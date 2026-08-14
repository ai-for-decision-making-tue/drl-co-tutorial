from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import rl4co
import torch
from lightning.pytorch import Callback, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from rl4co.envs.routing import CVRPEnv
from rl4co.models import AttentionModelPolicy, POMO
from rl4co.utils import RL4COTrainer
from rl4co.utils.ops import unbatchify


class EpochReporter(Callback):
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        metrics = {}
        for name, value in trainer.callback_metrics.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                metrics[name] = float(value.detach().cpu())
        print(
            "EPOCH_METRICS",
            json.dumps({"epoch": trainer.current_epoch, **metrics}, sort_keys=True),
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train POMO on synthetic CVRP15 instances.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "checkpoints")
    parser.add_argument("--run-name", default="pomo-cvrp15")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--train-data-size", type=int, default=100_000)
    parser.add_argument("--val-data-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-size", type=int, default=2_048)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-encoder-layers", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--eval-seed", type=int, default=4321)
    parser.add_argument("--precision", default="16-mixed")
    return parser.parse_args()


def make_policy(env: CVRPEnv, embed_dim: int, num_encoder_layers: int) -> AttentionModelPolicy:
    return AttentionModelPolicy(
        env_name=env.name,
        embed_dim=embed_dim,
        num_encoder_layers=num_encoder_layers,
        normalization="instance",
        use_graph_context=False,
    )


def make_fixed_eval_data(env: CVRPEnv, size: int, seed: int):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return env.generator(batch_size=[size]).cpu()


@torch.inference_mode()
def evaluate_policy(
    policy: AttentionModelPolicy,
    env: CVRPEnv,
    data,
    batch_size: int,
) -> dict[str, float]:
    device = next(policy.parameters()).device
    policy.eval()
    greedy_costs = []
    pomo_costs = []

    for start in range(0, data.batch_size[0], batch_size):
        raw = data[start : start + batch_size].clone()
        td = env.reset(raw).to(device)

        greedy = policy(
            td.clone(),
            env,
            phase="test",
            decode_type="greedy",
            return_actions=False,
        )
        greedy_costs.append(-greedy["reward"].detach().cpu())

        num_starts = env.get_num_starts(td)
        multi = policy(
            td.clone(),
            env,
            phase="test",
            decode_type="multistart_greedy",
            num_starts=num_starts,
            return_actions=False,
        )
        reward = unbatchify(multi["reward"], num_starts)
        pomo_costs.append(-reward.max(dim=-1).values.detach().cpu())

    greedy_costs = torch.cat(greedy_costs)
    pomo_costs = torch.cat(pomo_costs)
    return {
        "greedy_mean_cost": float(greedy_costs.mean()),
        "greedy_std_cost": float(greedy_costs.std()),
        "pomo_mean_cost": float(pomo_costs.mean()),
        "pomo_std_cost": float(pomo_costs.std()),
    }


def extract_policy_state(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    prefix = "policy."
    return {
        key[len(prefix) :]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith(prefix)
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training configuration.")

    run_dir = (args.output_dir / args.run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    env = CVRPEnv(generator_params={"num_loc": 15})
    policy = make_policy(env, args.embed_dim, args.num_encoder_layers)
    model = POMO(
        env,
        policy=policy,
        num_augment=1,
        batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        train_data_size=args.train_data_size,
        val_data_size=args.val_data_size,
        test_data_size=args.val_data_size,
        optimizer_kwargs={"lr": args.learning_rate},
    )

    eval_data = make_fixed_eval_data(env, args.eval_size, args.eval_seed)
    model.policy.cuda()
    initial_metrics = evaluate_policy(model.policy, env, eval_data, args.eval_batch_size)
    model.policy.cpu()
    model.train()
    print("INITIAL_METRICS", json.dumps(initial_metrics, sort_keys=True), flush=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir / "lightning-checkpoints",
        filename="epoch-{epoch:03d}-step-{step}",
        monitor="val/reward",
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    logger = CSVLogger(save_dir=run_dir, name="training-logs")

    trainer = RL4COTrainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=1,
        precision=args.precision,
        logger=logger,
        callbacks=[checkpoint_callback, EpochReporter()],
        default_root_dir=run_dir,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        enable_model_summary=False,
        enable_progress_bar=False,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer.fit(model)
    elapsed_seconds = time.perf_counter() - started

    best_checkpoint = checkpoint_callback.best_model_path
    policy_state = extract_policy_state(best_checkpoint)
    policy_path = run_dir / "pomo_cvrp15_policy.pt"
    torch.save(policy_state, policy_path)

    model.policy.load_state_dict(policy_state)
    model.policy.cuda()
    trained_metrics = evaluate_policy(model.policy, env, eval_data, args.eval_batch_size)
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3

    summary = {
        "rl4co_version": rl4co.__version__,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "best_checkpoint": str(Path(best_checkpoint).resolve()),
        "best_validation_reward": float(checkpoint_callback.best_model_score),
        "policy_state_dict": str(policy_path.resolve()),
        "initial_metrics": initial_metrics,
        "trained_metrics": trained_metrics,
        "elapsed_seconds": elapsed_seconds,
        "peak_vram_gb": peak_vram_gb,
    }
    summary_path = run_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("TRAINING_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    print(f"POLICY_PATH {policy_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
