"""Evaluate an L2D policy against dispatching rules on JSP instances."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd
import torch

from solution_methods.dispatching_rules.run_dispatching_rules import (
    run_dispatching_rules,
)
from solution_methods.helper_functions import load_job_shop_env, load_parameters, set_seeds
from solution_methods.L2D.network.actor_critic import ActorCritic
from solution_methods.L2D.src.env_test import NipsJSPEnv_test
from solution_methods.L2D.src.mb_agg import g_pool_cal


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "L2D.toml"
DEFAULT_CHECKPOINT_PATH = Path(__file__).resolve().parent / "saved_models" / "6_6_1_99.pth"
FIRST_TEN_TAILLARD_INSTANCES = tuple(
    f"/jsp/taillard/ta{instance_id:02d}" for instance_id in range(11, 21)
)
DEFAULT_DISPATCHING_RULES = ("FIFO", "SPT", "MOR", "MWR", "LOR", "LWR")
SUPPORTED_DISPATCHING_RULES = set(DEFAULT_DISPATCHING_RULES)


def _resolve_checkpoint_path(checkpoint_path: str | Path) -> Path:
    """Resolve repository-relative, working-directory-relative, or absolute weights."""
    path = Path(checkpoint_path).expanduser()
    candidates = [path]

    if not path.is_absolute():
        candidates.extend((Path.cwd() / path, REPOSITORY_ROOT / path))

    # L2D config files traditionally use paths such as /saved_models/model.pth.
    path_without_anchor = str(checkpoint_path).lstrip("/\\")
    candidates.append(Path(__file__).resolve().parent / path_without_anchor)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    checked_paths = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"L2D checkpoint not found. Checked: {checked_paths}")


def _extract_state_dict(checkpoint) -> Mapping[str, torch.Tensor]:
    """Accept a plain state dict or common wrapped checkpoint dictionaries."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError("the checkpoint must contain a PyTorch state dictionary")

    for key in ("state_dict", "model_state_dict", "policy_state_dict"):
        state_dict = checkpoint.get(key)
        if isinstance(state_dict, Mapping):
            return state_dict

    return checkpoint


def _build_policy(
        n_jobs: int,
        n_machines: int,
        network_parameters: Mapping,
        state_dict: Mapping[str, torch.Tensor],
        device: torch.device) -> ActorCritic:
    policy = ActorCritic(
        n_j=n_jobs,
        n_m=n_machines,
        num_layers=network_parameters["num_layers"],
        learn_eps=False,
        neighbor_pooling_type=network_parameters["neighbor_pooling_type"],
        input_dim=network_parameters["input_dim"],
        hidden_dim=network_parameters["hidden_dim"],
        num_mlp_layers_feature_extract=network_parameters[
            "num_mlp_layers_feature_extract"
        ],
        num_mlp_layers_actor=network_parameters["num_mlp_layers_actor"],
        hidden_dim_actor=network_parameters["hidden_dim_actor"],
        num_mlp_layers_critic=network_parameters["num_mlp_layers_critic"],
        hidden_dim_critic=network_parameters["hidden_dim_critic"],
        device=device,
    ).to(device)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def _evaluate_l2d_policy(
        job_shop,
        policy: ActorCritic,
        parameters: Mapping,
        device: torch.device) -> float:
    """Run one deterministic greedy L2D rollout and return its makespan."""
    number_of_nodes = job_shop.nr_of_jobs * job_shop.nr_of_machines
    environment = NipsJSPEnv_test(
        n_j=job_shop.nr_of_jobs,
        n_m=job_shop.nr_of_machines,
        env_parameters=parameters["env_parameters"],
    )
    graph_pool = g_pool_cal(
        graph_pool_type=parameters["network_parameters"]["graph_pool_type"],
        batch_size=torch.Size([1, number_of_nodes, number_of_nodes]),
        n_nodes=number_of_nodes,
        device=device,
    )
    adjacency, features, candidates, mask = environment.reset(job_shop)

    while not environment.done():
        feature_tensor = torch.from_numpy(np.copy(features)).to(device)
        adjacency_tensor = torch.from_numpy(np.copy(adjacency)).to(device).to_sparse()
        candidate_tensor = torch.from_numpy(np.copy(candidates)).to(device)
        mask_tensor = torch.from_numpy(np.copy(mask)).to(device)

        with torch.no_grad():
            probabilities, _ = policy(
                x=feature_tensor,
                graph_pool=graph_pool,
                padded_nei=None,
                adj=adjacency_tensor,
                candidate=candidate_tensor.unsqueeze(0),
                mask=mask_tensor.unsqueeze(0),
            )
            action_index = int(probabilities.squeeze().argmax().item())
            action = int(candidates[action_index])

        adjacency, features, _, _, candidates, mask = environment.step(action)

    return float(job_shop.makespan)


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_l2d_against_dispatching_rules(
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
        instances: Iterable[str] = FIRST_TEN_TAILLARD_INSTANCES,
        dispatching_rules: Iterable[str] = DEFAULT_DISPATCHING_RULES,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        env_parameter_overrides: Optional[Mapping] = None,
        save_csv: Optional[str | Path] = None,
        verbose: bool = True) -> pd.DataFrame:
    """Benchmark greedy L2D and dispatching rules on fresh instance copies.

    The L2D checkpoint is loaded once. Runtime measurements contain only the
    scheduling rollout: instance parsing and checkpoint/model construction are
    deliberately excluded. Every method receives a newly parsed ``JobShop`` so
    schedules from an earlier method cannot leak into a later result.

    A 6x6 L2D checkpoint can be evaluated on the first ten 15x15 Taillard
    instances because the learned graph and MLP layers are size-independent.
    This is an out-of-distribution test; pass ``15_15_1_99.pth`` when a
    size-matched comparison is desired.
    """
    instance_names = tuple(instances)
    rules = tuple(rule.upper() for rule in dispatching_rules)
    if not instance_names:
        raise ValueError("instances must contain at least one problem instance")
    if not rules:
        raise ValueError("dispatching_rules must contain at least one rule")

    unsupported_rules = sorted(set(rules) - SUPPORTED_DISPATCHING_RULES)
    if unsupported_rules:
        raise ValueError(
            f"unsupported dispatching rules: {unsupported_rules}; "
            f"choose from {sorted(SUPPORTED_DISPATCHING_RULES)}"
        )

    parameters = load_parameters(config_path)
    if env_parameter_overrides is not None:
        parameters["env_parameters"].update(env_parameter_overrides)
    requested_device = device or parameters["test_parameters"]["device"]
    torch_device = torch.device(requested_device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available")

    evaluation_seed = parameters["test_parameters"]["seed"] if seed is None else seed
    set_seeds(evaluation_seed)

    resolved_checkpoint = _resolve_checkpoint_path(checkpoint_path)
    checkpoint = torch.load(resolved_checkpoint, map_location="cpu", weights_only=True)
    state_dict = _extract_state_dict(checkpoint)

    results = []
    policies = {}
    for instance_name in instance_names:
        # Load once to discover dimensions, then discard it. Timed runs below
        # always receive their own freshly parsed environment.
        instance_information = load_job_shop_env(instance_name)
        dimensions = (
            instance_information.nr_of_jobs,
            instance_information.nr_of_machines,
        )
        if dimensions not in policies:
            policies[dimensions] = _build_policy(
                n_jobs=dimensions[0],
                n_machines=dimensions[1],
                network_parameters=parameters["network_parameters"],
                state_dict=state_dict,
                device=torch_device,
            )

        l2d_environment = load_job_shop_env(instance_name)
        _synchronize_device(torch_device)
        start_time = perf_counter()
        l2d_makespan = _evaluate_l2d_policy(
            l2d_environment,
            policies[dimensions],
            parameters,
            torch_device,
        )
        _synchronize_device(torch_device)
        runtime_seconds = perf_counter() - start_time
        results.append({
            "instance": Path(instance_name).name,
            "jobs": dimensions[0],
            "machines": dimensions[1],
            "method": "L2D",
            "makespan": l2d_makespan,
            "runtime_seconds": runtime_seconds,
            "checkpoint": str(resolved_checkpoint),
        })
        if verbose:
            print(
                f"{Path(instance_name).name:>4} | {'L2D':>4} | "
                f"makespan={l2d_makespan:>7.0f} | {runtime_seconds:.3f}s"
            )

        for rule in rules:
            dispatching_environment = load_job_shop_env(instance_name)
            dispatching_parameters = {
                "instance": {
                    "online_arrivals": False,
                    "problem_instance": instance_name,
                    "dispatching_rule": rule,
                    # A JSP operation has only one eligible machine, making
                    # SPT and EET assignment equivalent. SPT also supports all
                    # dispatching priorities in the existing implementation.
                    "machine_assignment_rule": "SPT",
                }
            }
            start_time = perf_counter()
            dispatching_makespan, _ = run_dispatching_rules(
                dispatching_environment,
                **dispatching_parameters,
            )
            runtime_seconds = perf_counter() - start_time
            results.append({
                "instance": Path(instance_name).name,
                "jobs": dimensions[0],
                "machines": dimensions[1],
                "method": rule,
                "makespan": float(dispatching_makespan),
                "runtime_seconds": runtime_seconds,
                "checkpoint": None,
            })
            if verbose:
                print(
                    f"{Path(instance_name).name:>4} | {rule:>4} | "
                    f"makespan={dispatching_makespan:>7.0f} | {runtime_seconds:.3f}s"
                )

    result_frame = pd.DataFrame(results)
    if save_csv is not None:
        output_path = Path(save_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_frame.to_csv(output_path, index=False)

    return result_frame


def make_makespan_table(results: pd.DataFrame, include_mean: bool = True) -> pd.DataFrame:
    """Create an instance-by-method makespan table from long-form results."""
    table = results.pivot(index="instance", columns="method", values="makespan")
    if include_mean:
        table.loc["mean"] = table.mean(axis=0)
    return table


def summarize_evaluation(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize average quality, runtime, rank, and per-instance wins."""
    ranked_results = results.copy()
    ranked_results["rank"] = ranked_results.groupby("instance")["makespan"].rank(
        method="min"
    )
    ranked_results["win"] = ranked_results["rank"].eq(1)

    summary = ranked_results.groupby("method", as_index=True).agg(
        mean_makespan=("makespan", "mean"),
        median_makespan=("makespan", "median"),
        mean_rank=("rank", "mean"),
        wins=("win", "sum"),
        mean_runtime_seconds=("runtime_seconds", "mean"),
    )
    return summary.sort_values(["mean_rank", "mean_makespan"])
