"""Evaluate a DANIEL policy against dispatching rules on JSP instances."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Iterable, Mapping, Optional

import pandas as pd
import torch

from solution_methods.DANIEL.network.main_model import DANIEL
from solution_methods.DANIEL.src.common_utils import greedy_select_action
from solution_methods.DANIEL.src.env_test import FJSPEnv_test
from solution_methods.dispatching_rules.run_dispatching_rules import (
    run_dispatching_rules,
)
from solution_methods.helper_functions import load_job_shop_env, load_parameters, set_seeds


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "DANIEL.toml"
DEFAULT_CHECKPOINT_PATH = (
    Path(__file__).resolve().parent / "save" / "JSSP" / "6x6+jssp.pth"
)
FIRST_TEN_TAILLARD_INSTANCES = tuple(
    f"/jsp/taillard/ta{instance_id:02d}" for instance_id in range(1, 11)
)
DEFAULT_DISPATCHING_RULES = ("FIFO", "SPT", "MOR", "MWR", "LOR", "LWR")
SUPPORTED_DISPATCHING_RULES = set(DEFAULT_DISPATCHING_RULES)


def _resolve_checkpoint_path(checkpoint_path: str | Path) -> Path:
    """Resolve repository-relative, working-directory-relative, or absolute weights."""
    path = Path(checkpoint_path).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.extend((Path.cwd() / path, REPOSITORY_ROOT / path))

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    checked_paths = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"DANIEL checkpoint not found. Checked: {checked_paths}")


def _extract_state_dict(checkpoint) -> Mapping[str, torch.Tensor]:
    """Accept a plain state dict or common wrapped checkpoint dictionaries."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError("the checkpoint must contain a PyTorch state dictionary")

    for key in ("state_dict", "model_state_dict", "policy_state_dict"):
        state_dict = checkpoint.get(key)
        if isinstance(state_dict, Mapping):
            return state_dict

    return checkpoint


def _evaluate_daniel_policy(job_shop, policy: DANIEL, parameters: Mapping) -> float:
    """Run one deterministic greedy DANIEL rollout and return its makespan."""
    environment = FJSPEnv_test(job_shop, parameters)
    state = environment.state

    while True:
        with torch.no_grad():
            probabilities, _ = policy(
                fea_j=state.fea_j_tensor,
                op_mask=state.op_mask_tensor,
                candidate=state.candidate_tensor,
                fea_m=state.fea_m_tensor,
                mch_mask=state.mch_mask_tensor,
                comp_idx=state.comp_idx_tensor,
                dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                fea_pairs=state.fea_pairs_tensor,
            )
            action = greedy_select_action(probabilities)

        state, _, done = environment.step(actions=action.cpu().numpy())
        if done.all():
            break

    return float(job_shop.makespan)


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_daniel_against_dispatching_rules(
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
        instances: Iterable[str] = FIRST_TEN_TAILLARD_INSTANCES,
        dispatching_rules: Iterable[str] = DEFAULT_DISPATCHING_RULES,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        save_csv: Optional[str | Path] = None,
        verbose: bool = True) -> pd.DataFrame:
    """Benchmark greedy DANIEL and dispatching rules on fresh JSP instances.

    The checkpoint is loaded once. Timings contain only environment setup and
    scheduling; instance parsing and model construction are excluded. Each
    method receives a fresh ``JobShop`` instance to prevent schedule leakage.

    DANIEL's neural layers are size-independent, so a 6x6 JSSP checkpoint can
    run on the first ten 15x15 Taillard instances. This remains an
    out-of-distribution generalization test.
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
    requested_device = device or parameters["device"]["name"]
    torch_device = torch.device(requested_device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available")
    parameters["device"]["name"] = requested_device

    evaluation_seed = parameters["test_parameters"]["seed"] if seed is None else seed
    set_seeds(evaluation_seed)

    resolved_checkpoint = _resolve_checkpoint_path(checkpoint_path)
    checkpoint = torch.load(resolved_checkpoint, map_location="cpu", weights_only=True)
    policy = DANIEL(parameters).to(torch_device)
    policy.load_state_dict(_extract_state_dict(checkpoint))
    policy.eval()

    results = []
    for instance_name in instance_names:
        instance_information = load_job_shop_env(instance_name)
        dimensions = (
            instance_information.nr_of_jobs,
            instance_information.nr_of_machines,
        )

        daniel_environment = load_job_shop_env(instance_name)
        _synchronize_device(torch_device)
        start_time = perf_counter()
        daniel_makespan = _evaluate_daniel_policy(
            daniel_environment,
            policy,
            parameters,
        )
        _synchronize_device(torch_device)
        runtime_seconds = perf_counter() - start_time
        results.append({
            "instance": Path(instance_name).name,
            "jobs": dimensions[0],
            "machines": dimensions[1],
            "method": "DANIEL",
            "makespan": daniel_makespan,
            "runtime_seconds": runtime_seconds,
            "checkpoint": str(resolved_checkpoint),
        })
        if verbose:
            print(
                f"{Path(instance_name).name:>4} | {'DANIEL':>6} | "
                f"makespan={daniel_makespan:>7.0f} | {runtime_seconds:.3f}s"
            )

        for rule in rules:
            dispatching_environment = load_job_shop_env(instance_name)
            dispatching_parameters = {
                "instance": {
                    "online_arrivals": False,
                    "problem_instance": instance_name,
                    "dispatching_rule": rule,
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
                    f"{Path(instance_name).name:>4} | {rule:>6} | "
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

