"""Run one sealed paired audit of robust continuation and adapter banks."""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import jax
from flax import nnx

from jax_experiments.analysis import final_task_sweep
from jax_experiments.analysis import regime_adapter_fork as protocol
from jax_experiments.analysis.regime_adapter_policy_bank import (
    LoadedPolicyBank,
    load_policy_bank,
)
from jax_experiments.analysis.run_regime_adapter_branch import (
    validate_published_bundle,
)
from jax_experiments.analysis.run_regime_adapter_calibration import (
    validate_calibration,
)
from jax_experiments.common.checkpoint import load_checkpoint
from jax_experiments.common.logging import Logger
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.train import make_algo, make_env


OUTPUT_FILES = ("summary.csv", "task_returns.csv", "switching_returns.csv")


@dataclass(frozen=True)
class EvaluationCase:
    label: str
    kind: str
    controller_map: tuple[int, ...] | None = None


def cases(utility_map) -> tuple[EvaluationCase, ...]:
    values = tuple(int(value) for value in utility_map)
    return (
        EvaluationCase("robust_continue", "robust"),
        EvaluationCase(
            "frozen_base", "bank", (-1,) * len(protocol.MODES)),
        EvaluationCase("identity_adapter", "bank", protocol.MODES),
        EvaluationCase("utility_adapter", "bank", values),
        *(EvaluationCase(
            f"fixed_adapter_{mode}", "bank",
            (mode,) * len(protocol.MODES)) for mode in protocol.MODES),
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _validate_case(
        directory: Path, seed: int, delta: float, event_seed: int,
        case: EvaluationCase) -> None:
    expected_counts = {
        "summary.csv": 3,
        "task_returns.csv": len(protocol.MODES),
        "switching_returns.csv": protocol.SWITCHING_EPISODES,
    }
    for filename, expected_count in expected_counts.items():
        rows = _read_rows(directory / filename)
        if len(rows) != expected_count:
            raise ValueError(
                f"{case.label}/{filename} has {len(rows)} rows")
        for row in rows:
            if (row.get("controller_case") != case.label
                    or int(float(row["training_seed"])) != seed
                    or float(row["residual_delta"]) != delta
                    or int(float(row["event_seed"])) != event_seed
                    or int(float(row["aggregate_controller_steps"]))
                    != protocol.ROBUST_FINAL_TOTAL_STEPS):
                raise ValueError(f"{case.label}/{filename} identity changed")
    task_rows = _read_rows(directory / "task_returns.csv")
    if ({int(float(row["mode_id_mean"])) for row in task_rows}
            != set(protocol.MODES)):
        raise ValueError(f"{case.label} stationary mode sweep is incomplete")
    switching_rows = _read_rows(directory / "switching_returns.csv")
    if {row["metric_semantics"] for row in switching_rows} != {
            "fixed_horizon_stream_sum"}:
        raise ValueError(f"{case.label} switching audit is not strict horizon")


def validate_audit(seed: int, delta: float, event_seed: int) -> dict:
    seed = protocol.require_seed(seed)
    delta = protocol.require_delta(delta)
    destination = protocol.audit_dir(seed, delta, event_seed)
    payload = protocol.read_json(destination / protocol.AUDIT_MANIFEST_NAME)
    utility = protocol.read_json(
        protocol.calibration_dir(seed, delta) / "utility_map.json")
    expected_cases = cases(utility["controller_map"])
    expected_files = {
        f"{case.label}/{filename}"
        for case in expected_cases for filename in OUTPUT_FILES
    }
    expected_identity = {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "env": protocol.ENV,
        "training_seed": seed,
        "residual_delta": delta,
        "event_seed": int(event_seed),
    }
    if (payload.get("schema") != protocol.AUDIT_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("identity") != expected_identity
            or set(payload.get("files") or {}) != expected_files):
        raise ValueError(f"invalid regime-adapter audit: {destination}")
    for relative, expected in payload["files"].items():
        path = destination / relative
        if not path.is_file() or protocol.file_record(path) != expected:
            raise ValueError(f"regime-adapter audit file changed: {path}")
    for case in expected_cases:
        _validate_case(
            destination / case.label, seed, delta, event_seed, case)
    return payload


def _load_robust(seed: int):
    validate_published_bundle(seed, "robust_continue")
    directory = protocol.robust_bundle_dir(seed)
    config = final_task_sweep.load_config(directory)
    config.stochastic_mode_fixed_id = -1
    env = make_env(config, seed_offset=0)
    tasks = env.sample_tasks(config.task_num)
    agent = make_algo(config.algo, env.obs_dim, env.act_dim, config)
    agent.set_task_metadata(tasks)
    replay = ReplayBuffer(
        env.obs_dim, env.act_dim, capacity=1,
        belief_dim=agent.belief_dim)
    temporary = tempfile.TemporaryDirectory()
    logger = Logger(temporary.name)
    next_iteration, total_steps = load_checkpoint(
        str(directory / "checkpoints"), agent, replay, logger,
        config.algo, load_replay_buffer=False)
    if (next_iteration != protocol.ROBUST_FINAL_NEXT_ITERATION
            or total_steps != protocol.ROBUST_FINAL_TOTAL_STEPS):
        temporary.cleanup()
        if hasattr(env, "close"):
            env.close()
        raise ValueError("robust continuation checkpoint is stale")
    return config, tasks, agent, env, temporary


def _evaluate(
        seed: int, delta: float, event_seed: int, case: EvaluationCase,
        output: Path, robust, bank: LoadedPolicyBank) -> None:
    robust_config, robust_tasks, robust_agent, _, _ = robust
    if case.kind == "robust":
        config = deepcopy(robust_config)
        tasks = robust_tasks
        agent = robust_agent
        algo_label = "regime_sac"
        checkpoint_next = protocol.ROBUST_FINAL_NEXT_ITERATION
        per_controller_steps = protocol.ROBUST_FINAL_TOTAL_STEPS
    else:
        config = deepcopy(bank.config)
        config.stochastic_mode_fixed_id = -1
        tasks = bank.tasks
        agent = bank.agent
        agent.set_controller_map(case.controller_map)
        algo_label = "regime_adapter_bank"
        checkpoint_next = protocol.ADAPTER_FINAL_NEXT_ITERATION
        per_controller_steps = protocol.ADAPTER_FINAL_TOTAL_STEPS
    eval_env = make_env(config, seed_offset=event_seed - seed)
    eval_env.build_rollout_fn(
        nnx.graphdef(agent.policy), direct_policy_context=True)
    try:
        task_rows = final_task_sweep.evaluate_task_split(
            agent, eval_env, config, tasks, "test",
            protocol.EPISODES_PER_TASK, len(protocol.MODES),
            20_300_722 + event_seed)
        switching_rows, trace_rows = final_task_sweep.evaluate_switching(
            agent, eval_env, config, tasks,
            protocol.SWITCHING_EPISODES, protocol.DWELL_STEPS,
            20_310_722 + event_seed)
        meta = {
            "run_name": case.label,
            "env": protocol.ENV.removesuffix("-v2"),
            "algo": algo_label,
            "training_seed": seed,
            "event_seed": event_seed,
            "residual_delta": delta,
            "controller_case": case.label,
            "controller_map": (
                "" if case.controller_map is None
                else ",".join(str(value) for value in case.controller_map)),
            "checkpoint_next_iter": checkpoint_next,
            "per_controller_steps": per_controller_steps,
            "aggregate_controller_steps": protocol.ROBUST_FINAL_TOTAL_STEPS,
            "budget_semantics": (
                "5.6M shared robust pretrain plus 2.8M branch-specific "
                "aggregate transitions"),
            "heldout_task_stream": "sealed",
        }
        for rows in (task_rows, switching_rows, trace_rows):
            for row in rows:
                row.update(meta)
        summary_rows = final_task_sweep.summarize(
            meta, task_rows, switching_rows, trace_rows,
            float(config.bapr_v2_gate_error_threshold), 50)
        final_task_sweep.write_csv(output / "task_returns.csv", task_rows)
        final_task_sweep.write_csv(
            output / "switching_returns.csv", switching_rows)
        final_task_sweep.write_csv(output / "summary.csv", summary_rows)
        _validate_case(output, seed, delta, event_seed, case)
    finally:
        if hasattr(eval_env, "close"):
            eval_env.close()


def run(seed: int, delta: float, event_seed: int) -> None:
    seed = protocol.require_seed(seed)
    delta = protocol.require_delta(delta)
    event_seed = int(event_seed)
    validate_calibration(seed, delta)
    destination = protocol.audit_dir(seed, delta, event_seed)
    if protocol.audit_manifest(seed, delta, event_seed).is_file():
        validate_audit(seed, delta, event_seed)
        print(f"REGIME ADAPTER AUDIT ALREADY COMPLETE: {destination}")
        return
    utility = protocol.read_json(
        protocol.calibration_dir(seed, delta) / "utility_map.json")
    selected_cases = cases(utility["controller_map"])
    robust = _load_robust(seed)
    bank = load_policy_bank(seed, delta)
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp.", dir=destination.parent))
    try:
        records = {}
        for case in selected_cases:
            output = temporary / case.label
            print(
                f"REGIME ADAPTER AUDIT seed={seed} delta={delta} "
                f"event={event_seed} case={case.label}", flush=True)
            _evaluate(seed, delta, event_seed, case, output, robust, bank)
            for filename in OUTPUT_FILES:
                relative = f"{case.label}/{filename}"
                records[relative] = protocol.file_record(
                    temporary / relative)
            jax.clear_caches()
        payload = {
            "schema": protocol.AUDIT_SCHEMA,
            "status": "complete",
            "identity": {
                "protocol_version": protocol.PROTOCOL_VERSION,
                "env": protocol.ENV,
                "training_seed": seed,
                "residual_delta": delta,
                "event_seed": event_seed,
            },
            "utility_map": utility["controller_map"],
            "calibration_manifest": protocol.file_record(
                protocol.calibration_manifest(seed, delta)),
            "files": records,
        }
        protocol.write_json_atomic(
            temporary / protocol.AUDIT_MANIFEST_NAME, payload)
        os.replace(temporary, destination)
    finally:
        bank.close()
        robust[4].cleanup()
        if hasattr(robust[3], "close"):
            robust[3].close()
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_audit(seed, delta, event_seed)
    print(f"REGIME ADAPTER AUDIT COMPLETE: {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=protocol.TRAINING_SEEDS,
                        required=True)
    parser.add_argument("--delta", type=float,
                        choices=protocol.RESIDUAL_DELTAS, required=True)
    parser.add_argument("--event-seed", type=int,
                        choices=protocol.AUDIT_EVENT_SEEDS, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for scheduler staging")
    run(args.seed, args.delta, args.event_seed)


if __name__ == "__main__":
    main()
