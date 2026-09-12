"""Run one strict audit of the late-base min-target adapter bank."""
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
from jax_experiments.analysis import regime_adapter_fork as source_protocol
from jax_experiments.analysis import regime_adapter_latebase_min as protocol
from jax_experiments.analysis.regime_adapter_policy_bank import (
    LoadedPolicyBank,
    load_policy_bank_from,
)
from jax_experiments.analysis import (
    run_regime_adapter_latebase_min_branch as branch,
)
from jax_experiments.analysis import run_regime_adapter_branch as source_runner
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


CASES = (
    EvaluationCase("late_robust", "robust"),
    EvaluationCase(
        "frozen_base", "bank", (-1,) * len(protocol.MODES)),
    EvaluationCase("identity_adapter", "bank", protocol.MODES),
    *(EvaluationCase(
        f"fixed_adapter_{mode}", "bank",
        (mode,) * len(protocol.MODES)) for mode in protocol.MODES),
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _portable_manifest_key(path: str | Path) -> str:
    parts = Path(path).parts
    try:
        index = parts.index("jax_experiments")
    except ValueError as error:
        raise ValueError(
            f"training provenance is outside jax_experiments: {path}"
        ) from error
    return Path(*parts[index:]).as_posix()


def _training_manifest_paths(seed: int) -> tuple[Path, ...]:
    return (
        protocol.source_bundle_dir(seed)
        / source_protocol.BUNDLE_MANIFEST_NAME,
        *(protocol.bundle_manifest(seed, mode) for mode in protocol.MODES),
    )


def _current_training_manifests(seed: int) -> dict:
    return {
        _portable_manifest_key(path): protocol.file_record(path)
        for path in _training_manifest_paths(seed)
    }


def _source_record_from_bundle(directory: Path) -> str:
    payload = protocol.read_json(
        directory / "checkpoints" / protocol.BRANCH_BOOTSTRAP_NAME)
    record = payload.get("canonical_manifest")
    if not isinstance(record, dict):
        raise ValueError("late-base branch lacks canonical provenance")
    return str(sorted(record.items()))


def _expected_steps(case: EvaluationCase) -> tuple[int, int]:
    if case.kind == "robust":
        return protocol.SOURCE_TOTAL_STEPS, protocol.SOURCE_TOTAL_STEPS
    return protocol.FINAL_TOTAL_STEPS, protocol.BANK_AGGREGATE_TOTAL_STEPS


def _validate_case(
        directory: Path, seed: int, event_seed: int,
        case: EvaluationCase) -> None:
    expected_counts = {
        "summary.csv": 3,
        "task_returns.csv": len(protocol.MODES),
        "switching_returns.csv": protocol.SWITCHING_EPISODES,
    }
    expected_per_controller, expected_aggregate = _expected_steps(case)
    for filename, expected_count in expected_counts.items():
        rows = _read_rows(directory / filename)
        if len(rows) != expected_count:
            raise ValueError(
                f"{case.label}/{filename} has {len(rows)} rows")
        for row in rows:
            if (row.get("controller_case") != case.label
                    or int(float(row["training_seed"])) != seed
                    or float(row["residual_delta"]) != protocol.DELTA
                    or int(float(row["event_seed"])) != event_seed
                    or int(float(row["per_controller_steps"]))
                    != expected_per_controller
                    or int(float(row["aggregate_controller_steps"]))
                    != expected_aggregate
                    or row.get("compute_matched") != "False"):
                raise ValueError(f"{case.label}/{filename} identity changed")
    task_rows = _read_rows(directory / "task_returns.csv")
    if ({int(float(row["mode_id_mean"])) for row in task_rows}
            != set(protocol.MODES)):
        raise ValueError(f"{case.label} stationary sweep is incomplete")
    switching_rows = _read_rows(directory / "switching_returns.csv")
    if {row["metric_semantics"] for row in switching_rows} != {
            "fixed_horizon_stream_sum"}:
        raise ValueError(f"{case.label} switching audit is not strict horizon")


def validate_audit(seed: int, event_seed: int) -> dict:
    seed = protocol.require_seed(seed)
    event_seed = int(event_seed)
    if event_seed not in protocol.EVENT_SEEDS:
        raise ValueError(f"unknown late-base event seed {event_seed}")
    destination = protocol.audit_dir(seed, event_seed)
    payload = protocol.read_json(
        destination / protocol.AUDIT_MANIFEST_NAME)
    expected_files = {
        f"{case.label}/{filename}"
        for case in CASES for filename in OUTPUT_FILES
    }
    expected_identity = {
        **protocol.canonical_identity(seed),
        "event_seed": event_seed,
    }
    if (payload.get("schema") != protocol.AUDIT_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("identity") != expected_identity
            or set(payload.get("files") or {}) != expected_files
            or payload.get("training_bundle_manifests")
            != _current_training_manifests(seed)):
        raise ValueError(f"invalid late-base audit: {destination}")
    for relative, expected in payload["files"].items():
        path = destination / relative
        if not path.is_file() or protocol.file_record(path) != expected:
            raise ValueError(f"late-base audit file changed: {path}")
    for case in CASES:
        _validate_case(
            destination / case.label, seed, event_seed, case)
    return payload


def _load_bank(seed: int) -> LoadedPolicyBank:
    return load_policy_bank_from(
        seed,
        protocol.DELTA,
        validate_bundle=branch.validate_published_bundle,
        adapter_bundle_dir=protocol.adapter_bundle_dir,
        expected_next_iteration=protocol.FINAL_NEXT_ITERATION,
        expected_total_steps=protocol.FINAL_TOTAL_STEPS,
        source_record_from_bundle=_source_record_from_bundle,
    )


def _load_robust(seed: int):
    source_runner.validate_published_bundle(seed, "robust_continue")
    directory = protocol.source_bundle_dir(seed)
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
    if (next_iteration != protocol.SOURCE_NEXT_ITERATION
            or total_steps != protocol.SOURCE_TOTAL_STEPS):
        temporary.cleanup()
        if hasattr(env, "close"):
            env.close()
        raise ValueError("late robust checkpoint is stale")
    return config, tasks, agent, env, temporary


def _evaluate(
        seed: int, event_seed: int, case: EvaluationCase,
        output: Path, robust, bank: LoadedPolicyBank) -> None:
    robust_config, robust_tasks, robust_agent, _, _ = robust
    if case.kind == "robust":
        config = deepcopy(robust_config)
        tasks = robust_tasks
        agent = robust_agent
        algo_label = "regime_sac"
        checkpoint_next = protocol.SOURCE_NEXT_ITERATION
    else:
        config = deepcopy(bank.config)
        config.stochastic_mode_fixed_id = -1
        tasks = bank.tasks
        agent = bank.agent
        agent.set_controller_map(case.controller_map)
        algo_label = "regime_adapter_latebase_min_bank"
        checkpoint_next = protocol.FINAL_NEXT_ITERATION
    per_controller_steps, aggregate_steps = _expected_steps(case)

    eval_env = make_env(config, seed_offset=event_seed - seed)
    eval_env.build_rollout_fn(
        nnx.graphdef(agent.policy), direct_policy_context=True)
    try:
        task_rows = final_task_sweep.evaluate_task_split(
            agent, eval_env, config, tasks, "test",
            protocol.EPISODES_PER_TASK, len(protocol.MODES),
            20_426_726 + event_seed)
        switching_rows, trace_rows = final_task_sweep.evaluate_switching(
            agent, eval_env, config, tasks,
            protocol.SWITCHING_EPISODES, protocol.DWELL_STEPS,
            20_436_726 + event_seed)
        meta = {
            "run_name": case.label,
            "env": protocol.ENV.removesuffix("-v2"),
            "algo": algo_label,
            "training_seed": seed,
            "event_seed": event_seed,
            "residual_delta": protocol.DELTA,
            "controller_case": case.label,
            "controller_map": (
                "" if case.controller_map is None
                else ",".join(str(value) for value in case.controller_map)),
            "checkpoint_next_iter": checkpoint_next,
            "per_controller_steps": per_controller_steps,
            "aggregate_controller_steps": aggregate_steps,
            "compute_matched": False,
            "critic_target_mode": "min",
            "alpha_frozen": True,
            "budget_semantics": (
                "Diagnostic upper bound from the mature 8.4M robust base; "
                "each fixed-mode residual receives 0.7M additional steps."),
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
        _validate_case(output, seed, event_seed, case)
    finally:
        if hasattr(eval_env, "close"):
            eval_env.close()


def run(seed: int, event_seed: int) -> None:
    seed = protocol.require_seed(seed)
    event_seed = int(event_seed)
    if event_seed not in protocol.EVENT_SEEDS:
        raise ValueError(f"unknown late-base event seed {event_seed}")
    destination = protocol.audit_dir(seed, event_seed)
    if protocol.audit_manifest(seed, event_seed).is_file():
        validate_audit(seed, event_seed)
        print(f"LATE-BASE AUDIT ALREADY COMPLETE: {destination}")
        return

    robust = _load_robust(seed)
    bank = _load_bank(seed)
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
        for case in CASES:
            output = temporary / case.label
            print(
                f"LATE-BASE AUDIT seed={seed} event={event_seed} "
                f"case={case.label}",
                flush=True)
            _evaluate(seed, event_seed, case, output, robust, bank)
            for filename in OUTPUT_FILES:
                relative = f"{case.label}/{filename}"
                records[relative] = protocol.file_record(
                    temporary / relative)
            jax.clear_caches()
        payload = {
            "schema": protocol.AUDIT_SCHEMA,
            "status": "complete",
            "identity": {
                **protocol.canonical_identity(seed),
                "event_seed": event_seed,
            },
            "training_bundle_manifests": (
                _current_training_manifests(seed)),
            "routing_rule": "Fixed identity map [0,1,2,3]; no calibration.",
            "compute_matched": False,
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
    validate_audit(seed, event_seed)
    print(f"LATE-BASE AUDIT COMPLETE: {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, choices=protocol.TRAINING_SEEDS, required=True)
    parser.add_argument(
        "--event-seed", type=int, choices=protocol.EVENT_SEEDS,
        required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for scheduler staging")
    run(args.seed, args.event_seed)


if __name__ == "__main__":
    main()
