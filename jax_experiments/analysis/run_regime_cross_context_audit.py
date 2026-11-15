"""Evaluate robust and all oracle contexts on one paired event stream."""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import tempfile
from pathlib import Path

import jax
from flax import nnx

from jax_experiments.analysis import final_task_sweep
from jax_experiments.analysis import regime_control_headroom as base
from jax_experiments.analysis import regime_cross_context as protocol
from jax_experiments.analysis.run_regime_control_headroom_controller import (
    validate_bundle,
)
from jax_experiments.common.checkpoint import load_checkpoint
from jax_experiments.common.logging import Logger
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.train import make_algo, make_env


OUTPUT_FILES = ("summary.csv", "task_returns.csv", "switching_returns.csv")
EXPECTED_ROWS = {
    "summary.csv": 3,
    "task_returns.csv": len(base.MODES),
    "switching_returns.csv": base.SWITCHING_EPISODES,
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _validate_case(
        directory: Path, env: str, seed: int, event_seed: int,
        case: protocol.EvaluationCase) -> None:
    expected_env = base.env_slug(env)
    expected_run_name = base.run_dir(
        env, case.source_role, seed).name
    for filename, expected_count in EXPECTED_ROWS.items():
        rows = _read_rows(directory / filename)
        if len(rows) != expected_count:
            raise ValueError(
                f"{filename} has {len(rows)} rows, expected "
                f"{expected_count}")
        for row in rows:
            if row.get("run_name") != expected_run_name:
                raise ValueError(f"{filename} has wrong run_name")
            if row.get("env") != expected_env:
                raise ValueError(f"{filename} has wrong environment")
            if row.get("algo") != "regime_sac":
                raise ValueError(f"{filename} has wrong algorithm")
            if row.get("source_role") != case.source_role:
                raise ValueError(f"{filename} has wrong source role")
            if row.get("eval_regime_context") != case.label:
                raise ValueError(f"{filename} has wrong case label")
            if row.get("eval_context_kind") != case.context_kind:
                raise ValueError(f"{filename} has wrong context kind")
            expected_fixed = (
                -1 if case.fixed_mode_id is None
                else case.fixed_mode_id)
            if int(float(row["eval_fixed_mode_id"])) != expected_fixed:
                raise ValueError(f"{filename} has wrong fixed context")
            if int(float(row["event_seed"])) != event_seed:
                raise ValueError(f"{filename} has wrong event seed")
            if int(float(row["checkpoint_next_iter"])) != base.MAX_ITERS:
                raise ValueError(f"{filename} has stale checkpoint iteration")
            if (int(float(row["checkpoint_total_steps"]))
                    != base.FINAL_TOTAL_STEPS):
                raise ValueError(f"{filename} has stale checkpoint budget")

    task_rows = _read_rows(directory / "task_returns.csv")
    modes = {int(float(row["mode_id_mean"])) for row in task_rows}
    if modes != set(base.MODES):
        raise ValueError(f"stationary mode sweep is incomplete: {modes}")
    if {row["split"] for row in task_rows} != {"test"}:
        raise ValueError("stationary audit must use the test split")
    switching_rows = _read_rows(directory / "switching_returns.csv")
    if {row["metric_semantics"] for row in switching_rows} != {
            "fixed_horizon_stream_sum"}:
        raise ValueError("switching audit is not strict fixed-horizon")


def validate_audit(
        env: str, seed: int, event_seed: int) -> dict[str, object]:
    env = base.require_env(env)
    seed = base.require_training_seed(seed)
    event_seed = base.require_event_seed(event_seed)
    directory = protocol.audit_dir(env, seed, event_seed)
    payload = base.read_json(directory / "audit_manifest.json")
    if (payload.get("schema") != protocol.AUDIT_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("identity")
            != protocol.identity(env, seed, event_seed)):
        raise ValueError(f"invalid cross-context audit: {directory}")
    records = payload.get("files") or {}
    expected_paths = {
        f"{case.label}/{filename}"
        for case in protocol.EVALUATION_CASES
        for filename in OUTPUT_FILES
    }
    if set(records) != expected_paths:
        raise ValueError(f"incomplete cross-context audit: {directory}")
    for relative, expected in records.items():
        path = directory / relative
        if not path.is_file() or base.file_record(path) != expected:
            raise ValueError(f"cross-context audit file changed: {path}")
    for case in protocol.EVALUATION_CASES:
        _validate_case(
            directory / case.label, env, seed, event_seed, case)
    return payload


def _evaluate_case(
        env_name: str, seed: int, event_seed: int,
        case: protocol.EvaluationCase, output: Path) -> None:
    run_dir = base.bundle_dir(env_name, case.source_role, seed)
    config = final_task_sweep.load_config(run_dir)
    if (config.algo != "regime_sac"
            or config.regime_context_source != case.source_role):
        raise ValueError(
            f"{case.label} requires a {case.source_role} "
            "regime_sac checkpoint")

    source_env = make_env(config, seed_offset=0)
    train_tasks = source_env.sample_tasks(config.task_num)
    test_tasks = source_env.sample_tasks(config.test_task_num)
    agent = make_algo(config.algo, source_env.obs_dim, source_env.act_dim, config)
    agent.set_task_metadata(train_tasks)
    replay = ReplayBuffer(
        source_env.obs_dim, source_env.act_dim, capacity=1,
        belief_dim=agent.belief_dim)
    with tempfile.TemporaryDirectory() as log_dir:
        logger = Logger(log_dir)
        next_iteration, total_steps = load_checkpoint(
            str(run_dir / "checkpoints"), agent, replay, logger,
            config.algo, load_replay_buffer=False)
    final_task_sweep.require_checkpoint_iteration(
        next_iteration, base.MAX_ITERS)
    if int(total_steps) != base.FINAL_TOTAL_STEPS:
        raise ValueError(
            f"checkpoint has {total_steps} steps, expected "
            f"{base.FINAL_TOTAL_STEPS}")
    agent.set_eval_context_override(
        case.context_kind, case.fixed_mode_id)

    eval_env = make_env(config, seed_offset=event_seed - seed)
    eval_env.build_rollout_fn(
        nnx.graphdef(agent.policy), direct_policy_context=True)
    task_rows = final_task_sweep.evaluate_task_split(
        agent, eval_env, config, test_tasks, "test",
        base.EPISODES_PER_TASK, len(base.MODES),
        20_270_722 + event_seed)
    switching_rows, trace_rows = final_task_sweep.evaluate_switching(
        agent, eval_env, config, test_tasks,
        base.SWITCHING_EPISODES, base.DWELL_STEPS,
        20_280_722 + event_seed)

    run_meta = {
        "run_name": run_dir.name,
        "env": base.env_slug(env_name),
        "algo": config.algo,
        "seed": int(config.seed),
        "event_seed": int(event_seed),
        "source_role": case.source_role,
        "checkpoint_next_iter": int(next_iteration),
        "checkpoint_total_steps": int(total_steps),
        "eval_regime_context": case.label,
        "eval_context_kind": case.context_kind,
        "eval_fixed_mode_id": (
            -1 if case.fixed_mode_id is None
            else case.fixed_mode_id),
        "heldout_task_stream": "validation",
    }
    for rows in (task_rows, switching_rows, trace_rows):
        for row in rows:
            row.update(run_meta)
    summary_rows = final_task_sweep.summarize(
        run_meta, task_rows, switching_rows, trace_rows,
        float(config.bapr_v2_gate_error_threshold), 50)

    final_task_sweep.write_csv(output / "task_returns.csv", task_rows)
    final_task_sweep.write_csv(
        output / "switching_returns.csv", switching_rows)
    final_task_sweep.write_csv(output / "summary.csv", summary_rows)
    _validate_case(
        output, env_name, seed, event_seed, case)
    if hasattr(eval_env, "close"):
        eval_env.close()
    if hasattr(source_env, "close"):
        source_env.close()


def run(env: str, seed: int, event_seed: int) -> None:
    env = base.require_env(env)
    seed = base.require_training_seed(seed)
    event_seed = base.require_event_seed(event_seed)
    bundle_payloads = {
        role: validate_bundle(env, role, seed)
        for role in base.ROLES
    }
    destination = protocol.audit_dir(env, seed, event_seed)
    if protocol.audit_manifest(env, seed, event_seed).is_file():
        try:
            validate_audit(env, seed, event_seed)
        except (KeyError, OSError, TypeError, ValueError):
            pass
        else:
            print(f"CROSS CONTEXT AUDIT ALREADY COMPLETE: {destination}")
            return
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
        for case in protocol.EVALUATION_CASES:
            case_output = temporary / case.label
            print(
                f"CROSS CONTEXT AUDIT env={env} seed={seed} "
                f"event_seed={event_seed} case={case.label}",
                flush=True)
            _evaluate_case(
                env, seed, event_seed, case, case_output)
            for filename in OUTPUT_FILES:
                relative = f"{case.label}/{filename}"
                records[relative] = base.file_record(
                    temporary / relative)
            jax.clear_caches()
        payload = {
            "schema": protocol.AUDIT_SCHEMA,
            "status": "complete",
            "identity": protocol.identity(env, seed, event_seed),
            "bundles": {
                role: {
                    "manifest": base.file_record(
                        base.bundle_manifest(env, role, seed)),
                    "checkpoint": bundle_payloads[role]["checkpoint"],
                }
                for role in base.ROLES
            },
            "files": records,
        }
        base.write_json_atomic(
            temporary / "audit_manifest.json", payload)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_audit(env, seed, event_seed)
    print(f"CROSS CONTEXT AUDIT COMPLETE: {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=base.ENVS, required=True)
    parser.add_argument(
        "--seed", choices=base.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument(
        "--event-seed", choices=base.AUDIT_EVENT_SEEDS,
        type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for scheduler input staging")
    run(args.env, args.seed, args.event_seed)


if __name__ == "__main__":
    main()
