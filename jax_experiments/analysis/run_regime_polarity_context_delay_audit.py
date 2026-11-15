"""Evaluate context causality and delayed-oracle reachability."""
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
from jax_experiments.analysis import regime_polarity_context_delay as protocol
from jax_experiments.analysis import regime_polarity_headroom as base
from jax_experiments.common.checkpoint import load_checkpoint
from jax_experiments.common.logging import Logger
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.train import make_algo, make_env


TRACE_FIELDS = (
    "episode",
    "step",
    "physics_action_task_id",
    "action_task_id",
    "eval_context_delay_remaining",
    "physics_task_after",
    "switched",
    "reward",
    "done",
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _indicator(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized in ("true", "1", "1.0"):
        return True
    if normalized in ("false", "0", "0.0"):
        return False
    raise ValueError(f"invalid boolean value {value!r}")


def _validate_bundle(env: str, role: str, seed: int) -> dict[str, object]:
    directory = base.bundle_dir(env, role, seed)
    payload = base.read_json(directory / "bundle_manifest.json")
    required = {
        "checkpoints/params.pkl",
        "checkpoints/train_state.pkl",
        "logs/protocol_signature.json",
    }
    if (payload.get("schema") != base.BUNDLE_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("identity") != base.identity(env, role, seed)):
        raise ValueError(f"invalid polarity source bundle: {directory}")
    records = payload.get("files") or {}
    if set(records) != required:
        raise ValueError(f"incomplete polarity source bundle: {directory}")
    for relative, expected in records.items():
        path = directory / relative
        if not path.is_file() or base.file_record(path) != expected:
            raise ValueError(f"polarity source bundle changed: {path}")
    expected_checkpoint = {
        "iteration": base.FINAL_ITERATION,
        "next_iteration": base.MAX_ITERS,
        "total_steps": base.FINAL_TOTAL_STEPS,
        "update_count": base.FINAL_UPDATE_COUNT,
        "algo": "regime_sac",
    }
    if payload.get("checkpoint") != expected_checkpoint:
        raise ValueError(f"wrong polarity checkpoint budget: {directory}")
    return payload


def _expected_context_rows(
        rows: list[dict[str, str]],
        case: protocol.EvaluationCase,
        event_seed: int,
) -> None:
    by_episode: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        by_episode.setdefault(int(row["episode"]), []).append(row)
    shuffled = protocol.shuffled_mode_map(event_seed)
    for episode, episode_rows in by_episode.items():
        episode_rows.sort(key=lambda row: int(row["step"]))
        expected_mode = None
        pending_mode = None
        delay_remaining = 0
        for row in episode_rows:
            physics_mode = int(row["physics_action_task_id"])
            actual_mode = int(row["action_task_id"])
            actual_remaining = int(row["eval_context_delay_remaining"])
            if case.context_kind == "checkpoint":
                wanted_mode = -1
                wanted_remaining = 0
            elif case.context_kind == "true":
                wanted_mode = physics_mode
                wanted_remaining = 0
            elif case.context_kind == "zero":
                wanted_mode = -1
                wanted_remaining = 0
            elif case.context_kind == "fixed":
                wanted_mode = int(case.fixed_mode_id)
                wanted_remaining = 0
            elif case.context_kind == "cyclic":
                wanted_mode = (physics_mode + 1) % len(base.MODES)
                wanted_remaining = 0
            elif case.context_kind == "shuffled":
                wanted_mode = shuffled[physics_mode]
                wanted_remaining = 0
            elif case.context_kind == "delayed":
                if expected_mode is None:
                    expected_mode = physics_mode
                    pending_mode = physics_mode
                elif physics_mode == expected_mode:
                    pending_mode = physics_mode
                    delay_remaining = 0
                elif physics_mode != pending_mode:
                    pending_mode = physics_mode
                    delay_remaining = int(case.delay_steps)
                    if delay_remaining == 0:
                        expected_mode = physics_mode
                elif delay_remaining > 1:
                    delay_remaining -= 1
                else:
                    expected_mode = physics_mode
                    delay_remaining = 0
                wanted_mode = int(expected_mode)
                wanted_remaining = delay_remaining
            else:
                raise AssertionError(case.context_kind)
            if (actual_mode != wanted_mode
                    or actual_remaining != wanted_remaining):
                raise ValueError(
                    "context schedule mismatch at "
                    f"episode={episode}, step={row['step']}: "
                    f"got mode={actual_mode}, remaining={actual_remaining}; "
                    f"expected mode={wanted_mode}, "
                    f"remaining={wanted_remaining}")


def _validate_case(
        directory: Path,
        env: str,
        seed: int,
        event_seed: int,
        case: protocol.EvaluationCase,
) -> None:
    expected_counts = {
        "summary.csv": 3,
        "switching_returns.csv": base.SWITCHING_EPISODES,
        "switching_trace.csv": (
            base.SWITCHING_EPISODES * base.MAX_EPISODE_STEPS),
    }
    if case.stationary:
        expected_counts["task_returns.csv"] = len(base.MODES)
    expected_run_name = base.bundle_dir(
        env, case.source_role, seed).name
    expected_map = "|".join(
        str(mode) for mode in protocol.shuffled_mode_map(event_seed))
    for filename, expected_count in expected_counts.items():
        rows = _read_rows(directory / filename)
        if len(rows) != expected_count:
            raise ValueError(
                f"{filename} has {len(rows)} rows, expected "
                f"{expected_count}")
        for row in rows:
            if row.get("run_name") != expected_run_name:
                raise ValueError(f"{filename} has wrong run_name")
            if row.get("env") != base.env_slug(env):
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
            expected_delay = (
                0 if case.delay_steps is None else case.delay_steps)
            if int(float(row["eval_delay_steps"])) != expected_delay:
                raise ValueError(f"{filename} has wrong context delay")
            if row.get("eval_shuffled_mode_map") != expected_map:
                raise ValueError(f"{filename} has wrong shuffled map")
            if int(float(row["event_seed"])) != event_seed:
                raise ValueError(f"{filename} has wrong event seed")
            if int(float(row["checkpoint_next_iter"])) != base.MAX_ITERS:
                raise ValueError(f"{filename} has stale checkpoint iteration")
            if (int(float(row["checkpoint_total_steps"]))
                    != base.FINAL_TOTAL_STEPS):
                raise ValueError(f"{filename} has stale checkpoint budget")

    if case.stationary:
        task_rows = _read_rows(directory / "task_returns.csv")
        modes = {int(float(row["mode_id_mean"])) for row in task_rows}
        if modes != set(base.MODES):
            raise ValueError(
                f"stationary mode sweep is incomplete: {modes}")
        if {row["split"] for row in task_rows} != {"test"}:
            raise ValueError("stationary audit must use the test split")
    switching_rows = _read_rows(directory / "switching_returns.csv")
    if {row["metric_semantics"] for row in switching_rows} != {
            "fixed_horizon_stream_sum"}:
        raise ValueError("switching audit is not strict fixed-horizon")
    trace_rows = _read_rows(directory / "switching_trace.csv")
    for row in trace_rows:
        _indicator(row["done"])
        _indicator(row["switched"])
    _expected_context_rows(trace_rows, case, event_seed)


def validate_audit(
        env: str, seed: int, event_seed: int) -> dict[str, object]:
    env = protocol.require_env(env)
    seed = base.require_training_seed(seed)
    event_seed = base.require_event_seed(event_seed)
    directory = protocol.audit_dir(env, seed, event_seed)
    payload = base.read_json(directory / "audit_manifest.json")
    if (payload.get("schema") != protocol.AUDIT_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("identity")
            != protocol.identity(env, seed, event_seed)):
        raise ValueError(f"invalid context-delay audit: {directory}")
    records = payload.get("files") or {}
    expected_paths = {
        f"{case.label}/{filename}"
        for case in protocol.EVALUATION_CASES
        for filename in protocol.output_files(case)
    }
    if set(records) != expected_paths:
        raise ValueError(f"incomplete context-delay audit: {directory}")
    for relative, expected in records.items():
        path = directory / relative
        if not path.is_file() or base.file_record(path) != expected:
            raise ValueError(f"context-delay audit file changed: {path}")
    for case in protocol.EVALUATION_CASES:
        _validate_case(
            directory / case.label, env, seed, event_seed, case)
    return payload


def _configure_case(agent, case: protocol.EvaluationCase, event_seed: int):
    kwargs = {}
    if case.context_kind == "shuffled":
        kwargs["shuffled_mode_map"] = protocol.shuffled_mode_map(event_seed)
    if case.context_kind == "delayed":
        kwargs["delay_steps"] = case.delay_steps
    agent.set_eval_context_override(
        case.context_kind, case.fixed_mode_id, **kwargs)


def _evaluate_case(
        env_name: str,
        seed: int,
        event_seed: int,
        case: protocol.EvaluationCase,
        output: Path,
) -> None:
    run_dir = base.bundle_dir(env_name, case.source_role, seed)
    config = final_task_sweep.load_config(run_dir)
    if (config.algo != "regime_sac"
            or config.regime_context_source != case.source_role
            or config.stochastic_mode_family != base.FAMILY):
        raise ValueError(
            f"{case.label} requires a polarity {case.source_role} "
            "regime_sac checkpoint")

    source_env = make_env(config, seed_offset=0)
    train_tasks = source_env.sample_tasks(config.task_num)
    test_tasks = source_env.sample_tasks(config.test_task_num)
    agent = make_algo(
        config.algo, source_env.obs_dim, source_env.act_dim, config)
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
    _configure_case(agent, case, event_seed)

    eval_env = make_env(config, seed_offset=event_seed - seed)
    eval_env.build_rollout_fn(
        nnx.graphdef(agent.policy), direct_policy_context=True)
    task_rows = []
    if case.stationary:
        task_rows = final_task_sweep.evaluate_task_split(
            agent, eval_env, config, test_tasks, "test",
            base.EPISODES_PER_TASK, len(base.MODES),
            20_270_727 + event_seed)
    switching_rows, trace_rows = final_task_sweep.evaluate_switching(
        agent, eval_env, config, test_tasks,
        base.SWITCHING_EPISODES, base.DWELL_STEPS,
        20_280_727 + event_seed)

    shuffled = "|".join(
        str(mode) for mode in protocol.shuffled_mode_map(event_seed))
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
        "eval_delay_steps": (
            0 if case.delay_steps is None else case.delay_steps),
        "eval_shuffled_mode_map": shuffled,
        "heldout_task_stream": "validation",
    }
    for rows in (task_rows, switching_rows, trace_rows):
        for row in rows:
            row.update(run_meta)
    summary_rows = final_task_sweep.summarize(
        run_meta, task_rows, switching_rows, trace_rows,
        float(config.bapr_v2_gate_error_threshold), 50)
    compact_trace = [
        {
            **run_meta,
            **{field: row[field] for field in TRACE_FIELDS},
        }
        for row in trace_rows
    ]

    if case.stationary:
        final_task_sweep.write_csv(
            output / "task_returns.csv", task_rows)
    final_task_sweep.write_csv(
        output / "switching_returns.csv", switching_rows)
    final_task_sweep.write_csv(
        output / "switching_trace.csv", compact_trace)
    final_task_sweep.write_csv(output / "summary.csv", summary_rows)
    _validate_case(output, env_name, seed, event_seed, case)
    if hasattr(eval_env, "close"):
        eval_env.close()
    if hasattr(source_env, "close"):
        source_env.close()


def run(env: str, seed: int, event_seed: int) -> None:
    env = protocol.require_env(env)
    seed = base.require_training_seed(seed)
    event_seed = base.require_event_seed(event_seed)
    bundle_payloads = {
        role: _validate_bundle(env, role, seed)
        for role in base.ROLES
    }
    destination = protocol.audit_dir(env, seed, event_seed)
    manifest = protocol.audit_manifest(env, seed, event_seed)
    if manifest.is_file():
        try:
            validate_audit(env, seed, event_seed)
        except (KeyError, OSError, TypeError, ValueError):
            pass
        else:
            print(f"CONTEXT DELAY AUDIT ALREADY COMPLETE: {destination}")
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
                f"CONTEXT DELAY AUDIT env={env} seed={seed} "
                f"event_seed={event_seed} case={case.label}",
                flush=True)
            _evaluate_case(
                env, seed, event_seed, case, case_output)
            for filename in protocol.output_files(case):
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
    print(f"CONTEXT DELAY AUDIT COMPLETE: {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=protocol.ENVS, required=True)
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
