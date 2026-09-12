"""Run five paired CPU event-stream audits for one headroom checkpoint."""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from jax_experiments.analysis import regime_control_headroom as protocol
from jax_experiments.analysis.run_regime_control_headroom_controller import (
    validate_bundle,
)


OUTPUT_FILES = (
    "summary.csv",
    "task_returns.csv",
    "switching_returns.csv",
    "switching_trace.csv",
)
EXPECTED_ROWS = {
    "summary.csv": 3,
    "task_returns.csv": len(protocol.MODES),
    "switching_returns.csv": protocol.SWITCHING_EPISODES,
    "switching_trace.csv": (
        protocol.SWITCHING_EPISODES * protocol.MAX_EPISODE_STEPS),
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _validate_csv_outputs(
        directory: Path, env: str, role: str, seed: int) -> None:
    expected_run_name = protocol.run_dir(env, "robust", seed).name
    expected_env = protocol.env_slug(env)
    for filename, count in EXPECTED_ROWS.items():
        rows = _read_rows(directory / filename)
        if len(rows) != count:
            raise ValueError(
                f"{filename} has {len(rows)} rows, expected {count}")
        for row in rows:
            if row.get("run_name") != expected_run_name:
                raise ValueError(f"{filename} has wrong run_name")
            if row.get("algo") != "regime_sac":
                raise ValueError(f"{filename} has wrong algorithm")
            if row.get("env") != expected_env:
                raise ValueError(f"{filename} has wrong environment")
            if str(row.get("eval_oracle_mode_id")) != "dynamic":
                raise ValueError(f"{filename} has non-dynamic context")
            if int(float(row["checkpoint_next_iter"])) != protocol.MAX_ITERS:
                raise ValueError(f"{filename} has stale checkpoint iteration")
            if (int(float(row["checkpoint_total_steps"]))
                    != protocol.FINAL_TOTAL_STEPS):
                raise ValueError(f"{filename} has stale checkpoint budget")

    task_rows = _read_rows(directory / "task_returns.csv")
    modes = {int(float(row["mode_id_mean"])) for row in task_rows}
    if modes != set(protocol.MODES):
        raise ValueError(f"stationary mode sweep is incomplete: {modes}")
    if {row["split"] for row in task_rows} != {"test"}:
        raise ValueError("stationary audit must use the held-out/test split")
    switching_rows = _read_rows(directory / "switching_returns.csv")
    if {row["metric_semantics"] for row in switching_rows} != {
            "fixed_horizon_stream_sum"}:
        raise ValueError("switching audit is not strict fixed-horizon")

    trace_rows = _read_rows(directory / "switching_trace.csv")
    robust_context_mode_id = getattr(
        protocol, "ROBUST_TRACE_CONTEXT_MODE_ID", None)
    if role == "robust" and robust_context_mode_id is not None:
        if any(
                int(row["action_task_id"]) != robust_context_mode_id
                for row in trace_rows):
            raise ValueError(
                "robust action context is not the sealed all-zero input")
    elif any(
            int(row["action_task_id"])
            != int(row["physics_action_task_id"])
            for row in trace_rows):
        raise ValueError(
            f"{role} action context is misaligned with the physics mode")



def _identity(env: str, role: str, seed: int) -> dict[str, object]:
    return {
        **protocol.identity(env, role, seed),
        "event_seeds": list(protocol.AUDIT_EVENT_SEEDS),
        "episodes_per_task": protocol.EPISODES_PER_TASK,
        "switching_episodes": protocol.SWITCHING_EPISODES,
        "switching_period_steps": protocol.DWELL_STEPS,
        "effective_eval_seed_rule": "event_seed",
    }


def validate_audit(env: str, role: str, seed: int) -> dict[str, object]:
    env = protocol.require_env(env)
    role = protocol.require_role(role)
    seed = protocol.require_training_seed(seed)
    directory = protocol.audit_dir(env, role, seed)
    payload = protocol.read_json(directory / "audit_manifest.json")
    if (payload.get("schema") != protocol.AUDIT_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("identity") != _identity(env, role, seed)):
        raise ValueError(f"invalid headroom audit: {directory}")
    records = payload.get("files") or {}
    expected_paths = {
        f"event_seed_{event_seed}/{filename}"
        for event_seed in protocol.AUDIT_EVENT_SEEDS
        for filename in OUTPUT_FILES
    }
    if set(records) != expected_paths:
        raise ValueError(f"incomplete headroom audit: {directory}")
    for relative, expected in records.items():
        path = directory / relative
        if not path.is_file() or protocol.file_record(path) != expected:
            raise ValueError(f"headroom audit file changed: {path}")
    for event_seed in protocol.AUDIT_EVENT_SEEDS:
        _validate_csv_outputs(
            directory / f"event_seed_{event_seed}", env, role, seed)
    return payload


def _evaluation_command(
        env: str, role: str, seed: int, event_seed: int,
        output: Path) -> list[str]:
    run_dir = protocol.bundle_dir(env, role, seed)
    # make_env uses config.seed + eval_seed_offset. Subtracting the training
    # seed makes every robust/oracle/training-seed arm see the exact same
    # physical event stream for a given sealed event_seed.
    eval_seed_offset = event_seed - seed
    return [
        sys.executable, "-u", "-m",
        "jax_experiments.analysis.final_task_sweep",
        "--run-dir", str(run_dir),
        "--out-dir", str(output),
        "--episodes-per-task", str(protocol.EPISODES_PER_TASK),
        "--switching-episodes", str(protocol.SWITCHING_EPISODES),
        "--switching-period-steps", str(protocol.DWELL_STEPS),
        "--heldout-task-stream", "validation",
        "--detection-window-steps", "50",
        "--bapr-v2-context-source", "checkpoint",
        "--bapr-v2-advantage", "off",
        "--rng-seed", str(20_260_722 + event_seed),
        "--eval-seed-offset", str(eval_seed_offset),
        "--max-tasks", "4",
        "--stationary-test-only",
        "--min-checkpoint-next-iter", str(protocol.MAX_ITERS),
        "--resume-from", str(
            run_dir / "checkpoints" / "train_state.pkl"),
    ]


def run(env: str, role: str, seed: int) -> None:
    env = protocol.require_env(env)
    role = protocol.require_role(role)
    seed = protocol.require_training_seed(seed)
    validate_bundle(env, role, seed)
    destination = protocol.audit_dir(env, role, seed)
    if protocol.audit_manifest(env, role, seed).is_file():
        try:
            validate_audit(env, role, seed)
        except (KeyError, OSError, TypeError, ValueError):
            pass
        else:
            print(f"HEADROOM AUDIT ALREADY COMPLETE: {destination}")
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
        commands = []
        for event_seed in protocol.AUDIT_EVENT_SEEDS:
            event_dir = temporary / f"event_seed_{event_seed}"
            command = _evaluation_command(
                env, role, seed, event_seed, event_dir)
            commands.append(command)
            print("HEADROOM AUDIT:", " ".join(command), flush=True)
            subprocess.run(command, cwd=protocol.ROOT, check=True)
            _validate_csv_outputs(event_dir, env, role, seed)
            for filename in OUTPUT_FILES:
                relative = f"event_seed_{event_seed}/{filename}"
                records[relative] = protocol.file_record(
                    temporary / relative)
        bundle_payload = validate_bundle(env, role, seed)
        payload = {
            "schema": protocol.AUDIT_SCHEMA,
            "status": "complete",
            "identity": _identity(env, role, seed),
            "bundle_manifest": protocol.file_record(
                protocol.bundle_manifest(env, role, seed)),
            "bundle_checkpoint": bundle_payload["checkpoint"],
            "commands": commands,
            "files": records,
        }
        protocol.write_json_atomic(
            temporary / "audit_manifest.json", payload)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_audit(env, role, seed)
    print(f"HEADROOM AUDIT COMPLETE: {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=protocol.ENVS, required=True)
    parser.add_argument("--role", choices=protocol.ROLES, required=True)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for scheduler input staging")
    run(args.env, args.role, args.seed)


if __name__ == "__main__":
    main()
