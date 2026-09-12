"""Run strict held-out audits for one numerical-semantics controller."""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from jax_experiments.analysis import resac_escp_semantics_v2 as protocol
from jax_experiments.analysis.run_resac_escp_semantics_v2 import (
    validate_bundle,
)


OUTPUT_FILES = (
    "summary.csv",
    "task_returns.csv",
    "switching_returns.csv",
    "switching_trace.csv",
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _validate_event(
    directory: Path,
    env: str,
    role: str,
    seed: int,
) -> None:
    summary = _rows(directory / "summary.csv")
    task_rows = _rows(directory / "task_returns.csv")
    switching = _rows(directory / "switching_returns.csv")
    trace = _rows(directory / "switching_trace.csv")
    if (len(summary) != 3
            or [(row["metric_group"], row["split"]) for row in summary]
            != [
                ("stationary", "train"),
                ("stationary", "test"),
                ("switching", "test_sequence"),
            ]):
        raise ValueError("semantics audit summary is incomplete")
    train_summary = summary[0]
    if (int(float(train_summary["n_tasks"])) != 0
            or train_summary["return_mean"].lower() != "nan"):
        raise ValueError("semantics audit unexpectedly evaluated train tasks")
    if len(task_rows) != protocol.AUDIT_TASKS:
        raise ValueError("semantics audit held-out task count is wrong")
    if {row["split"] for row in task_rows} != {"test"}:
        raise ValueError("semantics audit leaked training tasks")
    if len(switching) != protocol.AUDIT_SWITCHING_EPISODES:
        raise ValueError("semantics audit switching episodes are incomplete")
    if len(trace) != (
            protocol.AUDIT_SWITCHING_EPISODES
            * protocol.MAX_EPISODE_STEPS):
        raise ValueError("semantics audit switching trace is incomplete")
    for row in (*summary, *task_rows, *switching, *trace):
        if (row.get("algo") != protocol.require_role(role)
                or row.get("env") != protocol.env_slug(env)
                or int(float(row["seed"]))
                != protocol.require_training_seed(seed)
                or int(float(row["checkpoint_next_iter"]))
                != protocol.MAX_ITERS
                or int(float(row["checkpoint_total_steps"]))
                != protocol.FINAL_TOTAL_STEPS):
            raise ValueError("semantics audit provenance is wrong")


def _identity(env: str, role: str, seed: int) -> dict[str, object]:
    return {
        **protocol.identity(env, role, seed),
        "event_seeds": list(protocol.EVENT_SEEDS),
        "heldout_tasks_per_event": protocol.AUDIT_TASKS,
        "episodes_per_task": protocol.AUDIT_EPISODES_PER_TASK,
        "switching_episodes": protocol.AUDIT_SWITCHING_EPISODES,
        "switching_period_steps": protocol.SWITCHING_PERIOD_STEPS,
        "evaluation_policy": "deterministic_mean",
    }


def validate_audit(env: str, role: str, seed: int) -> dict:
    directory = protocol.audit_dir(env, role, seed)
    payload = protocol.read_json(directory / "audit_manifest.json")
    if (payload.get("schema") != protocol.AUDIT_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("identity") != _identity(env, role, seed)):
        raise ValueError(f"invalid semantics audit: {directory}")
    records = payload.get("files") or {}
    expected = {
        f"event_seed_{event_seed}/{filename}"
        for event_seed in protocol.EVENT_SEEDS
        for filename in OUTPUT_FILES
    }
    if set(records) != expected:
        raise ValueError(f"incomplete semantics audit: {directory}")
    for relative, record in records.items():
        path = directory / relative
        if not path.is_file() or protocol.file_record(path) != record:
            raise ValueError(f"semantics audit file changed: {path}")
    for event_seed in protocol.EVENT_SEEDS:
        _validate_event(
            directory / f"event_seed_{event_seed}", env, role, seed)
    return payload


def _command(
    env: str,
    role: str,
    seed: int,
    event_seed: int,
    output: Path,
) -> list[str]:
    run_dir = protocol.bundle_dir(env, role, seed)
    return [
        sys.executable, "-u", "-m",
        "jax_experiments.analysis.final_task_sweep",
        "--run-dir", str(run_dir),
        "--out-dir", str(output),
        "--episodes-per-task", str(protocol.AUDIT_EPISODES_PER_TASK),
        "--max-tasks", str(protocol.AUDIT_TASKS),
        "--switching-episodes", str(protocol.AUDIT_SWITCHING_EPISODES),
        "--switching-period-steps", str(protocol.SWITCHING_PERIOD_STEPS),
        "--heldout-task-stream", "validation",
        "--stationary-test-only",
        "--rng-seed", str(event_seed),
        "--eval-seed-offset", str(event_seed - seed),
        "--min-checkpoint-next-iter", str(protocol.MAX_ITERS),
        "--resume-from", str(run_dir / "checkpoints/train_state.pkl"),
        "--resume",
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
            print(f"SEMANTICS AUDIT ALREADY COMPLETE: {destination}")
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
        commands = []
        records = {}
        for event_seed in protocol.EVENT_SEEDS:
            event_dir = temporary / f"event_seed_{event_seed}"
            command = _command(env, role, seed, event_seed, event_dir)
            commands.append(command)
            print("SEMANTICS AUDIT:", " ".join(command), flush=True)
            subprocess.run(command, cwd=protocol.ROOT, check=True)
            _validate_event(event_dir, env, role, seed)
            for filename in OUTPUT_FILES:
                relative = f"event_seed_{event_seed}/{filename}"
                records[relative] = protocol.file_record(temporary / relative)
        protocol.write_json_atomic(temporary / "audit_manifest.json", {
            "schema": protocol.AUDIT_SCHEMA,
            "status": "complete",
            "identity": _identity(env, role, seed),
            "bundle_manifest": protocol.file_record(
                protocol.bundle_manifest(env, role, seed)),
            "commands": commands,
            "files": records,
        })
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_audit(env, role, seed)
    print(f"SEMANTICS AUDIT COMPLETE: {destination}", flush=True)


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
