#!/usr/bin/env python3
"""Run one paired strict audit for the shared-regime headroom screen."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from jax_experiments.analysis import bapr_regime_screen as protocol
from jax_experiments.analysis.run_bapr_regime_screen_controller import (
    validate_bundle,
)


EPISODES_PER_TASK = 5
SWITCHING_EPISODES = 5
SWITCHING_PERIOD_STEPS = 250
OUTPUT_FILES = (
    "summary.csv",
    "task_returns.csv",
    "switching_returns.csv",
    "switching_trace.csv",
)


def _multiplier(role: str) -> int:
    return 5 if protocol.require_role(role) == "regime_oracle" else 1


def _expected_rows(role: str) -> dict[str, int]:
    multiplier = _multiplier(role)
    return {
        "summary.csv": 3 * multiplier,
        "task_returns.csv": 4 * multiplier,
        "switching_returns.csv": SWITCHING_EPISODES * multiplier,
        "switching_trace.csv": (
            SWITCHING_EPISODES * 1000 * multiplier),
    }


def _context_args(role: str) -> list[str]:
    role = protocol.require_role(role)
    if role == "regime_robust":
        return [
            "--bapr-v2-context-source", "robust",
            "--bapr-v2-advantage", "off",
        ]
    if role == "regime_oracle":
        return [
            "--bapr-v2-context-source", "oracle",
            "--bapr-v2-advantage", "off",
            "--oracle-context-ladder",
        ]
    return [
        "--bapr-v2-context-source", "checkpoint",
        "--bapr-v2-advantage", "off",
    ]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _expected_mode_labels(role: str) -> set[str]:
    if role == "regime_oracle":
        return {"dynamic", "0", "1", "2", "3"}
    return {"dynamic"}


def _validate_csv_outputs(directory: Path, role: str) -> None:
    expected_run_name = role
    expected_algo = protocol.expected_algo(role)
    expected_modes = _expected_mode_labels(role)
    for filename, count in _expected_rows(role).items():
        rows = _read_rows(directory / filename)
        if len(rows) != count:
            raise ValueError(
                f"{filename} has {len(rows)} rows, expected {count}")
        if {
            str(row.get("eval_oracle_mode_id")) for row in rows
        } != expected_modes:
            raise ValueError(f"{filename} has wrong oracle context ladder")
        for row in rows:
            if row.get("run_name") != expected_run_name:
                raise ValueError(f"{filename} has wrong run_name")
            if row.get("algo") != expected_algo:
                raise ValueError(f"{filename} has wrong algorithm")
            if int(float(row["checkpoint_next_iter"])) != protocol.MAX_ITERS:
                raise ValueError(f"{filename} has stale checkpoint iteration")
            if (int(float(row["checkpoint_total_steps"]))
                    != protocol.FINAL_TOTAL_STEPS):
                raise ValueError(f"{filename} has stale checkpoint budget")

    task_rows = _read_rows(directory / "task_returns.csv")
    for label in expected_modes:
        rows = [
            row for row in task_rows
            if str(row["eval_oracle_mode_id"]) == label]
        modes = {int(float(row["mode_id_mean"])) for row in rows}
        if modes != {0, 1, 2, 3}:
            raise ValueError(
                f"stationary task matrix is incomplete for context {label}")


def _manifest_identity(role: str, event_seed: int) -> dict[str, object]:
    return {
        "role": protocol.require_role(role),
        "event_seed": protocol.require_audit_event_seed(event_seed),
        "env": protocol.ENV,
        "family": protocol.FAMILY,
        "training_seed": protocol.TRAINING_SEED,
        "episodes_per_task": EPISODES_PER_TASK,
        "switching_episodes": SWITCHING_EPISODES,
        "switching_period_steps": SWITCHING_PERIOD_STEPS,
    }


def validate_audit(role: str, event_seed: int) -> dict[str, object]:
    directory = protocol.audit_dir(role, event_seed)
    payload = protocol.read_json(directory / "audit_manifest.json")
    if (payload.get("schema") != protocol.AUDIT_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("identity") != _manifest_identity(
                role, event_seed)):
        raise ValueError(f"invalid shared-regime audit: {directory}")
    records = payload.get("files") or {}
    if set(records) != set(OUTPUT_FILES):
        raise ValueError(f"incomplete shared-regime audit: {directory}")
    for filename, expected in records.items():
        path = directory / filename
        if not path.is_file() or protocol.file_record(path) != expected:
            raise ValueError(f"shared-regime audit file changed: {path}")
    _validate_csv_outputs(directory, role)
    return payload


def _evaluation_command(role: str, event_seed: int, output: Path) -> list[str]:
    run_dir = protocol.bundle_dir(role)
    return [
        sys.executable, "-u", "-m",
        "jax_experiments.analysis.final_task_sweep",
        "--run-dir", str(run_dir),
        "--out-dir", str(output),
        "--episodes-per-task", str(EPISODES_PER_TASK),
        "--switching-episodes", str(SWITCHING_EPISODES),
        "--switching-period-steps", str(SWITCHING_PERIOD_STEPS),
        "--heldout-task-stream", "validation",
        "--detection-window-steps", "50",
        *_context_args(role),
        "--rng-seed", str(20260721 + event_seed),
        "--eval-seed-offset", str(event_seed),
        "--max-tasks", "4",
        "--stationary-test-only",
        "--min-checkpoint-next-iter", str(protocol.MAX_ITERS),
        "--resume-from", str(
            run_dir / "checkpoints" / "train_state.pkl"),
    ]


def run(role: str, event_seed: int) -> None:
    role = protocol.require_role(role)
    event_seed = protocol.require_audit_event_seed(event_seed)
    validate_bundle(role)
    destination = protocol.audit_dir(role, event_seed)
    if protocol.audit_manifest(role, event_seed).is_file():
        try:
            validate_audit(role, event_seed)
        except (KeyError, OSError, TypeError, ValueError):
            pass
        else:
            print(f"SHARED REGIME AUDIT ALREADY COMPLETE: {destination}")
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
        command = _evaluation_command(role, event_seed, temporary)
        print("SHARED REGIME AUDIT:", " ".join(command), flush=True)
        subprocess.run(command, cwd=protocol.ROOT, check=True)
        _validate_csv_outputs(temporary, role)
        bundle_payload = validate_bundle(role)
        payload = {
            "schema": protocol.AUDIT_SCHEMA,
            "status": "complete",
            "identity": _manifest_identity(role, event_seed),
            "bundle_manifest": protocol.file_record(
                protocol.bundle_manifest(role)),
            "bundle_checkpoint": bundle_payload["checkpoint"],
            "command": command,
            "files": {
                filename: protocol.file_record(temporary / filename)
                for filename in OUTPUT_FILES
            },
        }
        protocol.write_json_atomic(
            temporary / "audit_manifest.json", payload)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_audit(role, event_seed)
    print(f"SHARED REGIME AUDIT COMPLETE: {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=protocol.AUDIT_ROLES, required=True)
    parser.add_argument(
        "--event-seed", choices=protocol.AUDIT_EVENT_SEEDS,
        type=int, required=True)
    parser.add_argument(
        "--resume", action="store_true",
        help="Declarative scheduler staging marker; evaluation never trains.")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for scheduler input staging")
    run(args.role, args.event_seed)


if __name__ == "__main__":
    main()
