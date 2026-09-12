"""Train one fresh controller for the frozen-v5 confirmation graph."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_confirmation_v6 as protocol,
)


def training_command(role: str, seed: int) -> list[str]:
    role = protocol.require_role(role)
    seed = protocol.require_training_seed(seed)
    run_dir = protocol.run_dir(role, seed)
    command = [
        sys.executable, "-u", "-m", "jax_experiments.train",
        "--algo", "sac",
        "--env", protocol.ENV,
        "--seed", str(seed),
        "--max_iters", str(protocol.MAX_ITERS),
        "--save_root", str(run_dir.parent),
        "--run_name", run_dir.name,
        "--env_type", "stochastic_mode",
        "--stochastic_mode_family", protocol.FAMILY,
        "--stochastic_mode_dwell_steps", str(protocol.DWELL_STEPS),
        "--stochastic_mode_dwell_distribution", "fixed",
        "--task_num", "4",
        "--test_task_num", "4",
        "--samples_per_iter", str(protocol.SAMPLES_PER_ITER),
        "--updates_per_iter", str(protocol.UPDATES_PER_ITER),
        "--start_train_steps", str(protocol.START_TRAIN_STEPS),
        "--context_warmup_iters", "50",
        "--ensemble_size", "2",
        "--hidden_dim", "256",
        "--lr", "0.0003",
        "--max_episode_steps", str(protocol.MAX_EPISODE_STEPS),
        "--backend", "spring",
        "--eval_protocol", "stationary",
        "--log_interval", "50",
        "--eval_episodes", "3",
        "--save_interval", "50",
        "--resume",
    ]
    fixed_mode = protocol.role_fixed_mode(role)
    if fixed_mode >= 0:
        command += ["--stochastic_mode_fixed_id", str(fixed_mode)]
    return command


def expected_config(role: str, seed: int) -> dict[str, object]:
    role = protocol.require_role(role)
    return {
        "algo": "sac",
        "env_name": protocol.ENV,
        "seed": protocol.require_training_seed(seed),
        "max_iters": protocol.MAX_ITERS,
        "env_type": "stochastic_mode",
        "stochastic_mode_family": protocol.FAMILY,
        "stochastic_mode_dwell_steps": protocol.DWELL_STEPS,
        "stochastic_mode_dwell_distribution": "fixed",
        "stochastic_mode_fixed_id": protocol.role_fixed_mode(role),
        "task_num": 4,
        "test_task_num": 4,
        "samples_per_iter": protocol.SAMPLES_PER_ITER,
        "updates_per_iter": protocol.UPDATES_PER_ITER,
        "start_train_steps": protocol.START_TRAIN_STEPS,
        "context_warmup_iters": 50,
        "ensemble_size": 2,
        "hidden_dim": 256,
        "lr": 0.0003,
        "max_episode_steps": protocol.MAX_EPISODE_STEPS,
        "brax_backend": "spring",
    }


def validate_signature(role: str, seed: int) -> dict:
    path = protocol.run_dir(role, seed) / "logs/protocol_signature.json"
    signature = protocol.read_json(path)
    config = signature.get("config") or {}
    mismatches = {
        key: {"actual": config.get(key), "expected": expected}
        for key, expected in expected_config(role, seed).items()
        if config.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"confirmation controller config mismatch: {mismatches}")
    return signature


def validate_bundle(role: str, seed: int) -> dict:
    directory = protocol.bundle_dir(role, seed)
    payload = protocol.read_json(directory / "bundle_manifest.json")
    required = {
        "checkpoints/params.pkl",
        "checkpoints/train_state.pkl",
        "logs/protocol_signature.json",
    }
    if (
        payload.get("schema") != protocol.BUNDLE_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("identity") != protocol.identity(role, seed)
        or payload.get("checkpoint") != protocol.expected_checkpoint(role)
    ):
        raise ValueError(f"invalid confirmation controller bundle: {directory}")
    records = payload.get("files") or {}
    if set(records) != required:
        raise ValueError(f"incomplete confirmation controller bundle: {directory}")
    for relative, expected in records.items():
        path = directory / relative
        if not path.is_file() or protocol.file_record(path) != expected:
            raise ValueError(f"confirmation controller bundle changed: {path}")
    return payload


def publish_bundle(role: str, seed: int) -> dict:
    run_dir = protocol.run_dir(role, seed)
    checkpoint = protocol.checkpoint_record(run_dir)
    if checkpoint != protocol.expected_checkpoint(role):
        raise ValueError(f"incomplete confirmation checkpoint: {checkpoint}")
    validate_signature(role, seed)
    destination = protocol.bundle_dir(role, seed)
    if (destination / "bundle_manifest.json").is_file():
        return validate_bundle(role, seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp.", dir=destination.parent))
    try:
        records = {}
        for relative in (
            Path("checkpoints/params.pkl"),
            Path("checkpoints/train_state.pkl"),
            Path("logs/protocol_signature.json"),
        ):
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(run_dir / relative, target)
            records[relative.as_posix()] = protocol.file_record(target)
        protocol.write_json_atomic(temporary / "bundle_manifest.json", {
            "schema": protocol.BUNDLE_SCHEMA,
            "status": "complete",
            "identity": protocol.identity(role, seed),
            "checkpoint": checkpoint,
            "files": records,
        })
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return validate_bundle(role, seed)


def run(role: str, seed: int) -> None:
    role = protocol.require_role(role)
    seed = protocol.require_training_seed(seed)
    manifest = protocol.bundle_manifest(role, seed)
    if manifest.is_file():
        validate_bundle(role, seed)
        print(f"CONFIRMATION CONTROLLER ALREADY COMPLETE: {manifest}")
        return
    command = training_command(role, seed)
    print("CONFIRMATION CONTROLLER TRAIN:", " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(protocol.ROOT)
    subprocess.run(command, cwd=protocol.ROOT, env=environment, check=True)
    payload = publish_bundle(role, seed)
    replay = protocol.run_dir(role, seed) / "checkpoints/replay_buffer.npz"
    if replay.is_file():
        replay.unlink()
    print("CONFIRMATION CONTROLLER COMPLETE: " + json.dumps(
        payload["identity"], sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=protocol.ROLES, required=True)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for checkpoint-safe execution")
    run(args.role, args.seed)


if __name__ == "__main__":
    main()
