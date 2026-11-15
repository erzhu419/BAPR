"""Train and publish one same-protocol final baseline controller."""
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
    regime_polarity_fallback_final_comparison_v1 as protocol,
)


def training_command(role: str, seed: int) -> list[str]:
    role = protocol.require_baseline_role(role)
    seed = protocol.require_training_seed(seed)
    run_dir = protocol.baseline_run_dir(role, seed)
    values = [
        sys.executable, "-u", "-m", "jax_experiments.train",
        "--algo", role,
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
        "--ensemble_size", "10",
        "--hidden_dim", "256",
        "--lr", str(protocol.BASELINE_LR[role]),
        "--max_episode_steps", str(protocol.MAX_EPISODE_STEPS),
        "--eval_protocol", "stationary",
        "--log_interval", "50",
        "--eval_episodes", "3",
        "--save_interval", "50",
        "--resume",
    ]
    if role == "resac":
        values += [
            "--weight_reg", str(protocol.RESAC_WEIGHT_REG),
            "--beta_ood", str(protocol.RESAC_BETA_OOD),
            "--beta", str(protocol.RESAC_BETA),
        ]
    return values


def expected_config(role: str, seed: int) -> dict[str, object]:
    role = protocol.require_baseline_role(role)
    expected: dict[str, object] = {
        "algo": role,
        "env_name": protocol.ENV,
        "seed": protocol.require_training_seed(seed),
        "max_iters": protocol.MAX_ITERS,
        "env_type": "stochastic_mode",
        "stochastic_mode_family": protocol.FAMILY,
        "stochastic_mode_dwell_steps": protocol.DWELL_STEPS,
        "stochastic_mode_dwell_distribution": "fixed",
        "task_num": 4,
        "test_task_num": 4,
        "samples_per_iter": protocol.SAMPLES_PER_ITER,
        "updates_per_iter": protocol.UPDATES_PER_ITER,
        "start_train_steps": protocol.START_TRAIN_STEPS,
        "ensemble_size": 10,
        "hidden_dim": 256,
        "lr": protocol.BASELINE_LR[role],
        "max_episode_steps": protocol.MAX_EPISODE_STEPS,
    }
    if role == "resac":
        expected.update({
            "weight_reg": protocol.RESAC_WEIGHT_REG,
            "beta_ood": protocol.RESAC_BETA_OOD,
            "beta": protocol.RESAC_BETA,
        })
    return expected


def validate_signature(role: str, seed: int) -> dict:
    path = (
        protocol.baseline_run_dir(role, seed)
        / "logs" / "protocol_signature.json")
    signature = protocol.read_json(path)
    config = signature.get("config") or {}
    mismatches = {
        key: {"actual": config.get(key), "expected": value}
        for key, value in expected_config(role, seed).items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"final baseline config mismatch: {mismatches}")
    return signature


def validate_bundle(role: str, seed: int) -> dict:
    directory = protocol.baseline_bundle_dir(role, seed)
    payload = protocol.read_json(directory / "bundle_manifest.json")
    required = {
        "checkpoints/params.pkl",
        "checkpoints/train_state.pkl",
        "logs/protocol_signature.json",
    }
    if (payload.get("schema") != protocol.BASELINE_BUNDLE_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("identity")
            != protocol.baseline_identity(role, seed)):
        raise ValueError(f"invalid final baseline bundle: {directory}")
    records = payload.get("files") or {}
    if set(records) != required:
        raise ValueError(f"incomplete final baseline bundle: {directory}")
    for relative, expected in records.items():
        path = directory / relative
        if not path.is_file() or protocol.file_record(path) != expected:
            raise ValueError(f"final baseline file changed: {path}")
    expected_checkpoint = {
        "iteration": protocol.FINAL_ITERATION,
        "next_iteration": protocol.MAX_ITERS,
        "total_steps": protocol.FINAL_TOTAL_STEPS,
        "update_count": protocol.FINAL_UPDATE_COUNT,
        "algo": protocol.require_baseline_role(role),
    }
    if payload.get("checkpoint") != expected_checkpoint:
        raise ValueError(f"wrong final baseline budget: {directory}")
    return payload


def publish_bundle(role: str, seed: int) -> dict:
    role = protocol.require_baseline_role(role)
    seed = protocol.require_training_seed(seed)
    run_dir = protocol.baseline_run_dir(role, seed)
    checkpoint = protocol.checkpoint_record(run_dir)
    expected = {
        "iteration": protocol.FINAL_ITERATION,
        "next_iteration": protocol.MAX_ITERS,
        "total_steps": protocol.FINAL_TOTAL_STEPS,
        "update_count": protocol.FINAL_UPDATE_COUNT,
        "algo": role,
    }
    if checkpoint != expected:
        raise ValueError(f"incomplete final baseline checkpoint: {checkpoint}")
    validate_signature(role, seed)

    destination = protocol.baseline_bundle_dir(role, seed)
    if (destination / "bundle_manifest.json").is_file():
        return validate_bundle(role, seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp.", dir=destination.parent))
    try:
        files = (
            Path("checkpoints") / "params.pkl",
            Path("checkpoints") / "train_state.pkl",
            Path("logs") / "protocol_signature.json",
        )
        records = {}
        for relative in files:
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(run_dir / relative, target)
            records[relative.as_posix()] = protocol.file_record(target)
        payload = {
            "schema": protocol.BASELINE_BUNDLE_SCHEMA,
            "status": "complete",
            "identity": protocol.baseline_identity(role, seed),
            "checkpoint": checkpoint,
            "files": records,
            "registration": protocol.FROZEN_REGISTRATION_RECORD,
        }
        protocol.write_json_atomic(
            temporary / "bundle_manifest.json", payload)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return validate_bundle(role, seed)


def run(role: str, seed: int) -> None:
    role = protocol.require_baseline_role(role)
    seed = protocol.require_training_seed(seed)
    protocol.validate_registration()
    manifest = protocol.baseline_bundle_manifest(role, seed)
    if manifest.is_file():
        validate_bundle(role, seed)
        print(f"FINAL BASELINE ALREADY COMPLETE: {manifest}", flush=True)
        return
    command = training_command(role, seed)
    print("FINAL BASELINE TRAIN:", " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(protocol.ROOT)
    subprocess.run(
        command, cwd=protocol.ROOT, env=environment, check=True)
    payload = publish_bundle(role, seed)
    replay = (
        protocol.baseline_run_dir(role, seed)
        / "checkpoints" / "replay_buffer.npz")
    if replay.is_file():
        replay.unlink()
    print("FINAL BASELINE COMPLETE: " + json.dumps(
        payload["identity"], sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=protocol.BASELINE_ROLES, required=True)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for checkpoint-safe execution")
    run(args.role, args.seed)


if __name__ == "__main__":
    main()
