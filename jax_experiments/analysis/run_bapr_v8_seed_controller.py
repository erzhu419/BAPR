"""Train and publish one independent controller for BAPR-v8 validation."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from jax_experiments.analysis import bapr_v8_seed_validation as protocol


def training_command(seed: int, role: str, mode: int | None) -> list[str]:
    run_dir = protocol.run_dir(seed, role, mode)
    values = [
        sys.executable, "-u", "-m", "jax_experiments.train",
        "--algo", protocol.expected_algo(role),
        "--env", protocol.ENV,
        "--seed", str(seed),
        "--max_iters", str(protocol.MAX_ITERS),
        "--save_root", str(run_dir.parent),
        "--run_name", run_dir.name,
        "--env_type", "stochastic_mode",
        "--stochastic_mode_family", protocol.FAMILY,
        "--stochastic_mode_dwell_steps", "500",
        "--stochastic_mode_dwell_distribution", "fixed",
        "--task_num", "4",
        "--test_task_num", "4",
        "--samples_per_iter", str(protocol.SAMPLES_PER_ITER),
        "--updates_per_iter", str(protocol.UPDATES_PER_ITER),
        "--start_train_steps", "10000",
        "--ensemble_size", "10",
        "--hidden_dim", "256",
        "--lr", str(protocol.expected_lr(role)),
        "--max_episode_steps", "1000",
        "--eval_protocol", "stationary",
        "--log_interval", "20",
        "--eval_episodes", "3",
        "--save_interval", "50",
        "--resume",
    ]
    if role == "specialist":
        values += ["--stochastic_mode_fixed_id", str(mode)]
    if role == "resac":
        values += [
            "--weight_reg", "0.01",
            "--beta_ood", "0.01",
            "--beta", "-2.0",
        ]
    return values


def _validate_signature(run_dir: Path, seed: int, role: str,
                        mode: int | None) -> dict:
    signature_path = run_dir / "logs" / "protocol_signature.json"
    signature = protocol.read_json(signature_path)
    config = signature.get("config") or {}
    expected = {
        "algo": protocol.expected_algo(role),
        "env_name": protocol.ENV,
        "seed": seed,
        "max_iters": protocol.MAX_ITERS,
        "env_type": "stochastic_mode",
        "stochastic_mode_family": protocol.FAMILY,
        "stochastic_mode_fixed_id": -1 if mode is None else mode,
        "task_num": 4,
        "test_task_num": 4,
        "samples_per_iter": protocol.SAMPLES_PER_ITER,
        "updates_per_iter": protocol.UPDATES_PER_ITER,
        "lr": protocol.expected_lr(role),
    }
    if role == "resac":
        expected.update({"weight_reg": 0.01, "beta_ood": 0.01})
    mismatches = {
        key: {"actual": config.get(key), "expected": value}
        for key, value in expected.items() if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"training config mismatch: {mismatches}")
    return signature


def validate_bundle(seed: int, role: str, mode: int | None) -> dict:
    directory = protocol.bundle_dir(seed, role, mode)
    payload = protocol.read_json(directory / "bundle_manifest.json")
    identity = {
        "seed": seed,
        "role": role,
        "mode": mode,
        "algo": protocol.expected_algo(role),
        "env": protocol.ENV,
        "family": protocol.FAMILY,
    }
    if (payload.get("schema") != protocol.BUNDLE_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("identity") != identity):
        raise ValueError(f"invalid bundle identity: {directory}")
    checkpoint = payload.get("checkpoint") or {}
    expected_checkpoint = {
        "iteration": protocol.FINAL_ITERATION,
        "next_iteration": protocol.MAX_ITERS,
        "total_steps": protocol.FINAL_TOTAL_STEPS,
        "update_count": protocol.FINAL_UPDATE_COUNT,
        "algo": protocol.expected_algo(role),
    }
    if checkpoint != expected_checkpoint:
        raise ValueError(f"invalid bundle checkpoint: {checkpoint}")
    files = payload.get("files") or {}
    required_files = {
        "checkpoints/params.pkl",
        "checkpoints/train_state.pkl",
        "logs/protocol_signature.json",
    }
    if set(files) != required_files:
        raise ValueError(f"invalid bundle file set: {set(files)}")
    for relative, expected in files.items():
        path = directory / relative
        if not path.is_file() or protocol.file_record(path) != expected:
            raise ValueError(f"bundle file changed: {path}")
    return payload


def publish_bundle(seed: int, role: str, mode: int | None) -> dict:
    run_dir = protocol.run_dir(seed, role, mode)
    checkpoint = protocol.checkpoint_record(run_dir)
    expected = {
        "iteration": protocol.FINAL_ITERATION,
        "next_iteration": protocol.MAX_ITERS,
        "total_steps": protocol.FINAL_TOTAL_STEPS,
        "update_count": protocol.FINAL_UPDATE_COUNT,
        "algo": protocol.expected_algo(role),
    }
    if checkpoint != expected:
        raise ValueError(f"incomplete final checkpoint: {checkpoint}")
    _validate_signature(run_dir, seed, role, mode)

    destination = protocol.bundle_dir(seed, role, mode)
    if (destination / "bundle_manifest.json").is_file():
        return validate_bundle(seed, role, mode)
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
            source = run_dir / relative
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            records[relative.as_posix()] = protocol.file_record(target)
        payload = {
            "schema": protocol.BUNDLE_SCHEMA,
            "status": "complete",
            "identity": {
                "seed": seed,
                "role": role,
                "mode": mode,
                "algo": protocol.expected_algo(role),
                "env": protocol.ENV,
                "family": protocol.FAMILY,
            },
            "checkpoint": checkpoint,
            "files": records,
        }
        protocol.write_json_atomic(temporary / "bundle_manifest.json", payload)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return validate_bundle(seed, role, mode)


def run(seed: int, role: str, mode: int | None) -> None:
    protocol.require_seed(seed)
    protocol.require_role(role, mode)
    manifest = protocol.bundle_manifest(seed, role, mode)
    if manifest.is_file():
        validate_bundle(seed, role, mode)
        print(f"BAPR V8 CONTROLLER ALREADY COMPLETE: {manifest}", flush=True)
        return

    command = training_command(seed, role, mode)
    print("BAPR V8 TRAIN:", " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(protocol.ROOT)
    subprocess.run(command, cwd=protocol.ROOT, env=environment, check=True)
    payload = publish_bundle(seed, role, mode)

    # Once the immutable eval bundle exists, the completed replay is no longer
    # needed. This happens only after the final budget and all file hashes pass.
    replay = protocol.run_dir(seed, role, mode) / "checkpoints" / "replay_buffer.npz"
    if replay.is_file():
        replay.unlink()
    print(
        "BAPR V8 CONTROLLER COMPLETE: "
        + json.dumps(payload["identity"], sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True,
                        choices=protocol.TRAINING_SEEDS)
    parser.add_argument("--role", required=True, choices=protocol.ROLES)
    parser.add_argument("--mode", type=int, choices=protocol.MODES)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (args.role == "specialist") != (args.mode is not None):
        parser.error("--mode is required exactly for role=specialist")
    run(args.seed, args.role, args.mode)


if __name__ == "__main__":
    main()
