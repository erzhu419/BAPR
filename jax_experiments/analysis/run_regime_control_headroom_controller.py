"""Train and publish one equal-budget robust/oracle regime controller."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from jax_experiments.analysis import regime_control_headroom as protocol


def training_command(env: str, role: str, seed: int) -> list[str]:
    env = protocol.require_env(env)
    role = protocol.require_role(role)
    seed = protocol.require_training_seed(seed)
    run_dir = protocol.run_dir(env, role, seed)
    return [
        sys.executable, "-u", "-m", "jax_experiments.train",
        "--algo", "regime_sac",
        "--env", env,
        "--seed", str(seed),
        "--max_iters", str(protocol.MAX_ITERS),
        "--save_root", str(run_dir.parent),
        "--run_name", run_dir.name,
        "--env_type", "stochastic_mode",
        "--stochastic_mode_family", protocol.FAMILY,
        "--stochastic_mode_dwell_steps", str(protocol.DWELL_STEPS),
        "--stochastic_mode_dwell_distribution", "fixed",
        "--regime_context_source", role,
        "--task_num", "4",
        "--test_task_num", "4",
        "--samples_per_iter", str(protocol.SAMPLES_PER_ITER),
        "--updates_per_iter", str(protocol.UPDATES_PER_ITER),
        "--start_train_steps", "4000",
        "--context_warmup_iters", "0",
        "--ensemble_size", "10",
        "--hidden_dim", "256",
        "--lr", "0.0003",
        "--max_episode_steps", str(protocol.MAX_EPISODE_STEPS),
        "--eval_protocol", "stationary",
        "--log_interval", "50",
        "--eval_episodes", "3",
        "--save_interval", "50",
        "--resume",
    ]


def _expected_config(env: str, role: str, seed: int) -> dict[str, object]:
    return {
        "algo": "regime_sac",
        "env_name": protocol.require_env(env),
        "seed": protocol.require_training_seed(seed),
        "max_iters": protocol.MAX_ITERS,
        "env_type": "stochastic_mode",
        "stochastic_mode_family": protocol.FAMILY,
        "stochastic_mode_dwell_steps": protocol.DWELL_STEPS,
        "stochastic_mode_dwell_distribution": "fixed",
        "regime_context_source": protocol.require_role(role),
        "task_num": 4,
        "test_task_num": 4,
        "samples_per_iter": protocol.SAMPLES_PER_ITER,
        "updates_per_iter": protocol.UPDATES_PER_ITER,
        "start_train_steps": 4000,
        "context_warmup_iters": 0,
        "ensemble_size": 10,
        "hidden_dim": 256,
        "lr": 0.0003,
        "max_episode_steps": protocol.MAX_EPISODE_STEPS,
    }


def validate_signature(env: str, role: str, seed: int) -> dict:
    signature = protocol.read_json(
        protocol.run_dir(env, role, seed)
        / "logs" / "protocol_signature.json")
    config = signature.get("config") or {}
    expected = _expected_config(env, role, seed)
    mismatches = {
        key: {"actual": config.get(key), "expected": value}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"headroom config mismatch: {mismatches}")
    return signature


def validate_bundle(env: str, role: str, seed: int) -> dict:
    directory = protocol.bundle_dir(env, role, seed)
    payload = protocol.read_json(directory / "bundle_manifest.json")
    required = {
        "checkpoints/params.pkl",
        "checkpoints/train_state.pkl",
        "logs/protocol_signature.json",
    }
    if (payload.get("schema") != protocol.BUNDLE_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("identity") != protocol.identity(env, role, seed)):
        raise ValueError(f"invalid headroom bundle: {directory}")
    records = payload.get("files") or {}
    if set(records) != required:
        raise ValueError(f"incomplete headroom bundle: {directory}")
    for relative, expected in records.items():
        path = directory / relative
        if not path.is_file() or protocol.file_record(path) != expected:
            raise ValueError(f"headroom bundle file changed: {path}")
    expected_checkpoint = {
        "iteration": protocol.FINAL_ITERATION,
        "next_iteration": protocol.MAX_ITERS,
        "total_steps": protocol.FINAL_TOTAL_STEPS,
        "update_count": protocol.FINAL_UPDATE_COUNT,
        "algo": "regime_sac",
    }
    if payload.get("checkpoint") != expected_checkpoint:
        raise ValueError(f"wrong headroom checkpoint budget: {directory}")
    return payload


def publish_bundle(env: str, role: str, seed: int) -> dict:
    run_dir = protocol.run_dir(env, role, seed)
    checkpoint = protocol.checkpoint_record(run_dir)
    expected = {
        "iteration": protocol.FINAL_ITERATION,
        "next_iteration": protocol.MAX_ITERS,
        "total_steps": protocol.FINAL_TOTAL_STEPS,
        "update_count": protocol.FINAL_UPDATE_COUNT,
        "algo": "regime_sac",
    }
    if checkpoint != expected:
        raise ValueError(f"incomplete headroom checkpoint: {checkpoint}")
    validate_signature(env, role, seed)
    destination = protocol.bundle_dir(env, role, seed)
    if (destination / "bundle_manifest.json").is_file():
        return validate_bundle(env, role, seed)
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
            "schema": protocol.BUNDLE_SCHEMA,
            "status": "complete",
            "identity": protocol.identity(env, role, seed),
            "checkpoint": checkpoint,
            "files": records,
        }
        protocol.write_json_atomic(
            temporary / "bundle_manifest.json", payload)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return validate_bundle(env, role, seed)


def run(env: str, role: str, seed: int) -> None:
    env = protocol.require_env(env)
    role = protocol.require_role(role)
    seed = protocol.require_training_seed(seed)
    manifest = protocol.bundle_manifest(env, role, seed)
    if manifest.is_file():
        validate_bundle(env, role, seed)
        print(f"HEADROOM CONTROLLER ALREADY COMPLETE: {manifest}")
        return
    command = training_command(env, role, seed)
    print("HEADROOM TRAIN:", " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(protocol.ROOT)
    subprocess.run(
        command, cwd=protocol.ROOT, env=environment, check=True)
    payload = publish_bundle(env, role, seed)
    replay = protocol.run_dir(env, role, seed) / "checkpoints/replay_buffer.npz"
    if replay.is_file():
        replay.unlink()
    print("HEADROOM CONTROLLER COMPLETE: " + json.dumps(
        payload["identity"], sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=protocol.ENVS, required=True)
    parser.add_argument("--role", choices=protocol.ROLES, required=True)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for checkpoint-safe execution")
    run(args.env, args.role, args.seed)


if __name__ == "__main__":
    main()
