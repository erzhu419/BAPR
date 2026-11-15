"""Train and publish one paper-aligned JAX baseline smoke controller."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from jax_experiments.analysis import resac_paper_fidelity_smoke_v1 as protocol


def training_command(env: str, role: str, seed: int) -> list[str]:
    env = protocol.require_env(env)
    role = protocol.require_role(role)
    seed = protocol.require_training_seed(seed)
    run_dir = protocol.run_dir(env, role, seed)
    values = [
        sys.executable, "-u", "-m", "jax_experiments.train",
        "--algo", role,
        "--env", env,
        "--seed", str(seed),
        "--max_iters", str(protocol.MAX_ITERS),
        "--save_root", str(run_dir.parent),
        "--run_name", run_dir.name,
        "--env_type", "continuous",
        "--varying_params", "gravity",
        "--task_scale_distribution", "pow1p5",
        "--log_scale_limit", "3.0",
        "--changing_period", "20000",
        "--changing_interval", "4000",
        "--task_num", str(protocol.TASK_NUM),
        "--test_task_num", str(protocol.TEST_TASK_NUM),
        "--samples_per_iter", str(protocol.SAMPLES_PER_ITER),
        "--updates_per_iter", str(protocol.UPDATES_PER_ITER),
        "--start_train_steps", str(protocol.INITIAL_RANDOM_STEPS),
        "--initial_random_steps", str(protocol.INITIAL_RANDOM_STEPS),
        "--ensemble_size", str(protocol.ENSEMBLE_SIZE[role]),
        "--hidden_dim", "256",
        "--lr", str(protocol.LR),
        "--max_episode_steps", str(protocol.MAX_EPISODE_STEPS),
        "--backend", "spring",
        "--eval_protocol", "stationary",
        "--log_interval", "100",
        "--eval_episodes", "3",
        "--save_interval", "25",
        "--resume",
    ]
    if role == "resac":
        values += [
            "--beta", str(protocol.RESAC_BETA),
            "--beta_ood", str(protocol.RESAC_BETA_OOD),
            "--weight_reg", str(protocol.RESAC_WEIGHT_REG),
            "--beta_bc", str(protocol.RESAC_BETA_BC),
            "--critic_actor_ratio", str(protocol.RESAC_CRITIC_ACTOR_RATIO),
            "--clip_norm", str(protocol.CLIP_NORM),
        ]
    return values


def expected_config(env: str, role: str, seed: int) -> dict[str, object]:
    role = protocol.require_role(role)
    expected: dict[str, object] = {
        "algo": role,
        "env_name": protocol.require_env(env),
        "seed": protocol.require_training_seed(seed),
        "max_iters": protocol.MAX_ITERS,
        "env_type": "continuous",
        "varying_params": ["gravity"],
        "task_scale_distribution": "pow1p5",
        "log_scale_limit": 3.0,
        "changing_period": 20_000,
        "changing_interval": 4_000,
        "task_num": protocol.TASK_NUM,
        "test_task_num": protocol.TEST_TASK_NUM,
        "samples_per_iter": protocol.SAMPLES_PER_ITER,
        "updates_per_iter": protocol.UPDATES_PER_ITER,
        "start_train_steps": protocol.INITIAL_RANDOM_STEPS,
        "initial_random_steps": protocol.INITIAL_RANDOM_STEPS,
        "ensemble_size": protocol.ENSEMBLE_SIZE[role],
        "hidden_dim": 256,
        "lr": protocol.LR,
        "max_episode_steps": protocol.MAX_EPISODE_STEPS,
        "brax_backend": "spring",
    }
    if role == "resac":
        expected.update({
            "beta": protocol.RESAC_BETA,
            "beta_ood": protocol.RESAC_BETA_OOD,
            "weight_reg": protocol.RESAC_WEIGHT_REG,
            "beta_bc": protocol.RESAC_BETA_BC,
            "critic_actor_ratio": protocol.RESAC_CRITIC_ACTOR_RATIO,
            "clip_norm": protocol.CLIP_NORM,
        })
    return expected


def validate_signature(env: str, role: str, seed: int) -> dict:
    path = (
        protocol.run_dir(env, role, seed)
        / "logs" / "protocol_signature.json")
    signature = protocol.read_json(path)
    config = signature.get("config") or {}
    mismatches = {
        key: {"actual": config.get(key), "expected": expected}
        for key, expected in expected_config(env, role, seed).items()
        if config.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"fidelity-smoke config mismatch: {mismatches}")
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
            or payload.get("identity") != protocol.identity(env, role, seed)
            or payload.get("checkpoint") != protocol.expected_checkpoint(role)):
        raise ValueError(f"invalid fidelity-smoke bundle: {directory}")
    records = payload.get("files") or {}
    if set(records) != required:
        raise ValueError(f"incomplete fidelity-smoke bundle: {directory}")
    for relative, expected in records.items():
        path = directory / relative
        if not path.is_file() or protocol.file_record(path) != expected:
            raise ValueError(f"fidelity-smoke bundle changed: {path}")
    return payload


def publish_bundle(env: str, role: str, seed: int) -> dict:
    run_dir = protocol.run_dir(env, role, seed)
    checkpoint = protocol.checkpoint_record(run_dir)
    if checkpoint != protocol.expected_checkpoint(role):
        raise ValueError(f"incomplete fidelity-smoke checkpoint: {checkpoint}")
    validate_signature(env, role, seed)
    destination = protocol.bundle_dir(env, role, seed)
    if (destination / "bundle_manifest.json").is_file():
        return validate_bundle(env, role, seed)
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
            "identity": protocol.identity(env, role, seed),
            "checkpoint": checkpoint,
            "files": records,
        })
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
        print(f"FIDELITY SMOKE ALREADY COMPLETE: {manifest}", flush=True)
        return
    command = training_command(env, role, seed)
    print("FIDELITY SMOKE TRAIN:", " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(protocol.ROOT)
    subprocess.run(command, cwd=protocol.ROOT, env=environment, check=True)
    payload = publish_bundle(env, role, seed)
    replay = protocol.run_dir(env, role, seed) / "checkpoints/replay_buffer.npz"
    if replay.is_file():
        replay.unlink()
    print("FIDELITY SMOKE COMPLETE: " + json.dumps(
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
