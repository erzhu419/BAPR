"""Train and publish one corrected RE-SAC/ESCP semantics controller."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from jax_experiments.analysis import resac_escp_semantics_v2 as protocol


COMMON_LOGS = (
    "alpha.npy",
    "critic_loss.npy",
    "eval_reward.npy",
    "eval_reward_std.npy",
    "iteration.npy",
    "policy_loss.npy",
    "q_mean.npy",
    "q_std_mean.npy",
    "total_steps.npy",
)
ROLE_LOGS = {
    "escp": (
        "context_active.npy",
        "context_grad_norm.npy",
        "context_train.npy",
        "critic_grad_norm.npy",
        "finite_update_rate.npy",
        "observed_task_count_estimate.npy",
        "policy_grad_norm.npy",
        "rmdm_loss.npy",
    ),
    "resac": (
        "actor_update_rate.npy",
        "bc_loss.npy",
        "beta_lcb.npy",
        "reg_bonus_mean.npy",
        "reg_bonus_std.npy",
        "resac_independent_ratio.npy",
    ),
}


def training_command(env: str, role: str, seed: int) -> list[str]:
    env = protocol.require_env(env)
    role = protocol.require_train_role(role)
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
        "--hidden_dim", str(protocol.HIDDEN_DIM),
        "--lr", str(protocol.LR),
        "--max_episode_steps", str(protocol.MAX_EPISODE_STEPS),
        "--backend", "spring",
        "--eval_protocol", "stationary",
        "--log_interval", "100",
        "--eval_episodes", "3",
        "--save_interval", "25",
        "--resume",
    ]
    if role == "escp":
        values += [
            "--ensemble_size", str(protocol.ESCP_ENSEMBLE_SIZE),
            "--escp_target_mode", "twin_min",
            "--escp_actor_mode", "twin_min",
            "--escp_context_min_steps",
            str(protocol.ESCP_CONTEXT_MIN_STEPS),
            "--escp_context_min_tasks",
            str(protocol.ESCP_CONTEXT_MIN_TASKS),
            "--escp_alpha_max", str(protocol.ESCP_ALPHA_MAX),
            "--context_warmup_iters", "0",
            "--clip_norm", str(protocol.ESCP_CLIP_NORM),
        ]
    else:
        values += [
            "--ensemble_size", str(protocol.RESAC_ENSEMBLE_SIZE[env]),
            "--beta", str(protocol.RESAC_BETA_START),
            "--weight_reg", "0",
            "--beta_ood", "0",
            "--resac_independent_ratio",
            str(protocol.RESAC_INDEPENDENT_RATIO),
            "--resac_anchor_lambda", str(protocol.RESAC_ANCHOR[env]),
            "--resac_adaptive_beta",
            "--resac_beta_start", str(protocol.RESAC_BETA_START),
            "--resac_beta_end", str(protocol.RESAC_BETA_END[env]),
            "--resac_beta_warmup", str(protocol.RESAC_BETA_WARMUP),
            "--resac_critic_actor_ratio", "1",
            "--resac_beta_bc", "0",
            "--resac_clip_norm", "0",
            "--ema_tau", str(protocol.RESAC_EMA_TAU),
            "--use_ema_eval",
        ]
    return values


def expected_config(env: str, role: str, seed: int) -> dict[str, object]:
    role = protocol.require_train_role(role)
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
        "hidden_dim": protocol.HIDDEN_DIM,
        "lr": protocol.LR,
        "max_episode_steps": protocol.MAX_EPISODE_STEPS,
        "brax_backend": "spring",
    }
    if role == "escp":
        expected.update({
            "ensemble_size": protocol.ESCP_ENSEMBLE_SIZE,
            "escp_target_mode": "twin_min",
            "escp_actor_mode": "twin_min",
            "escp_context_min_steps": protocol.ESCP_CONTEXT_MIN_STEPS,
            "escp_context_min_tasks": protocol.ESCP_CONTEXT_MIN_TASKS,
            "escp_alpha_max": protocol.ESCP_ALPHA_MAX,
            "context_warmup_iters": 0,
            "clip_norm": protocol.ESCP_CLIP_NORM,
            "escp_finite_guard": True,
        })
    else:
        expected.update({
            "ensemble_size": protocol.RESAC_ENSEMBLE_SIZE[env],
            "beta": protocol.RESAC_BETA_START,
            "weight_reg": 0.0,
            "beta_ood": 0.0,
            "resac_independent_ratio": protocol.RESAC_INDEPENDENT_RATIO,
            "resac_anchor_lambda": protocol.RESAC_ANCHOR[env],
            "resac_adaptive_beta": True,
            "resac_beta_start": protocol.RESAC_BETA_START,
            "resac_beta_end": protocol.RESAC_BETA_END[env],
            "resac_beta_warmup": protocol.RESAC_BETA_WARMUP,
            "resac_critic_actor_ratio": 1,
            "resac_beta_bc": 0.0,
            "resac_clip_norm": 0.0,
            "ema_tau": protocol.RESAC_EMA_TAU,
            "use_ema_eval": True,
            "use_ema_rollout": False,
        })
    return expected


def validate_signature(env: str, role: str, seed: int) -> dict:
    path = protocol.run_dir(env, role, seed) / "logs/protocol_signature.json"
    signature = protocol.read_json(path)
    config = signature.get("config") or {}
    mismatches = {
        key: {"actual": config.get(key), "expected": expected}
        for key, expected in expected_config(env, role, seed).items()
        if config.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"semantics config mismatch: {mismatches}")
    return signature


def _required_relative_logs(role: str) -> tuple[Path, ...]:
    role = protocol.require_train_role(role)
    return tuple(
        Path("logs") / name for name in (*COMMON_LOGS, *ROLE_LOGS[role]))


def validate_bundle(env: str, role: str, seed: int) -> dict:
    role = protocol.require_role(role)
    if role == "sac":
        from jax_experiments.analysis.run_resac_paper_fidelity_smoke_v1 import (
            validate_bundle as validate_legacy_bundle,
        )
        return validate_legacy_bundle(env, role, seed)
    directory = protocol.bundle_dir(env, role, seed)
    payload = protocol.read_json(directory / "bundle_manifest.json")
    required = {
        "checkpoints/params.pkl",
        "checkpoints/train_state.pkl",
        "logs/protocol_signature.json",
        *(path.as_posix() for path in _required_relative_logs(role)),
    }
    if (payload.get("schema") != protocol.BUNDLE_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("identity") != protocol.identity(env, role, seed)
            or payload.get("checkpoint")
            != protocol.expected_checkpoint(role)):
        raise ValueError(f"invalid semantics bundle: {directory}")
    records = payload.get("files") or {}
    if set(records) != required:
        raise ValueError(f"incomplete semantics bundle: {directory}")
    for relative, expected in records.items():
        path = directory / relative
        if not path.is_file() or protocol.file_record(path) != expected:
            raise ValueError(f"semantics bundle changed: {path}")
    return payload


def publish_bundle(env: str, role: str, seed: int) -> dict:
    run_dir = protocol.run_dir(env, role, seed)
    checkpoint = protocol.checkpoint_record(run_dir)
    if checkpoint != protocol.expected_checkpoint(role):
        raise ValueError(f"incomplete semantics checkpoint: {checkpoint}")
    validate_signature(env, role, seed)
    destination = protocol.bundle_dir(env, role, seed)
    if (destination / "bundle_manifest.json").is_file():
        return validate_bundle(env, role, seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp.", dir=destination.parent))
    try:
        relatives = (
            Path("checkpoints/params.pkl"),
            Path("checkpoints/train_state.pkl"),
            Path("logs/protocol_signature.json"),
            *_required_relative_logs(role),
        )
        records = {}
        for relative in relatives:
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
    role = protocol.require_train_role(role)
    seed = protocol.require_training_seed(seed)
    manifest = protocol.bundle_manifest(env, role, seed)
    if manifest.is_file():
        validate_bundle(env, role, seed)
        print(f"SEMANTICS TRAIN ALREADY COMPLETE: {manifest}", flush=True)
        return
    command = training_command(env, role, seed)
    print("SEMANTICS TRAIN:", " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(protocol.ROOT)
    subprocess.run(command, cwd=protocol.ROOT, env=environment, check=True)
    payload = publish_bundle(env, role, seed)
    replay = protocol.run_dir(env, role, seed) / "checkpoints/replay_buffer.npz"
    if replay.is_file():
        replay.unlink()
    print("SEMANTICS TRAIN COMPLETE: " + json.dumps(
        payload["identity"], sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=protocol.ENVS, required=True)
    parser.add_argument("--role", choices=protocol.TRAIN_ROLES, required=True)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for checkpoint-safe execution")
    run(args.env, args.role, args.seed)


if __name__ == "__main__":
    main()
