"""Train and publish one corrected same-budget polarity baseline."""
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
    regime_polarity_corrected_baselines_v2 as protocol,
)


def training_command(method: str, seed: int) -> list[str]:
    method = protocol.require_trained_method(method)
    seed = protocol.require_seed(seed)
    run = protocol.run_dir(method, seed)
    values = [
        sys.executable, "-u", "-m", "jax_experiments.train",
        "--algo", protocol.algo_for(method),
        "--env", protocol.ENV,
        "--seed", str(seed),
        "--max_iters", str(protocol.MAX_ITERS),
        "--save_root", str(run.parent),
        "--run_name", run.name,
        "--env_type", "stochastic_mode",
        "--stochastic_mode_family", protocol.FAMILY,
        "--stochastic_mode_dwell_steps", str(protocol.DWELL_STEPS),
        "--stochastic_mode_dwell_distribution", "fixed",
        "--task_num", "4",
        "--test_task_num", "4",
        "--samples_per_iter", str(protocol.SAMPLES_PER_ITER),
        "--updates_per_iter", str(protocol.UPDATES_PER_ITER),
        "--start_train_steps", str(protocol.START_TRAIN_STEPS),
        "--initial_random_steps", "0",
        "--hidden_dim", "256",
        "--max_episode_steps", str(protocol.MAX_EPISODE_STEPS),
        "--backend", "spring",
        "--eval_protocol", "stationary",
        "--log_interval", "50",
        "--eval_episodes", "3",
        "--save_interval", "25",
        "--resume",
    ]
    if method == "escp_recurrent":
        config = protocol.ESCP_CONFIG
        values += [
            "--ensemble_size", str(config["ensemble_size"]),
            "--ep_dim", str(config["ep_dim"]),
            "--lr", str(config["policy_lr"]),
            "--clip_norm", str(config["clip_norm"]),
            "--context_warmup_iters", "0",
            "--rmdm_max_tasks", "4",
            "--rbf_radius", str(config["rbf_radius"]),
            "--consistency_loss_weight", str(config["consistency_weight"]),
            "--diversity_loss_weight", str(config["diversity_weight"]),
            "--escp_context_mode", "recurrent",
            "--escp_history_length", str(config["history_length"]),
            "--escp_target_mode", "twin_min",
            "--escp_actor_mode", "twin_min",
            "--escp_context_min_steps", str(config["context_min_steps"]),
            "--escp_context_min_tasks", str(config["context_min_tasks"]),
            "--escp_alpha_max", "1.0",
            "--escp_policy_lr", str(config["policy_lr"]),
            "--escp_critic_lr", str(config["critic_lr"]),
            "--escp_context_lr", str(config["context_lr"]),
            "--escp_alpha_lr", str(config["alpha_lr"]),
            "--escp_target_entropy_ratio",
            str(config["target_entropy_ratio"]),
            "--escp_bottleneck_sigma", str(config["bottleneck_sigma"]),
            "--escp_prototype_tau", str(config["prototype_tau"]),
        ]
    else:
        config = protocol.RESAC_CONFIG
        values += [
            "--ensemble_size", str(config["ensemble_size"]),
            "--lr", str(config["lr"]),
            "--beta", str(config["beta_start"]),
            "--weight_reg", "0",
            "--beta_ood", "0",
            "--resac_independent_ratio", str(config["independent_ratio"]),
            "--resac_anchor_lambda", str(config["anchor_lambda"]),
            "--resac_adaptive_beta",
            "--resac_beta_start", str(config["beta_start"]),
            "--resac_beta_end", str(config["beta_end"]),
            "--resac_beta_warmup", str(config["beta_warmup"]),
            "--resac_critic_actor_ratio", "1",
            "--resac_beta_bc", "0",
            "--resac_clip_norm", "0",
            "--ema_tau", str(config["ema_tau"]),
            "--use_ema_eval",
        ]
    return values


def expected_config(method: str, seed: int) -> dict[str, object]:
    method = protocol.require_trained_method(method)
    expected: dict[str, object] = {
        "algo": protocol.algo_for(method),
        "env_name": protocol.ENV,
        "seed": protocol.require_seed(seed),
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
        "initial_random_steps": 0,
        "hidden_dim": 256,
        "max_episode_steps": protocol.MAX_EPISODE_STEPS,
        "brax_backend": "spring",
    }
    if method == "escp_recurrent":
        config = protocol.ESCP_CONFIG
        expected.update({
            "ensemble_size": config["ensemble_size"],
            "ep_dim": config["ep_dim"],
            "lr": config["policy_lr"],
            "clip_norm": config["clip_norm"],
            "context_warmup_iters": 0,
            "rmdm_max_tasks": 4,
            "rbf_radius": config["rbf_radius"],
            "consistency_loss_weight": config["consistency_weight"],
            "diversity_loss_weight": config["diversity_weight"],
            "escp_context_mode": "recurrent",
            "escp_history_length": config["history_length"],
            "escp_target_mode": "twin_min",
            "escp_actor_mode": "twin_min",
            "escp_context_min_steps": config["context_min_steps"],
            "escp_context_min_tasks": config["context_min_tasks"],
            "escp_alpha_max": 1.0,
            "escp_policy_lr": config["policy_lr"],
            "escp_critic_lr": config["critic_lr"],
            "escp_context_lr": config["context_lr"],
            "escp_alpha_lr": config["alpha_lr"],
            "escp_target_entropy_ratio": config["target_entropy_ratio"],
            "escp_bottleneck_sigma": config["bottleneck_sigma"],
            "escp_prototype_tau": config["prototype_tau"],
        })
    else:
        config = protocol.RESAC_CONFIG
        expected.update({
            "ensemble_size": config["ensemble_size"],
            "lr": config["lr"],
            "beta": config["beta_start"],
            "weight_reg": 0.0,
            "beta_ood": 0.0,
            "resac_independent_ratio": config["independent_ratio"],
            "resac_anchor_lambda": config["anchor_lambda"],
            "resac_adaptive_beta": True,
            "resac_beta_start": config["beta_start"],
            "resac_beta_end": config["beta_end"],
            "resac_beta_warmup": config["beta_warmup"],
            "resac_critic_actor_ratio": 1,
            "resac_beta_bc": 0.0,
            "resac_clip_norm": 0.0,
            "ema_tau": config["ema_tau"],
            "use_ema_eval": True,
            "use_ema_rollout": False,
        })
    return expected


def validate_signature(method: str, seed: int) -> dict:
    path = protocol.run_dir(method, seed) / "logs/protocol_signature.json"
    signature = protocol.read_json(path)
    config = signature.get("config") or {}
    mismatches = {
        key: {"actual": config.get(key), "expected": value}
        for key, value in expected_config(method, seed).items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"corrected baseline config mismatch: {mismatches}")
    return signature


def validate_bundle(method: str, seed: int) -> dict:
    directory = protocol.bundle_dir(method, seed)
    payload = protocol.read_json(directory / "bundle_manifest.json")
    required = {
        "checkpoints/params.pkl",
        "checkpoints/train_state.pkl",
        "logs/protocol_signature.json",
    }
    if (payload.get("schema") != protocol.BUNDLE_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("identity") != protocol.identity(method, seed)
            or payload.get("registration")
            != protocol.file_record(protocol.REGISTRATION_PATH)
            or payload.get("checkpoint")
            != protocol.expected_checkpoint(method)):
        raise ValueError(f"invalid corrected baseline bundle: {directory}")
    records = payload.get("files") or {}
    if set(records) != required:
        raise ValueError(f"incomplete corrected baseline bundle: {directory}")
    for relative, expected in records.items():
        path = directory / relative
        if not path.is_file() or protocol.file_record(path) != expected:
            raise ValueError(f"corrected baseline bundle changed: {path}")
    return payload


def publish_bundle(method: str, seed: int) -> dict:
    run = protocol.run_dir(method, seed)
    checkpoint = protocol.checkpoint_record(run)
    if checkpoint != protocol.expected_checkpoint(method):
        raise ValueError(f"incomplete corrected checkpoint: {checkpoint}")
    validate_signature(method, seed)
    destination = protocol.bundle_dir(method, seed)
    if protocol.bundle_manifest(method, seed).is_file():
        return validate_bundle(method, seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp.", dir=destination.parent))
    try:
        records = {}
        for relative in (
                Path("checkpoints/params.pkl"),
                Path("checkpoints/train_state.pkl"),
                Path("logs/protocol_signature.json")):
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(run / relative, target)
            records[relative.as_posix()] = protocol.file_record(target)
        protocol.write_json_atomic(temporary / "bundle_manifest.json", {
            "schema": protocol.BUNDLE_SCHEMA,
            "status": "complete",
            "identity": protocol.identity(method, seed),
            "checkpoint": checkpoint,
            "registration": protocol.file_record(
                protocol.REGISTRATION_PATH),
            "files": records,
        })
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return validate_bundle(method, seed)


def run(method: str, seed: int) -> None:
    method = protocol.require_trained_method(method)
    seed = protocol.require_seed(seed)
    protocol.validate_registration()
    manifest = protocol.bundle_manifest(method, seed)
    if manifest.is_file():
        validate_bundle(method, seed)
        print(f"CORRECTED BASELINE ALREADY COMPLETE: {manifest}", flush=True)
        return
    command = training_command(method, seed)
    print("CORRECTED BASELINE TRAIN:", " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(protocol.ROOT)
    subprocess.run(command, cwd=protocol.ROOT, env=environment, check=True)
    payload = publish_bundle(method, seed)
    replay = protocol.run_dir(method, seed) / "checkpoints/replay_buffer.npz"
    if replay.is_file():
        replay.unlink()
    print("CORRECTED BASELINE COMPLETE: " + json.dumps(
        payload["identity"], sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", choices=protocol.TRAINED_METHODS, required=True)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS,
        type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for checkpoint-safe execution")
    run(args.method, args.seed)


if __name__ == "__main__":
    main()
