"""Train and publish one controller for the shared-regime screen."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from jax_experiments.analysis import bapr_regime_screen as protocol


def training_command(role: str) -> list[str]:
    role = protocol.require_role(role)
    run_dir = protocol.run_dir(role)
    values = [
        sys.executable, "-u", "-m", "jax_experiments.train",
        "--algo", protocol.expected_algo(role),
        "--env", protocol.ENV,
        "--seed", str(protocol.TRAINING_SEED),
        "--max_iters", str(protocol.MAX_ITERS),
        "--save_root", str(run_dir.parent),
        "--run_name", run_dir.name,
        "--env_type", "stochastic_mode",
        "--stochastic_mode_family", protocol.FAMILY,
        "--stochastic_mode_dwell_steps", "250",
        "--stochastic_mode_dwell_distribution", "fixed",
        "--task_num", "4", "--test_task_num", "4",
        "--samples_per_iter", str(protocol.SAMPLES_PER_ITER),
        "--updates_per_iter", str(protocol.UPDATES_PER_ITER),
        "--start_train_steps", "4000",
        "--ensemble_size", "10", "--hidden_dim", "256",
        "--lr", str(protocol.expected_lr(role)),
        "--max_episode_steps", "1000",
        "--eval_protocol", "stationary",
        "--log_interval", "50", "--eval_episodes", "3",
        "--save_interval", "50", "--resume",
    ]
    if role == "resac":
        values += [
            "--weight_reg", "0.01", "--beta_ood", "0.01",
            "--beta", "-2.0",
        ]
    if role.startswith("regime_"):
        values += [
            "--bapr_v2_mode", "supervised",
            "--bapr_v2_latent_dim", "4",
            "--bapr_v2_policy_mode", "residual",
            "--bapr_v2_training_schedule", "joint",
            "--bapr_v2_context_hidden_dim", "128",
            "--bapr_v2_context_length", "64",
            "--bapr_v2_context_chunks", "8",
            "--bapr_v2_context_burnin", "16",
            "--bapr_v2_min_history", "16",
            "--bapr_v2_switch_rollout_steps", "250",
            "--bapr_v2_residual_delta", "0.5",
            "--bapr_v2_action_deviation_weight", "0.01",
            "--bapr_v2_base_aux_weight", "0.0",
            "--bapr_v2_context_dropout", "0.1",
            "--bapr_v3_context_ensemble_size", "5",
            "--bapr_v3_variance_model", "mode_empirical",
            "--bapr_v3_variance_ceiling", "0.5",
            "--bapr_v3_variance_ema", "0.05",
            "--bapr_v3_hazard_rate", "0.004",
            "--bapr_v3_evidence_scale", "4.0",
            "--context_warmup_iters", "0",
        ]
        if role == "regime_robust":
            values += [
                "--bapr_v2_base_pretrain_iters", "1401",
                "--bapr_regime_inference_iters", "1",
                "--bapr_regime_adaptation_source", "learned",
            ]
        else:
            values += [
                "--bapr_v2_base_pretrain_iters", "500",
                "--bapr_regime_inference_iters", "200",
                "--bapr_regime_adaptation_source", "oracle",
                "--no_bapr_regime_advantage_fallback",
            ]
    return values


def _expected_config(role: str) -> dict:
    expected = {
        "algo": protocol.expected_algo(role),
        "env_name": protocol.ENV,
        "seed": protocol.TRAINING_SEED,
        "max_iters": protocol.MAX_ITERS,
        "env_type": "stochastic_mode",
        "stochastic_mode_family": protocol.FAMILY,
        "stochastic_mode_dwell_steps": 250,
        "task_num": 4,
        "test_task_num": 4,
        "samples_per_iter": protocol.SAMPLES_PER_ITER,
        "updates_per_iter": protocol.UPDATES_PER_ITER,
        "start_train_steps": 4000,
        "lr": protocol.expected_lr(role),
    }
    if role == "resac":
        expected.update({"weight_reg": 0.01, "beta_ood": 0.01})
    if role.startswith("regime_"):
        expected.update({
            "bapr_v2_mode": "supervised",
            "bapr_v2_policy_mode": "residual",
            "bapr_v3_variance_model": "mode_empirical",
            "bapr_v2_reg_weight": 0.0,
        })
    return expected


def validate_signature(role: str) -> dict:
    signature = protocol.read_json(
        protocol.run_dir(role) / "logs" / "protocol_signature.json")
    config = signature.get("config") or {}
    mismatches = {
        key: {"actual": config.get(key), "expected": value}
        for key, value in _expected_config(role).items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"shared-regime config mismatch: {mismatches}")
    return signature


def validate_bundle(role: str) -> dict:
    directory = protocol.bundle_dir(role)
    payload = protocol.read_json(directory / "bundle_manifest.json")
    required_files = {
        "checkpoints/params.pkl",
        "checkpoints/train_state.pkl",
        "logs/protocol_signature.json",
    }
    expected_identity = {
        "seed": protocol.TRAINING_SEED,
        "role": role,
        "algo": protocol.expected_algo(role),
        "env": protocol.ENV,
        "family": protocol.FAMILY,
    }
    if (payload.get("schema") != protocol.BUNDLE_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("identity") != expected_identity):
        raise ValueError(f"invalid shared-regime bundle: {directory}")
    file_records = payload.get("files") or {}
    if set(file_records) != required_files:
        raise ValueError(
            f"shared-regime bundle has incomplete file set: {directory}")
    for relative, expected in file_records.items():
        path = directory / relative
        if not path.is_file() or protocol.file_record(path) != expected:
            raise ValueError(f"shared-regime bundle file changed: {path}")
    return payload


def publish_bundle(role: str) -> dict:
    run_dir = protocol.run_dir(role)
    checkpoint = protocol.checkpoint_record(run_dir)
    expected_checkpoint = {
        "iteration": protocol.FINAL_ITERATION,
        "next_iteration": protocol.MAX_ITERS,
        "total_steps": protocol.FINAL_TOTAL_STEPS,
        "update_count": protocol.FINAL_UPDATE_COUNT,
        "algo": protocol.expected_algo(role),
    }
    if checkpoint != expected_checkpoint:
        raise ValueError(f"incomplete shared-regime checkpoint: {checkpoint}")
    validate_signature(role)
    destination = protocol.bundle_dir(role)
    if (destination / "bundle_manifest.json").is_file():
        return validate_bundle(role)
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
            "identity": {
                "seed": protocol.TRAINING_SEED,
                "role": role,
                "algo": protocol.expected_algo(role),
                "env": protocol.ENV,
                "family": protocol.FAMILY,
            },
            "checkpoint": checkpoint,
            "files": records,
        }
        protocol.write_json_atomic(
            temporary / "bundle_manifest.json", payload)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return validate_bundle(role)


def run(role: str) -> None:
    role = protocol.require_role(role)
    manifest = protocol.bundle_manifest(role)
    if manifest.is_file():
        validate_bundle(role)
        print(f"SHARED REGIME CONTROLLER ALREADY COMPLETE: {manifest}")
        return
    command = training_command(role)
    print("SHARED REGIME TRAIN:", " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(protocol.ROOT)
    subprocess.run(
        command, cwd=protocol.ROOT, env=environment, check=True)
    payload = publish_bundle(role)
    replay = protocol.run_dir(role) / "checkpoints" / "replay_buffer.npz"
    if replay.is_file():
        replay.unlink()
    print("SHARED REGIME CONTROLLER COMPLETE: " + json.dumps(
        payload["identity"], sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=protocol.ROLES)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.role)


if __name__ == "__main__":
    main()
