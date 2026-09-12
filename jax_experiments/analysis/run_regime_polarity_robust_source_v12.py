"""Train one fresh robust SAC source and publish a policy-only bundle."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from flax import nnx

from jax_experiments.analysis import final_task_sweep
from jax_experiments.analysis import (
    regime_polarity_robust_warmstart_confirmation_v12 as protocol,
)
from jax_experiments.common.checkpoint import _to_numpy_tree, load_checkpoint
from jax_experiments.common.logging import Logger
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.train import make_algo, make_env


def training_command(seed: int) -> list[str]:
    seed = protocol.require_training_seed(seed)
    run_dir = protocol.source_run_dir(seed)
    return [
        sys.executable, "-u", "-m", "jax_experiments.train",
        "--algo", "sac",
        "--env", protocol.ENV,
        "--seed", str(seed),
        "--max_iters", str(protocol.SOURCE_NEXT_ITERATION),
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
        "--start_train_steps", str(protocol.SOURCE_START_TRAIN_STEPS),
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


def expected_config(seed: int) -> dict[str, Any]:
    return {
        "algo": "sac",
        "env_name": protocol.ENV,
        "seed": protocol.require_training_seed(seed),
        "max_iters": protocol.SOURCE_NEXT_ITERATION,
        "env_type": "stochastic_mode",
        "stochastic_mode_family": protocol.FAMILY,
        "stochastic_mode_dwell_steps": protocol.DWELL_STEPS,
        "stochastic_mode_dwell_distribution": "fixed",
        "stochastic_mode_fixed_id": -1,
        "task_num": 4,
        "test_task_num": 4,
        "samples_per_iter": protocol.SAMPLES_PER_ITER,
        "updates_per_iter": protocol.UPDATES_PER_ITER,
        "start_train_steps": protocol.SOURCE_START_TRAIN_STEPS,
        "context_warmup_iters": 50,
        "ensemble_size": 2,
        "hidden_dim": 256,
        "lr": 0.0003,
        "max_episode_steps": protocol.MAX_EPISODE_STEPS,
        "brax_backend": "spring",
    }


def validate_signature(seed: int) -> dict[str, Any]:
    path = protocol.source_run_dir(seed) / "logs/protocol_signature.json"
    signature = protocol.read_json(path)
    config = signature.get("config") or {}
    mismatches = {
        key: {"actual": config.get(key), "expected": expected}
        for key, expected in expected_config(seed).items()
        if config.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"v12 robust source config mismatch: {mismatches}")
    return signature


def validate_bundle(seed: int) -> dict[str, Any]:
    seed = protocol.require_training_seed(seed)
    directory = protocol.source_bundle(seed)
    payload = protocol.read_json(protocol.source_manifest(seed))
    expected_files = {
        "policy/" + protocol.POLICY_NAME,
        "logs/protocol_signature.json",
    }
    if (
        payload.get("schema") != protocol.SOURCE_BUNDLE_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("identity") != protocol.source_identity(seed)
        or payload.get("checkpoint") != protocol.expected_source_checkpoint()
        or set(payload.get("files") or {}) != expected_files
    ):
        raise ValueError(f"invalid v12 robust source bundle: {directory}")
    for relative, record in payload["files"].items():
        path = directory / relative
        if not path.is_file() or protocol.file_record(path) != record:
            raise ValueError(f"changed v12 robust source file: {path}")
    return payload


def _load_final_agent(seed: int):
    run_dir = protocol.source_run_dir(seed)
    config = final_task_sweep.load_config(run_dir)
    env = make_env(config, seed_offset=0)
    tasks = env.sample_tasks(len(protocol.MODES))
    agent = make_algo("sac", env.obs_dim, env.act_dim, config)
    if hasattr(agent, "set_task_metadata"):
        agent.set_task_metadata(tasks)
    replay = ReplayBuffer(
        env.obs_dim,
        env.act_dim,
        capacity=1,
        belief_dim=getattr(agent, "belief_dim", 0),
    )
    with tempfile.TemporaryDirectory() as temporary:
        logger = Logger(temporary)
        next_iteration, total_steps = load_checkpoint(
            str(run_dir / "checkpoints"),
            agent,
            replay,
            logger,
            "sac",
            load_replay_buffer=False,
        )
    return config, env, agent, next_iteration, total_steps


def publish_bundle(seed: int) -> dict[str, Any]:
    seed = protocol.require_training_seed(seed)
    destination = protocol.source_bundle(seed)
    if protocol.source_manifest(seed).is_file():
        return validate_bundle(seed)
    run_dir = protocol.source_run_dir(seed)
    config, env, agent, next_iteration, total_steps = _load_final_agent(seed)
    try:
        checkpoint = protocol.checkpoint_record(run_dir)
        if (
            checkpoint != protocol.expected_source_checkpoint()
            or next_iteration != protocol.SOURCE_NEXT_ITERATION
            or total_steps != protocol.SOURCE_TOTAL_STEPS
            or int(agent.update_count) != protocol.SOURCE_UPDATE_COUNT
            or int(config.stochastic_mode_fixed_id) != -1
        ):
            raise ValueError(f"incomplete v12 robust source: {run_dir}")
        signature = run_dir / "logs/protocol_signature.json"
        validate_signature(seed)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp.", dir=destination.parent))
        try:
            policy_path = temporary / "policy" / protocol.POLICY_NAME
            policy_path.parent.mkdir(parents=True, exist_ok=True)
            with policy_path.open("wb") as handle:
                pickle.dump(
                    _to_numpy_tree(nnx.state(agent.policy, nnx.Param)),
                    handle,
                )
            signature_target = temporary / "logs" / signature.name
            signature_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(signature, signature_target)
            files = {
                path.relative_to(temporary).as_posix(): protocol.file_record(path)
                for path in (policy_path, signature_target)
            }
            protocol.write_json_atomic(
                temporary / "bundle_manifest.json",
                {
                    "schema": protocol.SOURCE_BUNDLE_SCHEMA,
                    "status": "complete",
                    "identity": protocol.source_identity(seed),
                    "checkpoint": checkpoint,
                    "files": files,
                    "contents": "policy parameters only; no critic, optimizer, or replay",
                },
            )
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    finally:
        if hasattr(env, "close"):
            env.close()
    return validate_bundle(seed)


def run(seed: int) -> None:
    seed = protocol.require_training_seed(seed)
    protocol.validate_registration()
    if protocol.source_manifest(seed).is_file():
        validate_bundle(seed)
        print(f"V12 ROBUST SOURCE ALREADY COMPLETE: seed={seed}", flush=True)
        return
    command = training_command(seed)
    print("V12 ROBUST SOURCE TRAIN:", " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(protocol.ROOT)
    subprocess.run(command, cwd=protocol.ROOT, env=environment, check=True)
    payload = publish_bundle(seed)
    replay = protocol.source_run_dir(seed) / "checkpoints/replay_buffer.npz"
    if replay.is_file():
        replay.unlink()
    print(
        "V12 ROBUST SOURCE COMPLETE: "
        + json.dumps(payload["identity"], sort_keys=True),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for checkpoint-safe execution")
    run(args.seed)


if __name__ == "__main__":
    main()
