"""Train one V33 from-scratch SAC recipe and publish policy parameters."""
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
    regime_polarity_canonical_training_ablation_v33 as protocol,
)
from jax_experiments.common.checkpoint import _to_numpy_tree, load_checkpoint
from jax_experiments.common.logging import Logger
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.train import make_algo, make_env


def training_command(recipe: str, seed: int) -> list[str]:
    recipe = protocol.require_direct_recipe(recipe)
    seed = protocol.require_training_seed(seed)
    run_dir = protocol.direct_run_dir(recipe, seed)
    fixed_mode, canonical_reference = protocol.direct_flags(recipe)
    command = [
        sys.executable, "-u", "-m", "jax_experiments.train",
        "--algo", "sac",
        "--env", protocol.ENV,
        "--seed", str(seed),
        "--max_iters", str(protocol.FINAL_NEXT_ITERATION),
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
        "--initial_random_steps", "0",
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
    if fixed_mode >= 0:
        command += ["--stochastic_mode_fixed_id", str(fixed_mode)]
    if canonical_reference >= 0:
        command += [
            "--stochastic_mode_canonical_reference_id",
            str(canonical_reference),
        ]
    return command


def expected_config(recipe: str, seed: int) -> dict[str, Any]:
    fixed_mode, canonical_reference = protocol.direct_flags(recipe)
    return {
        "algo": "sac",
        "env_name": protocol.ENV,
        "seed": protocol.require_training_seed(seed),
        "max_iters": protocol.FINAL_NEXT_ITERATION,
        "env_type": "stochastic_mode",
        "stochastic_mode_family": protocol.FAMILY,
        "stochastic_mode_dwell_steps": protocol.DWELL_STEPS,
        "stochastic_mode_dwell_distribution": "fixed",
        "stochastic_mode_fixed_id": fixed_mode,
        "stochastic_mode_canonical_reference_id": canonical_reference,
        "task_num": 4,
        "test_task_num": 4,
        "samples_per_iter": protocol.SAMPLES_PER_ITER,
        "updates_per_iter": protocol.UPDATES_PER_ITER,
        "start_train_steps": protocol.START_TRAIN_STEPS,
        "initial_random_steps": 0,
        "context_warmup_iters": 50,
        "ensemble_size": 2,
        "hidden_dim": 256,
        "lr": 3e-4,
        "max_episode_steps": protocol.MAX_EPISODE_STEPS,
        "brax_backend": "spring",
    }


def validate_signature(recipe: str, seed: int) -> dict[str, Any]:
    path = protocol.direct_run_dir(recipe, seed) / "logs/protocol_signature.json"
    if not path.is_file():
        path = protocol.direct_bundle_dir(recipe, seed) / "logs/protocol_signature.json"
    signature = protocol.read_json(path)
    config = signature.get("config") or {}
    mismatches = {
        key: {"actual": config.get(key), "expected": expected}
        for key, expected in expected_config(recipe, seed).items()
        if config.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"V33 direct recipe config mismatch: {mismatches}")
    return signature


def _load_final_agent(recipe: str, seed: int):
    run_dir = protocol.direct_run_dir(recipe, seed)
    config = final_task_sweep.load_config(run_dir)
    env = make_env(config, seed_offset=0)
    agent = make_algo("sac", env.obs_dim, env.act_dim, config)
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


def validate_bundle(recipe: str, seed: int) -> dict[str, Any]:
    recipe = protocol.require_direct_recipe(recipe)
    seed = protocol.require_training_seed(seed)
    directory = protocol.direct_bundle_dir(recipe, seed)
    payload = protocol.read_json(directory / "bundle_manifest.json")
    expected_files = {
        "policy/" + protocol.POLICY_NAME,
        "logs/protocol_signature.json",
    }
    if (
        payload.get("schema") != protocol.DIRECT_BUNDLE_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("identity") != protocol.direct_identity(recipe, seed)
        or payload.get("checkpoint") != protocol.expected_checkpoint()
        or set(payload.get("files") or {}) != expected_files
    ):
        raise ValueError(f"invalid V33 direct policy bundle: {directory}")
    for relative, record in payload["files"].items():
        path = directory / relative
        if not path.is_file() or protocol.file_record(path) != record:
            raise ValueError(f"changed V33 direct bundle file: {path}")
    validate_signature(recipe, seed)
    return payload


def publish_bundle(recipe: str, seed: int) -> dict[str, Any]:
    recipe = protocol.require_direct_recipe(recipe)
    seed = protocol.require_training_seed(seed)
    destination = protocol.direct_bundle_dir(recipe, seed)
    if protocol.direct_manifest(recipe, seed).is_file():
        return validate_bundle(recipe, seed)
    run_dir = protocol.direct_run_dir(recipe, seed)
    config, env, agent, next_iteration, total_steps = _load_final_agent(recipe, seed)
    try:
        fixed_mode, canonical_reference = protocol.direct_flags(recipe)
        checkpoint = protocol.checkpoint_record(run_dir)
        if (
            checkpoint != protocol.expected_checkpoint()
            or next_iteration != protocol.FINAL_NEXT_ITERATION
            or total_steps != protocol.FINAL_TOTAL_STEPS
            or int(agent.update_count) != protocol.FINAL_UPDATE_COUNT
            or int(config.stochastic_mode_fixed_id) != fixed_mode
            or int(config.stochastic_mode_canonical_reference_id)
            != canonical_reference
        ):
            raise ValueError(f"incomplete V33 direct training output: {run_dir}")
        signature = run_dir / "logs/protocol_signature.json"
        validate_signature(recipe, seed)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp.", dir=destination.parent))
        try:
            policy_path = temporary / "policy" / protocol.POLICY_NAME
            policy_path.parent.mkdir(parents=True, exist_ok=True)
            with policy_path.open("wb") as handle:
                pickle.dump(
                    _to_numpy_tree(nnx.state(agent.policy, nnx.Param)), handle)
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
                    "schema": protocol.DIRECT_BUNDLE_SCHEMA,
                    "status": "complete",
                    "identity": protocol.direct_identity(recipe, seed),
                    "checkpoint": checkpoint,
                    "files": files,
                    "contents": "policy parameters and protocol signature only",
                },
            )
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    finally:
        if hasattr(env, "close"):
            env.close()
    return validate_bundle(recipe, seed)


def run(recipe: str, seed: int) -> None:
    recipe = protocol.require_direct_recipe(recipe)
    seed = protocol.require_training_seed(seed)
    protocol.validate_registration()
    if protocol.direct_manifest(recipe, seed).is_file():
        validate_bundle(recipe, seed)
        print(f"V33 DIRECT RECIPE ALREADY COMPLETE: {recipe} seed={seed}", flush=True)
        return
    command = training_command(recipe, seed)
    print("V33 DIRECT RECIPE TRAIN:", " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(protocol.ROOT)
    subprocess.run(command, cwd=protocol.ROOT, env=environment, check=True)
    payload = publish_bundle(recipe, seed)
    replay = protocol.direct_run_dir(recipe, seed) / "checkpoints/replay_buffer.npz"
    if replay.is_file():
        replay.unlink()
    print(
        "V33 DIRECT RECIPE COMPLETE: "
        + json.dumps(payload["identity"], sort_keys=True),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=protocol.DIRECT_RECIPES, required=True)
    parser.add_argument("--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for checkpoint-safe execution")
    run(args.recipe, args.seed)


if __name__ == "__main__":
    main()
