"""Train one robust-warm-started fixed-mode SAC development controller."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from jax_experiments.analysis import final_task_sweep
from jax_experiments.analysis import (
    regime_polarity_robust_warmstart_specialist_v11 as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_safe_utility_confirmation_v9 as parent,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_expected_action_confirmation_audit_v6 as source_audit,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_safe_utility_confirmation_audit_v9 as parent_audit,
)
from jax_experiments.common.checkpoint import (
    _to_numpy_tree,
    load_checkpoint,
    save_checkpoint,
)
from jax_experiments.common.logging import Logger
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.train import make_algo, make_env


def _bind_parent() -> None:
    parent_audit._bind()
    parent.validate_registration()


def _reset_optimizers(agent) -> None:
    agent.policy_opt_state = agent.policy_opt.init(
        nnx.state(agent.policy, nnx.Param))
    agent.critic_opt_state = agent.critic_opt.init(
        nnx.state(agent.critic, nnx.Param))
    agent.alpha_opt_state = agent.alpha_opt.init(agent.log_alpha)


def _target_config(source_config, run_dir: Path, mode: int):
    config = deepcopy(source_config)
    config.algo = "sac"
    config.save_root = str(run_dir.parent)
    config.run_name = run_dir.name
    config.max_iters = protocol.FINAL_NEXT_ITERATION
    config.start_train_steps = 0
    config.stochastic_mode_fixed_id = protocol.require_mode(mode)
    config.log_interval = 50
    config.eval_episodes = 3
    config.eval_protocol = "stationary"
    config.save_interval = 50
    config.resume = True
    config.min_resume_iteration = protocol.SOURCE_NEXT_ITERATION
    return config


def _actor_equivalence(source_agent, target_agent) -> dict[str, Any]:
    observations = jax.random.normal(
        jax.random.PRNGKey(311_081), (64, source_agent.obs_dim))
    source_actions = source_agent.policy.deterministic(observations)
    target_actions = target_agent.policy.deterministic(observations)
    error = float(jnp.max(jnp.abs(source_actions - target_actions)))
    return {
        "pass": bool(error <= 1e-7),
        "max_abs_action_error": error,
        "observations": int(observations.shape[0]),
        "atol": 1e-7,
    }


def _bootstrap(
    variant: str,
    seed: int,
    mode: int,
    run_dir: Path | None = None,
    require_registration: bool = True,
) -> dict[str, Any]:
    variant = protocol.require_variant(variant)
    seed = protocol.require_training_seed(seed)
    mode = protocol.require_mode(mode)
    run_dir = run_dir or protocol.run_dir(variant, seed, mode)
    manifest_path = run_dir / "checkpoints" / protocol.BOOTSTRAP_NAME
    expected_identity = protocol.identity(variant, seed, mode)
    if manifest_path.is_file():
        payload = protocol.read_json(manifest_path)
        if (
            payload.get("schema")
            != "bapr.robust-warmstart-specialist-bootstrap.v11"
            or payload.get("identity") != expected_identity
            or payload.get("actor_equivalence", {}).get("pass") is not True
        ):
            raise ValueError(f"invalid existing v11 bootstrap: {manifest_path}")
        return payload
    if run_dir.exists():
        raise RuntimeError(
            f"partial v11 branch exists without bootstrap: {run_dir}")

    _bind_parent()
    source = source_audit._load_controller("robust_sac", seed)
    source_agent = source["agent"]
    source_config = source["config"]
    config = _target_config(source_config, run_dir, mode)
    env = make_env(config, seed_offset=0)
    try:
        if variant == "full_state":
            agent = source_agent
            agent.config = config
            _reset_optimizers(agent)
        else:
            agent = make_algo("sac", env.obs_dim, env.act_dim, config)
            nnx.update(
                agent.policy,
                nnx.state(source_agent.policy, nnx.Param),
            )
            agent.update_count = int(source_agent.update_count)
        equivalence = _actor_equivalence(source_agent, agent)
        if not equivalence["pass"]:
            raise RuntimeError(
                f"robust actor warm-start mismatch: {equivalence}")

        run_dir.mkdir(parents=True, exist_ok=False)
        replay = ReplayBuffer(
            env.obs_dim,
            env.act_dim,
            capacity=config.replay_size,
            belief_dim=getattr(agent, "belief_dim", 0),
        )
        logger = Logger(str(run_dir / "logs"))
        save_checkpoint(
            str(run_dir / "checkpoints"),
            agent,
            replay,
            logger,
            iteration=protocol.SOURCE_NEXT_ITERATION - 1,
            total_steps=protocol.SOURCE_TOTAL_STEPS,
            algo="sac",
        )
        registration = (
            protocol.file_record(protocol.REGISTRATION_PATH)
            if require_registration else {"smoke_only": True}
        )
        payload = {
            "schema": "bapr.robust-warmstart-specialist-bootstrap.v11",
            "status": "complete",
            "identity": expected_identity,
            "registration": registration,
            "source_bundle_manifest": protocol.file_record(
                protocol.source_manifest(seed)),
            "source_checkpoint": parent.expected_checkpoint("robust_sac"),
            "fork_checkpoint": {
                "iteration": protocol.SOURCE_NEXT_ITERATION - 1,
                "next_iteration": protocol.SOURCE_NEXT_ITERATION,
                "total_steps": protocol.SOURCE_TOTAL_STEPS,
                "update_count": protocol.SOURCE_UPDATE_COUNT,
                "algo": "sac",
            },
            "actor_equivalence": equivalence,
            "replay_reset": True,
            "optimizer_reset": True,
            "controller_initialization": (
                ["actor", "critic", "target_critic", "alpha"]
                if variant == "full_state" else ["actor"]
            ),
        }
        protocol.write_json_atomic(manifest_path, payload)
        return payload
    except Exception:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    finally:
        if hasattr(env, "close"):
            env.close()


def training_command(
    variant: str,
    seed: int,
    mode: int,
    run_dir: Path | None = None,
    final_next_iteration: int | None = None,
) -> list[str]:
    variant = protocol.require_variant(variant)
    seed = protocol.require_training_seed(seed)
    mode = protocol.require_mode(mode)
    run_dir = run_dir or protocol.run_dir(variant, seed, mode)
    final_next_iteration = int(
        final_next_iteration or protocol.FINAL_NEXT_ITERATION)
    if not (
        protocol.SOURCE_NEXT_ITERATION < final_next_iteration
        <= protocol.FINAL_NEXT_ITERATION
    ):
        raise ValueError("invalid v11 training endpoint")
    return [
        sys.executable, "-u", "-m", "jax_experiments.train",
        "--algo", "sac",
        "--env", protocol.ENV,
        "--seed", str(seed),
        "--max_iters", str(final_next_iteration),
        "--save_root", str(run_dir.parent),
        "--run_name", run_dir.name,
        "--env_type", "stochastic_mode",
        "--stochastic_mode_family", protocol.FAMILY,
        "--stochastic_mode_dwell_steps", str(protocol.DWELL_STEPS),
        "--stochastic_mode_dwell_distribution", "fixed",
        "--stochastic_mode_fixed_id", str(mode),
        "--task_num", "4",
        "--test_task_num", "4",
        "--samples_per_iter", str(protocol.SAMPLES_PER_ITER),
        "--updates_per_iter", str(protocol.UPDATES_PER_ITER),
        "--start_train_steps", "0",
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
        "--min_resume_iteration", str(protocol.SOURCE_NEXT_ITERATION),
    ]


def _load_final_agent(run_dir: Path):
    config = final_task_sweep.load_config(run_dir)
    env = make_env(config, seed_offset=0)
    tasks = env.sample_tasks(len(protocol.MODES))
    agent = make_algo(config.algo, env.obs_dim, env.act_dim, config)
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
            config.algo,
            load_replay_buffer=False,
        )
    return config, env, agent, next_iteration, total_steps


def validate_bundle(variant: str, seed: int, mode: int) -> dict[str, Any]:
    variant = protocol.require_variant(variant)
    seed = protocol.require_training_seed(seed)
    mode = protocol.require_mode(mode)
    directory = protocol.bundle_dir(variant, seed, mode)
    payload = protocol.read_json(directory / "bundle_manifest.json")
    expected_files = {
        "policy/" + protocol.POLICY_NAME,
        "logs/protocol_signature.json",
        "provenance/" + protocol.BOOTSTRAP_NAME,
    }
    if (
        payload.get("schema") != protocol.BUNDLE_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("identity") != protocol.identity(variant, seed, mode)
        or payload.get("checkpoint") != protocol.expected_checkpoint()
        or payload.get("source_bundle_manifest")
        != protocol.file_record(protocol.source_manifest(seed))
        or set(payload.get("files") or {}) != expected_files
    ):
        raise ValueError(f"invalid v11 policy bundle: {directory}")
    for relative, record in payload["files"].items():
        path = directory / relative
        if not path.is_file() or protocol.file_record(path) != record:
            raise ValueError(f"changed v11 policy bundle file: {path}")
    return payload


def publish_bundle(variant: str, seed: int, mode: int) -> dict[str, Any]:
    variant = protocol.require_variant(variant)
    seed = protocol.require_training_seed(seed)
    mode = protocol.require_mode(mode)
    destination = protocol.bundle_dir(variant, seed, mode)
    if (destination / "bundle_manifest.json").is_file():
        return validate_bundle(variant, seed, mode)
    run_dir = protocol.run_dir(variant, seed, mode)
    config, env, agent, next_iteration, total_steps = _load_final_agent(run_dir)
    try:
        checkpoint = protocol.checkpoint_record(run_dir)
        if (
            checkpoint != protocol.expected_checkpoint()
            or next_iteration != protocol.FINAL_NEXT_ITERATION
            or total_steps != protocol.FINAL_TOTAL_STEPS
            or int(config.stochastic_mode_fixed_id) != mode
            or int(config.start_train_steps) != 0
        ):
            raise ValueError(f"incomplete v11 training output: {run_dir}")
        signature = run_dir / "logs" / "protocol_signature.json"
        bootstrap = run_dir / "checkpoints" / protocol.BOOTSTRAP_NAME
        if not signature.is_file() or not bootstrap.is_file():
            raise FileNotFoundError("v11 training provenance is incomplete")

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
            bootstrap_target = temporary / "provenance" / bootstrap.name
            bootstrap_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(bootstrap, bootstrap_target)
            files = {
                path.relative_to(temporary).as_posix(): protocol.file_record(path)
                for path in (policy_path, signature_target, bootstrap_target)
            }
            protocol.write_json_atomic(
                temporary / "bundle_manifest.json",
                {
                    "schema": protocol.BUNDLE_SCHEMA,
                    "status": "complete",
                    "identity": protocol.identity(variant, seed, mode),
                    "checkpoint": checkpoint,
                    "source_bundle_manifest": protocol.file_record(
                        protocol.source_manifest(seed)),
                    "files": files,
                    "contents": "policy parameters only; no critic or optimizer",
                },
            )
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    finally:
        if hasattr(env, "close"):
            env.close()
    return validate_bundle(variant, seed, mode)


def run(variant: str, seed: int, mode: int) -> None:
    variant = protocol.require_variant(variant)
    seed = protocol.require_training_seed(seed)
    mode = protocol.require_mode(mode)
    protocol.validate_registration()
    if protocol.bundle_manifest(variant, seed, mode).is_file():
        validate_bundle(variant, seed, mode)
        print(
            "V11 WARMSTART CONTROLLER ALREADY COMPLETE: "
            f"variant={variant} seed={seed} mode={mode}",
            flush=True,
        )
        return
    run_dir = protocol.run_dir(variant, seed, mode)
    _bootstrap(variant, seed, mode, run_dir)
    command = training_command(variant, seed, mode, run_dir)
    print("V11 WARMSTART TRAIN:", " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(protocol.ROOT)
    subprocess.run(
        command,
        cwd=protocol.ROOT,
        env=environment,
        check=True,
    )
    payload = publish_bundle(variant, seed, mode)
    print(
        "V11 WARMSTART CONTROLLER COMPLETE: "
        + json.dumps(payload["identity"], sort_keys=True),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=protocol.VARIANTS, required=True)
    parser.add_argument("--seed", choices=protocol.TRAINING_SEEDS,
                        type=int, required=True)
    parser.add_argument("--mode", choices=protocol.MODES,
                        type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for checkpoint-safe execution")
    run(args.variant, args.seed, args.mode)


if __name__ == "__main__":
    main()
