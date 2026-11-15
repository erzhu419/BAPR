"""Train one V24 Ant specialist on post-switch recovery states."""
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
    regime_polarity_ant_switch_recovery_v24 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_ant_full_state_specialist_v22 as source,
)
from jax_experiments.analysis import (
    run_regime_polarity_robust_warmstart_specialist_v11 as historical,
)
from jax_experiments.algos.sac_switch_recovery import SACSwitchRecovery
from jax_experiments.common.checkpoint import (
    _patch_flax_variablestate_unpickle,
    _to_numpy_tree,
    load_checkpoint,
    save_checkpoint,
)
from jax_experiments.common.logging import Logger
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.train import make_env


def _load_source(seed: int):
    _patch_flax_variablestate_unpickle()
    return source._load_source(seed)


def _target_config(source_config, run_dir: Path, mode: int, variant: str):
    config = deepcopy(source_config)
    config.algo = "sac"
    config.save_root = str(run_dir.parent)
    config.run_name = run_dir.name
    config.max_iters = protocol.FINAL_NEXT_ITERATION
    config.samples_per_iter = protocol.PHYSICAL_SAMPLES_PER_ITER
    config.updates_per_iter = protocol.UPDATES_PER_ITER
    config.start_train_steps = 0
    config.stochastic_mode_fixed_id = protocol.require_mode(mode)
    config.log_interval = 50
    config.eval_episodes = 3
    config.eval_protocol = "stationary"
    config.save_interval = 50
    config.resume = True
    config.min_resume_iteration = protocol.SOURCE_NEXT_ITERATION
    config.switch_recovery_target_mode = protocol.require_mode(mode)
    config.switch_recovery_segment_steps = protocol.SWITCH_SEGMENT_STEPS
    config.switch_recovery_termination_penalty = (
        protocol.termination_penalty(variant))
    return config


def _make_agent(config, obs_dim: int, act_dim: int) -> SACSwitchRecovery:
    return SACSwitchRecovery(obs_dim, act_dim, config, seed=config.seed)


def _controller_equivalence(source_agent, target_agent) -> dict[str, Any]:
    observations = jax.random.normal(
        jax.random.PRNGKey(324_101), (64, source_agent.obs_dim))
    actions = jax.random.uniform(
        jax.random.PRNGKey(324_102),
        (64, source_agent.act_dim),
        minval=-1.0,
        maxval=1.0,
    )
    errors = {
        "actor_max_abs_error": float(jnp.max(jnp.abs(
            source_agent.policy.deterministic(observations)
            - target_agent.policy.deterministic(observations)))),
        "fallback_actor_max_abs_error": float(jnp.max(jnp.abs(
            source_agent.policy.deterministic(observations)
            - target_agent.fallback_policy.deterministic(observations)))),
        "critic_max_abs_error": float(jnp.max(jnp.abs(
            source_agent.critic(observations, actions)
            - target_agent.critic(observations, actions)))),
        "target_critic_max_abs_error": float(jnp.max(jnp.abs(
            source_agent.target_critic(observations, actions)
            - target_agent.target_critic(observations, actions)))),
        "log_alpha_abs_error": float(jnp.abs(
            source_agent.log_alpha - target_agent.log_alpha)),
    }
    return {
        "pass": bool(max(errors.values()) <= 1e-7),
        **errors,
        "atol": 1e-7,
    }


def _fallback_equivalence(source_agent, target_agent) -> dict[str, Any]:
    observations = jax.random.normal(
        jax.random.PRNGKey(324_103), (128, source_agent.obs_dim))
    error = float(jnp.max(jnp.abs(
        source_agent.policy.deterministic(observations)
        - target_agent.fallback_policy.deterministic(observations))))
    return {
        "pass": bool(error <= 1e-7),
        "max_abs_action_error": error,
        "observations": int(observations.shape[0]),
        "atol": 1e-7,
    }


def _runtime_payload(variant: str, seed: int, mode: int) -> dict[str, Any]:
    return {
        "schema": "bapr.ant-switch-recovery-runtime.v24",
        "identity": protocol.identity(variant, seed, mode),
        "physical_samples_per_iter": protocol.PHYSICAL_SAMPLES_PER_ITER,
        "candidate_samples_per_iter": protocol.CANDIDATE_SAMPLES_PER_ITER,
        "robust_prefix_samples_per_iter": (
            protocol.PHYSICAL_SAMPLES_PER_ITER
            - protocol.CANDIDATE_SAMPLES_PER_ITER),
        "segment_steps": protocol.SWITCH_SEGMENT_STEPS,
        "termination_penalty": protocol.termination_penalty(variant),
        "predecessor_policy": "frozen_matched_robust_actor",
        "specialist_replay": "post_switch_target_mode_only",
    }


def _bootstrap(
    variant: str,
    seed: int,
    mode: int,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    variant = protocol.require_variant(variant)
    seed = protocol.require_training_seed(seed)
    mode = protocol.require_mode(mode)
    run_dir = run_dir or protocol.run_dir(variant, seed, mode)
    manifest_path = run_dir / "checkpoints" / protocol.BOOTSTRAP_NAME
    runtime_path = run_dir / "checkpoints" / protocol.RUNTIME_NAME
    identity = protocol.identity(variant, seed, mode)
    if manifest_path.is_file() and runtime_path.is_file():
        manifest = protocol.read_json(manifest_path)
        runtime = protocol.read_json(runtime_path)
        if (
            manifest.get("schema") != protocol.BOOTSTRAP_SCHEMA
            or manifest.get("identity") != identity
            or manifest.get("controller_equivalence", {}).get("pass")
            is not True
            or runtime != _runtime_payload(variant, seed, mode)
        ):
            raise ValueError(f"invalid existing V24 bootstrap: {run_dir}")
        return manifest
    if run_dir.exists():
        raise RuntimeError(f"partial V24 branch exists without bootstrap: {run_dir}")

    source_config, source_agent = _load_source(seed)
    config = _target_config(source_config, run_dir, mode, variant)
    env = make_env(config, seed_offset=0)
    try:
        agent = _make_agent(config, env.obs_dim, env.act_dim)
        nnx.update(agent.policy, nnx.state(source_agent.policy, nnx.Param))
        nnx.update(agent.critic, nnx.state(source_agent.critic, nnx.Param))
        nnx.update(
            agent.target_critic,
            nnx.state(source_agent.target_critic, nnx.Param),
        )
        agent.set_fallback_policy_state(
            nnx.state(source_agent.policy, nnx.Param))
        agent.log_alpha = jnp.asarray(source_agent.log_alpha)
        historical._reset_optimizers(agent)
        agent.update_count = protocol.SOURCE_UPDATE_COUNT
        equivalence = _controller_equivalence(source_agent, agent)
        if not equivalence["pass"]:
            raise RuntimeError(
                f"V24 full-controller warm-start mismatch: {equivalence}")

        run_dir.mkdir(parents=True, exist_ok=False)
        replay = ReplayBuffer(
            env.obs_dim,
            env.act_dim,
            capacity=config.replay_size,
            belief_dim=0,
        )
        logger = Logger(str(run_dir / "logs"))
        save_checkpoint(
            str(run_dir / "checkpoints"),
            agent,
            replay,
            logger,
            iteration=protocol.SOURCE_ITERATION,
            total_steps=protocol.SOURCE_TOTAL_STEPS,
            algo="sac",
        )
        manifest = {
            "schema": protocol.BOOTSTRAP_SCHEMA,
            "status": "complete",
            "identity": identity,
            "registration": protocol.file_record(protocol.REGISTRATION_PATH),
            "source_bundle_manifest": protocol.file_record(
                protocol.source_manifest(seed)),
            "source_checkpoint": protocol.expected_source_checkpoint(),
            "fork_checkpoint": protocol.expected_source_checkpoint(),
            "controller_equivalence": equivalence,
            "controller_initialization": [
                "actor", "critic", "target_critic", "alpha"
            ],
            "frozen_fallback_initialization": "source_actor",
            "replay_reset": True,
            "optimizer_reset": True,
        }
        protocol.write_json_atomic(manifest_path, manifest)
        protocol.write_json_atomic(
            runtime_path, _runtime_payload(variant, seed, mode))
        return manifest
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
) -> list[str]:
    variant = protocol.require_variant(variant)
    seed = protocol.require_training_seed(seed)
    mode = protocol.require_mode(mode)
    run_dir = run_dir or protocol.run_dir(variant, seed, mode)
    return [
        sys.executable,
        "-u",
        "-m",
        "jax_experiments.analysis."
        "train_regime_polarity_ant_switch_recovery_v24",
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
        "--stochastic_mode_fixed_id", str(mode),
        "--task_num", "4",
        "--test_task_num", "4",
        "--samples_per_iter", str(protocol.PHYSICAL_SAMPLES_PER_ITER),
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


def _runtime_environment(variant: str, mode: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(protocol.ROOT)
    environment["BAPR_SWITCH_RECOVERY_TARGET_MODE"] = str(
        protocol.require_mode(mode))
    environment["BAPR_SWITCH_RECOVERY_SEGMENT_STEPS"] = str(
        protocol.SWITCH_SEGMENT_STEPS)
    environment["BAPR_SWITCH_RECOVERY_TERMINATION_PENALTY"] = str(
        protocol.termination_penalty(variant))
    return environment


def _load_final_agent(variant: str, seed: int, mode: int):
    run_dir = protocol.run_dir(variant, seed, mode)
    config = final_task_sweep.load_config(run_dir)
    config.switch_recovery_target_mode = protocol.require_mode(mode)
    config.switch_recovery_segment_steps = protocol.SWITCH_SEGMENT_STEPS
    config.switch_recovery_termination_penalty = (
        protocol.termination_penalty(variant))
    env = make_env(config, seed_offset=0)
    agent = _make_agent(config, env.obs_dim, env.act_dim)
    replay = ReplayBuffer(env.obs_dim, env.act_dim, capacity=1)
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
        "provenance/" + protocol.RUNTIME_NAME,
    }
    if (
        payload.get("schema") != protocol.BUNDLE_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("identity") != protocol.identity(variant, seed, mode)
        or payload.get("checkpoint") != protocol.expected_checkpoint()
        or payload.get("source_bundle_manifest")
        != protocol.file_record(protocol.source_manifest(seed))
        or payload.get("frozen_fallback_equivalence", {}).get("pass")
        is not True
        or set(payload.get("files") or {}) != expected_files
    ):
        raise ValueError(f"invalid V24 policy bundle: {directory}")
    for relative, record in payload["files"].items():
        path = directory / relative
        if not path.is_file() or protocol.file_record(path) != record:
            raise ValueError(f"changed V24 policy bundle file: {path}")
    return payload


def publish_bundle(variant: str, seed: int, mode: int) -> dict[str, Any]:
    variant = protocol.require_variant(variant)
    seed = protocol.require_training_seed(seed)
    mode = protocol.require_mode(mode)
    destination = protocol.bundle_dir(variant, seed, mode)
    if protocol.bundle_manifest(variant, seed, mode).is_file():
        return validate_bundle(variant, seed, mode)
    run_dir = protocol.run_dir(variant, seed, mode)
    config, env, agent, next_iteration, total_steps = _load_final_agent(
        variant, seed, mode)
    try:
        _, source_agent = _load_source(seed)
        fallback_equivalence = _fallback_equivalence(source_agent, agent)
        checkpoint = protocol.checkpoint_record(run_dir)
        runtime = protocol.read_json(
            run_dir / "checkpoints" / protocol.RUNTIME_NAME)
        if (
            checkpoint != protocol.expected_checkpoint()
            or next_iteration != protocol.FINAL_NEXT_ITERATION
            or total_steps != protocol.FINAL_TOTAL_STEPS
            or int(agent.update_count) != protocol.FINAL_UPDATE_COUNT
            or int(config.samples_per_iter)
            != protocol.PHYSICAL_SAMPLES_PER_ITER
            or runtime != _runtime_payload(variant, seed, mode)
            or fallback_equivalence["pass"] is not True
        ):
            raise ValueError(f"incomplete V24 training output: {run_dir}")
        signature = run_dir / "logs" / "protocol_signature.json"
        bootstrap = run_dir / "checkpoints" / protocol.BOOTSTRAP_NAME
        runtime_path = run_dir / "checkpoints" / protocol.RUNTIME_NAME
        if not signature.is_file() or not bootstrap.is_file():
            raise FileNotFoundError("V24 training provenance is incomplete")

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
            provenance = temporary / "provenance"
            provenance.mkdir(parents=True, exist_ok=True)
            shutil.copy2(bootstrap, provenance / bootstrap.name)
            shutil.copy2(runtime_path, provenance / runtime_path.name)
            files = {
                path.relative_to(temporary).as_posix(): protocol.file_record(path)
                for path in (
                    policy_path,
                    signature_target,
                    provenance / bootstrap.name,
                    provenance / runtime_path.name,
                )
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
                    "frozen_fallback_equivalence": fallback_equivalence,
                    "files": files,
                    "contents": "final policy and compact provenance only",
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
            f"V24 SWITCH RECOVERY ALREADY COMPLETE: {variant} "
            f"seed={seed} mode={mode}",
            flush=True,
        )
        return
    run_dir = protocol.run_dir(variant, seed, mode)
    _bootstrap(variant, seed, mode, run_dir)
    command = training_command(variant, seed, mode, run_dir)
    print("V24 SWITCH RECOVERY TRAIN:", " ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=protocol.ROOT,
        env=_runtime_environment(variant, mode),
        check=True,
    )
    payload = publish_bundle(variant, seed, mode)
    print(
        "V24 SWITCH RECOVERY COMPLETE: "
        + json.dumps(payload["identity"], sort_keys=True),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=protocol.VARIANTS, required=True)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--mode", choices=protocol.MODES, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for checkpoint-safe execution")
    run(args.variant, args.seed, args.mode)


if __name__ == "__main__":
    main()
