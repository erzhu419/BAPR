"""Train one V26 shared Ant mode-conditioned risk controller."""
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
    regime_polarity_ant_joint_mode_risk_v26 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_ant_full_state_specialist_v22 as source,
)
from jax_experiments.algos.joint_mode_risk_sac import JointModeRiskSAC
from jax_experiments.common.checkpoint import (
    _patch_flax_variablestate_unpickle,
    _restore_tree_like,
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


def _target_config(source_config, run_dir: Path, variant: str):
    variant = protocol.require_variant(variant)
    config = deepcopy(source_config)
    config.algo = "sac"
    config.save_root = str(run_dir.parent)
    config.run_name = run_dir.name
    config.max_iters = protocol.FINAL_NEXT_ITERATION[variant]
    config.samples_per_iter = protocol.PHYSICAL_SAMPLES_PER_ITER
    config.updates_per_iter = protocol.UPDATES_PER_ITER
    config.start_train_steps = 0
    config.context_warmup_iters = 0
    config.stochastic_mode_fixed_id = -1
    config.stochastic_mode_dwell_steps = protocol.DWELL_STEPS
    config.stochastic_mode_dwell_distribution = "fixed"
    config.regime_context_source = "oracle"
    config.task_num = len(protocol.MODES)
    config.test_task_num = len(protocol.MODES)
    config.log_interval = 50
    config.eval_episodes = 3
    config.eval_protocol = "stationary"
    config.save_interval = 50
    config.resume = True
    config.min_resume_iteration = protocol.SOURCE_NEXT_ITERATION
    config.joint_mode_risk_lambda = protocol.RISK_LAMBDA
    config.joint_mode_risk_actor_start_update = (
        protocol.RISK_ACTOR_START_UPDATE)
    return config


def _make_agent(config, obs_dim: int, act_dim: int) -> JointModeRiskSAC:
    return JointModeRiskSAC(obs_dim, act_dim, config, seed=config.seed)


def _copy_policy_from_source(target, source_policy) -> None:
    if len(target.layers) != len(source_policy.layers):
        raise ValueError("policy depth changed")
    for index, (target_layer, source_layer) in enumerate(
        zip(target.layers, source_policy.layers)
    ):
        source_kernel = jnp.asarray(source_layer.kernel.value)
        if index == 0:
            kernel = jnp.zeros_like(target_layer.kernel.value)
            if kernel.shape[0] <= source_kernel.shape[0]:
                raise ValueError("conditioned policy has no context rows")
            kernel = kernel.at[:source_kernel.shape[0], :].set(source_kernel)
            target_layer.kernel.value = kernel
        else:
            target_layer.kernel.value = source_kernel
        target_layer.bias.value = jnp.asarray(source_layer.bias.value)
    for target_head, source_head in (
        (target.mean_head, source_policy.mean_head),
        (target.log_std_head, source_policy.log_std_head),
    ):
        target_head.kernel.value = jnp.asarray(source_head.kernel.value)
        target_head.bias.value = jnp.asarray(source_head.bias.value)


def _copy_critic_from_source(
    target,
    source_critic,
    obs_dim: int,
    context_dim: int,
) -> None:
    if len(target.layers) != len(source_critic.layers):
        raise ValueError("critic depth changed")
    for index, (target_layer, source_layer) in enumerate(
        zip(target.layers, source_critic.layers)
    ):
        source_kernel = jnp.asarray(source_layer.kernel.value)
        if index == 0:
            kernel = jnp.zeros_like(target_layer.kernel.value)
            if kernel.shape[1] != source_kernel.shape[1] + context_dim:
                raise ValueError("conditioned critic width changed")
            kernel = kernel.at[:, :obs_dim, :].set(
                source_kernel[:, :obs_dim, :])
            kernel = kernel.at[:, obs_dim + context_dim:, :].set(
                source_kernel[:, obs_dim:, :])
            target_layer.kernel.value = kernel
        else:
            target_layer.kernel.value = source_kernel
        target_layer.bias.value = jnp.asarray(source_layer.bias.value)


def warmstart_from_source(source_agent, target_agent) -> None:
    _copy_policy_from_source(target_agent.policy, source_agent.policy)
    _copy_policy_from_source(target_agent.fallback_policy, source_agent.policy)
    _copy_critic_from_source(
        target_agent.critic,
        source_agent.critic,
        target_agent.obs_dim,
        target_agent.context_dim,
    )
    _copy_critic_from_source(
        target_agent.target_critic,
        source_agent.target_critic,
        target_agent.obs_dim,
        target_agent.context_dim,
    )
    target_agent.log_alpha = jnp.asarray(source_agent.log_alpha)
    target_agent.policy_opt_state = target_agent.policy_opt.init(
        nnx.state(target_agent.policy, nnx.Param))
    target_agent.critic_opt_state = target_agent.critic_opt.init(
        nnx.state(target_agent.critic, nnx.Param))
    target_agent.alpha_opt_state = target_agent.alpha_opt.init(
        target_agent.log_alpha)
    target_agent.risk_opt_state = target_agent.risk_opt.init(
        nnx.state(target_agent.risk_critic, nnx.Param))
    target_agent.update_count = int(source_agent.update_count)


def _controller_equivalence(source_agent, target_agent) -> dict[str, Any]:
    observations = jax.random.normal(
        jax.random.PRNGKey(326_101), (64, source_agent.obs_dim))
    actions = jax.random.uniform(
        jax.random.PRNGKey(326_102),
        (64, source_agent.act_dim),
        minval=-1.0,
        maxval=1.0,
    )
    source_action = source_agent.policy.deterministic(observations)
    source_q = source_agent.critic(observations, actions)
    source_target_q = source_agent.target_critic(observations, actions)
    errors = {}
    for mode in protocol.MODES:
        context = jnp.broadcast_to(
            jax.nn.one_hot(mode, len(protocol.MODES))[None, :],
            (observations.shape[0], len(protocol.MODES)),
        )
        critic_obs = jnp.concatenate([observations, context], axis=-1)
        errors[f"candidate_actor_mode_{mode}"] = float(jnp.max(jnp.abs(
            source_action
            - target_agent.policy.deterministic(observations, context))))
        errors[f"critic_mode_{mode}"] = float(jnp.max(jnp.abs(
            source_q - target_agent.critic(critic_obs, actions))))
        errors[f"target_critic_mode_{mode}"] = float(jnp.max(jnp.abs(
            source_target_q
            - target_agent.target_critic(critic_obs, actions))))
    zero_context = jnp.zeros(
        (observations.shape[0], len(protocol.MODES)), dtype=jnp.float32)
    errors["fallback_actor"] = float(jnp.max(jnp.abs(
        source_action
        - target_agent.fallback_policy.deterministic(
            observations, zero_context))))
    errors["log_alpha"] = float(jnp.abs(
        source_agent.log_alpha - target_agent.log_alpha))
    atol = protocol.CONTROLLER_EQUIVALENCE_ATOL
    return {
        "pass": bool(max(errors.values()) <= atol),
        "max_abs_errors": errors,
        "atol": atol,
    }


def _fallback_equivalence(source_agent, target_agent) -> dict[str, Any]:
    observations = jax.random.normal(
        jax.random.PRNGKey(326_103), (128, source_agent.obs_dim))
    context = jnp.zeros(
        (observations.shape[0], len(protocol.MODES)), dtype=jnp.float32)
    error = float(jnp.max(jnp.abs(
        source_agent.policy.deterministic(observations)
        - target_agent.fallback_policy.deterministic(observations, context))))
    atol = protocol.CONTROLLER_EQUIVALENCE_ATOL
    return {
        "pass": bool(error <= atol),
        "max_abs_action_error": error,
        "observations": int(observations.shape[0]),
        "atol": atol,
    }


def _runtime_payload(variant: str, seed: int) -> dict[str, Any]:
    variant = protocol.require_variant(variant)
    return {
        "schema": "bapr.ant-joint-mode-risk-runtime.v26",
        "identity": protocol.identity(variant, seed),
        "physical_samples_per_iter": protocol.PHYSICAL_SAMPLES_PER_ITER,
        "target_samples_per_iter": protocol.TARGET_SAMPLES_PER_ITER,
        "per_mode_target_samples_per_iter": (
            protocol.TARGET_SAMPLES_PER_ITER // len(protocol.MODES)
        ),
        "segment_steps": protocol.SWITCH_SEGMENT_STEPS,
        "risk_objective": "relative",
        "risk_lambda": protocol.RISK_LAMBDA,
        "predecessor_policy": "frozen_robust_actor",
        "target_behavior": "balanced_candidate_robust",
        "controller": "single_shared_true_mode_conditioned_actor",
    }


def _bootstrap(
    variant: str,
    seed: int,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    variant = protocol.require_variant(variant)
    seed = protocol.require_training_seed(seed)
    run_dir = run_dir or protocol.run_dir(variant, seed)
    manifest_path = run_dir / "checkpoints" / protocol.BOOTSTRAP_NAME
    runtime_path = run_dir / "checkpoints" / protocol.RUNTIME_NAME
    identity = protocol.identity(variant, seed)
    if manifest_path.is_file() and runtime_path.is_file():
        manifest = protocol.read_json(manifest_path)
        runtime = protocol.read_json(runtime_path)
        if (
            manifest.get("schema") != protocol.BOOTSTRAP_SCHEMA
            or manifest.get("identity") != identity
            or manifest.get("controller_equivalence", {}).get("pass")
            is not True
            or runtime != _runtime_payload(variant, seed)
        ):
            raise ValueError(f"invalid existing V26 bootstrap: {run_dir}")
        return manifest
    if run_dir.exists():
        raise RuntimeError(f"partial V26 branch exists without bootstrap: {run_dir}")

    source_config, source_agent = _load_source(seed)
    config = _target_config(source_config, run_dir, variant)
    env = make_env(config, seed_offset=0)
    try:
        agent = _make_agent(config, env.obs_dim, env.act_dim)
        warmstart_from_source(source_agent, agent)
        equivalence = _controller_equivalence(source_agent, agent)
        if not equivalence["pass"]:
            raise RuntimeError(
                f"V26 robust warm-start mismatch: {equivalence}")
        run_dir.mkdir(parents=True, exist_ok=False)
        replay = ReplayBuffer(
            env.obs_dim,
            env.act_dim,
            capacity=config.replay_size,
            belief_dim=agent.belief_dim,
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
                "conditioned_actor",
                "conditioned_critic",
                "conditioned_target_critic",
                "alpha",
            ],
            "risk_critic_initialization": "fresh",
            "frozen_fallback_initialization": "source_actor",
            "replay_reset": True,
            "optimizer_reset": True,
        }
        protocol.write_json_atomic(manifest_path, manifest)
        protocol.write_json_atomic(
            runtime_path, _runtime_payload(variant, seed))
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
    run_dir: Path | None = None,
) -> list[str]:
    variant = protocol.require_variant(variant)
    seed = protocol.require_training_seed(seed)
    run_dir = run_dir or protocol.run_dir(variant, seed)
    return [
        sys.executable,
        "-u",
        "-m",
        "jax_experiments.analysis.train_regime_polarity_ant_joint_mode_risk_v26",
        "--algo", "sac",
        "--env", protocol.ENV,
        "--seed", str(seed),
        "--max_iters", str(protocol.FINAL_NEXT_ITERATION[variant]),
        "--save_root", str(run_dir.parent),
        "--run_name", run_dir.name,
        "--env_type", "stochastic_mode",
        "--stochastic_mode_family", protocol.FAMILY,
        "--stochastic_mode_dwell_steps", str(protocol.DWELL_STEPS),
        "--stochastic_mode_dwell_distribution", "fixed",
        "--task_num", str(len(protocol.MODES)),
        "--test_task_num", str(len(protocol.MODES)),
        "--samples_per_iter", str(protocol.PHYSICAL_SAMPLES_PER_ITER),
        "--updates_per_iter", str(protocol.UPDATES_PER_ITER),
        "--start_train_steps", "0",
        "--context_warmup_iters", "0",
        "--ensemble_size", "2",
        "--hidden_dim", "256",
        "--lr", "0.0003",
        "--regime_context_source", "oracle",
        "--max_episode_steps", str(protocol.MAX_EPISODE_STEPS),
        "--backend", "spring",
        "--eval_protocol", "stationary",
        "--log_interval", "50",
        "--eval_episodes", "3",
        "--save_interval", "50",
        "--resume",
        "--min_resume_iteration", str(protocol.SOURCE_NEXT_ITERATION),
    ]


def _runtime_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(protocol.ROOT)
    environment["BAPR_JOINT_MODE_RISK_LAMBDA"] = str(protocol.RISK_LAMBDA)
    environment["BAPR_JOINT_MODE_RISK_ACTOR_START_UPDATE"] = str(
        protocol.RISK_ACTOR_START_UPDATE)
    environment["BAPR_JOINT_MODE_SEGMENT_STEPS"] = str(
        protocol.SWITCH_SEGMENT_STEPS)
    return environment


def _load_final_agent(variant: str, seed: int):
    run_dir = protocol.run_dir(variant, seed)
    config = final_task_sweep.load_config(run_dir)
    config.regime_context_source = "oracle"
    config.joint_mode_risk_lambda = protocol.RISK_LAMBDA
    config.joint_mode_risk_actor_start_update = (
        protocol.RISK_ACTOR_START_UPDATE)
    env = make_env(config, seed_offset=0)
    agent = _make_agent(config, env.obs_dim, env.act_dim)
    replay = ReplayBuffer(
        env.obs_dim, env.act_dim, capacity=1, belief_dim=agent.belief_dim)
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


def validate_bundle(variant: str, seed: int) -> dict[str, Any]:
    variant = protocol.require_variant(variant)
    seed = protocol.require_training_seed(seed)
    directory = protocol.bundle_dir(variant, seed)
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
        or payload.get("identity") != protocol.identity(variant, seed)
        or payload.get("checkpoint") != protocol.expected_checkpoint(variant)
        or payload.get("source_bundle_manifest")
        != protocol.file_record(protocol.source_manifest(seed))
        or payload.get("frozen_fallback_equivalence", {}).get("pass")
        is not True
        or set(payload.get("files") or {}) != expected_files
    ):
        raise ValueError(f"invalid V26 policy bundle: {directory}")
    for relative, record in payload["files"].items():
        path = directory / relative
        if not path.is_file() or protocol.file_record(path) != record:
            raise ValueError(f"changed V26 policy bundle file: {path}")
    return payload


def publish_bundle(variant: str, seed: int) -> dict[str, Any]:
    variant = protocol.require_variant(variant)
    seed = protocol.require_training_seed(seed)
    destination = protocol.bundle_dir(variant, seed)
    if protocol.bundle_manifest(variant, seed).is_file():
        return validate_bundle(variant, seed)
    run_dir = protocol.run_dir(variant, seed)
    config, env, agent, next_iteration, total_steps = _load_final_agent(
        variant, seed)
    try:
        _, source_agent = _load_source(seed)
        fallback_equivalence = _fallback_equivalence(source_agent, agent)
        checkpoint = protocol.checkpoint_record(run_dir)
        runtime = protocol.read_json(
            run_dir / "checkpoints" / protocol.RUNTIME_NAME)
        if (
            checkpoint != protocol.expected_checkpoint(variant)
            or next_iteration != protocol.FINAL_NEXT_ITERATION[variant]
            or total_steps != protocol.FINAL_TOTAL_STEPS[variant]
            or int(agent.update_count)
            != protocol.FINAL_UPDATE_COUNT[variant]
            or int(config.samples_per_iter)
            != protocol.PHYSICAL_SAMPLES_PER_ITER
            or runtime != _runtime_payload(variant, seed)
            or fallback_equivalence["pass"] is not True
        ):
            raise ValueError(f"incomplete V26 training output: {run_dir}")
        signature = run_dir / "logs" / "protocol_signature.json"
        bootstrap = run_dir / "checkpoints" / protocol.BOOTSTRAP_NAME
        runtime_path = run_dir / "checkpoints" / protocol.RUNTIME_NAME
        if not signature.is_file() or not bootstrap.is_file():
            raise FileNotFoundError("V26 training provenance is incomplete")

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
                    "identity": protocol.identity(variant, seed),
                    "checkpoint": checkpoint,
                    "source_bundle_manifest": protocol.file_record(
                        protocol.source_manifest(seed)),
                    "frozen_fallback_equivalence": fallback_equivalence,
                    "files": files,
                    "contents": "final joint policy and compact provenance only",
                },
            )
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    finally:
        if hasattr(env, "close"):
            env.close()
    return validate_bundle(variant, seed)


def run(variant: str, seed: int) -> None:
    variant = protocol.require_variant(variant)
    seed = protocol.require_training_seed(seed)
    protocol.validate_registration()
    if protocol.bundle_manifest(variant, seed).is_file():
        validate_bundle(variant, seed)
        print(f"V26 JOINT MODE ALREADY COMPLETE: {variant} seed={seed}")
        return
    run_dir = protocol.run_dir(variant, seed)
    _bootstrap(variant, seed, run_dir)
    command = training_command(variant, seed, run_dir)
    print("V26 JOINT MODE TRAIN:", " ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=protocol.ROOT,
        env=_runtime_environment(),
        check=True,
    )
    payload = publish_bundle(variant, seed)
    print(
        "V26 JOINT MODE COMPLETE: "
        + json.dumps(payload["identity"], sort_keys=True),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=protocol.VARIANTS, required=True)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for checkpoint-safe execution")
    run(args.variant, args.seed)


if __name__ == "__main__":
    main()
