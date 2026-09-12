"""Train one equal-budget frozen-anchor v2 controller branch."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.analysis import final_task_sweep
from jax_experiments.analysis import (
    regime_polarity_frozen_anchor as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_anchored_branch as v1_runner,
)
from jax_experiments.common.checkpoint import load_checkpoint, save_checkpoint
from jax_experiments.common.logging import Logger
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.train import make_algo, make_env


ACTION_ATOL = 1e-5
CRITIC_ATOL = 2e-3
CRITIC_RTOL = 2e-3


def _required_bundle_files() -> set[str]:
    return {
        "checkpoints/params.pkl",
        "checkpoints/train_state.pkl",
        "checkpoints/" + protocol.BOOTSTRAP_NAME,
        "logs/protocol_signature.json",
    }


def _variant_training_config(variant: dict[str, object]) -> dict[str, object]:
    return {
        "bapr_v2_train_advantage_constraint": bool(
            variant.get("train_advantage_constraint", False)),
        "bapr_v2_train_advantage_lcb_scale": float(
            variant.get("train_advantage_lcb_scale", 1.0)),
        "bapr_v2_train_advantage_margin": float(
            variant.get("train_advantage_margin", 0.0)),
        "bapr_v2_train_advantage_temperature": float(
            variant.get("train_advantage_temperature", 0.01)),
        "bapr_v2_train_advantage_weight": float(
            variant.get("train_advantage_weight", 1.0)),
        "bapr_v2_train_update_filter": bool(
            variant.get("train_update_filter", False)),
        "bapr_v2_train_update_tolerance": float(
            variant.get("train_update_tolerance", 0.005)),
        "bapr_v2_train_update_floor": float(
            variant.get("train_update_floor", -0.01)),
    }


def _copy_value(destination, value) -> None:
    destination.value = jnp.asarray(value, dtype=destination.value.dtype)


def _load_source(seed: int):
    seed = protocol.require_training_seed(seed)
    v1_runner.validate_published(seed, "robust_continue")
    bundle = protocol.source_bundle_dir(seed)
    config = final_task_sweep.load_config(bundle)
    if (
        config.algo != "regime_sac"
        or config.regime_context_source != "robust"
        or config.env_name != protocol.ENV
        or config.stochastic_mode_family != protocol.FAMILY
    ):
        raise ValueError("frozen anchor requires the v1 robust continuation")
    env = make_env(config, seed_offset=0)
    tasks = env.sample_tasks(config.task_num)
    agent = make_algo(config.algo, env.obs_dim, env.act_dim, config)
    agent.set_task_metadata(tasks)
    replay = ReplayBuffer(
        env.obs_dim,
        env.act_dim,
        capacity=1,
        belief_dim=agent.belief_dim,
    )
    temporary = tempfile.TemporaryDirectory()
    logger = Logger(temporary.name)
    next_iteration, total_steps = load_checkpoint(
        str(bundle / "checkpoints"),
        agent,
        replay,
        logger,
        config.algo,
        load_replay_buffer=False,
    )
    if (
        next_iteration != protocol.SOURCE_NEXT_ITERATION
        or total_steps != protocol.SOURCE_TOTAL_STEPS
        or agent.update_count != protocol.SOURCE_UPDATE_COUNT
    ):
        temporary.cleanup()
        raise ValueError("frozen-anchor source has the wrong budget")
    return config, env, tasks, agent, logger, temporary


def _robust_config(source, run_dir: Path):
    config = deepcopy(source)
    config.save_root = str(run_dir.parent)
    config.run_name = run_dir.name
    config.max_iters = protocol.BRANCH_FINAL_NEXT_ITERATION
    config.start_train_steps = 0
    config.stochastic_mode_fixed_id = -1
    config.log_interval = 50
    config.eval_episodes = 3
    config.save_interval = 25
    config.resume = True
    return config


def _adaptive_config(source, role: str, run_dir: Path):
    variant = protocol.VARIANT_CONFIGS[protocol.require_variant(role)]
    config = deepcopy(source)
    config.algo = variant["algo"]
    config.save_root = str(run_dir.parent)
    config.run_name = run_dir.name
    config.max_iters = protocol.BRANCH_FINAL_NEXT_ITERATION
    config.start_train_steps = 0
    config.stochastic_mode_fixed_id = -1
    config.bapr_v2_mode = "supervised"
    config.bapr_v2_latent_dim = len(protocol.MODES)
    config.bapr_v2_policy_context_source = "stored"
    config.bapr_v2_training_schedule = "joint"
    config.bapr_v2_policy_mode = "residual"
    config.bapr_v2_context_hidden_dim = 128
    config.bapr_v2_context_length = 64
    config.bapr_v2_context_chunks = 8
    config.bapr_v2_context_burnin = 16
    config.bapr_v2_min_history = 16
    config.bapr_v2_switch_rollout_steps = protocol.DWELL_STEPS
    config.bapr_v2_residual_delta = variant["residual_delta"]
    config.bapr_v2_action_deviation_weight = (
        variant["action_deviation_weight"])
    for field, value in _variant_training_config(variant).items():
        setattr(config, field, value)
    config.bapr_v2_base_aux_weight = 0.0
    config.bapr_v2_context_dropout = 0.0
    config.bapr_v2_advantage_gate = False
    config.bapr_v2_actor_objective = "mean"
    config.bapr_v2_critic_target_mode = "min"
    config.bapr_v2_beta_ood = 0.0
    config.bapr_v2_reg_weight = 0.0
    config.bapr_v3_context_ensemble_size = 5
    config.bapr_v3_variance_model = "mode_empirical"
    config.bapr_v3_variance_ceiling = 0.5
    config.bapr_v3_variance_ema = 0.05
    config.bapr_v4_training_source_period = 1
    config.bapr_v4_training_robust_slots = 0
    config.bapr_v4_context_bootstrap_model = ""
    config.bapr_v4_context_bootstrap_manifest = ""
    config.context_warmup_iters = 0
    config.log_interval = 25
    config.eval_episodes = 3
    config.save_interval = 25
    config.resume = True
    return config


def _copy_source_policy_to_mode(source_agent, target_agent) -> None:
    source = source_agent.policy
    target = target_agent.policy
    if len(source.layers) != len(target.base_layers):
        raise ValueError("source and mode actor depths differ")
    for index, (source_layer, base_layer) in enumerate(
        zip(source.layers, target.base_layers)
    ):
        kernel = source_layer.kernel.value
        if index == 0:
            expected = source_agent.obs_dim + source_agent.context_dim
            if kernel.shape[0] != expected:
                raise ValueError("source actor input width changed")
            kernel = kernel[:source_agent.obs_dim]
        _copy_value(base_layer.kernel, kernel)
        _copy_value(base_layer.bias, source_layer.bias.value)
    for source_head, base_head in (
        (source.mean_head, target.base_mean),
        (source.log_std_head, target.base_log_std),
    ):
        _copy_value(base_head.kernel, source_head.kernel.value)
        _copy_value(base_head.bias, source_head.bias.value)
    for mode_head in (target.mode_mean, target.mode_log_std):
        _copy_value(
            mode_head.kernel, jnp.zeros_like(mode_head.kernel.value))
        _copy_value(
            mode_head.bias, jnp.zeros_like(mode_head.bias.value))
    if not target.zero_mode_residual_output():
        raise ValueError("mode residuals are not zero after source copy")


def _copy_source_critic_to_options(
    source,
    target,
    *,
    obs_dim: int,
    source_context_dim: int,
) -> None:
    v1_runner.copy_source_ensemble_critic(
        source,
        target.base_critic,
        obs_dim=obs_dim,
        source_context_dim=source_context_dim,
        target_context_dim=0,
    )
    if len(source.layers) != len(target.option_layers):
        raise ValueError("source and mode critic depths differ")
    for index, (source_layer, option_layer) in enumerate(
        zip(source.layers, target.option_layers)
    ):
        kernel = source_layer.kernel.value
        if index == 0:
            source_action = obs_dim + source_context_dim
            action_dim = kernel.shape[1] - source_action
            reduced = jnp.zeros(
                (kernel.shape[0], obs_dim + action_dim, kernel.shape[2]),
                dtype=kernel.dtype,
            )
            reduced = reduced.at[:, :obs_dim].set(kernel[:, :obs_dim])
            reduced = reduced.at[:, obs_dim:].set(
                kernel[:, source_action:])
            kernel = reduced
        tiled_kernel = jnp.tile(
            kernel, (len(protocol.MODES), 1, 1))
        tiled_bias = jnp.tile(
            source_layer.bias.value, (len(protocol.MODES), 1, 1))
        if tiled_kernel.shape != option_layer.kernel.value.shape:
            raise ValueError("mode critic kernel shape mismatch")
        _copy_value(option_layer.kernel, tiled_kernel)
        _copy_value(option_layer.bias, tiled_bias)


def _copy_source_controller(source_agent, target_agent, role: str) -> None:
    role = protocol.require_variant(role)
    if (
        protocol.VARIANT_CONFIGS[role]["algo"]
        == "frozen_anchored_regime_sac"
    ):
        v1_runner.copy_source_controller(source_agent, target_agent)
        return

    _copy_source_policy_to_mode(source_agent, target_agent)
    for source_critic, target_critic in (
        (source_agent.critic, target_agent.critic),
        (source_agent.target_critic, target_agent.target_critic),
    ):
        _copy_source_critic_to_options(
            source_critic,
            target_critic,
            obs_dim=source_agent.obs_dim,
            source_context_dim=source_agent.context_dim,
        )
    source_log_alpha = jnp.asarray(source_agent.log_alpha)
    if source_log_alpha.ndim != 0:
        raise ValueError("robust source alpha must be scalar")
    target_agent.log_alpha = jnp.full(
        (len(protocol.MODES) + 1,),
        source_log_alpha,
        dtype=source_log_alpha.dtype,
    )
    target_agent.update_count = int(source_agent.update_count)


def _equivalence(source_agent, target_agent) -> dict[str, object]:
    obs_key, act_key = jax.random.split(jax.random.PRNGKey(20260729))
    obs = jax.random.normal(obs_key, (64, source_agent.obs_dim))
    act = jnp.tanh(jax.random.normal(
        act_key, (64, source_agent.act_dim)))
    source_context = jnp.zeros(
        (64, source_agent.context_dim), dtype=obs.dtype)
    source_action = source_agent.policy.deterministic(obs, source_context)
    source_q = source_agent.critic(
        jnp.concatenate([obs, source_context], axis=-1), act)
    action_errors = []
    critic_errors = []
    contexts = [
        jnp.zeros(
            (64, len(protocol.MODES) + 1), dtype=obs.dtype)
    ]
    contexts.extend([
        jnp.broadcast_to(
            jnp.concatenate([
                jax.nn.one_hot(mode, len(protocol.MODES)),
                jnp.ones((1,), dtype=obs.dtype),
            ])[None],
            (64, len(protocol.MODES) + 1),
        )
        for mode in protocol.MODES
    ])
    for context in contexts:
        action = target_agent.policy.deterministic(obs, context)
        q_value = target_agent.critic(
            jnp.concatenate([obs, context], axis=-1), act)
        action_errors.append(float(jnp.max(jnp.abs(
            source_action - action))))
        critic_errors.append(float(jnp.max(jnp.abs(source_q - q_value))))
    action_error = max(action_errors)
    critic_error = max(critic_errors)
    critic_scale = float(jnp.max(jnp.abs(source_q)))
    critic_threshold = (
        CRITIC_ATOL + CRITIC_RTOL * max(critic_scale, 1.0))
    return {
        "pass": bool(
            action_error <= ACTION_ATOL
            and critic_error <= critic_threshold),
        "max_abs_action_error": action_error,
        "max_abs_critic_error": critic_error,
        "max_abs_source_q": critic_scale,
        "critic_threshold": critic_threshold,
        "action_atol": ACTION_ATOL,
        "critic_atol": CRITIC_ATOL,
        "critic_rtol": CRITIC_RTOL,
        "contexts_checked": len(contexts),
    }


def _policy_hashes(agent, role: str) -> dict[str, str]:
    if (
        protocol.VARIANT_CONFIGS[role]["algo"]
        == "frozen_anchored_regime_sac"
    ):
        values = protocol.anchored_policy_hashes(agent.policy)
        return {"base": values["base"], "adaptive": values["residual"]}
    return protocol.mode_policy_hashes(agent.policy)


def _component_hashes(agent, role: str) -> dict[str, str]:
    policy = _policy_hashes(agent, role)
    critic = protocol.critic_hashes(agent.critic)
    target = protocol.critic_hashes(agent.target_critic)
    return {
        "base_policy": policy["base"],
        "adaptive_policy": policy["adaptive"],
        "base_critic": critic["base"],
        "adaptive_critic": critic["adaptive"],
        "base_target_critic": target["base"],
        "adaptive_target_critic": target["adaptive"],
    }


def _bootstrap(seed: int, role: str, run_dir: Path) -> dict:
    role = protocol.require_branch_role(role)
    manifest = run_dir / "checkpoints" / protocol.BOOTSTRAP_NAME
    if manifest.is_file():
        payload = protocol.read_json(manifest)
        if (
            payload.get("schema") != protocol.BOOTSTRAP_SCHEMA
            or payload.get("identity") != protocol.branch_identity(role, seed)
        ):
            raise ValueError(f"invalid frozen-anchor bootstrap: {manifest}")
        return payload
    if run_dir.exists():
        raise RuntimeError(
            f"partial frozen-anchor branch lacks bootstrap: {run_dir}")

    source_config, env, tasks, source_agent, logger, temporary = (
        _load_source(seed))
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
        if role == "robust_long":
            config = _robust_config(source_config, run_dir)
            agent = source_agent
            agent.config = config
            v1_runner._reset_optimizers(agent)
            equivalence = {
                "pass": True,
                "max_abs_action_error": 0.0,
                "max_abs_critic_error": 0.0,
                "contexts_checked": 1,
            }
            components = {
                "source_policy": protocol.source_policy_sha256(agent.policy),
            }
        else:
            config = _adaptive_config(source_config, role, run_dir)
            agent = make_algo(
                config.algo, env.obs_dim, env.act_dim, config)
            agent.set_task_metadata(tasks)
            _copy_source_controller(source_agent, agent, role)
            v1_runner._reset_optimizers(agent)
            equivalence = _equivalence(source_agent, agent)
            if not equivalence["pass"]:
                raise RuntimeError(
                    "frozen-anchor bootstrap changed source function: "
                    f"{equivalence}")
            components = _component_hashes(agent, role)

        replay = ReplayBuffer(
            env.obs_dim,
            env.act_dim,
            capacity=config.replay_size,
            belief_dim=getattr(agent, "belief_dim", 0),
        )
        save_checkpoint(
            str(run_dir / "checkpoints"),
            agent,
            replay,
            logger,
            iteration=protocol.SOURCE_FINAL_ITERATION,
            total_steps=protocol.SOURCE_TOTAL_STEPS,
            algo=config.algo,
        )
        payload = {
            "schema": protocol.BOOTSTRAP_SCHEMA,
            "status": "complete",
            "identity": protocol.branch_identity(role, seed),
            "semantics": (
                "fork from the completed 8.4M-transition robust controller; "
                "empty replay and reset optimizers; adaptive branches freeze "
                "the robust actor for every subsequent update"),
            "source": {
                "bundle_manifest": protocol.file_record(
                    protocol.source_manifest(seed)),
                "checkpoint_next_iteration": protocol.SOURCE_NEXT_ITERATION,
                "checkpoint_total_steps": protocol.SOURCE_TOTAL_STEPS,
                "checkpoint_update_count": protocol.SOURCE_UPDATE_COUNT,
            },
            "empty_replay": True,
            "optimizer_states_reset": True,
            "robust_actor_frozen": role != "robust_long",
            "function_equivalence": equivalence,
            "initial_components": components,
        }
        protocol.write_json_atomic(manifest, payload)
        return payload
    except Exception:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    finally:
        temporary.cleanup()
        if hasattr(env, "close"):
            env.close()


def _training_command(seed: int, role: str, run_dir: Path) -> list[str]:
    role = protocol.require_branch_role(role)
    if role == "robust_long":
        values = [
            "--algo", "regime_sac",
            "--regime_context_source", "robust",
        ]
    else:
        variant = protocol.VARIANT_CONFIGS[role]
        conservative = _variant_training_config(variant)
        values = [
            "--algo", str(variant["algo"]),
            "--bapr_v2_mode", "supervised",
            "--bapr_v2_latent_dim", "4",
            "--bapr_v2_policy_context_source", "stored",
            "--bapr_v2_training_schedule", "joint",
            "--bapr_v2_policy_mode", "residual",
            "--bapr_v2_context_hidden_dim", "128",
            "--bapr_v2_context_length", "64",
            "--bapr_v2_context_chunks", "8",
            "--bapr_v2_context_burnin", "16",
            "--bapr_v2_min_history", "16",
            "--bapr_v2_switch_rollout_steps", str(protocol.DWELL_STEPS),
            "--bapr_v2_residual_delta",
            str(variant["residual_delta"]),
            "--bapr_v2_action_deviation_weight",
            str(variant["action_deviation_weight"]),
            "--bapr_v2_base_aux_weight", "0.0",
            "--bapr_v2_context_dropout", "0.0",
            "--bapr_v2_actor_objective", "mean",
            "--bapr_v2_critic_target_mode", "min",
            "--bapr_v2_beta_ood", "0.0",
            "--bapr_v2_reg_weight", "0.0",
            "--bapr_v3_context_ensemble_size", "5",
            "--bapr_v3_variance_model", "mode_empirical",
            "--bapr_v3_variance_ceiling", "0.5",
            "--bapr_v3_variance_ema", "0.05",
            "--bapr_v4_training_source_period", "1",
            "--bapr_v4_training_robust_slots", "0",
        ]
        for field in (
            "bapr_v2_train_advantage_lcb_scale",
            "bapr_v2_train_advantage_margin",
            "bapr_v2_train_advantage_temperature",
            "bapr_v2_train_advantage_weight",
            "bapr_v2_train_update_tolerance",
            "bapr_v2_train_update_floor",
        ):
            values.extend([
                "--" + field,
                str(conservative[field]),
            ])
        if conservative["bapr_v2_train_advantage_constraint"]:
            values.append("--bapr_v2_train_advantage_constraint")
        if conservative["bapr_v2_train_update_filter"]:
            values.append("--bapr_v2_train_update_filter")
    return [
        sys.executable,
        "-u",
        "-m",
        "jax_experiments.train",
        *values,
        "--env",
        protocol.ENV,
        "--seed",
        str(protocol.require_training_seed(seed)),
        "--max_iters",
        str(protocol.BRANCH_FINAL_NEXT_ITERATION),
        "--save_root",
        str(run_dir.parent),
        "--run_name",
        run_dir.name,
        "--env_type",
        "stochastic_mode",
        "--stochastic_mode_family",
        protocol.FAMILY,
        "--stochastic_mode_dwell_steps",
        str(protocol.DWELL_STEPS),
        "--stochastic_mode_dwell_distribution",
        "fixed",
        "--task_num",
        "4",
        "--test_task_num",
        "4",
        "--samples_per_iter",
        str(protocol.SAMPLES_PER_ITER),
        "--updates_per_iter",
        str(protocol.UPDATES_PER_ITER),
        "--start_train_steps",
        "0",
        "--context_warmup_iters",
        "0",
        "--ensemble_size",
        "10",
        "--hidden_dim",
        "256",
        "--lr",
        "0.0003",
        "--max_episode_steps",
        str(protocol.MAX_EPISODE_STEPS),
        "--eval_protocol",
        "stationary",
        "--log_interval",
        "25" if role != "robust_long" else "50",
        "--eval_episodes",
        "3",
        "--save_interval",
        "25",
        "--min_resume_iteration",
        str(protocol.SOURCE_NEXT_ITERATION),
        "--resume",
    ]


def _load_final(run_dir: Path):
    config = final_task_sweep.load_config(run_dir)
    env = make_env(config, seed_offset=0)
    tasks = env.sample_tasks(config.task_num)
    agent = make_algo(config.algo, env.obs_dim, env.act_dim, config)
    if hasattr(agent, "set_task_metadata"):
        agent.set_task_metadata(tasks)
    replay = ReplayBuffer(
        env.obs_dim,
        env.act_dim,
        capacity=1,
        belief_dim=getattr(agent, "belief_dim", 0),
    )
    temporary = tempfile.TemporaryDirectory()
    logger = Logger(temporary.name)
    next_iteration, total_steps = load_checkpoint(
        str(run_dir / "checkpoints"),
        agent,
        replay,
        logger,
        config.algo,
        load_replay_buffer=False,
    )
    return config, env, agent, next_iteration, total_steps, temporary


def validate_branch(seed: int, role: str, run_dir: Path) -> dict:
    role = protocol.require_branch_role(role)
    bootstrap = protocol.read_json(
        run_dir / "checkpoints" / protocol.BOOTSTRAP_NAME)
    if (
        bootstrap.get("schema") != protocol.BOOTSTRAP_SCHEMA
        or bootstrap.get("identity") != protocol.branch_identity(role, seed)
        or bootstrap.get("empty_replay") is not True
        or bootstrap.get("optimizer_states_reset") is not True
    ):
        raise ValueError(f"invalid frozen-anchor bootstrap: {run_dir}")
    config, env, agent, next_iteration, total_steps, temporary = (
        _load_final(run_dir))
    try:
        checkpoint = protocol.checkpoint_record(run_dir)
        expected = protocol.expected_branch_checkpoint(role)
        if checkpoint != expected:
            raise ValueError(
                f"frozen-anchor branch has wrong budget: {checkpoint}")
        if (
            next_iteration != expected["next_iteration"]
            or total_steps != expected["total_steps"]
        ):
            raise ValueError("loaded frozen-anchor state disagrees")
        signature = protocol.read_json(
            run_dir / "logs" / "protocol_signature.json")
        config_record = signature.get("config") or {}
        if (
            signature.get("checkpoint_loaded") is not True
            or signature.get("start_iteration")
            != protocol.SOURCE_NEXT_ITERATION
            or signature.get("total_steps_at_start")
            != protocol.SOURCE_TOTAL_STEPS
            or config_record.get("start_train_steps") != 0
        ):
            raise ValueError("frozen-anchor branch missed fork boundary")
        result = {
            "identity": protocol.branch_identity(role, seed),
            "checkpoint": checkpoint,
            "bootstrap": bootstrap,
            "protocol_signature": signature,
        }
        if role == "robust_long":
            if (
                config.algo != "regime_sac"
                or config.regime_context_source != "robust"
            ):
                raise ValueError("robust-long config changed")
            return result

        variant = protocol.VARIANT_CONFIGS[role]
        required = {
            "algo": variant["algo"],
            "bapr_v2_training_schedule": "joint",
            "bapr_v2_critic_target_mode": "min",
            "bapr_v4_training_source_period": 1,
            "bapr_v4_training_robust_slots": 0,
            "bapr_v2_residual_delta": variant["residual_delta"],
            "bapr_v2_action_deviation_weight": (
                variant["action_deviation_weight"]),
            **_variant_training_config(variant),
        }
        mismatches = {
            key: {"actual": getattr(config, key), "expected": value}
            for key, value in required.items()
            if getattr(config, key) != value
        }
        if mismatches:
            raise ValueError(
                f"frozen-anchor final config changed: {mismatches}")
        final_components = _component_hashes(agent, role)
        initial = bootstrap["initial_components"]
        if final_components["base_policy"] != initial["base_policy"]:
            raise ValueError("frozen robust actor changed after bootstrap")
        conservative_config = _variant_training_config(variant)
        adaptive_changed = (
            final_components["adaptive_policy"]
            != initial["adaptive_policy"])
        if not conservative_config["bapr_v2_train_update_filter"]:
            if not adaptive_changed:
                raise ValueError("adaptive policy did not update")
        else:
            acceptance = np.asarray(np.load(
                run_dir / "logs" / "v2_train_update_accept_rate.npy",
                allow_pickle=False,
            )[-protocol.BRANCH_EXTRA_ITERS:], dtype=np.float64)
            advantage = np.asarray(np.load(
                run_dir / "logs" / "v2_train_advantage_lcb.npy",
                allow_pickle=False,
            )[-protocol.BRANCH_EXTRA_ITERS:], dtype=np.float64)
            shortfall = np.asarray(np.load(
                run_dir / "logs" / "v2_train_advantage_shortfall.npy",
                allow_pickle=False,
            )[-protocol.BRANCH_EXTRA_ITERS:], dtype=np.float64)
            if (
                acceptance.size != protocol.BRANCH_EXTRA_ITERS
                or advantage.size != protocol.BRANCH_EXTRA_ITERS
                or shortfall.size != protocol.BRANCH_EXTRA_ITERS
                or not np.all(np.isfinite(acceptance))
                or not np.all(np.isfinite(advantage))
                or not np.all(np.isfinite(shortfall))
                or np.any((acceptance < 0.0) | (acceptance > 1.0))
            ):
                raise ValueError("invalid conservative-training metrics")
            if not adaptive_changed and np.any(acceptance > 0.0):
                raise ValueError(
                    "accepted residual updates left policy unchanged")
            result["conservative_training"] = {
                "adaptive_policy_changed": adaptive_changed,
                "update_accept_rate_mean": float(np.mean(acceptance)),
                "update_accept_rate_min": float(np.min(acceptance)),
                "update_accept_rate_max": float(np.max(acceptance)),
                "advantage_lcb_mean": float(np.mean(advantage)),
                "advantage_lcb_min": float(np.min(advantage)),
                "advantage_shortfall_mean": float(np.mean(shortfall)),
            }
        train_base = np.load(
            run_dir / "logs" / "v2_train_base.npy", allow_pickle=False)
        if np.any(np.asarray(
                train_base[-protocol.BRANCH_EXTRA_ITERS:]) != 0.0):
            raise ValueError("frozen branch attempted a base actor update")
        sources = np.load(
            run_dir / "logs" / "v4_training_context_source.npy",
            allow_pickle=False,
        )
        if np.any(np.asarray(
                sources[-protocol.BRANCH_EXTRA_ITERS:]) != 1.0):
            raise ValueError("frozen branch used a non-oracle rollout")
        modes = np.load(
            run_dir / "logs" / "mode_id.npy", allow_pickle=False)
        observed = {
            int(value)
            for value in modes[protocol.SOURCE_NEXT_ITERATION:]
        }
        if (
            protocol.BRANCH_EXTRA_ITERS >= 20
            and observed != set(protocol.MODES)
        ):
            raise ValueError(
                f"frozen branch missed modes: {observed}")
        result["final_components"] = final_components
        return result
    finally:
        temporary.cleanup()
        if hasattr(env, "close"):
            env.close()


def validate_published(seed: int, role: str) -> dict:
    directory = protocol.branch_bundle_dir(role, seed)
    payload = protocol.read_json(directory / "bundle_manifest.json")
    if (
        payload.get("schema") != protocol.BRANCH_BUNDLE_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("identity") != protocol.branch_identity(role, seed)
        or payload.get("checkpoint")
        != protocol.expected_branch_checkpoint(role)
        or set(payload.get("files") or {}) != _required_bundle_files()
    ):
        raise ValueError(f"invalid frozen-anchor bundle: {directory}")
    for relative, record in payload["files"].items():
        path = directory / relative
        if not path.is_file() or protocol.file_record(path) != record:
            raise ValueError(f"frozen-anchor bundle changed: {path}")
    return payload


def publish_bundle(seed: int, role: str, run_dir: Path) -> dict:
    validation = validate_branch(seed, role, run_dir)
    destination = protocol.branch_bundle_dir(role, seed)
    if (destination / "bundle_manifest.json").is_file():
        return validate_published(seed, role)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp.", dir=destination.parent))
    try:
        records = {}
        for relative in (
            Path("checkpoints") / "params.pkl",
            Path("checkpoints") / "train_state.pkl",
            Path("checkpoints") / protocol.BOOTSTRAP_NAME,
            Path("logs") / "protocol_signature.json",
        ):
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(run_dir / relative, target)
            records[relative.as_posix()] = protocol.file_record(target)
        payload = {
            "schema": protocol.BRANCH_BUNDLE_SCHEMA,
            "status": "complete",
            "identity": protocol.branch_identity(role, seed),
            "checkpoint": validation["checkpoint"],
            "budget_semantics": {
                "shared_source_steps": protocol.SOURCE_TOTAL_STEPS,
                "additional_steps": (
                    protocol.BRANCH_EXTRA_ITERS
                    * protocol.SAMPLES_PER_ITER),
                "controller_updates": protocol.BRANCH_UPDATE_COUNT,
                "base_actor_frozen": role != "robust_long",
                "oracle_rollouts_only": role != "robust_long",
                "dual_context_relabel": role != "robust_long",
            },
            "components": validation.get("final_components"),
            "conservative_training": validation.get(
                "conservative_training"),
            "files": records,
        }
        protocol.write_json_atomic(
            temporary / "bundle_manifest.json", payload)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return validate_published(seed, role)


def run(seed: int, role: str) -> None:
    seed = protocol.require_training_seed(seed)
    role = protocol.require_branch_role(role)
    run_dir = protocol.branch_run_dir(role, seed)
    destination = protocol.branch_bundle_dir(role, seed)
    if (destination / "bundle_manifest.json").is_file():
        validate_published(seed, role)
        print(f"FROZEN ANCHOR ALREADY COMPLETE: {destination}")
        return
    _bootstrap(seed, role, run_dir)
    expected = protocol.expected_branch_checkpoint(role)
    current = protocol.checkpoint_record(run_dir)
    if current["next_iteration"] < expected["next_iteration"]:
        command = _training_command(seed, role, run_dir)
        print("FROZEN ANCHOR TRAIN:", " ".join(command), flush=True)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(protocol.ROOT)
        subprocess.run(
            command,
            cwd=protocol.ROOT,
            env=environment,
            check=True,
        )
    payload = publish_bundle(seed, role, run_dir)
    replay = run_dir / "checkpoints" / "replay_buffer.npz"
    if replay.is_file():
        replay.unlink()
    print(
        "FROZEN ANCHOR COMPLETE: "
        + json.dumps(payload["identity"], sort_keys=True),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        choices=protocol.TRAINING_SEEDS,
        type=int,
        required=True,
    )
    parser.add_argument(
        "--role",
        choices=protocol.BRANCH_ROLES,
        required=True,
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for checkpoint-safe execution")
    run(args.seed, args.role)


if __name__ == "__main__":
    main()
