"""Shared strict-evaluation helpers for the anchored residual protocol."""
from __future__ import annotations

import copy
import tempfile
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.analysis import final_task_sweep
from jax_experiments.analysis import (
    regime_polarity_anchored_residual as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_expected_action_system_id_model as model_lib,
)
from jax_experiments.analysis import (
    run_regime_polarity_anchored_branch as branch_runner,
)
from jax_experiments.analysis import (
    run_regime_polarity_inverse_system_id_audit as inverse_audit,
)
from jax_experiments.common.checkpoint import load_checkpoint
from jax_experiments.common.logging import Logger
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.train import (
    _reset_eval_switch_schedule,
    _select_eval_switch_sequence,
    make_algo,
    make_env,
)


CALIBRATION_ARMS = (
    "robust_continue",
    "anchored_base",
    "oracle_residual",
)
AUDIT_ARMS = (
    "robust_continue",
    "anchored_base",
    "oracle_residual",
    "oracle_safe",
    "learned_raw",
    "learned_safe",
)
LEARNED_ARMS = ("learned_raw", "learned_safe")
ANCHORED_ARMS = tuple(
    arm for arm in AUDIT_ARMS if arm != "robust_continue")
InverseSystemIDEstimator = inverse_audit.InverseSystemIDEstimator


def load_controller(role: str, seed: int):
    role = protocol.require_branch_role(role)
    seed = protocol.require_training_seed(seed)
    branch_runner.validate_published(seed, role)
    directory = protocol.branch_bundle_dir(role, seed)
    config = final_task_sweep.load_config(directory)
    expected_algo = (
        "regime_sac" if role == "robust_continue"
        else "anchored_regime_sac"
    )
    if (config.algo != expected_algo
            or config.env_name != protocol.ENV
            or config.env_type != "stochastic_mode"
            or config.stochastic_mode_family != protocol.FAMILY):
        raise ValueError(f"wrong anchored controller config: {directory}")
    env = make_env(config, seed_offset=0)
    tasks = env.sample_tasks(config.task_num)
    agent = make_algo(config.algo, env.obs_dim, env.act_dim, config)
    agent.set_task_metadata(tasks)
    replay = ReplayBuffer(
        env.obs_dim, env.act_dim, capacity=1,
        belief_dim=getattr(agent, "belief_dim", 0))
    with tempfile.TemporaryDirectory() as log_dir:
        logger = Logger(log_dir)
        next_iteration, total_steps = load_checkpoint(
            str(directory / "checkpoints"),
            agent,
            replay,
            logger,
            config.algo,
            load_replay_buffer=False,
        )
    if (next_iteration != protocol.BRANCH_FINAL_NEXT_ITERATION
            or total_steps != protocol.BRANCH_TOTAL_STEPS
            or agent.update_count != protocol.BRANCH_UPDATE_COUNT):
        raise ValueError(f"stale anchored controller: {directory}")
    if hasattr(env, "close"):
        env.close()
    return config, agent, nnx.state(agent.policy, nnx.Param)


def load_estimator():
    protocol.validate_frozen_estimator()
    model, filter_config, gains, variance, manifest = model_lib.load_model(
        # HalfCheetah dimensions are checked again against the controller by
        # the caller before evaluation.
        obs_dim=17,
        act_dim=6,
    )
    estimator = InverseSystemIDEstimator(
        model_lib.one_step_evidence(model),
        nnx.state(model, nnx.Param),
        gains,
        variance,
        filter_config,
    )
    return estimator, manifest


def policy_action_fn(agent):
    graphdef = nnx.graphdef(agent.policy)

    @jax.jit
    def action(params, observation, context):
        policy = nnx.merge(graphdef, params)
        return policy.deterministic(
            jnp.asarray(observation)[None],
            jnp.asarray(context)[None],
        )[0]

    return action


def arm_source(arm: str) -> str:
    if arm == "robust_continue":
        return "robust_continue"
    if arm in ANCHORED_ARMS:
        return "anchored"
    raise ValueError(f"unknown anchored audit arm {arm!r}")


def arm_context(
    arm: str,
    physical_mode: int,
    posterior: np.ndarray,
    mode_mask: np.ndarray,
) -> np.ndarray:
    mode = protocol.require_mode(physical_mode)
    posterior = np.asarray(posterior, dtype=np.float32)
    mode_mask = np.asarray(mode_mask, dtype=bool)
    if posterior.shape != (len(protocol.MODES),):
        raise ValueError("anchored posterior has the wrong width")
    if mode_mask.shape != (len(protocol.MODES),):
        raise ValueError("anchored mode mask has the wrong width")
    if arm == "robust_continue":
        return np.zeros((len(protocol.MODES),), dtype=np.float32)
    if arm == "anchored_base":
        return np.zeros((len(protocol.MODES) + 1,), dtype=np.float32)
    if arm in ("oracle_residual", "oracle_safe"):
        latent = np.eye(
            len(protocol.MODES), dtype=np.float32)[mode]
        enabled = arm == "oracle_residual" or bool(mode_mask[mode])
        return np.concatenate([
            latent if enabled else np.zeros_like(latent),
            np.asarray([float(enabled)], dtype=np.float32),
        ])
    if arm == "learned_raw":
        return np.concatenate([
            posterior,
            np.ones((1,), dtype=np.float32),
        ])
    if arm == "learned_safe":
        selected = int(np.argmax(posterior))
        enabled = (
            float(np.max(posterior)) >= protocol.CONFIDENCE_THRESHOLD
            and bool(mode_mask[selected])
        )
        return np.concatenate([
            posterior if enabled else np.zeros_like(posterior),
            np.asarray([float(enabled)], dtype=np.float32),
        ])
    raise ValueError(f"unknown anchored audit arm {arm!r}")


def strict_stationary(
    config,
    arm: str,
    policy_state,
    action_fn,
    event_seed: int,
    *,
    mode_mask: np.ndarray,
    estimator=None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries = []
    metric_rows = []
    for mode in protocol.MODES:
        run_config = copy.deepcopy(config)
        run_config.stochastic_mode_fixed_id = int(mode)
        env = make_env(
            run_config,
            seed_offset=int(event_seed) - int(config.seed),
        )
        tasks = env.sample_tasks(len(protocol.MODES))
        env.set_nonstationary_para(tasks)
        env.set_task(tasks[mode])
        returns = []
        terminated = []
        posteriors = []
        labels = []
        for _ in range(protocol.EPISODES_PER_TASK):
            obs = env.reset()
            estimator_state = (
                estimator.initial_state() if estimator is not None else None)
            episode_return = 0.0
            episode_terminated = False
            for _ in range(protocol.MAX_EPISODE_STEPS):
                posterior = (
                    estimator.probabilities(estimator_state)
                    if estimator is not None
                    else np.full(
                        (len(protocol.MODES),),
                        1.0 / len(protocol.MODES),
                        dtype=np.float64,
                    )
                )
                context = arm_context(
                    arm, mode, posterior, mode_mask)
                action = np.asarray(
                    action_fn(policy_state, obs, context))
                next_obs, reward, done, info = env.step(action)
                if int(info["mode_used"]) != int(mode):
                    raise ValueError("stationary anchored mode changed")
                if arm in LEARNED_ARMS:
                    posteriors.append(posterior.copy())
                    labels.append(int(mode))
                    estimator_state, _, _, _ = estimator.step(
                        estimator_state,
                        obs,
                        action,
                        reward,
                        next_obs,
                    )
                episode_return += float(reward)
                obs = next_obs
                if done:
                    episode_terminated = True
                    obs = env.reset()
                    if arm in LEARNED_ARMS:
                        estimator_state = estimator.initial_state()
            returns.append(episode_return)
            terminated.append(float(episode_terminated))
        summaries.append({
            "arm": arm,
            "mode": int(mode),
            "return_mean": float(np.mean(returns)),
            "return_std": float(np.std(returns)),
            "returns": [float(value) for value in returns],
            "terminated_rate": float(np.mean(terminated)),
        })
        if arm in LEARNED_ARMS:
            metric_rows.append({
                "arm": arm,
                "mode": int(mode),
                **protocol.posterior_metrics(
                    np.asarray(posteriors),
                    np.asarray(labels, dtype=np.int32),
                ),
            })
        if hasattr(env, "close"):
            env.close()
    return summaries, metric_rows


def strict_switching(
    config,
    arm: str,
    policy_state,
    action_fn,
    estimator,
    event_seed: int,
    *,
    mode_mask: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    run_config = copy.deepcopy(config)
    run_config.stochastic_mode_fixed_id = -1
    env = make_env(
        run_config,
        seed_offset=int(event_seed) - int(config.seed),
    )
    tasks = env.sample_tasks(len(protocol.MODES))
    returns = []
    terminated = []
    trace: dict[str, list[Any]] = {
        "episode": [],
        "step": [],
        "physics_mode": [],
        "posterior_before": [],
        "posterior_after": [],
        "context": [],
        "log_likelihood": [],
        "aleatoric": [],
        "epistemic": [],
        "reward": [],
        "done": [],
    }
    for episode in range(protocol.SWITCHING_EPISODES):
        sequence, _ = _select_eval_switch_sequence(env, tasks, episode)
        _reset_eval_switch_schedule(
            env, sequence, run_config, protocol.DWELL_STEPS)
        obs = env.reset()
        estimator_state = estimator.initial_state()
        episode_return = 0.0
        episode_terminated = False
        for step in range(protocol.MAX_EPISODE_STEPS):
            physical_mode = int(env.task_id_for_next_step())
            posterior = estimator.probabilities(estimator_state)
            context = arm_context(
                arm, physical_mode, posterior, mode_mask)
            action = np.asarray(action_fn(policy_state, obs, context))
            next_obs, reward, done, info = env.step(action)
            if int(info["mode_used"]) != physical_mode:
                raise ValueError("switching anchored label is not causal")
            if arm in LEARNED_ARMS:
                next_state, evidence, aleatoric, epistemic = estimator.step(
                    estimator_state,
                    obs,
                    action,
                    reward,
                    next_obs,
                )
                next_posterior = estimator.probabilities(next_state)
                trace["episode"].append(episode)
                trace["step"].append(step)
                trace["physics_mode"].append(physical_mode)
                trace["posterior_before"].append(posterior.copy())
                trace["posterior_after"].append(next_posterior.copy())
                trace["context"].append(context.copy())
                trace["log_likelihood"].append(evidence)
                trace["aleatoric"].append(aleatoric)
                trace["epistemic"].append(epistemic)
                trace["reward"].append(float(reward))
                trace["done"].append(bool(done))
                estimator_state = next_state
            episode_return += float(reward)
            obs = next_obs
            if done:
                episode_terminated = True
                # Preserve the physical switch clock and posterior across the
                # simulator-state reset, matching the strict prior audits.
                obs = env.reset()
        returns.append(episode_return)
        terminated.append(float(episode_terminated))
    if hasattr(env, "close"):
        env.close()
    arrays = {
        key: np.asarray(value)
        for key, value in trace.items()
    }
    summary = {
        "arm": arm,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "returns": [float(value) for value in returns],
        "terminated_rate": float(np.mean(terminated)),
    }
    if arm in LEARNED_ARMS:
        summary["posterior_metrics"] = protocol.posterior_metrics(
            arrays["posterior_before"],
            arrays["physics_mode"],
        )
        enabled = arrays["context"][:, -1]
        summary["fallback"] = {
            "adaptation_enabled_fraction": float(np.mean(enabled)),
            "fallback_fraction": float(np.mean(1.0 - enabled)),
        }
    return summary, arrays
