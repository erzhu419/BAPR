"""Train V25 Ant specialists with a learned termination-risk constraint."""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments import train as train_module
from jax_experiments.algos.sac_switch_recovery_risk import (
    SACSwitchRecoveryRisk,
)
from jax_experiments.analysis import (
    train_regime_polarity_ant_switch_recovery_v24 as switch_training,
)


_ORIGINAL_MAKE_ALGO = train_module.make_algo


def _make_algo(algo_name, obs_dim, act_dim, config):
    if algo_name != "sac":
        return _ORIGINAL_MAKE_ALGO(algo_name, obs_dim, act_dim, config)
    config.switch_recovery_target_mode = int(
        os.environ["BAPR_SWITCH_RECOVERY_TARGET_MODE"])
    config.switch_recovery_segment_steps = int(
        os.environ["BAPR_SWITCH_RECOVERY_SEGMENT_STEPS"])
    config.switch_recovery_termination_penalty = 0.0
    config.switch_recovery_risk_objective = os.environ[
        "BAPR_SWITCH_RECOVERY_RISK_OBJECTIVE"]
    config.switch_recovery_risk_lambda = float(
        os.environ["BAPR_SWITCH_RECOVERY_RISK_LAMBDA"])
    config.switch_recovery_risk_actor_start_update = int(
        os.environ["BAPR_SWITCH_RECOVERY_RISK_ACTOR_START_UPDATE"])
    return SACSwitchRecoveryRisk(
        obs_dim, act_dim, config, seed=config.seed)


def collect_constrained_risk_samples(
    agent,
    env,
    replay_buffer,
    config,
    n_steps: int,
    current_iter: int,
):
    """Collect target-mode data under balanced candidate/robust behavior."""
    if not isinstance(agent, SACSwitchRecoveryRisk):
        return switch_training._collect_samples(
            agent, env, replay_buffer, config, n_steps, current_iter)
    segment_steps = int(agent.switch_recovery_segment_steps)
    pair_steps = 2 * segment_steps
    if n_steps <= 0 or n_steps % pair_steps:
        raise ValueError(
            "constrained-risk physical steps must contain complete pairs")
    if int(getattr(env, "num_modes", -1)) != 4:
        raise ValueError("constrained-risk collection requires four modes")
    activate = getattr(env, "_activate_mode", None)
    if not callable(activate):
        raise TypeError("constrained-risk collection requires mode activation")

    target_mode = int(agent.switch_recovery_target_mode)
    candidate_params = nnx.state(agent.policy, nnx.Param)
    fallback_params = nnx.state(agent.fallback_policy, nnx.Param)
    cycles = n_steps // pair_steps
    if cycles % 2:
        raise ValueError("constrained-risk behavior mix requires even cycles")
    root_key = jax.random.PRNGKey(
        int(config.seed) + 1_000_003 * (int(current_iter) + 2))
    keys = jax.random.split(root_key, 2 * cycles)
    target_chunks = [[] for _ in range(5)]
    raw_segment_returns = []
    candidate_segments = 0
    fallback_segments = 0

    for cycle in range(cycles):
        predecessor = switch_training.predecessor_mode(
            target_mode, int(config.seed), int(current_iter), cycle)
        activate(predecessor)
        env.rollout(
            fallback_params,
            segment_steps,
            keys[2 * cycle],
            continue_state=cycle > 0,
        )
        activate(target_mode)
        use_candidate = cycle % 2 == 1
        behavior_params = candidate_params if use_candidate else fallback_params
        transitions, _ = env.rollout(
            behavior_params,
            segment_steps,
            keys[2 * cycle + 1],
            continue_state=True,
        )
        for chunks, values in zip(target_chunks, transitions):
            chunks.append(values)
        raw_segment_returns.append(float(jnp.sum(transitions[2])))
        candidate_segments += int(use_candidate)
        fallback_segments += int(not use_candidate)

    obs, act, raw_rew, next_obs, done = tuple(
        jnp.concatenate(chunks, axis=0) for chunks in target_chunks)
    task_ids = jnp.full((obs.shape[0],), target_mode, dtype=jnp.int32)
    replay_buffer.push_batch_jax(
        obs,
        act,
        raw_rew.reshape(-1, 1),
        next_obs,
        done.reshape(-1, 1),
        task_ids,
    )
    agent._last_switch_recovery_termination_rate = float(jnp.mean(done))
    agent._last_switch_recovery_raw_reward = float(jnp.mean(raw_rew))
    recent_rollout = {
        "obs": obs,
        "act": act,
        "rew": raw_rew.reshape(-1, 1),
        "raw_rew": raw_rew.reshape(-1, 1),
        "next_obs": next_obs,
        "done": done.reshape(-1, 1),
        "task_id": task_ids,
        "rollout_task_id": target_mode,
        "rollout_task_ids": np.full(cycles, target_mode, dtype=np.int32),
        "physical_steps": int(n_steps),
        "candidate_steps": int(obs.shape[0]),
        "candidate_behavior_segments": candidate_segments,
        "fallback_behavior_segments": fallback_segments,
    }
    return raw_segment_returns, recent_rollout


def main() -> None:
    required = (
        "BAPR_SWITCH_RECOVERY_TARGET_MODE",
        "BAPR_SWITCH_RECOVERY_SEGMENT_STEPS",
        "BAPR_SWITCH_RECOVERY_RISK_OBJECTIVE",
        "BAPR_SWITCH_RECOVERY_RISK_LAMBDA",
        "BAPR_SWITCH_RECOVERY_RISK_ACTOR_START_UPDATE",
    )
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise SystemExit("missing V25 environment: " + ",".join(missing))
    train_module.make_algo = _make_algo
    train_module.collect_samples = collect_constrained_risk_samples
    train_module.main()


if __name__ == "__main__":
    main()
