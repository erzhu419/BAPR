"""Run SAC with Ant switch-state rollout collection for V24."""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments import train as train_module
from jax_experiments.algos.sac_switch_recovery import SACSwitchRecovery


_ORIGINAL_MAKE_ALGO = train_module.make_algo
_ORIGINAL_COLLECT_SAMPLES = train_module.collect_samples


def predecessor_mode(
    target_mode: int,
    seed: int,
    iteration: int,
    cycle: int,
) -> int:
    """Return a deterministic balanced predecessor distinct from target."""
    alternatives = tuple(mode for mode in range(4) if mode != target_mode)
    offset = int(seed) + 17 * int(iteration) + int(cycle)
    return alternatives[offset % len(alternatives)]


def _make_algo(algo_name, obs_dim, act_dim, config):
    if algo_name != "sac":
        return _ORIGINAL_MAKE_ALGO(algo_name, obs_dim, act_dim, config)
    config.switch_recovery_target_mode = int(
        os.environ["BAPR_SWITCH_RECOVERY_TARGET_MODE"])
    config.switch_recovery_segment_steps = int(
        os.environ["BAPR_SWITCH_RECOVERY_SEGMENT_STEPS"])
    config.switch_recovery_termination_penalty = float(
        os.environ["BAPR_SWITCH_RECOVERY_TERMINATION_PENALTY"])
    return SACSwitchRecovery(obs_dim, act_dim, config, seed=config.seed)


def collect_switch_recovery_samples(
    agent,
    env,
    replay_buffer,
    config,
    n_steps: int,
    current_iter: int,
):
    """Collect equal robust-prefix and post-switch specialist transitions."""
    segment_steps = int(agent.switch_recovery_segment_steps)
    pair_steps = 2 * segment_steps
    if n_steps <= 0 or n_steps % pair_steps:
        raise ValueError(
            "switch-recovery physical steps must be divisible by two "
            f"segments: n_steps={n_steps}, segment_steps={segment_steps}")
    if int(getattr(env, "num_modes", -1)) != 4:
        raise ValueError("switch-recovery collection requires four modes")
    activate = getattr(env, "_activate_mode", None)
    if not callable(activate):
        raise TypeError("switch-recovery collection requires mode activation")

    target_mode = int(agent.switch_recovery_target_mode)
    candidate_params = nnx.state(agent.policy, nnx.Param)
    fallback_params = nnx.state(agent.fallback_policy, nnx.Param)
    cycles = n_steps // pair_steps
    root_key = jax.random.PRNGKey(
        int(config.seed) + 1_000_003 * (int(current_iter) + 2))
    keys = jax.random.split(root_key, 2 * cycles)
    target_chunks = [[] for _ in range(5)]
    raw_segment_returns = []

    for cycle in range(cycles):
        predecessor = predecessor_mode(
            target_mode, int(config.seed), int(current_iter), cycle)
        activate(predecessor)
        env.rollout(
            fallback_params,
            segment_steps,
            keys[2 * cycle],
            continue_state=cycle > 0,
        )
        activate(target_mode)
        transitions, _ = env.rollout(
            candidate_params,
            segment_steps,
            keys[2 * cycle + 1],
            continue_state=True,
        )
        for chunks, values in zip(target_chunks, transitions):
            chunks.append(values)
        raw_segment_returns.append(float(jnp.sum(transitions[2])))

    obs, act, raw_rew, next_obs, done = tuple(
        jnp.concatenate(chunks, axis=0) for chunks in target_chunks)
    penalty = jnp.asarray(
        agent.switch_recovery_termination_penalty, dtype=raw_rew.dtype)
    train_rew = raw_rew - penalty * done
    task_ids = jnp.full(
        (obs.shape[0],), target_mode, dtype=jnp.int32)
    replay_buffer.push_batch_jax(
        obs,
        act,
        train_rew.reshape(-1, 1),
        next_obs,
        done.reshape(-1, 1),
        task_ids,
    )
    agent._last_switch_recovery_termination_rate = float(jnp.mean(done))
    agent._last_switch_recovery_raw_reward = float(jnp.mean(raw_rew))
    recent_rollout = {
        "obs": obs,
        "act": act,
        "rew": train_rew.reshape(-1, 1),
        "raw_rew": raw_rew.reshape(-1, 1),
        "next_obs": next_obs,
        "done": done.reshape(-1, 1),
        "task_id": task_ids,
        "rollout_task_id": target_mode,
        "rollout_task_ids": np.full(cycles, target_mode, dtype=np.int32),
        "physical_steps": int(n_steps),
        "candidate_steps": int(obs.shape[0]),
    }
    return raw_segment_returns, recent_rollout


def _collect_samples(
    agent,
    env,
    replay_buffer,
    config,
    n_steps: int,
    current_iter: int = 0,
):
    if not isinstance(agent, SACSwitchRecovery):
        return _ORIGINAL_COLLECT_SAMPLES(
            agent, env, replay_buffer, config, n_steps, current_iter)
    return collect_switch_recovery_samples(
        agent, env, replay_buffer, config, n_steps, current_iter)


def main() -> None:
    missing = [
        name for name in (
            "BAPR_SWITCH_RECOVERY_TARGET_MODE",
            "BAPR_SWITCH_RECOVERY_SEGMENT_STEPS",
            "BAPR_SWITCH_RECOVERY_TERMINATION_PENALTY",
        )
        if name not in os.environ
    ]
    if missing:
        raise SystemExit("missing V24 environment: " + ",".join(missing))
    train_module.make_algo = _make_algo
    train_module.collect_samples = _collect_samples
    train_module.main()


if __name__ == "__main__":
    main()
