"""Train the V26 shared Ant mode-conditioned risk controller."""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments import train as train_module
from jax_experiments.algos.joint_mode_risk_sac import JointModeRiskSAC
from jax_experiments.analysis import (
    train_regime_polarity_ant_switch_recovery_v24 as switch_training,
)


_ORIGINAL_MAKE_ALGO = train_module.make_algo
_ORIGINAL_COLLECT_SAMPLES = train_module.collect_samples


def _make_algo(algo_name, obs_dim, act_dim, config):
    if algo_name != "sac":
        return _ORIGINAL_MAKE_ALGO(algo_name, obs_dim, act_dim, config)
    config.regime_context_source = "oracle"
    config.joint_mode_risk_lambda = float(
        os.environ["BAPR_JOINT_MODE_RISK_LAMBDA"])
    config.joint_mode_risk_actor_start_update = int(
        os.environ["BAPR_JOINT_MODE_RISK_ACTOR_START_UPDATE"])
    return JointModeRiskSAC(obs_dim, act_dim, config, seed=config.seed)


def collection_plan(
    seed: int, current_iter: int, cycles: int, mode_count: int,
) -> tuple[tuple[int, bool], ...]:
    """Return a mode-balanced and behavior-balanced segment plan."""
    if mode_count <= 1 or cycles % (2 * mode_count):
        raise ValueError("collection plan cannot balance modes and behavior")
    offset = (int(seed) + int(current_iter)) % mode_count
    return tuple(
        (
            (cycle + offset) % mode_count,
            (cycle // mode_count) % 2 == 1,
        )
        for cycle in range(cycles)
    )


def collect_joint_mode_samples(
    agent,
    env,
    replay_buffer,
    config,
    n_steps: int,
    current_iter: int,
):
    """Collect balanced post-switch data for all modes and both behaviors."""
    if not isinstance(agent, JointModeRiskSAC):
        return _ORIGINAL_COLLECT_SAMPLES(
            agent, env, replay_buffer, config, n_steps, current_iter)
    segment_steps = int(os.environ["BAPR_JOINT_MODE_SEGMENT_STEPS"])
    pair_steps = 2 * segment_steps
    mode_count = int(agent.context_dim)
    cycles = n_steps // pair_steps
    if n_steps <= 0 or n_steps % pair_steps:
        raise ValueError("joint-mode physical steps require complete pairs")
    if cycles % (2 * mode_count):
        raise ValueError("joint-mode cycles cannot balance modes and behavior")
    if int(getattr(env, "num_modes", -1)) != mode_count:
        raise ValueError("joint-mode collection mode count changed")
    activate = getattr(env, "_activate_mode", None)
    if not callable(activate):
        raise TypeError("joint-mode collection requires mode activation")

    candidate_params = nnx.state(agent.policy, nnx.Param)
    fallback_params = nnx.state(agent.fallback_policy, nnx.Param)
    root_key = jax.random.PRNGKey(
        int(config.seed) + 1_000_033 * (int(current_iter) + 2))
    keys = jax.random.split(root_key, 2 * cycles)
    target_chunks = [[] for _ in range(5)]
    context_chunks = []
    task_id_chunks = []
    raw_segment_returns = []
    mode_counts = np.zeros((mode_count,), dtype=np.int32)
    candidate_segments = 0
    fallback_segments = 0
    plan = collection_plan(config.seed, current_iter, cycles, mode_count)
    zero_context = jnp.zeros((mode_count,), dtype=jnp.float32)

    for cycle, (target_mode, use_candidate) in enumerate(plan):
        predecessor = switch_training.predecessor_mode(
            target_mode, int(config.seed), int(current_iter), cycle)
        activate(predecessor)
        env.rollout(
            fallback_params,
            segment_steps,
            keys[2 * cycle],
            belief_vec=zero_context,
            continue_state=cycle > 0,
        )
        activate(target_mode)
        target_context = jax.nn.one_hot(
            target_mode, mode_count, dtype=jnp.float32)
        behavior_params = candidate_params if use_candidate else fallback_params
        behavior_context = target_context if use_candidate else zero_context
        transitions, _ = env.rollout(
            behavior_params,
            segment_steps,
            keys[2 * cycle + 1],
            belief_vec=behavior_context,
            continue_state=True,
        )
        for chunks, values in zip(target_chunks, transitions):
            chunks.append(values)
        context_chunks.append(jnp.broadcast_to(
            target_context[None, :], (segment_steps, mode_count)))
        task_id_chunks.append(jnp.full(
            (segment_steps,), target_mode, dtype=jnp.int32))
        raw_segment_returns.append(float(jnp.sum(transitions[2])))
        mode_counts[target_mode] += segment_steps
        candidate_segments += int(use_candidate)
        fallback_segments += int(not use_candidate)

    obs, act, rew, next_obs, done = tuple(
        jnp.concatenate(chunks, axis=0) for chunks in target_chunks)
    contexts = jnp.concatenate(context_chunks, axis=0)
    task_ids = jnp.concatenate(task_id_chunks, axis=0)
    replay_buffer.push_batch_jax(
        obs,
        act,
        rew.reshape(-1, 1),
        next_obs,
        done.reshape(-1, 1),
        task_ids,
        belief=contexts,
        next_belief=contexts,
    )
    agent._last_mode_counts = jnp.asarray(mode_counts)
    agent._last_candidate_fraction = float(
        candidate_segments / max(cycles, 1))
    agent._last_termination_rate = float(jnp.mean(done))
    recent_rollout = {
        "obs": obs,
        "act": act,
        "rew": rew.reshape(-1, 1),
        "next_obs": next_obs,
        "done": done.reshape(-1, 1),
        "task_id": task_ids,
        "context": contexts,
        "next_context": contexts,
        "rollout_task_id": int(task_ids[-1]),
        "rollout_task_ids": np.asarray(
            [target_mode for target_mode, _ in plan],
            dtype=np.int32,
        ),
        "physical_steps": int(n_steps),
        "target_steps": int(obs.shape[0]),
        "mode_target_steps": mode_counts.tolist(),
        "candidate_behavior_segments": candidate_segments,
        "fallback_behavior_segments": fallback_segments,
    }
    return raw_segment_returns, recent_rollout


def main() -> None:
    required = (
        "BAPR_JOINT_MODE_RISK_LAMBDA",
        "BAPR_JOINT_MODE_RISK_ACTOR_START_UPDATE",
        "BAPR_JOINT_MODE_SEGMENT_STEPS",
    )
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise SystemExit("missing V26 environment: " + ",".join(missing))
    train_module.make_algo = _make_algo
    train_module.collect_samples = collect_joint_mode_samples
    train_module.main()


if __name__ == "__main__":
    main()
