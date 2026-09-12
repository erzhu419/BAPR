"""Focused checks for the stable ESCP paper-core compatibility path."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jax_experiments.algos.escp import ESCP
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.configs.default import Config
from jax_experiments.networks.escp_recurrent import (
    RecurrentEnvironmentProbe,
)


def _config() -> Config:
    config = Config()
    config.hidden_dim = 8
    config.ensemble_size = 2
    config.ep_dim = 2
    config.batch_size = 4
    config.clip_norm = 1.0
    config.samples_per_iter = 10
    config.initial_random_steps = 0
    config.changing_period = 20
    config.task_num = 40
    config.context_warmup_iters = 0
    config.escp_target_mode = "twin_min"
    config.escp_actor_mode = "twin_min"
    config.escp_context_min_steps = 100
    config.escp_context_min_tasks = 5
    config.escp_alpha_max = 1.0
    config.escp_finite_guard = True
    return config


def _batch(updates: int, *, nan_reward: bool = False):
    rewards = jnp.ones((updates, 4, 1), dtype=jnp.float32)
    if nan_reward:
        rewards = rewards.at[0, 0, 0].set(jnp.nan)
    return {
        "obs": jnp.zeros((updates, 4, 3), dtype=jnp.float32),
        "act": jnp.zeros((updates, 4, 2), dtype=jnp.float32),
        "rew": rewards,
        "next_obs": jnp.full((updates, 4, 3), 0.1, dtype=jnp.float32),
        "done": jnp.zeros((updates, 4, 1), dtype=jnp.float32),
        "task_id": jnp.asarray(
            [[0, 1, 2, 3]] * updates, dtype=jnp.int32),
    }


def _recurrent_config() -> Config:
    config = _config()
    config.escp_context_mode = "recurrent"
    config.escp_history_length = 4
    config.rmdm_max_tasks = 4
    config.task_num = 4
    config.escp_context_min_steps = 0
    config.escp_context_min_tasks = 0
    config.escp_policy_lr = 3e-4
    config.escp_critic_lr = 1e-3
    config.escp_context_lr = 3e-4
    config.escp_alpha_lr = 1e-2
    config.escp_target_entropy_ratio = 1.5
    config.escp_bottleneck_sigma = 1e-2
    return config


def _recurrent_batch(updates: int = 2):
    buffer = ReplayBuffer(3, 2, capacity=16)
    observations = jnp.arange(24, dtype=jnp.float32).reshape(8, 3) / 10.0
    actions = jnp.zeros((8, 2), dtype=jnp.float32)
    rewards = jnp.ones((8, 1), dtype=jnp.float32)
    next_observations = observations + 0.1
    dones = jnp.asarray(
        [[0], [0], [1], [0], [0], [0], [0], [0]],
        dtype=jnp.float32)
    task_ids = jnp.asarray(
        [0, 0, 0, 1, 1, 2, 2, 3], dtype=jnp.int32)
    buffer.push_batch_jax(
        observations, actions, rewards, next_observations, dones,
        task_ids)
    return buffer.sample_stacked_sequences(
        updates, 4, 4, rng_key=jax.random.PRNGKey(13))


def _parameter_arrays(agent: ESCP):
    return [
        np.asarray(value).copy()
        for module in (agent.policy, agent.critic, agent.context_net)
        for value in jax.tree.leaves(nnx.state(module, nnx.Param))
    ]


def test_context_phase_waits_for_steps_and_task_coverage():
    agent = ESCP(3, 2, _config(), seed=3)
    assert agent._context_phase(0) == (False, False, 1)
    assert agent._context_phase(9) == (True, True, 6)


def test_paper_core_update_is_finite_and_clamps_alpha():
    agent = ESCP(3, 2, _config(), seed=5)
    metrics = agent.multi_update(_batch(3), current_iter=9)

    assert metrics["finite_update_rate"] == 1.0
    assert metrics["context_train"] is True
    assert metrics["context_active"] is True
    assert metrics["alpha"] <= 1.0
    assert all(
        np.isfinite(value)
        for key, value in metrics.items()
        if key not in {"context_train", "context_active"})


def test_finite_guard_rejects_bad_scan_before_applying_parameters():
    agent = ESCP(3, 2, _config(), seed=7)
    before = _parameter_arrays(agent)

    with pytest.raises(FloatingPointError, match="global_update=0"):
        agent.multi_update(
            _batch(2, nan_reward=True), current_iter=9)

    after = _parameter_arrays(agent)
    assert agent.update_count == 0
    for left, right in zip(before, after):
        np.testing.assert_array_equal(left, right)


def test_recurrent_probe_is_causal_and_honors_reset_boundaries():
    probe = RecurrentEnvironmentProbe(3, 2, rngs=nnx.Rngs(17))
    prefix = jnp.arange(9, dtype=jnp.float32).reshape(1, 3, 3)
    left = jnp.concatenate(
        [prefix, jnp.zeros((1, 1, 3), dtype=jnp.float32)], axis=1)
    right = jnp.concatenate(
        [prefix, jnp.full((1, 1, 3), 99.0, dtype=jnp.float32)], axis=1)
    previous_actions = jnp.zeros((1, 4, 2), dtype=jnp.float32)
    _, left_context = probe.sequence(left, previous_actions)
    _, right_context = probe.sequence(right, previous_actions)
    np.testing.assert_allclose(
        left_context[:, :3], right_context[:, :3], atol=0.0, rtol=0.0)

    reset_mask = jnp.asarray([[False, False, False, True]])
    _, reset_context = probe.sequence(
        right, previous_actions, reset_mask)
    _, isolated_context = probe.sequence(
        right[:, -1:, :], previous_actions[:, -1:, :])
    np.testing.assert_allclose(
        reset_context[:, -1], isolated_context[:, -1], atol=1e-6)


def test_sequence_replay_is_reset_aware_after_ring_wrap():
    buffer = ReplayBuffer(1, 1, capacity=6)
    for step in range(9):
        buffer.push(
            np.asarray([step], dtype=np.float32),
            np.asarray([100 + step], dtype=np.float32),
            float(step), np.asarray([step + 0.5], dtype=np.float32),
            False, task_id=step % 4,
            episode_start=step in (0, 3, 6))
    batch = buffer.sample_stacked_sequences(
        4, 16, 4, rng_key=jax.random.PRNGKey(23))
    observations = np.asarray(batch["obs"])[..., 0]
    previous_actions = np.asarray(batch["prev_act"])[..., 0]
    resets = np.asarray(batch["reset_before"])
    next_observations = np.asarray(batch["next_obs"])[..., 0]
    next_previous_actions = np.asarray(batch["next_prev_act"])[..., 0]
    next_resets = np.asarray(batch["next_reset_before"])

    assert np.all(observations[..., -1] >= 3.0)
    np.testing.assert_allclose(
        next_observations[..., -1], observations[..., -1] + 0.5)
    np.testing.assert_allclose(
        next_previous_actions[..., -1],
        100.0 + observations[..., -1])
    assert not np.any(next_resets[..., -1])
    expected_previous = 99.0 + observations
    np.testing.assert_allclose(
        previous_actions[~resets], expected_previous[~resets])
    np.testing.assert_array_equal(
        previous_actions[resets], np.zeros_like(previous_actions[resets]))


def test_recurrent_escp_update_is_finite_and_updates_prototypes():
    agent = ESCP(3, 2, _recurrent_config(), seed=29)
    metrics = agent.multi_update(_recurrent_batch(), current_iter=20)
    assert metrics["finite_update_rate"] == 1.0
    assert metrics["context_train"] is True
    assert bool(np.any(np.asarray(agent.context_prototype_valid)))
    assert all(
        np.isfinite(value)
        for key, value in metrics.items()
        if key not in {"context_train", "context_active"})


def test_recurrent_online_state_resets_after_termination():
    agent = ESCP(3, 2, _recurrent_config(), seed=31)
    initial = agent.snapshot_recurrent_context()
    agent.select_action(np.ones(3, dtype=np.float32), deterministic=True)
    changed = agent.snapshot_recurrent_context()
    assert not np.allclose(np.asarray(initial[0]), np.asarray(changed[0]))
    agent.finish_recurrent_step(True)
    reset = agent.snapshot_recurrent_context()
    np.testing.assert_array_equal(initial[0], reset[0])
    np.testing.assert_array_equal(initial[1], reset[1])
