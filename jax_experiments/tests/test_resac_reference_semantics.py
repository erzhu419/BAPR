"""Focused checks for paper-aligned RE-SAC update semantics."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.algos.resac import RESAC
from jax_experiments.configs.default import Config


def _config(*, beta_bc: float = 0.001, ratio: int = 2) -> Config:
    config = Config()
    config.hidden_dim = 8
    config.ensemble_size = 2
    config.batch_size = 4
    config.resac_beta_bc = beta_bc
    config.resac_critic_actor_ratio = ratio
    config.resac_clip_norm = 1.0
    return config


def _batch(updates: int) -> dict[str, jax.Array]:
    return {
        "obs": jnp.zeros((updates, 4, 3), dtype=jnp.float32),
        "act": jnp.full((updates, 4, 2), 0.75, dtype=jnp.float32),
        "rew": jnp.ones((updates, 4, 1), dtype=jnp.float32),
        "next_obs": jnp.full((updates, 4, 3), 0.1, dtype=jnp.float32),
        "done": jnp.zeros((updates, 4, 1), dtype=jnp.float32),
    }


def _policy_leaves(agent: RESAC) -> list[np.ndarray]:
    return [
        np.asarray(value)
        for value in jax.tree.leaves(nnx.state(agent.policy, nnx.Param))
    ]


def test_actor_ratio_phase_uses_checkpointed_update_count():
    agent = RESAC(3, 2, _config(ratio=2), seed=7)

    first = agent.multi_update(_batch(3))
    assert agent.update_count == 3
    assert np.isclose(first["actor_update_rate"], 2.0 / 3.0)

    second = agent.multi_update(_batch(1))
    assert agent.update_count == 4
    assert second["actor_update_rate"] == 0.0

    resumed = RESAC(3, 2, _config(ratio=2), seed=11)
    resumed.update_count = 3
    resumed_step = resumed.multi_update(_batch(1))
    assert resumed_step["actor_update_rate"] == 0.0


def test_behavior_cloning_term_changes_actor_update():
    without_bc = RESAC(3, 2, _config(beta_bc=0.0, ratio=1), seed=13)
    with_bc = RESAC(3, 2, _config(beta_bc=0.5, ratio=1), seed=13)

    no_bc_metrics = without_bc.multi_update(_batch(1))
    bc_metrics = with_bc.multi_update(_batch(1))

    assert bc_metrics["bc_loss"] > 0.0
    assert np.isclose(no_bc_metrics["bc_loss"], bc_metrics["bc_loss"])
    assert any(
        not np.array_equal(left, right)
        for left, right in zip(
            _policy_leaves(without_bc), _policy_leaves(with_bc))
    )


def test_positive_weight_regularizer_remains_configured():
    config = _config()
    assert config.weight_reg == 0.01
    assert config.beta_ood == 0.01
    assert config.beta_bc == 0.001
    assert config.critic_actor_ratio == 2
    assert config.resac_beta_bc == 0.001
    assert config.resac_critic_actor_ratio == 2


def test_b0_controls_enable_target_blend_and_ema_without_reg_shift():
    config = _config(ratio=2)
    config.weight_reg = 0.0
    config.beta_ood = 0.0
    config.resac_independent_ratio = 0.75
    config.resac_anchor_lambda = 0.01
    config.use_ema_eval = True
    agent = RESAC(3, 2, config, seed=17)
    before = [
        np.asarray(value).copy()
        for value in jax.tree.leaves(
            nnx.state(agent.ema_policy, nnx.Param))
    ]

    metrics = agent.multi_update(_batch(2), current_iter=10)
    after = [
        np.asarray(value)
        for value in jax.tree.leaves(
            nnx.state(agent.ema_policy, nnx.Param))
    ]

    assert metrics["resac_independent_ratio"] == 0.75
    assert metrics["reg_bonus_mean"] == 0.0
    assert metrics["reg_bonus_std"] == 0.0
    assert any(
        not np.array_equal(left, right)
        for left, right in zip(before, after))


def test_checkpoint_anchor_is_remapped_to_current_policy_treedef():
    config = _config(beta_bc=0.0, ratio=1)
    config.weight_reg = 0.0
    config.beta_ood = 0.0
    config.resac_independent_ratio = 0.75
    config.resac_anchor_lambda = 0.01
    agent = RESAC(3, 2, config, seed=23)
    current_leaves, current_def = jax.tree.flatten(agent._anchor_params)

    # A plain list deliberately has a different treedef while preserving the
    # ordered values and shapes, matching the metadata-only drift seen across
    # the two Flax NNX runtimes used by scheduler nodes.
    saved_anchor = [np.asarray(value) for value in current_leaves]
    agent.load_checkpoint_state({
        "kind": "resac_compat_v1",
        "anchor_params": saved_anchor,
        "best_eval": 17.0,
        "current_beta": -1.5,
    })

    restored_leaves, restored_def = jax.tree.flatten(agent._anchor_params)
    assert restored_def == current_def
    assert all(
        np.array_equal(np.asarray(current), np.asarray(restored))
        for current, restored in zip(current_leaves, restored_leaves))
    metrics = agent.multi_update(_batch(1), current_iter=10)
    assert all(np.isfinite(value) for value in metrics.values())
