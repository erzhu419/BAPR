"""Tests for the robust-anchored residual controller."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.algos.anchored_regime_sac import AnchoredRegimeSAC
from jax_experiments.configs.default import Config
from jax_experiments.networks.anchored_dual_critic import (
    AnchoredDualEnsembleCritic,
)
from jax_experiments.networks.anchored_residual_policy import (
    AnchoredResidualGaussianPolicy,
)


def _context(mode: int, valid: float = 1.0, batch: int = 1):
    posterior = jax.nn.one_hot(
        jnp.full((batch,), mode), 4, dtype=jnp.float32)
    return jnp.concatenate([
        posterior,
        jnp.full((batch, 1), valid, dtype=jnp.float32),
    ], axis=-1)


def _config():
    config = Config()
    config.algo = "anchored_regime_sac"
    config.env_type = "stochastic_mode"
    config.task_num = 4
    config.test_task_num = 4
    config.bapr_v2_latent_dim = 4
    config.bapr_v2_policy_mode = "residual"
    config.bapr_v2_training_schedule = "joint"
    config.bapr_v2_critic_target_mode = "min"
    config.bapr_v2_residual_delta = 0.5
    config.bapr_v2_context_hidden_dim = 8
    config.bapr_v3_context_ensemble_size = 2
    config.bapr_v4_training_source_period = 2
    config.bapr_v4_training_robust_slots = 1
    config.hidden_dim = 8
    config.ensemble_size = 2
    config.batch_size = 4
    config.bapr_v2_base_aux_weight = 0.0
    config.bapr_v2_context_dropout = 0.0
    return config


def _batch():
    task_ids = jnp.asarray([[0, 1, 2, 3]], dtype=jnp.int32)
    return {
        "obs": jnp.zeros((1, 4, 3), dtype=jnp.float32),
        "act": jnp.zeros((1, 4, 2), dtype=jnp.float32),
        "rew": jnp.ones((1, 4, 1), dtype=jnp.float32),
        "next_obs": jnp.full((1, 4, 3), 0.1, dtype=jnp.float32),
        "done": jnp.zeros((1, 4, 1), dtype=jnp.float32),
        "task_id": task_ids,
        "belief": jnp.zeros((1, 4, 5), dtype=jnp.float32),
        "next_belief": jnp.zeros((1, 4, 5), dtype=jnp.float32),
    }


def test_residual_starts_at_exact_robust_policy():
    policy = AnchoredResidualGaussianPolicy(
        3, 2, 8, 4, residual_delta=0.5, rngs=nnx.Rngs(1))
    obs = jnp.asarray([[0.2, -0.1, 0.3]], dtype=jnp.float32)
    robust = np.asarray(policy.deterministic(obs, None))
    for mode in range(4):
        np.testing.assert_array_equal(
            np.asarray(policy.deterministic(obs, _context(mode))),
            robust,
        )


def test_actor_context_gate_isolates_gradient_paths():
    policy = AnchoredResidualGaussianPolicy(
        3, 2, 8, 4, residual_delta=0.5, rngs=nnx.Rngs(2))
    graphdef = nnx.graphdef(policy)
    params = nnx.state(policy, nnx.Param)
    obs = jnp.asarray([[0.2, -0.1, 0.3]], dtype=jnp.float32)

    def adaptive_loss(values):
        model = nnx.merge(graphdef, values)
        mean, _ = model(obs, _context(2))
        return jnp.sum(mean)

    def robust_loss(values):
        model = nnx.merge(graphdef, values)
        mean, _ = model(obs, jnp.zeros((1, 5), dtype=jnp.float32))
        return jnp.sum(mean)

    adaptive_grad = jax.grad(adaptive_loss)(params)
    robust_grad = jax.grad(robust_loss)(params)
    adaptive_base = jax.tree.leaves(adaptive_grad["base_layers"])
    adaptive_base += jax.tree.leaves(adaptive_grad["base_mean"])
    adaptive_base += jax.tree.leaves(adaptive_grad["base_log_std"])
    robust_residual = jax.tree.leaves(robust_grad["residual_layers"])
    robust_residual += jax.tree.leaves(robust_grad["residual_mean"])
    assert all(np.allclose(np.asarray(value), 0.0)
               for value in adaptive_base)
    assert all(np.allclose(np.asarray(value), 0.0)
               for value in robust_residual)
    assert any(not np.allclose(np.asarray(value), 0.0)
               for value in jax.tree.leaves(
                   adaptive_grad["residual_mean"]))


def test_dual_critic_isolates_robust_and_adaptive_parameters():
    critic = AnchoredDualEnsembleCritic(
        3, 2, 5, 8, 4, 2, n_layers=2, rngs=nnx.Rngs(3))
    graphdef = nnx.graphdef(critic)
    params = nnx.state(critic, nnx.Param)
    obs = jnp.asarray([[0.1, 0.2, -0.3]], dtype=jnp.float32)
    act = jnp.asarray([[0.4, -0.2]], dtype=jnp.float32)

    def loss(values, context):
        model = nnx.merge(graphdef, values)
        return jnp.sum(model(
            jnp.concatenate([obs, context], axis=-1), act))

    adaptive_grad = jax.grad(loss)(
        params, _context(1))
    robust_grad = jax.grad(loss)(
        params, jnp.zeros((1, 5), dtype=jnp.float32))
    assert all(np.allclose(np.asarray(value), 0.0)
               for value in jax.tree.leaves(
                   adaptive_grad["base_critic"]))
    assert all(np.allclose(np.asarray(value), 0.0)
               for value in jax.tree.leaves(
                   robust_grad["adaptive_critic"]))


def test_anchored_update_is_finite_and_dual_context():
    agent = AnchoredRegimeSAC(3, 2, _config(), seed=4)
    agent.set_task_metadata([{"mode_id": mode} for mode in range(4)])
    relabelled = agent._dual_context_batch(_batch())
    assert relabelled["obs"].shape == (1, 8, 3)
    assert agent.log_alpha.shape == (5,)
    metrics = agent.multi_update(_batch(), current_iter=1)
    assert np.isfinite(metrics["critic_loss"])
    assert np.isfinite(metrics["policy_loss"])
    assert np.isfinite(metrics["alpha"])
    assert metrics["anchored_residual"] == 1.0
