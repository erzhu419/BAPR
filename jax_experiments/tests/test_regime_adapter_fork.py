"""Unit tests for the frozen-base independent-adapter protocol."""
from __future__ import annotations

from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np

from jax_experiments.algos.bapr_regime import BAPRRegime
from jax_experiments.algos.regime_sac import RegimeSAC
from jax_experiments.analysis import regime_adapter_fork as protocol
from jax_experiments.configs.default import Config


def _source_config() -> Config:
    config = Config()
    config.algo = "regime_sac"
    config.env_type = "stochastic_mode"
    config.task_num = 4
    config.test_task_num = 4
    config.hidden_dim = 16
    config.ensemble_size = 2
    config.regime_context_source = "robust"
    return config


def _target_config() -> Config:
    config = deepcopy(_source_config())
    config.algo = "bapr_regime"
    config.bapr_v2_mode = "supervised"
    config.bapr_v2_latent_dim = 4
    config.bapr_v2_policy_context_source = "stored"
    config.bapr_v2_policy_mode = "residual"
    config.bapr_v2_training_schedule = "joint"
    config.bapr_v2_base_pretrain_iters = 1
    config.bapr_regime_inference_iters = 1
    config.bapr_regime_adaptation_source = "oracle"
    config.bapr_v3_variance_model = "mode_empirical"
    return config


def _tasks():
    return [{"mode_id": index} for index in range(4)]


def test_budget_is_aggregate_matched() -> None:
    adapter_steps = (
        len(protocol.MODES)
        * protocol.ADAPTER_EXTRA_ITERS_PER_MODE
        * protocol.SAMPLES_PER_ITER)
    robust_steps = protocol.ROBUST_EXTRA_ITERS * protocol.SAMPLES_PER_ITER
    assert adapter_steps == robust_steps == 2_800_000


def test_source_controller_copy_is_function_exact_at_zero_context() -> None:
    source = RegimeSAC(7, 3, _source_config(), seed=5)
    source.set_task_metadata(_tasks())
    target = BAPRRegime(7, 3, _target_config(), seed=11)
    target.set_task_metadata(_tasks())

    protocol.copy_source_controller_to_adapter(source, target)
    obs_key, act_key = jax.random.split(jax.random.PRNGKey(77))
    obs = jax.random.normal(obs_key, (13, 7))
    act = jnp.tanh(jax.random.normal(act_key, (13, 3)))
    source_context = jnp.zeros((13, source.context_dim))
    target_context = jnp.zeros((13, target.context_dim))

    np.testing.assert_allclose(
        np.asarray(source.policy.deterministic(obs, source_context)),
        np.asarray(target.policy.base_deterministic(obs)), atol=1e-7)
    np.testing.assert_allclose(
        np.asarray(source.critic(
            jnp.concatenate([obs, source_context], axis=-1), act)),
        np.asarray(target.critic(
            jnp.concatenate([obs, target_context], axis=-1), act)),
        atol=1e-7)
    np.testing.assert_allclose(
        np.asarray(source.target_critic(
            jnp.concatenate([obs, source_context], axis=-1), act)),
        np.asarray(target.target_critic(
            jnp.concatenate([obs, target_context], axis=-1), act)),
        atol=1e-7)


def test_copy_zeroes_residual_and_new_critic_context_coordinates() -> None:
    source = RegimeSAC(5, 2, _source_config(), seed=3)
    source.set_task_metadata(_tasks())
    target = BAPRRegime(5, 2, _target_config(), seed=9)
    target.set_task_metadata(_tasks())
    protocol.copy_source_controller_to_adapter(source, target)

    np.testing.assert_array_equal(
        np.asarray(target.policy.residual_mean.kernel.value), 0.0)
    np.testing.assert_array_equal(
        np.asarray(target.policy.residual_mean.bias.value), 0.0)
    first = np.asarray(target.critic.layers[0].kernel.value)
    context_slice = first[:, 5:5 + target.context_dim]
    np.testing.assert_array_equal(context_slice, 0.0)

