import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.algos.bapr_v5 import BAPRv5
from jax_experiments.configs.default import Config
from jax_experiments.networks.hard_option_critic import (
    HardOptionEnsembleCritic,
)
from jax_experiments.networks.hard_option_policy import HardOptionGaussianPolicy


def make_config():
    config = Config()
    config.task_num = 4
    config.test_task_num = 4
    config.bapr_v2_latent_dim = 4
    config.hidden_dim = 8
    config.ensemble_size = 2
    config.batch_size = 4
    config.bapr_v2_context_hidden_dim = 8
    config.bapr_v3_context_ensemble_size = 2
    config.bapr_v3_likelihood = "point"
    config.bapr_v2_base_aux_weight = 0.0
    config.bapr_v4_training_source_period = 5
    config.bapr_v4_training_robust_slots = 1
    return config


def one_hot_context(mode, batch=1):
    option = jax.nn.one_hot(
        jnp.full((batch,), mode), 4, dtype=jnp.float32)
    return jnp.concatenate(
        [option, jnp.ones((batch, 1), dtype=jnp.float32)], axis=-1)


def test_hard_option_policy_starts_equal_without_sharing_outputs():
    policy = HardOptionGaussianPolicy(
        3, 2, 8, 4, rngs=nnx.Rngs(2))
    obs = jnp.asarray([[0.2, -0.1, 0.3]], dtype=jnp.float32)
    robust = np.asarray(policy.deterministic(obs, None))
    for mode in range(4):
        np.testing.assert_allclose(
            np.asarray(policy.deterministic(
                obs, one_hot_context(mode))), robust, atol=1e-7)

    policy.option_mean.bias.value = (
        policy.option_mean.bias.value.at[2, 0, 0].set(1.0))
    np.testing.assert_allclose(
        np.asarray(policy.deterministic(obs, None)), robust, atol=1e-7)
    np.testing.assert_allclose(
        np.asarray(policy.deterministic(
            obs, one_hot_context(1))), robust, atol=1e-7)
    assert not np.allclose(
        np.asarray(policy.deterministic(obs, one_hot_context(2))), robust)


def test_hard_option_critic_starts_control_equivalent():
    critic = HardOptionEnsembleCritic(
        3, 2, 5, 8, 4, 2, n_layers=2, rngs=nnx.Rngs(3))
    obs = jnp.asarray([[0.1, 0.2, -0.3]], dtype=jnp.float32)
    act = jnp.asarray([[0.4, -0.2]], dtype=jnp.float32)
    robust_aug = jnp.concatenate([obs, jnp.zeros((1, 5))], axis=-1)
    robust = np.asarray(critic(robust_aug, act))
    for mode in range(4):
        option_aug = jnp.concatenate(
            [obs, one_hot_context(mode)], axis=-1)
        np.testing.assert_allclose(
            np.asarray(critic(option_aug, act)), robust, atol=1e-7)


def test_bapr_v5_dual_context_relabel_and_update():
    agent = BAPRv5(3, 2, make_config(), seed=4)
    agent.set_task_metadata([{"mode_id": mode} for mode in range(4)])
    task_ids = jnp.asarray([[0, 1, 2, 3]], dtype=jnp.int32)
    batch = {
        "obs": jnp.zeros((1, 4, 3), dtype=jnp.float32),
        "act": jnp.zeros((1, 4, 2), dtype=jnp.float32),
        "rew": jnp.ones((1, 4, 1), dtype=jnp.float32),
        "next_obs": jnp.full((1, 4, 3), 0.1, dtype=jnp.float32),
        "done": jnp.zeros((1, 4, 1), dtype=jnp.float32),
        "task_id": task_ids,
        "belief": jnp.zeros((1, 4, 5), dtype=jnp.float32),
        "next_belief": jnp.zeros((1, 4, 5), dtype=jnp.float32),
    }
    relabelled = agent._dual_context_batch(batch)
    assert relabelled["obs"].shape == (1, 8, 3)
    np.testing.assert_array_equal(
        np.asarray(relabelled["belief"][0, :4]), np.zeros((4, 5)))
    np.testing.assert_array_equal(
        np.asarray(relabelled["belief"][0, 4:, :4]), np.eye(4))
    np.testing.assert_array_equal(
        np.asarray(relabelled["belief"][0, 4:, 4]), np.ones(4))

    metrics = agent.multi_update(batch, current_iter=1)
    assert np.isfinite(metrics["critic_loss"])
    assert np.isfinite(metrics["policy_loss"])
    assert metrics["v5_dual_context_relabel"] == 1.0
