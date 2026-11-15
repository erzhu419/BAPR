import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.algos.bapr_v4 import BAPRv4
from jax_experiments.configs.default import Config
from jax_experiments.networks.persistent_option_context import (
    PersistentOptionRegimeContext,
)
from jax_experiments.networks.persistent_option_policy import (
    PersistentOptionGaussianPolicy,
)


def make_context(hold_steps=4):
    return PersistentOptionRegimeContext(
        obs_dim=3, act_dim=2, num_modes=4, hidden_dim=8,
        ensemble_size=2, mode="supervised", likelihood="point",
        min_history=2, option_hold_steps=hold_steps,
        option_confidence_threshold=0.8,
        option_margin_threshold=0.1,
        option_hysteresis_margin=0.02,
        rngs=nnx.Rngs(1),
    )


def test_persistent_option_commits_then_waits_for_boundary():
    model = make_context(hold_steps=4)
    posterior0 = jnp.asarray([0.92, 0.03, 0.03, 0.02])
    option, age = model._advance_option(
        posterior0, jnp.asarray(2), jnp.asarray(-1), jnp.asarray(0))
    assert int(option) == 0
    assert int(age) == 0

    posterior2 = jnp.asarray([0.02, 0.03, 0.93, 0.02])
    for expected_age in (1, 2, 3):
        option, age = model._advance_option(
            posterior2, jnp.asarray(10), option, age)
        assert int(option) == 0
        assert int(age) == expected_age
    option, age = model._advance_option(
        posterior2, jnp.asarray(10), option, age)
    assert int(option) == 2
    assert int(age) == 0


def test_persistent_option_uses_robust_fallback_at_boundary():
    model = make_context(hold_steps=2)
    uncertain = jnp.asarray([0.30, 0.25, 0.25, 0.20])
    option, age = model._advance_option(
        uncertain, jnp.asarray(20), jnp.asarray(2), jnp.asarray(1))
    assert int(option) == -1
    assert int(age) == 0
    context = np.asarray(model.policy_context(
        (uncertain, jnp.asarray(0.0), jnp.asarray(0.0),
         jnp.asarray(0.0), jnp.asarray(20), option, age),
        jnp.asarray([0.0, 0.0, 1.0, 0.0])))
    np.testing.assert_array_equal(context, np.zeros(5, dtype=np.float32))


def test_shared_option_policy_starts_control_equivalent():
    policy = PersistentOptionGaussianPolicy(
        3, 2, 16, 4, rngs=nnx.Rngs(4))
    obs = jnp.asarray([[0.2, -0.1, 0.3]], dtype=jnp.float32)
    robust = np.asarray(policy.deterministic(obs, None))
    for mode in range(4):
        context = jnp.concatenate([
            jax_one_hot(mode, 4), jnp.ones((1,), dtype=jnp.float32)
        ])[None, :]
        np.testing.assert_allclose(
            np.asarray(policy.deterministic(obs, context)), robust,
            atol=1e-7)


def jax_one_hot(index, size):
    return jnp.eye(size, dtype=jnp.float32)[index]


def test_bapr_v4_training_source_mix_is_deterministic():
    config = Config()
    config.task_num = 4
    config.test_task_num = 4
    config.bapr_v2_latent_dim = 4
    config.hidden_dim = 8
    config.ensemble_size = 2
    config.bapr_v2_context_hidden_dim = 8
    config.bapr_v3_context_ensemble_size = 2
    config.bapr_v3_likelihood = "point"
    config.bapr_v4_training_source_period = 4
    config.bapr_v4_training_robust_slots = 1
    agent = BAPRv4(3, 2, config, seed=5)
    assert [agent.rollout_context_source(iteration)
            for iteration in range(8)] == [
                agent.CONTEXT_ROBUST,
                agent.CONTEXT_ORACLE,
                agent.CONTEXT_ORACLE,
                agent.CONTEXT_ORACLE,
                agent.CONTEXT_ROBUST,
                agent.CONTEXT_ORACLE,
                agent.CONTEXT_ORACLE,
                agent.CONTEXT_ORACLE,
            ]
    assert len(agent.adaptation_state) == 7
