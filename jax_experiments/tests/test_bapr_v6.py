import jax.numpy as jnp
import numpy as np

from jax_experiments.algos.bapr_v6 import BAPRv6
from jax_experiments.tests.test_bapr_v5 import make_config


def _batch(updates=2, batch_size=8):
    task_ids = jnp.tile(
        jnp.arange(batch_size, dtype=jnp.int32) % 4, (updates, 1))
    return {
        "obs": jnp.zeros((updates, batch_size, 3), dtype=jnp.float32),
        "act": jnp.zeros((updates, batch_size, 2), dtype=jnp.float32),
        "rew": jnp.ones((updates, batch_size, 1), dtype=jnp.float32),
        "next_obs": jnp.full(
            (updates, batch_size, 3), 0.1, dtype=jnp.float32),
        "done": jnp.zeros(
            (updates, batch_size, 1), dtype=jnp.float32),
        "task_id": task_ids,
        "belief": jnp.zeros(
            (updates, batch_size, 5), dtype=jnp.float32),
        "next_belief": jnp.zeros(
            (updates, batch_size, 5), dtype=jnp.float32),
    }


def test_bapr_v6_balances_controller_examples_and_alpha_state():
    config = make_config()
    config.batch_size = 8
    agent = BAPRv6(3, 2, config, seed=12)
    agent.set_task_metadata([{"mode_id": mode} for mode in range(4)])
    balanced = agent._balanced_context_batch(_batch())

    assert balanced["obs"].shape == (2, 10, 3)
    contexts = np.asarray(balanced["belief"][0])
    np.testing.assert_array_equal(contexts[:2], np.zeros((2, 5)))
    for mode in range(4):
        assert int(np.sum(contexts[:, mode] * contexts[:, -1])) == 2
    assert agent.log_alpha.shape == (5,)


def test_bapr_v6_compiled_update_is_finite():
    config = make_config()
    config.batch_size = 8
    agent = BAPRv6(3, 2, config, seed=13)
    agent.set_task_metadata([{"mode_id": mode} for mode in range(4)])
    metrics = agent.multi_update(_batch(), current_iter=1)

    assert np.isfinite(metrics["critic_loss"])
    assert np.isfinite(metrics["policy_loss"])
    assert np.isfinite(metrics["alpha"])
    assert metrics["v6_balanced_per_head_replay"] == 1.0
    assert metrics["v6_per_context_alpha"] == 1.0
    assert metrics["v6_effective_batch_multiplier"] == 1.25
