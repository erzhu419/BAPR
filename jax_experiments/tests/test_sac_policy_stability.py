from types import SimpleNamespace
import math

import jax
import jax.numpy as jnp
from flax import nnx

from jax_experiments.algos.sac_base import SACBase
from jax_experiments.algos.sac_policy_stability import SACPolicyStability


def _config(period=1, select=True):
    return SimpleNamespace(
        hidden_dim=16,
        ensemble_size=2,
        alpha=0.2,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        auto_alpha=True,
        sac_actor_update_period=period,
        sac_select_best_eval=select,
    )


def _batch(updates=3, batch_size=8):
    return {
        "obs": jnp.ones((updates, batch_size, 3)),
        "act": jnp.zeros((updates, batch_size, 2)),
        "rew": jnp.ones((updates, batch_size, 1)),
        "next_obs": jnp.full((updates, batch_size, 3), 0.5),
        "done": jnp.zeros((updates, batch_size, 1)),
    }


def _leaves(state):
    return jax.tree.leaves(nnx.state(state, nnx.Param))


def test_period_one_matches_sac_base_update_exactly():
    config = _config(period=1, select=False)
    reference = SACBase(3, 2, config, seed=5)
    candidate = SACPolicyStability(3, 2, config, seed=5)
    batch = _batch(updates=3)

    reference.multi_update(batch)
    metrics = candidate.multi_update(batch)

    for reference_module, candidate_module in (
        (reference.policy, candidate.policy),
        (reference.critic, candidate.critic),
        (reference.target_critic, candidate.target_critic),
    ):
        assert all(
            jnp.array_equal(left, right)
            for left, right in zip(
                _leaves(reference_module), _leaves(candidate_module)
            )
        )
    assert jnp.array_equal(reference.log_alpha, candidate.log_alpha)
    assert reference.update_count == candidate.update_count == 3
    assert metrics["actor_update_fraction"] == 1.0


def test_actor_period_thins_policy_updates_but_not_total_updates():
    agent = SACPolicyStability(3, 2, _config(period=2), seed=7)
    before = _leaves(agent.policy)
    metrics = agent.multi_update(_batch(updates=3))
    after = _leaves(agent.policy)

    assert agent.update_count == 3
    assert math.isclose(
        metrics["actor_update_fraction"], 2.0 / 3.0, rel_tol=1e-6)
    assert any(not jnp.array_equal(a, b) for a, b in zip(before, after))


def test_best_validation_policy_survives_later_updates_and_checkpoint_state():
    config = _config(period=1, select=True)
    agent = SACPolicyStability(3, 2, config, seed=11)
    agent.report_eval(10.0)
    selected = agent.selected_policy_state()
    agent.multi_update(_batch(updates=2))
    agent.report_eval(9.0)

    selected_after = agent.selected_policy_state()
    assert all(
        jnp.array_equal(left, right)
        for left, right in zip(
            jax.tree.leaves(selected), jax.tree.leaves(selected_after)
        )
    )
    assert agent.selection_record()["selected_update_count"] == 0

    restored = SACPolicyStability(3, 2, config, seed=13)
    restored.update_count = agent.update_count
    restored.load_checkpoint_state(agent.checkpoint_state())
    assert restored.selection_record() == agent.selection_record()
    assert all(
        jnp.array_equal(left, right)
        for left, right in zip(
            jax.tree.leaves(agent.selected_policy_state()),
            jax.tree.leaves(restored.selected_policy_state()),
        )
    )
