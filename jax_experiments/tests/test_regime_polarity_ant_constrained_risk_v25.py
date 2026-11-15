from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.algos.sac_switch_recovery_risk import (
    SACSwitchRecoveryRisk,
    risk_constraint,
)
from jax_experiments.analysis import (
    regime_polarity_ant_constrained_risk_v25 as protocol,
)
from jax_experiments.analysis import (
    train_regime_polarity_ant_constrained_risk_v25 as trainer,
)
from jax_experiments.tests import (
    test_regime_polarity_ant_switch_recovery_v24 as v24_test,
)
from scripts import submit_regime_polarity_ant_constrained_risk_v25 as submit


def _config(objective="relative", actor_start=0):
    return SimpleNamespace(
        hidden_dim=8,
        ensemble_size=2,
        alpha=0.2,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        auto_alpha=True,
        seed=7,
        switch_recovery_target_mode=2,
        switch_recovery_segment_steps=5,
        switch_recovery_termination_penalty=0.0,
        switch_recovery_risk_objective=objective,
        switch_recovery_risk_lambda=500.0,
        switch_recovery_risk_actor_start_update=actor_start,
    )


def test_risk_constraint_absolute_and_relative_semantics():
    candidate = jnp.asarray([0.1, 0.7])
    fallback = jnp.asarray([0.2, 0.4])
    assert jnp.allclose(
        risk_constraint(candidate, fallback, "absolute"), candidate)
    assert jnp.allclose(
        risk_constraint(candidate, fallback, "relative"),
        jnp.asarray([0.0, 0.3]),
    )


def test_collector_balances_candidate_and_robust_target_behavior():
    config = _config()
    agent = SACSwitchRecoveryRisk(3, 2, config, seed=config.seed)
    env = v24_test._FakeEnv()
    replay = v24_test._FakeReplay()
    _, rollout = trainer.collect_constrained_risk_samples(
        agent, env, replay, config, n_steps=20, current_iter=11)
    _, _, rewards, _, done, task_ids = replay.values
    assert rollout["candidate_behavior_segments"] == 1
    assert rollout["fallback_behavior_segments"] == 1
    assert rollout["candidate_steps"] == 10
    assert jnp.all(task_ids == 2)
    assert jnp.all(rewards.reshape(-1)[jnp.asarray([4, 9])] == 1.0)
    assert jnp.all(done.reshape(-1)[jnp.asarray([4, 9])] == 1.0)


def test_constrained_scan_compiles_and_updates_all_critics():
    agent = SACSwitchRecoveryRisk(3, 2, _config(), seed=7)
    before = [
        np.array(value, copy=True)
        for value in jax.tree.leaves(
            nnx.state(agent.risk_critic, nnx.Param))
    ]
    key = jax.random.PRNGKey(91)
    obs = jax.random.normal(key, (2, 4, 3))
    batch = {
        "obs": obs,
        "act": jnp.tanh(obs[..., :2]),
        "rew": jnp.ones((2, 4, 1)),
        "next_obs": obs + 0.1,
        "done": jnp.zeros((2, 4, 1)).at[:, -1, 0].set(1.0),
    }
    metrics = agent.multi_update(batch)
    after = nnx.state(agent.risk_critic, nnx.Param)
    assert agent.update_count == 2
    assert metrics["risk_actor_enabled"] == 1.0
    assert all(np.isfinite(float(value)) for value in metrics.values())
    after_leaves = jax.tree.leaves(after)
    assert any(
        not jnp.allclose(old, new)
        for old, new in zip(before, after_leaves)
    )


def test_custom_checkpoint_restores_fallback_and_risk_state():
    source = SACSwitchRecoveryRisk(3, 2, _config(), seed=11)
    restored = SACSwitchRecoveryRisk(3, 2, _config(), seed=29)
    restored.load_checkpoint_state(source.checkpoint_state())
    observations = jax.random.normal(jax.random.PRNGKey(31), (16, 3))
    actions = jax.random.uniform(
        jax.random.PRNGKey(32), (16, 2), minval=-1.0, maxval=1.0)
    assert jnp.allclose(
        source.fallback_policy.deterministic(observations),
        restored.fallback_policy.deterministic(observations),
    )
    assert jnp.allclose(
        source.risk_critic(observations, actions),
        restored.risk_critic(observations, actions),
    )


def test_protocol_and_scheduler_graph_are_frozen():
    assert protocol.VARIANTS == ("risk_q_absolute", "risk_q_relative")
    assert protocol.RISK_LAMBDA == 500.0
    assert protocol.RISK_WARMUP_UPDATES == 12_500
    assert protocol.RISK_ACTOR_START_UPDATE == 362_500
    assert protocol.TARGET_MODE_SAMPLES_PER_ITER == 4_000
    rows = submit.candidates("high")
    assert len(rows) == 31
    specs = [spec for _, spec, _ in rows]
    gpu = [spec for spec in specs if spec["vram"] > 0]
    cpu = [spec for spec in specs if spec["vram"] == 0]
    assert len(gpu) == 24
    assert len(cpu) == 7
    assert all("local" not in spec["allowed_nodes"] for spec in gpu)
    assert all(spec["vram"] == 3300 for spec in gpu)
    assert all(
        "results_regime_polarity_ant_switch_recovery_analysis_v24"
        in " ".join(spec["stage_input_paths"])
        for spec in specs
    )
    assert all(
        "results_regime_polarity_ant_matching_checkpoint_analysis_v23"
        in " ".join(spec["stage_input_paths"])
        for spec in specs
    )
