"""Tests for the Ant V26 joint mode-conditioned controller protocol."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from flax import nnx

from jax_experiments.algos.joint_mode_risk_sac import JointModeRiskSAC
from jax_experiments.algos.sac_base import SACBase
from jax_experiments.analysis import (
    regime_polarity_ant_joint_mode_risk_v26 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_ant_joint_mode_risk_v26 as runner,
)
from jax_experiments.analysis import (
    train_regime_polarity_ant_joint_mode_risk_v26 as trainer,
)


def _config(**overrides):
    values = {
        "env_type": "stochastic_mode",
        "regime_context_source": "oracle",
        "task_num": 4,
        "hidden_dim": 16,
        "ensemble_size": 2,
        "alpha": 0.2,
        "lr": 3e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "auto_alpha": True,
        "joint_mode_risk_lambda": 500.0,
        "joint_mode_risk_actor_start_update": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _submit_module():
    path = (
        protocol.ROOT / "scripts"
        / "submit_regime_polarity_ant_joint_mode_risk_v26.py"
    )
    spec = importlib.util.spec_from_file_location("_test_submit_v26", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_v26_budget_and_split_integrity():
    protocol.assert_protocol_integrity()
    assert protocol.FINETUNE_ITERS["joint_data_matched"] == 4 * (
        protocol.FINETUNE_ITERS["joint_equal_budget"])
    assert protocol.FINAL_TOTAL_STEPS["joint_equal_budget"] == 11_200_000
    assert protocol.FINAL_TOTAL_STEPS["joint_data_matched"] == 28_000_000
    assert protocol.FINAL_UPDATE_COUNT["joint_equal_budget"] == 525_000
    assert protocol.FINAL_UPDATE_COUNT["joint_data_matched"] == 1_050_000


def test_collection_plan_balances_each_mode_and_behavior():
    plan = trainer.collection_plan(85_003, 1_400, 16, 4)
    assert len(plan) == 16
    for mode in protocol.MODES:
        rows = [candidate for target, candidate in plan if target == mode]
        assert rows.count(True) == 2
        assert rows.count(False) == 2


def test_robust_warmstart_is_functionally_equivalent_for_every_context():
    source = SACBase(7, 3, _config(), seed=3)
    target = JointModeRiskSAC(7, 3, _config(), seed=5)
    runner.warmstart_from_source(source, target)
    result = runner._controller_equivalence(source, target)
    assert result["pass"] is True
    assert (
        max(result["max_abs_errors"].values())
        <= protocol.CONTROLLER_EQUIVALENCE_ATOL
    )


def test_joint_risk_update_is_finite_and_context_sensitive():
    agent = JointModeRiskSAC(7, 3, _config(), seed=7)
    agent.set_task_metadata([{"mode_id": mode} for mode in protocol.MODES])
    update_steps = 2
    batch_size = 8
    key = jax.random.PRNGKey(19)
    obs = jax.random.normal(key, (update_steps, batch_size, 7))
    act = jnp.tanh(jax.random.normal(
        jax.random.PRNGKey(20), (update_steps, batch_size, 3)))
    mode_ids = jnp.arange(batch_size) % len(protocol.MODES)
    context = jax.nn.one_hot(mode_ids, len(protocol.MODES))
    context = jnp.broadcast_to(
        context[None, :, :], (update_steps, batch_size, len(protocol.MODES)))
    metrics = agent.multi_update({
        "obs": obs,
        "act": act,
        "rew": jax.random.normal(
            jax.random.PRNGKey(21), (update_steps, batch_size, 1)),
        "next_obs": obs + 0.01,
        "done": jnp.zeros((update_steps, batch_size, 1)),
        "belief": context,
        "next_belief": context,
    })
    assert agent.update_count == update_steps
    assert all(math.isfinite(float(value)) for value in metrics.values())


def test_custom_checkpoint_restores_fallback_and_risk_state():
    source = JointModeRiskSAC(7, 3, _config(), seed=11)
    restored = JointModeRiskSAC(7, 3, _config(), seed=13)
    state = source.checkpoint_state()
    restored.load_checkpoint_state(state)
    obs = jax.random.normal(jax.random.PRNGKey(22), (16, 7))
    action = jnp.zeros((16, 3), dtype=jnp.float32)
    context = jax.nn.one_hot(
        jnp.arange(16) % len(protocol.MODES), len(protocol.MODES))
    critic_obs = jnp.concatenate([obs, context], axis=-1)
    assert jnp.array_equal(
        source.fallback_policy.deterministic(obs, context),
        restored.fallback_policy.deterministic(obs, context),
    )
    assert jnp.array_equal(
        source.risk_critic(critic_obs, action),
        restored.risk_critic(critic_obs, action),
    )


def test_scheduler_dag_is_batched_remote_and_dependency_gated():
    submit = _submit_module()
    rows = submit.candidates("high")
    assert len(rows) == 13
    specs = [spec for _, spec, _ in rows]
    train = [spec for spec in specs if "/train/" in spec["signature"]]
    audits = [spec for spec in specs if "/audit/" in spec["signature"]]
    aggregate = [spec for spec in specs if spec["signature"].endswith("/analysis")]
    assert len(train) == 6
    assert len(audits) == 6
    assert len(aggregate) == 1
    assert all("local" not in spec["allowed_nodes"] for spec in train)
    assert all(spec["ckpt_dir"].endswith("/checkpoints") for spec in train)
    assert all(spec["wait_for_files"] for spec in audits + aggregate)
    assert all(spec["resume_managed_by_cmd"] is True for spec in train)
