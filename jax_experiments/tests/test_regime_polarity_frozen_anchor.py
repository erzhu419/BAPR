"""Tests for the frozen-anchor v2 controller and scheduler graph."""
from __future__ import annotations

from copy import deepcopy
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jax_experiments.algos.regime_sac import RegimeSAC
from jax_experiments.analysis import (
    regime_polarity_frozen_anchor as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_frozen_anchor_branch as branch,
)
from jax_experiments.configs.default import Config
from jax_experiments.networks.residual_policy import (
    conservative_q_advantage,
    modewise_update_acceptance,
    modewise_update_diagnostics,
)
from jax_experiments.train import make_algo


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import submit_regime_polarity_frozen_anchor as submit


def _source_config():
    config = Config()
    config.algo = "regime_sac"
    config.env_type = "stochastic_mode"
    config.task_num = 4
    config.test_task_num = 4
    config.regime_context_source = "robust"
    config.hidden_dim = 16
    config.ensemble_size = 2
    return config


def _variant_config(role: str):
    source = _source_config()
    config = branch._adaptive_config(
        source, role, protocol.branch_run_dir(role, 1103))
    config.hidden_dim = 16
    config.ensemble_size = 2
    config.bapr_v2_context_hidden_dim = 8
    config.bapr_v3_context_ensemble_size = 2
    config.batch_size = 4
    return config


def _batch():
    return {
        "obs": jnp.zeros((1, 4, 5), dtype=jnp.float32),
        "act": jnp.zeros((1, 4, 2), dtype=jnp.float32),
        "rew": jnp.ones((1, 4, 1), dtype=jnp.float32),
        "next_obs": jnp.full((1, 4, 5), 0.1, dtype=jnp.float32),
        "done": jnp.zeros((1, 4, 1), dtype=jnp.float32),
        "task_id": jnp.arange(4, dtype=jnp.int32)[None],
        "belief": jnp.zeros((1, 4, 5), dtype=jnp.float32),
        "next_belief": jnp.zeros((1, 4, 5), dtype=jnp.float32),
    }


def test_protocol_reuses_anchor_but_adds_equal_budget():
    assert protocol.SOURCE_TOTAL_STEPS == 8_400_000
    assert protocol.BRANCH_TOTAL_STEPS == 11_200_000
    assert protocol.BRANCH_UPDATE_COUNT == 700_000
    assert protocol.BRANCH_ROLES == (
        "robust_long",
        "shared_small",
        "shared_wide",
        "mode_residual",
    )


def test_all_variants_start_at_the_same_robust_function():
    source = RegimeSAC(5, 2, _source_config(), seed=17)
    tasks = [{"mode_id": mode} for mode in protocol.MODES]
    source.set_task_metadata(tasks)
    for role in protocol.VARIANTS:
        config = _variant_config(role)
        target = make_algo(config.algo, 5, 2, config)
        target.set_task_metadata(tasks)
        branch._copy_source_controller(source, target, role)
        equivalence = branch._equivalence(source, target)
        assert equivalence["pass"], (role, equivalence)

        obs = jax.random.normal(jax.random.PRNGKey(31), (16, 5))
        robust = np.asarray(target.policy.base_deterministic(obs))
        for mode in protocol.MODES:
            context = jnp.broadcast_to(
                jnp.concatenate([
                    jax.nn.one_hot(mode, 4),
                    jnp.ones((1,), dtype=jnp.float32),
                ])[None],
                (len(obs), 5),
            )
            np.testing.assert_array_equal(
                np.asarray(target.policy.deterministic(obs, context)),
                robust,
            )


def test_adaptive_update_cannot_change_base_policy():
    source = RegimeSAC(5, 2, _source_config(), seed=41)
    tasks = [{"mode_id": mode} for mode in protocol.MODES]
    source.set_task_metadata(tasks)
    for role in protocol.VARIANTS:
        config = _variant_config(role)
        target = make_algo(config.algo, 5, 2, config)
        target.set_task_metadata(tasks)
        branch._copy_source_controller(source, target, role)
        before = branch._component_hashes(target, role)
        metrics = target.multi_update(_batch(), current_iter=2100)
        after = branch._component_hashes(target, role)
        assert before["base_policy"] == after["base_policy"]
        assert before["adaptive_policy"] != after["adaptive_policy"]
        assert metrics["v2_train_base"] == 0.0
        assert target.rollout_context_source(2100) == target.CONTEXT_ORACLE


def test_modewise_update_filter_rejects_any_represented_mode_regression():
    context = jnp.asarray([
        [1, 0, 0, 0, 1],
        [0, 1, 0, 0, 1],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0],
    ], dtype=jnp.float32)
    current = jnp.asarray([0.01, 0.02, 0.03, 0.04, -10.0])
    candidate = jnp.asarray([0.02, 0.03, 0.01, 0.05, -20.0])
    accepted, before, after, represented = modewise_update_acceptance(
        current,
        candidate,
        context,
        tolerance=0.005,
        floor=-0.01,
    )
    assert not bool(accepted)
    np.testing.assert_array_equal(np.asarray(represented), np.ones(4, bool))
    assert float(after[2]) < float(before[2]) - 0.005

    accepted, _, _, _ = modewise_update_acceptance(
        current,
        candidate.at[2].set(0.028),
        context,
        tolerance=0.005,
        floor=-0.01,
    )
    assert bool(accepted)


def test_conservative_advantage_has_finite_zero_residual_gradient():
    q_base = jnp.asarray([
        [1.0, -2.0, 3.0],
        [1.0, -2.0, 3.0],
        [1.0, -2.0, 3.0],
    ])

    def objective(q_adaptive):
        return conservative_q_advantage(q_adaptive, q_base).sum()

    value, gradient = jax.value_and_grad(objective)(q_base)
    assert float(value) == 0.0
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_modewise_update_diagnostics_separate_rejection_causes():
    context = jnp.eye(4, dtype=jnp.float32)
    context = jnp.concatenate(
        [context, jnp.ones((4, 1), dtype=jnp.float32)], axis=-1)
    current = jnp.asarray([0.0, 0.0, 0.0, 0.0])
    candidate = jnp.asarray([0.01, -0.02, jnp.nan, 0.03])
    diagnostics = modewise_update_diagnostics(
        current,
        candidate,
        context,
        tolerance=0.005,
        floor=-0.01,
    )
    (
        accepted,
        _,
        _,
        represented,
        candidate_min,
        candidate_mean,
        regression_margin_min,
        floor_margin_min,
        reject_nonfinite,
        reject_regression,
        reject_floor,
    ) = diagnostics
    assert not bool(accepted)
    np.testing.assert_array_equal(np.asarray(represented), np.ones(4, bool))
    assert np.isclose(float(candidate_min), -0.02)
    assert np.isclose(float(candidate_mean), (0.01 - 0.02 + 0.03) / 3)
    assert np.isclose(float(regression_margin_min), -0.015)
    assert np.isclose(float(floor_margin_min), -0.01)
    assert bool(reject_nonfinite)
    assert bool(reject_regression)
    assert bool(reject_floor)


def test_conservative_filter_rolls_back_residual_actor_update():
    source = RegimeSAC(5, 2, _source_config(), seed=41)
    tasks = [{"mode_id": mode} for mode in protocol.MODES]
    source.set_task_metadata(tasks)
    config = _variant_config("shared_small")
    config.bapr_v2_train_advantage_constraint = True
    config.bapr_v2_train_update_filter = True
    config.bapr_v2_train_update_floor = 1e6
    target = make_algo(config.algo, 5, 2, config)
    target.set_task_metadata(tasks)
    branch._copy_source_controller(source, target, "shared_small")
    before = branch._component_hashes(target, "shared_small")
    metrics = target.multi_update(_batch(), current_iter=2100)
    after = branch._component_hashes(target, "shared_small")
    assert before["base_policy"] == after["base_policy"]
    assert before["adaptive_policy"] == after["adaptive_policy"]
    assert metrics["v2_train_update_accept_rate"] == 0.0
    assert metrics["v2_train_candidate_nonfinite_rate"] == 0.0
    assert metrics["v2_train_update_reject_floor_rate"] == 1.0


def test_variant_view_maps_generic_audit_roles():
    view = protocol.variant_view("mode_residual")
    assert view.BRANCH_ROLES == ("robust_continue", "anchored")
    assert view.posterior_metrics is protocol.posterior_metrics
    assert (
        view.branch_bundle_dir("robust_continue", 1103)
        == protocol.branch_bundle_dir("robust_long", 1103)
    )
    assert (
        view.branch_bundle_dir("anchored", 1103)
        == protocol.branch_bundle_dir("mode_residual", 1103)
    )
    assert view.calibration_dir(1103) == protocol.calibration_dir(
        "mode_residual", 1103)


def test_scheduler_matrix_is_measured_and_file_gated():
    rows = submit.candidates("all", "high")
    assert len(rows) == 33
    specs = [spec for _, spec, _ in rows]
    json.dumps(specs)
    gpu = [spec for spec in specs if spec["vram"] > 0]
    cpu = [spec for spec in specs if spec["vram"] == 0]
    assert len(gpu) == 12
    assert len(cpu) == 21
    assert all("jtl311linux" not in spec["allowed_nodes"] for spec in specs)
    assert all(
        set(spec["allowed_nodes"]) == set(submit.GPU_NODES)
        for spec in gpu)
    assert all(
        set(spec["allowed_nodes"]) == set(submit.CPU_NODES)
        for spec in cpu)
    assert {
        spec["vram"] for spec in gpu
    } == {1800, 2800, 4200}
    assert all(
        len(spec["wait_for_files"]) == 5 for spec in gpu)

    calibrations = [
        spec for spec in cpu if "/calibration/" in spec["signature"]
    ]
    audits = [
        spec for spec in cpu if "/audit/" in spec["signature"]
    ]
    analyses = [
        spec for spec in cpu if spec["signature"].endswith("/analysis")
    ]
    assert len(calibrations) == 9
    assert len(audits) == 9
    assert len(analyses) == 3
    assert all(len(spec["wait_for_files"]) == 10
               for spec in calibrations)
    assert all(len(spec["wait_for_files"]) == 13 for spec in audits)
    assert all(len(spec["wait_for_files"]) == 3 for spec in analyses)
