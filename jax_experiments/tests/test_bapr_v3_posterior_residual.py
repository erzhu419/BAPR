"""Protocol tests for posterior-conditioned residual control."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from jax_experiments.analysis import (
    analyze_bapr_v3_posterior_residual_screen as analysis,
)
from jax_experiments.analysis import (
    bapr_v3_learned_control_router as estimator,
)
from jax_experiments.analysis import bapr_v3_posterior_residual as protocol
from jax_experiments.analysis import bapr_v3_sequence_router as sequence
from jax_experiments.analysis import (
    bapr_v3_structured_channel_confirmation as confirmation,
)
from jax_experiments.analysis import bapr_v3_utility_aware_router as utility


def _table():
    rows = {
        "0": {"mean_returns": {"4": 10.0, "0": 20.0, "2": 0.0, "3": 0.0}},
        "1": {"mean_returns": {"4": 20.0, "0": 0.0, "2": 10.0, "3": 10.0}},
        "2": {"mean_returns": {"4": 10.0, "0": 0.0, "2": 20.0, "3": 0.0}},
        "3": {"mean_returns": {"4": 10.0, "0": 0.0, "2": 0.0, "3": 20.0}},
    }
    return {"nondominated_controllers": [4, 0, 2, 3], "rows": rows}


def load_submit_module():
    path = Path(__file__).resolve().parents[2] / "scripts" \
        / "submit_bapr_v3_posterior_residual_screen.py"
    spec = importlib.util.spec_from_file_location(
        "_test_submit_bapr_v3_posterior_residual_screen", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_uniform_posterior_falls_back_to_robust_without_threshold():
    selected, strength, diagnostics = protocol.posterior_residual_decision(
        np.full(4, 0.25), _table(), 1.0)
    assert selected == utility.ROBUST_CONTROLLER
    assert strength == 0.0
    assert diagnostics["best_expected_advantage"] <= 0.0


def test_one_hot_posterior_selects_specialist_with_bounded_strength():
    selected, strength, _ = protocol.posterior_residual_decision(
        np.asarray([1.0, 0.0, 0.0, 0.0]), _table(), 0.5)
    assert selected == 0
    assert strength == 0.5


def test_blend_is_exactly_robust_at_zero_and_clipped_at_cap():
    robust = np.asarray([-0.5, 0.5], dtype=np.float32)
    specialist = np.asarray([1.0, -1.0], dtype=np.float32)
    assert np.array_equal(
        protocol.blend_action(robust, specialist, 0.0), robust)
    assert np.allclose(
        protocol.blend_action(robust, specialist, 0.5),
        np.asarray([0.25, -0.25], dtype=np.float32))


def test_development_and_confirmation_seeds_are_disjoint():
    prior = {
        *estimator.TRAIN_EVENT_SEEDS,
        *estimator.VALIDATION_EVENT_SEEDS,
        *utility.VALIDATION_EVENT_SEEDS,
        *utility.HOLDOUT_EVENT_SEEDS,
        *sequence.TRAIN_EVENT_SEEDS,
        *sequence.VALIDATION_EVENT_SEEDS,
        *sequence.HOLDOUT_EVENT_SEEDS,
        *confirmation.EVENT_SEEDS,
    }
    assert set(protocol.DEVELOPMENT_EVENT_SEEDS) \
        == set(utility.VALIDATION_EVENT_SEEDS)
    assert not prior.intersection(protocol.SEALED_CONFIRMATION_EVENT_SEEDS)


def test_promotion_gate_rejects_mean_only_improvement():
    gate = analysis.variant_gate(
        [100.0, -1.0], [200.0, 200.0], [0.0, 0.0],
        0.0, 0.0, 0.5, True)
    assert not gate["full_cycle_improves_each_seed"]
    assert not gate["passed"]


def test_submission_is_scheduler_only_and_artifact_pinned():
    module = load_submit_module()
    signature, task, output = module.evaluation_spec(
        "cap025", 6100, "high")
    assert signature.endswith("/cap025/event-seed-6100")
    assert output == protocol.group_path("cap025", 6100)
    assert task["require_node"] == "jtl311linux"
    assert task["ckpt_glob"] == "router_manifest.json"
    assert "--resume" in task["cmd"]
    assert "slurm" not in task["cmd"].lower()
    assert "auto-adopt" not in task["cmd"].lower()
