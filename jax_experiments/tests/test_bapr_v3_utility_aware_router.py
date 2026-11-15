from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from flax import nnx

from jax_experiments.analysis import bapr_v3_learned_control_router as estimator
from jax_experiments.analysis import bapr_v3_utility_aware_router as router
from jax_experiments.networks.probabilistic_regime_context import (
    ProbabilisticRegimeContext,
)


def _table():
    means = {
        0: {4: 2270.0, 0: 3160.0, 1: 124.0, 2: 1672.0, 3: 858.0},
        1: {4: 2137.0, 0: 1200.0, 1: 1475.0, 2: 1962.0, 3: 1911.0},
        2: {4: 2643.0, 0: 1702.0, 1: 179.0, 2: 3203.0, 3: 897.0},
        3: {4: 2173.0, 0: 1316.0, 1: 1136.0, 2: 1654.0, 3: 2649.0},
    }
    return {
        "nondominated_controllers": [4, 0, 2, 3],
        "oracle_controller_map": [0, 4, 2, 3],
        "rows": {
            str(mode): {
                "mean_returns": {
                    str(controller): value
                    for controller, value in values.items()
                }
            }
            for mode, values in means.items()
        },
    }


def _config(min_history=8):
    return estimator.RouterConfig(
        hazard_rate=0.005,
        evidence_scale=0.25,
        confidence_threshold=0.8,
        margin_threshold=0.02,
        min_history=min_history,
        hysteresis_margin=0.02,
    )


def _load_submit_module():
    path = Path(__file__).resolve().parents[2] / "scripts" \
        / "submit_bapr_v3_utility_aware_router.py"
    spec = importlib.util.spec_from_file_location(
        "_test_submit_bapr_v3_utility_router", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_utility_router_selects_explicit_robust_for_mode_one():
    selected, diagnostics = router.select_utility_controller(
        np.asarray([0.01, 0.97, 0.01, 0.01]), 20, _table(), _config())
    assert selected == router.ROBUST_CONTROLLER
    assert diagnostics["fallback"] == 0.0
    assert diagnostics["deliberate_robust"] == 1.0


def test_utility_router_selects_specialist_two_for_mode_two():
    selected, diagnostics = router.select_utility_controller(
        np.asarray([0.01, 0.01, 0.97, 0.01]), 20, _table(), _config())
    assert selected == 2
    assert diagnostics["best_expected_advantage"] > 0.0


def test_utility_router_uses_expected_advantage_under_mode_ambiguity():
    selected, _ = router.select_utility_controller(
        np.asarray([0.0, 0.82, 0.18, 0.0]), 20, _table(),
        estimator.RouterConfig(
            hazard_rate=0.005, evidence_scale=0.25,
            confidence_threshold=0.5, margin_threshold=0.0,
            min_history=1, hysteresis_margin=0.0))
    assert selected == router.ROBUST_CONTROLLER


def test_low_confidence_is_distinct_from_deliberate_robust():
    selected, diagnostics = router.select_utility_controller(
        np.asarray([0.01, 0.97, 0.01, 0.01]), 7, _table(), _config())
    assert selected == router.FALLBACK_CONTROLLER
    assert diagnostics["fallback"] == 1.0
    assert diagnostics["deliberate_robust"] == 0.0


def test_decision_variant_changes_only_confidence_and_history():
    base = _config()
    selected = router.decision_config(base, "c70h4")
    assert selected.confidence_threshold == 0.70
    assert selected.min_history == 4
    assert selected.hazard_rate == base.hazard_rate
    assert selected.evidence_scale == base.evidence_scale
    assert selected.margin_threshold == base.margin_threshold
    assert selected.hysteresis_margin == base.hysteresis_margin

    faster = router.decision_config(base, "h010e100c80h4")
    assert faster.hazard_rate == 0.010
    assert faster.evidence_scale == 1.00
    assert faster.confidence_threshold == 0.80
    assert faster.min_history == 4
    assert faster.margin_threshold == base.margin_threshold

    bounded = router.decision_config(base, "d095c80h8")
    assert bounded.posterior_decay == 0.95
    assert bounded.hazard_rate == base.hazard_rate
    assert bounded.evidence_scale == base.evidence_scale
    assert bounded.confidence_threshold == base.confidence_threshold

    reset = router.decision_config(base, "cp050a25c80h8")
    assert reset.posterior_decay == 1.0
    assert reset.change_reset_threshold == 0.50
    assert reset.change_reset_alpha == 0.25
    assert reset.change_reset_mix == 1.0

    cusum = router.decision_config(base, "cs4d025c80h8")
    assert cusum.change_reset_threshold == 0.0
    assert cusum.change_cusum_threshold == 4.0
    assert cusum.change_cusum_drift == 0.25


def test_bounded_memory_forgets_old_mode_evidence():
    evidence = np.concatenate([
        np.tile(np.asarray([[0.0, -2.0]]), (200, 1)),
        np.tile(np.asarray([[-2.0, 0.0]]), (100, 1)),
    ])
    sticky = estimator.sticky_filter(
        evidence,
        estimator.RouterConfig(
            hazard_rate=1e-6, evidence_scale=0.25,
            confidence_threshold=0.8, margin_threshold=0.02,
            min_history=8, posterior_decay=1.0))
    bounded = estimator.sticky_filter(
        evidence,
        estimator.RouterConfig(
            hazard_rate=1e-6, evidence_scale=0.25,
            confidence_threshold=0.8, margin_threshold=0.02,
            min_history=8, posterior_decay=0.95))
    assert bounded[-1, 1] > 0.99
    assert np.argmax(bounded[220]) == 1
    assert bounded[220, 1] > sticky[220, 1]


def test_old_router_manifest_defaults_to_unbounded_memory():
    restored = estimator.RouterConfig.from_dict({
        "hazard_rate": 0.005,
        "evidence_scale": 0.25,
        "confidence_threshold": 0.8,
        "margin_threshold": 0.02,
        "min_history": 8,
        "hysteresis_margin": 0.02,
    })
    assert restored.posterior_decay == 1.0
    assert restored.change_reset_threshold == 0.0
    assert restored.change_cusum_threshold == 0.0


def test_change_point_prior_resets_only_contradictory_belief():
    model = ProbabilisticRegimeContext(
        3, 2, num_modes=4, hidden_dim=8, ensemble_size=2,
        change_reset_threshold=0.5, change_reset_alpha=1.0,
        change_reset_mix=1.0, rngs=nnx.Rngs(0))
    posterior = np.asarray([0.97, 0.01, 0.01, 0.01])

    stable_prior, stable_ema, stable_trigger = model._change_point_prior(
        posterior, np.asarray([0.0, -2.0, -2.0, -2.0]), 0.0)
    assert not bool(stable_trigger)
    assert int(np.argmax(stable_prior)) == 0
    assert float(stable_ema) == 0.0

    reset_prior, reset_ema, reset_trigger = model._change_point_prior(
        posterior, np.asarray([-2.0, 0.0, -2.0, -2.0]), 0.0)
    assert bool(reset_trigger)
    np.testing.assert_allclose(reset_prior, np.full((4,), 0.25))
    assert float(reset_ema) == 0.0


def test_cusum_supporting_evidence_cancels_noise_before_reset():
    model = ProbabilisticRegimeContext(
        3, 2, num_modes=4, hidden_dim=8, ensemble_size=2,
        change_cusum_threshold=2.0, change_cusum_drift=0.25,
        change_reset_mix=1.0, rngs=nnx.Rngs(0))
    posterior = np.asarray([0.97, 0.01, 0.01, 0.01])

    _, score, triggered = model._cusum_change_point_prior(
        posterior, np.asarray([-0.5, 0.0, -2.0, -2.0]), 0.0)
    assert not bool(triggered)
    assert np.isclose(float(score), 0.25)

    _, score, triggered = model._cusum_change_point_prior(
        posterior, np.asarray([0.0, -1.0, -2.0, -2.0]), score)
    assert not bool(triggered)
    assert float(score) == 0.0

    for expected_trigger in (False, False, True):
        prior, score, triggered = model._cusum_change_point_prior(
            posterior, np.asarray([-1.0, 0.0, -2.0, -2.0]), score)
        assert bool(triggered) is expected_trigger
    np.testing.assert_allclose(prior, np.full((4,), 0.25))
    assert float(score) == 0.0


def test_fallback_executes_the_correct_action_for_robust_mode():
    decisions = np.full((64,), router.FALLBACK_CONTROLLER, dtype=np.int32)
    true_modes = np.ones((64,), dtype=np.int32)
    metrics = router.routing_metrics(
        decisions, true_modes, (0, 4, 2, 3), burnin=0)
    assert metrics["coverage"] == 0.0
    assert metrics["action_accuracy"] == 1.0
    assert metrics["wrong_route_rate"] == 0.0


def test_specialist_one_is_dominated_by_robust_calibration_utility():
    table = _table()
    means = {
        controller: [
            table["rows"][str(mode)]["mean_returns"][str(controller)]
            for mode in range(4)
        ]
        for controller in router.ALL_CONTROLLERS
    }
    assert router._nondominated_controllers(means) == [4, 0, 2, 3]


def test_utility_splits_are_disjoint_and_submission_is_scheduler_only():
    prior = {
        *estimator.TRAIN_EVENT_SEEDS,
        *estimator.VALIDATION_EVENT_SEEDS,
        *estimator.HOLDOUT_EVENT_SEEDS,
    }
    assert not prior.intersection(router.VALIDATION_EVENT_SEEDS)
    assert not prior.intersection(router.HOLDOUT_EVENT_SEEDS)
    assert not set(router.VALIDATION_EVENT_SEEDS).intersection(
        router.HOLDOUT_EVENT_SEEDS)

    module = _load_submit_module()
    _, spec, output = module.audit_spec("validation", 6100, "high")
    assert output == router.audit_group_path("validation", 6100)
    assert spec["require_node"] == "jtl311linux"
    assert spec["ckpt_glob"] == "router_manifest.json"
    assert "--resume" in spec["cmd"]
    assert "slurm" not in spec["cmd"].lower()
    assert "auto-adopt" not in spec["cmd"].lower()

    signature, variant_spec, variant_output = module.audit_spec(
        "validation", 6100, "high", "c70h4")
    assert "/c70h4/event-seed-6100" in signature
    assert variant_output == router.audit_group_path(
        "validation", 6100, "c70h4")
    assert "--decision-variant c70h4" in variant_spec["cmd"]
