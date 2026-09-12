"""Tests for the estimator-first actuator-polarity screen."""
from __future__ import annotations

import numpy as np

from jax_experiments.analysis import regime_polarity_posterior as protocol
from jax_experiments.analysis.run_regime_polarity_posterior_audit import (
    _arm_context,
)
from scripts import submit_regime_polarity_posterior as submitter


def test_controller_and_event_splits_are_disjoint():
    assert set(protocol.TRAIN_CONTROLLER_SEEDS).isdisjoint(
        protocol.VALIDATION_CONTROLLER_SEEDS)
    assert set(protocol.TRAIN_CONTROLLER_SEEDS).isdisjoint(
        protocol.TEST_CONTROLLER_SEEDS)
    assert set(protocol.VALIDATION_CONTROLLER_SEEDS).isdisjoint(
        protocol.TEST_CONTROLLER_SEEDS)
    assert set(protocol.TRAIN_EVENT_SEEDS).isdisjoint(
        protocol.VALIDATION_EVENT_SEEDS)
    assert set(protocol.TRAIN_EVENT_SEEDS).isdisjoint(
        protocol.TEST_EVENT_SEEDS)
    assert set(protocol.VALIDATION_EVENT_SEEDS).isdisjoint(
        protocol.TEST_EVENT_SEEDS)
    assert protocol.MODEL_CONFIG["variance_model"] == "mode_empirical"


def test_causal_filter_does_not_use_current_transition_for_action():
    config = protocol.FilterConfig(
        hazard_rate=0.004,
        evidence_scale=2.0,
        posterior_decay=1.0,
    )
    evidence = np.asarray([
        [5.0, 0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0, 0.0],
    ])
    before, after = protocol.causal_posteriors(evidence, config)
    np.testing.assert_allclose(before[0], np.full((4,), 0.25))
    np.testing.assert_allclose(before[1], after[0])
    assert int(np.argmax(before[1])) == 0
    assert after[1, 0] > before[1, 0]


def test_posterior_metrics_measure_action_time_switch_delay():
    labels = np.asarray([0] * 40 + [1] * 40, dtype=np.int32)
    posterior = np.eye(4)[labels].astype(np.float64)
    posterior[40:45] = np.eye(4)[0]
    metrics = protocol.posterior_metrics(
        posterior, labels, burnin=5, stability=3)
    assert metrics["switch_count"] == 1
    assert metrics["switch_delays"] == [5]
    assert metrics["median_switch_delay"] == 5.0
    assert metrics["mode_accuracy"] == 1.0


def test_soft_and_map_contexts_are_direct_posterior_inputs():
    posterior = np.asarray([0.1, 0.6, 0.2, 0.1], dtype=np.float64)
    np.testing.assert_allclose(
        _arm_context("learned_soft", 3, posterior), posterior)
    np.testing.assert_array_equal(
        _arm_context("learned_map", 3, posterior),
        np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        _arm_context("robust", 2, posterior), np.zeros((4,)))
    np.testing.assert_array_equal(
        _arm_context("oracle", 2, posterior),
        np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    )


def test_scheduler_graph_is_file_gated_and_excludes_311():
    training = submitter.candidates("training", "high")
    audits = submitter.candidates("audit", "high")
    analysis = submitter.candidates("analysis", "high")
    assert len(training) == 1
    assert len(audits) == 5
    assert len(analysis) == 1
    assert training[0][1]["vram"] == submitter.ESTIMATED_VRAM_MB
    assert len(training[0][1]["wait_for_files"]) == 24
    assert training[0][1]["resume_managed_by_cmd"] is True
    assert "auto-adopt" not in training[0][1]["cmd"]
    assert "jtl311linux" not in training[0][1]["allowed_nodes"]
    for _, spec, _ in audits:
        assert spec["vram"] == 0
        assert spec["cpu"] == 8
        assert len(spec["wait_for_files"]) == 10
        assert spec["allowed_nodes"] == submitter.CPU_NODES
        assert "JAX_PLATFORMS=cpu" in spec["cmd"]
    assert len(analysis[0][1]["wait_for_files"]) == 5

