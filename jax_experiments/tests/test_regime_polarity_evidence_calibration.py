"""Tests for the v2 causal evidence-calibration screen."""
from __future__ import annotations

import numpy as np

from jax_experiments.analysis import regime_polarity_evidence_calibration as protocol
from jax_experiments.analysis.train_regime_polarity_evidence_calibration import (
    _fit_weighted_ridge,
)
from scripts import submit_regime_polarity_calibrated as submitter


def test_data_splits_remain_sealed():
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


def test_temporal_features_are_causal():
    evidence = np.asarray([
        [4.0, 1.0, 1.0, 1.0],
        [1.0, 4.0, 1.0, 1.0],
    ])
    features = protocol.temporal_features(evidence, ema_alpha=0.5)
    first = protocol.centered_evidence(evidence[0])
    second = protocol.centered_evidence(evidence[1])
    np.testing.assert_allclose(features[0, :4], first)
    np.testing.assert_allclose(features[0, 4:], 0.5 * first)
    np.testing.assert_allclose(features[1, :4], second)
    np.testing.assert_allclose(
        features[1, 4:], 0.25 * first + 0.5 * second)


def test_weighted_affine_calibration_corrects_permuted_heads():
    signatures = np.asarray([
        [0.0, 4.0, 1.0, 2.0],
        [2.0, 0.0, 4.0, 1.0],
        [1.0, 2.0, 0.0, 4.0],
        [4.0, 1.0, 2.0, 0.0],
    ])
    evidence = np.repeat(signatures, 64, axis=0)
    labels = np.repeat(np.arange(4, dtype=np.int32), 64)
    features = protocol.temporal_features(evidence, ema_alpha=0.0)
    fitted = _fit_weighted_ridge([features], [labels], ridge=1e-3)
    calibrator = protocol.CausalEvidenceCalibrator(
        protocol.CalibratorConfig(
            ema_alpha=0.0,
            ridge=1e-3,
            temperature=0.1,
        ),
        *fitted,
    )
    predictions = []
    for signature in signatures:
        state = calibrator.initial_state()
        state = calibrator.update_from_evidence(state, signature)
        predictions.append(
            int(np.argmax(calibrator.probabilities(state))))
    np.testing.assert_array_equal(predictions, np.arange(4))


def test_action_posterior_has_exact_one_transition_causality():
    feature_mean = np.zeros((8,))
    feature_scale = np.ones((8,))
    weights = np.zeros((8, 4))
    weights[4:, :] = 5.0 * np.eye(4)
    calibrator = protocol.CausalEvidenceCalibrator(
        protocol.CalibratorConfig(
            ema_alpha=0.0,
            ridge=0.1,
            temperature=1.0,
        ),
        feature_mean,
        feature_scale,
        weights,
        np.zeros((4,)),
    )
    evidence = np.asarray([
        [5.0, 0.0, 0.0, 0.0],
        [0.0, 5.0, 0.0, 0.0],
    ])
    action_time = calibrator.action_posteriors(evidence)
    np.testing.assert_allclose(action_time[0], np.full((4,), 0.25))
    assert int(np.argmax(action_time[1])) == 0


def test_scheduler_graph_is_file_gated_and_excludes_311():
    training = submitter.candidates("training", "high")
    audits = submitter.candidates("audit", "high")
    analysis = submitter.candidates("analysis", "high")
    assert len(training) == 1
    assert len(audits) == 5
    assert len(analysis) == 1
    assert training[0][1]["vram"] == submitter.ESTIMATED_VRAM_MB
    assert len(training[0][1]["wait_for_files"]) == 26
    assert "auto-adopt" not in training[0][1]["cmd"]
    assert "jtl311linux" not in training[0][1]["allowed_nodes"]
    for _, spec, _ in audits:
        assert spec["vram"] == 0
        assert spec["cpu"] == 8
        assert len(spec["wait_for_files"]) == 12
        assert spec["allowed_nodes"] == submitter.CPU_NODES
        assert "JAX_PLATFORMS=cpu" in spec["cmd"]
    assert len(analysis[0][1]["wait_for_files"]) == 5
