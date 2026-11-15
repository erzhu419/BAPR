"""Tests for the no-realized-action supervision ablation."""
from __future__ import annotations

import inspect

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_expected_action_system_id as protocol,
)
from jax_experiments.analysis import (
    train_regime_polarity_expected_action_system_id as trainer,
)
from scripts import submit_regime_polarity_expected_action_system_id as submitter


def test_expected_target_uses_only_mode_transform_and_command():
    obs = np.zeros((4, 17), dtype=np.float32)
    act = np.asarray([
        [0.8, -0.7, 0.6, -0.5, 0.4, -0.3],
        [0.7, -0.6, 0.5, -0.4, 0.3, -0.2],
        [0.6, -0.5, 0.4, -0.3, 0.2, -0.1],
        [0.5, -0.4, 0.3, -0.2, 0.1, 0.0],
    ], dtype=np.float32)
    transitions = (
        obs,
        act,
        np.zeros((4,), dtype=np.float32),
        obs,
        np.zeros((4,), dtype=np.float32),
    )
    labels = np.arange(4, dtype=np.int32)
    sequence = trainer._sequence(transitions, labels, {"kind": "stationary"})
    expected = act * protocol.mode_gain_vectors(6)[labels]
    np.testing.assert_allclose(sequence["target_action"], expected)
    assert "executed_action" not in sequence


def test_collectors_do_not_request_realized_action_field():
    stationary = inspect.getsource(trainer._collect_stationary)
    switching = inspect.getsource(trainer._collect_switching)
    assert "return_executed_action" not in stationary
    assert "return_executed_action" not in switching


def test_v4_reuses_frozen_splits_and_gates():
    assert protocol.TRAIN_CONTROLLER_SEEDS == (8, 16)
    assert protocol.VALIDATION_CONTROLLER_SEEDS == (24,)
    assert protocol.TEST_CONTROLLER_SEEDS == (101, 211, 307, 419, 523)
    assert protocol.MIN_MODE_ACCURACY == 0.85
    assert protocol.MAX_MEDIAN_SWITCH_DELAY == 25.0
    assert protocol.MAX_P90_SWITCH_DELAY == 50.0
    assert protocol.MAX_BRIER_SCORE == 0.25


def test_scheduler_graph_is_file_gated_and_gpu_claim_is_small():
    training = submitter.candidates("training", "high")
    audits = submitter.candidates("audit", "high")
    analysis = submitter.candidates("analysis", "high")
    assert len(training) == 1
    assert len(audits) == 5
    assert len(analysis) == 1
    assert training[0][1]["vram"] == 2500
    assert len(training[0][1]["wait_for_files"]) == 24
    assert "jtl110gpu2" not in training[0][1]["allowed_nodes"]
    assert "jtl311linux" not in training[0][1]["allowed_nodes"]
    assert "auto-adopt" not in training[0][1]["cmd"]
    for _, spec, _ in audits:
        assert spec["vram"] == 0
        assert len(spec["wait_for_files"]) == 10
        assert spec["allowed_nodes"] == submitter.CPU_NODES
    assert len(analysis[0][1]["wait_for_files"]) == 5
