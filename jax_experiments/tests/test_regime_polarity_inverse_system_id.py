"""Tests for the v3 executed-action inverse system-ID screen."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.analysis import (
    regime_polarity_inverse_system_id as protocol,
)
from jax_experiments.networks.executed_action_inverse import (
    ExecutedActionInverse,
    candidate_action_evidence,
)
from scripts import submit_regime_polarity_inverse_system_id as submitter


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


def test_mode_gain_vectors_match_halfcheetah_polarity_patterns():
    gains = protocol.mode_gain_vectors(6)
    np.testing.assert_array_equal(gains, np.asarray([
        [-1, -1, -1, 1, 1, 1],
        [1, 1, 1, -1, -1, -1],
        [-1, 1, -1, 1, -1, 1],
        [1, -1, 1, -1, 1, -1],
    ], dtype=np.float32))


def test_candidate_evidence_recovers_matching_transform():
    gains = jnp.asarray(protocol.mode_gain_vectors(6))
    commanded = jnp.asarray([[
        0.8, -0.7, 0.6, -0.5, 0.4, -0.3,
    ]])
    wanted = 2
    executed = jnp.clip(commanded * gains[wanted], -1.0, 1.0)
    predictions = jnp.broadcast_to(executed, (5, 1, 6))
    evidence, aleatoric, epistemic = candidate_action_evidence(
        predictions,
        commanded,
        gains,
        jnp.full((6,), 0.01),
    )
    assert int(jnp.argmax(evidence[0])) == wanted
    assert evidence.shape == aleatoric.shape == epistemic.shape == (1, 4)


def test_inverse_model_has_independent_head_batches():
    model = ExecutedActionInverse(
        obs_dim=5,
        act_dim=3,
        hidden_dim=16,
        ensemble_size=4,
        n_layers=2,
        rngs=nnx.Rngs(7),
    )
    obs = jnp.zeros((4, 8, 5))
    next_obs = jnp.ones((4, 8, 5)) * 0.1
    predicted = model.predict_head_batches(obs, next_obs)
    assert predicted.shape == (4, 8, 3)
    assert not np.allclose(
        np.asarray(predicted[0]), np.asarray(predicted[1]))


def test_scheduler_graph_is_file_gated_and_excludes_unavailable_nodes():
    training = submitter.candidates("training", "high")
    audits = submitter.candidates("audit", "high")
    analysis = submitter.candidates("analysis", "high")
    assert len(training) == 1
    assert len(audits) == 5
    assert len(analysis) == 1
    assert training[0][1]["vram"] == 2500
    assert len(training[0][1]["wait_for_files"]) == 24
    assert "auto-adopt" not in training[0][1]["cmd"]
    assert "jtl110gpu2" not in training[0][1]["allowed_nodes"]
    assert "jtl311linux" not in training[0][1]["allowed_nodes"]
    for _, spec, _ in audits:
        assert spec["vram"] == 0
        assert spec["cpu"] == 8
        assert len(spec["wait_for_files"]) == 10
        assert spec["allowed_nodes"] == submitter.CPU_NODES
        assert "JAX_PLATFORMS=cpu" in spec["cmd"]
    assert len(analysis[0][1]["wait_for_files"]) == 5
