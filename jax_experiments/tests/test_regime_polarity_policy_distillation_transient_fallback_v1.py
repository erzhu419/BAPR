"""Tests for the causal transient-fallback state machine."""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from jax_experiments.analysis import (
    regime_polarity_policy_distillation_transient_fallback_v1 as protocol,
)


def test_registered_splits_and_configs_are_unique():
    events = (*protocol.SCREEN_EVENT_SEEDS, *protocol.AUDIT_EVENT_SEEDS)
    assert len(events) == len(set(events))
    names = [config.name for config in protocol.FALLBACK_CONFIGS]
    assert len(names) == len(set(names)) == 8
    assert protocol.ROBUST_ARM not in names


def test_gate_exits_only_after_registered_stability_count():
    config = protocol.FallbackConfig("test", 1.0, stable_steps=3)
    state = protocol.initial_fallback_state()
    posterior = np.asarray([0.98, 0.01, 0.005, 0.005])
    evidence = np.asarray([0.0, -3.0, -4.0, -5.0])
    for expected_count in (1, 2):
        state = protocol.update_fallback_state(
            state, posterior, posterior, evidence, config)
        assert state.fallback is True
        assert state.stable_count == expected_count
    state = protocol.update_fallback_state(
        state, posterior, posterior, evidence, config)
    assert state.fallback is False
    assert state.stable_count == 3


def test_contradictory_evidence_triggers_fallback_despite_high_confidence():
    config = protocol.FallbackConfig("test", 1.0, stable_steps=1)
    state = protocol.FallbackState(
        fallback=False, candidate_mode=0, stable_count=1)
    old_posterior = np.asarray([0.999, 0.0005, 0.0003, 0.0002])
    still_old = np.asarray([0.98, 0.01, 0.005, 0.005])
    evidence = np.asarray([-3.0, 0.0, -4.0, -5.0])
    next_state = protocol.update_fallback_state(
        state, old_posterior, still_old, evidence, config)
    assert next_state.fallback is True
    assert next_state.trigger_count == 1


def test_supported_belief_does_not_false_trigger():
    config = protocol.FallbackConfig("test", 1.0, stable_steps=1)
    state = protocol.FallbackState(
        fallback=False, candidate_mode=2, stable_count=1)
    posterior = np.asarray([0.001, 0.001, 0.997, 0.001])
    evidence = np.asarray([-4.0, -3.0, 0.0, -5.0])
    assert protocol.update_fallback_state(
        state, posterior, posterior, evidence, config) == state


def test_gate_has_no_physical_mode_or_switch_clock_input():
    parameters = inspect.signature(protocol.update_fallback_state).parameters
    assert "mode" not in parameters
    assert "switch_clock" not in parameters


@pytest.mark.parametrize(
    "kwargs",
    [
        {"contradiction_threshold": -1.0, "stable_steps": 1},
        {"contradiction_threshold": 1.0, "stable_steps": 0},
        {
            "contradiction_threshold": 1.0,
            "stable_steps": 1,
            "enter_confidence": 0.9,
            "exit_confidence": 0.8,
        },
    ],
)
def test_invalid_gate_configs_fail_closed(kwargs):
    with pytest.raises(ValueError):
        protocol.FallbackConfig("invalid", **kwargs)
