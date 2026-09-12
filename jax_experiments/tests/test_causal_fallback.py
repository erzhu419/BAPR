"""Behavioral identity tests for the deployable causal fallback gate."""
from __future__ import annotations

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_policy_distillation_transient_fallback_v1 as protocol,
)
from jax_experiments.common import causal_fallback


def _supported_mode(mode: int) -> tuple[np.ndarray, np.ndarray]:
    posterior = np.full((4,), 0.01, dtype=np.float64)
    posterior[mode] = 0.97
    evidence = np.full((4,), -4.0, dtype=np.float64)
    evidence[mode] = 0.0
    return posterior, evidence


def test_protocol_uses_the_deployable_transition_function():
    assert protocol.FallbackConfig is causal_fallback.FallbackConfig
    assert protocol.FallbackState is causal_fallback.FallbackState
    assert protocol.update_fallback_state is causal_fallback.update_fallback_state


def test_runtime_gate_matches_the_pure_audited_updates():
    config = causal_fallback.FallbackConfig(
        "evidence_1p0_k1", contradiction_threshold=1.0, stable_steps=1)
    gate = causal_fallback.CausalFallbackGate(config)
    expected = causal_fallback.initial_fallback_state()

    posterior0, evidence0 = _supported_mode(0)
    expected = causal_fallback.update_fallback_state(
        expected, posterior0, posterior0, evidence0, config)
    assert gate.observe(posterior0, posterior0, evidence0) == expected
    assert gate.fallback is False

    posterior1, evidence1 = _supported_mode(1)
    expected = causal_fallback.update_fallback_state(
        expected, posterior0, posterior1, evidence1, config)
    assert gate.observe(posterior0, posterior1, evidence1) == expected
    assert gate.fallback is True
    assert gate.state.trigger_count == 1


def test_runtime_gate_checkpoint_round_trip():
    config = causal_fallback.FallbackConfig(
        "evidence_1p0_k1", contradiction_threshold=1.0, stable_steps=1)
    gate = causal_fallback.CausalFallbackGate(config)
    posterior, evidence = _supported_mode(2)
    gate.observe(posterior, posterior, evidence)

    restored = causal_fallback.CausalFallbackGate(config)
    restored.load_checkpoint_state(gate.checkpoint_state())
    assert restored.state == gate.state
    assert restored.reset() == causal_fallback.initial_fallback_state()


class _Estimator:
    def initial_state(self):
        return np.asarray([0.97, 0.01, 0.01, 0.01], dtype=np.float64)

    @staticmethod
    def probabilities(state):
        return np.asarray(state, dtype=np.float64)

    def step(self, state, obs, action, reward, next_obs):
        del state, obs, action, reward, next_obs
        posterior, evidence = _supported_mode(1)
        return posterior, evidence, np.zeros(4), np.zeros(4)


def test_policy_router_updates_only_after_the_current_action():
    config = causal_fallback.FallbackConfig(
        "evidence_1p0_k1", contradiction_threshold=1.0, stable_steps=1)
    router = causal_fallback.CausalFallbackPolicy(
        lambda obs: np.asarray([-0.5, 0.5], dtype=np.float32),
        lambda obs, posterior: np.asarray(
            [posterior[0], posterior[1]], dtype=np.float32),
        _Estimator(),
        config,
    )

    first = router.select_action(np.zeros(3))
    assert router.last_source == "robust"
    assert np.allclose(first, [-0.5, 0.5])
    router.observe_transition(np.zeros(3), first, 0.0, np.ones(3))
    assert router.fallback is False
    second = router.select_action(np.ones(3))
    assert router.last_source == "adaptive"
    assert np.allclose(second, [0.01, 0.97])

    restored = causal_fallback.CausalFallbackPolicy(
        lambda obs: np.asarray([-0.5, 0.5], dtype=np.float32),
        lambda obs, posterior: np.asarray(
            [posterior[0], posterior[1]], dtype=np.float32),
        _Estimator(),
        config,
    )
    restored.load_checkpoint_state(router.checkpoint_state())
    assert restored.gate.state == router.gate.state
    assert np.allclose(restored.posterior, router.posterior)
    assert restored.fallback_action_fraction == router.fallback_action_fraction


if __name__ == "__main__":
    test_protocol_uses_the_deployable_transition_function()
    test_runtime_gate_matches_the_pure_audited_updates()
    test_runtime_gate_checkpoint_round_trip()
    test_policy_router_updates_only_after_the_current_action()
