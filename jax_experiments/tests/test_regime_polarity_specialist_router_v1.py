import numpy as np
import pytest

from jax_experiments.analysis import (
    regime_polarity_specialist_router_v1 as protocol,
)
from jax_experiments.common.causal_fallback import FallbackConfig


class _Estimator:
    def initial_state(self):
        return 0

    def probabilities(self, state):
        if int(state) == 0:
            return np.full((4,), 0.25, dtype=np.float64)
        return np.asarray([0.02, 0.03, 0.93, 0.02], dtype=np.float64)

    def step(self, state, observation, action, reward, next_observation):
        del state, observation, action, reward, next_observation
        return 1, np.asarray([0.0, 0.0, 1.0, 0.0]), None, None


def _action(value):
    return lambda observation: np.full((2,), value, dtype=np.float32)


def _router(reduction):
    return protocol.CausalSpecialistRouter(
        _action(-1.0),
        tuple(_action(float(mode)) for mode in protocol.MODES),
        _Estimator(),
        reduction,
        FallbackConfig(
            name="test",
            contradiction_threshold=0.0,
            stable_steps=1,
            enter_confidence=0.6,
            exit_confidence=0.9,
        ),
    )


def test_map_router_falls_back_then_selects_posterior_mode():
    router = _router("map")
    first = router.select_action(np.zeros((3,), dtype=np.float32))
    assert first.source == "robust"
    np.testing.assert_array_equal(first.action, [-1.0, -1.0])
    router.observe_transition(
        np.zeros((3,)), first.action, 0.0, np.zeros((3,)))
    second = router.select_action(np.zeros((3,), dtype=np.float32))
    assert second.source == "specialist"
    assert second.selected_mode == 2
    np.testing.assert_array_equal(second.action, [2.0, 2.0])


def test_soft_router_uses_posterior_weighted_specialist_action():
    router = _router("soft")
    first = router.select_action(np.zeros((3,), dtype=np.float32))
    router.observe_transition(
        np.zeros((3,)), first.action, 0.0, np.zeros((3,)))
    second = router.select_action(np.zeros((3,), dtype=np.float32))
    assert second.source == "specialist_soft"
    expected = 0.02 * 0.0 + 0.03 * 1.0 + 0.93 * 2.0 + 0.02 * 3.0
    np.testing.assert_allclose(second.action, [expected, expected])


def test_router_rejects_unknown_reduction():
    with pytest.raises(ValueError, match="unknown specialist reduction"):
        _router("median")
