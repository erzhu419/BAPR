import numpy as np

from jax_experiments.analysis.regime_polarity_specialist_sticky_router_v3 import (
    StickySpecialistOption,
)


class _Estimator:
    def initial_state(self):
        return {"mode": 0}

    def probabilities(self, state):
        values = np.full(4, 0.01)
        values[state["mode"]] = 0.97
        return values

    def step(self, state, observation, action, reward, next_observation):
        mode = int(next_observation[0])
        evidence = np.full(4, -3.0)
        evidence[mode] = 3.0
        return {"mode": mode}, evidence, None, None


def _action(value):
    return lambda observation: np.asarray([value], dtype=np.float32)


def test_sticky_option_never_falls_back_after_initial_identification():
    router = StickySpecialistOption(
        _action(-1), tuple(_action(mode) for mode in range(4)),
        _Estimator(), switch_confirmation_steps=3)
    observation = np.asarray([0])
    for _ in range(3):
        assert router.select_action(observation).source == "robust"
        router.observe_transition(
            observation, np.asarray([-1]), 0.0, np.asarray([2]))
    assert router.active_mode == 2

    for _ in range(2):
        step = router.select_action(observation)
        assert step.source == "specialist"
        assert step.selected_mode == 2
        router.observe_transition(
            observation, step.action, 0.0, np.asarray([1]))
        assert router.active_mode == 2

    step = router.select_action(observation)
    assert step.source == "specialist"
    assert step.selected_mode == 2
    router.observe_transition(
        observation, step.action, 0.0, np.asarray([1]))
    assert router.active_mode == 1
    assert router.switch_count == 1
    assert router.robust_action_fraction == 0.5
