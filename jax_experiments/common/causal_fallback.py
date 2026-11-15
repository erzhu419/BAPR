"""Causal robust fallback for posterior-conditioned controllers."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class FallbackConfig:
    """Thresholds for entering and leaving robust fallback."""

    name: str
    contradiction_threshold: float
    stable_steps: int
    enter_confidence: float = 0.60
    exit_confidence: float = 0.90

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("fallback config requires a name")
        if self.contradiction_threshold < 0.0:
            raise ValueError("contradiction threshold must be nonnegative")
        if self.stable_steps <= 0:
            raise ValueError("stable_steps must be positive")
        if not 0.0 <= self.enter_confidence <= self.exit_confidence <= 1.0:
            raise ValueError("invalid fallback confidence thresholds")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FallbackState:
    """Serializable state for a causal fallback gate."""

    fallback: bool = True
    candidate_mode: int = -1
    stable_count: int = 0
    trigger_count: int = 0


def initial_fallback_state() -> FallbackState:
    """Start conservatively in robust fallback."""
    return FallbackState()


def _finite_vector(values, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size < 2 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector with at least two modes")
    return vector


def evidence_margin(log_likelihood, mode: int) -> float:
    """Return best-alternative minus selected-mode log likelihood."""
    evidence = _finite_vector(log_likelihood, "fallback evidence")
    mode = int(mode)
    if not 0 <= mode < evidence.size:
        raise ValueError("fallback mode is outside the evidence vector")
    alternatives = np.delete(evidence, mode)
    return float(np.max(alternatives) - evidence[mode])


def update_fallback_state(
    state: FallbackState,
    posterior_before,
    posterior_after,
    log_likelihood,
    config: FallbackConfig,
) -> FallbackState:
    """Update fallback only after observing the latest transition."""
    before = _finite_vector(posterior_before, "posterior_before")
    after = _finite_vector(posterior_after, "posterior_after")
    evidence = _finite_vector(log_likelihood, "fallback evidence")
    if after.shape != before.shape or evidence.shape != before.shape:
        raise ValueError("fallback posterior and evidence shapes differ")

    believed_mode = int(np.argmax(before))
    next_mode = int(np.argmax(after))
    next_confidence = float(np.max(after))
    contradiction = (
        evidence_margin(evidence, believed_mode)
        > config.contradiction_threshold
    )
    mode_changed = next_mode != believed_mode
    low_confidence = next_confidence < config.enter_confidence

    if not state.fallback and (contradiction or mode_changed or low_confidence):
        return FallbackState(
            fallback=True,
            candidate_mode=next_mode,
            stable_count=0,
            trigger_count=state.trigger_count + 1,
        )
    if not state.fallback:
        return state

    supported = (
        next_confidence >= config.exit_confidence
        and evidence_margin(evidence, next_mode)
        <= config.contradiction_threshold
    )
    if supported:
        count = (
            state.stable_count + 1
            if state.candidate_mode == next_mode else 1
        )
        candidate_mode = next_mode
    else:
        count = 0
        candidate_mode = next_mode
    if count >= config.stable_steps:
        return FallbackState(
            fallback=False,
            candidate_mode=candidate_mode,
            stable_count=count,
            trigger_count=state.trigger_count,
        )
    return FallbackState(
        fallback=True,
        candidate_mode=candidate_mode,
        stable_count=count,
        trigger_count=state.trigger_count,
    )


class CausalFallbackGate:
    """Stateful runtime facade over the audited pure transition function."""

    def __init__(
        self,
        config: FallbackConfig,
        state: FallbackState | None = None,
    ) -> None:
        self.config = config
        self.state = state or initial_fallback_state()

    @property
    def fallback(self) -> bool:
        return self.state.fallback

    def reset(self) -> FallbackState:
        self.state = initial_fallback_state()
        return self.state

    def observe(
        self,
        posterior_before,
        posterior_after,
        log_likelihood,
    ) -> FallbackState:
        self.state = update_fallback_state(
            self.state,
            posterior_before,
            posterior_after,
            log_likelihood,
            self.config,
        )
        return self.state

    def checkpoint_state(self) -> dict[str, Any]:
        return asdict(self.state)

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        restored = FallbackState(**state)
        if restored.stable_count < 0 or restored.trigger_count < 0:
            raise ValueError("invalid fallback checkpoint counters")
        self.state = restored


class CausalFallbackPolicy:
    """Deployable robust/adaptive policy router with causal estimator updates.

    The router never receives physical mode or a switch clock. The action at
    time ``t`` uses only estimator state built from transitions before ``t``;
    the transition produced by that action can affect only later actions.
    """

    def __init__(
        self,
        robust_action: Callable[[np.ndarray], np.ndarray],
        adaptive_action: Callable[[np.ndarray, np.ndarray], np.ndarray],
        estimator: Any,
        config: FallbackConfig,
    ) -> None:
        self._robust_action = robust_action
        self._adaptive_action = adaptive_action
        self.estimator = estimator
        self.gate = CausalFallbackGate(config)
        self.estimator_state = None
        self.action_count = 0
        self.fallback_action_count = 0
        self.last_source = "robust"
        self.reset()

    @property
    def posterior(self) -> np.ndarray:
        values = np.asarray(
            self.estimator.probabilities(self.estimator_state),
            dtype=np.float64,
        )
        return _finite_vector(values, "estimator posterior").copy()

    @property
    def fallback(self) -> bool:
        return self.gate.fallback

    @property
    def fallback_action_fraction(self) -> float:
        return float(self.fallback_action_count / max(self.action_count, 1))

    def reset(self) -> None:
        self.estimator_state = self.estimator.initial_state()
        self.gate.reset()
        self.action_count = 0
        self.fallback_action_count = 0
        self.last_source = "robust"

    def select_action(self, observation) -> np.ndarray:
        observation = np.asarray(observation)
        if self.gate.fallback:
            action = self._robust_action(observation)
            self.last_source = "robust"
            self.fallback_action_count += 1
        else:
            action = self._adaptive_action(observation, self.posterior)
            self.last_source = "adaptive"
        self.action_count += 1
        action = np.asarray(action, dtype=np.float32)
        if action.ndim != 1 or not np.all(np.isfinite(action)):
            raise ValueError("fallback policy produced an invalid action")
        return action

    def observe_transition(
        self,
        observation,
        action,
        reward: float,
        next_observation,
    ) -> FallbackState:
        posterior_before = self.posterior
        next_state, evidence, _, _ = self.estimator.step(
            self.estimator_state,
            observation,
            action,
            reward,
            next_observation,
        )
        posterior_after = np.asarray(
            self.estimator.probabilities(next_state), dtype=np.float64)
        self.gate.observe(
            posterior_before, posterior_after, evidence)
        self.estimator_state = next_state
        return self.gate.state

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "estimator_state": np.asarray(self.estimator_state).copy(),
            "gate": self.gate.checkpoint_state(),
            "action_count": int(self.action_count),
            "fallback_action_count": int(self.fallback_action_count),
            "last_source": self.last_source,
        }

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        action_count = int(state["action_count"])
        fallback_count = int(state["fallback_action_count"])
        source = str(state["last_source"])
        if (action_count < 0 or fallback_count < 0
                or fallback_count > action_count
                or source not in ("robust", "adaptive")):
            raise ValueError("invalid fallback policy checkpoint counters")
        estimator_state = np.asarray(state["estimator_state"])
        _finite_vector(
            self.estimator.probabilities(estimator_state),
            "checkpoint estimator posterior",
        )
        self.gate.load_checkpoint_state(state["gate"])
        self.estimator_state = estimator_state.copy()
        self.action_count = action_count
        self.fallback_action_count = fallback_count
        self.last_source = source
