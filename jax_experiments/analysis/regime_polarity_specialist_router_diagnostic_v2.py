"""Switching-only diagnosis for independent-specialist causal routing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_specialist_router_v1 as parent,
)
from jax_experiments.common.causal_fallback import evidence_margin


ROOT = parent.ROOT
PROTOCOL_VERSION = "v2-independent-specialist-router-diagnostic"
ENV = parent.ENV
FAMILY = parent.FAMILY
MODES = parent.MODES
TRAINING_SEEDS = parent.TRAINING_SEEDS
EVENT_SEEDS = (153_211, 153_223)

ARMS = (
    "robust_sac",
    "dynamic_oracle",
    "posterior_map_fallback",
    "true_mode_current_gate",
    "true_mode_robust10",
    "posterior_map_no_gate",
    "posterior_debounced_option",
)
LEARNED_ARMS = (
    "posterior_map_fallback",
    "true_mode_current_gate",
    "posterior_map_no_gate",
    "posterior_debounced_option",
)
PRIVILEGED_ARMS = (
    "dynamic_oracle", "true_mode_current_gate", "true_mode_robust10")

MAX_EPISODE_STEPS = parent.MAX_EPISODE_STEPS
DWELL_STEPS = parent.DWELL_STEPS
SWITCHING_EPISODES = parent.SWITCHING_EPISODES
FALLBACK_CONFIG = parent.FALLBACK_CONFIG
WARMUP_STEPS = 10
OPTION_STABLE_STEPS = 3
CONTRADICTION_STEPS = 2
MIN_ORACLE_RECOVERY = parent.MIN_ORACLE_RECOVERY
MIN_SWITCHING_RETURN = parent.MIN_SWITCHING_RETURN

AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_specialist_router_diagnostic_audit_v2"
)
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_specialist_router_diagnostic_analysis_v2"
)
REPORT = (
    ROOT / "reports"
    / "regime_polarity_specialist_router_diagnostic_v2_2026-08-29.md"
)

EVENT_SCHEMA = "bapr.regime-polarity-specialist-router-diagnostic-event.v2"
AUDIT_SCHEMA = "bapr.regime-polarity-specialist-router-diagnostic-audit.v2"
ANALYSIS_SCHEMA = "bapr.regime-polarity-specialist-router-diagnostic-analysis.v2"

file_record = parent.file_record
read_json = parent.read_json
write_json_atomic = parent.write_json_atomic
write_text_atomic = parent.write_text_atomic
source_records = parent.source_records
estimator_records = parent.estimator_records


def require_training_seed(seed: int) -> int:
    return parent.require_training_seed(seed)


def require_event_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in EVENT_SEEDS:
        raise ValueError(f"unknown specialist diagnostic event seed {seed}")
    return seed


def audit_dir(seed: int) -> Path:
    return AUDIT_ROOT / f"seed_{require_training_seed(seed)}"


def audit_manifest(seed: int) -> Path:
    return audit_dir(seed) / "audit_manifest.json"


def event_result(seed: int, event_seed: int) -> Path:
    return (
        audit_dir(seed)
        / f"event_seed_{require_event_seed(event_seed)}"
        / "results.json"
    )


def analysis_json() -> Path:
    return ANALYSIS_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return ANALYSIS_ROOT / "analysis.md"


@dataclass(frozen=True)
class OptionStep:
    action: np.ndarray
    source: str
    selected_mode: int


class DebouncedSpecialistOption:
    """Persistent specialist option with causal, debounced handoff."""

    def __init__(
        self,
        robust_action: Callable[[np.ndarray], np.ndarray],
        specialist_actions: tuple[Callable[[np.ndarray], np.ndarray], ...],
        estimator: Any,
    ) -> None:
        if len(specialist_actions) != len(MODES):
            raise ValueError("debounced option requires one specialist per mode")
        self.robust_action = robust_action
        self.specialist_actions = tuple(specialist_actions)
        self.estimator = estimator
        self.estimator_state = None
        self.active_mode = -1
        self.candidate_mode = -1
        self.candidate_count = 0
        self.contradiction_count = 0
        self.fallback = True
        self.action_count = 0
        self.fallback_action_count = 0
        self.trigger_count = 0
        self.reset()

    @property
    def posterior(self) -> np.ndarray:
        values = np.asarray(
            self.estimator.probabilities(self.estimator_state),
            dtype=np.float64,
        )
        if values.shape != (len(MODES),) or not np.all(np.isfinite(values)):
            raise ValueError("debounced option received an invalid posterior")
        total = float(np.sum(values))
        if total <= 0.0:
            raise ValueError("debounced option posterior has no mass")
        return values / total

    def reset(self) -> None:
        self.estimator_state = self.estimator.initial_state()
        self.active_mode = -1
        self.candidate_mode = -1
        self.candidate_count = 0
        self.contradiction_count = 0
        self.fallback = True
        self.action_count = 0
        self.fallback_action_count = 0
        self.trigger_count = 0

    def select_action(self, observation) -> OptionStep:
        if self.fallback or self.active_mode < 0:
            action = self.robust_action(observation)
            source = "robust"
            self.fallback_action_count += 1
        else:
            action = self.specialist_actions[self.active_mode](observation)
            source = "specialist"
        self.action_count += 1
        action = np.asarray(action, dtype=np.float32)
        if action.ndim != 1 or not np.all(np.isfinite(action)):
            raise ValueError("debounced option produced an invalid action")
        return OptionStep(action, source, int(self.active_mode))

    def observe_transition(
        self, observation, action, reward: float, next_observation
    ) -> None:
        next_state, evidence, _, _ = self.estimator.step(
            self.estimator_state,
            observation,
            action,
            reward,
            next_observation,
        )
        posterior = np.asarray(
            self.estimator.probabilities(next_state), dtype=np.float64)
        mode = int(np.argmax(posterior))
        confidence = float(np.max(posterior))
        supported = (
            evidence_margin(evidence, mode)
            <= FALLBACK_CONFIG.contradiction_threshold
        )

        if self.active_mode >= 0:
            contradicted = (
                evidence_margin(evidence, self.active_mode)
                > FALLBACK_CONFIG.contradiction_threshold
            )
            self.contradiction_count = (
                self.contradiction_count + 1 if contradicted else 0
            )
            if self.contradiction_count >= CONTRADICTION_STEPS:
                if not self.fallback:
                    self.trigger_count += 1
                self.fallback = True

        if confidence >= FALLBACK_CONFIG.exit_confidence and supported:
            if mode == self.candidate_mode:
                self.candidate_count += 1
            else:
                self.candidate_mode = mode
                self.candidate_count = 1
        else:
            self.candidate_mode = -1
            self.candidate_count = 0

        if self.active_mode >= 0 and mode != self.active_mode:
            if not self.fallback:
                self.trigger_count += 1
            self.fallback = True
        if self.candidate_count >= OPTION_STABLE_STEPS:
            self.active_mode = self.candidate_mode
            self.fallback = False
            self.candidate_count = 0
            self.contradiction_count = 0
        self.estimator_state = next_state

    @property
    def fallback_action_fraction(self) -> float:
        return float(self.fallback_action_count / max(self.action_count, 1))
