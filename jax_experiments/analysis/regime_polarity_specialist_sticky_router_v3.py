"""Persistent specialist routing without within-option robust fallback."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_specialist_router_diagnostic_v2 as parent,
)
from jax_experiments.common.causal_fallback import evidence_margin


ROOT = parent.ROOT
PROTOCOL_VERSION = "v3-sticky-independent-specialist-router"
ENV = parent.ENV
FAMILY = parent.FAMILY
MODES = parent.MODES
TRAINING_SEEDS = parent.TRAINING_SEEDS
EVENT_SEEDS = (153_241, 153_253, 153_269)

PRIMARY_ARM = "posterior_sticky_confirm3"
STICKY_ARMS = {
    "posterior_sticky_confirm2": 2,
    PRIMARY_ARM: 3,
    "posterior_sticky_confirm5": 5,
}
ARMS = (
    "robust_sac",
    "dynamic_oracle",
    "posterior_map_no_gate",
    *STICKY_ARMS,
)

MAX_EPISODE_STEPS = parent.MAX_EPISODE_STEPS
DWELL_STEPS = parent.DWELL_STEPS
SWITCHING_EPISODES = parent.SWITCHING_EPISODES
FALLBACK_CONFIG = parent.FALLBACK_CONFIG
INITIAL_CONFIRMATION_STEPS = 3
MIN_ORACLE_RECOVERY = parent.MIN_ORACLE_RECOVERY
MIN_SWITCHING_RETURN = parent.MIN_SWITCHING_RETURN
MIN_GAIN = 0.10

AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_specialist_sticky_router_audit_v3"
)
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_specialist_sticky_router_analysis_v3"
)
REPORT = (
    ROOT / "reports"
    / "regime_polarity_specialist_sticky_router_v3_2026-08-29.md"
)

EVENT_SCHEMA = "bapr.regime-polarity-specialist-sticky-router-event.v3"
AUDIT_SCHEMA = "bapr.regime-polarity-specialist-sticky-router-audit.v3"
ANALYSIS_SCHEMA = "bapr.regime-polarity-specialist-sticky-router-analysis.v3"

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
        raise ValueError(f"unknown sticky-router event seed {seed}")
    return seed


def require_arm(arm: str) -> str:
    arm = str(arm)
    if arm not in ARMS:
        raise ValueError(f"unknown sticky-router arm {arm!r}")
    return arm


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


class StickySpecialistOption:
    """Keep one specialist active until a new mode is persistently supported."""

    def __init__(
        self,
        robust_action: Callable[[np.ndarray], np.ndarray],
        specialist_actions: tuple[Callable[[np.ndarray], np.ndarray], ...],
        estimator: Any,
        switch_confirmation_steps: int,
    ) -> None:
        if len(specialist_actions) != len(MODES):
            raise ValueError("sticky option requires one specialist per mode")
        if int(switch_confirmation_steps) <= 0:
            raise ValueError("switch confirmation must be positive")
        self.robust_action = robust_action
        self.specialist_actions = tuple(specialist_actions)
        self.estimator = estimator
        self.switch_confirmation_steps = int(switch_confirmation_steps)
        self.estimator_state = None
        self.active_mode = -1
        self.candidate_mode = -1
        self.candidate_count = 0
        self.action_count = 0
        self.robust_action_count = 0
        self.switch_count = 0
        self.reset()

    @property
    def posterior(self) -> np.ndarray:
        values = np.asarray(
            self.estimator.probabilities(self.estimator_state),
            dtype=np.float64,
        )
        if values.shape != (len(MODES),) or not np.all(np.isfinite(values)):
            raise ValueError("sticky option received an invalid posterior")
        total = float(np.sum(values))
        if total <= 0.0:
            raise ValueError("sticky option posterior has no mass")
        return values / total

    def reset(self) -> None:
        self.estimator_state = self.estimator.initial_state()
        self.active_mode = -1
        self.candidate_mode = -1
        self.candidate_count = 0
        self.action_count = 0
        self.robust_action_count = 0
        self.switch_count = 0

    def select_action(self, observation) -> OptionStep:
        if self.active_mode < 0:
            action = self.robust_action(observation)
            source = "robust"
            self.robust_action_count += 1
        else:
            action = self.specialist_actions[self.active_mode](observation)
            source = "specialist"
        self.action_count += 1
        action = np.asarray(action, dtype=np.float32)
        if action.ndim != 1 or not np.all(np.isfinite(action)):
            raise ValueError("sticky option produced an invalid action")
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
            confidence >= FALLBACK_CONFIG.exit_confidence
            and evidence_margin(evidence, mode)
            <= FALLBACK_CONFIG.contradiction_threshold
        )
        target_is_new = self.active_mode < 0 or mode != self.active_mode
        if supported and target_is_new:
            if mode == self.candidate_mode:
                self.candidate_count += 1
            else:
                self.candidate_mode = mode
                self.candidate_count = 1
        else:
            self.candidate_mode = -1
            self.candidate_count = 0

        required = (
            INITIAL_CONFIRMATION_STEPS
            if self.active_mode < 0 else self.switch_confirmation_steps
        )
        if self.candidate_count >= required:
            if self.active_mode >= 0:
                self.switch_count += 1
            self.active_mode = self.candidate_mode
            self.candidate_mode = -1
            self.candidate_count = 0
        self.estimator_state = next_state

    @property
    def robust_action_fraction(self) -> float:
        return float(self.robust_action_count / max(self.action_count, 1))
