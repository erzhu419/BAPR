"""Causal routing over independent actuator-polarity specialists."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_expected_action_system_id as estimator_protocol,
)
from jax_experiments.analysis import (
    regime_polarity_fallback_final_comparison_v1 as frozen_bapr,
)
from jax_experiments.analysis import (
    regime_polarity_source_headroom_v1 as source,
)
from jax_experiments.common.causal_fallback import (
    CausalFallbackGate,
    FallbackConfig,
)


ROOT = source.ROOT
PROTOCOL_VERSION = "v1-causal-independent-specialist-router"
ENV = source.ENV
FAMILY = source.FAMILY
MODES = source.MODES
TRAINING_SEEDS = source.TRAINING_SEEDS
EVENT_SEEDS = (153_101, 153_113, 153_127)

ARMS = (
    "robust_sac",
    "dynamic_oracle",
    "posterior_map_fallback",
    "posterior_soft_fallback",
)
LEARNED_ARMS = ARMS[2:]

MAX_EPISODE_STEPS = source.MAX_EPISODE_STEPS
DWELL_STEPS = source.DWELL_STEPS
STATIONARY_EPISODES = source.EPISODES_PER_TASK
SWITCHING_EPISODES = source.SWITCHING_EPISODES
FALLBACK_CONFIG = frozen_bapr.FALLBACK_CONFIG

MIN_GAIN = 0.10
MIN_ORACLE_RECOVERY = 0.70
MIN_SWITCHING_RETURN = 2200.0

AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_specialist_router_audit_v1"
)
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_specialist_router_analysis_v1"
)
REPORT = (
    ROOT / "reports"
    / "regime_polarity_specialist_router_v1_2026-08-29.md"
)

EVENT_SCHEMA = "bapr.regime-polarity-specialist-router-event.v1"
AUDIT_SCHEMA = "bapr.regime-polarity-specialist-router-audit.v1"
ANALYSIS_SCHEMA = "bapr.regime-polarity-specialist-router-analysis.v1"

file_record = source.file_record
read_json = source.read_json
write_json_atomic = source.write_json_atomic
write_text_atomic = source.write_text_atomic


def require_training_seed(seed: int) -> int:
    return source.require_training_seed(seed)


def require_event_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in EVENT_SEEDS:
        raise ValueError(f"unknown specialist-router event seed {seed}")
    return seed


def require_arm(arm: str) -> str:
    arm = str(arm)
    if arm not in ARMS:
        raise ValueError(f"unknown specialist-router arm {arm!r}")
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


def source_records(seed: int) -> dict[str, Any]:
    seed = require_training_seed(seed)
    return {
        role: file_record(source.bundle_manifest(role, seed))
        for role in source.ROLES
    }


def estimator_records() -> dict[str, Any]:
    return {
        "manifest": file_record(estimator_protocol.MODEL_MANIFEST),
        "parameters": file_record(estimator_protocol.MODEL_PATH),
    }


@dataclass(frozen=True)
class RouterStep:
    action: np.ndarray
    source: str
    selected_mode: int


class CausalSpecialistRouter:
    """Route causally to independent specialists after evidence stabilizes."""

    def __init__(
        self,
        robust_action: Callable[[np.ndarray], np.ndarray],
        specialist_actions: tuple[Callable[[np.ndarray], np.ndarray], ...],
        estimator: Any,
        reduction: str,
        fallback_config: FallbackConfig = FALLBACK_CONFIG,
    ) -> None:
        if len(specialist_actions) != len(MODES):
            raise ValueError("specialist router requires one policy per mode")
        if reduction not in {"map", "soft"}:
            raise ValueError(f"unknown specialist reduction {reduction!r}")
        self.robust_action = robust_action
        self.specialist_actions = tuple(specialist_actions)
        self.estimator = estimator
        self.reduction = reduction
        self.gate = CausalFallbackGate(fallback_config)
        self.estimator_state = None
        self.action_count = 0
        self.fallback_action_count = 0
        self.reset()

    @property
    def posterior(self) -> np.ndarray:
        posterior = np.asarray(
            self.estimator.probabilities(self.estimator_state),
            dtype=np.float64,
        )
        if (
            posterior.shape != (len(MODES),)
            or not np.all(np.isfinite(posterior))
            or np.any(posterior < 0.0)
            or float(np.sum(posterior)) <= 0.0
        ):
            raise ValueError("specialist router received an invalid posterior")
        return posterior / float(np.sum(posterior))

    def reset(self) -> None:
        self.estimator_state = self.estimator.initial_state()
        self.gate.reset()
        self.action_count = 0
        self.fallback_action_count = 0

    def select_action(self, observation) -> RouterStep:
        observation = np.asarray(observation)
        posterior = self.posterior
        selected_mode = int(np.argmax(posterior))
        if self.gate.fallback:
            action = self.robust_action(observation)
            source_name = "robust"
            self.fallback_action_count += 1
        elif self.reduction == "map":
            action = self.specialist_actions[selected_mode](observation)
            source_name = "specialist"
        else:
            actions = np.stack(
                [action(observation) for action in self.specialist_actions],
                axis=0,
            )
            action = np.sum(actions * posterior[:, None], axis=0)
            source_name = "specialist_soft"
        action = np.asarray(action, dtype=np.float32)
        if action.ndim != 1 or not np.all(np.isfinite(action)):
            raise ValueError("specialist router produced an invalid action")
        self.action_count += 1
        return RouterStep(action, source_name, selected_mode)

    def observe_transition(
        self,
        observation,
        action,
        reward: float,
        next_observation,
    ) -> None:
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
        self.gate.observe(posterior_before, posterior_after, evidence)
        self.estimator_state = next_state

    @property
    def fallback_action_fraction(self) -> float:
        return float(self.fallback_action_count / max(self.action_count, 1))
