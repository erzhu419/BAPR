"""Frozen development protocol for posterior-conditioned action residuals."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from jax_experiments.analysis import (
    bapr_v3_structured_channel_confirmation as confirmation,
)
from jax_experiments.analysis import bapr_v3_utility_aware_router as utility


ROOT = utility.ROOT
FAMILY = utility.FAMILY
ENV = utility.ENV
DECISION_VARIANT = confirmation.DECISION_VARIANT

# These are development streams already used to select the frozen CUSUM
# filter. They may screen controller capacity, but cannot confirm a claim.
DEVELOPMENT_EVENT_SEEDS = utility.VALIDATION_EVENT_SEEDS
SEALED_CONFIRMATION_EVENT_SEEDS = (11100, 11200, 11300, 11400, 11500)

RESIDUAL_VARIANTS = {
    "cap025": 0.25,
    "cap050": 0.50,
    "cap075": 0.75,
    "cap100": 1.00,
}

# Prospective promotion gate for the two-seed capacity screen.
MIN_FULL_CYCLE_MEAN_GAIN = 50.0
STATIONARY_MEAN_MARGIN = 50.0
STATIONARY_PER_SEED_MARGIN = 100.0
TERMINATION_RATE_MARGIN = 0.0
MIN_ADAPTATION_RATE = 0.05

RESULT_ROOT = (
    ROOT / "jax_experiments"
    / "results_bapr_v3_structured_channel_posterior_residual_screen_v1"
)
ANALYSIS_ROOT = RESULT_ROOT / "analysis"
ANALYSIS_JSON = ANALYSIS_ROOT / "summary.json"
ANALYSIS_REPORT = ANALYSIS_ROOT / "report.md"

GROUP_SCHEMA = "bapr.v3-posterior-residual-screen-group.v1"
ANALYSIS_SCHEMA = "bapr.v3-posterior-residual-screen-analysis.v1"


def configure() -> None:
    utility.configure()


def residual_cap(variant: str) -> float:
    try:
        return float(RESIDUAL_VARIANTS[str(variant)])
    except KeyError as exc:
        raise ValueError(f"unknown posterior-residual variant {variant!r}") \
            from exc


def group_path(variant: str, event_seed: int) -> Path:
    residual_cap(variant)
    if int(event_seed) not in DEVELOPMENT_EVENT_SEEDS:
        raise ValueError(f"unregistered development seed {event_seed}")
    return (
        RESULT_ROOT / str(variant) / f"event_seed_{int(event_seed)}"
        / "group.json"
    )


def file_record(path: Path):
    return utility.estimator.file_record(path)


def write_json_atomic(path: Path, payload: Any) -> None:
    utility.estimator.write_json_atomic(path, payload)


def _probabilities(posterior: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(posterior, dtype=np.float64)
    if (probabilities.shape != (4,)
            or not np.all(np.isfinite(probabilities))
            or np.any(probabilities < 0.0)
            or float(np.sum(probabilities)) <= 0.0):
        raise ValueError("posterior must be a finite nonnegative 4-vector")
    return probabilities / np.sum(probabilities)


def residual_advantage_scale(table: dict[str, Any]) -> float:
    """Derive the only residual scale from calibration-frozen utilities."""
    matrix, controllers = utility.utility_matrix(table)
    robust_index = controllers.index(utility.ROBUST_CONTROLLER)
    specialist_indices = [
        index for index, controller in enumerate(controllers)
        if controller != utility.ROBUST_CONTROLLER
    ]
    if not specialist_indices:
        raise ValueError("utility table has no nondominated specialist")
    positive = []
    for mode in range(matrix.shape[0]):
        advantage = (
            float(np.max(matrix[mode, specialist_indices]))
            - float(matrix[mode, robust_index])
        )
        if advantage > 0.0:
            positive.append(advantage)
    if not positive:
        raise ValueError("utility table has no positive specialist headroom")
    scale = float(np.median(np.asarray(positive, dtype=np.float64)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("invalid residual advantage scale")
    return scale


def posterior_residual_decision(
    posterior: np.ndarray,
    table: dict[str, Any],
    cap: float,
) -> tuple[int, float, dict[str, float]]:
    """Map a causal mode posterior to a robust-anchored residual strength."""
    if not np.isfinite(cap) or not 0.0 < float(cap) <= 1.0:
        raise ValueError("residual cap must be in (0, 1]")
    probabilities = _probabilities(posterior)
    matrix, controllers = utility.utility_matrix(table)
    expected = probabilities @ matrix
    robust_index = controllers.index(utility.ROBUST_CONTROLLER)
    specialist_indices = [
        index for index, controller in enumerate(controllers)
        if controller != utility.ROBUST_CONTROLLER
    ]
    best_index = max(
        specialist_indices,
        key=lambda index: (float(expected[index]), -controllers[index]),
    )
    best_controller = int(controllers[best_index])
    robust_utility = float(expected[robust_index])
    best_utility = float(expected[best_index])
    advantage = best_utility - robust_utility
    scale = residual_advantage_scale(table)
    strength = float(cap) * float(np.clip(advantage / scale, 0.0, 1.0))
    selected = (
        best_controller if strength > 0.0 else utility.ROBUST_CONTROLLER)
    ordered = np.sort(probabilities)[::-1]
    positive_probabilities = probabilities[probabilities > 0.0]
    entropy = -float(np.sum(
        positive_probabilities * np.log(positive_probabilities))) / np.log(4.0)
    return selected, strength, {
        "confidence": float(ordered[0]),
        "margin": float(ordered[0] - ordered[1]),
        "posterior_entropy": entropy,
        "best_expected_utility": best_utility,
        "robust_expected_utility": robust_utility,
        "best_expected_advantage": advantage,
        "advantage_scale": scale,
        "adaptation_strength": strength,
    }


def blend_action(
    robust_action: np.ndarray,
    specialist_action: np.ndarray,
    strength: float,
) -> np.ndarray:
    robust = np.asarray(robust_action, dtype=np.float32)
    specialist = np.asarray(specialist_action, dtype=np.float32)
    if robust.shape != specialist.shape:
        raise ValueError("robust and specialist actions must have equal shape")
    if (not np.all(np.isfinite(robust))
            or not np.all(np.isfinite(specialist))
            or not np.isfinite(strength)
            or not 0.0 <= float(strength) <= 1.0):
        raise ValueError("invalid residual action inputs")
    return np.clip(
        robust + float(strength) * (specialist - robust), -1.0, 1.0)
