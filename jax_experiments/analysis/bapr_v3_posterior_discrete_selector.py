"""Frozen protocol for posterior utility selection without action blending."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from jax_experiments.analysis import bapr_v3_posterior_residual as prior


ROOT = prior.ROOT
FAMILY = prior.FAMILY
ENV = prior.ENV
DECISION_VARIANT = prior.DECISION_VARIANT

DEVELOPMENT_EVENT_SEEDS = prior.DEVELOPMENT_EVENT_SEEDS
SEALED_CONFIRMATION_EVENT_SEEDS = prior.SEALED_CONFIRMATION_EVENT_SEEDS

MIN_FULL_CYCLE_MEAN_GAIN = prior.MIN_FULL_CYCLE_MEAN_GAIN
STATIONARY_MEAN_MARGIN = prior.STATIONARY_MEAN_MARGIN
STATIONARY_PER_SEED_MARGIN = prior.STATIONARY_PER_SEED_MARGIN
TERMINATION_RATE_MARGIN = prior.TERMINATION_RATE_MARGIN
MIN_ADAPTATION_RATE = prior.MIN_ADAPTATION_RATE

RESULT_ROOT = (
    ROOT / "jax_experiments"
    / "results_bapr_v3_structured_channel_posterior_discrete_selector_v1"
)
ANALYSIS_ROOT = RESULT_ROOT / "analysis"
ANALYSIS_JSON = ANALYSIS_ROOT / "summary.json"
ANALYSIS_REPORT = ANALYSIS_ROOT / "report.md"

GROUP_SCHEMA = "bapr.v3-posterior-discrete-selector-group.v1"
ANALYSIS_SCHEMA = "bapr.v3-posterior-discrete-selector-analysis.v1"


def configure() -> None:
    prior.configure()


def group_path(event_seed: int) -> Path:
    if int(event_seed) not in DEVELOPMENT_EVENT_SEEDS:
        raise ValueError(f"unregistered posterior-discrete seed {event_seed}")
    return RESULT_ROOT / f"event_seed_{int(event_seed)}" / "group.json"


def file_record(path: Path):
    return prior.file_record(path)


def write_json_atomic(path: Path, payload: Any) -> None:
    prior.write_json_atomic(path, payload)


def select_expected_utility_controller(
    posterior: np.ndarray,
    table: dict[str, Any],
) -> tuple[int, dict[str, float]]:
    """Choose one real frozen controller by posterior expected utility."""
    probabilities = np.asarray(posterior, dtype=np.float64)
    if (probabilities.shape != (4,)
            or not np.all(np.isfinite(probabilities))
            or np.any(probabilities < 0.0)
            or float(np.sum(probabilities)) <= 0.0):
        raise ValueError("posterior must be a finite nonnegative 4-vector")
    probabilities /= np.sum(probabilities)
    matrix, controllers = prior.utility.utility_matrix(table)
    expected = probabilities @ matrix
    robust_index = controllers.index(prior.utility.ROBUST_CONTROLLER)
    selected_index = int(np.argmax(expected))
    selected = int(controllers[selected_index])
    ordered = np.sort(probabilities)[::-1]
    return selected, {
        "confidence": float(ordered[0]),
        "margin": float(ordered[0] - ordered[1]),
        "posterior_entropy": float(-np.sum(
            probabilities[probabilities > 0.0]
            * np.log(probabilities[probabilities > 0.0])) / np.log(4.0)),
        "best_expected_utility": float(expected[selected_index]),
        "robust_expected_utility": float(expected[robust_index]),
        "best_expected_advantage": float(
            expected[selected_index] - expected[robust_index]),
    }

