"""Privileged capacity protocol for replacing only hard-router fallback."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from jax_experiments.analysis import (
    bapr_v3_posterior_discrete_selector as discrete,
)


ROOT = discrete.ROOT
FAMILY = discrete.FAMILY
ENV = discrete.ENV
DECISION_VARIANT = discrete.DECISION_VARIANT
DEVELOPMENT_EVENT_SEEDS = discrete.DEVELOPMENT_EVENT_SEEDS
SEALED_CONFIRMATION_EVENT_SEEDS = discrete.SEALED_CONFIRMATION_EVENT_SEEDS

MIN_FULL_CYCLE_MEAN_GAIN = 50.0
TERMINATION_RATE_MARGIN = 0.0

RESULT_ROOT = (
    ROOT / "jax_experiments"
    / "results_bapr_v3_structured_channel_fallback_oracle_v1"
)
ANALYSIS_ROOT = RESULT_ROOT / "analysis"
ANALYSIS_JSON = ANALYSIS_ROOT / "summary.json"
ANALYSIS_REPORT = ANALYSIS_ROOT / "report.md"

GROUP_SCHEMA = "bapr.v3-fallback-oracle-group.v1"
ANALYSIS_SCHEMA = "bapr.v3-fallback-oracle-analysis.v1"


def configure() -> None:
    discrete.configure()


def group_path(event_seed: int) -> Path:
    if int(event_seed) not in DEVELOPMENT_EVENT_SEEDS:
        raise ValueError(f"unregistered fallback-oracle seed {event_seed}")
    return RESULT_ROOT / f"event_seed_{int(event_seed)}" / "group.json"


def file_record(path: Path):
    return discrete.file_record(path)


def write_json_atomic(path: Path, payload: Any) -> None:
    discrete.write_json_atomic(path, payload)


def select_controller(
    hard_selected: int,
    hard_fallback: bool,
    physics_mode: int,
    oracle_map: Iterable[int],
) -> int:
    """Preserve committed hard decisions; privilege only fallback steps."""
    mapping = tuple(int(value) for value in oracle_map)
    if len(mapping) != 4 or not 0 <= int(physics_mode) < 4:
        raise ValueError("fallback oracle requires a four-mode controller map")
    return int(mapping[int(physics_mode)] if hard_fallback else hard_selected)

