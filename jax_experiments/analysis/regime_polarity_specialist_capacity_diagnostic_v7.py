"""Stationary controller-capacity diagnostic for failed v6 policy banks."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_confirmation_v6 as source,
)


ROOT = source.ROOT
PROTOCOL_VERSION = "v7-stationary-controller-capacity-diagnostic"
ENV = source.ENV
FAMILY = source.FAMILY
MODES = source.MODES
ROLES = source.ROLES

# These are exactly the v6 policy banks that missed the frozen primary gate.
TRAINING_SEEDS = (5003, 5021, 5077)
# Fresh evaluation streams, not used by v6 estimator or router selection.
EVENT_SEEDS = (155_401, 155_417, 155_433)
EPISODES_PER_TASK = 5
MAX_EPISODE_STEPS = source.MAX_EPISODE_STEPS
MIN_DIAGONAL_WINS = 3
MIN_STATIONARY_ORACLE_GAIN = 0.10

AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_specialist_capacity_diagnostic_v7"
)
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_specialist_capacity_analysis_v7"
)
REPORT = (
    ROOT / "reports"
    / "regime_polarity_specialist_capacity_diagnostic_v7_2026-08-30.md"
)

EVENT_SCHEMA = "bapr.regime-polarity-specialist-capacity-event.v7"
AUDIT_SCHEMA = "bapr.regime-polarity-specialist-capacity-audit.v7"
ANALYSIS_SCHEMA = "bapr.regime-polarity-specialist-capacity-analysis.v7"

file_record = source.file_record
read_json = source.read_json
write_json_atomic = source.write_json_atomic
write_text_atomic = source.write_text_atomic


def require_training_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in TRAINING_SEEDS:
        raise ValueError(f"unknown capacity-diagnostic seed {seed}")
    return seed


def require_event_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in EVENT_SEEDS:
        raise ValueError(f"unknown capacity-diagnostic event seed {seed}")
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


def source_records(seed: int) -> dict[str, Any]:
    """Read the immutable bundle records captured by the completed v6 audit."""
    seed = require_training_seed(seed)
    manifest = read_json(source.audit_manifest(seed))
    records = manifest.get("source_bundles")
    if set(records or {}) != set(ROLES):
        raise ValueError(f"invalid v6 source records for seed {seed}")
    return records


def live_source_records(seed: int) -> dict[str, Any]:
    seed = require_training_seed(seed)
    records = source.source_records(seed)
    if records != source_records(seed):
        raise ValueError(f"v6 bundles changed after audit for seed {seed}")
    return records
