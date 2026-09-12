"""Robust-inclusive utility routing on fresh v6 policy banks."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_specialist_capacity_diagnostic_v7 as capacity,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_confirmation_v6 as source,
)


ROOT = source.ROOT
PROTOCOL_VERSION = "v8-robust-inclusive-specialist-utility"
ENV = source.ENV
FAMILY = source.FAMILY
MODES = source.MODES
ROLES = source.ROLES
TRAINING_SEEDS = source.TRAINING_SEEDS
CALIBRATION_EVENT_SEEDS = capacity.EVENT_SEEDS
HOLDOUT_EVENT_SEEDS = (155_501, 155_517, 155_533)
CAPACITY_AUDIT_SEEDS = capacity.TRAINING_SEEDS

MAX_EPISODE_STEPS = source.MAX_EPISODE_STEPS
SWITCHING_EPISODES = source.SWITCHING_EPISODES
MIN_CALIBRATION_GAIN = 0.05
MIN_HEADROOM_GAIN = 0.10
MIN_ORACLE_RECOVERY = 0.70
MIN_PRIMARY_GAIN = 0.10
MAX_NO_HEADROOM_REGRESSION = 0.05

PRIMARY_ARM = "posterior_sticky_confirm3_safe_utility"
ARMS = (
    "robust_sac",
    "dynamic_specialist_oracle",
    "true_mode_safe_utility",
    "posterior_map_safe_utility",
    PRIMARY_ARM,
)

AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_specialist_safe_utility_audit_v8"
)
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_specialist_safe_utility_analysis_v8"
)
REPORT = (
    ROOT / "reports"
    / "regime_polarity_specialist_safe_utility_v8_2026-08-30.md"
)

CALIBRATION_SCHEMA = "bapr.regime-polarity-safe-utility-calibration.v8"
EVENT_SCHEMA = "bapr.regime-polarity-safe-utility-event.v8"
AUDIT_SCHEMA = "bapr.regime-polarity-safe-utility-audit.v8"
ANALYSIS_SCHEMA = "bapr.regime-polarity-safe-utility-analysis.v8"

file_record = source.file_record
read_json = source.read_json
write_json_atomic = source.write_json_atomic
write_text_atomic = source.write_text_atomic


def require_training_seed(seed: int) -> int:
    return source.require_training_seed(seed)


def require_holdout_event_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in HOLDOUT_EVENT_SEEDS:
        raise ValueError(f"unknown safe-utility holdout seed {seed}")
    return seed


def audit_dir(seed: int) -> Path:
    return AUDIT_ROOT / f"seed_{require_training_seed(seed)}"


def calibration_result(seed: int) -> Path:
    return audit_dir(seed) / "calibration.json"


def event_result(seed: int, event_seed: int) -> Path:
    return (
        audit_dir(seed)
        / f"event_seed_{require_holdout_event_seed(event_seed)}"
        / "results.json"
    )


def audit_manifest(seed: int) -> Path:
    return audit_dir(seed) / "audit_manifest.json"


def analysis_json() -> Path:
    return ANALYSIS_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return ANALYSIS_ROOT / "analysis.md"


def source_records(seed: int) -> dict[str, Any]:
    seed = require_training_seed(seed)
    manifest = read_json(source.audit_manifest(seed))
    records = manifest.get("source_bundles")
    if set(records or {}) != set(ROLES):
        raise ValueError(f"invalid v6 source records for seed {seed}")
    return records


def estimator_records() -> dict[str, Any]:
    return source.estimator_records()
