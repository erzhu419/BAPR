"""Frozen protocol for the multi-seed regime-adapter confirmation."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from jax_experiments.analysis import regime_adapter_fork as development


ROOT = Path(os.environ.get(
    "BAPR_WORKSPACE_ROOT", Path(__file__).resolve().parents[2])).resolve()
PROTOCOL_VERSION = "v1"
ENV = development.ENV
FAMILY = development.FAMILY
MODES = development.MODES
DELTA = 0.5
CONTROLLER_MAP = MODES

DEVELOPMENT_SEED = 8
HOLDOUT_SEEDS = (16, 24, 32, 40)
TRAINING_SEEDS = (DEVELOPMENT_SEED,) + HOLDOUT_SEEDS
EVENT_SEEDS = (76100, 76200, 76300, 76400, 76500)

AUDIT_ROOT = (
    ROOT / "jax_experiments" / "results_regime_adapter_confirmation_v1")
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_adapter_confirmation_analysis_v1")
PROTOCOL_REPORT = (
    ROOT / "reports" / "regime_adapter_confirmation_protocol_2026-07-23.md")

AUDIT_MANIFEST_NAME = "audit_manifest.json"
AUDIT_SCHEMA = "bapr.regime-adapter-confirmation-audit.v1"
ANALYSIS_SCHEMA = "bapr.regime-adapter-confirmation-analysis.v1"


def require_training_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in TRAINING_SEEDS:
        raise ValueError(f"unknown confirmation training seed {seed}")
    return seed


def require_event_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in EVENT_SEEDS:
        raise ValueError(f"unknown confirmation event seed {seed}")
    return seed


def audit_dir(seed: int, event_seed: int) -> Path:
    return (
        AUDIT_ROOT / f"seed_{require_training_seed(seed)}"
        / f"event_{require_event_seed(event_seed)}")


def audit_manifest(seed: int, event_seed: int) -> Path:
    return audit_dir(seed, event_seed) / AUDIT_MANIFEST_NAME


def analysis_json() -> Path:
    return ANALYSIS_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return ANALYSIS_ROOT / "analysis.md"


def identity(seed: int, event_seed: int) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "env": ENV,
        "family": FAMILY,
        "training_seed": require_training_seed(seed),
        "event_seed": require_event_seed(event_seed),
        "residual_delta": DELTA,
        "controller_map": list(CONTROLLER_MAP),
        "selection_role": (
            "development" if int(seed) == DEVELOPMENT_SEED else "holdout"),
    }


def training_bundle_dirs(seed: int) -> tuple[Path, ...]:
    seed = require_training_seed(seed)
    return development.all_training_bundle_dirs(seed, DELTA)


def required_training_paths(seed: int) -> tuple[Path, ...]:
    return tuple(
        path
        for directory in training_bundle_dirs(seed)
        for path in development.required_bundle_paths(directory)
    )


read_json = development.read_json
write_json_atomic = development.write_json_atomic
file_record = development.file_record
