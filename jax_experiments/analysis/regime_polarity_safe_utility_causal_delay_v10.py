"""Frozen causal-delay diagnostic for the v9 safe-utility policy banks."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_specialist_safe_utility_confirmation_v9 as parent,
)


ROOT = parent.ROOT
PROTOCOL_VERSION = "v10-safe-utility-causal-delay-diagnostic"
TRAINING_SEEDS = parent.TRAINING_SEEDS
EVENT_SEEDS = (156_201, 156_217, 156_233)
DELAYS = (1, 2, 4, 8)
MODES = parent.MODES
ROLES = parent.ROLES
ENV = parent.ENV
FAMILY = parent.FAMILY
DWELL_STEPS = parent.DWELL_STEPS
MAX_EPISODE_STEPS = parent.MAX_EPISODE_STEPS
SWITCHING_EPISODES = parent.SWITCHING_EPISODES

MIN_HEADROOM_GAIN = 0.10
MIN_CAUSAL_RETENTION = 0.70
MIN_REPRODUCIBLE_SEEDS = 4

BASE_ARMS = (
    "robust_sac",
    "true_mode_safe_utility",
    "posterior_map_safe_utility",
)
STALE_ARMS = tuple(f"stale_delay_{delay}" for delay in DELAYS)
ROBUST_HANDOFF_ARMS = tuple(
    f"robust_handoff_{delay}" for delay in DELAYS)
ARMS = (*BASE_ARMS, *STALE_ARMS, *ROBUST_HANDOFF_ARMS)

AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_safe_utility_causal_delay_audit_v10"
)
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_safe_utility_causal_delay_analysis_v10"
)
REGISTRATION_ROOT = (
    ROOT / "jax_experiments" / "deployments"
    / "regime_polarity_safe_utility_causal_delay_v10"
)
REGISTRATION_PATH = REGISTRATION_ROOT / "registration.json"
PREREG_REPORT = (
    ROOT / "reports"
    / "regime_polarity_safe_utility_causal_delay_v10_preregistration_2026-08-31.md"
)
REPORT = (
    ROOT / "reports"
    / "regime_polarity_safe_utility_causal_delay_v10_2026-08-31.md"
)

EVENT_SCHEMA = "bapr.safe-utility-causal-delay-event.v10"
AUDIT_SCHEMA = "bapr.safe-utility-causal-delay-audit.v10"
ANALYSIS_SCHEMA = "bapr.safe-utility-causal-delay-analysis.v10"
REGISTRATION_SCHEMA = "bapr.safe-utility-causal-delay-registration.v10"

read_json = parent.read_json
write_json_atomic = parent.write_json_atomic
write_text_atomic = parent.write_text_atomic
file_record = parent.file_record


def require_training_seed(seed: int) -> int:
    return parent.require_training_seed(seed)


def require_event_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in EVENT_SEEDS:
        raise ValueError(f"unknown v10 event seed {seed}")
    return seed


def require_arm(arm: str) -> str:
    arm = str(arm)
    if arm not in ARMS:
        raise ValueError(f"unknown v10 arm {arm!r}")
    return arm


def delay_for_arm(arm: str) -> int:
    arm = require_arm(arm)
    for prefix in ("stale_delay_", "robust_handoff_"):
        if arm.startswith(prefix):
            return int(arm.removeprefix(prefix))
    return 0


def audit_dir(seed: int) -> Path:
    return AUDIT_ROOT / f"seed_{require_training_seed(seed)}"


def event_result(seed: int, event_seed: int) -> Path:
    return (
        audit_dir(seed)
        / f"event_seed_{require_event_seed(event_seed)}"
        / "results.json"
    )


def audit_manifest(seed: int) -> Path:
    return audit_dir(seed) / "audit_manifest.json"


def analysis_json() -> Path:
    return ANALYSIS_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return ANALYSIS_ROOT / "analysis.md"


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def registration_source_paths() -> tuple[Path, ...]:
    paths = (
        ROOT / "jax_experiments/analysis/regime_polarity_safe_utility_causal_delay_v10.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_safe_utility_causal_delay_audit_v10.py",
        ROOT / "jax_experiments/analysis/analyze_regime_polarity_safe_utility_causal_delay_v10.py",
        parent.REGISTRATION_PATH,
        PREREG_REPORT,
    )
    return tuple(path.resolve() for path in paths)


def registration_payload() -> dict[str, Any]:
    parent.validate_registration()
    paths = registration_source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"v10 registration sources missing: {missing}")
    return {
        "schema": REGISTRATION_SCHEMA,
        "status": "registered",
        "created_before_audit": True,
        "identity": {
            "protocol_version": PROTOCOL_VERSION,
            "parent_protocol_version": parent.PROTOCOL_VERSION,
            "training_seeds": list(TRAINING_SEEDS),
            "event_seeds": list(EVENT_SEEDS),
            "delays": list(DELAYS),
            "arms": list(ARMS),
        },
        "decision_rule": {
            "minimum_safe_oracle_gain": MIN_HEADROOM_GAIN,
            "minimum_delay_4_headroom_retention": MIN_CAUSAL_RETENTION,
            "minimum_reproducible_seeds": MIN_REPRODUCIBLE_SEEDS,
            "estimator_retraining_requires_causal_margin": True,
        },
        "data_policy": {
            "checkpoint_only": True,
            "new_training": False,
            "result_payload": "json_and_markdown_only",
        },
        "source_records": {
            _relative(path): file_record(path) for path in paths
        },
    }


def create_registration() -> dict[str, Any]:
    payload = registration_payload()
    REGISTRATION_ROOT.mkdir(parents=True, exist_ok=True)
    if REGISTRATION_PATH.is_file():
        if read_json(REGISTRATION_PATH) != payload:
            raise ValueError("existing v10 registration changed")
    else:
        write_json_atomic(REGISTRATION_PATH, payload)
    return validate_registration()


def validate_registration() -> dict[str, Any]:
    if not REGISTRATION_PATH.is_file():
        raise FileNotFoundError(f"missing v10 registration: {REGISTRATION_PATH}")
    payload = read_json(REGISTRATION_PATH)
    if payload != registration_payload():
        raise ValueError("v10 registration or source closure changed")
    return payload


def assert_split_integrity() -> None:
    prior = set(parent.CALIBRATION_EVENT_SEEDS) | set(parent.HOLDOUT_EVENT_SEEDS)
    if len(EVENT_SEEDS) != len(set(EVENT_SEEDS)):
        raise ValueError("v10 event seeds must be distinct")
    if prior & set(EVENT_SEEDS):
        raise ValueError("v10 reused a v9 event seed")


assert_split_integrity()
