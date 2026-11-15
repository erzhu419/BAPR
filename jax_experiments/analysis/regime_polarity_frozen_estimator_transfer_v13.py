"""Frozen-estimator transfer audit on the confirmed v12 policy banks."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_robust_warmstart_confirmation_v12 as parent,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_system_id_v5 as estimator,
)
from jax_experiments.analysis import (
    run_regime_polarity_robust_warmstart_confirmation_audit_v12_fix1 as parent_fix,
)


ROOT = parent.ROOT
PROTOCOL_VERSION = "v13-frozen-estimator-transfer"
ENV = parent.ENV
FAMILY = parent.FAMILY
MODES = parent.MODES
TRAINING_SEEDS = parent.TRAINING_SEEDS

EVENT_SEEDS = (168_201, 168_217, 168_233)
SWITCHING_SCHEDULES = {
    168_201: (0, 2, 1, 3),
    168_217: (3, 1, 2, 0),
    168_233: (2, 3, 0, 1),
}
DELAY_STEPS = 4
DWELL_STEPS = parent.DWELL_STEPS
MAX_EPISODE_STEPS = parent.MAX_EPISODE_STEPS
SWITCHING_EPISODES = parent.SWITCHING_EPISODES

ARMS = (
    "robust_sac",
    "true_mode_safe_utility",
    "delayed_oracle_4_safe_utility",
    "posterior_map_safe_utility",
)
PRIMARY_ARM = "posterior_map_safe_utility"

MIN_HEADROOM_GAIN = 0.10
MIN_CAUSAL_RETENTION = 0.70
MIN_POSTERIOR_GAIN = 0.10
MIN_MODE_ACCURACY = 0.95
REQUIRED_SEED_PASSES = 4

AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_frozen_estimator_transfer_audit_v13"
)
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_frozen_estimator_transfer_analysis_v13"
)
REGISTRATION_ROOT = (
    ROOT / "jax_experiments" / "deployments"
    / "regime_polarity_frozen_estimator_transfer_v13"
)
REGISTRATION_PATH = REGISTRATION_ROOT / "registration.json"
PREREG_REPORT = (
    ROOT / "reports"
    / "regime_polarity_frozen_estimator_transfer_v13_preregistration_2026-09-01.md"
)
REPORT = (
    ROOT / "reports"
    / "regime_polarity_frozen_estimator_transfer_v13_2026-09-01.md"
)

EVENT_SCHEMA = "bapr.frozen-estimator-transfer-event.v13"
AUDIT_SCHEMA = "bapr.frozen-estimator-transfer-audit.v13"
ANALYSIS_SCHEMA = "bapr.frozen-estimator-transfer-analysis.v13"
REGISTRATION_SCHEMA = "bapr.frozen-estimator-transfer-registration.v13"

read_json = parent.read_json
write_json_atomic = parent.write_json_atomic
write_text_atomic = parent.write_text_atomic
file_record = parent.file_record


def require_training_seed(seed: int) -> int:
    return parent.require_training_seed(seed)


def require_event_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in EVENT_SEEDS:
        raise ValueError(f"unknown v13 event seed {seed}")
    return seed


def require_arm(arm: str) -> str:
    arm = str(arm)
    if arm not in ARMS:
        raise ValueError(f"unknown v13 arm {arm!r}")
    return arm


def switching_sequence(event_seed: int, episode: int) -> tuple[int, ...]:
    base = SWITCHING_SCHEDULES[require_event_seed(event_seed)]
    shift = int(episode) % len(base)
    return tuple(base[shift:] + base[:shift])


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


def estimator_records() -> dict[str, Any]:
    return estimator.estimator_records()


def policy_bank_records(seed: int) -> dict[str, Any]:
    seed = require_training_seed(seed)
    return {
        "source": file_record(parent.source_manifest(seed)),
        "specialists": parent.bundle_records(seed),
        "v12_audit_manifest": file_record(parent.audit_manifest(seed)),
        "v12_audit": file_record(parent.audit_result(seed)),
    }


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def registration_source_paths() -> tuple[Path, ...]:
    paths: list[Path] = [
        ROOT / "jax_experiments/analysis/regime_polarity_frozen_estimator_transfer_v13.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_frozen_estimator_transfer_audit_v13.py",
        ROOT / "jax_experiments/analysis/analyze_regime_polarity_frozen_estimator_transfer_v13.py",
        ROOT / "jax_experiments/analysis/regime_polarity_specialist_expected_action_model_v5.py",
        ROOT / "jax_experiments/analysis/regime_polarity_specialist_expected_action_system_id_v5.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_inverse_system_id_audit.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_specialist_safe_utility_audit_v8.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_specialist_expected_action_confirmation_audit_v6.py",
        ROOT / "jax_experiments/networks/executed_action_inverse.py",
        parent.REGISTRATION_PATH,
        parent_fix.FIX_REGISTRATION_PATH,
        parent.analysis_json(),
        parent.analysis_markdown(),
        estimator.MODEL_MANIFEST,
        estimator.MODEL_PATH,
        estimator.REPORT,
        PREREG_REPORT,
    ]
    for seed in TRAINING_SEEDS:
        paths.extend([
            parent.source_manifest(seed),
            parent.audit_manifest(seed),
            parent.audit_result(seed),
        ])
        paths.extend(
            parent.bundle_manifest("actor_only", seed, mode)
            for mode in MODES
        )
    return tuple(path.resolve() for path in paths)


def registration_payload() -> dict[str, Any]:
    parent.validate_registration()
    parent_fix.validate_fix_registration()
    parent_analysis = read_json(parent.analysis_json())
    if (
        parent_analysis.get("schema") != parent.ANALYSIS_SCHEMA
        or parent_analysis.get("status") != "complete"
        or parent_analysis.get("confirmed") is not True
        or int(parent_analysis.get("seed_passes", -1))
        < parent.REQUIRED_SEED_PASSES
    ):
        raise ValueError("v12 policy-bank confirmation is not complete")
    paths = registration_source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"v13 registration sources missing: {missing}")
    return {
        "schema": REGISTRATION_SCHEMA,
        "status": "registered",
        "created_before_audit": True,
        "identity": {
            "protocol_version": PROTOCOL_VERSION,
            "parent_protocol_version": parent.PROTOCOL_VERSION,
            "estimator_protocol_version": estimator.PROTOCOL_VERSION,
            "training_seeds": list(TRAINING_SEEDS),
            "event_seeds": list(EVENT_SEEDS),
            "switching_schedules": {
                str(seed): list(sequence)
                for seed, sequence in SWITCHING_SCHEDULES.items()
            },
            "arms": list(ARMS),
            "delay_steps": DELAY_STEPS,
            "dwell_steps": DWELL_STEPS,
            "max_episode_steps": MAX_EPISODE_STEPS,
            "switching_episodes": SWITCHING_EPISODES,
        },
        "decision_rule": {
            "minimum_safe_oracle_gain": MIN_HEADROOM_GAIN,
            "minimum_delay_4_headroom_retention": MIN_CAUSAL_RETENTION,
            "minimum_posterior_gain": MIN_POSTERIOR_GAIN,
            "minimum_posterior_headroom_recovery": MIN_CAUSAL_RETENTION,
            "minimum_mode_accuracy": MIN_MODE_ACCURACY,
            "required_event_wins": len(EVENT_SEEDS),
            "required_seed_passes": REQUIRED_SEED_PASSES,
            "require_zero_termination": True,
            "estimator_retraining_requires_causal_margin": True,
        },
        "data_policy": {
            "checkpoint_only": True,
            "new_training": False,
            "controllers": "frozen v12 policy-only bundles",
            "estimator": "frozen v5 expected-action estimator",
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
            raise ValueError("existing v13 registration changed")
    else:
        write_json_atomic(REGISTRATION_PATH, payload)
    return validate_registration()


def validate_registration() -> dict[str, Any]:
    if not REGISTRATION_PATH.is_file():
        raise FileNotFoundError(f"missing v13 registration: {REGISTRATION_PATH}")
    payload = read_json(REGISTRATION_PATH)
    if payload != registration_payload():
        raise ValueError("v13 registration or source closure changed")
    return payload


def assert_protocol_integrity() -> None:
    prior_events = (
        set(parent.CALIBRATION_EVENT_SEEDS)
        | set(parent.STATIONARY_HOLDOUT_EVENT_SEEDS)
        | set(parent.SWITCHING_EVENT_SEEDS)
        | set(estimator.TRAIN_EVENT_SEEDS)
        | set(estimator.VALIDATION_EVENT_SEEDS)
        | set(estimator.AUDIT_EVENT_SEEDS)
    )
    if prior_events & set(EVENT_SEEDS):
        raise ValueError("v13 reused an earlier estimator or policy event seed")
    if set(SWITCHING_SCHEDULES) != set(EVENT_SEEDS):
        raise ValueError("v13 switching schedules are incomplete")
    cycles = []
    for sequence in SWITCHING_SCHEDULES.values():
        if len(sequence) != len(MODES) or set(sequence) != set(MODES):
            raise ValueError("each v13 switching schedule must use every mode")
        cycles.append(tuple(sequence))
    if len(set(cycles)) != len(cycles):
        raise ValueError("v13 switching schedules are not distinct")


assert_protocol_integrity()
