"""Registered Ant finite-horizon paired branch-risk diagnostic."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_ant_action_compensation_v29 as parent,
)


ROOT = parent.ROOT
PROTOCOL_VERSION = "v30-ant-finite-horizon-branch-risk-development"
ENV = parent.ENV
FAMILY = parent.FAMILY
MODES = parent.MODES
TRAINING_SEEDS = parent.TRAINING_SEEDS

# Frozen before any V30 branch rollout is inspected.
SOURCE_EVENT_SEEDS = (203_001, 203_017, 203_033)
BRANCH_KEY_BASE = 211_003
SOURCE_MAX_STEPS = parent.MAX_EPISODE_STEPS
FIXED_SNAPSHOT_STEPS = (0, 25, 50, 100, 200, 400, 600, 800)
PRETERMINATION_OFFSETS = (1, 2, 4, 8, 16, 32, 64)
RISK_HORIZONS = (1, 2, 4, 8, 16, 32, 64, 128, 250)
MAX_RISK_HORIZON = max(RISK_HORIZONS)
CONTINUATIONS_PER_SNAPSHOT = 8

CANDIDATE_ARM = "selected_reference_true_mode_compensation"
FALLBACK_ARM = "robust_sac_full_continuation"
ARMS = (CANDIDATE_ARM, FALLBACK_ARM)

# This is a headroom gate for a later risk-model study, not a safety claim.
MIN_UNIQUE_CANDIDATE_FAILURES_PER_INFORMATIVE_SEED = 4
MIN_INFORMATIVE_SEEDS = 2
MIN_ABSOLUTE_RISK_REDUCTION = 0.02
MIN_RESCUE_FRACTION = 0.50
MAX_HARM_FRACTION = 0.10
MIN_MODE_WINS = 3
EXACT_BRANCH_RETURN_ATOL = 1e-5

AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_ant_branch_risk_audit_v30"
)
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_ant_branch_risk_analysis_v30"
)
REGISTRATION_ROOT = (
    ROOT / "jax_experiments" / "deployments"
    / "regime_polarity_ant_branch_risk_v30"
)
REGISTRATION_PATH = REGISTRATION_ROOT / "registration.json"
ANALYSIS_JSON = ANALYSIS_ROOT / "analysis.json"
ANALYSIS_MARKDOWN = ANALYSIS_ROOT / "analysis.md"
PREREG_REPORT = (
    ROOT / "reports"
    / "regime_polarity_ant_branch_risk_v30_preregistration_2026-09-12.md"
)
REPORT = (
    ROOT / "reports"
    / "regime_polarity_ant_branch_risk_v30_2026-09-12.md"
)
DIAGNOSIS_NOTE = ROOT / "markdown" / "GPT_diagnosis.md"

AUDIT_SCHEMA = "bapr.regime-polarity-ant-branch-risk-audit.v30"
ANALYSIS_SCHEMA = "bapr.regime-polarity-ant-branch-risk-analysis.v30"
REGISTRATION_SCHEMA = "bapr.regime-polarity-ant-branch-risk-registration.v30"

file_record = parent.file_record
read_json = parent.read_json
write_json_atomic = parent.write_json_atomic
write_text_atomic = parent.write_text_atomic
source_bundle = parent.source_bundle
source_required_paths = parent.source_required_paths
specialist_bundle = parent.specialist_bundle
specialist_required_paths = parent.specialist_required_paths


def require_training_seed(seed: int) -> int:
    return parent.require_training_seed(seed)


def require_mode(mode: int) -> int:
    return parent.require_mode(mode)


def require_source_event_seed(seed: int) -> int:
    value = int(seed)
    if value not in SOURCE_EVENT_SEEDS:
        raise ValueError(f"unknown V30 source event seed {value}")
    return value


def selected_reference_mode(seed: int) -> int:
    seed = require_training_seed(seed)
    payload = read_json(parent.audit_result(seed))
    return require_mode(payload["calibration"]["selected_reference_mode"])


def audit_dir(seed: int) -> Path:
    return AUDIT_ROOT / f"seed_{require_training_seed(seed)}"


def audit_result(seed: int) -> Path:
    return audit_dir(seed) / "audit.json"


def audit_manifest(seed: int) -> Path:
    return audit_dir(seed) / "audit_manifest.json"


def analysis_json() -> Path:
    return ANALYSIS_JSON


def analysis_markdown() -> Path:
    return ANALYSIS_MARKDOWN


def frozen_input_records(seed: int) -> dict[str, Any]:
    seed = require_training_seed(seed)
    reference_mode = selected_reference_mode(seed)
    return {
        "v29_audit_manifest": file_record(parent.audit_manifest(seed)),
        "v29_audit": file_record(parent.audit_result(seed)),
        "robust_source_manifest": file_record(parent.policy_parent.source_manifest(seed)),
        "selected_specialist_manifest": file_record(
            parent.policy_parent.bundle_manifest(
                parent.policy_parent.CONTROL_VARIANT, seed, reference_mode)),
    }


def audit_required_paths(seed: int) -> tuple[Path, ...]:
    seed = require_training_seed(seed)
    reference_mode = selected_reference_mode(seed)
    return tuple(dict.fromkeys(path.resolve() for path in (
        REGISTRATION_PATH,
        parent.audit_manifest(seed),
        parent.audit_result(seed),
        *source_required_paths(seed),
        *specialist_required_paths(seed, reference_mode),
    )))


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def registration_source_paths() -> tuple[Path, ...]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "jax_experiments/analysis/run_regime_polarity_ant_branch_risk_audit_v30.py",
        ROOT / "jax_experiments/analysis/analyze_regime_polarity_ant_branch_risk_v30.py",
        ROOT / "scripts/submit_regime_polarity_ant_branch_risk_v30.py",
        ROOT / "jax_experiments/analysis/regime_polarity_ant_action_compensation_v29.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_ant_action_compensation_audit_v29.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_specialist_stability_audit_v19.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_specialist_safe_utility_audit_v8.py",
        ROOT / "jax_experiments/envs/stochastic_mode_env.py",
        ROOT / "jax_experiments/envs/brax_env.py",
        ROOT / "jax_experiments/train.py",
        parent.REGISTRATION_PATH,
        parent.analysis_json(),
        PREREG_REPORT,
        DIAGNOSIS_NOTE,
    )
    return tuple(dict.fromkeys(path.resolve() for path in paths))


def registration_payload() -> dict[str, Any]:
    parent.validate_registration()
    for seed in TRAINING_SEEDS:
        from jax_experiments.analysis import (
            run_regime_polarity_ant_action_compensation_audit_v29 as audit,
        )
        audit.validate_audit(seed)
    paths = registration_source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V30 registration sources missing: {missing}")
    return {
        "schema": REGISTRATION_SCHEMA,
        "status": "registered",
        "created_before_evaluation": True,
        "identity": {
            "protocol_version": PROTOCOL_VERSION,
            "environment": ENV,
            "family": FAMILY,
            "training_seeds": list(TRAINING_SEEDS),
            "source_event_seeds": list(SOURCE_EVENT_SEEDS),
            "selected_reference_modes": {
                str(seed): selected_reference_mode(seed)
                for seed in TRAINING_SEEDS
            },
            "modes": list(MODES),
            "arms": list(ARMS),
            "risk_horizons": list(RISK_HORIZONS),
        },
        "frozen_boundary": {
            "no_parameter_training": True,
            "v29_reference_selection_reused_without_reselection": True,
            "new_event_streams": True,
            "same_simulator_state_for_each_paired_branch": True,
            "common_action_noise_within_each_pair": True,
            "candidate_and_fallback_are_full_continuations": True,
            "physical_termination_stops_each_branch": True,
            "horizon_end_is_truncation_not_termination": True,
            "no_risk_critic_or_estimator_is_trained": True,
        },
        "snapshot_design": {
            "behavior": CANDIDATE_ARM,
            "source_max_steps": SOURCE_MAX_STEPS,
            "fixed_steps": list(FIXED_SNAPSHOT_STEPS),
            "pretermination_offsets": list(PRETERMINATION_OFFSETS),
            "continuations_per_snapshot": CONTINUATIONS_PER_SNAPSHOT,
            "actual_mode_interventions": list(MODES),
        },
        "decision_gate": {
            "primary_horizon": MAX_RISK_HORIZON,
            "minimum_unique_candidate_failures_per_informative_seed": (
                MIN_UNIQUE_CANDIDATE_FAILURES_PER_INFORMATIVE_SEED),
            "minimum_informative_seeds": MIN_INFORMATIVE_SEEDS,
            "minimum_absolute_candidate_minus_fallback_risk": (
                MIN_ABSOLUTE_RISK_REDUCTION),
            "minimum_rescue_fraction_given_candidate_failure": (
                MIN_RESCUE_FRACTION),
            "maximum_harm_fraction_given_candidate_survival": (
                MAX_HARM_FRACTION),
            "minimum_actual_mode_wins": MIN_MODE_WINS,
            "authorization": (
                "train a new finite-horizon continuation-risk model only if "
                "full robust continuation has preregistered rescue headroom"
            ),
        },
        "scope": {
            "development_mechanism_audit": True,
            "does_not_reopen_v29_estimator_gate": True,
            "does_not_claim_a_safety_guarantee": True,
            "paired_mode_interventions_are_not_independent_policy_seeds": True,
        },
        "sync_policy": (
            "compact JSON audits and aggregate only; simulator states, branch "
            "trajectories, checkpoints, and replay buffers remain unsynchronized"
        ),
        "source_records": {
            _relative(path): file_record(path) for path in paths
        },
    }


def create_registration() -> dict[str, Any]:
    payload = registration_payload()
    REGISTRATION_ROOT.mkdir(parents=True, exist_ok=True)
    if REGISTRATION_PATH.is_file():
        if read_json(REGISTRATION_PATH) != payload:
            raise ValueError("existing V30 registration changed")
    else:
        write_json_atomic(REGISTRATION_PATH, payload)
    return validate_registration()


def validate_registration() -> dict[str, Any]:
    if not REGISTRATION_PATH.is_file():
        raise FileNotFoundError(f"missing V30 registration: {REGISTRATION_PATH}")
    payload = read_json(REGISTRATION_PATH)
    if payload != registration_payload():
        raise ValueError("V30 registration or source closure changed")
    return payload


def assert_protocol_integrity() -> None:
    previous_events = {
        *parent.CALIBRATION_EVENT_SEEDS,
        *parent.STATIONARY_HOLDOUT_EVENT_SEEDS,
        *parent.SWITCHING_EVENT_SEEDS,
        *parent.policy_parent.CALIBRATION_EVENT_SEEDS,
        *parent.policy_parent.STATIONARY_HOLDOUT_EVENT_SEEDS,
        *parent.policy_parent.SWITCHING_EVENT_SEEDS,
    }
    if previous_events & set(SOURCE_EVENT_SEEDS):
        raise ValueError("V30 source events overlap V22/V29 evidence")
    if sorted(RISK_HORIZONS) != list(RISK_HORIZONS):
        raise ValueError("V30 risk horizons must be increasing")
    if any(step >= SOURCE_MAX_STEPS for step in FIXED_SNAPSHOT_STEPS):
        raise ValueError("V30 fixed snapshot exceeds source horizon")


assert_protocol_integrity()
