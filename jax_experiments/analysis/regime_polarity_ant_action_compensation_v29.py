"""Registered Ant oracle action-coordinate compensation audit."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_action_compensation_v28 as compensation_parent,
)
from jax_experiments.analysis import (
    regime_polarity_ant_full_state_headroom_v22 as policy_parent,
)
from jax_experiments.analysis import (
    run_regime_polarity_action_compensation_structural_v28 as structural_parent,
)


ROOT = policy_parent.ROOT
PROTOCOL_VERSION = "v29-ant-oracle-action-compensation-development"
ENV = policy_parent.ENV
FAMILY = policy_parent.FAMILY
MODES = policy_parent.MODES
TRAINING_SEEDS = policy_parent.TRAINING_SEEDS

# New streams, frozen before any V29 rollout is inspected.
CALIBRATION_EVENT_SEEDS = (195_001, 195_017, 195_033)
STATIONARY_HOLDOUT_EVENT_SEEDS = (195_101, 195_117, 195_133)
SWITCHING_EVENT_SEEDS = (195_201, 195_217, 195_233)
SWITCHING_SCHEDULES = {
    195_201: (3, 1, 0, 2),
    195_217: (0, 2, 1, 3),
    195_233: (2, 1, 3, 0),
}

DWELL_STEPS = policy_parent.DWELL_STEPS
MAX_EPISODE_STEPS = policy_parent.MAX_EPISODE_STEPS
CALIBRATION_EPISODES_PER_MODE = policy_parent.EPISODES_PER_TASK
STATIONARY_EPISODES_PER_MODE = policy_parent.EPISODES_PER_TASK
SWITCHING_EPISODES = policy_parent.SWITCHING_EPISODES

ROBUST_ARM = "robust_sac"
NO_COMPENSATION_ARM = "reference_no_compensation"
ORACLE_COMPENSATION_ARM = "reference_true_mode_compensation"
DYNAMIC_BANK_ARM = "v22_dynamic_specialist_oracle"
ARMS = (
    ROBUST_ARM,
    NO_COMPENSATION_ARM,
    ORACLE_COMPENSATION_ARM,
    DYNAMIC_BANK_ARM,
)

MIN_SWITCHING_GAIN_OVER_ROBUST = 0.10
REQUIRED_SEED_WINS = len(TRAINING_SEEDS)
REQUIRED_EVENT_WINS = len(TRAINING_SEEDS) * len(SWITCHING_EVENT_SEEDS)
EXACT_ACTION_ATOL = compensation_parent.EXACT_ACTION_ATOL
EXACT_RETURN_ATOL = compensation_parent.EXACT_RETURN_ATOL

AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_ant_action_compensation_audit_v29"
)
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_ant_action_compensation_analysis_v29"
)
REGISTRATION_ROOT = (
    ROOT / "jax_experiments" / "deployments"
    / "regime_polarity_ant_action_compensation_v29"
)
REGISTRATION_PATH = REGISTRATION_ROOT / "registration.json"
ANALYSIS_JSON = ANALYSIS_ROOT / "analysis.json"
ANALYSIS_MARKDOWN = ANALYSIS_ROOT / "analysis.md"
PREREG_REPORT = (
    ROOT / "reports"
    / "regime_polarity_ant_action_compensation_v29_preregistration_2026-09-12.md"
)
REPORT = (
    ROOT / "reports"
    / "regime_polarity_ant_action_compensation_v29_2026-09-12.md"
)

AUDIT_SCHEMA = "bapr.regime-polarity-ant-action-compensation-audit.v29"
ANALYSIS_SCHEMA = "bapr.regime-polarity-ant-action-compensation-analysis.v29"
REGISTRATION_SCHEMA = (
    "bapr.regime-polarity-ant-action-compensation-registration.v29")

file_record = policy_parent.file_record
read_json = policy_parent.read_json
write_json_atomic = policy_parent.write_json_atomic
write_text_atomic = policy_parent.write_text_atomic
mode_gain_vectors = compensation_parent.mode_gain_vectors
source_bundle = policy_parent.source_bundle
source_required_paths = policy_parent.source_required_paths


def require_training_seed(seed: int) -> int:
    return policy_parent.require_training_seed(seed)


def require_mode(mode: int) -> int:
    return policy_parent.require_mode(mode)


def require_switching_event_seed(seed: int) -> int:
    value = int(seed)
    if value not in SWITCHING_EVENT_SEEDS:
        raise ValueError(f"unknown V29 switching event seed {value}")
    return value


def switching_sequence(event_seed: int, episode: int) -> tuple[int, ...]:
    base = SWITCHING_SCHEDULES[require_switching_event_seed(event_seed)]
    shift = int(episode) % len(base)
    return tuple(base[shift:] + base[:shift])


def compensate_action(
    action: np.ndarray, reference_mode: int, actual_mode: int,
) -> np.ndarray:
    value = np.asarray(action)
    gains = np.asarray(mode_gain_vectors(int(value.shape[-1])), dtype=np.float32)
    multiplier = gains[require_mode(actual_mode)] * gains[
        require_mode(reference_mode)]
    return np.clip(value * multiplier, -1.0, 1.0).astype(
        value.dtype, copy=False)


def specialist_bundle(mode: int, seed: int) -> Path:
    return policy_parent.bundle_dir(
        policy_parent.CONTROL_VARIANT,
        require_training_seed(seed),
        require_mode(mode),
    )


def specialist_required_paths(seed: int, mode: int) -> tuple[Path, ...]:
    return policy_parent.bundle_required_paths(
        policy_parent.CONTROL_VARIANT,
        require_training_seed(seed),
        require_mode(mode),
    )


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
    return {
        "robust_source": file_record(policy_parent.source_manifest(seed)),
        "v22_specialists": {
            str(mode): file_record(policy_parent.bundle_manifest(
                policy_parent.CONTROL_VARIANT, seed, mode))
            for mode in MODES
        },
    }


def audit_required_paths(seed: int) -> tuple[Path, ...]:
    seed = require_training_seed(seed)
    paths: list[Path] = [
        REGISTRATION_PATH,
        structural_parent.protocol.STRUCTURAL_MANIFEST,
        *source_required_paths(seed),
    ]
    for mode in MODES:
        paths.extend(specialist_required_paths(seed, mode))
    return tuple(dict.fromkeys(path.resolve() for path in paths))


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def registration_source_paths() -> tuple[Path, ...]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "jax_experiments/analysis/run_regime_polarity_ant_action_compensation_audit_v29.py",
        ROOT / "jax_experiments/analysis/analyze_regime_polarity_ant_action_compensation_v29.py",
        ROOT / "scripts/submit_regime_polarity_ant_action_compensation_v29.py",
        ROOT / "jax_experiments/analysis/regime_polarity_ant_full_state_headroom_v22.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_ant_full_state_audit_v22.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_ant_full_state_specialist_v22.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_specialist_stability_audit_v19.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_robust_warmstart_specialist_audit_v11.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_specialist_safe_utility_audit_v8.py",
        ROOT / "jax_experiments/analysis/regime_polarity_action_compensation_v28.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_action_compensation_structural_v28.py",
        ROOT / "jax_experiments/common/checkpoint.py",
        ROOT / "jax_experiments/train.py",
        ROOT / "jax_experiments/envs/stochastic_mode_env.py",
        ROOT / "jax_experiments/envs/brax_env.py",
        policy_parent.REGISTRATION_PATH,
        policy_parent.analysis_json(),
        structural_parent.protocol.STRUCTURAL_MANIFEST,
        PREREG_REPORT,
    )
    return tuple(dict.fromkeys(path.resolve() for path in paths))


def registration_payload() -> dict[str, Any]:
    policy_parent.validate_registration()
    paths = registration_source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V29 registration sources missing: {missing}")
    structural = read_json(structural_parent.protocol.STRUCTURAL_MANIFEST)
    if structural.get("status") != "complete":
        raise ValueError("V28 structural compensation audit is not complete")
    return {
        "schema": REGISTRATION_SCHEMA,
        "status": "registered",
        "created_before_evaluation": True,
        "identity": {
            "protocol_version": PROTOCOL_VERSION,
            "environment": ENV,
            "family": FAMILY,
            "training_seeds": list(TRAINING_SEEDS),
            "modes": list(MODES),
            "calibration_event_seeds": list(CALIBRATION_EVENT_SEEDS),
            "stationary_holdout_event_seeds": list(
                STATIONARY_HOLDOUT_EVENT_SEEDS),
            "switching_event_seeds": list(SWITCHING_EVENT_SEEDS),
            "switching_schedules": {
                str(seed): list(sequence)
                for seed, sequence in SWITCHING_SCHEDULES.items()
            },
            "arms": list(ARMS),
        },
        "frozen_boundary": {
            "no_parameter_training": True,
            "v22_policy_bundles_unchanged": True,
            "reference_selected_on_calibration_only": True,
            "holdout_events_not_used_for_selection": True,
            "true_mode_used_only_by_oracle_arms": True,
            "no_ant_estimator_is_trained_or_evaluated": True,
        },
        "decision_gate": {
            "exact_action_tolerance": EXACT_ACTION_ATOL,
            "exact_return_tolerance": EXACT_RETURN_ATOL,
            "minimum_switching_gain_over_robust_per_seed": (
                MIN_SWITCHING_GAIN_OVER_ROBUST),
            "required_seed_wins": REQUIRED_SEED_WINS,
            "required_event_wins": REQUIRED_EVENT_WINS,
            "oracle_stationary_and_switching_termination_rate": 0.0,
            "estimator_authorization": (
                "only if oracle compensation is exact, beats robust and the "
                "uncompensated reference on all seeds and switching events, "
                "gains at least 10 percent over robust for every seed, beats "
                "robust stationary on every seed, and has zero stationary and "
                "switching termination"
            ),
            "dynamic_bank_comparison": "diagnostic, not an estimator gate",
        },
        "accounting": {
            "new_training_interactions": 0,
            "reused_v22_policy_count_per_seed": 5,
            "development_policy_seeds": len(TRAINING_SEEDS),
        },
        "scope": {
            "development_upper_bound_only": True,
            "ant_causal_estimator_available": False,
            "does_not_confirm_cross_environment_generalization": True,
        },
        "sync_policy": (
            "compact policy bundles, manifests, and JSON audits only; no "
            "checkpoints, replay buffers, or trajectory arrays"
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
            raise ValueError("existing V29 registration changed")
    else:
        write_json_atomic(REGISTRATION_PATH, payload)
    return validate_registration()


def validate_registration() -> dict[str, Any]:
    if not REGISTRATION_PATH.is_file():
        raise FileNotFoundError(f"missing V29 registration: {REGISTRATION_PATH}")
    payload = read_json(REGISTRATION_PATH)
    if payload != registration_payload():
        raise ValueError("V29 registration or source closure changed")
    return payload


def assert_protocol_integrity() -> None:
    splits = (
        set(CALIBRATION_EVENT_SEEDS),
        set(STATIONARY_HOLDOUT_EVENT_SEEDS),
        set(SWITCHING_EVENT_SEEDS),
    )
    if any(
        left & right
        for index, left in enumerate(splits)
        for right in splits[index + 1:]
    ):
        raise ValueError("V29 event splits overlap")
    previous = {
        *policy_parent.CALIBRATION_EVENT_SEEDS,
        *policy_parent.STATIONARY_HOLDOUT_EVENT_SEEDS,
        *policy_parent.SWITCHING_EVENT_SEEDS,
        *compensation_parent.CALIBRATION_EVENT_SEEDS,
        *compensation_parent.STATIONARY_HOLDOUT_EVENT_SEEDS,
        *compensation_parent.SWITCHING_EVENT_SEEDS,
    }
    if previous & set().union(*splits):
        raise ValueError("V29 reused a V22 or V28 event stream")
    if set(SWITCHING_SCHEDULES) != set(SWITCHING_EVENT_SEEDS):
        raise ValueError("V29 switching schedules are incomplete")
    if any(
        len(sequence) != len(MODES) or set(sequence) != set(MODES)
        for sequence in SWITCHING_SCHEDULES.values()
    ):
        raise ValueError("each V29 switching schedule must use every mode")
    gains = np.asarray(mode_gain_vectors(8), dtype=np.float32)
    if not np.array_equal(gains * gains, np.ones_like(gains)):
        raise ValueError("Ant polarity modes are no longer involutions")


assert_protocol_integrity()
