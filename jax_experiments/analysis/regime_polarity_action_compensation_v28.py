"""Frozen V21 action-coordinate compensation mechanism audit."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_full_state_final_confirmation_v21 as parent,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_system_id_v5 as v5_model,
)


ROOT = parent.ROOT
PROTOCOL_VERSION = "v28-polarity-action-compensation-audit"
ENV = parent.ENV
FAMILY = parent.FAMILY
MODES = parent.MODES
TRAINING_SEEDS = parent.TRAINING_SEEDS

# These streams are disjoint from all V18-V27 development and confirmation
# streams. They are frozen before any V28 rollout is inspected.
CALIBRATION_EVENT_SEEDS = (194_001, 194_017, 194_033)
STATIONARY_HOLDOUT_EVENT_SEEDS = (194_101, 194_117, 194_133)
SWITCHING_EVENT_SEEDS = (194_201, 194_217, 194_233)
SWITCHING_SCHEDULES = {
    194_201: (1, 0, 3, 2),
    194_217: (3, 2, 0, 1),
    194_233: (2, 3, 1, 0),
}

DWELL_STEPS = parent.DWELL_STEPS
MAX_EPISODE_STEPS = parent.MAX_EPISODE_STEPS
CALIBRATION_EPISODES_PER_MODE = parent.CALIBRATION_EPISODES_PER_MODE
STATIONARY_EPISODES_PER_MODE = parent.STATIONARY_EPISODES_PER_MODE
SWITCHING_EPISODES = parent.SWITCHING_EPISODES

ROBUST_ARM = "robust_sac"
NO_COMPENSATION_ARM = "reference_no_compensation"
ORACLE_COMPENSATION_ARM = "reference_true_mode_compensation"
CAUSAL_COMPENSATION_ARM = "reference_v5_map_compensation"
V21_BANK_ARM = "bapr_v21_posterior_map"
SAC5_ARM = "sac5_v21_posterior_map"
ARMS = (
    ROBUST_ARM,
    NO_COMPENSATION_ARM,
    ORACLE_COMPENSATION_ARM,
    CAUSAL_COMPENSATION_ARM,
    V21_BANK_ARM,
    SAC5_ARM,
)

REQUIRED_SEED_WINS = 4
REQUIRED_EVENT_WINS = 12
MIN_ORACLE_RECOVERY = 0.70
REQUIRED_RECOVERY_SEEDS = 4
EXACT_ACTION_ATOL = 1e-6
EXACT_TRAJECTORY_ATOL = 1e-5
EXACT_RETURN_ATOL = 1e-3

AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_action_compensation_audit_v28"
)
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_action_compensation_analysis_v28"
)
STRUCTURAL_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_action_compensation_structural_v28"
)
REGISTRATION_ROOT = (
    ROOT / "jax_experiments" / "deployments"
    / "regime_polarity_action_compensation_v28"
)
REGISTRATION_PATH = REGISTRATION_ROOT / "registration.json"
STRUCTURAL_RESULT = STRUCTURAL_ROOT / "structural_audit.json"
STRUCTURAL_MANIFEST = STRUCTURAL_ROOT / "structural_manifest.json"
ANALYSIS_JSON = ANALYSIS_ROOT / "analysis.json"
ANALYSIS_MARKDOWN = ANALYSIS_ROOT / "analysis.md"
PREREG_REPORT = (
    ROOT / "reports"
    / "regime_polarity_action_compensation_v28_preregistration_2026-09-12.md"
)
REPORT = (
    ROOT / "reports"
    / "regime_polarity_action_compensation_v28_2026-09-12.md"
)

AUDIT_SCHEMA = "bapr.regime-polarity-action-compensation-audit.v28"
STRUCTURAL_SCHEMA = "bapr.regime-polarity-action-compensation-structural.v28"
ANALYSIS_SCHEMA = "bapr.regime-polarity-action-compensation-analysis.v28"
REGISTRATION_SCHEMA = (
    "bapr.regime-polarity-action-compensation-registration.v28")

file_record = parent.file_record
read_json = parent.read_json
write_json_atomic = parent.write_json_atomic
write_text_atomic = parent.write_text_atomic
mode_gain_vectors = v5_model.mode_gain_vectors

# Existing V21 bundle APIs are reused without changing their identities.
source_bundle = parent.source_bundle
source_required_paths = parent.source_required_paths
specialist_bundle_dir = parent.specialist_bundle_dir
specialist_required_paths = parent.specialist_required_paths
sac_bundle_dir = parent.sac_bundle_dir
bundle_required_paths = parent.bundle_required_paths
frozen_policy_records = parent.frozen_policy_records
new_bundle_records = parent.new_bundle_records
require_mode = parent.require_mode
require_training_seed = parent.require_training_seed
require_replica_slot = parent.require_replica_slot
identity = parent.identity
SAC_REPLICA_SLOTS = parent.SAC_REPLICA_SLOTS
EVAL_PARAMS_NAME = parent.EVAL_PARAMS_NAME
MIN_CALIBRATION_GAIN = parent.MIN_CALIBRATION_GAIN


def require_switching_event_seed(seed: int) -> int:
    value = int(seed)
    if value not in SWITCHING_EVENT_SEEDS:
        raise ValueError(f"unknown V28 switching event seed {value}")
    return value


def switching_sequence(event_seed: int, episode: int) -> tuple[int, ...]:
    base = SWITCHING_SCHEDULES[require_switching_event_seed(event_seed)]
    shift = int(episode) % len(base)
    return tuple(base[shift:] + base[:shift])


def compensation_multiplier(
    act_dim: int, reference_mode: int, estimated_mode: int,
) -> np.ndarray:
    gains = np.asarray(mode_gain_vectors(int(act_dim)), dtype=np.float32)
    reference_mode = require_mode(reference_mode)
    estimated_mode = require_mode(estimated_mode)
    return gains[estimated_mode] * gains[reference_mode]


def compensate_action(
    action: np.ndarray, reference_mode: int, estimated_mode: int,
) -> np.ndarray:
    value = np.asarray(action)
    multiplier = compensation_multiplier(
        int(value.shape[-1]), reference_mode, estimated_mode)
    return np.clip(value * multiplier, -1.0, 1.0).astype(
        value.dtype, copy=False)


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


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def frozen_input_records(seed: int) -> dict[str, Any]:
    seed = require_training_seed(seed)
    return {
        "v21_policies": frozen_policy_records(seed),
        "v21_sac5": {
            f"replica_{slot}": file_record(
                parent.bundle_manifest("sac_replica", seed, slot))
            for slot in SAC_REPLICA_SLOTS
        },
        "v5_estimator": {
            "manifest": file_record(v5_model.MODEL_MANIFEST),
            "parameters": file_record(v5_model.MODEL_PATH),
        },
    }


def audit_required_paths(seed: int) -> tuple[Path, ...]:
    seed = require_training_seed(seed)
    paths: list[Path] = [
        REGISTRATION_PATH,
        STRUCTURAL_MANIFEST,
        *source_required_paths(seed),
        v5_model.MODEL_MANIFEST,
        v5_model.MODEL_PATH,
    ]
    for mode in MODES:
        paths.extend(specialist_required_paths(seed, mode))
    for slot in SAC_REPLICA_SLOTS:
        paths.extend(bundle_required_paths("sac_replica", seed, slot))
    return tuple(dict.fromkeys(path.resolve() for path in paths))


def registration_source_paths() -> tuple[Path, ...]:
    paths = (
        ROOT / "jax_experiments/analysis/regime_polarity_action_compensation_v28.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_action_compensation_structural_v28.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_action_compensation_audit_v28.py",
        ROOT / "jax_experiments/analysis/analyze_regime_polarity_action_compensation_v28.py",
        ROOT / "scripts/submit_regime_polarity_action_compensation_v28.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_full_state_confirmation_audit_v21.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_v5_final_comparison_audit_v18.py",
        ROOT / "jax_experiments/analysis/regime_polarity_full_state_final_confirmation_v21.py",
        ROOT / "jax_experiments/analysis/regime_polarity_specialist_expected_action_system_id_v5.py",
        ROOT / "jax_experiments/networks/executed_action_inverse.py",
        ROOT / "jax_experiments/common/checkpoint.py",
        ROOT / "jax_experiments/envs/stochastic_mode_env.py",
        ROOT / "jax_experiments/envs/brax_env.py",
        parent.REGISTRATION_PATH,
        parent.analysis_json(),
        parent.analysis_markdown(),
        parent.REPORT,
        v5_model.MODEL_MANIFEST,
        v5_model.MODEL_PATH,
        PREREG_REPORT,
    )
    return tuple(dict.fromkeys(path.resolve() for path in paths))


def registration_payload() -> dict[str, Any]:
    parent.validate_registration()
    paths = registration_source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V28 registration sources missing: {missing}")
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
            "v21_policy_bundles_unchanged": True,
            "v5_estimator_unchanged": True,
            "reference_selected_on_calibration_only": True,
            "holdout_events_not_used_for_selection": True,
            "causal_action_uses_posterior_before_current_transition": True,
            "oracle_mode_used_only_by_oracle_arm": True,
        },
        "decision_gate": {
            "exact_action_tolerance": EXACT_ACTION_ATOL,
            "exact_trajectory_tolerance": EXACT_TRAJECTORY_ATOL,
            "exact_return_tolerance": EXACT_RETURN_ATOL,
            "required_seed_wins_of_5": REQUIRED_SEED_WINS,
            "required_event_wins_of_15": REQUIRED_EVENT_WINS,
            "minimum_oracle_headroom_recovery": MIN_ORACLE_RECOVERY,
            "required_recovery_seeds": REQUIRED_RECOVERY_SEEDS,
            "primary_comparators": [NO_COMPENSATION_ARM, ROBUST_ARM],
            "bank_necessity_rule": (
                "positive paired mean and 95% interval plus at least 4/5 "
                "seed wins decides either V21-bank or compensation superiority; "
                "otherwise unresolved"
            ),
        },
        "accounting": {
            "new_training_interactions": 0,
            "reused_policy_count_bapr": 5,
            "reused_policy_count_sac5": 5,
            "v21_bapr_training_interactions": 16_800_000,
            "v21_sac5_training_interactions": 28_000_000,
            "estimator_training_reported_separately": True,
        },
        "scope": {
            "halfcheetah_direct_causal_audit": True,
            "ant_causal_estimator_available": False,
            "does_not_generalize_sign_symmetry_to_bus_or_gain_loss": True,
        },
        "sync_policy": (
            "compact policy parameters, model parameters, manifests, and JSON "
            "audits only; no replay buffers or checkpoints"
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
            raise ValueError("existing V28 registration changed")
    else:
        write_json_atomic(REGISTRATION_PATH, payload)
    return validate_registration()


def validate_registration() -> dict[str, Any]:
    if not REGISTRATION_PATH.is_file():
        raise FileNotFoundError(f"missing V28 registration: {REGISTRATION_PATH}")
    payload = read_json(REGISTRATION_PATH)
    if payload != registration_payload():
        raise ValueError("V28 registration or source closure changed")
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
        raise ValueError("V28 event splits overlap")
    previous = set(parent.CALIBRATION_EVENT_SEEDS)
    previous.update(parent.STATIONARY_HOLDOUT_EVENT_SEEDS)
    previous.update(parent.SWITCHING_EVENT_SEEDS)
    if previous & set().union(*splits):
        raise ValueError("V28 reused a V21 event stream")
    if min(set().union(*splits)) < 194_000:
        raise ValueError("V28 event namespace changed")
    if set(SWITCHING_SCHEDULES) != set(SWITCHING_EVENT_SEEDS):
        raise ValueError("V28 switching schedules are incomplete")
    if any(
        len(sequence) != len(MODES) or set(sequence) != set(MODES)
        for sequence in SWITCHING_SCHEDULES.values()
    ):
        raise ValueError("each V28 switching schedule must use every mode")
    for act_dim in (3, 6, 8):
        gains = np.asarray(mode_gain_vectors(act_dim))
        if not np.array_equal(gains * gains, np.ones_like(gains)):
            raise ValueError("actuator-polarity gains are no longer involutions")


assert_protocol_integrity()
