"""Independent confirmation of frozen posterior-MAP safe utility routing."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_corrected_baselines_v2 as corrected,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_confirmation_v6 as source,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_router_diagnostic_v2 as diagnostic,
)


ROOT = source.ROOT
PROTOCOL_VERSION = "v9-independent-posterior-map-safe-utility-confirmation"
ENV = source.ENV
FAMILY = source.FAMILY
MODES = source.MODES
ROLES = source.ROLES
BASELINE_METHODS = ("escp_recurrent", "resac_b0")
TRAINED_METHODS = BASELINE_METHODS

# This cohort and both event splits were absent from the protocol source tree
# when v9 was frozen. Calibration streams select only the per-mode utility map;
# switching holdouts are never used for selection.
TRAINING_SEEDS = (61_003, 61_021, 61_039, 61_057, 61_079)
CALIBRATION_EVENT_SEEDS = (156_001, 156_017, 156_033)
HOLDOUT_EVENT_SEEDS = (156_101, 156_117, 156_133)
EVENT_SEEDS = HOLDOUT_EVENT_SEEDS
CAPACITY_AUDIT_SEEDS: tuple[int, ...] = ()

MAX_ITERS = source.MAX_ITERS
FINAL_ITERATION = source.FINAL_ITERATION
SAMPLES_PER_ITER = source.SAMPLES_PER_ITER
UPDATES_PER_ITER = source.UPDATES_PER_ITER
START_TRAIN_STEPS = source.START_TRAIN_STEPS
FINAL_TOTAL_STEPS = source.FINAL_TOTAL_STEPS
FINAL_UPDATE_COUNT = source.FINAL_UPDATE_COUNT
DWELL_STEPS = source.DWELL_STEPS
MAX_EPISODE_STEPS = source.MAX_EPISODE_STEPS
SWITCHING_EPISODES = source.SWITCHING_EPISODES
AUDIT_SWITCHING_EPISODES = SWITCHING_EPISODES
EPISODES_PER_TASK = 3
AUDIT_EPISODES_PER_TASK = EPISODES_PER_TASK
WARMUP_STEPS = diagnostic.WARMUP_STEPS

ESCP_CONFIG = dict(corrected.ESCP_CONFIG)
RESAC_CONFIG = dict(corrected.RESAC_CONFIG)

MIN_CALIBRATION_GAIN = 0.05
MIN_HEADROOM_GAIN = 0.10
MIN_ORACLE_RECOVERY = 0.70
MIN_PRIMARY_GAIN = 0.10
MAX_NO_HEADROOM_REGRESSION = 0.05
MIN_BASELINE_SEED_WINS = 4

PRIMARY_ARM = "posterior_map_safe_utility"
BAPR_ARMS = (
    "robust_sac",
    "dynamic_specialist_oracle",
    "true_mode_safe_utility",
    PRIMARY_ARM,
)
ARMS = (*BAPR_ARMS, *BASELINE_METHODS)

CONTROLLER_RUN_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_safe_utility_confirmation_v9"
    / "controllers"
)
BASELINE_RUN_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_safe_utility_confirmation_v9"
    / "baselines"
)
BUNDLE_ROOT = (
    ROOT / "jax_experiments"
    / "eval_bundles_regime_polarity_safe_utility_confirmation_v9"
)
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_safe_utility_confirmation_audit_v9"
)
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_safe_utility_confirmation_analysis_v9"
)
REGISTRATION_ROOT = (
    ROOT / "jax_experiments" / "deployments"
    / "regime_polarity_safe_utility_confirmation_v9"
)
REGISTRATION_PATH = REGISTRATION_ROOT / "registration.json"
REPORT = (
    ROOT / "reports"
    / "regime_polarity_safe_utility_confirmation_v9_2026-08-30.md"
)
PREREG_REPORT = (
    ROOT / "reports"
    / "regime_polarity_safe_utility_confirmation_v9_preregistration_2026-08-30.md"
)

BUNDLE_SCHEMA = "bapr.regime-polarity-safe-utility-confirmation-bundle.v9"
CALIBRATION_SCHEMA = "bapr.regime-polarity-safe-utility-calibration.v9"
EVENT_SCHEMA = "bapr.regime-polarity-safe-utility-event.v9"
AUDIT_SCHEMA = "bapr.regime-polarity-safe-utility-audit.v9"
ANALYSIS_SCHEMA = "bapr.regime-polarity-safe-utility-analysis.v9"
REGISTRATION_SCHEMA = "bapr.regime-polarity-safe-utility-registration.v9"

file_record = source.file_record
read_json = source.read_json
write_json_atomic = source.write_json_atomic
write_text_atomic = source.write_text_atomic
checkpoint_record = source.checkpoint_record


def require_role(role: str) -> str:
    role = str(role)
    if role not in ROLES:
        raise ValueError(f"unknown v9 controller role {role!r}")
    return role


def role_fixed_mode(role: str) -> int:
    role = require_role(role)
    return int(role.removeprefix("specialist_")) if role.startswith(
        "specialist_") else -1


def require_training_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in TRAINING_SEEDS:
        raise ValueError(f"unknown v9 training seed {seed}")
    return seed


require_seed = require_training_seed


def require_trained_method(method: str) -> str:
    method = str(method)
    if method not in BASELINE_METHODS:
        raise ValueError(f"unknown v9 baseline method {method!r}")
    return method


def require_holdout_event_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in HOLDOUT_EVENT_SEEDS:
        raise ValueError(f"unknown v9 holdout event seed {seed}")
    return seed


def algo_for(method: str) -> str:
    method = require_trained_method(method)
    return "escp" if method == "escp_recurrent" else "resac"


def run_dir(name: str, seed: int) -> Path:
    seed = require_training_seed(seed)
    if name in ROLES:
        return CONTROLLER_RUN_ROOT / require_role(name) / f"seed_{seed}"
    method = require_trained_method(name)
    return BASELINE_RUN_ROOT / method / f"seed_{seed}"


def bundle_dir(name: str, seed: int) -> Path:
    seed = require_training_seed(seed)
    if name in ROLES:
        name = require_role(name)
    else:
        name = require_trained_method(name)
    return BUNDLE_ROOT / name / f"seed_{seed}"


def bundle_manifest(name: str, seed: int) -> Path:
    return bundle_dir(name, seed) / "bundle_manifest.json"


def bundle_required_paths(name: str, seed: int) -> tuple[Path, ...]:
    directory = bundle_dir(name, seed)
    return (
        directory / "bundle_manifest.json",
        directory / "checkpoints/params.pkl",
        directory / "checkpoints/train_state.pkl",
        directory / "logs/protocol_signature.json",
    )


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


def identity(name: str, seed: int) -> dict[str, Any]:
    seed = require_training_seed(seed)
    common = {
        "protocol_version": PROTOCOL_VERSION,
        "env": ENV,
        "family": FAMILY,
        "training_seed": seed,
        "max_iters": MAX_ITERS,
        "total_steps": FINAL_TOTAL_STEPS,
        "update_count": FINAL_UPDATE_COUNT,
    }
    if name in ROLES:
        role = require_role(name)
        return {
            **common,
            "benchmark_role": "fresh_final_policy_bank",
            "role": role,
            "algo": "sac",
            "fixed_mode": role_fixed_mode(role),
        }
    method = require_trained_method(name)
    return {
        **common,
        "benchmark_role": "fresh_same_controller_budget_baseline",
        "method": method,
        "algo": algo_for(method),
        "controller_budget_match": True,
    }


def expected_checkpoint(name: str) -> dict[str, Any]:
    algo = "sac" if name in ROLES else algo_for(name)
    return {
        "iteration": FINAL_ITERATION,
        "next_iteration": MAX_ITERS,
        "total_steps": FINAL_TOTAL_STEPS,
        "update_count": FINAL_UPDATE_COUNT,
        "algo": algo,
    }


def source_records(seed: int) -> dict[str, Any]:
    seed = require_training_seed(seed)
    return {
        role: file_record(bundle_manifest(role, seed)) for role in ROLES
    }


def baseline_records(seed: int) -> dict[str, Any]:
    seed = require_training_seed(seed)
    return {
        method: file_record(bundle_manifest(method, seed))
        for method in BASELINE_METHODS
    }


def estimator_records() -> dict[str, Any]:
    return source.estimator_records()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def registration_source_paths() -> tuple[Path, ...]:
    paths = (
        ROOT / "jax_experiments/analysis/regime_polarity_specialist_safe_utility_confirmation_v9.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_specialist_safe_utility_source_v9.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_specialist_safe_utility_baseline_v9.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_specialist_safe_utility_confirmation_audit_v9.py",
        ROOT / "jax_experiments/analysis/analyze_regime_polarity_specialist_safe_utility_confirmation_v9.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_confirmation_source_controller_v6.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_corrected_baseline_v2.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_corrected_audit_v2.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_specialist_safe_utility_audit_v8.py",
        source.estimator_records_path() if hasattr(
            source, "estimator_records_path") else source.frozen_v5.MODEL_MANIFEST,
        source.frozen_v5.MODEL_PATH,
        PREREG_REPORT,
    )
    return tuple(path.resolve() for path in paths)


def registration_payload() -> dict[str, Any]:
    paths = registration_source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"v9 registration sources missing: {missing}")
    return {
        "schema": REGISTRATION_SCHEMA,
        "status": "registered",
        "created_before_training": True,
        "identity": {
            "protocol_version": PROTOCOL_VERSION,
            "training_seeds": list(TRAINING_SEEDS),
            "calibration_event_seeds": list(CALIBRATION_EVENT_SEEDS),
            "holdout_event_seeds": list(HOLDOUT_EVENT_SEEDS),
            "primary_arm": PRIMARY_ARM,
            "baseline_methods": list(BASELINE_METHODS),
        },
        "selection_rule": {
            "minimum_calibration_gain": MIN_CALIBRATION_GAIN,
            "required_calibration_event_wins": len(CALIBRATION_EVENT_SEEDS),
            "require_zero_calibration_termination": True,
        },
        "bank_gate": {
            "minimum_headroom_gain": MIN_HEADROOM_GAIN,
            "minimum_primary_gain": MIN_PRIMARY_GAIN,
            "minimum_oracle_recovery": MIN_ORACLE_RECOVERY,
            "maximum_no_headroom_regression": MAX_NO_HEADROOM_REGRESSION,
        },
        "budget": {
            "per_controller_steps": FINAL_TOTAL_STEPS,
            "per_controller_updates": FINAL_UPDATE_COUNT,
            "bapr_policy_count": len(ROLES),
            "single_baseline_policy_count": 1,
            "frozen_estimator_training_excluded": True,
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
            raise ValueError("existing v9 registration changed")
    else:
        write_json_atomic(REGISTRATION_PATH, payload)
    return validate_registration()


def validate_registration() -> dict[str, Any]:
    if not REGISTRATION_PATH.is_file():
        raise FileNotFoundError(f"missing v9 registration: {REGISTRATION_PATH}")
    payload = read_json(REGISTRATION_PATH)
    if payload != registration_payload():
        raise ValueError("v9 registration or source closure changed")
    return payload


def assert_split_integrity() -> None:
    if len(TRAINING_SEEDS) != 5 or len(set(TRAINING_SEEDS)) != 5:
        raise ValueError("v9 requires five distinct policy-bank seeds")
    if set(TRAINING_SEEDS) & set(source.TRAINING_SEEDS):
        raise ValueError("v9 reused a v6 policy-bank seed")
    if set(CALIBRATION_EVENT_SEEDS) & set(HOLDOUT_EVENT_SEEDS):
        raise ValueError("v9 calibration and holdout streams overlap")


assert_split_integrity()
