"""Preregistered fresh ten-seed deployment confirmation for BAPR."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_corrected_baselines_v2 as corrected,
)
from jax_experiments.analysis import (
    regime_polarity_fallback_final_comparison_v1 as frozen,
)
from jax_experiments.analysis import (
    regime_polarity_final_mechanism_audit_v1 as mechanism,
)


ROOT = Path(os.environ.get(
    "BAPR_WORKSPACE_ROOT", Path(__file__).resolve().parents[2])).resolve()
PROTOCOL_VERSION = "v1-fresh-ten-seed-deployment-confirmation"
ENV = frozen.ENV
FAMILY = frozen.FAMILY
MODES = frozen.MODES
VARIANTS = ("mode_heads",)
TEACHER_GROUPS = VARIANTS

# These model-initialization seeds do not occur in any earlier registered
# polarity experiment. They form a new cohort; old five-seed observations are
# never appended to this experiment.
TRAINING_SEEDS = (
    52_021, 52_127, 52_237, 52_349, 52_457,
    52_567, 52_679, 52_783, 52_889, 52_999,
)
STUDENT_SEEDS = TRAINING_SEEDS
BASELINE_ROLES = ("sac",)
TRAINED_METHODS = ("escp_recurrent", "resac_b0")
METHODS = ("bapr", "sac", *TRAINED_METHODS)

# Student construction is a replication of the already frozen deployment
# pipeline. Only initialization changes. Confirmation events are unseen by all
# training, DAgger, validation, and model-selection stages.
TRAIN_EVENT_SEEDS = frozen.TRAIN_EVENT_SEEDS
SUPERVISED_VALIDATION_EVENT_SEEDS = (
    frozen.SUPERVISED_VALIDATION_EVENT_SEEDS)
DAGGER_EVENT_SEEDS = frozen.DAGGER_EVENT_SEEDS
CONTROL_VALIDATION_EVENT_SEEDS = frozen.CONTROL_VALIDATION_EVENT_SEEDS
FINAL_EVENT_SEEDS = (152_021, 152_127, 152_237, 152_349, 152_457)
EVENT_SEEDS = FINAL_EVENT_SEEDS

MAX_EPISODE_STEPS = frozen.MAX_EPISODE_STEPS
DWELL_STEPS = frozen.DWELL_STEPS
TRAIN_STATIONARY_EPISODES = frozen.TRAIN_STATIONARY_EPISODES
TRAIN_SWITCHING_EPISODES = frozen.TRAIN_SWITCHING_EPISODES
DAGGER_ROUNDS = frozen.DAGGER_ROUNDS
DAGGER_SWITCHING_EPISODES = frozen.DAGGER_SWITCHING_EPISODES
CONTROL_VALIDATION_SWITCHING_EPISODES = (
    frozen.CONTROL_VALIDATION_SWITCHING_EPISODES)
AUDIT_EPISODES_PER_TASK = frozen.AUDIT_EPISODES_PER_TASK
AUDIT_SWITCHING_EPISODES = frozen.AUDIT_SWITCHING_EPISODES
MODEL_CONFIG = frozen.MODEL_CONFIG

MAX_ITERS = frozen.MAX_ITERS
FINAL_ITERATION = frozen.FINAL_ITERATION
SAMPLES_PER_ITER = frozen.SAMPLES_PER_ITER
UPDATES_PER_ITER = frozen.UPDATES_PER_ITER
FINAL_TOTAL_STEPS = frozen.FINAL_TOTAL_STEPS
FINAL_UPDATE_COUNT = frozen.FINAL_UPDATE_COUNT
START_TRAIN_STEPS = frozen.START_TRAIN_STEPS
BASELINE_LR = {"sac": frozen.BASELINE_LR["sac"]}
ESCP_CONFIG = dict(corrected.ESCP_CONFIG)
RESAC_CONFIG = dict(corrected.RESAC_CONFIG)

FALLBACK_CONFIG = frozen.FALLBACK_CONFIG
ROBUST_SEED = frozen.ROBUST_SEED

# Three named baseline hypotheses are corrected as one family. The strongest
# per-seed envelope is diagnostic only because it is selected after observing
# each seed slot.
FAMILY_ALPHA = 0.05
MIN_SEED_WINS = 8
MIN_STATIONARY_RETENTION = 0.95
MAX_TERMINATION_GAP = 0.0

MODEL_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_deployment_confirmation_student_v1")
WORK_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_deployment_confirmation_work_v1")
BASELINE_RUN_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_deployment_confirmation_sac_runs_v1")
BASELINE_BUNDLE_ROOT = (
    ROOT / "jax_experiments"
    / "eval_bundles_regime_polarity_deployment_confirmation_sac_v1")
RUN_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_deployment_confirmation_corrected_runs_v1")
BUNDLE_ROOT = (
    ROOT / "jax_experiments"
    / "eval_bundles_regime_polarity_deployment_confirmation_corrected_v1")
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_deployment_confirmation_audit_v1")
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_deployment_confirmation_analysis_v1")
REGISTRATION_ROOT = (
    ROOT / "jax_experiments" / "deployments"
    / "regime_polarity_deployment_confirmation_v1")
REGISTRATION_PATH = REGISTRATION_ROOT / "registration.json"
PREREG_REPORT = (
    ROOT / "reports"
    / "regime_polarity_deployment_confirmation_v1_preregistration_2026-08-09.md")
RESULT_REPORT = (
    ROOT / "reports"
    / "regime_polarity_deployment_confirmation_v1_results_2026-08-09.md")

MODEL_SCHEMA = "bapr.regime-polarity-deployment-confirmation-student.v1"
CHECKPOINT_SCHEMA = (
    "bapr.regime-polarity-deployment-confirmation-student-checkpoint.v1")
BASELINE_BUNDLE_SCHEMA = (
    "bapr.regime-polarity-deployment-confirmation-sac-bundle.v1")
BUNDLE_SCHEMA = (
    "bapr.regime-polarity-deployment-confirmation-corrected-bundle.v1")
AUDIT_SCHEMA = "bapr.regime-polarity-deployment-confirmation-audit.v1"
ANALYSIS_SCHEMA = (
    "bapr.regime-polarity-deployment-confirmation-analysis.v1")
REGISTRATION_SCHEMA = (
    "bapr.regime-polarity-deployment-confirmation-registration.v1")

development = frozen.development
ensemble = frozen.ensemble
file_record = frozen.file_record
read_json = frozen.read_json
write_json_atomic = frozen.write_json_atomic
write_text_atomic = frozen.write_text_atomic
save_parameter_state = frozen.save_parameter_state
load_parameter_state = frozen.load_parameter_state
checkpoint_record = frozen.checkpoint_record

# The legacy final runner reads this module attribute. validate_registration()
# assigns the current immutable registration record before any output is made.
FROZEN_REGISTRATION_RECORD: dict[str, Any] | None = None


def require_variant(variant: str) -> str:
    variant = str(variant)
    if variant != "mode_heads":
        raise ValueError("confirmation permits only the frozen mode-head student")
    return variant


require_teacher_group = require_variant


def require_training_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in TRAINING_SEEDS:
        raise ValueError(f"unknown confirmation seed {seed}")
    return seed


require_student_seed = require_training_seed
require_seed = require_training_seed


def require_baseline_role(role: str) -> str:
    role = str(role)
    if role not in BASELINE_ROLES:
        raise ValueError(f"unknown confirmation SAC role {role!r}")
    return role


def require_trained_method(method: str) -> str:
    method = str(method)
    if method not in TRAINED_METHODS:
        raise ValueError(f"unknown corrected confirmation method {method!r}")
    return method


def require_method(method: str) -> str:
    method = str(method)
    if method not in METHODS:
        raise ValueError(f"unknown confirmation method {method!r}")
    return method


def require_event_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in EVENT_SEEDS:
        raise ValueError(f"unknown confirmation event seed {seed}")
    return seed


def algo_for(method: str) -> str:
    method = require_trained_method(method)
    return "escp" if method == "escp_recurrent" else "resac"


def source_groups(variant: str) -> tuple[str, ...]:
    return development.source_groups(require_variant(variant))


def controller_keys(variant: str) -> tuple[tuple[str, int], ...]:
    return development.controller_keys(require_variant(variant))


def source_bundle_dirs(variant: str) -> tuple[Path, ...]:
    return development.source_bundle_dirs(require_variant(variant))


def source_required_paths(variant: str) -> tuple[Path, ...]:
    return development.source_required_paths(require_variant(variant))


def model_dir(variant: str, seed: int) -> Path:
    return MODEL_ROOT / require_variant(variant) / f"student_seed_{require_seed(seed)}"


def work_dir(variant: str, seed: int) -> Path:
    return WORK_ROOT / require_variant(variant) / f"student_seed_{require_seed(seed)}"


def model_path(variant: str, seed: int) -> Path:
    return model_dir(variant, seed) / "student_params.npz"


def model_manifest(variant: str, seed: int) -> Path:
    return model_dir(variant, seed) / "model_manifest.json"


def baseline_run_dir(role: str, seed: int) -> Path:
    return BASELINE_RUN_ROOT / require_baseline_role(role) / f"seed_{require_seed(seed)}"


def baseline_bundle_dir(role: str, seed: int) -> Path:
    return BASELINE_BUNDLE_ROOT / require_baseline_role(role) / f"seed_{require_seed(seed)}"


def baseline_bundle_manifest(role: str, seed: int) -> Path:
    return baseline_bundle_dir(role, seed) / "bundle_manifest.json"


def baseline_bundle_required_paths(role: str, seed: int) -> tuple[Path, ...]:
    directory = baseline_bundle_dir(role, seed)
    return (
        directory / "bundle_manifest.json",
        directory / "checkpoints/params.pkl",
        directory / "checkpoints/train_state.pkl",
        directory / "logs/protocol_signature.json",
    )


def run_dir(method: str, seed: int) -> Path:
    return RUN_ROOT / require_trained_method(method) / f"seed_{require_seed(seed)}"


def bundle_dir(method: str, seed: int) -> Path:
    return BUNDLE_ROOT / require_trained_method(method) / f"seed_{require_seed(seed)}"


def bundle_manifest(method: str, seed: int) -> Path:
    return bundle_dir(method, seed) / "bundle_manifest.json"


def bundle_required_paths(method: str, seed: int) -> tuple[Path, ...]:
    directory = bundle_dir(method, seed)
    return (
        directory / "bundle_manifest.json",
        directory / "checkpoints/params.pkl",
        directory / "checkpoints/train_state.pkl",
        directory / "logs/protocol_signature.json",
    )


def audit_dir(method: str, seed: int) -> Path:
    return AUDIT_ROOT / require_method(method) / f"seed_{require_seed(seed)}"


def audit_manifest(method: str, seed: int) -> Path:
    return audit_dir(method, seed) / "audit_manifest.json"


def all_audit_manifests() -> tuple[Path, ...]:
    return tuple(
        audit_manifest(method, seed)
        for method in METHODS for seed in TRAINING_SEEDS)


def analysis_json() -> Path:
    return ANALYSIS_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return ANALYSIS_ROOT / "analysis.md"


def model_identity(variant: str, seed: int) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "fresh_deployment_student_replication",
        "confirmatory": True,
        "fresh_cohort": True,
        "env": ENV,
        "family": FAMILY,
        "variant": require_variant(variant),
        "teacher_group": "combined",
        "controller_keys": [list(value) for value in controller_keys(variant)],
        "reduction": development.REDUCTION,
        "student_seed": require_seed(seed),
        "train_event_seeds": list(TRAIN_EVENT_SEEDS),
        "dagger_event_seeds": list(DAGGER_EVENT_SEEDS),
        "supervised_validation_event_seeds": list(
            SUPERVISED_VALIDATION_EVENT_SEEDS),
        "control_validation_event_seeds": list(
            CONTROL_VALIDATION_EVENT_SEEDS),
        "dagger_rounds": DAGGER_ROUNDS,
        "model_config": MODEL_CONFIG,
        "student_online_inputs": ["observation", "soft_mode_posterior"],
        "checkpoint_selection": "frozen_development_control_validation",
        "confirmation_event_seeds_hidden": list(EVENT_SEEDS),
    }


def baseline_identity(role: str, seed: int) -> dict[str, Any]:
    role = require_baseline_role(role)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "fresh_same_environment_baseline",
        "confirmatory": True,
        "method": role,
        "algo": role,
        "training_seed": require_seed(seed),
        "env": ENV,
        "family": FAMILY,
        "max_iters": MAX_ITERS,
        "total_steps": FINAL_TOTAL_STEPS,
        "update_count": FINAL_UPDATE_COUNT,
        "controller_budget_match": True,
    }


def identity(method: str, seed: int) -> dict[str, Any]:
    method = require_trained_method(method)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "fresh_corrected_same_budget_baseline",
        "confirmatory": True,
        "method": method,
        "algo": algo_for(method),
        "training_seed": require_seed(seed),
        "env": ENV,
        "family": FAMILY,
        "max_iters": MAX_ITERS,
        "total_steps": FINAL_TOTAL_STEPS,
        "update_count": FINAL_UPDATE_COUNT,
        "controller_budget_match": True,
    }


def expected_checkpoint(method: str) -> dict[str, Any]:
    return {
        "iteration": FINAL_ITERATION,
        "next_iteration": MAX_ITERS,
        "total_steps": FINAL_TOTAL_STEPS,
        "update_count": FINAL_UPDATE_COUNT,
        "algo": algo_for(method),
    }


def audit_identity(method: str, seed: int) -> dict[str, Any]:
    method = require_method(method)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "untouched_fresh_ten_seed_confirmation",
        "confirmatory": True,
        "method": method,
        "seed": require_seed(seed),
        "event_seeds": list(EVENT_SEEDS),
        "env": ENV,
        "family": FAMILY,
        "strict_horizon": MAX_EPISODE_STEPS,
        "dwell_steps": DWELL_STEPS,
        "stationary_episodes_per_mode": AUDIT_EPISODES_PER_TASK,
        "switching_episodes": AUDIT_SWITCHING_EPISODES,
        "evaluation_policy": "deterministic_mean",
        "online_forbidden_for_bapr": [
            "mode_id", "action_gain", "executed_action", "switch_clock"],
    }


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def registration_source_paths() -> tuple[Path, ...]:
    new_paths = (
        ROOT / "jax_experiments/analysis/regime_polarity_deployment_confirmation_v1.py",
        ROOT / "jax_experiments/analysis/train_regime_polarity_deployment_confirmation_student_v1.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_deployment_confirmation_sac_v1.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_deployment_confirmation_corrected_v1.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_deployment_confirmation_audit_v1.py",
        ROOT / "jax_experiments/analysis/analyze_regime_polarity_deployment_confirmation_v1.py",
        ROOT / "scripts/submit_regime_polarity_deployment_confirmation_v1.py",
        ROOT / "jax_experiments/tests/test_regime_polarity_deployment_confirmation_v1.py",
        PREREG_REPORT,
        mechanism.REGISTRATION_PATH,
    )
    inherited = (
        *frozen.registration_source_paths(),
        *corrected.registration_source_paths(),
    )
    return tuple(dict.fromkeys(
        path.resolve() for path in (*new_paths, *inherited)))


def registration_payload() -> dict[str, Any]:
    paths = registration_source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"confirmation registration sources missing: {missing}")
    return {
        "schema": REGISTRATION_SCHEMA,
        "status": "registered",
        "created_before_confirmation_training": True,
        "identity": {
            "protocol_version": PROTOCOL_VERSION,
            "env": ENV,
            "family": FAMILY,
            "methods": list(METHODS),
            "student_architecture": "mode_heads",
            "training_seeds": list(TRAINING_SEEDS),
            "event_seeds": list(EVENT_SEEDS),
            "old_seed_results_excluded": True,
        },
        "budget": {
            "single_controller_max_iters": MAX_ITERS,
            "single_controller_total_steps": FINAL_TOTAL_STEPS,
            "single_controller_update_count": FINAL_UPDATE_COUNT,
            "bapr_role": "frozen_teacher_estimator_deployment_student_replication",
            "full_construction_budget_report": (
                "reports/bapr_deployment_compression_budget_2026-08-09.md"),
        },
        "release_gate": {
            "protocol_version": mechanism.PROTOCOL_VERSION,
            "registration": mechanism.registration_record(),
            "analysis_path": _relative(mechanism.analysis_json()),
            "pass_marker": _relative(mechanism.pass_marker()),
            "required_field": "mechanism_pass=true",
        },
        "statistics": {
            "family_alpha": FAMILY_ALPHA,
            "family": [
                "bapr>sac", "bapr>escp_recurrent", "bapr>resac_b0"],
            "multiplicity": "Holm one-sided paired tests",
            "simultaneous_interval": "Bonferroni one-sided lower bound",
            "minimum_seed_wins": MIN_SEED_WINS,
            "minimum_stationary_retention": MIN_STATIONARY_RETENTION,
            "maximum_termination_gap": MAX_TERMINATION_GAP,
            "strongest_seedwise_envelope": "diagnostic_only",
        },
        "source_records": {
            _relative(path): file_record(path) for path in paths
        },
    }


def create_registration() -> dict[str, Any]:
    payload = registration_payload()
    REGISTRATION_ROOT.mkdir(parents=True, exist_ok=True)
    if REGISTRATION_PATH.exists():
        current = read_json(REGISTRATION_PATH)
        if current != payload:
            raise ValueError("existing confirmation registration changed")
    else:
        write_json_atomic(REGISTRATION_PATH, payload)
    validate_registration()
    return payload


def validate_registration() -> dict[str, Any]:
    global FROZEN_REGISTRATION_RECORD
    if not REGISTRATION_PATH.is_file():
        raise FileNotFoundError(
            f"missing confirmation registration: {REGISTRATION_PATH}")
    payload = read_json(REGISTRATION_PATH)
    if payload != registration_payload():
        raise ValueError("confirmation registration or source closure changed")
    FROZEN_REGISTRATION_RECORD = file_record(REGISTRATION_PATH)
    return payload


def registration_record() -> dict[str, Any]:
    validate_registration()
    assert FROZEN_REGISTRATION_RECORD is not None
    return FROZEN_REGISTRATION_RECORD


def validate_mechanism_release() -> dict[str, Any]:
    validate_registration()
    mechanism.validate_registration()
    analysis_path = mechanism.analysis_json()
    marker = mechanism.pass_marker()
    if not analysis_path.is_file() or not marker.is_file():
        raise RuntimeError(
            "confirmation is gated: final mechanism audit has not passed")
    payload = read_json(analysis_path)
    expected_marker = f"{mechanism.PROTOCOL_VERSION}\nmechanism_pass=true\n"
    if (payload.get("schema") != mechanism.ANALYSIS_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("registration") != mechanism.registration_record()
            or payload.get("student_seeds") != list(mechanism.STUDENT_SEEDS)
            or payload.get("event_seeds") != list(mechanism.EVENT_SEEDS)
            or payload.get("mechanism_pass") is not True
            or marker.read_text(encoding="utf-8") != expected_marker):
        raise RuntimeError("confirmation is gated: invalid mechanism pass evidence")
    return {
        "analysis": file_record(analysis_path),
        "pass_marker": file_record(marker),
    }


def assert_split_integrity() -> None:
    if len(TRAINING_SEEDS) != 10 or len(set(TRAINING_SEEDS)) != 10:
        raise ValueError("confirmation requires exactly ten fresh model seeds")
    if set(TRAINING_SEEDS) & set(frozen.TRAINING_SEEDS):
        raise ValueError("confirmation model seeds overlap the prior cohort")
    if len(EVENT_SEEDS) != len(set(EVENT_SEEDS)):
        raise ValueError("confirmation event seeds are duplicated")
    prior_events = {
        *development.TRAIN_EVENT_SEEDS,
        *development.SUPERVISED_VALIDATION_EVENT_SEEDS,
        *development.DAGGER_EVENT_SEEDS,
        *development.CONTROL_VALIDATION_EVENT_SEEDS,
        *development.AUDIT_EVENT_SEEDS,
        *development.SEALED_CONFIRMATION_EVENT_SEEDS,
        *frozen.FINAL_EVENT_SEEDS,
        *mechanism.EVENT_SEEDS,
    }
    overlap = set(EVENT_SEEDS) & prior_events
    if overlap:
        raise ValueError(f"confirmation reused earlier event seeds: {overlap}")


assert_split_integrity()

