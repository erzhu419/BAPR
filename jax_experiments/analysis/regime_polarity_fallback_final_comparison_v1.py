"""Preregistered final comparison for the causal-fallback BAPR stack."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_policy_distillation_control_v2 as development,
)
from jax_experiments.analysis import (
    regime_polarity_policy_distillation_transient_causal_ceiling_v1 as ceiling,
)


ROOT = development.ROOT
PROTOCOL_VERSION = "v1-frozen-causal-fallback-final-comparison"
ENV = development.ENV
FAMILY = development.FAMILY
MODES = development.MODES
VARIANTS = ("mode_heads",)
TEACHER_GROUPS = VARIANTS

# These seeds have not appeared in any earlier BAPR protocol. The same labels
# are used across student initialization and baseline training to keep the
# final seed-level summaries balanced without pretending that the training
# processes share identical randomness.
TRAINING_SEEDS = (2009, 2113, 2213, 2311, 2417)
STUDENT_SEEDS = TRAINING_SEEDS
BASELINE_ROLES = ("sac", "escp", "resac")

# Student training reuses only the already frozen development data. Final
# event streams are new and cannot be used for model or threshold selection.
TRAIN_EVENT_SEEDS = development.TRAIN_EVENT_SEEDS
SUPERVISED_VALIDATION_EVENT_SEEDS = (
    development.SUPERVISED_VALIDATION_EVENT_SEEDS)
DAGGER_EVENT_SEEDS = development.DAGGER_EVENT_SEEDS
CONTROL_VALIDATION_EVENT_SEEDS = development.CONTROL_VALIDATION_EVENT_SEEDS
FINAL_EVENT_SEEDS = (106_301, 106_331, 106_367, 106_399, 106_451)

MAX_EPISODE_STEPS = development.MAX_EPISODE_STEPS
DWELL_STEPS = development.DWELL_STEPS
TRAIN_STATIONARY_EPISODES = development.TRAIN_STATIONARY_EPISODES
TRAIN_SWITCHING_EPISODES = development.TRAIN_SWITCHING_EPISODES
DAGGER_ROUNDS = development.DAGGER_ROUNDS
DAGGER_SWITCHING_EPISODES = development.DAGGER_SWITCHING_EPISODES
CONTROL_VALIDATION_SWITCHING_EPISODES = (
    development.CONTROL_VALIDATION_SWITCHING_EPISODES)
AUDIT_EPISODES_PER_TASK = 5
AUDIT_SWITCHING_EPISODES = 5
MODEL_CONFIG = development.MODEL_CONFIG

MAX_ITERS = 1400
FINAL_ITERATION = MAX_ITERS - 1
SAMPLES_PER_ITER = 4000
UPDATES_PER_ITER = 250
FINAL_TOTAL_STEPS = MAX_ITERS * SAMPLES_PER_ITER
FINAL_UPDATE_COUNT = MAX_ITERS * UPDATES_PER_ITER
START_TRAIN_STEPS = 4000
BASELINE_LR = {"sac": 3e-4, "escp": 3e-4, "resac": 1e-5}
RESAC_WEIGHT_REG = 0.01
RESAC_BETA_OOD = 0.01
RESAC_BETA = -2.0

FALLBACK_CONFIG = ceiling.parent.parent.require_config(
    ceiling.parent.SELECTED_CONFIG_NAME)
ROBUST_SEED = 719

# Primary claim gates are evaluated over five model-initialization summaries,
# each already averaged over the same five event clusters.
MIN_SEED_WINS = 4
MIN_STATIONARY_RETENTION = 0.95
MAX_TERMINATION_GAP = 0.0
CLUSTER_T_CRITICAL_95 = 2.7764451051977987

MODEL_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_fallback_final_student_model_v1")
WORK_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_fallback_final_student_work_v1")
BASELINE_RUN_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_fallback_final_baseline_runs_v1")
BASELINE_BUNDLE_ROOT = (
    ROOT / "jax_experiments"
    / "eval_bundles_regime_polarity_fallback_final_baselines_v1")
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_fallback_final_audit_v1")
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_fallback_final_analysis_v1")
REGISTRATION_ROOT = (
    ROOT / "jax_experiments"
    / "deployments" / "regime_polarity_fallback_final_v1")
REGISTRATION_PATH = REGISTRATION_ROOT / "registration.json"
REPORT = (
    ROOT / "reports"
    / "regime_polarity_fallback_final_comparison_v1_protocol_2026-08-02.md")

MODEL_SCHEMA = "bapr.regime-polarity-fallback-final-student-model.v1"
CHECKPOINT_SCHEMA = (
    "bapr.regime-polarity-fallback-final-student-checkpoint.v1")
BASELINE_BUNDLE_SCHEMA = (
    "bapr.regime-polarity-fallback-final-baseline-bundle.v1")
AUDIT_SCHEMA = "bapr.regime-polarity-fallback-final-audit.v1"
ANALYSIS_SCHEMA = "bapr.regime-polarity-fallback-final-analysis.v1"
REGISTRATION_SCHEMA = "bapr.regime-polarity-fallback-registration.v1"

ensemble = development.ensemble
file_record = development.file_record
read_json = development.read_json
write_json_atomic = development.write_json_atomic
write_text_atomic = development.write_text_atomic
save_parameter_state = development.save_parameter_state
load_parameter_state = development.load_parameter_state
checkpoint_record = development.ensemble.development.checkpoint_record

FROZEN_REGISTRATION_RECORD: dict[str, Any] = {
    "sha256": "e58b3ccfe947702d65dfff076cb3dde3d6d4bca767c4f778286009a9bb671b0f",
    "size": 23314,
}


def require_variant(variant: str) -> str:
    variant = str(variant)
    if variant != "mode_heads":
        raise ValueError("final comparison permits only mode_heads")
    return variant


require_teacher_group = require_variant


def require_training_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in TRAINING_SEEDS:
        raise ValueError(f"unknown final training seed {seed}")
    return seed


require_student_seed = require_training_seed


def require_baseline_role(role: str) -> str:
    role = str(role)
    if role not in BASELINE_ROLES:
        raise ValueError(f"unknown final baseline role {role!r}")
    return role


def require_event_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in FINAL_EVENT_SEEDS:
        raise ValueError(f"unknown final event seed {seed}")
    return seed


def source_groups(variant: str) -> tuple[str, ...]:
    return development.source_groups(require_variant(variant))


def controller_keys(variant: str) -> tuple[tuple[str, int], ...]:
    return development.controller_keys(require_variant(variant))


def source_bundle_dirs(variant: str) -> tuple[Path, ...]:
    return development.source_bundle_dirs(require_variant(variant))


def source_required_paths(variant: str) -> tuple[Path, ...]:
    return development.source_required_paths(require_variant(variant))


def model_dir(variant: str, student_seed: int) -> Path:
    return (
        MODEL_ROOT / require_variant(variant)
        / f"student_seed_{require_student_seed(student_seed)}")


def work_dir(variant: str, student_seed: int) -> Path:
    return (
        WORK_ROOT / require_variant(variant)
        / f"student_seed_{require_student_seed(student_seed)}")


def model_path(variant: str, student_seed: int) -> Path:
    return model_dir(variant, student_seed) / "student_params.npz"


def model_manifest(variant: str, student_seed: int) -> Path:
    return model_dir(variant, student_seed) / "model_manifest.json"


def baseline_run_dir(role: str, seed: int) -> Path:
    return (
        BASELINE_RUN_ROOT / require_baseline_role(role)
        / f"seed_{require_training_seed(seed)}")


def baseline_bundle_dir(role: str, seed: int) -> Path:
    return (
        BASELINE_BUNDLE_ROOT / require_baseline_role(role)
        / f"seed_{require_training_seed(seed)}")


def baseline_bundle_manifest(role: str, seed: int) -> Path:
    return baseline_bundle_dir(role, seed) / "bundle_manifest.json"


def baseline_bundle_required_paths(role: str, seed: int) -> tuple[Path, ...]:
    directory = baseline_bundle_dir(role, seed)
    return (
        directory / "bundle_manifest.json",
        directory / "checkpoints" / "params.pkl",
        directory / "checkpoints" / "train_state.pkl",
        directory / "logs" / "protocol_signature.json",
    )


def audit_dir(method: str, seed: int) -> Path:
    method = str(method)
    if method != "bapr" and method not in BASELINE_ROLES:
        raise ValueError(f"unknown final audit method {method!r}")
    return AUDIT_ROOT / method / f"seed_{require_training_seed(seed)}"


def audit_manifest(method: str, seed: int) -> Path:
    return audit_dir(method, seed) / "audit_manifest.json"


def all_audit_manifests() -> tuple[Path, ...]:
    return tuple(
        audit_manifest(method, seed)
        for method in ("bapr", *BASELINE_ROLES)
        for seed in TRAINING_SEEDS
    )


def analysis_json() -> Path:
    return ANALYSIS_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return ANALYSIS_ROOT / "analysis.md"


def model_identity(variant: str, student_seed: int) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "frozen_final_student_seed_replication",
        "confirmatory": True,
        "env": ENV,
        "family": FAMILY,
        "variant": require_variant(variant),
        "teacher_group": "combined",
        "controller_keys": [
            list(value) for value in controller_keys(variant)],
        "reduction": development.REDUCTION,
        "student_seed": require_student_seed(student_seed),
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
        "final_event_seeds_hidden": list(FINAL_EVENT_SEEDS),
    }


def baseline_identity(role: str, seed: int) -> dict[str, Any]:
    role = require_baseline_role(role)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "final_same_environment_baseline",
        "confirmatory": True,
        "env": ENV,
        "family": FAMILY,
        "role": role,
        "algo": role,
        "training_seed": require_training_seed(seed),
        "max_iters": MAX_ITERS,
        "total_steps": FINAL_TOTAL_STEPS,
        "update_count": FINAL_UPDATE_COUNT,
    }


def audit_identity(method: str, seed: int) -> dict[str, Any]:
    if method != "bapr":
        require_baseline_role(method)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "untouched_final_baseline_comparison",
        "confirmatory": True,
        "method": str(method),
        "seed": require_training_seed(seed),
        "event_seeds": list(FINAL_EVENT_SEEDS),
        "env": ENV,
        "family": FAMILY,
        "strict_horizon": MAX_EPISODE_STEPS,
        "dwell_steps": DWELL_STEPS,
        "online_forbidden_for_bapr": [
            "mode_id", "action_gain", "executed_action", "switch_clock"],
    }


def registration_source_paths() -> tuple[Path, ...]:
    robust = ceiling.parent.parent.frozen.development.ensemble.final.bundle_dir(
        ENV, "robust", ROBUST_SEED)
    return (
        ROOT / "jax_experiments/common/causal_fallback.py",
        ROOT / "jax_experiments/analysis/"
        "run_regime_polarity_fallback_final_audit_v1.py",
        ROOT / "jax_experiments/analysis/"
        "run_regime_polarity_fallback_final_baseline_v1.py",
        ROOT / "jax_experiments/analysis/"
        "train_regime_polarity_fallback_final_student_v1.py",
        ROOT / "jax_experiments/analysis/"
        "train_regime_polarity_policy_distillation_control_v2.py",
        ROOT / "jax_experiments/analysis/"
        "regime_polarity_policy_distillation_control_model.py",
        ROOT / "jax_experiments/analysis/"
        "regime_polarity_expected_action_system_id_model.py",
        ROOT / "jax_experiments/train.py",
        ROOT / "jax_experiments/algos/sac_base.py",
        ROOT / "jax_experiments/algos/escp.py",
        ROOT / "jax_experiments/algos/resac.py",
        ROOT / "jax_experiments/envs/brax_env.py",
        ROOT / "jax_experiments/envs/stochastic_mode_env.py",
        ceiling.parent.parent.frozen.development.ensemble.final.MODEL_MANIFEST,
        ceiling.parent.parent.frozen.development.ensemble.final.MODEL_PATH,
        robust / "bundle_manifest.json",
        robust / "checkpoints/params.pkl",
        robust / "checkpoints/train_state.pkl",
        robust / "logs/protocol_signature.json",
        ceiling.parent.parent.SELECTION_ROOT / "selection_manifest.json",
        ceiling.parent.analysis_json(),
        ceiling.analysis_json(),
        *(path for path in source_required_paths("mode_heads")),
    )


def registration_payload() -> dict[str, Any]:
    return {
        "schema": REGISTRATION_SCHEMA,
        "status": "frozen",
        "identity": {
            "protocol_version": PROTOCOL_VERSION,
            "env": ENV,
            "family": FAMILY,
            "student_architecture": "mode_heads",
            "student_training_seeds": list(STUDENT_SEEDS),
            "baseline_roles": list(BASELINE_ROLES),
            "baseline_training_seeds": list(TRAINING_SEEDS),
            "final_event_seeds": list(FINAL_EVENT_SEEDS),
            "robust_fallback_seed": ROBUST_SEED,
            "fallback_config": FALLBACK_CONFIG.to_dict(),
        },
        "source_records": {
            str(path.relative_to(ROOT)): file_record(path)
            for path in registration_source_paths()
        },
    }


def write_registration() -> dict[str, Any]:
    payload = registration_payload()
    if REGISTRATION_PATH.is_file():
        existing = read_json(REGISTRATION_PATH)
        if existing != payload:
            raise ValueError("existing final registration does not match inputs")
        return existing
    write_json_atomic(REGISTRATION_PATH, payload)
    return payload


def validate_registration() -> None:
    if FROZEN_REGISTRATION_RECORD is None:
        raise ValueError("final registration record has not been frozen")
    if (not REGISTRATION_PATH.is_file()
            or file_record(REGISTRATION_PATH) != FROZEN_REGISTRATION_RECORD
            or read_json(REGISTRATION_PATH) != registration_payload()):
        raise ValueError("final comparison registration changed")


def assert_split_integrity() -> None:
    if len(FINAL_EVENT_SEEDS) != len(set(FINAL_EVENT_SEEDS)):
        raise ValueError("final event seeds are not unique")
    prior = {
        *development.TRAIN_EVENT_SEEDS,
        *development.DAGGER_EVENT_SEEDS,
        *development.SUPERVISED_VALIDATION_EVENT_SEEDS,
        *development.CONTROL_VALIDATION_EVENT_SEEDS,
        *development.AUDIT_EVENT_SEEDS,
        *development.SEALED_CONFIRMATION_EVENT_SEEDS,
        *ceiling.EVENT_SEEDS,
        *ceiling.parent.EVENT_SEEDS,
        *ceiling.parent.parent.SCREEN_EVENT_SEEDS,
        *ceiling.parent.parent.AUDIT_EVENT_SEEDS,
        102_301, 102_331, 102_367, 102_397, 102_451,
    }
    if set(FINAL_EVENT_SEEDS) & prior:
        raise ValueError("final comparison reused an earlier event seed")


assert_split_integrity()
