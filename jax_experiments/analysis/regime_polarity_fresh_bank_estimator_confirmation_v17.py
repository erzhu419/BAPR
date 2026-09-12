"""Fresh-policy-bank confirmation of the frozen v16 estimator."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_robust_warmstart_confirmation_v12 as policy_parent,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_system_id_v5 as v5_parent,
)
from jax_experiments.analysis import (
    regime_polarity_switch_weighted_estimator_v16 as estimator_parent,
)


ROOT = policy_parent.ROOT
PROTOCOL_VERSION = "v17-fresh-bank-estimator-confirmation"
ENV = policy_parent.ENV
FAMILY = policy_parent.FAMILY
MODES = policy_parent.MODES
VARIANTS = ("actor_only",)
TRAINING_SEEDS = (81_003, 81_021, 81_039, 81_057, 81_079)

CALIBRATION_EVENT_SEEDS = (181_001, 181_017, 181_033)
STATIONARY_HOLDOUT_EVENT_SEEDS = (181_101, 181_117, 181_133)
SWITCHING_EVENT_SEEDS = (181_201, 181_217, 181_233)
SWITCHING_SCHEDULES = {
    181_201: (0, 3, 1, 2),
    181_217: (1, 2, 0, 3),
    181_233: (2, 1, 3, 0),
}

SOURCE_NEXT_ITERATION = policy_parent.SOURCE_NEXT_ITERATION
SOURCE_ITERATION = policy_parent.SOURCE_ITERATION
SOURCE_TOTAL_STEPS = policy_parent.SOURCE_TOTAL_STEPS
SOURCE_UPDATE_COUNT = policy_parent.SOURCE_UPDATE_COUNT
SOURCE_START_TRAIN_STEPS = policy_parent.SOURCE_START_TRAIN_STEPS
FINETUNE_ITERS = policy_parent.FINETUNE_ITERS
FINAL_NEXT_ITERATION = policy_parent.FINAL_NEXT_ITERATION
FINAL_ITERATION = policy_parent.FINAL_ITERATION
SAMPLES_PER_ITER = policy_parent.SAMPLES_PER_ITER
UPDATES_PER_ITER = policy_parent.UPDATES_PER_ITER
FINAL_TOTAL_STEPS = policy_parent.FINAL_TOTAL_STEPS
FINAL_UPDATE_COUNT = policy_parent.FINAL_UPDATE_COUNT

DWELL_STEPS = policy_parent.DWELL_STEPS
MAX_EPISODE_STEPS = policy_parent.MAX_EPISODE_STEPS
EPISODES_PER_TASK = policy_parent.EPISODES_PER_TASK
SWITCHING_EPISODES = policy_parent.SWITCHING_EPISODES
DELAY_STEPS = estimator_parent.DELAY_STEPS
SWITCH_WINDOW_STEPS = estimator_parent.SWITCH_WINDOW_STEPS

MIN_CALIBRATION_GAIN = policy_parent.MIN_CALIBRATION_GAIN
MIN_HOLDOUT_MODE_GAIN = policy_parent.MIN_HOLDOUT_MODE_GAIN
MIN_HOLDOUT_MODE_WINS = policy_parent.MIN_HOLDOUT_MODE_WINS
MIN_SWITCHING_GAIN = policy_parent.MIN_SWITCHING_GAIN
MIN_HEADROOM_GAIN = estimator_parent.MIN_HEADROOM_GAIN
MIN_CAUSAL_RETENTION = estimator_parent.MIN_CAUSAL_RETENTION
MIN_ESTIMATOR_GAIN = estimator_parent.MIN_ESTIMATOR_GAIN
REQUIRED_SEED_PASSES = 4
REQUIRED_V16_SEED_WINS = 4
REQUIRED_MECHANISM_SEED_PASSES = 4

ARMS = (
    "robust_sac",
    "true_mode_safe_utility",
    "delayed_oracle_4_safe_utility",
    "frozen_v5_posterior_map",
    "switch_weighted_v16_posterior_map",
)

RUN_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_fresh_bank_estimator_confirmation_v17"
)
SOURCE_RUN_ROOT = RUN_ROOT / "robust_sources"
SPECIALIST_RUN_ROOT = RUN_ROOT / "specialists"
SOURCE_BUNDLE_ROOT = (
    ROOT / "jax_experiments"
    / "eval_bundles_regime_polarity_robust_source_v17"
)
SPECIALIST_BUNDLE_ROOT = (
    ROOT / "jax_experiments"
    / "eval_bundles_regime_polarity_fresh_bank_estimator_confirmation_v17"
)
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_fresh_bank_estimator_confirmation_audit_v17"
)
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_fresh_bank_estimator_confirmation_analysis_v17"
)
REGISTRATION_ROOT = (
    ROOT / "jax_experiments" / "deployments"
    / "regime_polarity_fresh_bank_estimator_confirmation_v17"
)
REGISTRATION_PATH = REGISTRATION_ROOT / "registration.json"
PREREG_REPORT = (
    ROOT / "reports"
    / "regime_polarity_fresh_bank_estimator_confirmation_v17_preregistration_2026-09-01.md"
)
REPORT = (
    ROOT / "reports"
    / "regime_polarity_fresh_bank_estimator_confirmation_v17_2026-09-01.md"
)

BOOTSTRAP_NAME = "warmstart_bootstrap.json"
POLICY_NAME = "policy_params.pkl"
SOURCE_BUNDLE_SCHEMA = "bapr.robust-source-policy-bundle.v17"
BUNDLE_SCHEMA = "bapr.robust-warmstart-specialist-policy-bundle.v17"
AUDIT_SCHEMA = "bapr.fresh-bank-estimator-confirmation-audit.v17"
ANALYSIS_SCHEMA = "bapr.fresh-bank-estimator-confirmation-analysis.v17"
REGISTRATION_SCHEMA = "bapr.fresh-bank-estimator-confirmation-registration.v17"
BOOTSTRAP_SCHEMA = "bapr.robust-warmstart-specialist-bootstrap.v17"

file_record = policy_parent.file_record
read_json = policy_parent.read_json
write_json_atomic = policy_parent.write_json_atomic
write_text_atomic = policy_parent.write_text_atomic
checkpoint_record = policy_parent.checkpoint_record


def require_variant(variant: str) -> str:
    variant = str(variant)
    if variant not in VARIANTS:
        raise ValueError(f"unknown v17 variant {variant!r}")
    return variant


def require_training_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in TRAINING_SEEDS:
        raise ValueError(f"unknown v17 confirmation seed {seed}")
    return seed


require_seed = require_training_seed


def require_mode(mode: int) -> int:
    mode = int(mode)
    if mode not in MODES:
        raise ValueError(f"unknown actuator-polarity mode {mode}")
    return mode


def require_switching_event_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in SWITCHING_EVENT_SEEDS:
        raise ValueError(f"unknown v17 switching event seed {seed}")
    return seed


def switching_sequence(event_seed: int, episode: int) -> tuple[int, ...]:
    base = SWITCHING_SCHEDULES[require_switching_event_seed(event_seed)]
    shift = int(episode) % len(base)
    return tuple(base[shift:] + base[:shift])


def source_run_dir(seed: int) -> Path:
    return SOURCE_RUN_ROOT / f"seed_{require_training_seed(seed)}"


def source_bundle(seed: int) -> Path:
    return SOURCE_BUNDLE_ROOT / f"seed_{require_training_seed(seed)}"


def source_manifest(seed: int) -> Path:
    return source_bundle(seed) / "bundle_manifest.json"


def source_required_paths(seed: int) -> tuple[Path, ...]:
    directory = source_bundle(seed)
    return (
        directory / "bundle_manifest.json",
        directory / "policy" / POLICY_NAME,
        directory / "logs" / "protocol_signature.json",
    )


def run_dir(variant: str, seed: int, mode: int) -> Path:
    return (
        SPECIALIST_RUN_ROOT / require_variant(variant)
        / f"seed_{require_training_seed(seed)}"
        / f"mode_{require_mode(mode)}"
    )


def bundle_dir(variant: str, seed: int, mode: int) -> Path:
    return (
        SPECIALIST_BUNDLE_ROOT / require_variant(variant)
        / f"seed_{require_training_seed(seed)}"
        / f"mode_{require_mode(mode)}"
    )


def bundle_manifest(variant: str, seed: int, mode: int) -> Path:
    return bundle_dir(variant, seed, mode) / "bundle_manifest.json"


def bundle_required_paths(
    variant: str, seed: int, mode: int,
) -> tuple[Path, ...]:
    directory = bundle_dir(variant, seed, mode)
    return (
        directory / "bundle_manifest.json",
        directory / "policy" / POLICY_NAME,
        directory / "logs" / "protocol_signature.json",
        directory / "provenance" / BOOTSTRAP_NAME,
    )


def audit_dir(seed: int) -> Path:
    return AUDIT_ROOT / f"seed_{require_training_seed(seed)}"


def audit_result(seed: int) -> Path:
    return audit_dir(seed) / "audit.json"


def audit_manifest(seed: int) -> Path:
    return audit_dir(seed) / "audit_manifest.json"


def analysis_json() -> Path:
    return ANALYSIS_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return ANALYSIS_ROOT / "analysis.md"


def source_identity(seed: int) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "fresh_robust_sac_source",
        "env": ENV,
        "family": FAMILY,
        "training_seed": require_training_seed(seed),
        "algo": "sac",
        "final_next_iteration": SOURCE_NEXT_ITERATION,
        "final_total_steps": SOURCE_TOTAL_STEPS,
        "final_update_count": SOURCE_UPDATE_COUNT,
    }


def identity(variant: str, seed: int, mode: int) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "actor_only_robust_warmstart_fixed_mode_specialist",
        "variant": require_variant(variant),
        "env": ENV,
        "family": FAMILY,
        "training_seed": require_training_seed(seed),
        "fixed_mode": require_mode(mode),
        "algo": "sac",
        "source_next_iteration": SOURCE_NEXT_ITERATION,
        "final_next_iteration": FINAL_NEXT_ITERATION,
        "final_total_steps": FINAL_TOTAL_STEPS,
        "final_update_count": FINAL_UPDATE_COUNT,
    }


def expected_source_checkpoint() -> dict[str, Any]:
    return {
        "iteration": SOURCE_ITERATION,
        "next_iteration": SOURCE_NEXT_ITERATION,
        "total_steps": SOURCE_TOTAL_STEPS,
        "update_count": SOURCE_UPDATE_COUNT,
        "algo": "sac",
    }


def expected_checkpoint() -> dict[str, Any]:
    return {
        "iteration": FINAL_ITERATION,
        "next_iteration": FINAL_NEXT_ITERATION,
        "total_steps": FINAL_TOTAL_STEPS,
        "update_count": FINAL_UPDATE_COUNT,
        "algo": "sac",
    }


def bundle_records(seed: int) -> dict[str, Any]:
    return {
        str(mode): file_record(bundle_manifest("actor_only", seed, mode))
        for mode in MODES
    }


def estimator_records() -> dict[str, Any]:
    return {
        "v5": {
            "manifest": file_record(v5_parent.MODEL_MANIFEST),
            "parameters": file_record(v5_parent.MODEL_PATH),
        },
        "v16": estimator_parent.model_records(),
    }


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def registration_source_paths() -> tuple[Path, ...]:
    paths: list[Path] = [
        ROOT / "jax_experiments/analysis/regime_polarity_fresh_bank_estimator_confirmation_v17.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_robust_source_v17.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_robust_warmstart_specialist_v17.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_fresh_bank_estimator_confirmation_audit_v17.py",
        ROOT / "jax_experiments/analysis/analyze_regime_polarity_fresh_bank_estimator_confirmation_v17.py",
        policy_parent.REGISTRATION_PATH,
        policy_parent.analysis_json(),
        policy_parent.analysis_markdown(),
        policy_parent.REPORT,
        estimator_parent.REGISTRATION_PATH,
        estimator_parent.analysis_json(),
        estimator_parent.analysis_markdown(),
        estimator_parent.REPORT,
        estimator_parent.MODEL_MANIFEST,
        estimator_parent.MODEL_PATH,
        v5_parent.MODEL_MANIFEST,
        v5_parent.MODEL_PATH,
        v5_parent.REPORT,
        PREREG_REPORT,
    ]
    paths.extend(policy_parent.registration_source_paths())
    return tuple(dict.fromkeys(path.resolve() for path in paths))


def registration_payload() -> dict[str, Any]:
    policy_parent.validate_registration()
    estimator_parent.validate_registration()
    policy_analysis = read_json(policy_parent.analysis_json())
    estimator_analysis = read_json(estimator_parent.analysis_json())
    if (
        policy_analysis.get("confirmed") is not True
        or policy_analysis.get("seed_passes", 0) < policy_parent.REQUIRED_SEED_PASSES
    ):
        raise ValueError("v12 did not confirm the actor-only training protocol")
    if (
        estimator_analysis.get("estimator_development_pass") is not True
        or estimator_analysis.get("fresh_policy_bank_confirmation_authorized")
        is not True
    ):
        raise ValueError("v16 did not authorize fresh-bank confirmation")
    paths = registration_source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"v17 registration sources missing: {missing}")
    return {
        "schema": REGISTRATION_SCHEMA,
        "status": "registered",
        "created_before_training": True,
        "identity": {
            "protocol_version": PROTOCOL_VERSION,
            "variant": "actor_only",
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
        },
        "training": {
            "robust_source_steps": SOURCE_TOTAL_STEPS,
            "specialist_finetune_steps": FINETUNE_ITERS * SAMPLES_PER_ITER,
            "specialist_final_steps": FINAL_TOTAL_STEPS,
            "initialization": (
                "copy robust actor only; fresh critic, target, alpha, "
                "optimizers, replay"
            ),
            "estimator_updates": False,
        },
        "confirmation_gate": {
            "minimum_calibration_gain": MIN_CALIBRATION_GAIN,
            "minimum_holdout_mode_gain": MIN_HOLDOUT_MODE_GAIN,
            "minimum_holdout_mode_wins": MIN_HOLDOUT_MODE_WINS,
            "minimum_safe_oracle_gain": MIN_HEADROOM_GAIN,
            "minimum_delay_4_headroom_retention": MIN_CAUSAL_RETENTION,
            "minimum_v16_gain": MIN_ESTIMATOR_GAIN,
            "minimum_v16_headroom_recovery": MIN_CAUSAL_RETENTION,
            "required_policy_bank_passes": REQUIRED_SEED_PASSES,
            "required_strict_v16_passes": REQUIRED_SEED_PASSES,
            "required_v16_seed_wins_over_v5": REQUIRED_V16_SEED_WINS,
            "required_mechanism_seed_passes": REQUIRED_MECHANISM_SEED_PASSES,
            "require_positive_v16_minus_v5_ci95_lower": True,
            "require_all_switching_events_per_passing_seed": True,
            "require_zero_termination": True,
        },
        "mechanism_gate": {
            "switch_window_accuracy": "v16_not_lower_than_v5",
            "wrong_specialist_action_fraction": "v16_not_higher_than_v5",
        },
        "sync_policy": (
            "policy parameters and JSON provenance only; no full checkpoint"
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
            raise ValueError("existing v17 registration changed")
    else:
        write_json_atomic(REGISTRATION_PATH, payload)
    return validate_registration()


def validate_registration() -> dict[str, Any]:
    if not REGISTRATION_PATH.is_file():
        raise FileNotFoundError(f"missing v17 registration: {REGISTRATION_PATH}")
    payload = read_json(REGISTRATION_PATH)
    if payload != registration_payload():
        raise ValueError("v17 registration or source closure changed")
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
        raise ValueError("v17 calibration and holdout streams overlap")
    prior_policy_seeds = (
        set(policy_parent.TRAINING_SEEDS)
        | set(policy_parent.parent.TRAINING_SEEDS)
        | set(policy_parent.development.TRAINING_SEEDS)
    )
    if prior_policy_seeds & set(TRAINING_SEEDS):
        raise ValueError("v17 reused an earlier policy seed")
    prior_events = (
        set(policy_parent.CALIBRATION_EVENT_SEEDS)
        | set(policy_parent.STATIONARY_HOLDOUT_EVENT_SEEDS)
        | set(policy_parent.SWITCHING_EVENT_SEEDS)
        | set(estimator_parent.TRAIN_EVENT_SEEDS)
        | set(estimator_parent.VALIDATION_EVENT_SEEDS)
        | set(estimator_parent.AUDIT_EVENT_SEEDS)
    )
    if prior_events & set().union(*splits):
        raise ValueError("v17 reused an earlier evaluation stream")
    if set(SWITCHING_SCHEDULES) != set(SWITCHING_EVENT_SEEDS):
        raise ValueError("v17 switching schedules are incomplete")
    cycles = []
    for sequence in SWITCHING_SCHEDULES.values():
        if len(sequence) != len(MODES) or set(sequence) != set(MODES):
            raise ValueError("each v17 switching schedule must use every mode")
        cycles.append(tuple(sequence))
    if len(set(cycles)) != len(cycles):
        raise ValueError("v17 switching schedules are not distinct")


assert_protocol_integrity()
