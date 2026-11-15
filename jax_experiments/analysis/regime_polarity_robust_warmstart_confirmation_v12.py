"""Frozen five-seed confirmation of actor-only robust warm-start specialists."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_robust_warmstart_specialist_v11 as development,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_safe_utility_confirmation_v9 as parent,
)


ROOT = parent.ROOT
PROTOCOL_VERSION = "v12-actor-only-warmstart-confirmation"
ENV = parent.ENV
FAMILY = parent.FAMILY
MODES = parent.MODES
VARIANTS = ("actor_only",)
TRAINING_SEEDS = (71_003, 71_021, 71_039, 71_057, 71_079)

CALIBRATION_EVENT_SEEDS = (167_001, 167_017, 167_033)
STATIONARY_HOLDOUT_EVENT_SEEDS = (167_101, 167_117, 167_133)
SWITCHING_EVENT_SEEDS = (167_201, 167_217, 167_233)
SWITCHING_SCHEDULES = {
    167_201: (0, 1, 2, 3),
    167_217: (1, 3, 0, 2),
    167_233: (2, 0, 3, 1),
}

SOURCE_NEXT_ITERATION = parent.MAX_ITERS
SOURCE_ITERATION = SOURCE_NEXT_ITERATION - 1
SOURCE_TOTAL_STEPS = parent.FINAL_TOTAL_STEPS
SOURCE_UPDATE_COUNT = parent.FINAL_UPDATE_COUNT
SOURCE_START_TRAIN_STEPS = parent.START_TRAIN_STEPS
FINETUNE_ITERS = development.FINETUNE_ITERS
FINAL_NEXT_ITERATION = SOURCE_NEXT_ITERATION + FINETUNE_ITERS
FINAL_ITERATION = FINAL_NEXT_ITERATION - 1
SAMPLES_PER_ITER = parent.SAMPLES_PER_ITER
UPDATES_PER_ITER = parent.UPDATES_PER_ITER
FINAL_TOTAL_STEPS = SOURCE_TOTAL_STEPS + FINETUNE_ITERS * SAMPLES_PER_ITER
FINAL_UPDATE_COUNT = SOURCE_UPDATE_COUNT + FINETUNE_ITERS * UPDATES_PER_ITER

DWELL_STEPS = parent.DWELL_STEPS
MAX_EPISODE_STEPS = parent.MAX_EPISODE_STEPS
EPISODES_PER_TASK = development.EPISODES_PER_TASK
SWITCHING_EPISODES = development.SWITCHING_EPISODES

MIN_CALIBRATION_GAIN = development.MIN_CALIBRATION_GAIN
MIN_HOLDOUT_MODE_GAIN = development.MIN_HOLDOUT_MODE_GAIN
MIN_HOLDOUT_MODE_WINS = development.MIN_HOLDOUT_MODE_WINS
MIN_SWITCHING_GAIN = development.MIN_SWITCHING_GAIN
REQUIRED_SEED_PASSES = 4

RUN_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_robust_warmstart_confirmation_v12"
)
SOURCE_RUN_ROOT = RUN_ROOT / "robust_sources"
SPECIALIST_RUN_ROOT = RUN_ROOT / "specialists"
SOURCE_BUNDLE_ROOT = (
    ROOT / "jax_experiments"
    / "eval_bundles_regime_polarity_robust_source_v12"
)
SPECIALIST_BUNDLE_ROOT = (
    ROOT / "jax_experiments"
    / "eval_bundles_regime_polarity_robust_warmstart_confirmation_v12"
)
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_robust_warmstart_confirmation_audit_v12"
)
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_robust_warmstart_confirmation_analysis_v12"
)
REGISTRATION_ROOT = (
    ROOT / "jax_experiments" / "deployments"
    / "regime_polarity_robust_warmstart_confirmation_v12"
)
REGISTRATION_PATH = REGISTRATION_ROOT / "registration.json"
PREREG_REPORT = (
    ROOT / "reports"
    / "regime_polarity_robust_warmstart_confirmation_v12_preregistration_2026-08-31.md"
)
REPORT = (
    ROOT / "reports"
    / "regime_polarity_robust_warmstart_confirmation_v12_2026-08-31.md"
)

BOOTSTRAP_NAME = "warmstart_bootstrap.json"
POLICY_NAME = "policy_params.pkl"
SOURCE_BUNDLE_SCHEMA = "bapr.robust-source-policy-bundle.v12"
BUNDLE_SCHEMA = "bapr.robust-warmstart-specialist-policy-bundle.v12"
AUDIT_SCHEMA = "bapr.robust-warmstart-specialist-audit.v12"
ANALYSIS_SCHEMA = "bapr.robust-warmstart-specialist-analysis.v12"
REGISTRATION_SCHEMA = "bapr.robust-warmstart-specialist-registration.v12"

file_record = parent.file_record
read_json = parent.read_json
write_json_atomic = parent.write_json_atomic
write_text_atomic = parent.write_text_atomic
checkpoint_record = parent.checkpoint_record


def require_variant(variant: str) -> str:
    variant = str(variant)
    if variant not in VARIANTS:
        raise ValueError(f"unknown v12 variant {variant!r}")
    return variant


def require_training_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in TRAINING_SEEDS:
        raise ValueError(f"unknown v12 confirmation seed {seed}")
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
        raise ValueError(f"unknown v12 switching event seed {seed}")
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


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def registration_source_paths() -> tuple[Path, ...]:
    paths = (
        ROOT / "jax_experiments/analysis/regime_polarity_robust_warmstart_confirmation_v12.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_robust_source_v12.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_robust_warmstart_specialist_v12.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_robust_warmstart_confirmation_audit_v12.py",
        ROOT / "jax_experiments/analysis/analyze_regime_polarity_robust_warmstart_confirmation_v12.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_robust_warmstart_specialist_v11.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_robust_warmstart_specialist_audit_v11.py",
        ROOT / "jax_experiments/analysis/analyze_regime_polarity_robust_warmstart_specialist_v11.py",
        ROOT / "jax_experiments/common/checkpoint.py",
        ROOT / "jax_experiments/train.py",
        ROOT / "jax_experiments/envs/stochastic_mode_env.py",
        development.REGISTRATION_PATH,
        development.analysis_json(),
        development.REPORT,
        PREREG_REPORT,
    )
    return tuple(path.resolve() for path in paths)


def registration_payload() -> dict[str, Any]:
    paths = registration_source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"v12 registration sources missing: {missing}")
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
            "initialization": "copy robust actor only; fresh critic, target, alpha, optimizers, replay",
        },
        "gate": {
            "minimum_calibration_gain": MIN_CALIBRATION_GAIN,
            "minimum_holdout_mode_gain": MIN_HOLDOUT_MODE_GAIN,
            "minimum_holdout_mode_wins": MIN_HOLDOUT_MODE_WINS,
            "minimum_switching_gain": MIN_SWITCHING_GAIN,
            "required_seed_passes": REQUIRED_SEED_PASSES,
            "require_all_switching_events_per_passing_seed": True,
            "require_zero_termination": True,
            "require_distinct_switching_traces": True,
        },
        "sync_policy": "policy parameters and JSON provenance only; no full checkpoint",
        "source_records": {
            _relative(path): file_record(path) for path in paths
        },
    }


def create_registration() -> dict[str, Any]:
    payload = registration_payload()
    REGISTRATION_ROOT.mkdir(parents=True, exist_ok=True)
    if REGISTRATION_PATH.is_file():
        if read_json(REGISTRATION_PATH) != payload:
            raise ValueError("existing v12 registration changed")
    else:
        write_json_atomic(REGISTRATION_PATH, payload)
    return validate_registration()


def validate_registration() -> dict[str, Any]:
    if not REGISTRATION_PATH.is_file():
        raise FileNotFoundError(f"missing v12 registration: {REGISTRATION_PATH}")
    payload = read_json(REGISTRATION_PATH)
    if payload != registration_payload():
        raise ValueError("v12 registration or source closure changed")
    return payload


def assert_protocol_integrity() -> None:
    splits = (
        set(CALIBRATION_EVENT_SEEDS),
        set(STATIONARY_HOLDOUT_EVENT_SEEDS),
        set(SWITCHING_EVENT_SEEDS),
    )
    if any(left & right for index, left in enumerate(splits)
           for right in splits[index + 1:]):
        raise ValueError("v12 calibration and holdout streams overlap")
    prior_policy_seeds = set(parent.TRAINING_SEEDS) | set(development.TRAINING_SEEDS)
    if prior_policy_seeds & set(TRAINING_SEEDS):
        raise ValueError("v12 reused a development policy seed")
    prior_events = (
        set(parent.CALIBRATION_EVENT_SEEDS)
        | set(parent.HOLDOUT_EVENT_SEEDS)
        | set(development.CALIBRATION_EVENT_SEEDS)
        | set(development.STATIONARY_HOLDOUT_EVENT_SEEDS)
        | set(development.SWITCHING_EVENT_SEEDS)
    )
    if prior_events & set().union(*splits):
        raise ValueError("v12 reused an earlier evaluation stream")
    if set(SWITCHING_SCHEDULES) != set(SWITCHING_EVENT_SEEDS):
        raise ValueError("v12 switching schedules are incomplete")
    cycles = []
    for sequence in SWITCHING_SCHEDULES.values():
        if len(sequence) != len(MODES) or set(sequence) != set(MODES):
            raise ValueError("each v12 switching schedule must use every mode once")
        cycles.append(tuple(sequence))
    if len(set(cycles)) != len(cycles):
        raise ValueError("v12 switching schedules are not distinct")


assert_protocol_integrity()
