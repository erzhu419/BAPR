"""Development protocol for robust-warm-started fixed-mode SAC controllers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_specialist_safe_utility_confirmation_v9 as parent,
)


ROOT = parent.ROOT
PROTOCOL_VERSION = "v11-robust-warmstart-specialist-development"
ENV = parent.ENV
FAMILY = parent.FAMILY
MODES = parent.MODES
ROLES = parent.ROLES
VARIANTS = ("full_state", "actor_only")
TRAINING_SEEDS = (61_039, 61_057)

# These streams are new development data. They are split before any v11 run:
# calibration selects only robust versus the matching specialist, stationary
# holdout tests that selection, and switching holdout tests the assembled bank.
CALIBRATION_EVENT_SEEDS = (157_001, 157_017, 157_033)
STATIONARY_HOLDOUT_EVENT_SEEDS = (157_101, 157_117, 157_133)
SWITCHING_EVENT_SEEDS = (157_201, 157_217, 157_233)

SOURCE_NEXT_ITERATION = parent.MAX_ITERS
SOURCE_TOTAL_STEPS = parent.FINAL_TOTAL_STEPS
SOURCE_UPDATE_COUNT = parent.FINAL_UPDATE_COUNT
FINETUNE_ITERS = 700
FINAL_NEXT_ITERATION = SOURCE_NEXT_ITERATION + FINETUNE_ITERS
FINAL_ITERATION = FINAL_NEXT_ITERATION - 1
SAMPLES_PER_ITER = parent.SAMPLES_PER_ITER
UPDATES_PER_ITER = parent.UPDATES_PER_ITER
FINAL_TOTAL_STEPS = SOURCE_TOTAL_STEPS + FINETUNE_ITERS * SAMPLES_PER_ITER
FINAL_UPDATE_COUNT = SOURCE_UPDATE_COUNT + FINETUNE_ITERS * UPDATES_PER_ITER

DWELL_STEPS = parent.DWELL_STEPS
MAX_EPISODE_STEPS = parent.MAX_EPISODE_STEPS
EPISODES_PER_TASK = 3
SWITCHING_EPISODES = 5
WARMUP_STEPS = parent.WARMUP_STEPS

MIN_CALIBRATION_GAIN = 0.05
MIN_HOLDOUT_MODE_GAIN = 0.05
MIN_HOLDOUT_MODE_WINS = 3
MIN_SWITCHING_GAIN = 0.10

RUN_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_robust_warmstart_specialist_v11"
    / "controllers"
)
BUNDLE_ROOT = (
    ROOT / "jax_experiments"
    / "eval_bundles_regime_polarity_robust_warmstart_specialist_v11"
)
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_robust_warmstart_specialist_audit_v11"
)
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_robust_warmstart_specialist_analysis_v11"
)
REGISTRATION_ROOT = (
    ROOT / "jax_experiments" / "deployments"
    / "regime_polarity_robust_warmstart_specialist_v11"
)
REGISTRATION_PATH = REGISTRATION_ROOT / "registration.json"
PREREG_REPORT = (
    ROOT / "reports"
    / "regime_polarity_robust_warmstart_specialist_v11_preregistration_2026-08-31.md"
)
REPORT = (
    ROOT / "reports"
    / "regime_polarity_robust_warmstart_specialist_v11_2026-08-31.md"
)

BOOTSTRAP_NAME = "warmstart_bootstrap.json"
POLICY_NAME = "policy_params.pkl"
BUNDLE_SCHEMA = "bapr.robust-warmstart-specialist-policy-bundle.v11"
AUDIT_SCHEMA = "bapr.robust-warmstart-specialist-audit.v11"
ANALYSIS_SCHEMA = "bapr.robust-warmstart-specialist-analysis.v11"
REGISTRATION_SCHEMA = "bapr.robust-warmstart-specialist-registration.v11"

file_record = parent.file_record
read_json = parent.read_json
write_json_atomic = parent.write_json_atomic
write_text_atomic = parent.write_text_atomic
checkpoint_record = parent.checkpoint_record


def require_variant(variant: str) -> str:
    variant = str(variant)
    if variant not in VARIANTS:
        raise ValueError(f"unknown v11 warm-start variant {variant!r}")
    return variant


def require_training_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in TRAINING_SEEDS:
        raise ValueError(f"unknown v11 development seed {seed}")
    return seed


require_seed = require_training_seed


def require_mode(mode: int) -> int:
    mode = int(mode)
    if mode not in MODES:
        raise ValueError(f"unknown actuator-polarity mode {mode}")
    return mode


def run_dir(variant: str, seed: int, mode: int) -> Path:
    return (
        RUN_ROOT / require_variant(variant)
        / f"seed_{require_training_seed(seed)}"
        / f"mode_{require_mode(mode)}"
    )


def bundle_dir(variant: str, seed: int, mode: int) -> Path:
    return (
        BUNDLE_ROOT / require_variant(variant)
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


def audit_dir(variant: str, seed: int) -> Path:
    return (
        AUDIT_ROOT / require_variant(variant)
        / f"seed_{require_training_seed(seed)}"
    )


def audit_result(variant: str, seed: int) -> Path:
    return audit_dir(variant, seed) / "audit.json"


def audit_manifest(variant: str, seed: int) -> Path:
    return audit_dir(variant, seed) / "audit_manifest.json"


def analysis_json() -> Path:
    return ANALYSIS_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return ANALYSIS_ROOT / "analysis.md"


def source_bundle(seed: int) -> Path:
    return parent.bundle_dir("robust_sac", require_training_seed(seed))


def source_manifest(seed: int) -> Path:
    return parent.bundle_manifest("robust_sac", require_training_seed(seed))


def identity(variant: str, seed: int, mode: int) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "robust_warmstart_fixed_mode_specialist",
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


def expected_checkpoint() -> dict[str, Any]:
    return {
        "iteration": FINAL_ITERATION,
        "next_iteration": FINAL_NEXT_ITERATION,
        "total_steps": FINAL_TOTAL_STEPS,
        "update_count": FINAL_UPDATE_COUNT,
        "algo": "sac",
    }


def bundle_records(variant: str, seed: int) -> dict[str, Any]:
    return {
        str(mode): file_record(bundle_manifest(variant, seed, mode))
        for mode in MODES
    }


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def registration_source_paths() -> tuple[Path, ...]:
    paths = (
        ROOT / "jax_experiments/analysis/regime_polarity_robust_warmstart_specialist_v11.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_robust_warmstart_specialist_v11.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_robust_warmstart_specialist_audit_v11.py",
        ROOT / "jax_experiments/analysis/analyze_regime_polarity_robust_warmstart_specialist_v11.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_specialist_expected_action_confirmation_audit_v6.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_specialist_capacity_diagnostic_v7.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_specialist_router_diagnostic_v2.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_specialist_safe_utility_audit_v8.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_specialist_safe_utility_confirmation_audit_v9.py",
        ROOT / "jax_experiments/common/checkpoint.py",
        ROOT / "jax_experiments/train.py",
        parent.REGISTRATION_PATH,
        *(source_manifest(seed) for seed in TRAINING_SEEDS),
        PREREG_REPORT,
    )
    return tuple(path.resolve() for path in paths)


def registration_payload() -> dict[str, Any]:
    paths = registration_source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"v11 registration sources missing: {missing}")
    return {
        "schema": REGISTRATION_SCHEMA,
        "status": "registered",
        "created_before_training": True,
        "identity": {
            "protocol_version": PROTOCOL_VERSION,
            "variants": list(VARIANTS),
            "training_seeds": list(TRAINING_SEEDS),
            "modes": list(MODES),
            "calibration_event_seeds": list(CALIBRATION_EVENT_SEEDS),
            "stationary_holdout_event_seeds": list(
                STATIONARY_HOLDOUT_EVENT_SEEDS),
            "switching_event_seeds": list(SWITCHING_EVENT_SEEDS),
        },
        "initialization": {
            "common": "matched robust SAC actor at 5.6M transitions",
            "full_state": (
                "copy actor, critic, target critic, and alpha; reset Adam "
                "states and replay"
            ),
            "actor_only": (
                "copy actor only; initialize critic, target critic, alpha, "
                "Adam states, and replay freshly"
            ),
            "finetune_iterations": FINETUNE_ITERS,
            "finetune_steps": FINETUNE_ITERS * SAMPLES_PER_ITER,
        },
        "gate": {
            "minimum_calibration_gain": MIN_CALIBRATION_GAIN,
            "minimum_holdout_mode_gain": MIN_HOLDOUT_MODE_GAIN,
            "minimum_holdout_mode_wins": MIN_HOLDOUT_MODE_WINS,
            "minimum_switching_gain": MIN_SWITCHING_GAIN,
            "required_seed_wins": len(TRAINING_SEEDS),
            "require_zero_termination": True,
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
            raise ValueError("existing v11 registration changed")
    else:
        write_json_atomic(REGISTRATION_PATH, payload)
    return validate_registration()


def validate_registration() -> dict[str, Any]:
    if not REGISTRATION_PATH.is_file():
        raise FileNotFoundError(f"missing v11 registration: {REGISTRATION_PATH}")
    payload = read_json(REGISTRATION_PATH)
    if payload != registration_payload():
        raise ValueError("v11 registration or source closure changed")
    return payload


def assert_split_integrity() -> None:
    splits = (
        set(CALIBRATION_EVENT_SEEDS),
        set(STATIONARY_HOLDOUT_EVENT_SEEDS),
        set(SWITCHING_EVENT_SEEDS),
    )
    if any(left & right for index, left in enumerate(splits)
           for right in splits[index + 1:]):
        raise ValueError("v11 calibration and holdout streams overlap")
    prior = set(parent.CALIBRATION_EVENT_SEEDS) | set(parent.HOLDOUT_EVENT_SEEDS)
    if prior & set().union(*splits):
        raise ValueError("v11 reused a v9 evaluation stream")


assert_split_integrity()
