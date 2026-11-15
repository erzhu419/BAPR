"""Fresh-seed full-state specialist policy-stability protocol."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_specialist_stability_v19 as parent,
)


ROOT = parent.ROOT
PROTOCOL_VERSION = "v20-full-state-policy-stability-development"
ENV = parent.ENV
FAMILY = parent.FAMILY
MODES = parent.MODES
VARIANTS = (
    "full_state_final",
    "full_state_best",
    "period2_best",
)
CONTROL_VARIANT = "full_state_final"
TRAINING_SEEDS = (83_003, 83_021, 83_039)

CALIBRATION_EVENT_SEEDS = (185_001, 185_017, 185_033)
STATIONARY_HOLDOUT_EVENT_SEEDS = (185_101, 185_117, 185_133)
SWITCHING_EVENT_SEEDS = (185_201, 185_217, 185_233)
SWITCHING_SCHEDULES = {
    185_201: (1, 0, 3, 2),
    185_217: (2, 3, 1, 0),
    185_233: (3, 2, 0, 1),
}

SOURCE_NEXT_ITERATION = parent.SOURCE_NEXT_ITERATION
SOURCE_ITERATION = parent.SOURCE_ITERATION
SOURCE_TOTAL_STEPS = parent.SOURCE_TOTAL_STEPS
SOURCE_UPDATE_COUNT = parent.SOURCE_UPDATE_COUNT
SOURCE_START_TRAIN_STEPS = parent.SOURCE_START_TRAIN_STEPS
FINETUNE_ITERS = parent.FINETUNE_ITERS
FINAL_NEXT_ITERATION = parent.FINAL_NEXT_ITERATION
FINAL_ITERATION = parent.FINAL_ITERATION
SAMPLES_PER_ITER = parent.SAMPLES_PER_ITER
UPDATES_PER_ITER = parent.UPDATES_PER_ITER
FINAL_TOTAL_STEPS = parent.FINAL_TOTAL_STEPS
FINAL_UPDATE_COUNT = parent.FINAL_UPDATE_COUNT

DWELL_STEPS = parent.DWELL_STEPS
MAX_EPISODE_STEPS = parent.MAX_EPISODE_STEPS
EPISODES_PER_TASK = parent.EPISODES_PER_TASK
SWITCHING_EPISODES = parent.SWITCHING_EPISODES

MIN_CALIBRATION_GAIN = parent.MIN_CALIBRATION_GAIN
MIN_HOLDOUT_MODE_GAIN = parent.MIN_HOLDOUT_MODE_GAIN
MIN_HOLDOUT_MODE_WINS = parent.MIN_HOLDOUT_MODE_WINS
MIN_SWITCHING_GAIN = parent.MIN_SWITCHING_GAIN
REQUIRED_CELL_PASSES = len(TRAINING_SEEDS)
REQUIRED_PAIRED_SEED_WINS = parent.REQUIRED_PAIRED_SEED_WINS
REQUIRED_PAIRED_EVENT_WINS = parent.REQUIRED_PAIRED_EVENT_WINS
MIN_STATIONARY_RETENTION = parent.MIN_STATIONARY_RETENTION

ACTOR_UPDATE_PERIOD = {
    "full_state_final": 1,
    "full_state_best": 1,
    "period2_best": 2,
}
SELECT_BEST_VALIDATION = {
    "full_state_final": False,
    "full_state_best": True,
    "period2_best": True,
}

RUN_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_specialist_policy_stability_v20"
)
SOURCE_RUN_ROOT = RUN_ROOT / "robust_sources"
SPECIALIST_RUN_ROOT = RUN_ROOT / "specialists"
SOURCE_BUNDLE_ROOT = (
    ROOT / "jax_experiments"
    / "eval_bundles_regime_polarity_robust_source_v20"
)
SPECIALIST_BUNDLE_ROOT = (
    ROOT / "jax_experiments"
    / "eval_bundles_regime_polarity_specialist_policy_stability_v20"
)
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_specialist_policy_stability_audit_v20"
)
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_specialist_policy_stability_analysis_v20"
)
REGISTRATION_ROOT = (
    ROOT / "jax_experiments" / "deployments"
    / "regime_polarity_specialist_policy_stability_v20"
)
REGISTRATION_PATH = REGISTRATION_ROOT / "registration.json"
PREREG_REPORT = (
    ROOT / "reports"
    / "regime_polarity_specialist_policy_stability_v20_preregistration_2026-09-08.md"
)
REPORT = (
    ROOT / "reports"
    / "regime_polarity_specialist_policy_stability_v20_2026-09-08.md"
)
V19_DIAGNOSTIC = (
    ROOT / "reports"
    / "regime_polarity_specialist_stability_v19_diagnostic_2026-09-08.md"
)

BOOTSTRAP_NAME = parent.BOOTSTRAP_NAME
POLICY_NAME = parent.POLICY_NAME
CONTROLLER_STATE_NAME = parent.CONTROLLER_STATE_NAME
SELECTION_NAME = "policy_selection.json"
SOURCE_BUNDLE_SCHEMA = "bapr.robust-source-controller-bundle.v20"
BUNDLE_SCHEMA = "bapr.specialist-policy-stability-bundle.v20"
AUDIT_SCHEMA = "bapr.specialist-policy-stability-audit.v20"
ANALYSIS_SCHEMA = "bapr.specialist-policy-stability-analysis.v20"
REGISTRATION_SCHEMA = "bapr.specialist-policy-stability-registration.v20"
BOOTSTRAP_SCHEMA = "bapr.specialist-policy-stability-bootstrap.v20"

file_record = parent.file_record
read_json = parent.read_json
write_json_atomic = parent.write_json_atomic
write_text_atomic = parent.write_text_atomic
checkpoint_record = parent.checkpoint_record


def require_variant(variant: str) -> str:
    value = str(variant)
    if value not in VARIANTS:
        raise ValueError(f"unknown v20 specialist variant {value!r}")
    return value


def require_training_seed(seed: int) -> int:
    value = int(seed)
    if value not in TRAINING_SEEDS:
        raise ValueError(f"unknown v20 development seed {value}")
    return value


require_seed = require_training_seed


def require_mode(mode: int) -> int:
    value = int(mode)
    if value not in MODES:
        raise ValueError(f"unknown actuator-polarity mode {value}")
    return value


def require_switching_event_seed(seed: int) -> int:
    value = int(seed)
    if value not in SWITCHING_EVENT_SEEDS:
        raise ValueError(f"unknown v20 switching event seed {value}")
    return value


def switching_sequence(event_seed: int, episode: int) -> tuple[int, ...]:
    base = SWITCHING_SCHEDULES[require_switching_event_seed(event_seed)]
    shift = int(episode) % len(base)
    return tuple(base[shift:] + base[:shift])


def actor_update_period(variant: str) -> int:
    return ACTOR_UPDATE_PERIOD[require_variant(variant)]


def select_best_validation(variant: str) -> bool:
    return SELECT_BEST_VALIDATION[require_variant(variant)]


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
        directory / "controller" / CONTROLLER_STATE_NAME,
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
        directory / "provenance" / SELECTION_NAME,
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


def source_identity(seed: int) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "fresh_robust_sac_controller_source",
        "env": ENV,
        "family": FAMILY,
        "training_seed": require_training_seed(seed),
        "algo": "sac",
        "final_next_iteration": SOURCE_NEXT_ITERATION,
        "final_total_steps": SOURCE_TOTAL_STEPS,
        "final_update_count": SOURCE_UPDATE_COUNT,
    }


def identity(variant: str, seed: int, mode: int) -> dict[str, Any]:
    variant = require_variant(variant)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "specialist_policy_stability_development",
        "variant": variant,
        "env": ENV,
        "family": FAMILY,
        "training_seed": require_training_seed(seed),
        "fixed_mode": require_mode(mode),
        "algo": "sac",
        "source_next_iteration": SOURCE_NEXT_ITERATION,
        "final_next_iteration": FINAL_NEXT_ITERATION,
        "final_total_steps": FINAL_TOTAL_STEPS,
        "final_update_count": FINAL_UPDATE_COUNT,
        "actor_update_period": actor_update_period(variant),
        "select_best_validation": select_best_validation(variant),
    }


def expected_source_checkpoint() -> dict[str, Any]:
    return parent.expected_source_checkpoint()


def expected_checkpoint() -> dict[str, Any]:
    return parent.expected_checkpoint()


def bundle_records(variant: str, seed: int) -> dict[str, Any]:
    return {
        str(mode): file_record(bundle_manifest(variant, seed, mode))
        for mode in MODES
    }


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def registration_source_paths() -> tuple[Path, ...]:
    paths = (
        ROOT / "jax_experiments/analysis/regime_polarity_specialist_policy_stability_v20.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_robust_source_v20.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_specialist_policy_stability_v20.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_specialist_policy_stability_audit_v20.py",
        ROOT / "jax_experiments/analysis/analyze_regime_polarity_specialist_policy_stability_v20.py",
        ROOT / "jax_experiments/analysis/train_regime_polarity_sac_policy_stability_v20.py",
        ROOT / "jax_experiments/algos/sac_policy_stability.py",
        ROOT / "scripts/submit_regime_polarity_specialist_policy_stability_v20.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_robust_source_v19.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_specialist_stability_audit_v19.py",
        ROOT / "jax_experiments/analysis/analyze_regime_polarity_specialist_stability_v19.py",
        parent.REGISTRATION_PATH,
        parent.analysis_json(),
        parent.analysis_markdown(),
        parent.REPORT,
        V19_DIAGNOSTIC,
        PREREG_REPORT,
    )
    return tuple(dict.fromkeys(path.resolve() for path in paths))


def registration_payload() -> dict[str, Any]:
    parent.validate_registration()
    prior = read_json(parent.analysis_json())
    full_state = prior["comparisons"]["full_state"]
    if (
        prior.get("selected_candidate") is not None
        or full_state.get("mean_safe_switching_delta", 0.0) <= 0.0
        or full_state.get("paired_event_wins", 0) < 6
        or full_state.get("checks", {}).get("stationary_retention") is not False
        or prior["comparisons"]["critic_warmup"].get("pass") is not False
    ):
        raise ValueError("v19 does not authorize the v20 stability screen")
    paths = registration_source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"v20 registration sources missing: {missing}")
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
            "switching_schedules": {
                str(seed): list(sequence)
                for seed, sequence in SWITCHING_SCHEDULES.items()
            },
        },
        "frozen_boundary": {
            "environment_posterior_router_unchanged": True,
            "v18_and_v19_holdouts_untouched": True,
            "all_specialists_copy_full_controller_state": True,
            "development_question": (
                "whether validation checkpoint selection and a lower actor "
                "update frequency prevent post-fork policy erosion"
            ),
        },
        "training": {
            "robust_source_steps": SOURCE_TOTAL_STEPS,
            "specialist_finetune_steps": FINETUNE_ITERS * SAMPLES_PER_ITER,
            "common": (
                "copy actor, critic, target critic, and alpha; reset Adam "
                "states and replay"
            ),
            "variants": {
                variant: {
                    "actor_update_period": actor_update_period(variant),
                    "policy_selection": (
                        "best independent training-validation evaluation"
                        if select_best_validation(variant) else "final"
                    ),
                }
                for variant in VARIANTS
            },
            "validation": (
                "fixed-mode trainer evaluation stream with seed offset 1000; "
                "never reused for utility calibration or holdout reporting"
            ),
        },
        "development_gate": {
            "robust_relative_cell": {
                "minimum_holdout_mode_gain": MIN_HOLDOUT_MODE_GAIN,
                "minimum_holdout_mode_wins": MIN_HOLDOUT_MODE_WINS,
                "minimum_safe_switching_gain": MIN_SWITCHING_GAIN,
                "required_cell_passes": REQUIRED_CELL_PASSES,
                "require_zero_termination": True,
            },
            "paired_against_full_state_final": {
                "require_positive_mean_switching_delta": True,
                "required_seed_wins_of_3": REQUIRED_PAIRED_SEED_WINS,
                "required_event_wins_of_9": REQUIRED_PAIRED_EVENT_WINS,
                "require_better_worst_seed_safe_gain": True,
                "minimum_stationary_return_retention": (
                    MIN_STATIONARY_RETENTION),
                "do_not_reduce_total_holdout_mode_passes": True,
            },
            "selection": (
                "highest mean safe switching return among passing candidates; "
                "prefer full_state_best when within 2 percent"
            ),
        },
        "sync_policy": (
            "JSON, protocol signature, policy parameters, selection record, "
            "and compact source controller state only; never sync replay or "
            "full checkpoints"
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
            raise ValueError("existing v20 registration changed")
    else:
        write_json_atomic(REGISTRATION_PATH, payload)
    return validate_registration()


def validate_registration() -> dict[str, Any]:
    if not REGISTRATION_PATH.is_file():
        raise FileNotFoundError(f"missing v20 registration: {REGISTRATION_PATH}")
    payload = read_json(REGISTRATION_PATH)
    if payload != registration_payload():
        raise ValueError("v20 registration or source closure changed")
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
        raise ValueError("v20 calibration and holdout streams overlap")
    if set(TRAINING_SEEDS) & set(parent.TRAINING_SEEDS):
        raise ValueError("v20 reused a v19 policy seed")
    prior_events = (
        set(parent.CALIBRATION_EVENT_SEEDS)
        | set(parent.STATIONARY_HOLDOUT_EVENT_SEEDS)
        | set(parent.SWITCHING_EVENT_SEEDS)
    )
    if prior_events & set().union(*splits):
        raise ValueError("v20 reused a v19 evaluation stream")
    if set(SWITCHING_SCHEDULES) != set(SWITCHING_EVENT_SEEDS):
        raise ValueError("v20 switching schedules are incomplete")
    schedules = list(SWITCHING_SCHEDULES.values())
    if len(set(schedules)) != len(schedules):
        raise ValueError("v20 switching schedules are not distinct")
    if any(
        len(sequence) != len(MODES) or set(sequence) != set(MODES)
        for sequence in schedules
    ):
        raise ValueError("each v20 switching schedule must use every mode")
    if set(ACTOR_UPDATE_PERIOD) != set(VARIANTS):
        raise ValueError("v20 actor periods are incomplete")
    if set(SELECT_BEST_VALIDATION) != set(VARIANTS):
        raise ValueError("v20 selection rules are incomplete")


assert_protocol_integrity()
