"""Fresh-seed development protocol for specialist training stability."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_fresh_bank_estimator_confirmation_v17 as policy_parent,
)
from jax_experiments.analysis import (
    regime_polarity_robust_warmstart_specialist_v11 as historical,
)
from jax_experiments.analysis import (
    regime_polarity_v5_final_comparison_v18 as frozen_final,
)


ROOT = policy_parent.ROOT
PROTOCOL_VERSION = "v19-specialist-training-stability-development"
ENV = policy_parent.ENV
FAMILY = policy_parent.FAMILY
MODES = policy_parent.MODES
VARIANTS = ("actor_only_control", "full_state", "critic_warmup")
TRAINING_SEEDS = (82_003, 82_021, 82_039)

CALIBRATION_EVENT_SEEDS = (184_001, 184_017, 184_033)
STATIONARY_HOLDOUT_EVENT_SEEDS = (184_101, 184_117, 184_133)
SWITCHING_EVENT_SEEDS = (184_201, 184_217, 184_233)
SWITCHING_SCHEDULES = {
    184_201: (0, 2, 3, 1),
    184_217: (3, 1, 0, 2),
    184_233: (2, 0, 1, 3),
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

CRITIC_WARMUP_UPDATES = 25_000
ACTOR_UPDATE_AFTER = SOURCE_UPDATE_COUNT + CRITIC_WARMUP_UPDATES

DWELL_STEPS = policy_parent.DWELL_STEPS
MAX_EPISODE_STEPS = policy_parent.MAX_EPISODE_STEPS
EPISODES_PER_TASK = policy_parent.EPISODES_PER_TASK
SWITCHING_EPISODES = policy_parent.SWITCHING_EPISODES

MIN_CALIBRATION_GAIN = policy_parent.MIN_CALIBRATION_GAIN
MIN_HOLDOUT_MODE_GAIN = policy_parent.MIN_HOLDOUT_MODE_GAIN
MIN_HOLDOUT_MODE_WINS = policy_parent.MIN_HOLDOUT_MODE_WINS
MIN_SWITCHING_GAIN = policy_parent.MIN_SWITCHING_GAIN
REQUIRED_CELL_PASSES = len(TRAINING_SEEDS)
REQUIRED_PAIRED_SEED_WINS = 2
REQUIRED_PAIRED_EVENT_WINS = 6
MIN_STATIONARY_RETENTION = 0.95
CONTROL_VARIANT = "actor_only_control"

RUN_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_specialist_stability_v19"
)
SOURCE_RUN_ROOT = RUN_ROOT / "robust_sources"
SPECIALIST_RUN_ROOT = RUN_ROOT / "specialists"
SOURCE_BUNDLE_ROOT = (
    ROOT / "jax_experiments"
    / "eval_bundles_regime_polarity_robust_source_v19"
)
SPECIALIST_BUNDLE_ROOT = (
    ROOT / "jax_experiments"
    / "eval_bundles_regime_polarity_specialist_stability_v19"
)
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_specialist_stability_audit_v19"
)
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_specialist_stability_analysis_v19"
)
REGISTRATION_ROOT = (
    ROOT / "jax_experiments" / "deployments"
    / "regime_polarity_specialist_stability_v19"
)
REGISTRATION_PATH = REGISTRATION_ROOT / "registration.json"
PREREG_REPORT = (
    ROOT / "reports"
    / "regime_polarity_specialist_stability_v19_preregistration_2026-09-07.md"
)
REPORT = (
    ROOT / "reports"
    / "regime_polarity_specialist_stability_v19_2026-09-07.md"
)
V18_DIAGNOSTIC = (
    ROOT / "reports"
    / "regime_polarity_v5_final_comparison_v18_diagnostic_2026-09-07.md"
)

BOOTSTRAP_NAME = "warmstart_bootstrap.json"
POLICY_NAME = "policy_params.pkl"
CONTROLLER_STATE_NAME = "controller_state.pkl"
SOURCE_BUNDLE_SCHEMA = "bapr.robust-source-controller-bundle.v19"
BUNDLE_SCHEMA = "bapr.specialist-training-stability-policy-bundle.v19"
AUDIT_SCHEMA = "bapr.specialist-training-stability-audit.v19"
ANALYSIS_SCHEMA = "bapr.specialist-training-stability-analysis.v19"
REGISTRATION_SCHEMA = "bapr.specialist-training-stability-registration.v19"
BOOTSTRAP_SCHEMA = "bapr.specialist-training-stability-bootstrap.v19"

file_record = policy_parent.file_record
read_json = policy_parent.read_json
write_json_atomic = policy_parent.write_json_atomic
write_text_atomic = policy_parent.write_text_atomic
checkpoint_record = policy_parent.checkpoint_record


def require_variant(variant: str) -> str:
    variant = str(variant)
    if variant not in VARIANTS:
        raise ValueError(f"unknown v19 specialist variant {variant!r}")
    return variant


def require_training_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in TRAINING_SEEDS:
        raise ValueError(f"unknown v19 development seed {seed}")
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
        raise ValueError(f"unknown v19 switching event seed {seed}")
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
        "benchmark_role": "specialist_training_stability_development",
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
        "actor_update_after": (
            ACTOR_UPDATE_AFTER if variant == "critic_warmup" else None
        ),
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


def bundle_records(variant: str, seed: int) -> dict[str, Any]:
    return {
        str(mode): file_record(bundle_manifest(variant, seed, mode))
        for mode in MODES
    }


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def registration_source_paths() -> tuple[Path, ...]:
    paths = (
        ROOT / "jax_experiments/analysis/regime_polarity_specialist_stability_v19.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_robust_source_v19.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_specialist_stability_v19.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_specialist_stability_audit_v19.py",
        ROOT / "jax_experiments/analysis/analyze_regime_polarity_specialist_stability_v19.py",
        ROOT / "jax_experiments/analysis/train_regime_polarity_sac_actor_delay_v19.py",
        ROOT / "jax_experiments/algos/sac_actor_delay.py",
        ROOT / "scripts/submit_regime_polarity_specialist_stability_v19.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_robust_source_v12.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_robust_warmstart_specialist_v11.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_robust_warmstart_specialist_audit_v11.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_robust_warmstart_confirmation_audit_v12.py",
        ROOT / "jax_experiments/algos/sac_base.py",
        ROOT / "jax_experiments/common/checkpoint.py",
        ROOT / "jax_experiments/train.py",
        frozen_final.REGISTRATION_PATH,
        frozen_final.analysis_json(),
        frozen_final.analysis_markdown(),
        frozen_final.REPORT,
        V18_DIAGNOSTIC,
        historical.analysis_json(),
        historical.analysis_markdown(),
        historical.REPORT,
        PREREG_REPORT,
    )
    return tuple(dict.fromkeys(path.resolve() for path in paths))


def registration_payload() -> dict[str, Any]:
    frozen_final.validate_registration()
    final_analysis = read_json(frozen_final.analysis_json())
    if (
        final_analysis.get("standard_baseline_gate_pass") is not True
        or final_analysis.get("equal_policy_budget_gate_pass") is not False
        or final_analysis.get("strong_final_algorithm_claim") is not False
    ):
        raise ValueError("v18 does not authorize specialist stability work")
    paths = registration_source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"v19 registration sources missing: {missing}")
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
            "v18_candidate_and_holdouts_unchanged": True,
            "posterior_or_gate_changes": False,
            "environment_changes": False,
            "development_question": (
                "whether controller initialization reduces specialist-bank "
                "variance on fresh seeds"
            ),
        },
        "training": {
            "robust_source_steps": SOURCE_TOTAL_STEPS,
            "specialist_finetune_steps": FINETUNE_ITERS * SAMPLES_PER_ITER,
            "common": "empty specialist replay and reset Adam states",
            "actor_only_control": (
                "copy robust actor; fresh critic, target critic, and alpha"
            ),
            "full_state": (
                "copy robust actor, critic, target critic, and alpha"
            ),
            "critic_warmup": (
                "copy robust actor; fresh critic, target critic, and alpha; "
                f"freeze actor and alpha for {CRITIC_WARMUP_UPDATES} updates"
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
            "paired_against_actor_only_control": {
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
                "prefer full_state when within 2 percent"
            ),
        },
        "sync_policy": (
            "JSON, protocol signature, policy parameters, and compact source "
            "controller state only; never sync replay or full checkpoints"
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
            raise ValueError("existing v19 registration changed")
    else:
        write_json_atomic(REGISTRATION_PATH, payload)
    return validate_registration()


def validate_registration() -> dict[str, Any]:
    if not REGISTRATION_PATH.is_file():
        raise FileNotFoundError(f"missing v19 registration: {REGISTRATION_PATH}")
    payload = read_json(REGISTRATION_PATH)
    if payload != registration_payload():
        raise ValueError("v19 registration or source closure changed")
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
        raise ValueError("v19 calibration and holdout streams overlap")
    prior_training = (
        set(policy_parent.TRAINING_SEEDS) | set(historical.TRAINING_SEEDS)
    )
    if prior_training & set(TRAINING_SEEDS):
        raise ValueError("v19 reused an earlier policy seed")
    prior_events = (
        set(policy_parent.CALIBRATION_EVENT_SEEDS)
        | set(policy_parent.STATIONARY_HOLDOUT_EVENT_SEEDS)
        | set(policy_parent.SWITCHING_EVENT_SEEDS)
        | set(frozen_final.CALIBRATION_EVENT_SEEDS)
        | set(frozen_final.STATIONARY_HOLDOUT_EVENT_SEEDS)
        | set(frozen_final.SWITCHING_EVENT_SEEDS)
        | set(historical.CALIBRATION_EVENT_SEEDS)
        | set(historical.STATIONARY_HOLDOUT_EVENT_SEEDS)
        | set(historical.SWITCHING_EVENT_SEEDS)
    )
    if prior_events & set().union(*splits):
        raise ValueError("v19 reused an earlier evaluation stream")
    if set(SWITCHING_SCHEDULES) != set(SWITCHING_EVENT_SEEDS):
        raise ValueError("v19 switching schedules are incomplete")
    schedules = list(SWITCHING_SCHEDULES.values())
    if len(set(schedules)) != len(schedules):
        raise ValueError("v19 switching schedules are not distinct")
    if any(
        len(sequence) != len(MODES) or set(sequence) != set(MODES)
        for sequence in schedules
    ):
        raise ValueError("each v19 switching schedule must use every mode")
    if not (
        SOURCE_UPDATE_COUNT < ACTOR_UPDATE_AFTER < FINAL_UPDATE_COUNT
    ):
        raise ValueError("v19 critic warmup threshold is outside fine-tuning")


assert_protocol_integrity()
