"""Registered Ant headroom screen for the frozen V21 policy-bank recipe."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_specialist_policy_stability_v20 as shared,
)


ROOT = shared.ROOT
PROTOCOL_VERSION = "v22-ant-full-state-headroom-development"
ENV = "Ant-v2"
FAMILY = "actuator_polarity"
MODES = (0, 1, 2, 3)
VARIANTS = ("full_state_final",)
CONTROL_VARIANT = "full_state_final"
TRAINING_SEEDS = (85_003, 85_021, 85_039)

CALIBRATION_EVENT_SEEDS = (187_001, 187_017, 187_033)
STATIONARY_HOLDOUT_EVENT_SEEDS = (187_101, 187_117, 187_133)
SWITCHING_EVENT_SEEDS = (187_201, 187_217, 187_233)
SWITCHING_SCHEDULES = {
    187_201: (0, 3, 1, 2),
    187_217: (2, 0, 3, 1),
    187_233: (1, 3, 2, 0),
}

SOURCE_NEXT_ITERATION = 1_400
SOURCE_ITERATION = SOURCE_NEXT_ITERATION - 1
SAMPLES_PER_ITER = 4_000
UPDATES_PER_ITER = 250
SOURCE_TOTAL_STEPS = SOURCE_NEXT_ITERATION * SAMPLES_PER_ITER
SOURCE_UPDATE_COUNT = SOURCE_NEXT_ITERATION * UPDATES_PER_ITER
SOURCE_START_TRAIN_STEPS = 4_000
FINETUNE_ITERS = 700
FINAL_NEXT_ITERATION = SOURCE_NEXT_ITERATION + FINETUNE_ITERS
FINAL_ITERATION = FINAL_NEXT_ITERATION - 1
FINAL_TOTAL_STEPS = FINAL_NEXT_ITERATION * SAMPLES_PER_ITER
FINAL_UPDATE_COUNT = FINAL_NEXT_ITERATION * UPDATES_PER_ITER

DWELL_STEPS = 250
MAX_EPISODE_STEPS = 1_000
EPISODES_PER_TASK = 5
SWITCHING_EPISODES = 5

MIN_CALIBRATION_GAIN = 0.05
MIN_HOLDOUT_MODE_GAIN = 0.05
MIN_HOLDOUT_MODE_WINS = 3
MIN_SWITCHING_GAIN = 0.10
REQUIRED_CELL_PASSES = len(TRAINING_SEEDS)
REQUIRED_PAIRED_SEED_WINS = 3
REQUIRED_PAIRED_EVENT_WINS = 9
MIN_STATIONARY_RETENTION = 0.95

RUN_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_ant_full_state_headroom_v22"
)
SOURCE_RUN_ROOT = RUN_ROOT / "robust_sources"
SPECIALIST_RUN_ROOT = RUN_ROOT / "specialists"
SOURCE_BUNDLE_ROOT = (
    ROOT / "jax_experiments"
    / "eval_bundles_regime_polarity_ant_robust_source_v22"
)
SPECIALIST_BUNDLE_ROOT = (
    ROOT / "jax_experiments"
    / "eval_bundles_regime_polarity_ant_full_state_v22"
)
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_ant_full_state_headroom_audit_v22"
)
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_ant_full_state_headroom_analysis_v22"
)
REGISTRATION_ROOT = (
    ROOT / "jax_experiments" / "deployments"
    / "regime_polarity_ant_full_state_headroom_v22"
)
REGISTRATION_PATH = REGISTRATION_ROOT / "registration.json"
REPORT = (
    ROOT / "reports"
    / "regime_polarity_ant_full_state_headroom_v22_2026-09-10.md"
)

V21_ANALYSIS = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_full_state_final_confirmation_analysis_v21"
    / "analysis.json"
)
INITIAL_HEADROOM_ANALYSIS = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_headroom_analysis_v1" / "analysis.json"
)

POLICY_NAME = "policy_params.pkl"
CONTROLLER_STATE_NAME = "controller_state.pkl"
BOOTSTRAP_NAME = "warmstart_bootstrap.json"
SELECTION_NAME = "policy_selection.json"
SOURCE_BUNDLE_SCHEMA = "bapr.ant-robust-source-controller-bundle.v22"
BUNDLE_SCHEMA = "bapr.ant-full-state-specialist-bundle.v22"
BOOTSTRAP_SCHEMA = "bapr.ant-full-state-bootstrap.v22"
AUDIT_SCHEMA = "bapr.ant-full-state-headroom-audit.v22"
ANALYSIS_SCHEMA = "bapr.ant-full-state-headroom-analysis.v22"
REGISTRATION_SCHEMA = "bapr.ant-full-state-headroom-registration.v22"

file_record = shared.file_record
read_json = shared.read_json
write_json_atomic = shared.write_json_atomic
write_text_atomic = shared.write_text_atomic
checkpoint_record = shared.checkpoint_record


def require_variant(variant: str) -> str:
    value = str(variant)
    if value not in VARIANTS:
        raise ValueError(f"unknown Ant v22 specialist variant {value!r}")
    return value


def require_training_seed(seed: int) -> int:
    value = int(seed)
    if value not in TRAINING_SEEDS:
        raise ValueError(f"unknown Ant v22 development seed {value}")
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
        raise ValueError(f"unknown Ant v22 switching event seed {value}")
    return value


def switching_sequence(event_seed: int, episode: int) -> tuple[int, ...]:
    base = SWITCHING_SCHEDULES[require_switching_event_seed(event_seed)]
    shift = int(episode) % len(base)
    return tuple(base[shift:] + base[:shift])


def actor_update_period(variant: str) -> int:
    require_variant(variant)
    return 1


def select_best_validation(variant: str) -> bool:
    require_variant(variant)
    return False


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
    require_variant(variant)
    return (
        SPECIALIST_RUN_ROOT / f"seed_{require_training_seed(seed)}"
        / f"mode_{require_mode(mode)}"
    )


def bundle_dir(variant: str, seed: int, mode: int) -> Path:
    require_variant(variant)
    return (
        SPECIALIST_BUNDLE_ROOT / f"seed_{require_training_seed(seed)}"
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
    require_variant(variant)
    return AUDIT_ROOT / f"seed_{require_training_seed(seed)}"


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
        "benchmark_role": "ant_persistent_regime_robust_source",
        "env": ENV,
        "family": FAMILY,
        "training_seed": require_training_seed(seed),
        "algo": "sac",
        "final_next_iteration": SOURCE_NEXT_ITERATION,
        "final_total_steps": SOURCE_TOTAL_STEPS,
        "final_update_count": SOURCE_UPDATE_COUNT,
    }


def identity(variant: str, seed: int, mode: int) -> dict[str, Any]:
    require_variant(variant)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "ant_persistent_regime_full_state_specialist",
        "variant": CONTROL_VARIANT,
        "env": ENV,
        "family": FAMILY,
        "training_seed": require_training_seed(seed),
        "fixed_mode": require_mode(mode),
        "algo": "sac",
        "source_next_iteration": SOURCE_NEXT_ITERATION,
        "final_next_iteration": FINAL_NEXT_ITERATION,
        "final_total_steps": FINAL_TOTAL_STEPS,
        "final_update_count": FINAL_UPDATE_COUNT,
        "actor_update_period": 1,
        "select_best_validation": False,
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
        Path(__file__).resolve(),
        ROOT / "jax_experiments/analysis/run_regime_polarity_ant_robust_source_v22.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_ant_full_state_specialist_v22.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_ant_full_state_audit_v22.py",
        ROOT / "jax_experiments/analysis/analyze_regime_polarity_ant_full_state_headroom_v22.py",
        ROOT / "scripts/submit_regime_polarity_ant_full_state_headroom_v22.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_robust_source_v19.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_specialist_policy_stability_v20.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_specialist_stability_audit_v19.py",
        ROOT / "jax_experiments/algos/sac_policy_stability.py",
        ROOT / "jax_experiments/common/checkpoint.py",
        ROOT / "jax_experiments/envs/stochastic_mode_env.py",
        V21_ANALYSIS,
        INITIAL_HEADROOM_ANALYSIS,
    )
    return tuple(dict.fromkeys(path.resolve() for path in paths))


def registration_payload() -> dict[str, Any]:
    paths = registration_source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Ant v22 registration sources missing: {missing}")
    v21 = read_json(V21_ANALYSIS)
    ant_initial = read_json(INITIAL_HEADROOM_ANALYSIS)["environments"][ENV]
    if v21.get("strong_final_algorithm_claim") is not True:
        raise ValueError("V21 does not authorize frozen-recipe transfer")
    if ant_initial.get("switching_relative_gain", 0.0) < 0.15:
        raise ValueError("initial Ant screen lacks transfer headroom")
    return {
        "schema": REGISTRATION_SCHEMA,
        "status": "registered",
        "created_before_training": True,
        "identity": {
            "protocol_version": PROTOCOL_VERSION,
            "environment": ENV,
            "family": FAMILY,
            "variant": CONTROL_VARIANT,
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
            "halfcheetah_v21_untouched": True,
            "same_environment_family_and_dwell": True,
            "same_robust_and_full_state_specialist_recipe": True,
            "estimator_or_router_training": False,
            "question": (
                "whether Ant has stable robust-inclusive true-mode policy-bank "
                "headroom before porting the frozen causal estimator"
            ),
        },
        "budget": {
            "robust_steps": SOURCE_TOTAL_STEPS,
            "specialist_additional_steps": FINETUNE_ITERS * SAMPLES_PER_ITER,
            "specialist_initialization": (
                "copy actor, critic, target critic, and alpha from the matched "
                "robust source; reset optimizer and replay"
            ),
        },
        "gate": {
            "all_three_policy_seeds_pass": True,
            "minimum_stationary_mode_wins_per_seed": MIN_HOLDOUT_MODE_WINS,
            "minimum_per_mode_gain": MIN_HOLDOUT_MODE_GAIN,
            "minimum_switching_gain": MIN_SWITCHING_GAIN,
            "all_three_switching_events_win": True,
            "zero_termination": True,
        },
        "sync_policy": (
            "compact policy/controller bundles and JSON only; never sync replay "
            "or full checkpoints"
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
            raise ValueError("existing Ant v22 registration changed")
    else:
        write_json_atomic(REGISTRATION_PATH, payload)
    return validate_registration()


def validate_registration() -> dict[str, Any]:
    if not REGISTRATION_PATH.is_file():
        raise FileNotFoundError(f"missing Ant v22 registration: {REGISTRATION_PATH}")
    payload = read_json(REGISTRATION_PATH)
    if payload != registration_payload():
        raise ValueError("Ant v22 registration or source closure changed")
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
        raise ValueError("Ant v22 evaluation splits overlap")
    if set(SWITCHING_SCHEDULES) != set(SWITCHING_EVENT_SEEDS):
        raise ValueError("Ant v22 switching schedules are incomplete")
    if len(set(SWITCHING_SCHEDULES.values())) != len(SWITCHING_SCHEDULES):
        raise ValueError("Ant v22 switching schedules are not distinct")
    if any(
        len(sequence) != len(MODES) or set(sequence) != set(MODES)
        for sequence in SWITCHING_SCHEDULES.values()
    ):
        raise ValueError("each Ant v22 schedule must use every mode")


assert_protocol_integrity()
