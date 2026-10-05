"""Frozen development protocol for canonical-controller training recipes."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_action_compensation_power_confirmation_v32 as parent,
)
from jax_experiments.analysis import (
    regime_polarity_action_compensation_v28 as mechanism,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_system_id_v5 as v5_model,
)


ROOT = parent.ROOT
PROTOCOL_VERSION = "v33-canonical-controller-training-ablation"
ENV = parent.ENV
FAMILY = parent.FAMILY
MODES = parent.MODES
REFERENCE_MODE = 0
TRAINING_SEEDS = (89_003, 89_021, 89_039, 89_057, 89_079)

STATIONARY_HOLDOUT_EVENT_SEEDS = (198_101, 198_117, 198_133)
SWITCHING_EVENT_SEEDS = (198_201, 198_217, 198_233)
SWITCHING_SCHEDULES = {
    198_201: (2, 1, 3, 0),
    198_217: (0, 3, 1, 2),
    198_233: (3, 0, 2, 1),
}

SOURCE_NEXT_ITERATION = parent.SOURCE_NEXT_ITERATION
SOURCE_ITERATION = parent.SOURCE_ITERATION
SOURCE_TOTAL_STEPS = parent.SOURCE_TOTAL_STEPS
SOURCE_UPDATE_COUNT = parent.SOURCE_UPDATE_COUNT
SOURCE_START_TRAIN_STEPS = parent.SOURCE_START_TRAIN_STEPS
FINAL_NEXT_ITERATION = parent.FINAL_NEXT_ITERATION
FINAL_ITERATION = parent.FINAL_ITERATION
FINAL_TOTAL_STEPS = parent.FINAL_TOTAL_STEPS
FINAL_UPDATE_COUNT = parent.FINAL_UPDATE_COUNT
FINETUNE_ITERS = parent.FINETUNE_ITERS
MAX_ITERS = FINAL_NEXT_ITERATION
START_TRAIN_STEPS = SOURCE_START_TRAIN_STEPS
SAMPLES_PER_ITER = parent.SAMPLES_PER_ITER
UPDATES_PER_ITER = parent.UPDATES_PER_ITER
DWELL_STEPS = parent.DWELL_STEPS
MAX_EPISODE_STEPS = parent.MAX_EPISODE_STEPS
STATIONARY_EPISODES_PER_MODE = parent.STATIONARY_EPISODES_PER_MODE
EPISODES_PER_TASK = STATIONARY_EPISODES_PER_MODE
SWITCHING_EPISODES = parent.SWITCHING_EPISODES

WARMSTART_RECIPE = "warmstart_mode0"
ROBUST_RECIPE = "robust_multimode"
DIRECT_RECIPE = "direct_mode0"
CANONICAL_RECIPE = "canonicalized_multimode"
RECIPES = (
    WARMSTART_RECIPE,
    ROBUST_RECIPE,
    DIRECT_RECIPE,
    CANONICAL_RECIPE,
)
CANONICAL_RECIPES = (
    WARMSTART_RECIPE,
    DIRECT_RECIPE,
    CANONICAL_RECIPE,
)
DIRECT_RECIPES = (ROBUST_RECIPE, DIRECT_RECIPE, CANONICAL_RECIPE)
VARIANTS = (WARMSTART_RECIPE,)
CONTROL_VARIANT = WARMSTART_RECIPE
SPECIALIST_VARIANT = WARMSTART_RECIPE

ROBUST_ARM = ROBUST_RECIPE
ARMS = (ROBUST_ARM,) + tuple(
    f"{recipe}_{route}"
    for recipe in CANONICAL_RECIPES
    for route in ("no_compensation", "oracle", "causal")
)

# This is a development-selection rule, not a confirmatory efficacy gate.
REQUIRED_RECIPE_WINS = 4
MIN_STATIONARY_RETENTION = 0.95
REQUIRED_RETENTION_SEEDS = 4
EXACT_ACTION_ATOL = parent.EXACT_ACTION_ATOL
EXACT_RETURN_ATOL = parent.EXACT_RETURN_ATOL

RUN_ROOT = ROOT / "jax_experiments/results_regime_polarity_canonical_training_v33"
SOURCE_RUN_ROOT = RUN_ROOT / "robust_sources"
WARMSTART_RUN_ROOT = RUN_ROOT / "warmstart_mode0"
DIRECT_RUN_ROOT = RUN_ROOT / "direct"
SOURCE_BUNDLE_ROOT = (
    ROOT / "jax_experiments/eval_bundles_regime_polarity_canonical_training_source_v33"
)
WARMSTART_BUNDLE_ROOT = (
    ROOT / "jax_experiments/eval_bundles_regime_polarity_canonical_training_warmstart_v33"
)
DIRECT_BUNDLE_ROOT = (
    ROOT / "jax_experiments/eval_bundles_regime_polarity_canonical_training_direct_v33"
)
AUDIT_ROOT = ROOT / "jax_experiments/results_regime_polarity_canonical_training_audit_v33"
ANALYSIS_ROOT = ROOT / "jax_experiments/results_regime_polarity_canonical_training_analysis_v33"
REGISTRATION_ROOT = (
    ROOT / "jax_experiments/deployments/regime_polarity_canonical_training_v33"
)
REGISTRATION_PATH = REGISTRATION_ROOT / "registration.json"
PREREG_REPORT = (
    ROOT / "reports/regime_polarity_canonical_training_v33_preregistration_2026-09-22.md"
)
REPORT = ROOT / "reports/regime_polarity_canonical_training_v33_2026-09-22.md"

POLICY_NAME = parent.POLICY_NAME
CONTROLLER_STATE_NAME = parent.CONTROLLER_STATE_NAME
BOOTSTRAP_NAME = parent.BOOTSTRAP_NAME
SELECTION_NAME = parent.SELECTION_NAME
SOURCE_BUNDLE_SCHEMA = "bapr.canonical-training-source-bundle.v33"
BUNDLE_SCHEMA = "bapr.canonical-training-warmstart-bundle.v33"
BOOTSTRAP_SCHEMA = "bapr.canonical-training-warmstart-bootstrap.v33"
DIRECT_BUNDLE_SCHEMA = "bapr.canonical-training-direct-bundle.v33"
AUDIT_SCHEMA = "bapr.canonical-training-audit.v33"
ANALYSIS_SCHEMA = "bapr.canonical-training-analysis.v33"
REGISTRATION_SCHEMA = "bapr.canonical-training-registration.v33"

file_record = parent.file_record
read_json = parent.read_json
write_json_atomic = parent.write_json_atomic
write_text_atomic = parent.write_text_atomic
checkpoint_record = parent.checkpoint_record
mode_gain_vectors = mechanism.mode_gain_vectors
compensation_multiplier = mechanism.compensation_multiplier
compensate_action = mechanism.compensate_action


def require_training_seed(seed: int) -> int:
    value = int(seed)
    if value not in TRAINING_SEEDS:
        raise ValueError(f"unknown V33 training seed {value}")
    return value


require_seed = require_training_seed


def require_mode(mode: int) -> int:
    value = int(mode)
    if value not in MODES:
        raise ValueError(f"unknown actuator-polarity mode {value}")
    return value


def require_recipe(recipe: str) -> str:
    value = str(recipe)
    if value not in RECIPES:
        raise ValueError(f"unknown V33 recipe {value!r}")
    return value


def require_direct_recipe(recipe: str) -> str:
    value = require_recipe(recipe)
    if value not in DIRECT_RECIPES:
        raise ValueError(f"recipe {value!r} is not trained from scratch")
    return value


def require_variant(variant: str) -> str:
    value = str(variant)
    if value != WARMSTART_RECIPE:
        raise ValueError(f"unknown V33 warm-start variant {value!r}")
    return value


def actor_update_period(variant: str) -> int:
    require_variant(variant)
    return 1


def select_best_validation(variant: str) -> bool:
    require_variant(variant)
    return False


def require_switching_event_seed(seed: int) -> int:
    value = int(seed)
    if value not in SWITCHING_EVENT_SEEDS:
        raise ValueError(f"unknown V33 switching event seed {value}")
    return value


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
        directory / "logs/protocol_signature.json",
    )


def source_identity(seed: int) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "canonical_training_robust_source",
        "env": ENV,
        "family": FAMILY,
        "training_seed": require_training_seed(seed),
        "algo": "sac",
        "final_next_iteration": SOURCE_NEXT_ITERATION,
        "final_total_steps": SOURCE_TOTAL_STEPS,
        "final_update_count": SOURCE_UPDATE_COUNT,
    }


def expected_source_checkpoint() -> dict[str, Any]:
    return {
        "iteration": SOURCE_ITERATION,
        "next_iteration": SOURCE_NEXT_ITERATION,
        "total_steps": SOURCE_TOTAL_STEPS,
        "update_count": SOURCE_UPDATE_COUNT,
        "algo": "sac",
    }


def run_dir(variant: str, seed: int, mode: int) -> Path:
    require_variant(variant)
    if require_mode(mode) != REFERENCE_MODE:
        raise ValueError("V33 warm-start is fixed to reference mode 0")
    return WARMSTART_RUN_ROOT / f"seed_{require_training_seed(seed)}"


def bundle_dir(variant: str, seed: int, mode: int) -> Path:
    require_variant(variant)
    if require_mode(mode) != REFERENCE_MODE:
        raise ValueError("V33 warm-start is fixed to reference mode 0")
    return WARMSTART_BUNDLE_ROOT / f"seed_{require_training_seed(seed)}"


def bundle_manifest(variant: str, seed: int, mode: int) -> Path:
    return bundle_dir(variant, seed, mode) / "bundle_manifest.json"


def bundle_required_paths(variant: str, seed: int, mode: int) -> tuple[Path, ...]:
    directory = bundle_dir(variant, seed, mode)
    return (
        directory / "bundle_manifest.json",
        directory / "policy" / POLICY_NAME,
        directory / "logs/protocol_signature.json",
        directory / "provenance" / BOOTSTRAP_NAME,
        directory / "provenance" / SELECTION_NAME,
    )


def identity(variant: str, seed: int, mode: int) -> dict[str, Any]:
    require_variant(variant)
    if require_mode(mode) != REFERENCE_MODE:
        raise ValueError("V33 warm-start identity is fixed to mode 0")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "canonical_training_warmstart_reference",
        "recipe": WARMSTART_RECIPE,
        "env": ENV,
        "family": FAMILY,
        "training_seed": require_training_seed(seed),
        "fixed_mode": REFERENCE_MODE,
        "algo": "sac",
        "source_next_iteration": SOURCE_NEXT_ITERATION,
        "final_next_iteration": FINAL_NEXT_ITERATION,
        "final_total_steps": FINAL_TOTAL_STEPS,
        "final_update_count": FINAL_UPDATE_COUNT,
        "actor_update_period": 1,
        "select_best_validation": False,
    }


def expected_checkpoint() -> dict[str, Any]:
    return {
        "iteration": FINAL_ITERATION,
        "next_iteration": FINAL_NEXT_ITERATION,
        "total_steps": FINAL_TOTAL_STEPS,
        "update_count": FINAL_UPDATE_COUNT,
        "algo": "sac",
    }


def direct_run_dir(recipe: str, seed: int) -> Path:
    return DIRECT_RUN_ROOT / require_direct_recipe(recipe) / f"seed_{require_training_seed(seed)}"


def direct_bundle_dir(recipe: str, seed: int) -> Path:
    return DIRECT_BUNDLE_ROOT / require_direct_recipe(recipe) / f"seed_{require_training_seed(seed)}"


def direct_manifest(recipe: str, seed: int) -> Path:
    return direct_bundle_dir(recipe, seed) / "bundle_manifest.json"


def direct_required_paths(recipe: str, seed: int) -> tuple[Path, ...]:
    directory = direct_bundle_dir(recipe, seed)
    return (
        directory / "bundle_manifest.json",
        directory / "policy" / POLICY_NAME,
        directory / "logs/protocol_signature.json",
    )


def direct_flags(recipe: str) -> tuple[int, int]:
    recipe = require_direct_recipe(recipe)
    if recipe == ROBUST_RECIPE:
        return -1, -1
    if recipe == DIRECT_RECIPE:
        return REFERENCE_MODE, -1
    return -1, REFERENCE_MODE


def direct_identity(recipe: str, seed: int) -> dict[str, Any]:
    fixed_mode, canonical_reference = direct_flags(recipe)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "canonical_training_from_scratch",
        "recipe": require_direct_recipe(recipe),
        "env": ENV,
        "family": FAMILY,
        "training_seed": require_training_seed(seed),
        "fixed_mode": fixed_mode,
        "canonical_reference_mode": canonical_reference,
        "algo": "sac",
        "final_next_iteration": FINAL_NEXT_ITERATION,
        "final_total_steps": FINAL_TOTAL_STEPS,
        "final_update_count": FINAL_UPDATE_COUNT,
    }


def recipe_bundle_dir(recipe: str, seed: int) -> Path:
    recipe = require_recipe(recipe)
    if recipe == WARMSTART_RECIPE:
        return bundle_dir(recipe, seed, REFERENCE_MODE)
    return direct_bundle_dir(recipe, seed)


def recipe_manifest(recipe: str, seed: int) -> Path:
    return recipe_bundle_dir(recipe, seed) / "bundle_manifest.json"


def recipe_required_paths(recipe: str, seed: int) -> tuple[Path, ...]:
    recipe = require_recipe(recipe)
    if recipe == WARMSTART_RECIPE:
        return bundle_required_paths(recipe, seed, REFERENCE_MODE)
    return direct_required_paths(recipe, seed)


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


def frozen_input_records(seed: int) -> dict[str, Any]:
    seed = require_training_seed(seed)
    return {
        "recipes": {
            recipe: file_record(recipe_manifest(recipe, seed))
            for recipe in RECIPES
        },
        "v5_estimator_manifest": file_record(v5_model.MODEL_MANIFEST),
        "v5_estimator_parameters": file_record(v5_model.MODEL_PATH),
    }


def audit_required_paths(seed: int) -> tuple[Path, ...]:
    paths = [REGISTRATION_PATH, v5_model.MODEL_MANIFEST, v5_model.MODEL_PATH]
    for recipe in RECIPES:
        paths.extend(recipe_required_paths(recipe, seed))
    return tuple(dict.fromkeys(path.resolve() for path in paths))


def registration_source_paths() -> tuple[Path, ...]:
    names = (
        "regime_polarity_canonical_training_ablation_v33.py",
        "run_regime_polarity_canonical_training_source_v33.py",
        "run_regime_polarity_canonical_training_warmstart_v33.py",
        "run_regime_polarity_canonical_training_direct_v33.py",
        "run_regime_polarity_canonical_training_audit_v33.py",
        "analyze_regime_polarity_canonical_training_v33.py",
    )
    paths = [ROOT / "jax_experiments/analysis" / name for name in names]
    paths += [
        ROOT / "scripts/submit_regime_polarity_canonical_training_v33.py",
        ROOT / "jax_experiments/envs/stochastic_mode_env.py",
        ROOT / "jax_experiments/configs/default.py",
        ROOT / "jax_experiments/train.py",
        v5_model.MODEL_MANIFEST,
        v5_model.MODEL_PATH,
        PREREG_REPORT,
    ]
    return tuple(path.resolve() for path in paths)


def registration_payload() -> dict[str, Any]:
    paths = registration_source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V33 registration inputs missing: {missing}")
    return {
        "schema": REGISTRATION_SCHEMA,
        "status": "registered",
        "created_before_training": True,
        "identity": {
            "protocol_version": PROTOCOL_VERSION,
            "environment": ENV,
            "family": FAMILY,
            "training_seeds": list(TRAINING_SEEDS),
            "recipes": list(RECIPES),
            "reference_mode": REFERENCE_MODE,
            "stationary_event_seeds": list(STATIONARY_HOLDOUT_EVENT_SEEDS),
            "switching_event_seeds": list(SWITCHING_EVENT_SEEDS),
            "switching_schedules": {
                str(seed): list(sequence)
                for seed, sequence in SWITCHING_SCHEDULES.items()
            },
        },
        "selection_rule": {
            "role": "development_recipe_selection_not_confirmation",
            "candidate_must_improve_causal_mean_over_warmstart": True,
            "required_paired_seed_wins": REQUIRED_RECIPE_WINS,
            "minimum_stationary_causal_to_oracle_retention": (
                MIN_STATIONARY_RETENTION
            ),
            "required_retention_seeds": REQUIRED_RETENTION_SEEDS,
            "oracle_trajectory_equivalence_required": True,
            "select_by": "switching_causal_return_after_all_gates",
        },
        "accounting": {
            "policy_interactions_per_final_recipe_seed": FINAL_TOTAL_STEPS,
            "warmstart_source_interactions": SOURCE_TOTAL_STEPS,
            "warmstart_fixed_mode_finetune_interactions": (
                FINAL_TOTAL_STEPS - SOURCE_TOTAL_STEPS
            ),
            "v5_estimator_retrained": False,
            "comparators_retrained": False,
        },
        "evaluation_boundary": {
            "canonical_training_interface_disabled_during_audit": True,
            "same_frozen_v5_estimator": True,
            "new_event_streams": True,
        },
        "source_files": {
            path.resolve().relative_to(ROOT).as_posix(): file_record(path)
            for path in paths
        },
    }


def create_registration() -> dict[str, Any]:
    expected = registration_payload()
    if REGISTRATION_PATH.is_file():
        actual = read_json(REGISTRATION_PATH)
        if actual != expected:
            raise ValueError("existing V33 registration differs from source")
        return actual
    REGISTRATION_ROOT.mkdir(parents=True, exist_ok=True)
    write_json_atomic(REGISTRATION_PATH, expected)
    return expected


def validate_registration() -> dict[str, Any]:
    actual = read_json(REGISTRATION_PATH)
    expected = registration_payload()
    if actual != expected:
        raise ValueError("V33 registration is missing or stale")
    return actual


def assert_protocol_integrity() -> None:
    if FAMILY != "actuator_polarity":
        raise ValueError("canonical action training is valid only for polarity modes")
    if set(TRAINING_SEEDS) & set(parent.TRAINING_SEEDS):
        raise ValueError("V33 development seeds overlap V32 confirmation")
    if len(set(STATIONARY_HOLDOUT_EVENT_SEEDS + SWITCHING_EVENT_SEEDS)) != 6:
        raise ValueError("V33 event streams must be distinct")
    if FINAL_TOTAL_STEPS != 8_400_000 or FINAL_UPDATE_COUNT != 525_000:
        raise ValueError("V33 equal-budget endpoint changed")


assert_protocol_integrity()
