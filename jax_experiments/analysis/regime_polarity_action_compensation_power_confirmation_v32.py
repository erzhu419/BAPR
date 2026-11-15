"""Prospective ten-seed power confirmation of V31 action compensation."""
from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_action_compensation_confirmation_v31 as pilot,
)
from jax_experiments.analysis import (
    regime_polarity_full_state_final_confirmation_v21 as recipe,
)
from jax_experiments.analysis import (
    regime_polarity_action_compensation_v28 as mechanism,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_system_id_v5 as v5_model,
)


ROOT = pilot.ROOT
PROTOCOL_VERSION = "v32-canonical-action-compensation-power-confirmation"
ENV = pilot.ENV
FAMILY = pilot.FAMILY
MODES = pilot.MODES
REFERENCE_MODE = pilot.REFERENCE_MODE
TRAINING_SEEDS = (
    88_003, 88_021, 88_039, 88_057, 88_079,
    88_103, 88_121, 88_139, 88_157, 88_179,
)

STATIONARY_HOLDOUT_EVENT_SEEDS = (197_101, 197_117, 197_133)
SWITCHING_EVENT_SEEDS = (197_201, 197_217, 197_233)
SWITCHING_SCHEDULES = {
    197_201: (2, 0, 3, 1),
    197_217: (1, 3, 0, 2),
    197_233: (3, 2, 1, 0),
}

SOURCE_NEXT_ITERATION = pilot.SOURCE_NEXT_ITERATION
SOURCE_ITERATION = pilot.SOURCE_ITERATION
SOURCE_TOTAL_STEPS = pilot.SOURCE_TOTAL_STEPS
SOURCE_UPDATE_COUNT = pilot.SOURCE_UPDATE_COUNT
SOURCE_START_TRAIN_STEPS = pilot.SOURCE_START_TRAIN_STEPS
REFERENCE_VARIANT = pilot.REFERENCE_VARIANT
VARIANTS = pilot.VARIANTS
CONTROL_VARIANT = pilot.CONTROL_VARIANT
SPECIALIST_VARIANT = pilot.SPECIALIST_VARIANT
FINETUNE_ITERS = pilot.FINETUNE_ITERS
FINAL_NEXT_ITERATION = pilot.FINAL_NEXT_ITERATION
FINAL_ITERATION = pilot.FINAL_ITERATION
FINAL_TOTAL_STEPS = pilot.FINAL_TOTAL_STEPS
FINAL_UPDATE_COUNT = pilot.FINAL_UPDATE_COUNT
MAX_ITERS = pilot.MAX_ITERS
START_TRAIN_STEPS = pilot.START_TRAIN_STEPS
SAMPLES_PER_ITER = pilot.SAMPLES_PER_ITER
UPDATES_PER_ITER = pilot.UPDATES_PER_ITER
DWELL_STEPS = pilot.DWELL_STEPS
MAX_EPISODE_STEPS = pilot.MAX_EPISODE_STEPS
STATIONARY_EPISODES_PER_MODE = pilot.STATIONARY_EPISODES_PER_MODE
EPISODES_PER_TASK = pilot.EPISODES_PER_TASK
SWITCHING_EPISODES = pilot.SWITCHING_EPISODES

TRAINED_METHODS = pilot.TRAINED_METHODS
LONG_SAC_SLOT = pilot.LONG_SAC_SLOT
SAC_REPLICA_SLOTS = pilot.SAC_REPLICA_SLOTS
ESCP_CONFIG = dict(pilot.ESCP_CONFIG)
RESAC_CONFIG = dict(pilot.RESAC_CONFIG)

ROBUST_SOURCE_ARM = pilot.ROBUST_SOURCE_ARM
ROBUST_LONG_ARM = pilot.ROBUST_LONG_ARM
NO_COMPENSATION_ARM = pilot.NO_COMPENSATION_ARM
ORACLE_COMPENSATION_ARM = pilot.ORACLE_COMPENSATION_ARM
CAUSAL_COMPENSATION_ARM = pilot.CAUSAL_COMPENSATION_ARM
ESCP_ARM = pilot.ESCP_ARM
RESAC_ARM = pilot.RESAC_ARM
ARMS = pilot.ARMS

REQUIRED_SEED_WINS = 8
REQUIRED_EVENT_WINS = 24
MIN_ORACLE_HEADROOM = pilot.MIN_ORACLE_HEADROOM
MIN_ORACLE_RECOVERY = pilot.MIN_ORACLE_RECOVERY
REQUIRED_RECOVERY_SEEDS = 8
MIN_STATIONARY_ORACLE_RETENTION = pilot.MIN_STATIONARY_ORACLE_RETENTION
REQUIRED_STATIONARY_RETENTION_SEEDS = 8
EXACT_ACTION_ATOL = pilot.EXACT_ACTION_ATOL
EXACT_RETURN_ATOL = pilot.EXACT_RETURN_ATOL
MIN_CALIBRATION_GAIN = pilot.MIN_CALIBRATION_GAIN

# Exact two-sided one-sample noncentral-t power under the V31 paired pilot
# effects. These values select ten new seeds before any V32 training.
PLANNED_SAMPLE_SIZE = 10
PILOT_POWER_ESTIMATES = {
    "causal_vs_equal_budget_sac": 0.919,
    "causal_vs_equal_budget_resac": 0.929,
}

RUN_ROOT = ROOT / "jax_experiments/results_regime_polarity_action_compensation_power_v32"
SOURCE_RUN_ROOT = RUN_ROOT / "robust_sources"
REFERENCE_RUN_ROOT = RUN_ROOT / "canonical_references"
BASELINE_RUN_ROOT = RUN_ROOT / "baselines"
SOURCE_BUNDLE_ROOT = ROOT / "jax_experiments/eval_bundles_regime_polarity_action_compensation_source_v32"
REFERENCE_BUNDLE_ROOT = ROOT / "jax_experiments/eval_bundles_regime_polarity_action_compensation_reference_v32"
BASELINE_BUNDLE_ROOT = ROOT / "jax_experiments/eval_bundles_regime_polarity_action_compensation_baselines_v32"
AUDIT_ROOT = ROOT / "jax_experiments/results_regime_polarity_action_compensation_power_audit_v32"
ANALYSIS_ROOT = ROOT / "jax_experiments/results_regime_polarity_action_compensation_power_analysis_v32"
REGISTRATION_ROOT = ROOT / "jax_experiments/deployments/regime_polarity_action_compensation_power_v32"
REGISTRATION_PATH = REGISTRATION_ROOT / "registration.json"
PREREG_REPORT = ROOT / "reports/regime_polarity_action_compensation_power_v32_preregistration_2026-09-13.md"
REPORT = ROOT / "reports/regime_polarity_action_compensation_power_v32_2026-09-13.md"

POLICY_NAME = pilot.POLICY_NAME
CONTROLLER_STATE_NAME = pilot.CONTROLLER_STATE_NAME
BOOTSTRAP_NAME = pilot.BOOTSTRAP_NAME
SELECTION_NAME = pilot.SELECTION_NAME
EVAL_PARAMS_NAME = pilot.EVAL_PARAMS_NAME
SOURCE_BUNDLE_SCHEMA = "bapr.canonical-compensation-source-bundle.v32"
REFERENCE_BUNDLE_SCHEMA = "bapr.canonical-compensation-reference-bundle.v32"
REFERENCE_BOOTSTRAP_SCHEMA = "bapr.canonical-compensation-bootstrap.v32"
BASELINE_BUNDLE_SCHEMA = "bapr.canonical-compensation-baseline-bundle.v32"
BUNDLE_SCHEMA = BASELINE_BUNDLE_SCHEMA
EVAL_PARAMS_SCHEMA = "bapr.evaluation-parameters.v32"
AUDIT_SCHEMA = "bapr.canonical-compensation-power-audit.v32"
ANALYSIS_SCHEMA = "bapr.canonical-compensation-power-analysis.v32"
REGISTRATION_SCHEMA = "bapr.canonical-compensation-power-registration.v32"

file_record = pilot.file_record
read_json = pilot.read_json
write_json_atomic = pilot.write_json_atomic
write_text_atomic = pilot.write_text_atomic
checkpoint_record = pilot.checkpoint_record
mode_gain_vectors = mechanism.mode_gain_vectors
compensation_multiplier = mechanism.compensation_multiplier
compensate_action = mechanism.compensate_action


def require_training_seed(seed: int) -> int:
    value = int(seed)
    if value not in TRAINING_SEEDS:
        raise ValueError(f"unknown V32 policy seed {value}")
    return value


require_seed = require_training_seed


def require_mode(mode: int) -> int:
    value = int(mode)
    if value not in MODES:
        raise ValueError(f"unknown actuator-polarity mode {value}")
    return value


def require_variant(variant: str) -> str:
    value = str(variant)
    if value != REFERENCE_VARIANT:
        raise ValueError(f"unknown V32 reference variant {value!r}")
    return value


def actor_update_period(variant: str) -> int:
    require_variant(variant)
    return 1


def select_best_validation(variant: str) -> bool:
    require_variant(variant)
    return False


def require_method(method: str) -> str:
    value = str(method)
    if value not in TRAINED_METHODS:
        raise ValueError(f"unknown V32 baseline method {value!r}")
    return value


def require_replica_slot(slot: int) -> int:
    value = int(slot)
    if value != LONG_SAC_SLOT:
        raise ValueError(f"unknown V32 long-SAC slot {value}")
    return value


def replica_training_seed(seed: int, slot: int) -> int:
    require_replica_slot(slot)
    return require_training_seed(seed)


def require_switching_event_seed(seed: int) -> int:
    value = int(seed)
    if value not in SWITCHING_EVENT_SEEDS:
        raise ValueError(f"unknown V32 switching event seed {value}")
    return value


def switching_sequence(event_seed: int, episode: int) -> tuple[int, ...]:
    base = SWITCHING_SCHEDULES[require_switching_event_seed(event_seed)]
    shift = int(episode) % len(base)
    return tuple(base[shift:] + base[:shift])


def algo_for(method: str) -> str:
    return "escp" if require_method(method) == "escp_recurrent" else "resac"


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
        "benchmark_role": "power_confirmation_robust_source",
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


def reference_run_dir(seed: int) -> Path:
    return REFERENCE_RUN_ROOT / f"seed_{require_training_seed(seed)}/mode_{REFERENCE_MODE}"


def reference_bundle_dir(seed: int) -> Path:
    return REFERENCE_BUNDLE_ROOT / f"seed_{require_training_seed(seed)}/mode_{REFERENCE_MODE}"


def reference_bundle_manifest(seed: int) -> Path:
    return reference_bundle_dir(seed) / "bundle_manifest.json"


def reference_required_paths(seed: int) -> tuple[Path, ...]:
    directory = reference_bundle_dir(seed)
    return (
        directory / "bundle_manifest.json",
        directory / "policy" / POLICY_NAME,
        directory / "logs/protocol_signature.json",
        directory / "provenance" / BOOTSTRAP_NAME,
        directory / "provenance" / SELECTION_NAME,
    )


def reference_identity(seed: int) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "power_confirmation_fixed_canonical_reference",
        "variant": REFERENCE_VARIANT,
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


def expected_reference_checkpoint() -> dict[str, Any]:
    return {
        "iteration": FINAL_ITERATION,
        "next_iteration": FINAL_NEXT_ITERATION,
        "total_steps": FINAL_TOTAL_STEPS,
        "update_count": FINAL_UPDATE_COUNT,
        "algo": "sac",
    }


def baseline_run_dir(method: str, seed: int) -> Path:
    return BASELINE_RUN_ROOT / method / f"seed_{require_training_seed(seed)}"


def baseline_bundle_dir(method: str, seed: int) -> Path:
    return BASELINE_BUNDLE_ROOT / method / f"seed_{require_training_seed(seed)}"


def sac_run_dir(seed: int, slot: int) -> Path:
    require_replica_slot(slot)
    return BASELINE_RUN_ROOT / "robust_long" / f"seed_{require_training_seed(seed)}"


def sac_bundle_dir(seed: int, slot: int) -> Path:
    require_replica_slot(slot)
    return BASELINE_BUNDLE_ROOT / "robust_long" / f"seed_{require_training_seed(seed)}"


def bundle_dir(kind: str, seed: int, slot: int | None = None) -> Path:
    if kind == "sac_replica":
        if slot is None:
            raise ValueError("V32 long SAC requires its frozen slot")
        return sac_bundle_dir(seed, slot)
    if slot is not None:
        raise ValueError("slot is only valid for long SAC")
    return baseline_bundle_dir(require_method(kind), seed)


def bundle_manifest(kind: str, seed: int, slot: int | None = None) -> Path:
    return bundle_dir(kind, seed, slot) / "bundle_manifest.json"


def bundle_required_paths(kind: str, seed: int, slot: int | None = None) -> tuple[Path, ...]:
    directory = bundle_dir(kind, seed, slot)
    return (
        directory / "bundle_manifest.json",
        directory / "runtime" / EVAL_PARAMS_NAME,
        directory / "logs/protocol_signature.json",
    )


def identity(kind: str, seed: int, slot: int | None = None) -> dict[str, Any]:
    seed = require_training_seed(seed)
    if kind == "sac_replica":
        if slot is None:
            raise ValueError("V32 long SAC identity requires a slot")
        slot = require_replica_slot(slot)
        algo = "sac"
        training_seed = replica_training_seed(seed, slot)
    else:
        kind = require_method(kind)
        slot = None
        algo = algo_for(kind)
        training_seed = seed
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "power_confirmation_equal_interaction_baseline",
        "kind": kind,
        "outer_seed": seed,
        "replica_slot": slot,
        "training_seed": training_seed,
        "algo": algo,
        "env": ENV,
        "family": FAMILY,
        "max_iters": MAX_ITERS,
        "total_steps": FINAL_TOTAL_STEPS,
        "update_count": FINAL_UPDATE_COUNT,
    }


def expected_checkpoint(kind: str) -> dict[str, Any]:
    algo = "sac" if kind == "sac_replica" else algo_for(kind)
    return {
        "iteration": FINAL_ITERATION,
        "next_iteration": MAX_ITERS,
        "total_steps": FINAL_TOTAL_STEPS,
        "update_count": FINAL_UPDATE_COUNT,
        "algo": algo,
    }


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
        "robust_source": file_record(source_manifest(seed)),
        "canonical_reference": file_record(reference_bundle_manifest(seed)),
        "robust_long": file_record(bundle_manifest("sac_replica", seed, LONG_SAC_SLOT)),
        "escp_recurrent": file_record(bundle_manifest("escp_recurrent", seed)),
        "resac_b0": file_record(bundle_manifest("resac_b0", seed)),
        "v5_estimator": {
            "manifest": file_record(v5_model.MODEL_MANIFEST),
            "parameters": file_record(v5_model.MODEL_PATH),
        },
    }


def audit_required_paths(seed: int) -> tuple[Path, ...]:
    paths = [
        REGISTRATION_PATH,
        *source_required_paths(seed),
        *reference_required_paths(seed),
        *bundle_required_paths("sac_replica", seed, LONG_SAC_SLOT),
        *bundle_required_paths("escp_recurrent", seed),
        *bundle_required_paths("resac_b0", seed),
        v5_model.MODEL_MANIFEST,
        v5_model.MODEL_PATH,
    ]
    return tuple(dict.fromkeys(path.resolve() for path in paths))


def _pilot_power_basis() -> dict[str, Any]:
    prior = read_json(pilot.analysis_json())
    if prior.get("fresh_policy_confirmation_pass") is not False:
        raise ValueError("V31 must remain a failed pilot for V32 planning")
    rows = {}
    for key in (
        "causal_vs_equal_budget_sac",
        "causal_vs_equal_budget_resac",
    ):
        comparison = prior["comparisons"][key]
        differences = [
            float(value)
            for value in comparison["paired_seed_differences"].values()
        ]
        sample_sd = statistics.stdev(differences)
        rows[key] = {
            "v31_paired_mean": statistics.mean(differences),
            "v31_sample_sd": sample_sd,
            "standardized_effect": statistics.mean(differences) / sample_sd,
            "planned_n": PLANNED_SAMPLE_SIZE,
            "estimated_two_sided_power": PILOT_POWER_ESTIMATES[key],
        }
    return {
        "role": "sample_size_planning_only",
        "v31_confirmation_pass": False,
        "v31_diagnosis": prior["diagnosis"],
        "v31_not_pooled_with_v32": True,
        "comparisons": rows,
    }


def registration_source_paths() -> tuple[Path, ...]:
    paths = [
        ROOT / "jax_experiments/analysis/regime_polarity_action_compensation_power_confirmation_v32.py",
        ROOT / "jax_experiments/analysis/regime_polarity_action_compensation_reference_v32.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_action_compensation_source_v32.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_action_compensation_reference_v32.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_action_compensation_baseline_v32.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_action_compensation_power_audit_v32.py",
        ROOT / "jax_experiments/analysis/analyze_regime_polarity_action_compensation_power_v32.py",
        ROOT / "scripts/submit_regime_polarity_action_compensation_power_v32.py",
        ROOT / "jax_experiments/analysis/analyze_regime_polarity_action_compensation_confirmation_v31.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_action_compensation_confirmation_audit_v31.py",
        ROOT / "scripts/submit_regime_polarity_action_compensation_confirmation_v31.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_robust_source_v19.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_specialist_policy_stability_v20.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_v5_final_baseline_v18.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_v5_final_comparison_audit_v18.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_specialist_safe_utility_audit_v8.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_action_compensation_audit_v28.py",
        ROOT / "jax_experiments/analysis/regime_polarity_specialist_expected_action_model_v5.py",
        ROOT / "jax_experiments/common/checkpoint.py",
        ROOT / "jax_experiments/algos/escp.py",
        ROOT / "jax_experiments/algos/resac.py",
        ROOT / "jax_experiments/algos/sac_base.py",
        ROOT / "jax_experiments/envs/stochastic_mode_env.py",
        ROOT / "jax_experiments/envs/brax_env.py",
        ROOT / "jax_experiments/train.py",
        pilot.REGISTRATION_PATH,
        pilot.analysis_json(),
        pilot.analysis_markdown(),
        v5_model.MODEL_MANIFEST,
        v5_model.MODEL_PATH,
        PREREG_REPORT,
    ]
    return tuple(dict.fromkeys(path.resolve() for path in paths))


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def registration_payload() -> dict[str, Any]:
    pilot.validate_registration()
    assert_protocol_integrity()
    paths = registration_source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V32 registration sources missing: {missing}")
    return {
        "schema": REGISTRATION_SCHEMA,
        "status": "registered",
        "created_before_training": True,
        "identity": {
            "protocol_version": PROTOCOL_VERSION,
            "environment": ENV,
            "family": FAMILY,
            "training_seeds": list(TRAINING_SEEDS),
            "reference_mode": REFERENCE_MODE,
            "modes": list(MODES),
            "stationary_holdout_event_seeds": list(STATIONARY_HOLDOUT_EVENT_SEEDS),
            "switching_event_seeds": list(SWITCHING_EVENT_SEEDS),
            "switching_schedules": {
                str(seed): list(sequence)
                for seed, sequence in SWITCHING_SCHEDULES.items()
            },
            "arms": list(ARMS),
        },
        "development_basis": _pilot_power_basis(),
        "frozen_boundary": {
            "v31_algorithm_unchanged": True,
            "v31_failed_result_preserved": True,
            "v31_results_not_pooled": True,
            "ten_new_policy_seeds": True,
            "new_stationary_and_switching_events": True,
            "reference_mode_fixed_before_training": True,
            "no_reference_calibration": True,
            "v5_estimator_unchanged": True,
            "causal_action_uses_posterior_before_current_transition": True,
            "no_checkpoint_selection": True,
        },
        "budgets": {
            "canonical_compensation_training_path": {
                "robust_source_steps": SOURCE_TOTAL_STEPS,
                "reference_finetune_steps": FINETUNE_ITERS * SAMPLES_PER_ITER,
                "total_unique_interactions": FINAL_TOTAL_STEPS,
                "final_updates": FINAL_UPDATE_COUNT,
            },
            "equal_interaction_baselines": {
                "methods": ["robust_sac", *TRAINED_METHODS],
                "steps_per_method": FINAL_TOTAL_STEPS,
                "updates_per_method": FINAL_UPDATE_COUNT,
            },
        },
        "decision_gate": {
            "primary_arm": CAUSAL_COMPENSATION_ARM,
            "primary_comparators": [ROBUST_LONG_ARM, ESCP_ARM, RESAC_ARM],
            "required_seed_wins_of_10": REQUIRED_SEED_WINS,
            "required_event_wins_of_30": REQUIRED_EVENT_WINS,
            "positive_paired_mean_and_two_sided_ci95": True,
            "minimum_oracle_headroom": MIN_ORACLE_HEADROOM,
            "minimum_oracle_recovery": MIN_ORACLE_RECOVERY,
            "required_recovery_seeds_of_10": REQUIRED_RECOVERY_SEEDS,
            "minimum_stationary_oracle_retention": MIN_STATIONARY_ORACLE_RETENTION,
            "required_stationary_retention_seeds_of_10": REQUIRED_STATIONARY_RETENTION_SEEDS,
        },
        "claim_boundary": {
            "confirms_halfcheetah_actuator_polarity_only": True,
            "does_not_claim_general_invertible_dynamics": True,
            "does_not_claim_termination_safety": True,
            "training_cost_includes_robust_source": True,
        },
        "sync_policy": (
            "compact policy/controller parameters, protocol signatures, "
            "provenance, and JSON audits only; no replay or checkpoints"),
        "source_records": {_relative(path): file_record(path) for path in paths},
    }


def create_registration() -> dict[str, Any]:
    payload = registration_payload()
    REGISTRATION_ROOT.mkdir(parents=True, exist_ok=True)
    if REGISTRATION_PATH.is_file():
        if read_json(REGISTRATION_PATH) != payload:
            raise ValueError("existing V32 registration changed")
    else:
        write_json_atomic(REGISTRATION_PATH, payload)
    return validate_registration()


def validate_registration() -> dict[str, Any]:
    if not REGISTRATION_PATH.is_file():
        raise FileNotFoundError(f"missing V32 registration: {REGISTRATION_PATH}")
    payload = read_json(REGISTRATION_PATH)
    if payload != registration_payload():
        raise ValueError("V32 registration or frozen source closure changed")
    return payload


def assert_protocol_integrity() -> None:
    if len(TRAINING_SEEDS) != PLANNED_SAMPLE_SIZE or len(set(TRAINING_SEEDS)) != 10:
        raise ValueError("V32 requires ten unique policy seeds")
    old_seeds = set(pilot.TRAINING_SEEDS) | set(recipe.TRAINING_SEEDS) | set(mechanism.TRAINING_SEEDS)
    if set(TRAINING_SEEDS) & old_seeds:
        raise ValueError("V32 reused a prior confirmation policy seed")
    if set(TRAINING_SEEDS) & set(v5_model.TRAIN_SOURCE_SEEDS):
        raise ValueError("V32 reused an estimator-training policy seed")
    old_events = set(pilot.STATIONARY_HOLDOUT_EVENT_SEEDS) | set(pilot.SWITCHING_EVENT_SEEDS)
    if (set(STATIONARY_HOLDOUT_EVENT_SEEDS) | set(SWITCHING_EVENT_SEEDS)) & old_events:
        raise ValueError("V32 reused a V31 event stream")
    if set(STATIONARY_HOLDOUT_EVENT_SEEDS) & set(SWITCHING_EVENT_SEEDS):
        raise ValueError("V32 stationary and switching streams overlap")
    if set(SWITCHING_SCHEDULES) != set(SWITCHING_EVENT_SEEDS):
        raise ValueError("V32 switching schedules are incomplete")
    if any(len(sequence) != len(MODES) or set(sequence) != set(MODES)
           for sequence in SWITCHING_SCHEDULES.values()):
        raise ValueError("each V32 schedule must use all modes once")
    if FINAL_TOTAL_STEPS != 8_400_000 or SOURCE_TOTAL_STEPS != 5_600_000:
        raise ValueError("V32 interaction budgets changed")
    if REQUIRED_SEED_WINS != 8 or REQUIRED_EVENT_WINS != 24:
        raise ValueError("V32 population consistency gate changed")
    if min(PILOT_POWER_ESTIMATES.values()) < 0.90:
        raise ValueError("V32 planned power is below 90 percent")


assert_protocol_integrity()
