"""Fresh-policy confirmation of canonical action compensation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_action_compensation_v28 as mechanism,
)
from jax_experiments.analysis import (
    regime_polarity_full_state_final_confirmation_v21 as recipe,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_system_id_v5 as v5_model,
)


ROOT = recipe.ROOT
PROTOCOL_VERSION = "v31-canonical-action-compensation-confirmation"
ENV = recipe.ENV
FAMILY = recipe.FAMILY
MODES = recipe.MODES
REFERENCE_MODE = 0
TRAINING_SEEDS = (87_003, 87_021, 87_039, 87_057, 87_079)

STATIONARY_HOLDOUT_EVENT_SEEDS = (196_101, 196_117, 196_133)
SWITCHING_EVENT_SEEDS = (196_201, 196_217, 196_233)
SWITCHING_SCHEDULES = {
    196_201: (0, 2, 3, 1),
    196_217: (3, 1, 0, 2),
    196_233: (1, 3, 2, 0),
}

SOURCE_NEXT_ITERATION = recipe.SOURCE_NEXT_ITERATION
SOURCE_ITERATION = recipe.SOURCE_ITERATION
SOURCE_TOTAL_STEPS = recipe.SOURCE_TOTAL_STEPS
SOURCE_UPDATE_COUNT = recipe.SOURCE_UPDATE_COUNT
SOURCE_START_TRAIN_STEPS = recipe.SOURCE_START_TRAIN_STEPS

REFERENCE_VARIANT = "canonical_mode0_final"
VARIANTS = (REFERENCE_VARIANT,)
CONTROL_VARIANT = REFERENCE_VARIANT
SPECIALIST_VARIANT = REFERENCE_VARIANT
FINETUNE_ITERS = recipe.SPECIALIST_FINETUNE_ITERS
FINAL_NEXT_ITERATION = recipe.SPECIALIST_FINAL_NEXT_ITERATION
FINAL_ITERATION = recipe.SPECIALIST_FINAL_ITERATION
FINAL_TOTAL_STEPS = recipe.SPECIALIST_FINAL_TOTAL_STEPS
FINAL_UPDATE_COUNT = recipe.SPECIALIST_FINAL_UPDATE_COUNT

# Every final comparator receives the same 8.4M interaction budget as the
# robust-source plus canonical-reference training path.
MAX_ITERS = FINAL_NEXT_ITERATION
START_TRAIN_STEPS = SOURCE_START_TRAIN_STEPS
SAMPLES_PER_ITER = recipe.SAMPLES_PER_ITER
UPDATES_PER_ITER = recipe.UPDATES_PER_ITER
DWELL_STEPS = recipe.DWELL_STEPS
MAX_EPISODE_STEPS = recipe.MAX_EPISODE_STEPS
STATIONARY_EPISODES_PER_MODE = recipe.STATIONARY_EPISODES_PER_MODE
EPISODES_PER_TASK = STATIONARY_EPISODES_PER_MODE
SWITCHING_EPISODES = recipe.SWITCHING_EPISODES

TRAINED_METHODS = recipe.TRAINED_METHODS
LONG_SAC_SLOT = 0
SAC_REPLICA_SLOTS = (LONG_SAC_SLOT,)
ESCP_CONFIG = dict(recipe.ESCP_CONFIG)
RESAC_CONFIG = dict(recipe.RESAC_CONFIG)

ROBUST_SOURCE_ARM = "robust_sac_5p6m"
ROBUST_LONG_ARM = "robust_sac_8p4m"
NO_COMPENSATION_ARM = "canonical_no_compensation"
ORACLE_COMPENSATION_ARM = "canonical_true_mode_compensation"
CAUSAL_COMPENSATION_ARM = "canonical_v5_map_compensation"
ESCP_ARM = "escp_recurrent_8p4m"
RESAC_ARM = "resac_b0_8p4m"
ARMS = (
    ROBUST_SOURCE_ARM,
    ROBUST_LONG_ARM,
    NO_COMPENSATION_ARM,
    ORACLE_COMPENSATION_ARM,
    CAUSAL_COMPENSATION_ARM,
    ESCP_ARM,
    RESAC_ARM,
)

REQUIRED_SEED_WINS = 4
REQUIRED_EVENT_WINS = 12
MIN_ORACLE_HEADROOM = 0.10
MIN_ORACLE_RECOVERY = 0.70
REQUIRED_RECOVERY_SEEDS = 4
MIN_STATIONARY_ORACLE_RETENTION = 0.95
REQUIRED_STATIONARY_RETENTION_SEEDS = 4
EXACT_ACTION_ATOL = mechanism.EXACT_ACTION_ATOL
EXACT_RETURN_ATOL = mechanism.EXACT_RETURN_ATOL
MIN_CALIBRATION_GAIN = recipe.MIN_CALIBRATION_GAIN

RUN_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_action_compensation_confirmation_v31"
)
SOURCE_RUN_ROOT = RUN_ROOT / "robust_sources"
REFERENCE_RUN_ROOT = RUN_ROOT / "canonical_references"
BASELINE_RUN_ROOT = RUN_ROOT / "baselines"

SOURCE_BUNDLE_ROOT = (
    ROOT / "jax_experiments"
    / "eval_bundles_regime_polarity_action_compensation_source_v31"
)
REFERENCE_BUNDLE_ROOT = (
    ROOT / "jax_experiments"
    / "eval_bundles_regime_polarity_action_compensation_reference_v31"
)
BASELINE_BUNDLE_ROOT = (
    ROOT / "jax_experiments"
    / "eval_bundles_regime_polarity_action_compensation_baselines_v31"
)
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_action_compensation_confirmation_audit_v31"
)
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_action_compensation_confirmation_analysis_v31"
)
REGISTRATION_ROOT = (
    ROOT / "jax_experiments" / "deployments"
    / "regime_polarity_action_compensation_confirmation_v31"
)
REGISTRATION_PATH = REGISTRATION_ROOT / "registration.json"
PREREG_REPORT = (
    ROOT / "reports"
    / "regime_polarity_action_compensation_confirmation_v31_preregistration_2026-09-12.md"
)
REPORT = (
    ROOT / "reports"
    / "regime_polarity_action_compensation_confirmation_v31_2026-09-12.md"
)

POLICY_NAME = recipe.POLICY_NAME
CONTROLLER_STATE_NAME = recipe.CONTROLLER_STATE_NAME
BOOTSTRAP_NAME = recipe.BOOTSTRAP_NAME
SELECTION_NAME = recipe.SELECTION_NAME
EVAL_PARAMS_NAME = recipe.EVAL_PARAMS_NAME

SOURCE_BUNDLE_SCHEMA = "bapr.canonical-compensation-source-bundle.v31"
REFERENCE_BUNDLE_SCHEMA = "bapr.canonical-compensation-reference-bundle.v31"
REFERENCE_BOOTSTRAP_SCHEMA = "bapr.canonical-compensation-bootstrap.v31"
BASELINE_BUNDLE_SCHEMA = "bapr.canonical-compensation-baseline-bundle.v31"
BUNDLE_SCHEMA = BASELINE_BUNDLE_SCHEMA
EVAL_PARAMS_SCHEMA = "bapr.evaluation-parameters.v31"
AUDIT_SCHEMA = "bapr.canonical-compensation-confirmation-audit.v31"
ANALYSIS_SCHEMA = "bapr.canonical-compensation-confirmation-analysis.v31"
REGISTRATION_SCHEMA = "bapr.canonical-compensation-registration.v31"

file_record = recipe.file_record
read_json = recipe.read_json
write_json_atomic = recipe.write_json_atomic
write_text_atomic = recipe.write_text_atomic
checkpoint_record = recipe.checkpoint_record
mode_gain_vectors = mechanism.mode_gain_vectors
compensation_multiplier = mechanism.compensation_multiplier
compensate_action = mechanism.compensate_action


def require_training_seed(seed: int) -> int:
    value = int(seed)
    if value not in TRAINING_SEEDS:
        raise ValueError(f"unknown V31 policy seed {value}")
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
        raise ValueError(f"unknown V31 reference variant {value!r}")
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
        raise ValueError(f"unknown V31 baseline method {value!r}")
    return value


def require_replica_slot(slot: int) -> int:
    value = int(slot)
    if value != LONG_SAC_SLOT:
        raise ValueError(f"unknown V31 long-SAC slot {value}")
    return value


def replica_training_seed(seed: int, slot: int) -> int:
    require_replica_slot(slot)
    return require_training_seed(seed)


def require_switching_event_seed(seed: int) -> int:
    value = int(seed)
    if value not in SWITCHING_EVENT_SEEDS:
        raise ValueError(f"unknown V31 switching event seed {value}")
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
        directory / "logs" / "protocol_signature.json",
    )


def source_identity(seed: int) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "fresh_robust_source_for_canonical_reference",
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
    return (
        REFERENCE_RUN_ROOT / f"seed_{require_training_seed(seed)}"
        / f"mode_{REFERENCE_MODE}"
    )


def reference_bundle_dir(seed: int) -> Path:
    return (
        REFERENCE_BUNDLE_ROOT / f"seed_{require_training_seed(seed)}"
        / f"mode_{REFERENCE_MODE}"
    )


def reference_bundle_manifest(seed: int) -> Path:
    return reference_bundle_dir(seed) / "bundle_manifest.json"


def reference_required_paths(seed: int) -> tuple[Path, ...]:
    directory = reference_bundle_dir(seed)
    return (
        directory / "bundle_manifest.json",
        directory / "policy" / POLICY_NAME,
        directory / "logs" / "protocol_signature.json",
        directory / "provenance" / BOOTSTRAP_NAME,
        directory / "provenance" / SELECTION_NAME,
    )


def reference_identity(seed: int) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "fresh_fixed_canonical_reference",
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
            raise ValueError("V31 long SAC requires its frozen slot")
        return sac_bundle_dir(seed, slot)
    if slot is not None:
        raise ValueError("slot is only valid for long SAC")
    return baseline_bundle_dir(require_method(kind), seed)


def bundle_manifest(kind: str, seed: int, slot: int | None = None) -> Path:
    return bundle_dir(kind, seed, slot) / "bundle_manifest.json"


def bundle_required_paths(
    kind: str, seed: int, slot: int | None = None,
) -> tuple[Path, ...]:
    directory = bundle_dir(kind, seed, slot)
    return (
        directory / "bundle_manifest.json",
        directory / "runtime" / EVAL_PARAMS_NAME,
        directory / "logs" / "protocol_signature.json",
    )


def identity(
    kind: str, seed: int, slot: int | None = None,
) -> dict[str, Any]:
    seed = require_training_seed(seed)
    if kind == "sac_replica":
        if slot is None:
            raise ValueError("V31 long SAC identity requires a slot")
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
        "benchmark_role": "fresh_equal_interaction_baseline",
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
        "robust_long": file_record(
            bundle_manifest("sac_replica", seed, LONG_SAC_SLOT)),
        "escp_recurrent": file_record(bundle_manifest("escp_recurrent", seed)),
        "resac_b0": file_record(bundle_manifest("resac_b0", seed)),
        "v5_estimator": {
            "manifest": file_record(v5_model.MODEL_MANIFEST),
            "parameters": file_record(v5_model.MODEL_PATH),
        },
    }


def audit_required_paths(seed: int) -> tuple[Path, ...]:
    paths: list[Path] = [
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


def _old_mode0_basis() -> dict[str, Any]:
    rows = {}
    for seed in mechanism.TRAINING_SEEDS:
        payload = read_json(mechanism.audit_result(seed))
        mode0 = float(payload["calibration"]["reference_matrix"]["0"][
            "native_return_mean"])
        robust_rows = [
            event[mechanism.ROBUST_ARM][str(mode)]["return_mean"]
            for event in payload["stationary_holdout"].values()
            for mode in MODES
        ]
        robust = float(sum(robust_rows) / len(robust_rows))
        rows[str(seed)] = {
            "mode0_native_calibration_return": mode0,
            "robust_stationary_holdout_return": robust,
            "mode0_minus_robust": mode0 - robust,
        }
    if not all(row["mode0_minus_robust"] > 0.0 for row in rows.values()):
        raise ValueError("old V28 evidence does not support canonical mode 0")
    return {
        "role": "development_only_reference_choice",
        "selected_reference_mode": REFERENCE_MODE,
        "all_five_old_policy_seeds_positive": True,
        "rows": rows,
    }


def registration_source_paths() -> tuple[Path, ...]:
    paths = [
        ROOT / "jax_experiments/analysis/regime_polarity_action_compensation_confirmation_v31.py",
        ROOT / "jax_experiments/analysis/regime_polarity_action_compensation_reference_v31.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_action_compensation_source_v31.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_action_compensation_reference_v31.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_action_compensation_baseline_v31.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_action_compensation_confirmation_audit_v31.py",
        ROOT / "jax_experiments/analysis/analyze_regime_polarity_action_compensation_confirmation_v31.py",
        ROOT / "scripts/submit_regime_polarity_action_compensation_confirmation_v31.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_robust_source_v19.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_specialist_policy_stability_v20.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_v5_final_baseline_v18.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_action_compensation_audit_v28.py",
        ROOT / "jax_experiments/algos/escp.py",
        ROOT / "jax_experiments/algos/resac.py",
        ROOT / "jax_experiments/algos/sac_base.py",
        ROOT / "jax_experiments/envs/stochastic_mode_env.py",
        ROOT / "jax_experiments/envs/brax_env.py",
        ROOT / "jax_experiments/train.py",
        recipe.REGISTRATION_PATH,
        mechanism.REGISTRATION_PATH,
        mechanism.analysis_json(),
        mechanism.analysis_markdown(),
        mechanism.STRUCTURAL_MANIFEST,
        v5_model.MODEL_MANIFEST,
        v5_model.MODEL_PATH,
        PREREG_REPORT,
    ]
    for seed in mechanism.TRAINING_SEEDS:
        paths.extend((mechanism.audit_result(seed), mechanism.audit_manifest(seed)))
    return tuple(dict.fromkeys(path.resolve() for path in paths))


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def registration_payload() -> dict[str, Any]:
    recipe.validate_registration()
    mechanism.validate_registration()
    prior = read_json(mechanism.analysis_json())
    if prior.get("causal_compensation_claim_pass") is not True:
        raise ValueError("V28 did not authorize fresh-policy confirmation")
    paths = registration_source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V31 registration sources missing: {missing}")
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
            "stationary_holdout_event_seeds": list(
                STATIONARY_HOLDOUT_EVENT_SEEDS),
            "switching_event_seeds": list(SWITCHING_EVENT_SEEDS),
            "switching_schedules": {
                str(seed): list(sequence)
                for seed, sequence in SWITCHING_SCHEDULES.items()
            },
            "arms": list(ARMS),
        },
        "development_basis": {
            "v28_diagnosis": prior["diagnosis"],
            "v28_single_policy_superior_to_bank": (
                prior["bank_conclusion"]
                == "single_policy_compensation_is_superior"),
            "canonical_reference_choice": _old_mode0_basis(),
        },
        "frozen_boundary": {
            "reference_mode_selected_before_new_training": True,
            "no_new_reference_calibration": True,
            "v21_full_state_final_recipe_unchanged": True,
            "v5_estimator_unchanged": True,
            "v5_estimator_policy_seeds_disjoint": True,
            "causal_action_uses_posterior_before_current_transition": True,
            "no_checkpoint_selection": True,
            "new_policy_and_event_seeds": True,
        },
        "budgets": {
            "robust_source_diagnostic": {
                "steps": SOURCE_TOTAL_STEPS,
                "updates": SOURCE_UPDATE_COUNT,
            },
            "canonical_compensation_training_path": {
                "robust_source_steps": SOURCE_TOTAL_STEPS,
                "reference_finetune_steps": (
                    FINETUNE_ITERS * SAMPLES_PER_ITER),
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
            "primary_comparators": [
                ROBUST_LONG_ARM, ESCP_ARM, RESAC_ARM],
            "required_seed_wins_of_5": REQUIRED_SEED_WINS,
            "required_event_wins_of_15": REQUIRED_EVENT_WINS,
            "positive_paired_mean_and_ci95": True,
            "minimum_oracle_headroom": MIN_ORACLE_HEADROOM,
            "minimum_oracle_recovery": MIN_ORACLE_RECOVERY,
            "required_recovery_seeds": REQUIRED_RECOVERY_SEEDS,
            "minimum_stationary_oracle_retention": (
                MIN_STATIONARY_ORACLE_RETENTION),
            "required_stationary_retention_seeds": (
                REQUIRED_STATIONARY_RETENTION_SEEDS),
        },
        "claim_boundary": {
            "confirms_halfcheetah_actuator_polarity_only": True,
            "does_not_claim_general_invertible_dynamics": True,
            "does_not_claim_termination_safety": True,
            "training_cost_includes_robust_source": True,
        },
        "sync_policy": (
            "compact policy/controller parameters, protocol signatures, "
            "provenance, and JSON audits only; no replay or checkpoints"
        ),
        "source_records": {_relative(path): file_record(path) for path in paths},
    }


def create_registration() -> dict[str, Any]:
    payload = registration_payload()
    REGISTRATION_ROOT.mkdir(parents=True, exist_ok=True)
    if REGISTRATION_PATH.is_file():
        if read_json(REGISTRATION_PATH) != payload:
            raise ValueError("existing V31 registration changed")
    else:
        write_json_atomic(REGISTRATION_PATH, payload)
    return validate_registration()


def validate_registration() -> dict[str, Any]:
    if not REGISTRATION_PATH.is_file():
        raise FileNotFoundError(f"missing V31 registration: {REGISTRATION_PATH}")
    payload = read_json(REGISTRATION_PATH)
    if payload != registration_payload():
        raise ValueError("V31 registration or frozen source closure changed")
    return payload


def assert_protocol_integrity() -> None:
    if REFERENCE_MODE != 0:
        raise ValueError("V31 canonical reference changed")
    if set(TRAINING_SEEDS) & set(recipe.TRAINING_SEEDS):
        raise ValueError("V31 reused a V21 policy seed")
    if set(TRAINING_SEEDS) & set(v5_model.TRAIN_SOURCE_SEEDS):
        raise ValueError("V31 reused an estimator-training policy seed")
    if set(STATIONARY_HOLDOUT_EVENT_SEEDS) & set(SWITCHING_EVENT_SEEDS):
        raise ValueError("V31 stationary and switching streams overlap")
    if set(SWITCHING_SCHEDULES) != set(SWITCHING_EVENT_SEEDS):
        raise ValueError("V31 switching schedules are incomplete")
    if any(
        len(sequence) != len(MODES) or set(sequence) != set(MODES)
        for sequence in SWITCHING_SCHEDULES.values()
    ):
        raise ValueError("each V31 schedule must use all modes once")
    if FINAL_TOTAL_STEPS != 8_400_000 or SOURCE_TOTAL_STEPS != 5_600_000:
        raise ValueError("V31 interaction budgets changed")


assert_protocol_integrity()
