"""Independent confirmation of the full-state-final BAPR policy bank."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_system_id_v5 as v5_model,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_policy_stability_v20 as development,
)
from jax_experiments.analysis import (
    regime_polarity_v5_final_comparison_v18 as comparison,
)


ROOT = development.ROOT
PROTOCOL_VERSION = "v21-full-state-final-equal-policy-budget-confirmation"
ENV = development.ENV
FAMILY = development.FAMILY
MODES = development.MODES
TRAINING_SEEDS = (84_003, 84_021, 84_039, 84_057, 84_079)

CALIBRATION_EVENT_SEEDS = (186_001, 186_017, 186_033)
STATIONARY_HOLDOUT_EVENT_SEEDS = (186_101, 186_117, 186_133)
SWITCHING_EVENT_SEEDS = (186_201, 186_217, 186_233)
SWITCHING_SCHEDULES = {
    186_201: (0, 3, 1, 2),
    186_217: (2, 1, 3, 0),
    186_233: (3, 0, 2, 1),
}

SOURCE_NEXT_ITERATION = development.SOURCE_NEXT_ITERATION
SOURCE_ITERATION = development.SOURCE_ITERATION
SOURCE_TOTAL_STEPS = development.SOURCE_TOTAL_STEPS
SOURCE_UPDATE_COUNT = development.SOURCE_UPDATE_COUNT
SOURCE_START_TRAIN_STEPS = development.SOURCE_START_TRAIN_STEPS

SPECIALIST_VARIANT = "full_state_final"
SPECIALIST_FINETUNE_ITERS = development.FINETUNE_ITERS
SPECIALIST_FINAL_NEXT_ITERATION = development.FINAL_NEXT_ITERATION
SPECIALIST_FINAL_ITERATION = development.FINAL_ITERATION
SPECIALIST_FINAL_TOTAL_STEPS = development.FINAL_TOTAL_STEPS
SPECIALIST_FINAL_UPDATE_COUNT = development.FINAL_UPDATE_COUNT

TRAINED_METHODS = comparison.TRAINED_METHODS
SAC_REPLICA_SLOTS = comparison.SAC_REPLICA_SLOTS
SAC5_SLOTS = comparison.SAC5_SLOTS
MAX_ITERS = SOURCE_NEXT_ITERATION
FINAL_ITERATION = MAX_ITERS - 1
SAMPLES_PER_ITER = development.SAMPLES_PER_ITER
UPDATES_PER_ITER = development.UPDATES_PER_ITER
START_TRAIN_STEPS = SOURCE_START_TRAIN_STEPS
FINAL_TOTAL_STEPS = SOURCE_TOTAL_STEPS
FINAL_UPDATE_COUNT = SOURCE_UPDATE_COUNT
DWELL_STEPS = development.DWELL_STEPS
MAX_EPISODE_STEPS = development.MAX_EPISODE_STEPS
CALIBRATION_EPISODES_PER_MODE = comparison.CALIBRATION_EPISODES_PER_MODE
STATIONARY_EPISODES_PER_MODE = comparison.STATIONARY_EPISODES_PER_MODE
EPISODES_PER_TASK = STATIONARY_EPISODES_PER_MODE
SWITCHING_EPISODES = comparison.SWITCHING_EPISODES

ESCP_CONFIG = dict(comparison.ESCP_CONFIG)
RESAC_CONFIG = dict(comparison.RESAC_CONFIG)

PRIMARY_ARM = "bapr_v5_posterior_map"
STANDARD_BASELINES = ("robust_sac", "escp_recurrent", "resac_b0")
EQUAL_POLICY_BUDGET_BASELINE = "sac5_v5_posterior_map"
SWITCHING_ARMS = comparison.SWITCHING_ARMS
STATIONARY_ARMS = SWITCHING_ARMS

MIN_CALIBRATION_GAIN = development.MIN_CALIBRATION_GAIN
MIN_ORACLE_HEADROOM = comparison.MIN_ORACLE_HEADROOM
MIN_ORACLE_RECOVERY = comparison.MIN_ORACLE_RECOVERY
MIN_STATIONARY_RETENTION = comparison.MIN_STATIONARY_RETENTION
REQUIRED_SEED_WINS = comparison.REQUIRED_SEED_WINS
REQUIRED_EVENT_WINS = comparison.REQUIRED_EVENT_WINS
REQUIRED_RECOVERY_SEEDS = comparison.REQUIRED_RECOVERY_SEEDS
REQUIRED_STATIONARY_SEEDS = comparison.REQUIRED_STATIONARY_SEEDS

RUN_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_full_state_final_confirmation_v21"
)
SOURCE_RUN_ROOT = RUN_ROOT / "robust_sources"
SPECIALIST_RUN_ROOT = RUN_ROOT / "specialists"
BASELINE_RUN_ROOT = RUN_ROOT / "baselines"

SOURCE_BUNDLE_ROOT = (
    ROOT / "jax_experiments"
    / "eval_bundles_regime_polarity_robust_source_v21"
)
SPECIALIST_BUNDLE_ROOT = (
    ROOT / "jax_experiments"
    / "eval_bundles_regime_polarity_full_state_final_v21"
)
BASELINE_BUNDLE_ROOT = (
    ROOT / "jax_experiments"
    / "eval_bundles_regime_polarity_full_state_confirmation_v21"
)
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_full_state_final_confirmation_audit_v21"
)
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_full_state_final_confirmation_analysis_v21"
)
REGISTRATION_ROOT = (
    ROOT / "jax_experiments" / "deployments"
    / "regime_polarity_full_state_final_confirmation_v21"
)
REGISTRATION_PATH = REGISTRATION_ROOT / "registration.json"
PREREG_REPORT = (
    ROOT / "reports"
    / "regime_polarity_full_state_final_confirmation_v21_preregistration_2026-09-09.md"
)
REPORT = (
    ROOT / "reports"
    / "regime_polarity_full_state_final_confirmation_v21_2026-09-09.md"
)
V20_DIAGNOSTIC = (
    ROOT / "reports"
    / "regime_polarity_specialist_policy_stability_v20_diagnostic_2026-09-09.md"
)

POLICY_NAME = development.POLICY_NAME
CONTROLLER_STATE_NAME = development.CONTROLLER_STATE_NAME
BOOTSTRAP_NAME = development.BOOTSTRAP_NAME
SELECTION_NAME = development.SELECTION_NAME
EVAL_PARAMS_NAME = comparison.EVAL_PARAMS_NAME

SOURCE_BUNDLE_SCHEMA = "bapr.robust-source-controller-bundle.v21"
SPECIALIST_BUNDLE_SCHEMA = "bapr.full-state-final-specialist-bundle.v21"
SPECIALIST_BOOTSTRAP_SCHEMA = "bapr.full-state-final-bootstrap.v21"
BASELINE_BUNDLE_SCHEMA = "bapr.full-state-final-baseline-bundle.v21"
EVAL_PARAMS_SCHEMA = "bapr.evaluation-parameters.v21"
AUDIT_SCHEMA = "bapr.full-state-final-confirmation-audit.v21"
ANALYSIS_SCHEMA = "bapr.full-state-final-confirmation-analysis.v21"
REGISTRATION_SCHEMA = "bapr.full-state-final-confirmation-registration.v21"

file_record = development.file_record
read_json = development.read_json
write_json_atomic = development.write_json_atomic
write_text_atomic = development.write_text_atomic
checkpoint_record = development.checkpoint_record


def require_training_seed(seed: int) -> int:
    value = int(seed)
    if value not in TRAINING_SEEDS:
        raise ValueError(f"unknown v21 confirmation seed {value}")
    return value


require_seed = require_training_seed


def require_mode(mode: int) -> int:
    value = int(mode)
    if value not in MODES:
        raise ValueError(f"unknown actuator-polarity mode {value}")
    return value


def require_method(method: str) -> str:
    value = str(method)
    if value not in TRAINED_METHODS:
        raise ValueError(f"unknown v21 baseline method {value!r}")
    return value


def require_replica_slot(slot: int) -> int:
    value = int(slot)
    if value not in SAC_REPLICA_SLOTS:
        raise ValueError(f"unknown v21 SAC5 replica slot {value}")
    return value


def replica_training_seed(seed: int, slot: int) -> int:
    return require_training_seed(seed) + 1_000_000 * require_replica_slot(slot)


def require_switching_event_seed(seed: int) -> int:
    value = int(seed)
    if value not in SWITCHING_EVENT_SEEDS:
        raise ValueError(f"unknown v21 switching event seed {value}")
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
        "benchmark_role": "confirmation_robust_sac_controller_source",
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


def specialist_run_dir(seed: int, mode: int) -> Path:
    return (
        SPECIALIST_RUN_ROOT / f"seed_{require_training_seed(seed)}"
        / f"mode_{require_mode(mode)}"
    )


def specialist_bundle_dir(seed: int, mode: int) -> Path:
    return (
        SPECIALIST_BUNDLE_ROOT / f"seed_{require_training_seed(seed)}"
        / f"mode_{require_mode(mode)}"
    )


def specialist_bundle_manifest(seed: int, mode: int) -> Path:
    return specialist_bundle_dir(seed, mode) / "bundle_manifest.json"


def specialist_required_paths(seed: int, mode: int) -> tuple[Path, ...]:
    directory = specialist_bundle_dir(seed, mode)
    return (
        directory / "bundle_manifest.json",
        directory / "policy" / POLICY_NAME,
        directory / "logs" / "protocol_signature.json",
        directory / "provenance" / BOOTSTRAP_NAME,
        directory / "provenance" / SELECTION_NAME,
    )


def specialist_identity(seed: int, mode: int) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "frozen_full_state_final_specialist",
        "variant": SPECIALIST_VARIANT,
        "env": ENV,
        "family": FAMILY,
        "training_seed": require_training_seed(seed),
        "fixed_mode": require_mode(mode),
        "algo": "sac",
        "source_next_iteration": SOURCE_NEXT_ITERATION,
        "final_next_iteration": SPECIALIST_FINAL_NEXT_ITERATION,
        "final_total_steps": SPECIALIST_FINAL_TOTAL_STEPS,
        "final_update_count": SPECIALIST_FINAL_UPDATE_COUNT,
        "actor_update_period": 1,
        "select_best_validation": False,
    }


def expected_specialist_checkpoint() -> dict[str, Any]:
    return {
        "iteration": SPECIALIST_FINAL_ITERATION,
        "next_iteration": SPECIALIST_FINAL_NEXT_ITERATION,
        "total_steps": SPECIALIST_FINAL_TOTAL_STEPS,
        "update_count": SPECIALIST_FINAL_UPDATE_COUNT,
        "algo": "sac",
    }


def specialist_bundle_records(seed: int) -> dict[str, Any]:
    return {
        str(mode): file_record(specialist_bundle_manifest(seed, mode))
        for mode in MODES
    }


def baseline_run_dir(method: str, seed: int) -> Path:
    return (
        BASELINE_RUN_ROOT / require_method(method)
        / f"seed_{require_training_seed(seed)}"
    )


def baseline_bundle_dir(method: str, seed: int) -> Path:
    return (
        BASELINE_BUNDLE_ROOT / require_method(method)
        / f"seed_{require_training_seed(seed)}"
    )


def sac_run_dir(seed: int, slot: int) -> Path:
    return (
        BASELINE_RUN_ROOT / "sac5" / f"seed_{require_training_seed(seed)}"
        / f"replica_{require_replica_slot(slot)}"
    )


def sac_bundle_dir(seed: int, slot: int) -> Path:
    return (
        BASELINE_BUNDLE_ROOT / "sac5" / f"seed_{require_training_seed(seed)}"
        / f"replica_{require_replica_slot(slot)}"
    )


def bundle_dir(kind: str, seed: int, slot: int | None = None) -> Path:
    if kind == "sac_replica":
        if slot is None:
            raise ValueError("SAC replica bundle requires a slot")
        return sac_bundle_dir(seed, slot)
    if slot is not None:
        raise ValueError("non-SAC baseline bundle cannot have a slot")
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
            raise ValueError("SAC replica identity requires a slot")
        slot = require_replica_slot(slot)
        training_seed = replica_training_seed(seed, slot)
        algo = "sac"
    else:
        kind = require_method(kind)
        slot = None
        training_seed = seed
        algo = algo_for(kind)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "confirmation_equal_budget_baseline",
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


def frozen_policy_records(seed: int) -> dict[str, Any]:
    seed = require_training_seed(seed)
    return {
        "robust_source": file_record(source_manifest(seed)),
        "specialists": specialist_bundle_records(seed),
    }


def new_bundle_records(seed: int) -> dict[str, Any]:
    seed = require_training_seed(seed)
    return {
        **{
            method: file_record(bundle_manifest(method, seed))
            for method in TRAINED_METHODS
        },
        **{
            f"sac_replica_{slot}": file_record(
                bundle_manifest("sac_replica", seed, slot))
            for slot in SAC_REPLICA_SLOTS
        },
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


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def registration_source_paths() -> tuple[Path, ...]:
    paths = (
        ROOT / "jax_experiments/analysis/regime_polarity_full_state_final_confirmation_v21.py",
        ROOT / "jax_experiments/analysis/regime_polarity_full_state_specialist_v21.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_robust_source_v21.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_full_state_specialist_v21.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_full_state_confirmation_baseline_v21.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_full_state_confirmation_audit_v21.py",
        ROOT / "jax_experiments/analysis/analyze_regime_polarity_full_state_confirmation_v21.py",
        ROOT / "scripts/submit_regime_polarity_full_state_confirmation_v21.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_robust_source_v19.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_specialist_policy_stability_v20.py",
        ROOT / "jax_experiments/analysis/train_regime_polarity_sac_policy_stability_v20.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_v5_final_baseline_v18.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_v5_final_comparison_audit_v18.py",
        ROOT / "jax_experiments/analysis/analyze_regime_polarity_v5_final_comparison_v18.py",
        ROOT / "jax_experiments/algos/sac_policy_stability.py",
        ROOT / "jax_experiments/algos/sac_base.py",
        ROOT / "jax_experiments/algos/escp.py",
        ROOT / "jax_experiments/algos/resac.py",
        ROOT / "jax_experiments/common/checkpoint.py",
        ROOT / "jax_experiments/envs/stochastic_mode_env.py",
        ROOT / "jax_experiments/train.py",
        development.REGISTRATION_PATH,
        development.analysis_json(),
        development.analysis_markdown(),
        development.REPORT,
        V20_DIAGNOSTIC,
        comparison.REGISTRATION_PATH,
        comparison.analysis_json(),
        comparison.analysis_markdown(),
        comparison.REPORT,
        v5_model.MODEL_MANIFEST,
        v5_model.MODEL_PATH,
        PREREG_REPORT,
    )
    return tuple(dict.fromkeys(path.resolve() for path in paths))


def _development_evidence() -> dict[str, Any]:
    payload = read_json(development.analysis_json())
    rows = payload["cells"][SPECIALIST_VARIANT]
    seed_passes = sum(bool(row["pass"]) for row in rows.values())
    mode_passes = sum(
        int(row["stationary"]["mode_wins"]) for row in rows.values())
    event_wins = sum(
        int(row["switching"]["safe_event_wins"]) for row in rows.values())
    minimum_gain = min(
        float(mode_row["relative_gain"])
        for row in rows.values()
        for mode_row in row["stationary"]["modes"].values()
    )
    if (
        seed_passes != len(development.TRAINING_SEEDS)
        or mode_passes != len(development.TRAINING_SEEDS) * len(MODES)
        or event_wins != (
            len(development.TRAINING_SEEDS)
            * len(development.SWITCHING_EVENT_SEEDS))
        or minimum_gain <= 0.0
    ):
        raise ValueError("V20 full-state-final evidence does not authorize V21")
    return {
        "v20_seed_passes": seed_passes,
        "v20_stationary_mode_passes": mode_passes,
        "v20_switching_event_wins": event_wins,
        "v20_minimum_stationary_relative_gain": minimum_gain,
        "v20_mean_safe_switching_return": float(
            payload["mean_safe_switching_return"][SPECIALIST_VARIANT]),
        "v20_registered_candidate_selected": payload["selected_candidate"],
        "selection_arm_execution_deviation_recorded": True,
    }


def registration_payload() -> dict[str, Any]:
    development.validate_registration()
    comparison.validate_registration()
    evidence = _development_evidence()
    paths = registration_source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"v21 registration sources missing: {missing}")
    return {
        "schema": REGISTRATION_SCHEMA,
        "status": "registered",
        "created_before_training": True,
        "identity": {
            "protocol_version": PROTOCOL_VERSION,
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
        "development_basis": evidence,
        "execution_boundary": {
            "v20_registered_result_not_rewritten": True,
            "v20_best_selection_arms_not_used": True,
            "frozen_specialist_recipe": (
                "copy actor, critic, target critic, and alpha; reset all Adam "
                "states and replay; fixed-mode train to the final 2.8M-step "
                "endpoint with actor period 1"
            ),
            "frozen_estimator": "v5 executed-action posterior MAP",
            "environment_unchanged": True,
            "fresh_policy_and_event_seeds": True,
            "no_checkpoint_selection": True,
        },
        "budgets": {
            "robust_source": {
                "policies": 1,
                "steps_per_policy": SOURCE_TOTAL_STEPS,
                "updates_per_policy": SOURCE_UPDATE_COUNT,
            },
            "bapr_specialists": {
                "policies": len(MODES),
                "additional_steps_per_policy": (
                    SPECIALIST_FINETUNE_ITERS * SAMPLES_PER_ITER),
                "final_updates_per_policy": SPECIALIST_FINAL_UPDATE_COUNT,
            },
            "causal_sac5": {
                "policies": len(SAC5_SLOTS),
                "steps_per_policy": SOURCE_TOTAL_STEPS,
                "updates_per_policy": SOURCE_UPDATE_COUNT,
            },
            "standard_baselines": {
                "policies_per_method": 1,
                "steps_per_policy": SOURCE_TOTAL_STEPS,
                "updates_per_policy": SOURCE_UPDATE_COUNT,
            },
        },
        "decision_gate": {
            "comparators": [
                *STANDARD_BASELINES, EQUAL_POLICY_BUDGET_BASELINE],
            "positive_paired_mean_and_ci95": True,
            "required_seed_wins_of_5": REQUIRED_SEED_WINS,
            "required_event_wins_of_15": REQUIRED_EVENT_WINS,
            "minimum_oracle_headroom": MIN_ORACLE_HEADROOM,
            "minimum_oracle_recovery": MIN_ORACLE_RECOVERY,
            "required_oracle_recovery_seeds": REQUIRED_RECOVERY_SEEDS,
            "minimum_stationary_retention": MIN_STATIONARY_RETENTION,
            "required_stationary_retention_seeds": REQUIRED_STATIONARY_SEEDS,
            "no_higher_termination": True,
        },
        "sync_policy": (
            "compact policy/controller parameters, protocol signatures, JSON "
            "provenance, and audit results only; never sync replay buffers or "
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
            raise ValueError("existing v21 registration changed")
    else:
        write_json_atomic(REGISTRATION_PATH, payload)
    return validate_registration()


def validate_registration() -> dict[str, Any]:
    if not REGISTRATION_PATH.is_file():
        raise FileNotFoundError(f"missing v21 registration: {REGISTRATION_PATH}")
    payload = read_json(REGISTRATION_PATH)
    if payload != registration_payload():
        raise ValueError("v21 registration or source closure changed")
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
        raise ValueError("v21 calibration and holdout streams overlap")
    previous_policy_seeds = (
        set(development.TRAINING_SEEDS)
        | set(comparison.TRAINING_SEEDS)
    )
    if set(TRAINING_SEEDS) & previous_policy_seeds:
        raise ValueError("v21 reused a V18 or V20 policy seed")
    previous_events = (
        set(development.CALIBRATION_EVENT_SEEDS)
        | set(development.STATIONARY_HOLDOUT_EVENT_SEEDS)
        | set(development.SWITCHING_EVENT_SEEDS)
        | set(comparison.CALIBRATION_EVENT_SEEDS)
        | set(comparison.STATIONARY_HOLDOUT_EVENT_SEEDS)
        | set(comparison.SWITCHING_EVENT_SEEDS)
    )
    if previous_events & set().union(*splits):
        raise ValueError("v21 reused a V18 or V20 evaluation stream")
    if set(SWITCHING_SCHEDULES) != set(SWITCHING_EVENT_SEEDS):
        raise ValueError("v21 switching schedules are incomplete")
    schedules = list(SWITCHING_SCHEDULES.values())
    if len(set(schedules)) != len(schedules):
        raise ValueError("v21 switching schedules are not distinct")
    if any(
        len(sequence) != len(MODES) or set(sequence) != set(MODES)
        for sequence in schedules
    ):
        raise ValueError("each v21 switching schedule must use every mode")


assert_protocol_integrity()
