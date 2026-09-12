"""Preregistered final comparison for frozen v5 BAPR policy routing."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_corrected_baselines_v2 as corrected,
)
from jax_experiments.analysis import (
    regime_polarity_fresh_bank_estimator_confirmation_v17 as frozen,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_system_id_v5 as v5_model,
)


ROOT = Path(os.environ.get(
    "BAPR_WORKSPACE_ROOT", Path(__file__).resolve().parents[2])).resolve()
PROTOCOL_VERSION = "v18-v5-final-equal-policy-budget-comparison"
ENV = frozen.ENV
FAMILY = frozen.FAMILY
MODES = frozen.MODES
TRAINING_SEEDS = frozen.TRAINING_SEEDS

CALIBRATION_EVENT_SEEDS = (182_001, 182_017, 182_033)
STATIONARY_HOLDOUT_EVENT_SEEDS = (182_101, 182_117, 182_133)
SWITCHING_EVENT_SEEDS = (182_201, 182_217, 182_233)
SWITCHING_SCHEDULES = {
    182_201: (2, 0, 3, 1),
    182_217: (1, 3, 0, 2),
    182_233: (3, 2, 1, 0),
}

TRAINED_METHODS = ("escp_recurrent", "resac_b0")
SAC_REPLICA_SLOTS = (1, 2, 3, 4)
SAC5_SLOTS = (0, *SAC_REPLICA_SLOTS)

MAX_ITERS = frozen.SOURCE_NEXT_ITERATION
FINAL_ITERATION = MAX_ITERS - 1
SAMPLES_PER_ITER = frozen.SAMPLES_PER_ITER
UPDATES_PER_ITER = frozen.UPDATES_PER_ITER
START_TRAIN_STEPS = frozen.SOURCE_START_TRAIN_STEPS
FINAL_TOTAL_STEPS = MAX_ITERS * SAMPLES_PER_ITER
FINAL_UPDATE_COUNT = MAX_ITERS * UPDATES_PER_ITER
DWELL_STEPS = frozen.DWELL_STEPS
MAX_EPISODE_STEPS = frozen.MAX_EPISODE_STEPS
CALIBRATION_EPISODES_PER_MODE = 3
STATIONARY_EPISODES_PER_MODE = 5
SWITCHING_EPISODES = 5

ESCP_CONFIG = dict(corrected.ESCP_CONFIG)
RESAC_CONFIG = dict(corrected.RESAC_CONFIG)

PRIMARY_ARM = "bapr_v5_posterior_map"
STANDARD_BASELINES = ("robust_sac", "escp_recurrent", "resac_b0")
EQUAL_POLICY_BUDGET_BASELINE = "sac5_v5_posterior_map"
SWITCHING_ARMS = (
    "robust_sac",
    "bapr_true_mode_oracle",
    PRIMARY_ARM,
    "escp_recurrent",
    "resac_b0",
    "sac5_best_static",
    "sac5_true_mode_oracle",
    EQUAL_POLICY_BUDGET_BASELINE,
)
STATIONARY_ARMS = SWITCHING_ARMS

MIN_CALIBRATION_GAIN = frozen.MIN_CALIBRATION_GAIN
MIN_ORACLE_HEADROOM = 0.10
MIN_ORACLE_RECOVERY = 0.70
MIN_STATIONARY_RETENTION = 0.95
REQUIRED_SEED_WINS = 4
REQUIRED_EVENT_WINS = 12
REQUIRED_RECOVERY_SEEDS = 4
REQUIRED_STATIONARY_SEEDS = 4

RUN_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_v5_final_comparison_v18"
)
BUNDLE_ROOT = (
    ROOT / "jax_experiments"
    / "eval_bundles_regime_polarity_v5_final_comparison_v18"
)
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_v5_final_comparison_audit_v18"
)
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_v5_final_comparison_analysis_v18"
)
REGISTRATION_ROOT = (
    ROOT / "jax_experiments" / "deployments"
    / "regime_polarity_v5_final_comparison_v18"
)
REGISTRATION_PATH = REGISTRATION_ROOT / "registration.json"
PREREG_REPORT = (
    ROOT / "reports"
    / "regime_polarity_v5_final_comparison_v18_preregistration_2026-09-01.md"
)
REPORT = (
    ROOT / "reports"
    / "regime_polarity_v5_final_comparison_v18_2026-09-01.md"
)

EVAL_PARAMS_NAME = "eval_params.pkl"
BUNDLE_SCHEMA = "bapr.regime-polarity-v5-final-baseline-bundle.v18"
AUDIT_SCHEMA = "bapr.regime-polarity-v5-final-comparison-audit.v18"
ANALYSIS_SCHEMA = "bapr.regime-polarity-v5-final-comparison-analysis.v18"
REGISTRATION_SCHEMA = (
    "bapr.regime-polarity-v5-final-comparison-registration.v18")

file_record = frozen.file_record
read_json = frozen.read_json
write_json_atomic = frozen.write_json_atomic
write_text_atomic = frozen.write_text_atomic
checkpoint_record = frozen.checkpoint_record


def require_training_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in TRAINING_SEEDS:
        raise ValueError(f"unknown v18 outer seed {seed}")
    return seed


require_seed = require_training_seed


def require_mode(mode: int) -> int:
    mode = int(mode)
    if mode not in MODES:
        raise ValueError(f"unknown actuator-polarity mode {mode}")
    return mode


def require_method(method: str) -> str:
    method = str(method)
    if method not in TRAINED_METHODS:
        raise ValueError(f"unknown v18 trained method {method!r}")
    return method


def require_replica_slot(slot: int) -> int:
    slot = int(slot)
    if slot not in SAC_REPLICA_SLOTS:
        raise ValueError(f"unknown v18 SAC replica slot {slot}")
    return slot


def replica_training_seed(seed: int, slot: int) -> int:
    return require_training_seed(seed) + 1_000_000 * require_replica_slot(slot)


def require_switching_event_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in SWITCHING_EVENT_SEEDS:
        raise ValueError(f"unknown v18 switching event seed {seed}")
    return seed


def switching_sequence(event_seed: int, episode: int) -> tuple[int, ...]:
    base = SWITCHING_SCHEDULES[require_switching_event_seed(event_seed)]
    shift = int(episode) % len(base)
    return tuple(base[shift:] + base[:shift])


def algo_for(method: str) -> str:
    method = require_method(method)
    return "escp" if method == "escp_recurrent" else "resac"


def baseline_run_dir(method: str, seed: int) -> Path:
    return RUN_ROOT / method / f"seed_{require_training_seed(seed)}"


def baseline_bundle_dir(method: str, seed: int) -> Path:
    return BUNDLE_ROOT / method / f"seed_{require_training_seed(seed)}"


def sac_run_dir(seed: int, slot: int) -> Path:
    return (
        RUN_ROOT / "sac5" / f"seed_{require_training_seed(seed)}"
        / f"replica_{require_replica_slot(slot)}"
    )


def sac_bundle_dir(seed: int, slot: int) -> Path:
    return (
        BUNDLE_ROOT / "sac5" / f"seed_{require_training_seed(seed)}"
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


def identity(
    kind: str, seed: int, slot: int | None = None,
) -> dict[str, Any]:
    seed = require_training_seed(seed)
    if kind == "sac_replica":
        if slot is None:
            raise ValueError("SAC replica identity requires a slot")
        slot = require_replica_slot(slot)
        method = "sac"
        training_seed = replica_training_seed(seed, slot)
    else:
        method = require_method(kind)
        training_seed = seed
        slot = None
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "final_equal_budget_baseline",
        "kind": kind,
        "outer_seed": seed,
        "replica_slot": slot,
        "training_seed": training_seed,
        "algo": "sac" if kind == "sac_replica" else algo_for(method),
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


def frozen_policy_records(seed: int) -> dict[str, Any]:
    seed = require_training_seed(seed)
    return {
        "robust_source": file_record(frozen.source_manifest(seed)),
        "specialists": {
            str(mode): file_record(
                frozen.bundle_manifest("actor_only", seed, mode))
            for mode in MODES
        },
    }


def registration_source_paths() -> tuple[Path, ...]:
    return (
        ROOT / "jax_experiments/analysis/regime_polarity_v5_final_comparison_v18.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_v5_final_baseline_v18.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_v5_final_comparison_audit_v18.py",
        ROOT / "jax_experiments/analysis/analyze_regime_polarity_v5_final_comparison_v18.py",
        ROOT / "jax_experiments/algos/escp.py",
        ROOT / "jax_experiments/algos/resac.py",
        ROOT / "jax_experiments/algos/sac_base.py",
        ROOT / "jax_experiments/common/checkpoint.py",
        ROOT / "jax_experiments/envs/brax_env.py",
        ROOT / "jax_experiments/train.py",
        ROOT / "scripts/submit_regime_polarity_v5_final_comparison_v18.py",
        frozen.REGISTRATION_PATH,
        frozen.analysis_json(),
        frozen.analysis_markdown(),
        frozen.REPORT,
        v5_model.MODEL_MANIFEST,
        v5_model.MODEL_PATH,
        PREREG_REPORT,
    )


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def registration_payload() -> dict[str, Any]:
    frozen.validate_registration()
    frozen_analysis = read_json(frozen.analysis_json())
    if frozen_analysis.get("policy_bank_confirmation_pass") is not True:
        raise ValueError("v17 did not confirm the fresh actor-only policy banks")
    if frozen_analysis.get("close_executed_action_inverse_estimator_family") is not True:
        raise ValueError("v17 did not close estimator development before v18")
    paths = registration_source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    for seed in TRAINING_SEEDS:
        missing.extend(
            str(path) for path in frozen.source_required_paths(seed)
            if not path.is_file())
        for mode in MODES:
            missing.extend(
                str(path) for path in frozen.bundle_required_paths(
                    "actor_only", seed, mode)
                if not path.is_file())
    if missing:
        raise FileNotFoundError(f"v18 frozen inputs missing: {missing}")
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
        "frozen_candidate": {
            "policy_bank": "v17 actor-only robust-warm-start bank",
            "router": "v5 posterior MAP",
            "estimator_updates": False,
        },
        "budget": {
            "steps_per_policy": FINAL_TOTAL_STEPS,
            "updates_per_policy": FINAL_UPDATE_COUNT,
            "sac5_total_policies": len(SAC5_SLOTS),
            "sac5_slot_0": "reuse matched v17 robust SAC",
            "sac5_new_replicas": list(SAC_REPLICA_SLOTS),
        },
        "baselines": {
            "standard": list(STANDARD_BASELINES),
            "equal_total_policy_budget": EQUAL_POLICY_BUDGET_BASELINE,
            "diagnostics": [
                "bapr_true_mode_oracle",
                "sac5_best_static",
                "sac5_true_mode_oracle",
            ],
            "escp_config": ESCP_CONFIG,
            "resac_config": RESAC_CONFIG,
        },
        "decision_gate": {
            "primary_arm": PRIMARY_ARM,
            "comparators": [
                *STANDARD_BASELINES, EQUAL_POLICY_BUDGET_BASELINE],
            "required_seed_wins": REQUIRED_SEED_WINS,
            "required_event_wins_of_15": REQUIRED_EVENT_WINS,
            "require_positive_paired_mean": True,
            "require_positive_paired_ci95_lower": True,
            "require_no_termination_increase": True,
            "minimum_oracle_headroom": MIN_ORACLE_HEADROOM,
            "minimum_oracle_recovery": MIN_ORACLE_RECOVERY,
            "required_recovery_seeds": REQUIRED_RECOVERY_SEEDS,
            "minimum_stationary_retention": MIN_STATIONARY_RETENTION,
            "required_stationary_retention_seeds": REQUIRED_STATIONARY_SEEDS,
        },
        "sync_policy": (
            "JSON, protocol signature, and evaluation-only parameters; "
            "never sync replay or full checkpoints"
        ),
        "frozen_inputs": {
            str(seed): frozen_policy_records(seed)
            for seed in TRAINING_SEEDS
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
            raise ValueError("existing v18 registration changed")
    else:
        write_json_atomic(REGISTRATION_PATH, payload)
    return validate_registration()


def validate_registration() -> dict[str, Any]:
    if not REGISTRATION_PATH.is_file():
        raise FileNotFoundError(f"missing v18 registration: {REGISTRATION_PATH}")
    payload = read_json(REGISTRATION_PATH)
    if payload != registration_payload():
        raise ValueError("v18 registration or frozen source closure changed")
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
        raise ValueError("v18 calibration and holdout streams overlap")
    prior_events = (
        set(frozen.CALIBRATION_EVENT_SEEDS)
        | set(frozen.STATIONARY_HOLDOUT_EVENT_SEEDS)
        | set(frozen.SWITCHING_EVENT_SEEDS)
    )
    if prior_events & set().union(*splits):
        raise ValueError("v18 reused a v17 event stream")
    if set(SWITCHING_SCHEDULES) != set(SWITCHING_EVENT_SEEDS):
        raise ValueError("v18 switching schedules are incomplete")
    schedules = list(SWITCHING_SCHEDULES.values())
    if len(set(schedules)) != len(schedules):
        raise ValueError("v18 switching schedules are not distinct")
    if any(len(row) != len(MODES) or set(row) != set(MODES)
           for row in schedules):
        raise ValueError("each v18 switching schedule must use every mode")


assert_protocol_integrity()
