"""Switch-weighted evidence retraining on the confirmed v12 policy banks."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_conflict_fallback_confirmation_v15 as parent,
)
from jax_experiments.analysis import (
    regime_polarity_conflict_fallback_router_v14 as router_parent,
)
from jax_experiments.analysis import (
    regime_polarity_frozen_estimator_transfer_v13 as transfer_parent,
)
from jax_experiments.analysis import (
    regime_polarity_robust_warmstart_confirmation_v12 as policy_parent,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_system_id_v5 as estimator_parent,
)


ROOT = parent.ROOT
PROTOCOL_VERSION = "v16-switch-weighted-expected-action-estimator"
ENV = parent.ENV
FAMILY = parent.FAMILY
MODES = parent.MODES

TRAIN_POLICY_SEEDS = (71_003, 71_021, 71_039)
VALIDATION_POLICY_SEEDS = (71_057, 71_079)
AUDIT_POLICY_SEEDS = parent.TRAINING_SEEDS

TRAIN_EVENT_SEEDS = (171_001, 171_013)
VALIDATION_EVENT_SEEDS = (171_101, 171_113)
AUDIT_EVENT_SEEDS = (171_201, 171_217, 171_233)
SWITCHING_SCHEDULES = {
    171_001: (0, 1, 3, 2),
    171_013: (0, 2, 1, 3),
    171_101: (0, 3, 1, 2),
    171_113: (0, 1, 2, 3),
    171_201: (0, 2, 3, 1),
    171_217: (0, 3, 2, 1),
    171_233: (0, 2, 1, 3),
}

DWELL_STEPS = parent.DWELL_STEPS
MAX_EPISODE_STEPS = parent.MAX_EPISODE_STEPS
TRAINING_EPISODES = 4
AUDIT_EPISODES = parent.SWITCHING_EPISODES
SWITCH_WINDOW_STEPS = 8
SWITCH_SAMPLE_WEIGHT = 8.0
TRAIN_UPDATES = 2_000
BATCH_SIZE = 256
CLASSIFICATION_LOSS_WEIGHT = 0.10
MODEL_SEED = 20_260_901
MODEL_CONFIG = dict(estimator_parent.MODEL_CONFIG)
MODEL_CONFIG["learning_rate"] = 5e-5
FILTER_CONFIG = {
    "hazard_rate": 0.002,
    "evidence_scale": 1.0,
    "posterior_decay": 0.98,
}

ARMS = (
    "robust_sac",
    "true_mode_safe_utility",
    "delayed_oracle_4_safe_utility",
    "frozen_v5_posterior_map",
    "switch_weighted_v16_posterior_map",
)
DELAY_STEPS = parent.DELAY_STEPS
MIN_HEADROOM_GAIN = parent.MIN_HEADROOM_GAIN
MIN_CAUSAL_RETENTION = parent.MIN_CAUSAL_RETENTION
MIN_ESTIMATOR_GAIN = parent.MIN_ROUTER_GAIN
REQUIRED_SEED_PASSES = parent.REQUIRED_SEED_PASSES
REQUIRED_V5_SEED_WINS = 4
REQUIRED_VALIDATION_POLICY_WINS = len(VALIDATION_POLICY_SEEDS)

MODEL_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_switch_weighted_estimator_model_v16"
)
MODEL_PATH = MODEL_ROOT / "inverse_params.npz"
MODEL_MANIFEST = MODEL_ROOT / "model_manifest.json"
CHECKPOINT_ROOT = MODEL_ROOT / "checkpoints"
TRAIN_STATE_JSON = CHECKPOINT_ROOT / "train_state.json"
TRAIN_STATE_NPZ = CHECKPOINT_ROOT / "train_state.npz"
TRAIN_STATE_PKL = CHECKPOINT_ROOT / "train_state.pkl"
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_switch_weighted_estimator_audit_v16"
)
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_switch_weighted_estimator_analysis_v16"
)
REGISTRATION_ROOT = (
    ROOT / "jax_experiments" / "deployments"
    / "regime_polarity_switch_weighted_estimator_v16"
)
REGISTRATION_PATH = REGISTRATION_ROOT / "registration.json"
PREREG_REPORT = (
    ROOT / "reports"
    / "regime_polarity_switch_weighted_estimator_v16_preregistration_2026-09-01.md"
)
REPORT = (
    ROOT / "reports"
    / "regime_polarity_switch_weighted_estimator_v16_2026-09-01.md"
)

MODEL_SCHEMA = "bapr.switch-weighted-expected-action-model.v16"
EVENT_SCHEMA = "bapr.switch-weighted-expected-action-event.v16"
AUDIT_SCHEMA = "bapr.switch-weighted-expected-action-audit.v16"
ANALYSIS_SCHEMA = "bapr.switch-weighted-expected-action-analysis.v16"
REGISTRATION_SCHEMA = "bapr.switch-weighted-expected-action-registration.v16"
TRAIN_STATE_SCHEMA = "bapr.switch-weighted-expected-action-train-state.v16"

FilterConfig = estimator_parent.FilterConfig
mode_gain_vectors = estimator_parent.mode_gain_vectors
posterior_update = estimator_parent.posterior_update
read_json = parent.read_json
write_json_atomic = parent.write_json_atomic
write_text_atomic = parent.write_text_atomic
file_record = parent.file_record
save_parameter_state = estimator_parent.save_parameter_state
load_parameter_state = estimator_parent.load_parameter_state
policy_bank_records = parent.policy_bank_records


def require_policy_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in AUDIT_POLICY_SEEDS:
        raise ValueError(f"unknown v16 policy seed {seed}")
    return seed


def require_event_seed(seed: int) -> int:
    seed = int(seed)
    all_events = {
        *TRAIN_EVENT_SEEDS,
        *VALIDATION_EVENT_SEEDS,
        *AUDIT_EVENT_SEEDS,
    }
    if seed not in all_events:
        raise ValueError(f"unknown v16 event seed {seed}")
    return seed


def require_audit_event_seed(seed: int) -> int:
    seed = require_event_seed(seed)
    if seed not in AUDIT_EVENT_SEEDS:
        raise ValueError(f"event seed {seed} is not a v16 audit event")
    return seed


def require_arm(arm: str) -> str:
    arm = str(arm)
    if arm not in ARMS:
        raise ValueError(f"unknown v16 arm {arm!r}")
    return arm


def switching_sequence(event_seed: int, episode: int) -> tuple[int, ...]:
    base = SWITCHING_SCHEDULES[require_event_seed(event_seed)]
    shift = int(episode) % len(base)
    return tuple(base[shift:] + base[:shift])


def audit_dir(seed: int) -> Path:
    return AUDIT_ROOT / f"seed_{require_policy_seed(seed)}"


def event_result(seed: int, event_seed: int) -> Path:
    return (
        audit_dir(seed)
        / f"event_seed_{require_audit_event_seed(event_seed)}"
        / "results.json"
    )


def audit_manifest(seed: int) -> Path:
    return audit_dir(seed) / "audit_manifest.json"


def analysis_json() -> Path:
    return ANALYSIS_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return ANALYSIS_ROOT / "analysis.md"


def model_records() -> dict[str, Any]:
    return {
        "manifest": file_record(MODEL_MANIFEST),
        "parameters": file_record(MODEL_PATH),
    }


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def registration_source_paths() -> tuple[Path, ...]:
    paths: list[Path] = [
        ROOT / "jax_experiments/analysis/regime_polarity_switch_weighted_estimator_v16.py",
        ROOT / "jax_experiments/analysis/regime_polarity_switch_weighted_estimator_model_v16.py",
        ROOT / "jax_experiments/analysis/train_regime_polarity_switch_weighted_estimator_v16.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_switch_weighted_estimator_audit_v16.py",
        ROOT / "jax_experiments/analysis/analyze_regime_polarity_switch_weighted_estimator_v16.py",
        parent.REGISTRATION_PATH,
        parent.analysis_json(),
        parent.analysis_markdown(),
        parent.REPORT,
        policy_parent.REGISTRATION_PATH,
        policy_parent.analysis_json(),
        policy_parent.analysis_markdown(),
        policy_parent.REPORT,
        estimator_parent.MODEL_MANIFEST,
        estimator_parent.MODEL_PATH,
        estimator_parent.REPORT,
        PREREG_REPORT,
    ]
    for seed in AUDIT_POLICY_SEEDS:
        paths.append(policy_parent.source_manifest(seed))
        paths.extend(
            policy_parent.bundle_manifest("actor_only", seed, mode)
            for mode in MODES
        )
    return tuple(path.resolve() for path in paths)


def registration_payload() -> dict[str, Any]:
    parent.validate_registration()
    parent_analysis = read_json(parent.analysis_json())
    if (
        parent_analysis.get("schema") != parent.ANALYSIS_SCHEMA
        or parent_analysis.get("status") != "complete"
        or parent_analysis.get("router_confirmation_pass") is not False
        or parent_analysis.get("switch_focused_estimator_retraining_required")
        is not True
    ):
        raise ValueError("v15 did not authorize switch-focused estimator work")
    old_manifest = read_json(estimator_parent.MODEL_MANIFEST)
    if old_manifest.get("filter_config") != FILTER_CONFIG:
        raise ValueError("v16 filter differs from the frozen v5 filter")
    paths = registration_source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"v16 registration sources missing: {missing}")
    return {
        "schema": REGISTRATION_SCHEMA,
        "status": "registered",
        "created_before_training": True,
        "identity": {
            "protocol_version": PROTOCOL_VERSION,
            "parent_protocol_version": parent.PROTOCOL_VERSION,
            "train_policy_seeds": list(TRAIN_POLICY_SEEDS),
            "validation_policy_seeds": list(VALIDATION_POLICY_SEEDS),
            "audit_policy_seeds": list(AUDIT_POLICY_SEEDS),
            "train_event_seeds": list(TRAIN_EVENT_SEEDS),
            "validation_event_seeds": list(VALIDATION_EVENT_SEEDS),
            "audit_event_seeds": list(AUDIT_EVENT_SEEDS),
            "switching_schedules": {
                str(seed): list(sequence)
                for seed, sequence in SWITCHING_SCHEDULES.items()
            },
        },
        "training": {
            "initialization": "frozen_v5_expected_action_model",
            "model_config": MODEL_CONFIG,
            "model_seed": MODEL_SEED,
            "episodes_per_event": TRAINING_EPISODES,
            "updates": TRAIN_UPDATES,
            "batch_size_per_head": BATCH_SIZE,
            "switch_window_steps": SWITCH_WINDOW_STEPS,
            "switch_sample_weight": SWITCH_SAMPLE_WEIGHT,
            "classification_loss_weight": CLASSIFICATION_LOSS_WEIGHT,
            "filter_config": FILTER_CONFIG,
            "policy_updates": False,
        },
        "decision_rule": {
            "minimum_safe_oracle_gain": MIN_HEADROOM_GAIN,
            "minimum_delay_4_headroom_retention": MIN_CAUSAL_RETENTION,
            "minimum_v16_gain": MIN_ESTIMATOR_GAIN,
            "minimum_v16_headroom_recovery": MIN_CAUSAL_RETENTION,
            "required_event_wins": len(AUDIT_EVENT_SEEDS),
            "required_seed_passes": REQUIRED_SEED_PASSES,
            "required_seed_wins_over_frozen_v5": REQUIRED_V5_SEED_WINS,
            "required_validation_policy_wins": REQUIRED_VALIDATION_POLICY_WINS,
            "require_positive_paired_mean_over_frozen_v5": True,
            "require_zero_termination": True,
        },
        "data_policy": {
            "true_mode_use": "offline target and metric construction only",
            "deployed_inputs": [
                "observation", "commanded_action", "next_observation",
            ],
            "candidate_router": "plain posterior MAP",
            "candidate_forbidden_inputs": [
                "mode_id", "switch_clock", "dwell_boundary",
            ],
            "result_payload": "model_parameters_json_and_audit_json_only",
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
            raise ValueError("existing v16 registration changed")
    else:
        write_json_atomic(REGISTRATION_PATH, payload)
    return validate_registration()


def validate_registration() -> dict[str, Any]:
    if not REGISTRATION_PATH.is_file():
        raise FileNotFoundError(f"missing v16 registration: {REGISTRATION_PATH}")
    payload = read_json(REGISTRATION_PATH)
    if payload != registration_payload():
        raise ValueError("v16 registration or source closure changed")
    return payload


def assert_protocol_integrity() -> None:
    split_events = (
        set(TRAIN_EVENT_SEEDS),
        set(VALIDATION_EVENT_SEEDS),
        set(AUDIT_EVENT_SEEDS),
    )
    if any(left & right for index, left in enumerate(split_events)
           for right in split_events[index + 1:]):
        raise ValueError("v16 event splits overlap")
    prior_events = (
        set(parent.EVENT_SEEDS)
        | set(router_parent.EVENT_SEEDS)
        | set(transfer_parent.EVENT_SEEDS)
        | set(policy_parent.CALIBRATION_EVENT_SEEDS)
        | set(policy_parent.STATIONARY_HOLDOUT_EVENT_SEEDS)
        | set(policy_parent.SWITCHING_EVENT_SEEDS)
        | set(estimator_parent.TRAIN_EVENT_SEEDS)
        | set(estimator_parent.VALIDATION_EVENT_SEEDS)
        | set(estimator_parent.AUDIT_EVENT_SEEDS)
    )
    if prior_events & set().union(*split_events):
        raise ValueError("v16 reused an earlier event seed")
    if set(SWITCHING_SCHEDULES) != set().union(*split_events):
        raise ValueError("v16 switching schedules are incomplete")
    for sequence in SWITCHING_SCHEDULES.values():
        if len(sequence) != len(MODES) or set(sequence) != set(MODES):
            raise ValueError("each v16 schedule must use every mode once")
    if set(TRAIN_POLICY_SEEDS) & set(VALIDATION_POLICY_SEEDS):
        raise ValueError("v16 train and validation policy seeds overlap")
    if set(AUDIT_POLICY_SEEDS) != {
        *TRAIN_POLICY_SEEDS, *VALIDATION_POLICY_SEEDS,
    }:
        raise ValueError("v16 policy split does not cover the audit banks")


assert_protocol_integrity()
