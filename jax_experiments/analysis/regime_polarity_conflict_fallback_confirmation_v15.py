"""Holdout confirmation of the frozen v14 confirm-3 fallback router."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_conflict_fallback_router_v14 as parent,
)
from jax_experiments.analysis import (
    regime_polarity_frozen_estimator_transfer_v13 as transfer_parent,
)
from jax_experiments.analysis import (
    regime_polarity_robust_warmstart_confirmation_v12 as policy_parent,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_system_id_v5 as estimator,
)


ROOT = parent.ROOT
PROTOCOL_VERSION = "v15-conflict-fallback-confirmation"
ENV = parent.ENV
FAMILY = parent.FAMILY
MODES = parent.MODES
TRAINING_SEEDS = parent.TRAINING_SEEDS

EVENT_SEEDS = (170_201, 170_217, 170_233)
SWITCHING_SCHEDULES = {
    170_201: (2, 1, 3, 0),
    170_217: (0, 2, 3, 1),
    170_233: (3, 2, 1, 0),
}
DWELL_STEPS = parent.DWELL_STEPS
MAX_EPISODE_STEPS = parent.MAX_EPISODE_STEPS
SWITCHING_EPISODES = parent.SWITCHING_EPISODES
DELAY_STEPS = parent.DELAY_STEPS

SELECTED_ARM = "conflict_fallback_confirm3_safe_utility"
BASE_ARMS = parent.BASE_ARMS
CANDIDATE_ARMS = (SELECTED_ARM,)
ARMS = (*BASE_ARMS, SELECTED_ARM)
POSTERIOR_EXIT_CONFIDENCE = parent.POSTERIOR_EXIT_CONFIDENCE
CONFLICT_LOG_LIKELIHOOD_MARGIN = parent.CONFLICT_LOG_LIKELIHOOD_MARGIN
MIN_FALLBACK_ACTIONS_AFTER_CONFLICT = (
    parent.MIN_FALLBACK_ACTIONS_AFTER_CONFLICT)

MIN_HEADROOM_GAIN = parent.MIN_HEADROOM_GAIN
MIN_CAUSAL_RETENTION = parent.MIN_CAUSAL_RETENTION
MIN_ROUTER_GAIN = parent.MIN_ROUTER_GAIN
REQUIRED_SEED_PASSES = parent.REQUIRED_SEED_PASSES
REQUIRED_MAP_SEED_WINS = 4

AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_conflict_fallback_confirmation_audit_v15"
)
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_conflict_fallback_confirmation_analysis_v15"
)
REGISTRATION_ROOT = (
    ROOT / "jax_experiments" / "deployments"
    / "regime_polarity_conflict_fallback_confirmation_v15"
)
REGISTRATION_PATH = REGISTRATION_ROOT / "registration.json"
PREREG_REPORT = (
    ROOT / "reports"
    / "regime_polarity_conflict_fallback_confirmation_v15_preregistration_2026-09-01.md"
)
REPORT = (
    ROOT / "reports"
    / "regime_polarity_conflict_fallback_confirmation_v15_2026-09-01.md"
)

EVENT_SCHEMA = "bapr.evidence-conflict-fallback-event.v15"
AUDIT_SCHEMA = "bapr.evidence-conflict-fallback-audit.v15"
ANALYSIS_SCHEMA = "bapr.evidence-conflict-fallback-analysis.v15"
REGISTRATION_SCHEMA = "bapr.evidence-conflict-fallback-registration.v15"

read_json = parent.read_json
write_json_atomic = parent.write_json_atomic
write_text_atomic = parent.write_text_atomic
file_record = parent.file_record
policy_bank_records = parent.policy_bank_records
estimator_records = parent.estimator_records


def require_training_seed(seed: int) -> int:
    return parent.require_training_seed(seed)


def require_event_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in EVENT_SEEDS:
        raise ValueError(f"unknown v15 event seed {seed}")
    return seed


def require_arm(arm: str) -> str:
    arm = str(arm)
    if arm not in ARMS:
        raise ValueError(f"unknown v15 arm {arm!r}")
    return arm


def confirm_steps_for_arm(arm: str) -> int:
    return 3 if require_arm(arm) == SELECTED_ARM else 0


def switching_sequence(event_seed: int, episode: int) -> tuple[int, ...]:
    base = SWITCHING_SCHEDULES[require_event_seed(event_seed)]
    shift = int(episode) % len(base)
    return tuple(base[shift:] + base[:shift])


def audit_dir(seed: int) -> Path:
    return AUDIT_ROOT / f"seed_{require_training_seed(seed)}"


def event_result(seed: int, event_seed: int) -> Path:
    return (
        audit_dir(seed)
        / f"event_seed_{require_event_seed(event_seed)}"
        / "results.json"
    )


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
        ROOT / "jax_experiments/analysis/regime_polarity_conflict_fallback_confirmation_v15.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_conflict_fallback_confirmation_audit_v15.py",
        ROOT / "jax_experiments/analysis/analyze_regime_polarity_conflict_fallback_confirmation_v15.py",
        parent.REGISTRATION_PATH,
        parent.analysis_json(),
        parent.analysis_markdown(),
        parent.REPORT,
        PREREG_REPORT,
    )
    return tuple(path.resolve() for path in paths)


def registration_payload() -> dict[str, Any]:
    parent.validate_registration()
    development = read_json(parent.analysis_json())
    if (
        development.get("schema") != parent.ANALYSIS_SCHEMA
        or development.get("status") != "complete"
        or development.get("router_development_pass") is not True
        or development.get("selected_arm") != SELECTED_ARM
    ):
        raise ValueError("v14 did not select the frozen confirm-3 router")
    paths = registration_source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"v15 registration sources missing: {missing}")
    return {
        "schema": REGISTRATION_SCHEMA,
        "status": "registered",
        "created_before_audit": True,
        "identity": {
            "protocol_version": PROTOCOL_VERSION,
            "parent_protocol_version": parent.PROTOCOL_VERSION,
            "selected_arm": SELECTED_ARM,
            "training_seeds": list(TRAINING_SEEDS),
            "event_seeds": list(EVENT_SEEDS),
            "switching_schedules": {
                str(seed): list(sequence)
                for seed, sequence in SWITCHING_SCHEDULES.items()
            },
            "arms": list(ARMS),
            "dwell_steps": DWELL_STEPS,
            "max_episode_steps": MAX_EPISODE_STEPS,
            "switching_episodes": SWITCHING_EPISODES,
        },
        "router": {
            "confirmation_steps": 3,
            "conflict_log_likelihood_margin": (
                CONFLICT_LOG_LIKELIHOOD_MARGIN),
            "posterior_exit_confidence": POSTERIOR_EXIT_CONFIDENCE,
            "minimum_fallback_actions_after_conflict": (
                MIN_FALLBACK_ACTIONS_AFTER_CONFLICT),
            "candidate_forbidden_inputs": [
                "mode_id", "switch_clock", "dwell_boundary", "action_gain",
            ],
        },
        "decision_rule": {
            "minimum_safe_oracle_gain": MIN_HEADROOM_GAIN,
            "minimum_delay_4_headroom_retention": MIN_CAUSAL_RETENTION,
            "minimum_candidate_gain": MIN_ROUTER_GAIN,
            "minimum_candidate_headroom_recovery": MIN_CAUSAL_RETENTION,
            "required_event_wins": len(EVENT_SEEDS),
            "required_seed_passes": REQUIRED_SEED_PASSES,
            "required_seed_wins_over_plain_map": REQUIRED_MAP_SEED_WINS,
            "require_positive_paired_mean_over_plain_map": True,
            "require_zero_termination": True,
        },
        "data_policy": {
            "checkpoint_only": True,
            "new_training": False,
            "single_frozen_candidate": True,
            "result_payload": "json_and_markdown_only",
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
            raise ValueError("existing v15 registration changed")
    else:
        write_json_atomic(REGISTRATION_PATH, payload)
    return validate_registration()


def validate_registration() -> dict[str, Any]:
    if not REGISTRATION_PATH.is_file():
        raise FileNotFoundError(f"missing v15 registration: {REGISTRATION_PATH}")
    payload = read_json(REGISTRATION_PATH)
    if payload != registration_payload():
        raise ValueError("v15 registration or source closure changed")
    return payload


def assert_protocol_integrity() -> None:
    prior_events = (
        set(parent.EVENT_SEEDS)
        | set(transfer_parent.EVENT_SEEDS)
        | set(policy_parent.CALIBRATION_EVENT_SEEDS)
        | set(policy_parent.STATIONARY_HOLDOUT_EVENT_SEEDS)
        | set(policy_parent.SWITCHING_EVENT_SEEDS)
        | set(estimator.TRAIN_EVENT_SEEDS)
        | set(estimator.VALIDATION_EVENT_SEEDS)
        | set(estimator.AUDIT_EVENT_SEEDS)
    )
    if prior_events & set(EVENT_SEEDS):
        raise ValueError("v15 reused an earlier event seed")
    if set(SWITCHING_SCHEDULES) != set(EVENT_SEEDS):
        raise ValueError("v15 switching schedules are incomplete")
    cycles = []
    for sequence in SWITCHING_SCHEDULES.values():
        if len(sequence) != len(MODES) or set(sequence) != set(MODES):
            raise ValueError("each v15 schedule must use every mode once")
        cycles.append(tuple(sequence))
    if len(set(cycles)) != len(cycles):
        raise ValueError("v15 switching schedules are not distinct")


assert_protocol_integrity()
