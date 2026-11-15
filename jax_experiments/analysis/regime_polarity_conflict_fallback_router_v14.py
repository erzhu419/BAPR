"""Causal evidence-conflict fallback router development on frozen v12 banks."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_frozen_estimator_transfer_v13 as parent,
)
from jax_experiments.analysis import (
    regime_polarity_robust_warmstart_confirmation_v12 as policy_parent,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_system_id_v5 as estimator,
)


ROOT = parent.ROOT
PROTOCOL_VERSION = "v14-evidence-conflict-fallback-router-development"
ENV = parent.ENV
FAMILY = parent.FAMILY
MODES = parent.MODES
TRAINING_SEEDS = parent.TRAINING_SEEDS

EVENT_SEEDS = (169_201, 169_217, 169_233)
SWITCHING_SCHEDULES = {
    169_201: (0, 3, 1, 2),
    169_217: (1, 2, 0, 3),
    169_233: (3, 0, 2, 1),
}
DWELL_STEPS = parent.DWELL_STEPS
MAX_EPISODE_STEPS = parent.MAX_EPISODE_STEPS
SWITCHING_EPISODES = parent.SWITCHING_EPISODES
DELAY_STEPS = parent.DELAY_STEPS

BASE_ARMS = (
    "robust_sac",
    "true_mode_safe_utility",
    "delayed_oracle_4_safe_utility",
    "posterior_map_safe_utility",
)
CANDIDATE_ARMS = tuple(
    f"conflict_fallback_confirm{steps}_safe_utility"
    for steps in (1, 2, 3)
)
ARMS = (*BASE_ARMS, *CANDIDATE_ARMS)
POSTERIOR_EXIT_CONFIDENCE = 0.80
CONFLICT_LOG_LIKELIHOOD_MARGIN = 3.0
MIN_FALLBACK_ACTIONS_AFTER_CONFLICT = 1

MIN_HEADROOM_GAIN = parent.MIN_HEADROOM_GAIN
MIN_CAUSAL_RETENTION = parent.MIN_CAUSAL_RETENTION
MIN_ROUTER_GAIN = parent.MIN_POSTERIOR_GAIN
REQUIRED_SEED_PASSES = parent.REQUIRED_SEED_PASSES

AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_conflict_fallback_router_audit_v14"
)
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_conflict_fallback_router_analysis_v14"
)
REGISTRATION_ROOT = (
    ROOT / "jax_experiments" / "deployments"
    / "regime_polarity_conflict_fallback_router_v14"
)
REGISTRATION_PATH = REGISTRATION_ROOT / "registration.json"
PREREG_REPORT = (
    ROOT / "reports"
    / "regime_polarity_conflict_fallback_router_v14_preregistration_2026-09-01.md"
)
REPORT = (
    ROOT / "reports"
    / "regime_polarity_conflict_fallback_router_v14_2026-09-01.md"
)

EVENT_SCHEMA = "bapr.evidence-conflict-fallback-event.v14"
AUDIT_SCHEMA = "bapr.evidence-conflict-fallback-audit.v14"
ANALYSIS_SCHEMA = "bapr.evidence-conflict-fallback-analysis.v14"
REGISTRATION_SCHEMA = "bapr.evidence-conflict-fallback-registration.v14"

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
        raise ValueError(f"unknown v14 event seed {seed}")
    return seed


def require_arm(arm: str) -> str:
    arm = str(arm)
    if arm not in ARMS:
        raise ValueError(f"unknown v14 arm {arm!r}")
    return arm


def confirm_steps_for_arm(arm: str) -> int:
    arm = require_arm(arm)
    if arm not in CANDIDATE_ARMS:
        return 0
    middle = arm.removeprefix("conflict_fallback_confirm")
    return int(middle.removesuffix("_safe_utility"))


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
        ROOT / "jax_experiments/analysis/regime_polarity_conflict_fallback_router_v14.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_conflict_fallback_router_audit_v14.py",
        ROOT / "jax_experiments/analysis/analyze_regime_polarity_conflict_fallback_router_v14.py",
        parent.REGISTRATION_PATH,
        parent.analysis_json(),
        parent.analysis_markdown(),
        PREREG_REPORT,
    )
    return tuple(path.resolve() for path in paths)


def registration_payload() -> dict[str, Any]:
    parent.validate_registration()
    parent_analysis = read_json(parent.analysis_json())
    if (
        parent_analysis.get("schema") != parent.ANALYSIS_SCHEMA
        or parent_analysis.get("status") != "complete"
        or parent_analysis.get("causal_margin_reproducible") is not True
        or parent_analysis.get("estimator_retraining_authorized") is not True
    ):
        raise ValueError("v13 did not authorize estimator/router development")
    paths = registration_source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"v14 registration sources missing: {missing}")
    return {
        "schema": REGISTRATION_SCHEMA,
        "status": "registered",
        "created_before_audit": True,
        "identity": {
            "protocol_version": PROTOCOL_VERSION,
            "parent_protocol_version": parent.PROTOCOL_VERSION,
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
            "trigger": (
                "best alternative one-step evidence exceeds active-mode "
                "log likelihood by at least 3.0"
            ),
            "conflict_log_likelihood_margin": (
                CONFLICT_LOG_LIKELIHOOD_MARGIN),
            "fallback_controller": "matched robust SAC",
            "minimum_fallback_actions_after_conflict": (
                MIN_FALLBACK_ACTIONS_AFTER_CONFLICT),
            "exit_requires_evidence_posterior_agreement": True,
            "posterior_exit_confidence": POSTERIOR_EXIT_CONFIDENCE,
            "confirmation_steps": [1, 2, 3],
            "candidate_online_inputs": [
                "observation", "commanded_action", "next_observation",
                "frozen estimator posterior", "frozen estimator evidence",
            ],
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
            "require_zero_termination": True,
            "selection_order": [
                "seed_pass_count", "mean_return", "lower_fallback_fraction",
                "fewer_confirmation_steps",
            ],
        },
        "data_policy": {
            "checkpoint_only": True,
            "new_training": False,
            "result_payload": "json_and_markdown_only",
            "development_not_confirmation": True,
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
            raise ValueError("existing v14 registration changed")
    else:
        write_json_atomic(REGISTRATION_PATH, payload)
    return validate_registration()


def validate_registration() -> dict[str, Any]:
    if not REGISTRATION_PATH.is_file():
        raise FileNotFoundError(f"missing v14 registration: {REGISTRATION_PATH}")
    payload = read_json(REGISTRATION_PATH)
    if payload != registration_payload():
        raise ValueError("v14 registration or source closure changed")
    return payload


def assert_protocol_integrity() -> None:
    prior_events = (
        set(parent.EVENT_SEEDS)
        | set(policy_parent.CALIBRATION_EVENT_SEEDS)
        | set(policy_parent.STATIONARY_HOLDOUT_EVENT_SEEDS)
        | set(policy_parent.SWITCHING_EVENT_SEEDS)
        | set(estimator.TRAIN_EVENT_SEEDS)
        | set(estimator.VALIDATION_EVENT_SEEDS)
        | set(estimator.AUDIT_EVENT_SEEDS)
    )
    if prior_events & set(EVENT_SEEDS):
        raise ValueError("v14 reused an earlier event seed")
    if set(SWITCHING_SCHEDULES) != set(EVENT_SEEDS):
        raise ValueError("v14 switching schedules are incomplete")
    cycles = []
    for sequence in SWITCHING_SCHEDULES.values():
        if len(sequence) != len(MODES) or set(sequence) != set(MODES):
            raise ValueError("each v14 schedule must use every mode once")
        cycles.append(tuple(sequence))
    if len(set(cycles)) != len(cycles):
        raise ValueError("v14 switching schedules are not distinct")


assert_protocol_integrity()
