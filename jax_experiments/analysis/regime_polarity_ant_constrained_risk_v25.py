"""Registered Ant constrained termination-risk development protocol."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_ant_switch_recovery_v24 as parent,
)


ROOT = parent.ROOT
PROTOCOL_VERSION = "v25-ant-constrained-risk-development"
ENV = parent.ENV
FAMILY = parent.FAMILY
MODES = parent.MODES
VARIANTS = ("risk_q_absolute", "risk_q_relative")
TRAINING_SEEDS = parent.TRAINING_SEEDS

CALIBRATION_EVENT_SEEDS = (190_001, 190_017, 190_033)
STATIONARY_HOLDOUT_EVENT_SEEDS = (190_101, 190_117, 190_133)
SWITCHING_EVENT_SEEDS = (190_201, 190_217, 190_233)
SWITCHING_SCHEDULES = {
    190_201: (1, 3, 0, 2),
    190_217: (2, 1, 3, 0),
    190_233: (0, 2, 1, 3),
}

SOURCE_NEXT_ITERATION = parent.SOURCE_NEXT_ITERATION
SOURCE_ITERATION = parent.SOURCE_ITERATION
SOURCE_TOTAL_STEPS = parent.SOURCE_TOTAL_STEPS
SOURCE_UPDATE_COUNT = parent.SOURCE_UPDATE_COUNT
FINETUNE_ITERS = parent.FINETUNE_ITERS
FINAL_NEXT_ITERATION = parent.FINAL_NEXT_ITERATION
FINAL_ITERATION = parent.FINAL_ITERATION
PHYSICAL_SAMPLES_PER_ITER = parent.PHYSICAL_SAMPLES_PER_ITER
CANDIDATE_SAMPLES_PER_ITER = parent.CANDIDATE_SAMPLES_PER_ITER
TARGET_MODE_SAMPLES_PER_ITER = CANDIDATE_SAMPLES_PER_ITER
UPDATES_PER_ITER = parent.UPDATES_PER_ITER
FINAL_TOTAL_STEPS = parent.FINAL_TOTAL_STEPS
FINAL_UPDATE_COUNT = parent.FINAL_UPDATE_COUNT

DWELL_STEPS = parent.DWELL_STEPS
SWITCH_SEGMENT_STEPS = parent.SWITCH_SEGMENT_STEPS
MAX_EPISODE_STEPS = parent.MAX_EPISODE_STEPS
EPISODES_PER_TASK = parent.EPISODES_PER_TASK
SWITCHING_EPISODES = parent.SWITCHING_EPISODES
TRANSIENT_FALLBACK_STEPS = parent.TRANSIENT_FALLBACK_STEPS
RISK_LAMBDA = 500.0
RISK_WARMUP_UPDATES = 12_500
RISK_ACTOR_START_UPDATE = SOURCE_UPDATE_COUNT + RISK_WARMUP_UPDATES
RISK_OBJECTIVE = {
    "risk_q_absolute": "absolute",
    "risk_q_relative": "relative",
}
TERMINATION_PENALTY = {variant: 0.0 for variant in VARIANTS}

MIN_CALIBRATION_GAIN = parent.MIN_CALIBRATION_GAIN
MIN_HOLDOUT_MODE_GAIN = parent.MIN_HOLDOUT_MODE_GAIN
MIN_HOLDOUT_MODE_WINS = parent.MIN_HOLDOUT_MODE_WINS
MIN_SWITCHING_GAIN = parent.MIN_SWITCHING_GAIN
REQUIRED_CELL_PASSES = len(TRAINING_SEEDS)

RUN_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_ant_constrained_risk_v25"
)
SPECIALIST_RUN_ROOT = RUN_ROOT / "specialists"
SPECIALIST_BUNDLE_ROOT = (
    ROOT / "jax_experiments"
    / "eval_bundles_regime_polarity_ant_constrained_risk_v25"
)
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_ant_constrained_risk_audit_v25"
)
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_ant_constrained_risk_analysis_v25"
)
REGISTRATION_ROOT = (
    ROOT / "jax_experiments" / "deployments"
    / "regime_polarity_ant_constrained_risk_v25"
)
REGISTRATION_PATH = REGISTRATION_ROOT / "registration.json"
PRIOR_ANALYSIS_PATH = parent.analysis_json()
REPORT = (
    ROOT / "reports"
    / "regime_polarity_ant_constrained_risk_v25_2026-09-11.md"
)

POLICY_NAME = parent.POLICY_NAME
CONTROLLER_STATE_NAME = parent.CONTROLLER_STATE_NAME
BOOTSTRAP_NAME = parent.BOOTSTRAP_NAME
RUNTIME_NAME = "constrained_risk_runtime.json"
SOURCE_BUNDLE_SCHEMA = parent.SOURCE_BUNDLE_SCHEMA
BUNDLE_SCHEMA = "bapr.ant-constrained-risk-bundle.v25"
BOOTSTRAP_SCHEMA = "bapr.ant-constrained-risk-bootstrap.v25"
AUDIT_SCHEMA = "bapr.ant-constrained-risk-audit.v25"
ANALYSIS_SCHEMA = "bapr.ant-constrained-risk-analysis.v25"
REGISTRATION_SCHEMA = "bapr.ant-constrained-risk-registration.v25"

file_record = parent.file_record
read_json = parent.read_json
write_json_atomic = parent.write_json_atomic
write_text_atomic = parent.write_text_atomic
checkpoint_record = parent.checkpoint_record


def require_variant(variant: str) -> str:
    value = str(variant)
    if value not in VARIANTS:
        raise ValueError(f"unknown Ant V25 variant {value!r}")
    return value


def require_training_seed(seed: int) -> int:
    return parent.require_training_seed(seed)


require_seed = require_training_seed


def require_mode(mode: int) -> int:
    return parent.require_mode(mode)


def require_switching_event_seed(seed: int) -> int:
    value = int(seed)
    if value not in SWITCHING_EVENT_SEEDS:
        raise ValueError(f"unknown Ant V25 switching event seed {value}")
    return value


def switching_sequence(event_seed: int, episode: int) -> tuple[int, ...]:
    base = SWITCHING_SCHEDULES[require_switching_event_seed(event_seed)]
    shift = int(episode) % len(base)
    return tuple(base[shift:] + base[:shift])


def termination_penalty(variant: str) -> float:
    return TERMINATION_PENALTY[require_variant(variant)]


def risk_objective(variant: str) -> str:
    return RISK_OBJECTIVE[require_variant(variant)]


def source_bundle(seed: int) -> Path:
    return parent.source_bundle(require_training_seed(seed))


def source_manifest(seed: int) -> Path:
    return parent.source_manifest(require_training_seed(seed))


def source_required_paths(seed: int) -> tuple[Path, ...]:
    return parent.source_required_paths(require_training_seed(seed))


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
        directory / "provenance" / RUNTIME_NAME,
    )


def audit_dir(variant: str, seed: int) -> Path:
    return AUDIT_ROOT / require_variant(variant) / f"seed_{require_seed(seed)}"


def audit_result(variant: str, seed: int) -> Path:
    return audit_dir(variant, seed) / "audit.json"


def audit_manifest(variant: str, seed: int) -> Path:
    return audit_dir(variant, seed) / "audit_manifest.json"


def analysis_json() -> Path:
    return ANALYSIS_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return ANALYSIS_ROOT / "analysis.md"


def identity(variant: str, seed: int, mode: int) -> dict[str, Any]:
    variant = require_variant(variant)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "ant_constrained_risk_specialist",
        "variant": variant,
        "env": ENV,
        "family": FAMILY,
        "training_seed": require_training_seed(seed),
        "target_mode": require_mode(mode),
        "algo": "sac",
        "source_next_iteration": SOURCE_NEXT_ITERATION,
        "final_next_iteration": FINAL_NEXT_ITERATION,
        "final_total_physical_steps": FINAL_TOTAL_STEPS,
        "final_update_count": FINAL_UPDATE_COUNT,
        "physical_samples_per_iter": PHYSICAL_SAMPLES_PER_ITER,
        "candidate_samples_per_iter": CANDIDATE_SAMPLES_PER_ITER,
        "target_behavior_candidate_fraction": 0.5,
        "segment_steps": SWITCH_SEGMENT_STEPS,
        "termination_penalty": 0.0,
        "risk_objective": risk_objective(variant),
        "risk_lambda": RISK_LAMBDA,
        "risk_actor_start_update": RISK_ACTOR_START_UPDATE,
    }


def expected_source_checkpoint() -> dict[str, Any]:
    return parent.expected_source_checkpoint()


def expected_checkpoint() -> dict[str, Any]:
    return parent.expected_checkpoint()


def bundle_records(variant: str, seed: int) -> dict[str, Any]:
    return {
        str(mode): file_record(bundle_manifest(variant, seed, mode))
        for mode in MODES
    }


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def registration_source_paths() -> tuple[Path, ...]:
    paths = [
        Path(__file__).resolve(),
        ROOT / "jax_experiments/algos/sac_switch_recovery.py",
        ROOT / "jax_experiments/algos/sac_switch_recovery_risk.py",
        ROOT / "jax_experiments/analysis/train_regime_polarity_ant_switch_recovery_v24.py",
        ROOT / "jax_experiments/analysis/train_regime_polarity_ant_constrained_risk_v25.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_ant_switch_recovery_v24.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_ant_constrained_risk_v25.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_ant_switch_recovery_audit_v24.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_ant_constrained_risk_audit_v25.py",
        ROOT / "jax_experiments/analysis/analyze_regime_polarity_ant_constrained_risk_v25.py",
        ROOT / "scripts/submit_regime_polarity_ant_constrained_risk_v25.py",
        ROOT / "jax_experiments/algos/sac_base.py",
        ROOT / "jax_experiments/common/checkpoint.py",
        ROOT / "jax_experiments/common/replay_buffer.py",
        ROOT / "jax_experiments/envs/stochastic_mode_env.py",
        ROOT / "jax_experiments/train.py",
        parent.REGISTRATION_PATH,
        PRIOR_ANALYSIS_PATH,
    ]
    paths.extend(source_manifest(seed) for seed in TRAINING_SEEDS)
    return tuple(dict.fromkeys(path.resolve() for path in paths))


def registration_payload() -> dict[str, Any]:
    parent.validate_registration()
    prior = read_json(PRIOR_ANALYSIS_PATH)
    if (
        prior.get("gate_pass") is not False
        or prior.get("diagnosis")
        != "termination_risk_helps_but_ant_bank_remains_unstable"
    ):
        raise ValueError("Ant V24 does not authorize V25")
    paths = registration_source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Ant V25 registration sources missing: {missing}")
    return {
        "schema": REGISTRATION_SCHEMA,
        "status": "registered",
        "created_before_training": True,
        "identity": {
            "protocol_version": PROTOCOL_VERSION,
            "environment": ENV,
            "family": FAMILY,
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
        "question": (
            "whether a learned discounted termination-risk critic makes the "
            "V24 Ant switch-state specialists reliable, and whether risk "
            "relative to the immutable robust action preserves more utility "
            "than an absolute risk penalty"
        ),
        "training": {
            "physical_samples_per_iter": PHYSICAL_SAMPLES_PER_ITER,
            "candidate_samples_per_iter": CANDIDATE_SAMPLES_PER_ITER,
            "target_mode_samples_per_iter": TARGET_MODE_SAMPLES_PER_ITER,
            "target_behavior_mix": {
                "candidate_policy": 0.5,
                "frozen_robust_policy": 0.5,
            },
            "updates_per_iter": UPDATES_PER_ITER,
            "segment_steps": SWITCH_SEGMENT_STEPS,
            "risk_lambda": RISK_LAMBDA,
            "risk_warmup_updates": RISK_WARMUP_UPDATES,
            "risk_objectives": dict(RISK_OBJECTIVE),
            "risk_target": "discounted termination probability",
            "termination_reward_shaping": False,
            "controller_initialization": [
                "actor", "critic", "target_critic", "alpha"
            ],
            "risk_critic_initialization": "fresh",
            "replay_and_optimizers_reset": True,
            "frozen_robust_prefix_actor": True,
        },
        "evaluation": {
            "immutable_robust_fallback": True,
            "transient_fallback_steps": TRANSIENT_FALLBACK_STEPS,
            "utility_fallback_calibrated_on_disjoint_events": True,
            "risk_critic_not_used_for_routing": True,
        },
        "frozen_boundary": {
            "v22_through_v24_artifacts_unchanged": True,
            "robust_sources_reused": True,
            "all_four_specialists_retrained": True,
            "posterior_or_router_training": False,
            "same_v24_development_policy_seeds": True,
            "new_disjoint_evaluation_streams": True,
        },
        "gate": {
            "all_three_policy_seeds_pass": True,
            "minimum_stationary_mode_wins_per_seed": MIN_HOLDOUT_MODE_WINS,
            "minimum_per_mode_gain": MIN_HOLDOUT_MODE_GAIN,
            "minimum_switching_gain": MIN_SWITCHING_GAIN,
            "all_three_switching_events_win": True,
            "zero_transient_fallback_termination": True,
            "selection": (
                "highest mean transient-fallback switching return among "
                "passing variants; registered order breaks exact ties"
            ),
        },
        "stopping_rule": (
            "if neither learned-risk arm passes all three development seeds, "
            "close independent Ant specialist optimization and move to a "
            "joint robust-plus-mode controller"
        ),
        "sync_policy": (
            "compact policy/provenance bundles and JSON only; never sync "
            "replay or full checkpoints"
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
            raise ValueError("existing Ant V25 registration changed")
    else:
        write_json_atomic(REGISTRATION_PATH, payload)
    return validate_registration()


def validate_registration() -> dict[str, Any]:
    if not REGISTRATION_PATH.is_file():
        raise FileNotFoundError(f"missing Ant V25 registration: {REGISTRATION_PATH}")
    payload = read_json(REGISTRATION_PATH)
    if payload != registration_payload():
        raise ValueError("Ant V25 registration or source closure changed")
    return payload


def assert_protocol_integrity() -> None:
    if PHYSICAL_SAMPLES_PER_ITER != 2 * CANDIDATE_SAMPLES_PER_ITER:
        raise ValueError("V25 physical/candidate accounting changed")
    if CANDIDATE_SAMPLES_PER_ITER % SWITCH_SEGMENT_STEPS:
        raise ValueError("V25 candidate samples do not contain full segments")
    if set(RISK_OBJECTIVE) != set(VARIANTS):
        raise ValueError("V25 risk objectives do not cover every variant")
    if RISK_ACTOR_START_UPDATE >= FINAL_UPDATE_COUNT:
        raise ValueError("V25 risk warmup consumes the whole actor budget")
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
        raise ValueError("Ant V25 evaluation splits overlap")
    if set(SWITCHING_SCHEDULES) != set(SWITCHING_EVENT_SEEDS):
        raise ValueError("Ant V25 switching schedules are incomplete")
    if any(
        len(sequence) != len(MODES) or set(sequence) != set(MODES)
        for sequence in SWITCHING_SCHEDULES.values()
    ):
        raise ValueError("each Ant V25 schedule must use every mode")


assert_protocol_integrity()
