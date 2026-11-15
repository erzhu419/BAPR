"""Registered Ant switch-state specialist development protocol."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_ant_full_state_headroom_v22 as parent,
)


ROOT = parent.ROOT
PROTOCOL_VERSION = "v24-ant-switch-recovery-development"
ENV = parent.ENV
FAMILY = parent.FAMILY
MODES = parent.MODES
VARIANTS = ("switch_state", "switch_state_risk")
TRAINING_SEEDS = parent.TRAINING_SEEDS

CALIBRATION_EVENT_SEEDS = (189_001, 189_017, 189_033)
STATIONARY_HOLDOUT_EVENT_SEEDS = (189_101, 189_117, 189_133)
SWITCHING_EVENT_SEEDS = (189_201, 189_217, 189_233)
SWITCHING_SCHEDULES = {
    189_201: (0, 2, 3, 1),
    189_217: (3, 1, 0, 2),
    189_233: (2, 0, 1, 3),
}

SOURCE_NEXT_ITERATION = parent.SOURCE_NEXT_ITERATION
SOURCE_ITERATION = parent.SOURCE_ITERATION
SOURCE_TOTAL_STEPS = parent.SOURCE_TOTAL_STEPS
SOURCE_UPDATE_COUNT = parent.SOURCE_UPDATE_COUNT
FINETUNE_ITERS = parent.FINETUNE_ITERS
FINAL_NEXT_ITERATION = SOURCE_NEXT_ITERATION + FINETUNE_ITERS
FINAL_ITERATION = FINAL_NEXT_ITERATION - 1
PHYSICAL_SAMPLES_PER_ITER = 8_000
CANDIDATE_SAMPLES_PER_ITER = 4_000
UPDATES_PER_ITER = parent.UPDATES_PER_ITER
FINAL_TOTAL_STEPS = (
    SOURCE_TOTAL_STEPS + FINETUNE_ITERS * PHYSICAL_SAMPLES_PER_ITER
)
FINAL_UPDATE_COUNT = SOURCE_UPDATE_COUNT + FINETUNE_ITERS * UPDATES_PER_ITER

DWELL_STEPS = parent.DWELL_STEPS
SWITCH_SEGMENT_STEPS = DWELL_STEPS
MAX_EPISODE_STEPS = parent.MAX_EPISODE_STEPS
EPISODES_PER_TASK = parent.EPISODES_PER_TASK
SWITCHING_EPISODES = parent.SWITCHING_EPISODES
TRANSIENT_FALLBACK_STEPS = 8
TERMINATION_PENALTY = {
    "switch_state": 0.0,
    "switch_state_risk": 500.0,
}

MIN_CALIBRATION_GAIN = parent.MIN_CALIBRATION_GAIN
MIN_HOLDOUT_MODE_GAIN = parent.MIN_HOLDOUT_MODE_GAIN
MIN_HOLDOUT_MODE_WINS = parent.MIN_HOLDOUT_MODE_WINS
MIN_SWITCHING_GAIN = parent.MIN_SWITCHING_GAIN
REQUIRED_CELL_PASSES = len(TRAINING_SEEDS)

RUN_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_ant_switch_recovery_v24"
)
SPECIALIST_RUN_ROOT = RUN_ROOT / "specialists"
SPECIALIST_BUNDLE_ROOT = (
    ROOT / "jax_experiments"
    / "eval_bundles_regime_polarity_ant_switch_recovery_v24"
)
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_ant_switch_recovery_audit_v24"
)
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_ant_switch_recovery_analysis_v24"
)
REGISTRATION_ROOT = (
    ROOT / "jax_experiments" / "deployments"
    / "regime_polarity_ant_switch_recovery_v24"
)
REGISTRATION_PATH = REGISTRATION_ROOT / "registration.json"
PRIOR_ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_ant_matching_checkpoint_analysis_v23"
)
PRIOR_ANALYSIS_PATH = PRIOR_ANALYSIS_ROOT / "analysis.json"
REPORT = (
    ROOT / "reports"
    / "regime_polarity_ant_switch_recovery_v24_2026-09-11.md"
)

POLICY_NAME = parent.POLICY_NAME
CONTROLLER_STATE_NAME = parent.CONTROLLER_STATE_NAME
BOOTSTRAP_NAME = parent.BOOTSTRAP_NAME
RUNTIME_NAME = "switch_recovery_runtime.json"
SOURCE_BUNDLE_SCHEMA = parent.SOURCE_BUNDLE_SCHEMA
BUNDLE_SCHEMA = "bapr.ant-switch-recovery-bundle.v24"
BOOTSTRAP_SCHEMA = "bapr.ant-switch-recovery-bootstrap.v24"
AUDIT_SCHEMA = "bapr.ant-switch-recovery-audit.v24"
ANALYSIS_SCHEMA = "bapr.ant-switch-recovery-analysis.v24"
REGISTRATION_SCHEMA = "bapr.ant-switch-recovery-registration.v24"

file_record = parent.file_record
read_json = parent.read_json
write_json_atomic = parent.write_json_atomic
write_text_atomic = parent.write_text_atomic
checkpoint_record = parent.checkpoint_record


def require_variant(variant: str) -> str:
    value = str(variant)
    if value not in VARIANTS:
        raise ValueError(f"unknown Ant V24 variant {value!r}")
    return value


def require_training_seed(seed: int) -> int:
    return parent.require_training_seed(seed)


require_seed = require_training_seed


def require_mode(mode: int) -> int:
    return parent.require_mode(mode)


def require_switching_event_seed(seed: int) -> int:
    value = int(seed)
    if value not in SWITCHING_EVENT_SEEDS:
        raise ValueError(f"unknown Ant V24 switching event seed {value}")
    return value


def switching_sequence(event_seed: int, episode: int) -> tuple[int, ...]:
    base = SWITCHING_SCHEDULES[require_switching_event_seed(event_seed)]
    shift = int(episode) % len(base)
    return tuple(base[shift:] + base[:shift])


def termination_penalty(variant: str) -> float:
    return float(TERMINATION_PENALTY[require_variant(variant)])


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
        "benchmark_role": "ant_switch_recovery_specialist",
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
        "segment_steps": SWITCH_SEGMENT_STEPS,
        "termination_penalty": termination_penalty(variant),
    }


def expected_source_checkpoint() -> dict[str, Any]:
    return parent.expected_source_checkpoint()


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
    paths = [
        Path(__file__).resolve(),
        ROOT / "jax_experiments/algos/sac_switch_recovery.py",
        ROOT / "jax_experiments/analysis/train_regime_polarity_ant_switch_recovery_v24.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_ant_switch_recovery_v24.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_ant_switch_recovery_audit_v24.py",
        ROOT / "jax_experiments/analysis/analyze_regime_polarity_ant_switch_recovery_v24.py",
        ROOT / "scripts/submit_regime_polarity_ant_switch_recovery_v24.py",
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
        != "checkpoint_selection_does_not_stabilize_ant_policy_bank"
    ):
        raise ValueError("Ant V23 does not authorize V24")
    paths = registration_source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Ant V24 registration sources missing: {missing}")
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
            "whether matching the specialist rollout distribution to Ant "
            "switch states, with an explicit terminal-risk target, removes "
            "the V23 controller failures"
        ),
        "training": {
            "physical_samples_per_iter": PHYSICAL_SAMPLES_PER_ITER,
            "candidate_samples_per_iter": CANDIDATE_SAMPLES_PER_ITER,
            "robust_prefix_samples_per_iter": (
                PHYSICAL_SAMPLES_PER_ITER - CANDIDATE_SAMPLES_PER_ITER),
            "updates_per_iter": UPDATES_PER_ITER,
            "segment_steps": SWITCH_SEGMENT_STEPS,
            "controller_initialization": [
                "actor", "critic", "target_critic", "alpha"
            ],
            "replay_and_optimizers_reset": True,
            "frozen_robust_prefix_actor": True,
            "termination_penalties": dict(TERMINATION_PENALTY),
        },
        "evaluation": {
            "immutable_robust_fallback": True,
            "transient_fallback_steps": TRANSIENT_FALLBACK_STEPS,
            "utility_fallback_calibrated_on_disjoint_events": True,
        },
        "frozen_boundary": {
            "v22_and_v23_artifacts_unchanged": True,
            "robust_sources_reused": True,
            "all_four_specialists_retrained": True,
            "checkpoint_selection_disabled": True,
            "actor_period_tuning_closed": True,
            "estimator_or_router_training": False,
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
        "accounting": (
            "physical interaction includes frozen robust prefixes; only "
            "post-switch target-mode transitions enter specialist replay"
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
            raise ValueError("existing Ant V24 registration changed")
    else:
        write_json_atomic(REGISTRATION_PATH, payload)
    return validate_registration()


def validate_registration() -> dict[str, Any]:
    if not REGISTRATION_PATH.is_file():
        raise FileNotFoundError(f"missing Ant V24 registration: {REGISTRATION_PATH}")
    payload = read_json(REGISTRATION_PATH)
    if payload != registration_payload():
        raise ValueError("Ant V24 registration or source closure changed")
    return payload


def assert_protocol_integrity() -> None:
    if PHYSICAL_SAMPLES_PER_ITER != 2 * CANDIDATE_SAMPLES_PER_ITER:
        raise ValueError("V24 physical/candidate accounting changed")
    if CANDIDATE_SAMPLES_PER_ITER % SWITCH_SEGMENT_STEPS:
        raise ValueError("V24 candidate samples do not contain full segments")
    if set(TERMINATION_PENALTY) != set(VARIANTS):
        raise ValueError("V24 penalties do not cover every variant")
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
        raise ValueError("Ant V24 evaluation splits overlap")
    if set(SWITCHING_SCHEDULES) != set(SWITCHING_EVENT_SEEDS):
        raise ValueError("Ant V24 switching schedules are incomplete")
    if any(
        len(sequence) != len(MODES) or set(sequence) != set(MODES)
        for sequence in SWITCHING_SCHEDULES.values()
    ):
        raise ValueError("each Ant V24 schedule must use every mode")


assert_protocol_integrity()
