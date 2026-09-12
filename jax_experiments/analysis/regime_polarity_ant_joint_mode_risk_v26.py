"""Registered Ant joint mode-conditioned risk-controller protocol."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_ant_constrained_risk_v25 as parent,
)


ROOT = parent.ROOT
PROTOCOL_VERSION = "v26-ant-joint-mode-risk-development"
ENV = parent.ENV
FAMILY = parent.FAMILY
MODES = parent.MODES
VARIANTS = ("joint_equal_budget", "joint_data_matched")
TRAINING_SEEDS = parent.TRAINING_SEEDS

CALIBRATION_EVENT_SEEDS = (191_001, 191_017, 191_033)
STATIONARY_HOLDOUT_EVENT_SEEDS = (191_101, 191_117, 191_133)
SWITCHING_EVENT_SEEDS = (191_201, 191_217, 191_233)
SWITCHING_SCHEDULES = {
    191_201: (3, 1, 0, 2),
    191_217: (1, 2, 3, 0),
    191_233: (2, 0, 1, 3),
}

SOURCE_NEXT_ITERATION = parent.SOURCE_NEXT_ITERATION
SOURCE_ITERATION = parent.SOURCE_ITERATION
SOURCE_TOTAL_STEPS = parent.SOURCE_TOTAL_STEPS
SOURCE_UPDATE_COUNT = parent.SOURCE_UPDATE_COUNT
PHYSICAL_SAMPLES_PER_ITER = parent.PHYSICAL_SAMPLES_PER_ITER
TARGET_SAMPLES_PER_ITER = parent.CANDIDATE_SAMPLES_PER_ITER
UPDATES_PER_ITER = parent.UPDATES_PER_ITER
FINETUNE_ITERS = {
    "joint_equal_budget": parent.FINETUNE_ITERS,
    "joint_data_matched": 4 * parent.FINETUNE_ITERS,
}
FINAL_NEXT_ITERATION = {
    variant: SOURCE_NEXT_ITERATION + FINETUNE_ITERS[variant]
    for variant in VARIANTS
}
FINAL_ITERATION = {
    variant: FINAL_NEXT_ITERATION[variant] - 1
    for variant in VARIANTS
}
FINAL_TOTAL_STEPS = {
    variant: (
        SOURCE_TOTAL_STEPS
        + FINETUNE_ITERS[variant] * PHYSICAL_SAMPLES_PER_ITER
    )
    for variant in VARIANTS
}
FINAL_UPDATE_COUNT = {
    variant: (
        SOURCE_UPDATE_COUNT + FINETUNE_ITERS[variant] * UPDATES_PER_ITER
    )
    for variant in VARIANTS
}

DWELL_STEPS = parent.DWELL_STEPS
SWITCH_SEGMENT_STEPS = parent.SWITCH_SEGMENT_STEPS
MAX_EPISODE_STEPS = parent.MAX_EPISODE_STEPS
EPISODES_PER_TASK = parent.EPISODES_PER_TASK
SWITCHING_EPISODES = parent.SWITCHING_EPISODES
TRANSIENT_FALLBACK_STEPS = parent.TRANSIENT_FALLBACK_STEPS
RISK_LAMBDA = parent.RISK_LAMBDA
RISK_WARMUP_UPDATES = parent.RISK_WARMUP_UPDATES
RISK_ACTOR_START_UPDATE = parent.RISK_ACTOR_START_UPDATE
CONTROLLER_EQUIVALENCE_ATOL = 1e-4

MIN_CALIBRATION_GAIN = parent.MIN_CALIBRATION_GAIN
MIN_HOLDOUT_MODE_GAIN = parent.MIN_HOLDOUT_MODE_GAIN
MIN_HOLDOUT_MODE_WINS = parent.MIN_HOLDOUT_MODE_WINS
MIN_SWITCHING_GAIN = parent.MIN_SWITCHING_GAIN
REQUIRED_CELL_PASSES = len(TRAINING_SEEDS)

RUN_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_ant_joint_mode_risk_v26"
)
BUNDLE_ROOT = (
    ROOT / "jax_experiments"
    / "eval_bundles_regime_polarity_ant_joint_mode_risk_v26"
)
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_ant_joint_mode_risk_audit_v26"
)
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_ant_joint_mode_risk_analysis_v26"
)
REGISTRATION_ROOT = (
    ROOT / "jax_experiments" / "deployments"
    / "regime_polarity_ant_joint_mode_risk_v26"
)
REGISTRATION_PATH = REGISTRATION_ROOT / "registration.json"
PRIOR_ANALYSIS_PATH = parent.analysis_json()
REPORT = (
    ROOT / "reports"
    / "regime_polarity_ant_joint_mode_risk_v26_2026-09-11.md"
)

POLICY_NAME = parent.POLICY_NAME
BOOTSTRAP_NAME = parent.BOOTSTRAP_NAME
RUNTIME_NAME = "joint_mode_risk_runtime.json"
BUNDLE_SCHEMA = "bapr.ant-joint-mode-risk-bundle.v26"
BOOTSTRAP_SCHEMA = "bapr.ant-joint-mode-risk-bootstrap.v26"
AUDIT_SCHEMA = "bapr.ant-joint-mode-risk-audit.v26"
ANALYSIS_SCHEMA = "bapr.ant-joint-mode-risk-analysis.v26"
REGISTRATION_SCHEMA = "bapr.ant-joint-mode-risk-registration.v26"

file_record = parent.file_record
read_json = parent.read_json
write_json_atomic = parent.write_json_atomic
write_text_atomic = parent.write_text_atomic
checkpoint_record = parent.checkpoint_record


def require_variant(variant: str) -> str:
    value = str(variant)
    if value not in VARIANTS:
        raise ValueError(f"unknown Ant V26 variant {value!r}")
    return value


def require_training_seed(seed: int) -> int:
    return parent.require_training_seed(seed)


require_seed = require_training_seed


def require_mode(mode: int) -> int:
    return parent.require_mode(mode)


def require_switching_event_seed(seed: int) -> int:
    value = int(seed)
    if value not in SWITCHING_EVENT_SEEDS:
        raise ValueError(f"unknown Ant V26 switching event seed {value}")
    return value


def switching_sequence(event_seed: int, episode: int) -> tuple[int, ...]:
    base = SWITCHING_SCHEDULES[require_switching_event_seed(event_seed)]
    shift = int(episode) % len(base)
    return tuple(base[shift:] + base[:shift])


def source_bundle(seed: int) -> Path:
    return parent.source_bundle(require_training_seed(seed))


def source_manifest(seed: int) -> Path:
    return parent.source_manifest(require_training_seed(seed))


def source_required_paths(seed: int) -> tuple[Path, ...]:
    return parent.source_required_paths(require_training_seed(seed))


def run_dir(variant: str, seed: int) -> Path:
    return (
        RUN_ROOT / require_variant(variant)
        / f"seed_{require_training_seed(seed)}"
    )


def bundle_dir(variant: str, seed: int) -> Path:
    return (
        BUNDLE_ROOT / require_variant(variant)
        / f"seed_{require_training_seed(seed)}"
    )


def bundle_manifest(variant: str, seed: int) -> Path:
    return bundle_dir(variant, seed) / "bundle_manifest.json"


def bundle_required_paths(variant: str, seed: int) -> tuple[Path, ...]:
    directory = bundle_dir(variant, seed)
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


def expected_source_checkpoint() -> dict[str, Any]:
    return parent.expected_source_checkpoint()


def expected_checkpoint(variant: str) -> dict[str, Any]:
    variant = require_variant(variant)
    return {
        "iteration": FINAL_ITERATION[variant],
        "next_iteration": FINAL_NEXT_ITERATION[variant],
        "total_steps": FINAL_TOTAL_STEPS[variant],
        "update_count": FINAL_UPDATE_COUNT[variant],
        "algo": "sac",
    }


def identity(variant: str, seed: int) -> dict[str, Any]:
    variant = require_variant(variant)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "ant_joint_mode_risk_controller",
        "variant": variant,
        "env": ENV,
        "family": FAMILY,
        "training_seed": require_training_seed(seed),
        "algo": "sac",
        "source_next_iteration": SOURCE_NEXT_ITERATION,
        "final_next_iteration": FINAL_NEXT_ITERATION[variant],
        "final_total_physical_steps": FINAL_TOTAL_STEPS[variant],
        "final_update_count": FINAL_UPDATE_COUNT[variant],
        "physical_samples_per_iter": PHYSICAL_SAMPLES_PER_ITER,
        "target_samples_per_iter": TARGET_SAMPLES_PER_ITER,
        "per_mode_target_samples_per_iter": (
            TARGET_SAMPLES_PER_ITER // len(MODES)
        ),
        "target_behavior_candidate_fraction": 0.5,
        "segment_steps": SWITCH_SEGMENT_STEPS,
        "risk_objective": "relative",
        "risk_lambda": RISK_LAMBDA,
        "risk_actor_start_update": RISK_ACTOR_START_UPDATE,
    }


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def registration_source_paths() -> tuple[Path, ...]:
    paths = [
        Path(__file__).resolve(),
        ROOT / "jax_experiments/algos/joint_mode_risk_sac.py",
        ROOT / "jax_experiments/analysis/train_regime_polarity_ant_joint_mode_risk_v26.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_ant_joint_mode_risk_v26.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_ant_joint_mode_risk_audit_v26.py",
        ROOT / "jax_experiments/analysis/analyze_regime_polarity_ant_joint_mode_risk_v26.py",
        ROOT / "scripts/submit_regime_polarity_ant_joint_mode_risk_v26.py",
        ROOT / "jax_experiments/algos/regime_sac.py",
        ROOT / "jax_experiments/common/checkpoint.py",
        ROOT / "jax_experiments/common/replay_buffer.py",
        ROOT / "jax_experiments/envs/stochastic_mode_env.py",
        ROOT / "jax_experiments/networks/policy.py",
        ROOT / "jax_experiments/networks/ensemble_critic.py",
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
        != "learned_risk_does_not_stabilize_independent_ant_bank"
    ):
        raise ValueError("Ant V25 does not authorize V26")
    paths = registration_source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Ant V26 registration sources missing: {missing}")
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
            "whether one shared true-mode-conditioned actor removes the "
            "seed-specific collapse of independently optimized Ant "
            "specialists, and whether failure at equal budget is explained "
            "by fourfold lower per-mode data"
        ),
        "controller": {
            "single_shared_conditioned_actor": True,
            "single_shared_conditioned_critic": True,
            "single_shared_conditioned_risk_critic": True,
            "context": "privileged true one-hot mode",
            "initialization": (
                "robust source for every context within "
                f"{CONTROLLER_EQUIVALENCE_ATOL:g}"
            ),
            "immutable_robust_fallback": True,
            "risk_objective": "relative candidate-minus-robust",
            "risk_lambda": RISK_LAMBDA,
            "risk_warmup_updates": RISK_WARMUP_UPDATES,
        },
        "training": {
            "physical_samples_per_iter": PHYSICAL_SAMPLES_PER_ITER,
            "target_samples_per_iter": TARGET_SAMPLES_PER_ITER,
            "per_mode_target_samples_per_iter": (
                TARGET_SAMPLES_PER_ITER // len(MODES)
            ),
            "balanced_modes_each_iteration": True,
            "target_behavior_mix": {
                "candidate_policy": 0.5,
                "frozen_robust_policy": 0.5,
            },
            "updates_per_iter": UPDATES_PER_ITER,
            "segment_steps": SWITCH_SEGMENT_STEPS,
            "budgets": {
                variant: {
                    "finetune_iterations": FINETUNE_ITERS[variant],
                    "final_next_iteration": FINAL_NEXT_ITERATION[variant],
                    "final_total_physical_steps": FINAL_TOTAL_STEPS[variant],
                    "final_update_count": FINAL_UPDATE_COUNT[variant],
                }
                for variant in VARIANTS
            },
            "replay_and_optimizers_reset": True,
        },
        "evaluation": {
            "immutable_robust_fallback": True,
            "transient_fallback_steps": TRANSIENT_FALLBACK_STEPS,
            "utility_fallback_calibrated_on_disjoint_events": True,
            "new_disjoint_event_streams": True,
        },
        "gate": {
            "all_three_policy_seeds_pass": True,
            "minimum_stationary_mode_wins_per_seed": MIN_HOLDOUT_MODE_WINS,
            "minimum_per_mode_gain": MIN_HOLDOUT_MODE_GAIN,
            "minimum_switching_gain": MIN_SWITCHING_GAIN,
            "all_three_switching_events_win": True,
            "zero_transient_fallback_termination": True,
            "selection": (
                "prefer joint_equal_budget if it passes; otherwise accept "
                "joint_data_matched only as a controller-capacity result"
            ),
        },
        "stopping_rule": (
            "if neither budget passes all three development seeds, close "
            "Ant joint-controller optimization without another architecture, "
            "risk, fallback, or checkpoint sweep"
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
            raise ValueError("existing Ant V26 registration changed")
    else:
        write_json_atomic(REGISTRATION_PATH, payload)
    return validate_registration()


def validate_registration() -> dict[str, Any]:
    if not REGISTRATION_PATH.is_file():
        raise FileNotFoundError(f"missing Ant V26 registration: {REGISTRATION_PATH}")
    payload = read_json(REGISTRATION_PATH)
    if payload != registration_payload():
        raise ValueError("Ant V26 registration or source closure changed")
    return payload


def assert_protocol_integrity() -> None:
    if PHYSICAL_SAMPLES_PER_ITER != 2 * TARGET_SAMPLES_PER_ITER:
        raise ValueError("V26 physical/target accounting changed")
    pair_steps = 2 * SWITCH_SEGMENT_STEPS
    cycles = PHYSICAL_SAMPLES_PER_ITER // pair_steps
    if PHYSICAL_SAMPLES_PER_ITER % pair_steps or cycles % (2 * len(MODES)):
        raise ValueError("V26 collection cannot balance mode and behavior")
    if FINETUNE_ITERS["joint_data_matched"] != (
        len(MODES) * FINETUNE_ITERS["joint_equal_budget"]
    ):
        raise ValueError("V26 data-matched budget is not fourfold")
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
        raise ValueError("Ant V26 evaluation splits overlap")
    if any(
        len(sequence) != len(MODES) or set(sequence) != set(MODES)
        for sequence in SWITCHING_SCHEDULES.values()
    ):
        raise ValueError("each Ant V26 schedule must use every mode")


assert_protocol_integrity()
