"""Protocol for distilling the stable polarity policy ensemble."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_policy_ensemble as ensemble,
)


ROOT = ensemble.ROOT
PROTOCOL_VERSION = "v1-dagger-median"
ENV = ensemble.ENV
FAMILY = ensemble.FAMILY
MODES = ensemble.MODES
REDUCTION = "median"

TEACHER_GROUPS = ("development", "final", "combined")
STUDENT_SEEDS = (1409, 1511, 1601)

# The ensemble diagnostic used 97001-97003.  These splits remain disjoint
# from that screen and from the earlier confirmation protocols.
TRAIN_EVENT_SEEDS = (98_001, 98_002, 98_003)
DAGGER_EVENT_SEEDS = (98_101, 98_102)
VALIDATION_EVENT_SEEDS = (98_901, 98_902)
AUDIT_EVENT_SEEDS = (99_001, 99_002, 99_003)

MAX_EPISODE_STEPS = ensemble.MAX_EPISODE_STEPS
DWELL_STEPS = ensemble.DWELL_STEPS
TRAIN_STATIONARY_EPISODES = 1
TRAIN_SWITCHING_EPISODES = 3
DAGGER_ROUNDS = 2
DAGGER_SWITCHING_EPISODES = 3

MODEL_CONFIG = {
    "hidden_dim": 256,
    "n_layers": 2,
    "learning_rate": 3e-4,
    "weight_decay": 1e-6,
    "gradient_clip": 5.0,
    "batch_size": 512,
    "initial_updates": 15_000,
    "dagger_updates_per_round": 10_000,
    "validation_interval": 500,
    "pre_tanh_loss_weight": 0.05,
    "pre_tanh_clip": 0.995,
}

AUDIT_EPISODES_PER_TASK = ensemble.EPISODES_PER_TASK
AUDIT_SWITCHING_EPISODES = ensemble.SWITCHING_EPISODES

MIN_EVENT_SEED_WINS = len(AUDIT_EVENT_SEEDS)
MIN_HEADROOM_RECOVERY = 0.80
MIN_STUDENT_SEED_PASSES = 2
MAX_TERMINATED_RATE = 0.0

MODEL_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_policy_distillation_model_v1")
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_policy_distillation_audit_v1")
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_policy_distillation_analysis_v1")
REPORT = (
    ROOT / "reports"
    / "regime_polarity_policy_distillation_protocol_2026-08-01.md")

MODEL_SCHEMA = "bapr.regime-polarity-policy-distillation-model.v1"
AUDIT_SCHEMA = "bapr.regime-polarity-policy-distillation-audit.v1"
EVENT_SCHEMA = "bapr.regime-polarity-policy-distillation-event.v1"
ANALYSIS_SCHEMA = "bapr.regime-polarity-policy-distillation-analysis.v1"

file_record = ensemble.file_record
read_json = ensemble.read_json
write_json_atomic = ensemble.write_json_atomic
write_text_atomic = ensemble.write_text_atomic
save_parameter_state = ensemble.final.estimator_protocol.save_parameter_state
load_parameter_state = ensemble.final.estimator_protocol.load_parameter_state


def require_teacher_group(group: str) -> str:
    group = str(group)
    if group not in TEACHER_GROUPS:
        raise ValueError(
            f"unknown distillation teacher group {group!r}; "
            f"expected {TEACHER_GROUPS}")
    return group


def require_student_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in STUDENT_SEEDS:
        raise ValueError(f"unknown distillation student seed {seed}")
    return seed


def require_audit_event_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in AUDIT_EVENT_SEEDS:
        raise ValueError(f"unknown distillation audit event seed {seed}")
    return seed


def source_groups(group: str) -> tuple[str, ...]:
    group = require_teacher_group(group)
    return ensemble.GROUPS if group == "combined" else (group,)


def controller_keys(group: str) -> tuple[tuple[str, int], ...]:
    return tuple(
        (source_group, seed)
        for source_group in source_groups(group)
        for seed in ensemble.controller_seeds(source_group)
    )


def source_bundle_dirs(group: str) -> tuple[Path, ...]:
    return tuple(
        path
        for source_group in source_groups(group)
        for path in ensemble.source_bundle_dirs(source_group)
    )


def source_required_paths(group: str) -> tuple[Path, ...]:
    return tuple(
        path
        for source_group in source_groups(group)
        for path in ensemble.source_required_paths(source_group)
    )


def model_dir(group: str, student_seed: int) -> Path:
    return (
        MODEL_ROOT / require_teacher_group(group)
        / f"student_seed_{require_student_seed(student_seed)}")


def model_path(group: str, student_seed: int) -> Path:
    return model_dir(group, student_seed) / "student_params.npz"


def model_manifest(group: str, student_seed: int) -> Path:
    return model_dir(group, student_seed) / "model_manifest.json"


def audit_dir(group: str, student_seed: int, event_seed: int) -> Path:
    return (
        AUDIT_ROOT / require_teacher_group(group)
        / f"student_seed_{require_student_seed(student_seed)}"
        / f"event_seed_{require_audit_event_seed(event_seed)}")


def audit_manifest(group: str, student_seed: int, event_seed: int) -> Path:
    return audit_dir(group, student_seed, event_seed) / "audit_manifest.json"


def all_audit_manifests() -> tuple[Path, ...]:
    return tuple(
        audit_manifest(group, student_seed, event_seed)
        for group in TEACHER_GROUPS
        for student_seed in STUDENT_SEEDS
        for event_seed in AUDIT_EVENT_SEEDS
    )


def analysis_json() -> Path:
    return ANALYSIS_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return ANALYSIS_ROOT / "analysis.md"


def model_identity(group: str, student_seed: int) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "retrospective_policy_ensemble_distillation",
        "development_only": True,
        "env": ENV,
        "family": FAMILY,
        "teacher_group": require_teacher_group(group),
        "controller_keys": [list(value) for value in controller_keys(group)],
        "reduction": REDUCTION,
        "student_seed": require_student_seed(student_seed),
        "train_event_seeds": list(TRAIN_EVENT_SEEDS),
        "dagger_event_seeds": list(DAGGER_EVENT_SEEDS),
        "validation_event_seeds": list(VALIDATION_EVENT_SEEDS),
        "dagger_rounds": DAGGER_ROUNDS,
        "model_config": MODEL_CONFIG,
        "student_online_inputs": ["observation", "soft_mode_posterior"],
    }


def audit_identity(
    group: str,
    student_seed: int,
    event_seed: int,
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "retrospective_distilled_student_audit",
        "development_only": True,
        "env": ENV,
        "family": FAMILY,
        "teacher_group": require_teacher_group(group),
        "controller_keys": [list(value) for value in controller_keys(group)],
        "reduction": REDUCTION,
        "student_seed": require_student_seed(student_seed),
        "event_seed": require_audit_event_seed(event_seed),
        "online_inputs": [
            "observation",
            "commanded_action",
            "next_observation",
        ],
        "forbidden_online_inputs": [
            "mode_id",
            "action_gain",
            "executed_action",
            "switch_clock",
        ],
    }
