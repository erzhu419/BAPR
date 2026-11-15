"""Closed-loop development protocol for polarity policy compression."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_policy_distillation as base,
)


ROOT = base.ROOT
PROTOCOL_VERSION = "v2-closed-loop-return-aware"
ENV = base.ENV
FAMILY = base.FAMILY
MODES = base.MODES
REDUCTION = base.REDUCTION

# These are architecture/objective variants, not alternative teacher subsets.
# Every arm uses the same frozen combined-ten median teacher.
VARIANTS = ("wide_dagger", "mode_heads", "mode_heads_return")
TEACHER_GROUPS = VARIANTS
STUDENT_SEEDS = (1709, 1811, 1901)

# Development-only splits. The frozen confirmation seeds 100019, 100043,
# 100069, 100103, and 100151 are intentionally absent and remain sealed.
TRAIN_EVENT_SEEDS = base.TRAIN_EVENT_SEEDS
SUPERVISED_VALIDATION_EVENT_SEEDS = base.VALIDATION_EVENT_SEEDS
DAGGER_EVENT_SEEDS = (101_201, 101_219, 101_237, 101_261)
CONTROL_VALIDATION_EVENT_SEEDS = (101_401, 101_417)
AUDIT_EVENT_SEEDS = (101_603, 101_627, 101_653)
SEALED_CONFIRMATION_EVENT_SEEDS = (100_019, 100_043, 100_069, 100_103, 100_151)

MAX_EPISODE_STEPS = base.MAX_EPISODE_STEPS
DWELL_STEPS = base.DWELL_STEPS
TRAIN_STATIONARY_EPISODES = 1
TRAIN_SWITCHING_EPISODES = 3
DAGGER_ROUNDS = 4
DAGGER_SWITCHING_EPISODES = 3
CONTROL_VALIDATION_SWITCHING_EPISODES = 3

MODEL_CONFIG = {
    "learning_rate": 3e-4,
    "weight_decay": 1e-6,
    "gradient_clip": 5.0,
    "batch_size": 512,
    "initial_updates": 20_000,
    "dagger_updates_per_round": 10_000,
    "validation_interval": 500,
    "pre_tanh_loss_weight": 0.05,
    "pre_tanh_clip": 0.995,
    "wide_hidden_dim": 512,
    "wide_n_layers": 3,
    "mode_hidden_dim": 256,
    "mode_n_layers": 2,
    "return_regret_scale": 2.0,
    "switch_weight_scale": 2.0,
    "action_disagreement_scale": 2.0,
    "switch_weight_tau": 25.0,
    "action_disagreement_normalizer": 0.05,
    "max_sample_weight": 8.0,
    "termination_penalty": 2_000.0,
}

AUDIT_EPISODES_PER_TASK = base.AUDIT_EPISODES_PER_TASK
AUDIT_SWITCHING_EPISODES = base.AUDIT_SWITCHING_EPISODES
FIXED_ROBUST_ARM = "robust_final_seed_719"
MIN_EVENT_WINS_VS_POPULATION = len(AUDIT_EVENT_SEEDS)
MIN_EVENT_WINS_VS_FIXED_ROBUST = 2
MIN_STUDENT_SEED_PASSES = 2
MIN_HEADROOM_RECOVERY = 0.80
MAX_MEAN_STUDENT_TEACHER_GAP = 25.0
MAX_TERMINATED_RATE = 0.0

MODEL_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_policy_distillation_control_model_v2")
WORK_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_policy_distillation_control_work_v2")
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_policy_distillation_control_audit_v2")
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_policy_distillation_control_analysis_v2")
REPORT = (
    ROOT / "reports"
    / "regime_polarity_policy_distillation_control_v2_protocol_2026-08-01.md")

MODEL_SCHEMA = "bapr.regime-polarity-policy-distillation-control-model.v2"
CHECKPOINT_SCHEMA = (
    "bapr.regime-polarity-policy-distillation-control-checkpoint.v2")
AUDIT_SCHEMA = "bapr.regime-polarity-policy-distillation-control-audit.v2"
EVENT_SCHEMA = "bapr.regime-polarity-policy-distillation-control-event.v2"
ANALYSIS_SCHEMA = "bapr.regime-polarity-policy-distillation-control-analysis.v2"

ensemble = base.ensemble
file_record = base.file_record
read_json = base.read_json
write_json_atomic = base.write_json_atomic
write_text_atomic = base.write_text_atomic
save_parameter_state = base.save_parameter_state
load_parameter_state = base.load_parameter_state


def require_teacher_group(variant: str) -> str:
    variant = str(variant)
    if variant not in VARIANTS:
        raise ValueError(f"unknown compression variant {variant!r}; expected {VARIANTS}")
    return variant


require_variant = require_teacher_group


def require_student_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in STUDENT_SEEDS:
        raise ValueError(f"unknown compression student seed {seed}")
    return seed


def require_audit_event_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in AUDIT_EVENT_SEEDS:
        raise ValueError(f"unknown compression audit event seed {seed}")
    return seed


def source_groups(variant: str) -> tuple[str, ...]:
    require_variant(variant)
    return ("development", "final")


def controller_keys(variant: str) -> tuple[tuple[str, int], ...]:
    require_variant(variant)
    return base.controller_keys("combined")


def source_bundle_dirs(variant: str) -> tuple[Path, ...]:
    require_variant(variant)
    return base.source_bundle_dirs("combined")


def source_required_paths(variant: str) -> tuple[Path, ...]:
    require_variant(variant)
    return base.source_required_paths("combined")


def model_dir(variant: str, student_seed: int) -> Path:
    return (
        MODEL_ROOT / require_variant(variant)
        / f"student_seed_{require_student_seed(student_seed)}")


def work_dir(variant: str, student_seed: int) -> Path:
    return (
        WORK_ROOT / require_variant(variant)
        / f"student_seed_{require_student_seed(student_seed)}")


def model_path(variant: str, student_seed: int) -> Path:
    return model_dir(variant, student_seed) / "student_params.npz"


def model_manifest(variant: str, student_seed: int) -> Path:
    return model_dir(variant, student_seed) / "model_manifest.json"


def audit_dir(variant: str, student_seed: int, event_seed: int) -> Path:
    return (
        AUDIT_ROOT / require_variant(variant)
        / f"student_seed_{require_student_seed(student_seed)}"
        / f"event_seed_{require_audit_event_seed(event_seed)}")


def audit_manifest(variant: str, student_seed: int, event_seed: int) -> Path:
    return audit_dir(variant, student_seed, event_seed) / "audit_manifest.json"


def all_audit_manifests() -> tuple[Path, ...]:
    return tuple(
        audit_manifest(variant, student_seed, event_seed)
        for variant in VARIANTS
        for student_seed in STUDENT_SEEDS
        for event_seed in AUDIT_EVENT_SEEDS
    )


def analysis_json() -> Path:
    return ANALYSIS_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return ANALYSIS_ROOT / "analysis.md"


def model_identity(variant: str, student_seed: int) -> dict[str, Any]:
    variant = require_variant(variant)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "closed_loop_return_aware_compression_development",
        "development_only": True,
        "env": ENV,
        "family": FAMILY,
        "variant": variant,
        "teacher_group": "combined",
        "controller_keys": [list(value) for value in controller_keys(variant)],
        "reduction": REDUCTION,
        "student_seed": require_student_seed(student_seed),
        "train_event_seeds": list(TRAIN_EVENT_SEEDS),
        "dagger_event_seeds": list(DAGGER_EVENT_SEEDS),
        "supervised_validation_event_seeds": list(
            SUPERVISED_VALIDATION_EVENT_SEEDS),
        "control_validation_event_seeds": list(
            CONTROL_VALIDATION_EVENT_SEEDS),
        "dagger_rounds": DAGGER_ROUNDS,
        "model_config": MODEL_CONFIG,
        "student_online_inputs": ["observation", "soft_mode_posterior"],
        "checkpoint_selection": "independent_closed_loop_switching_return",
    }


def audit_identity(
    variant: str,
    student_seed: int,
    event_seed: int,
) -> dict[str, Any]:
    variant = require_variant(variant)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "closed_loop_compression_development_audit",
        "development_only": True,
        "env": ENV,
        "family": FAMILY,
        "variant": variant,
        "teacher_group": "combined",
        "controller_keys": [list(value) for value in controller_keys(variant)],
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


def assert_split_integrity() -> None:
    splits = (
        TRAIN_EVENT_SEEDS,
        DAGGER_EVENT_SEEDS,
        SUPERVISED_VALIDATION_EVENT_SEEDS,
        CONTROL_VALIDATION_EVENT_SEEDS,
        AUDIT_EVENT_SEEDS,
    )
    flattened = [seed for split in splits for seed in split]
    if len(flattened) != len(set(flattened)):
        raise ValueError("closed-loop compression splits overlap")
    if set(flattened) & set(SEALED_CONFIRMATION_EVENT_SEEDS):
        raise ValueError("closed-loop development reused a sealed confirmation seed")


assert_split_integrity()
