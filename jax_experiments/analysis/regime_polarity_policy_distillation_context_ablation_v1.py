"""Checkpoint-only context ablation for the frozen mode-head student."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_policy_distillation_control_confirmation_v2 as frozen,
)


ROOT = frozen.ROOT
PROTOCOL_VERSION = "v1-frozen-mode-head-context-ablation"
ENV = frozen.ENV
FAMILY = frozen.FAMILY
MODES = frozen.MODES
REDUCTION = frozen.REDUCTION

TEACHER_GROUP = frozen.TEACHER_GROUP
TEACHER_GROUPS = (TEACHER_GROUP,)
STUDENT_SEED = frozen.STUDENT_SEED
STUDENT_SEEDS = (STUDENT_SEED,)

# Diagnostic only. These events are disjoint from every development and
# confirmation split and cannot be used for model or threshold selection.
AUDIT_EVENT_SEEDS = (102_301, 102_331, 102_367, 102_397, 102_451)

MAX_EPISODE_STEPS = frozen.MAX_EPISODE_STEPS
DWELL_STEPS = frozen.DWELL_STEPS
AUDIT_EPISODES_PER_TASK = frozen.AUDIT_EPISODES_PER_TASK
AUDIT_SWITCHING_EPISODES = frozen.AUDIT_SWITCHING_EPISODES

FIXED_ROBUST_ARM = frozen.FIXED_STRONGEST_ROBUST_ARM
TEACHER_ARMS = ("teacher_oracle_median", "teacher_learned_median")
STUDENT_CONTEXT_ARMS = (
    "student_learned",
    "student_oracle",
    "student_uniform",
    *(f"student_fixed_{mode}" for mode in MODES),
    "student_cyclic",
    "student_shuffled",
)
ARM_LABELS = (FIXED_ROBUST_ARM, *TEACHER_ARMS, *STUDENT_CONTEXT_ARMS)

AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_policy_distillation_context_ablation_audit_v1"
)
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_policy_distillation_context_ablation_analysis_v1"
)
REPORT = (
    ROOT / "reports"
    / "regime_polarity_policy_distillation_context_ablation_v1_protocol_2026-08-02.md"
)

AUDIT_SCHEMA = (
    "bapr.regime-polarity-policy-distillation-context-ablation-audit.v1"
)
EVENT_SCHEMA = (
    "bapr.regime-polarity-policy-distillation-context-ablation-event.v1"
)
ANALYSIS_SCHEMA = (
    "bapr.regime-polarity-policy-distillation-context-ablation-analysis.v1"
)

ensemble = frozen.ensemble
file_record = frozen.file_record
read_json = frozen.read_json
write_json_atomic = frozen.write_json_atomic
write_text_atomic = frozen.write_text_atomic
source_groups = frozen.source_groups
controller_keys = frozen.controller_keys
source_bundle_dirs = frozen.source_bundle_dirs
source_required_paths = frozen.source_required_paths
model_dir = frozen.model_dir
model_path = frozen.model_path
model_manifest = frozen.model_manifest
selection_analysis_path = frozen.selection_analysis_path
validate_frozen_candidate = frozen.validate_frozen_candidate

FROZEN_STUDENT_MANIFEST_RECORD = frozen.FROZEN_STUDENT_MANIFEST_RECORD
FROZEN_STUDENT_PARAMETER_RECORD = frozen.FROZEN_STUDENT_PARAMETER_RECORD
FROZEN_SELECTION_ANALYSIS_RECORD = frozen.FROZEN_SELECTION_ANALYSIS_RECORD


def require_teacher_group(group: str) -> str:
    return frozen.require_teacher_group(group)


def require_student_seed(seed: int) -> int:
    return frozen.require_student_seed(seed)


def require_audit_event_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in AUDIT_EVENT_SEEDS:
        raise ValueError(f"unknown context-ablation event seed {seed}")
    return seed


def require_arm(arm: str) -> str:
    arm = str(arm)
    if arm not in ARM_LABELS:
        raise ValueError(f"unknown context-ablation arm {arm!r}")
    return arm


def shuffled_mode_map(event_seed: int) -> tuple[int, ...]:
    event_seed = require_audit_event_seed(event_seed)
    modes = list(MODES)
    rng = random.Random(event_seed + 28_020_826)
    while True:
        shuffled = modes.copy()
        rng.shuffle(shuffled)
        if all(left != right for left, right in zip(modes, shuffled)):
            return tuple(shuffled)


def audit_dir(group: str, student_seed: int, event_seed: int) -> Path:
    require_teacher_group(group)
    require_student_seed(student_seed)
    return AUDIT_ROOT / f"event_seed_{require_audit_event_seed(event_seed)}"


def audit_manifest(group: str, student_seed: int, event_seed: int) -> Path:
    return audit_dir(group, student_seed, event_seed) / "audit_manifest.json"


def all_audit_manifests() -> tuple[Path, ...]:
    return tuple(
        audit_manifest(TEACHER_GROUP, STUDENT_SEED, event_seed)
        for event_seed in AUDIT_EVENT_SEEDS
    )


def analysis_json() -> Path:
    return ANALYSIS_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return ANALYSIS_ROOT / "analysis.md"


def audit_identity(
    group: str,
    student_seed: int,
    event_seed: int,
) -> dict[str, Any]:
    event_seed = require_audit_event_seed(event_seed)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "post_confirmation_mechanism_diagnostic",
        "confirmatory": False,
        "selection_forbidden": True,
        "env": ENV,
        "family": FAMILY,
        "variant": require_teacher_group(group),
        "student_seed": require_student_seed(student_seed),
        "event_seed": event_seed,
        "arms": list(ARM_LABELS),
        "shuffled_mode_map": list(shuffled_mode_map(event_seed)),
        "frozen_student_manifest": FROZEN_STUDENT_MANIFEST_RECORD,
        "frozen_student_parameters": FROZEN_STUDENT_PARAMETER_RECORD,
        "frozen_selection_analysis": FROZEN_SELECTION_ANALYSIS_RECORD,
        "online_inputs": [
            "observation",
            "commanded_action",
            "next_observation",
        ],
        "forbidden_online_inputs_for_learned": [
            "mode_id",
            "action_gain",
            "executed_action",
            "switch_clock",
        ],
    }

