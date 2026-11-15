"""Frozen five-event confirmation for the selected distilled policy."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_policy_distillation as base,
)


ROOT = base.ROOT
PROTOCOL_VERSION = "v1-frozen-five-event"
ENV = base.ENV
FAMILY = base.FAMILY
MODES = base.MODES
REDUCTION = base.REDUCTION

TEACHER_GROUP = "combined"
TEACHER_GROUPS = (TEACHER_GROUP,)
STUDENT_SEED = 1511
STUDENT_SEEDS = (STUDENT_SEED,)

# Frozen before this confirmation was submitted. None appears in the teacher,
# distillation, validation, ensemble-diagnostic, or retrospective-audit split.
AUDIT_EVENT_SEEDS = (100_019, 100_043, 100_069, 100_103, 100_151)

MAX_EPISODE_STEPS = base.MAX_EPISODE_STEPS
DWELL_STEPS = base.DWELL_STEPS
AUDIT_EPISODES_PER_TASK = base.AUDIT_EPISODES_PER_TASK
AUDIT_SWITCHING_EPISODES = base.AUDIT_SWITCHING_EPISODES
MODEL_CONFIG = base.MODEL_CONFIG

FIXED_STRONGEST_ROBUST_ARM = "robust_final_seed_719"
MIN_EVENT_WINS_VS_POPULATION = len(AUDIT_EVENT_SEEDS)
MIN_EVENT_WINS_VS_FIXED_ROBUST = 4
MIN_HEADROOM_RECOVERY = 0.80
MAX_TERMINATED_RATE = 0.0
CLUSTER_T_CRITICAL_95 = 2.7764451051977987

MODEL_ROOT = base.MODEL_ROOT
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_policy_distillation_confirmation_audit_v1")
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_policy_distillation_confirmation_analysis_v1")
REPORT = (
    ROOT / "reports"
    / "regime_polarity_policy_distillation_confirmation_protocol_2026-08-01.md")

AUDIT_SCHEMA = (
    "bapr.regime-polarity-policy-distillation-confirmation-audit.v1")
EVENT_SCHEMA = (
    "bapr.regime-polarity-policy-distillation-confirmation-event.v1")
ANALYSIS_SCHEMA = (
    "bapr.regime-polarity-policy-distillation-confirmation-analysis.v1")

FROZEN_STUDENT_MANIFEST_RECORD = {
    "sha256": "20e180c458b5a57975c9b06d5f4a54796eb5ebe941c62b93a856110d8f328e7a",
    "size": 9561,
}
FROZEN_STUDENT_PARAMETER_RECORD = {
    "sha256": "cf28fb30dde6b58346319087763f7e51cacf0b603f62cae7ebbf60fc50dc4d25",
    "size": 278547,
}
FROZEN_SELECTION_ANALYSIS_RECORD = {
    "sha256": "60353c028c7523ae2bab51385b22807bf485b30053cb98804378e02b8415304b",
    "size": 35160,
}

ensemble = base.ensemble
file_record = base.file_record
read_json = base.read_json
write_json_atomic = base.write_json_atomic
write_text_atomic = base.write_text_atomic
source_groups = base.source_groups
controller_keys = base.controller_keys
source_bundle_dirs = base.source_bundle_dirs
source_required_paths = base.source_required_paths
model_dir = base.model_dir
model_path = base.model_path
model_manifest = base.model_manifest
model_identity = base.model_identity


def require_teacher_group(group: str) -> str:
    group = str(group)
    if group != TEACHER_GROUP:
        raise ValueError(
            f"confirmation requires teacher group {TEACHER_GROUP!r}, "
            f"got {group!r}")
    return group


def require_student_seed(seed: int) -> int:
    seed = int(seed)
    if seed != STUDENT_SEED:
        raise ValueError(
            f"confirmation requires frozen student seed {STUDENT_SEED}, "
            f"got {seed}")
    return seed


def require_audit_event_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in AUDIT_EVENT_SEEDS:
        raise ValueError(f"unknown confirmation event seed {seed}")
    return seed


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


def selection_analysis_path() -> Path:
    return base.analysis_json()


def audit_identity(
    group: str,
    student_seed: int,
    event_seed: int,
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "independent_frozen_student_confirmation",
        "confirmatory": True,
        "env": ENV,
        "family": FAMILY,
        "teacher_group": require_teacher_group(group),
        "controller_keys": [
            list(value) for value in controller_keys(TEACHER_GROUP)],
        "reduction": REDUCTION,
        "student_seed": require_student_seed(student_seed),
        "event_seed": require_audit_event_seed(event_seed),
        "frozen_student_manifest": FROZEN_STUDENT_MANIFEST_RECORD,
        "frozen_student_parameters": FROZEN_STUDENT_PARAMETER_RECORD,
        "frozen_selection_analysis": FROZEN_SELECTION_ANALYSIS_RECORD,
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


def validate_frozen_candidate() -> None:
    if file_record(model_manifest(TEACHER_GROUP, STUDENT_SEED)) \
            != FROZEN_STUDENT_MANIFEST_RECORD:
        raise ValueError("frozen distillation manifest changed")
    if file_record(model_path(TEACHER_GROUP, STUDENT_SEED)) \
            != FROZEN_STUDENT_PARAMETER_RECORD:
        raise ValueError("frozen distillation parameters changed")
    if file_record(selection_analysis_path()) \
            != FROZEN_SELECTION_ANALYSIS_RECORD:
        raise ValueError("frozen validation-selection analysis changed")

    manifest = read_json(model_manifest(TEACHER_GROUP, STUDENT_SEED))
    if (manifest.get("schema") != base.MODEL_SCHEMA
            or manifest.get("status") != "complete"
            or manifest.get("identity")
            != model_identity(TEACHER_GROUP, STUDENT_SEED)
            or manifest.get("parameter_file")
            != FROZEN_STUDENT_PARAMETER_RECORD):
        raise ValueError("frozen distillation manifest is internally invalid")
    selection = read_json(selection_analysis_path())
    combined = selection.get("groups", {}).get(TEACHER_GROUP, {})
    if (selection.get("promotion") is not True
            or int(combined.get("validation_selected_seed", -1))
            != STUDENT_SEED):
        raise ValueError("seed 1511 is no longer the validation-selected model")
