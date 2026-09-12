"""Frozen five-event confirmation for the selected mode-head student."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_policy_distillation_control_v2 as development,
)


ROOT = development.ROOT
PROTOCOL_VERSION = "v2-mode-heads-frozen-five-event"
ENV = development.ENV
FAMILY = development.FAMILY
MODES = development.MODES
REDUCTION = development.REDUCTION

TEACHER_GROUP = "mode_heads"
TEACHER_GROUPS = (TEACHER_GROUP,)
STUDENT_SEED = 1811
STUDENT_SEEDS = (STUDENT_SEED,)

# Frozen before any result on this split was generated. These seeds are absent
# from every training, DAgger, validation, development-audit, and prior
# confirmation split.
AUDIT_EVENT_SEEDS = (102_019, 102_043, 102_069, 102_103, 102_151)

MAX_EPISODE_STEPS = development.MAX_EPISODE_STEPS
DWELL_STEPS = development.DWELL_STEPS
AUDIT_EPISODES_PER_TASK = development.AUDIT_EPISODES_PER_TASK
AUDIT_SWITCHING_EPISODES = development.AUDIT_SWITCHING_EPISODES
MODEL_CONFIG = development.MODEL_CONFIG

FIXED_STRONGEST_ROBUST_ARM = development.FIXED_ROBUST_ARM
MIN_EVENT_WINS_VS_POPULATION = len(AUDIT_EVENT_SEEDS)
MIN_EVENT_WINS_VS_FIXED_ROBUST = 4
MIN_HEADROOM_RECOVERY = 0.80
MAX_TERMINATED_RATE = 0.0
CLUSTER_T_CRITICAL_95 = 2.7764451051977987

MODEL_ROOT = development.MODEL_ROOT
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_policy_distillation_control_confirmation_audit_v2"
)
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_policy_distillation_control_confirmation_analysis_v2"
)
REPORT = (
    ROOT / "reports"
    / "regime_polarity_policy_distillation_control_confirmation_v2_protocol_2026-08-02.md"
)

AUDIT_SCHEMA = (
    "bapr.regime-polarity-policy-distillation-control-confirmation-audit.v2"
)
EVENT_SCHEMA = (
    "bapr.regime-polarity-policy-distillation-control-confirmation-event.v2"
)
ANALYSIS_SCHEMA = (
    "bapr.regime-polarity-policy-distillation-control-confirmation-analysis.v2"
)

FROZEN_STUDENT_MANIFEST_RECORD = {
    "sha256": "4f3f8cce988e9e546b6009bbaefb3a235f8085c91955fdc7da957c28b744ec75",
    "size": 26975,
}
FROZEN_STUDENT_PARAMETER_RECORD = {
    "sha256": "2f54b0a40352a68349b52ee36ede26f0bbe97085e37b595ed9f7c37b400104d8",
    "size": 287370,
}
FROZEN_SELECTION_ANALYSIS_RECORD = {
    "sha256": "a2d9aed89262f687d60042837d698fd5968627d8e120262cc11a989a5f1947d9",
    "size": 53210,
}

ensemble = development.ensemble
file_record = development.file_record
read_json = development.read_json
write_json_atomic = development.write_json_atomic
write_text_atomic = development.write_text_atomic
model_identity = development.model_identity


def require_teacher_group(group: str) -> str:
    group = str(group)
    if group != TEACHER_GROUP:
        raise ValueError(
            f"confirmation requires variant {TEACHER_GROUP!r}, got {group!r}"
        )
    return group


require_variant = require_teacher_group


def require_student_seed(seed: int) -> int:
    seed = int(seed)
    if seed != STUDENT_SEED:
        raise ValueError(
            f"confirmation requires frozen student seed {STUDENT_SEED}, got {seed}"
        )
    return seed


def require_audit_event_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in AUDIT_EVENT_SEEDS:
        raise ValueError(f"unknown confirmation event seed {seed}")
    return seed


def source_groups(group: str) -> tuple[str, ...]:
    return development.source_groups(require_teacher_group(group))


def controller_keys(group: str) -> tuple[tuple[str, int], ...]:
    return development.controller_keys(require_teacher_group(group))


def source_bundle_dirs(group: str) -> tuple[Path, ...]:
    return development.source_bundle_dirs(require_teacher_group(group))


def source_required_paths(group: str) -> tuple[Path, ...]:
    return development.source_required_paths(require_teacher_group(group))


def model_dir(group: str, student_seed: int) -> Path:
    return development.model_dir(
        require_teacher_group(group), require_student_seed(student_seed)
    )


def model_path(group: str, student_seed: int) -> Path:
    return model_dir(group, student_seed) / "student_params.npz"


def model_manifest(group: str, student_seed: int) -> Path:
    return model_dir(group, student_seed) / "model_manifest.json"


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
    return development.analysis_json()


def audit_identity(
    group: str,
    student_seed: int,
    event_seed: int,
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "independent_mode_head_student_confirmation",
        "confirmatory": True,
        "env": ENV,
        "family": FAMILY,
        "variant": require_teacher_group(group),
        "teacher_group": "combined",
        "controller_keys": [
            list(value) for value in controller_keys(TEACHER_GROUP)
        ],
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
        raise ValueError("frozen mode-head manifest changed")
    if file_record(model_path(TEACHER_GROUP, STUDENT_SEED)) \
            != FROZEN_STUDENT_PARAMETER_RECORD:
        raise ValueError("frozen mode-head parameters changed")
    if file_record(selection_analysis_path()) \
            != FROZEN_SELECTION_ANALYSIS_RECORD:
        raise ValueError("frozen control-validation selection changed")

    manifest = read_json(model_manifest(TEACHER_GROUP, STUDENT_SEED))
    if (
        manifest.get("schema") != development.MODEL_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("identity")
        != development.model_identity(TEACHER_GROUP, STUDENT_SEED)
        or manifest.get("parameter_file") != FROZEN_STUDENT_PARAMETER_RECORD
    ):
        raise ValueError("frozen mode-head manifest is internally invalid")

    selection = read_json(selection_analysis_path())
    selected = selection.get("variants", {}).get(TEACHER_GROUP, {})
    if (
        selection.get("schema") != development.ANALYSIS_SCHEMA
        or selection.get("promotion") is not True
        or selection.get("selected_variant") != TEACHER_GROUP
        or int(selected.get("control_validation_selected_seed", -1))
        != STUDENT_SEED
        or selected.get("selected_student_pass") is not True
    ):
        raise ValueError("mode-head seed 1811 is no longer the frozen candidate")

