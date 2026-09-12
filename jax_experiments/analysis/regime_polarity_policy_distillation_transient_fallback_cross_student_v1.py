"""Frozen cross-student audit for the causal transient fallback."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_policy_distillation_control_v2 as development,
)
from jax_experiments.analysis import (
    regime_polarity_policy_distillation_transient_fallback_v1 as parent,
)


ROOT = parent.ROOT
PROTOCOL_VERSION = "v1-frozen-fallback-cross-student"
ENV = parent.ENV
FAMILY = parent.FAMILY
TEACHER_GROUP = parent.TEACHER_GROUP
STUDENT_SEEDS = development.STUDENT_SEEDS
EVENT_SEEDS = (104_301, 104_331, 104_367, 104_399, 104_451)

SELECTED_CONFIG_NAME = "evidence_1p0_k1"
SELECTED_CONFIG_VALUES = {
    "name": SELECTED_CONFIG_NAME,
    "contradiction_threshold": 1.0,
    "stable_steps": 1,
    "enter_confidence": 0.60,
    "exit_confidence": 0.90,
}

ROBUST_ARM = parent.ROBUST_ARM
LEARNED_ARM = parent.LEARNED_ARM
ORACLE_ARM = parent.ORACLE_ARM
ARMS = (ROBUST_ARM, LEARNED_ARM, ORACLE_ARM, SELECTED_CONFIG_NAME)

FROZEN_SELECTION_MANIFEST_RECORD = {
    "sha256": "02e3e965d19eda0df0eb62cb379518cfc2107593b94278e65ceb6f429f621c73",
    "size": 2746,
}
FROZEN_TRANSIENT_ANALYSIS_RECORD = {
    "sha256": "a92602eb21d6ac43428277c336d2438d5eaff590d7d35844fad4a58f6aab2eac",
    "size": 4031,
}
FROZEN_ESTIMATOR_MANIFEST_RECORD = {
    "sha256": "fc4b28662ac1f38081aab02723ebcbea2fa47fbe37eea27bb4e1a564b8d571d0",
    "size": 56089,
}
FROZEN_ESTIMATOR_PARAMETER_RECORD = {
    "sha256": "c1ff9021a3d3684116464c421c852eb16cd7ab286b2fd53bcd764d9a50393af1",
    "size": 2712127,
}
FROZEN_STUDENT_RECORDS = {
    1709: {
        "manifest": {
            "sha256": "da1ab1e2bf4f1641eb7f0a4ab28913403e59ad654a725be604942f8f14b2b9cc",
            "size": 26984,
        },
        "parameters": {
            "sha256": "969a9e94255d02c6a1ad9ca464adef9ca152f5c0da406a02f2410c1b152e4cc4",
            "size": 287375,
        },
    },
    1811: {
        "manifest": {
            "sha256": "4f3f8cce988e9e546b6009bbaefb3a235f8085c91955fdc7da957c28b744ec75",
            "size": 26975,
        },
        "parameters": {
            "sha256": "2f54b0a40352a68349b52ee36ede26f0bbe97085e37b595ed9f7c37b400104d8",
            "size": 287370,
        },
    },
    1901: {
        "manifest": {
            "sha256": "533c4907340002827365ec1f7fdb50893b1d24286f27908246eb0f1436209257",
            "size": 26972,
        },
        "parameters": {
            "sha256": "04e0f1203cdf2d56202eea378f7df8c22eaf6aa11233d6d539b9edcc6570336d",
            "size": 287386,
        },
    },
}

AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_policy_distillation_transient_fallback_cross_student_audit_v1"
)
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_policy_distillation_transient_fallback_cross_student_analysis_v1"
)
REPORT = (
    ROOT / "reports"
    / "regime_polarity_policy_distillation_transient_fallback_cross_student_v1_protocol_2026-08-02.md"
)

AUDIT_SCHEMA = "bapr.regime-polarity-transient-fallback-cross-student-audit.v1"
EVENT_SCHEMA = "bapr.regime-polarity-transient-fallback-cross-student-event.v1"
ANALYSIS_SCHEMA = "bapr.regime-polarity-transient-fallback-cross-student-analysis.v1"

file_record = parent.file_record
read_json = parent.read_json
write_json_atomic = parent.write_json_atomic
write_text_atomic = parent.write_text_atomic


def require_student_seed(seed: int) -> int:
    return development.require_student_seed(seed)


def require_event_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in EVENT_SEEDS:
        raise ValueError(f"unknown cross-student event seed {seed}")
    return seed


def student_records(seed: int) -> dict[str, dict[str, Any]]:
    return FROZEN_STUDENT_RECORDS[require_student_seed(seed)]


def audit_dir(student_seed: int, event_seed: int) -> Path:
    return (
        AUDIT_ROOT
        / f"student_seed_{require_student_seed(student_seed)}"
        / f"event_seed_{require_event_seed(event_seed)}"
    )


def audit_manifest(student_seed: int, event_seed: int) -> Path:
    return audit_dir(student_seed, event_seed) / "audit_manifest.json"


def all_audit_manifests() -> tuple[Path, ...]:
    return tuple(
        audit_manifest(student_seed, event_seed)
        for student_seed in STUDENT_SEEDS
        for event_seed in EVENT_SEEDS
    )


def analysis_json() -> Path:
    return ANALYSIS_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return ANALYSIS_ROOT / "analysis.md"


def identity(student_seed: int, event_seed: int) -> dict[str, Any]:
    student_seed = require_student_seed(student_seed)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "frozen_transient_fallback_cross_student_audit",
        "development_only": True,
        "env": ENV,
        "family": FAMILY,
        "variant": TEACHER_GROUP,
        "student_seed": student_seed,
        "event_seed": require_event_seed(event_seed),
        "selected_fallback_config": SELECTED_CONFIG_VALUES,
        "frozen_student_manifest": student_records(student_seed)["manifest"],
        "frozen_student_parameters": student_records(student_seed)["parameters"],
        "frozen_selection_manifest": FROZEN_SELECTION_MANIFEST_RECORD,
        "frozen_transient_analysis": FROZEN_TRANSIENT_ANALYSIS_RECORD,
        "online_inputs": [
            "observation",
            "commanded_action",
            "reward",
            "next_observation",
            "causal_mode_posterior",
            "one_step_mode_log_likelihood",
        ],
        "forbidden_online_inputs": [
            "mode_id",
            "action_gain",
            "executed_action",
            "switch_clock",
        ],
    }


def validate_upstream(student_seed: int) -> None:
    student_seed = require_student_seed(student_seed)
    if parent.require_config(SELECTED_CONFIG_NAME).to_dict() \
            != SELECTED_CONFIG_VALUES:
        raise ValueError("frozen fallback configuration changed")
    if file_record(parent.selection_manifest()) \
            != FROZEN_SELECTION_MANIFEST_RECORD:
        raise ValueError("frozen fallback selection manifest changed")
    if file_record(parent.analysis_json()) != FROZEN_TRANSIENT_ANALYSIS_RECORD:
        raise ValueError("frozen transient-fallback analysis changed")
    analysis = read_json(parent.analysis_json())
    if (
        analysis.get("schema") != parent.ANALYSIS_SCHEMA
        or analysis.get("status") != "complete"
        or analysis.get("selected_config") != SELECTED_CONFIG_NAME
        or analysis.get("authorize_belief_augmentation_training") is not True
    ):
        raise ValueError("upstream transient-fallback gate is not frozen-pass")
    if file_record(development.ensemble.final.MODEL_MANIFEST) \
            != FROZEN_ESTIMATOR_MANIFEST_RECORD:
        raise ValueError("frozen estimator manifest changed")
    if file_record(development.ensemble.final.MODEL_PATH) \
            != FROZEN_ESTIMATOR_PARAMETER_RECORD:
        raise ValueError("frozen estimator parameters changed")

    records = student_records(student_seed)
    manifest_path = development.model_manifest(TEACHER_GROUP, student_seed)
    parameter_path = development.model_path(TEACHER_GROUP, student_seed)
    if file_record(manifest_path) != records["manifest"]:
        raise ValueError(f"frozen student {student_seed} manifest changed")
    if file_record(parameter_path) != records["parameters"]:
        raise ValueError(f"frozen student {student_seed} parameters changed")
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema") != development.MODEL_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("identity")
        != development.model_identity(TEACHER_GROUP, student_seed)
        or manifest.get("parameter_file") != records["parameters"]
    ):
        raise ValueError(f"frozen student {student_seed} is internally invalid")


def assert_split_integrity() -> None:
    if len(EVENT_SEEDS) != len(set(EVENT_SEEDS)):
        raise ValueError("cross-student event seeds are not unique")
    prior = {
        *development.TRAIN_EVENT_SEEDS,
        *development.DAGGER_EVENT_SEEDS,
        *development.SUPERVISED_VALIDATION_EVENT_SEEDS,
        *development.CONTROL_VALIDATION_EVENT_SEEDS,
        *development.AUDIT_EVENT_SEEDS,
        *parent.frozen.AUDIT_EVENT_SEEDS,
        *parent.SCREEN_EVENT_SEEDS,
        *parent.AUDIT_EVENT_SEEDS,
        102_301, 102_331, 102_367, 102_397, 102_451,
    }
    if set(EVENT_SEEDS) & prior:
        raise ValueError("cross-student audit reused an earlier event seed")


assert_split_integrity()
