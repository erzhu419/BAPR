"""Frozen delayed-oracle causal-ceiling diagnostic for transient fallback."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_policy_distillation_transient_fallback_cross_student_v1 as parent,
)


ROOT = parent.ROOT
PROTOCOL_VERSION = "v1-frozen-delayed-oracle-causal-ceiling"
ENV = parent.ENV
FAMILY = parent.FAMILY
TEACHER_GROUP = parent.TEACHER_GROUP
STUDENT_SEEDS = parent.STUDENT_SEEDS
EVENT_SEEDS = (105_301, 105_331, 105_367, 105_399, 105_451)

ROBUST_ARM = parent.ROBUST_ARM
LEARNED_ARM = parent.LEARNED_ARM
ORACLE_ARM = parent.ORACLE_ARM
FALLBACK_ARM = parent.SELECTED_CONFIG_NAME
DELAY_STEPS = (1, 2, 5, 10)
DELAY_ARMS = tuple(f"student_oracle_delay_{delay}" for delay in DELAY_STEPS)
ARMS = (
    ROBUST_ARM,
    LEARNED_ARM,
    ORACLE_ARM,
    *DELAY_ARMS,
    FALLBACK_ARM,
)

FROZEN_CROSS_STUDENT_ANALYSIS_RECORD = {
    "sha256": "346390d2019a3f33f923df87ce7a11d864a04eee2ba53992e6e56ad44b716620",
    "size": 11306,
}

AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_policy_distillation_transient_causal_ceiling_audit_v1"
)
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_policy_distillation_transient_causal_ceiling_analysis_v1"
)
REPORT = (
    ROOT / "reports"
    / "regime_polarity_policy_distillation_transient_causal_ceiling_v1_protocol_2026-08-02.md"
)

AUDIT_SCHEMA = "bapr.regime-polarity-transient-causal-ceiling-audit.v1"
EVENT_SCHEMA = "bapr.regime-polarity-transient-causal-ceiling-event.v1"
ANALYSIS_SCHEMA = "bapr.regime-polarity-transient-causal-ceiling-analysis.v1"

file_record = parent.file_record
read_json = parent.read_json
write_json_atomic = parent.write_json_atomic
write_text_atomic = parent.write_text_atomic


def require_student_seed(seed: int) -> int:
    return parent.require_student_seed(seed)


def require_event_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in EVENT_SEEDS:
        raise ValueError(f"unknown causal-ceiling event seed {seed}")
    return seed


def delay_arm(delay: int) -> str:
    delay = int(delay)
    if delay not in DELAY_STEPS:
        raise ValueError(f"unknown delayed-oracle step count {delay}")
    return f"student_oracle_delay_{delay}"


def arm_delay(arm: str) -> int | None:
    arm = str(arm)
    if arm not in DELAY_ARMS:
        return None
    return int(arm.rsplit("_", 1)[1])


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
        "benchmark_role": "post_fallback_delayed_oracle_causal_ceiling",
        "development_only": True,
        "selection_forbidden": True,
        "env": ENV,
        "family": FAMILY,
        "variant": TEACHER_GROUP,
        "student_seed": student_seed,
        "event_seed": require_event_seed(event_seed),
        "arms": list(ARMS),
        "delay_semantics": (
            "old true context remains active for exactly delay actions after "
            "each hidden physical-mode switch; the new true context becomes "
            "available only after those transitions"
        ),
        "frozen_student_manifest":
            parent.student_records(student_seed)["manifest"],
        "frozen_student_parameters":
            parent.student_records(student_seed)["parameters"],
        "frozen_cross_student_analysis":
            FROZEN_CROSS_STUDENT_ANALYSIS_RECORD,
        "deployable_online_inputs": [
            "observation",
            "commanded_action",
            "reward",
            "next_observation",
            "causal_mode_posterior",
            "one_step_mode_log_likelihood",
        ],
        "diagnostic_only_inputs": ["mode_id_after_registered_delay"],
        "forbidden_online_inputs_for_deployable_arms": [
            "mode_id",
            "action_gain",
            "executed_action",
            "switch_clock",
        ],
    }


def validate_upstream(student_seed: int) -> None:
    student_seed = require_student_seed(student_seed)
    parent.validate_upstream(student_seed)
    if file_record(parent.analysis_json()) \
            != FROZEN_CROSS_STUDENT_ANALYSIS_RECORD:
        raise ValueError("frozen cross-student analysis changed")
    analysis = read_json(parent.analysis_json())
    if (
        analysis.get("schema") != parent.ANALYSIS_SCHEMA
        or analysis.get("status") != "complete"
        or analysis.get("cross_student_pass") is not False
        or int(analysis.get("student_passes", -1)) != 1
    ):
        raise ValueError("cross-student failure diagnosis is not frozen")


def assert_split_integrity() -> None:
    if len(EVENT_SEEDS) != len(set(EVENT_SEEDS)):
        raise ValueError("causal-ceiling event seeds are not unique")
    prior = {
        *parent.development.TRAIN_EVENT_SEEDS,
        *parent.development.DAGGER_EVENT_SEEDS,
        *parent.development.SUPERVISED_VALIDATION_EVENT_SEEDS,
        *parent.development.CONTROL_VALIDATION_EVENT_SEEDS,
        *parent.development.AUDIT_EVENT_SEEDS,
        *parent.parent.SCREEN_EVENT_SEEDS,
        *parent.parent.AUDIT_EVENT_SEEDS,
        *parent.EVENT_SEEDS,
        100_019, 100_043, 100_069, 100_103, 100_151,
        102_301, 102_331, 102_367, 102_397, 102_451,
    }
    if set(EVENT_SEEDS) & prior:
        raise ValueError("causal-ceiling audit reused an earlier event seed")


assert_split_integrity()
