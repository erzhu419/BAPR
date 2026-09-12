"""Protocol and causal gate for the frozen mode-head transient fallback."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_policy_distillation_control_confirmation_v2 as frozen,
)
from jax_experiments.common.causal_fallback import (
    FallbackConfig,
    FallbackState,
    evidence_margin,
    initial_fallback_state,
    update_fallback_state,
)


ROOT = frozen.ROOT
PROTOCOL_VERSION = "v1-causal-evidence-robust-fallback"
ENV = frozen.ENV
FAMILY = frozen.FAMILY
MODES = frozen.MODES
REDUCTION = frozen.REDUCTION

TEACHER_GROUP = frozen.TEACHER_GROUP
STUDENT_SEED = frozen.STUDENT_SEED

# Development selection and final audit are disjoint from every earlier
# training, development, confirmation, and mechanism-diagnostic split.
SCREEN_EVENT_SEEDS = (103_001, 103_019, 103_037, 103_061)
AUDIT_EVENT_SEEDS = (103_301, 103_331, 103_367, 103_399, 103_451)

MAX_EPISODE_STEPS = frozen.MAX_EPISODE_STEPS
DWELL_STEPS = frozen.DWELL_STEPS
SWITCHING_EPISODES = frozen.AUDIT_SWITCHING_EPISODES

ROBUST_ARM = frozen.FIXED_STRONGEST_ROBUST_ARM
LEARNED_ARM = "student_learned"
ORACLE_ARM = "student_oracle"
BASELINE_ARMS = (ROBUST_ARM, LEARNED_ARM, ORACLE_ARM)


FALLBACK_CONFIGS = tuple(
    FallbackConfig(
        name=f"evidence_{str(threshold).replace('.', 'p')}_k{stable_steps}",
        contradiction_threshold=threshold,
        stable_steps=stable_steps,
    )
    for stable_steps in (1, 3)
    for threshold in (0.5, 1.0, 2.0, 4.0)
)
FALLBACK_CONFIG_BY_NAME = {config.name: config for config in FALLBACK_CONFIGS}
SCREEN_ARMS = (*BASELINE_ARMS, *(config.name for config in FALLBACK_CONFIGS))


SCREEN_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_policy_distillation_transient_fallback_screen_v1"
)
SELECTION_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_policy_distillation_transient_fallback_selection_v1"
)
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_policy_distillation_transient_fallback_audit_v1"
)
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_policy_distillation_transient_fallback_analysis_v1"
)
REPORT = (
    ROOT / "reports"
    / "regime_polarity_policy_distillation_transient_fallback_v1_protocol_2026-08-02.md"
)

SCREEN_SCHEMA = "bapr.regime-polarity-transient-fallback-screen.v1"
SCREEN_EVENT_SCHEMA = "bapr.regime-polarity-transient-fallback-screen-event.v1"
SELECTION_SCHEMA = "bapr.regime-polarity-transient-fallback-selection.v1"
AUDIT_SCHEMA = "bapr.regime-polarity-transient-fallback-audit.v1"
AUDIT_EVENT_SCHEMA = "bapr.regime-polarity-transient-fallback-audit-event.v1"
ANALYSIS_SCHEMA = "bapr.regime-polarity-transient-fallback-analysis.v1"

file_record = frozen.file_record
read_json = frozen.read_json
write_json_atomic = frozen.write_json_atomic
write_text_atomic = frozen.write_text_atomic
source_bundle_dirs = frozen.source_bundle_dirs
source_required_paths = frozen.source_required_paths
model_dir = frozen.model_dir
model_path = frozen.model_path
model_manifest = frozen.model_manifest
validate_frozen_candidate = frozen.validate_frozen_candidate

FROZEN_STUDENT_MANIFEST_RECORD = frozen.FROZEN_STUDENT_MANIFEST_RECORD
FROZEN_STUDENT_PARAMETER_RECORD = frozen.FROZEN_STUDENT_PARAMETER_RECORD
FROZEN_SELECTION_ANALYSIS_RECORD = frozen.FROZEN_SELECTION_ANALYSIS_RECORD


def require_screen_event_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in SCREEN_EVENT_SEEDS:
        raise ValueError(f"unknown fallback screen event seed {seed}")
    return seed


def require_audit_event_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in AUDIT_EVENT_SEEDS:
        raise ValueError(f"unknown fallback audit event seed {seed}")
    return seed


def require_config(name: str) -> FallbackConfig:
    try:
        return FALLBACK_CONFIG_BY_NAME[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown fallback config {name!r}") from exc


def screen_dir(event_seed: int) -> Path:
    return SCREEN_ROOT / f"event_seed_{require_screen_event_seed(event_seed)}"


def screen_manifest(event_seed: int) -> Path:
    return screen_dir(event_seed) / "screen_manifest.json"


def selection_json() -> Path:
    return SELECTION_ROOT / "selection.json"


def selection_manifest() -> Path:
    return SELECTION_ROOT / "selection_manifest.json"


def audit_dir(event_seed: int) -> Path:
    return AUDIT_ROOT / f"event_seed_{require_audit_event_seed(event_seed)}"


def audit_manifest(event_seed: int) -> Path:
    return audit_dir(event_seed) / "audit_manifest.json"


def analysis_json() -> Path:
    return ANALYSIS_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return ANALYSIS_ROOT / "analysis.md"


def common_identity(role: str, event_seed: int) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": role,
        "development_only": True,
        "env": ENV,
        "family": FAMILY,
        "variant": TEACHER_GROUP,
        "student_seed": STUDENT_SEED,
        "event_seed": int(event_seed),
        "frozen_student_manifest": FROZEN_STUDENT_MANIFEST_RECORD,
        "frozen_student_parameters": FROZEN_STUDENT_PARAMETER_RECORD,
        "frozen_selection_analysis": FROZEN_SELECTION_ANALYSIS_RECORD,
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


def assert_split_integrity() -> None:
    all_new = (*SCREEN_EVENT_SEEDS, *AUDIT_EVENT_SEEDS)
    if len(all_new) != len(set(all_new)):
        raise ValueError("transient-fallback splits overlap")
    prior = {
        *frozen.development.TRAIN_EVENT_SEEDS,
        *frozen.development.DAGGER_EVENT_SEEDS,
        *frozen.development.SUPERVISED_VALIDATION_EVENT_SEEDS,
        *frozen.development.CONTROL_VALIDATION_EVENT_SEEDS,
        *frozen.development.AUDIT_EVENT_SEEDS,
        *frozen.AUDIT_EVENT_SEEDS,
        102_301, 102_331, 102_367, 102_397, 102_451,
    }
    if set(all_new) & prior:
        raise ValueError("transient-fallback protocol reused an earlier event")


assert_split_integrity()
