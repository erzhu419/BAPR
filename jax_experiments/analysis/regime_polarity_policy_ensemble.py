"""Checkpoint-only policy-ensemble diagnostic for polarity controllers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jax_experiments.analysis import regime_polarity_confirmation as development
from jax_experiments.analysis import (
    regime_polarity_final_confirmation as final,
)


ROOT = development.ROOT
PROTOCOL_VERSION = "v1-controller-variance-diagnostic"
ENV = "HalfCheetah-v2"
FAMILY = development.FAMILY
MODES = development.MODES
GROUPS = ("development", "final")
CONTROLLER_SEEDS = {
    "development": development.TRAINING_SEEDS,
    "final": final.TRAINING_SEEDS,
}
# Fresh relative to all polarity development and final-confirmation audits.
EVENT_SEEDS = (97_001, 97_002, 97_003)

DWELL_STEPS = development.DWELL_STEPS
MAX_EPISODE_STEPS = development.MAX_EPISODE_STEPS
EPISODES_PER_TASK = development.EPISODES_PER_TASK
SWITCHING_EPISODES = development.SWITCHING_EPISODES

AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_policy_ensemble_audit_v1")
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_policy_ensemble_analysis_v1")
REPORT = (
    ROOT / "reports"
    / "regime_polarity_policy_ensemble_diagnostic_2026-08-01.md")

AUDIT_SCHEMA = "bapr.regime-polarity-policy-ensemble-audit.v1"
EVENT_SCHEMA = "bapr.regime-polarity-policy-ensemble-event.v1"
ANALYSIS_SCHEMA = "bapr.regime-polarity-policy-ensemble-analysis.v1"

file_record = development.file_record
read_json = development.read_json
write_json_atomic = development.write_json_atomic
write_text_atomic = development.write_text_atomic


@dataclass(frozen=True)
class ArmSpec:
    label: str
    source_role: str
    context_kind: str
    reduction: str
    controller_seed: int | None = None


def require_group(group: str) -> str:
    group = str(group)
    if group not in GROUPS:
        raise ValueError(f"unknown controller group {group!r}; expected {GROUPS}")
    return group


def require_event_seed(event_seed: int) -> int:
    event_seed = int(event_seed)
    if event_seed not in EVENT_SEEDS:
        raise ValueError(
            f"unknown ensemble event seed {event_seed}; expected {EVENT_SEEDS}")
    return event_seed


def source_protocol(group: str):
    return development if require_group(group) == "development" else final


def controller_seeds(group: str) -> tuple[int, ...]:
    return tuple(CONTROLLER_SEEDS[require_group(group)])


def arms(group: str) -> tuple[ArmSpec, ...]:
    seeds = controller_seeds(group)
    return (
        *(ArmSpec(
            f"robust_seed_{seed}", "robust", "zero", "individual", seed)
          for seed in seeds),
        *(ArmSpec(
            f"oracle_seed_{seed}", "oracle", "true", "individual", seed)
          for seed in seeds),
        ArmSpec("robust_mean", "robust", "zero", "mean"),
        ArmSpec("robust_median", "robust", "zero", "median"),
        ArmSpec("oracle_mean", "oracle", "true", "mean"),
        ArmSpec("oracle_median", "oracle", "true", "median"),
        ArmSpec("learned_mean", "oracle", "learned", "mean"),
        ArmSpec("learned_median", "oracle", "learned", "median"),
    )


def arm_labels(group: str) -> tuple[str, ...]:
    return tuple(arm.label for arm in arms(group))


def arm_spec(group: str, label: str) -> ArmSpec:
    label = str(label)
    for candidate in arms(group):
        if candidate.label == label:
            return candidate
    raise ValueError(f"unknown ensemble arm {label!r} for group {group!r}")


def audit_dir(group: str, event_seed: int) -> Path:
    return (
        AUDIT_ROOT / require_group(group)
        / f"event_seed_{require_event_seed(event_seed)}")


def audit_manifest(group: str, event_seed: int) -> Path:
    return audit_dir(group, event_seed) / "audit_manifest.json"


def event_result(group: str, event_seed: int) -> Path:
    return audit_dir(group, event_seed) / "results.json"


def analysis_json() -> Path:
    return ANALYSIS_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return ANALYSIS_ROOT / "analysis.md"


def all_audit_manifests() -> tuple[Path, ...]:
    return tuple(
        audit_manifest(group, event_seed)
        for group in GROUPS
        for event_seed in EVENT_SEEDS
    )


def source_bundle_manifests(group: str) -> tuple[Path, ...]:
    source = source_protocol(group)
    return tuple(
        source.bundle_manifest(ENV, role, seed)
        for role in source.ROLES
        for seed in controller_seeds(group)
    )


def source_bundle_dirs(group: str) -> tuple[Path, ...]:
    source = source_protocol(group)
    return tuple(
        source.bundle_dir(ENV, role, seed)
        for role in source.ROLES
        for seed in controller_seeds(group)
    )


def source_required_paths(group: str) -> tuple[Path, ...]:
    source = source_protocol(group)
    return tuple(
        path
        for role in source.ROLES
        for seed in controller_seeds(group)
        for path in source.bundle_required_paths(ENV, role, seed)
    )


def validate_frozen_estimator() -> None:
    final.validate_frozen_estimator()


def identity(group: str, event_seed: int) -> dict[str, Any]:
    group = require_group(group)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "retrospective_controller_variance_diagnostic",
        "development_only": True,
        "env": ENV,
        "family": FAMILY,
        "controller_group": group,
        "controller_seeds": list(controller_seeds(group)),
        "event_seed": require_event_seed(event_seed),
        "arms": [arm.__dict__ for arm in arms(group)],
        "online_inputs": [
            "observation",
            "commanded_action",
            "next_observation",
        ],
        "online_forbidden_for_learned_arms": [
            "mode_id",
            "action_gain",
            "executed_action",
            "switch_clock",
        ],
    }
