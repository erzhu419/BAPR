"""Frozen event-grouped checkpoint-only cross-context protocol."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jax_experiments.analysis import regime_control_headroom as base


PROTOCOL_VERSION = "v1"
SIGNATURE_FAMILY = "event-grouped-cross-context"


@dataclass(frozen=True)
class EvaluationCase:
    label: str
    source_role: str
    context_kind: str
    fixed_mode_id: int | None = None


ORACLE_CONTEXT_CASES = (
    EvaluationCase("true", "oracle", "true"),
    EvaluationCase("zero", "oracle", "zero"),
    *(EvaluationCase(
        f"fixed_{mode}", "oracle", "fixed", mode)
      for mode in base.MODES),
    EvaluationCase("cyclic", "oracle", "cyclic"),
)
EVALUATION_CASES = (
    EvaluationCase("robust_model", "robust", "checkpoint"),
    *ORACLE_CONTEXT_CASES,
)
CASE_LABELS = tuple(case.label for case in EVALUATION_CASES)
ORACLE_CONTEXT_LABELS = tuple(
    case.label for case in ORACLE_CONTEXT_CASES)

AUDIT_ROOT = (
    base.ROOT / "jax_experiments"
    / "results_regime_cross_context_audit_v1")
ANALYSIS_ROOT = (
    base.ROOT / "jax_experiments"
    / "results_regime_cross_context_analysis_v1")
REPORT = (
    base.ROOT / "reports"
    / "regime_cross_context_diagnostic_2026-07-22.md")

AUDIT_SCHEMA = "bapr.regime-cross-context-event-audit.v1"
ANALYSIS_SCHEMA = "bapr.regime-cross-context-analysis.v1"


def require_case(label: str) -> str:
    label = str(label)
    if label not in CASE_LABELS:
        raise ValueError(
            f"unknown cross-context case {label!r}; expected "
            f"{CASE_LABELS}")
    return label


def evaluation_case(label: str) -> EvaluationCase:
    label = require_case(label)
    return next(case for case in EVALUATION_CASES if case.label == label)


def audit_dir(env: str, seed: int, event_seed: int) -> Path:
    return (
        AUDIT_ROOT / base.env_slug(env)
        / f"seed_{base.require_training_seed(seed)}"
        / f"event_seed_{base.require_event_seed(event_seed)}")


def case_dir(
        env: str, seed: int, event_seed: int, case: str) -> Path:
    return audit_dir(env, seed, event_seed) / require_case(case)


def audit_manifest(env: str, seed: int, event_seed: int) -> Path:
    return audit_dir(env, seed, event_seed) / "audit_manifest.json"


def analysis_json() -> Path:
    return ANALYSIS_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return ANALYSIS_ROOT / "analysis.md"


def identity(
        env: str, seed: int, event_seed: int) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "family": SIGNATURE_FAMILY,
        "source_protocol": base.PROTOCOL_VERSION,
        "env": base.require_env(env),
        "training_seed": base.require_training_seed(seed),
        "event_seed": base.require_event_seed(event_seed),
        "algo": "regime_sac",
        "evaluation_cases": [
            {
                "label": case.label,
                "source_role": case.source_role,
                "context_kind": case.context_kind,
                "fixed_mode_id": case.fixed_mode_id,
            }
            for case in EVALUATION_CASES
        ],
    }


def all_audit_manifests() -> tuple[Path, ...]:
    return tuple(
        audit_manifest(env, seed, event_seed)
        for env in base.ENVS
        for seed in base.TRAINING_SEEDS
        for event_seed in base.AUDIT_EVENT_SEEDS
    )
