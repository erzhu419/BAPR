"""Checkpoint-only context causality and delayed-oracle protocol."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from jax_experiments.analysis import regime_polarity_headroom as base


PROTOCOL_VERSION = "v1"
SIGNATURE_FAMILY = "polarity-context-delay"
ENVS = ("HalfCheetah-v2", "Ant-v2")
DELAY_STEPS = (1, 5, 10, 25, 50)


@dataclass(frozen=True)
class EvaluationCase:
    label: str
    source_role: str
    context_kind: str
    fixed_mode_id: int | None = None
    delay_steps: int | None = None
    stationary: bool = True


FULL_CASES = (
    EvaluationCase("robust_model", "robust", "checkpoint"),
    EvaluationCase("true", "oracle", "true"),
    EvaluationCase("zero", "oracle", "zero"),
    *(EvaluationCase(
        f"fixed_{mode}", "oracle", "fixed", fixed_mode_id=mode)
      for mode in base.MODES),
    EvaluationCase("cyclic", "oracle", "cyclic"),
    EvaluationCase("shuffled", "oracle", "shuffled"),
)
DELAY_CASES = tuple(
    EvaluationCase(
        f"delayed_{delay}", "oracle", "delayed",
        delay_steps=delay, stationary=False)
    for delay in DELAY_STEPS
)
EVALUATION_CASES = (*FULL_CASES, *DELAY_CASES)
CASE_LABELS = tuple(case.label for case in EVALUATION_CASES)
FULL_CASE_LABELS = tuple(case.label for case in FULL_CASES)
DELAY_CASE_LABELS = tuple(case.label for case in DELAY_CASES)
FIXED_LABELS = tuple(f"fixed_{mode}" for mode in base.MODES)

COMMON_OUTPUT_FILES = (
    "summary.csv", "switching_returns.csv", "switching_trace.csv")
STATIONARY_OUTPUT_FILES = ("task_returns.csv",)

AUDIT_ROOT = (
    base.ROOT / "jax_experiments"
    / "results_regime_polarity_context_delay_audit_v1")
ANALYSIS_ROOT = (
    base.ROOT / "jax_experiments"
    / "results_regime_polarity_context_delay_analysis_v1")
REPORT = (
    base.ROOT / "reports"
    / "regime_polarity_context_delay_protocol_2026-07-27.md")

AUDIT_SCHEMA = "bapr.regime-polarity-context-delay-audit.v1"
ANALYSIS_SCHEMA = "bapr.regime-polarity-context-delay-analysis.v1"


def require_env(env: str) -> str:
    env = str(env)
    if env not in ENVS:
        raise ValueError(
            f"unknown context-delay environment {env!r}; expected {ENVS}")
    return env


def require_case(label: str) -> str:
    label = str(label)
    if label not in CASE_LABELS:
        raise ValueError(
            f"unknown context-delay case {label!r}; expected {CASE_LABELS}")
    return label


def evaluation_case(label: str) -> EvaluationCase:
    label = require_case(label)
    return next(case for case in EVALUATION_CASES if case.label == label)


def shuffled_mode_map(event_seed: int) -> tuple[int, ...]:
    """Return a deterministic derangement for one paired event stream."""
    event_seed = base.require_event_seed(event_seed)
    modes = list(base.MODES)
    rng = random.Random(event_seed + 27_072_026)
    while True:
        shuffled = modes.copy()
        rng.shuffle(shuffled)
        if all(left != right for left, right in zip(modes, shuffled)):
            return tuple(shuffled)


def output_files(case: EvaluationCase | str) -> tuple[str, ...]:
    if isinstance(case, str):
        case = evaluation_case(case)
    if case.stationary:
        return (*COMMON_OUTPUT_FILES, *STATIONARY_OUTPUT_FILES)
    return COMMON_OUTPUT_FILES


def audit_dir(env: str, seed: int, event_seed: int) -> Path:
    return (
        AUDIT_ROOT / base.env_slug(require_env(env))
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
    event_seed = base.require_event_seed(event_seed)
    shuffled = shuffled_mode_map(event_seed)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "family": SIGNATURE_FAMILY,
        "source_protocol": base.PROTOCOL_VERSION,
        "env": require_env(env),
        "training_seed": base.require_training_seed(seed),
        "event_seed": event_seed,
        "shuffled_mode_map": list(shuffled),
        "evaluation_cases": [
            {
                "label": case.label,
                "source_role": case.source_role,
                "context_kind": case.context_kind,
                "fixed_mode_id": case.fixed_mode_id,
                "delay_steps": case.delay_steps,
                "stationary": case.stationary,
            }
            for case in EVALUATION_CASES
        ],
    }


def all_audit_manifests() -> tuple[Path, ...]:
    return tuple(
        audit_manifest(env, seed, event_seed)
        for env in ENVS
        for seed in base.TRAINING_SEEDS
        for event_seed in base.AUDIT_EVENT_SEEDS
    )
