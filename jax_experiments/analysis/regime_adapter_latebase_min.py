"""Protocol constants for the late-base min-target adapter diagnostic."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from jax_experiments.analysis import regime_adapter_fork as source


ROOT = Path(os.environ.get(
    "BAPR_WORKSPACE_ROOT", Path(__file__).resolve().parents[2])).resolve()
PROTOCOL_VERSION = "v1"
ENV = source.ENV
FAMILY = source.FAMILY
MODES = source.MODES
TRAINING_SEEDS = (24, 32)
EVENT_SEEDS = (79100, 79200, 79300, 79400, 79500)
DELTA = 0.5
DWELL_STEPS = source.DWELL_STEPS
EPISODES_PER_TASK = source.EPISODES_PER_TASK
SWITCHING_EPISODES = source.SWITCHING_EPISODES

SOURCE_NEXT_ITERATION = source.ROBUST_FINAL_NEXT_ITERATION
SOURCE_TOTAL_STEPS = source.ROBUST_FINAL_TOTAL_STEPS
SOURCE_UPDATE_COUNT = source.ROBUST_FINAL_UPDATE_COUNT
SAMPLES_PER_ITER = source.SAMPLES_PER_ITER
UPDATES_PER_ITER = source.UPDATES_PER_ITER
ADDITIONAL_ITERS = source.ADAPTER_EXTRA_ITERS_PER_MODE
FINAL_NEXT_ITERATION = SOURCE_NEXT_ITERATION + ADDITIONAL_ITERS
FINAL_TOTAL_STEPS = (
    SOURCE_TOTAL_STEPS + ADDITIONAL_ITERS * SAMPLES_PER_ITER)
FINAL_UPDATE_COUNT = (
    SOURCE_UPDATE_COUNT + ADDITIONAL_ITERS * UPDATES_PER_ITER)
PER_CONTROLLER_POST_FORK_STEPS = (
    FINAL_TOTAL_STEPS - SOURCE_TOTAL_STEPS)
BANK_AGGREGATE_POST_FORK_STEPS = (
    len(MODES) * PER_CONTROLLER_POST_FORK_STEPS)
BANK_AGGREGATE_TOTAL_STEPS = (
    SOURCE_TOTAL_STEPS + BANK_AGGREGATE_POST_FORK_STEPS)

CANONICAL_ROOT = (
    ROOT / "jax_experiments"
    / "eval_bundles_regime_adapter_latebase_min_canonical_v1")
RUN_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_adapter_latebase_min_v1")
BUNDLE_ROOT = (
    ROOT / "jax_experiments"
    / "eval_bundles_regime_adapter_latebase_min_v1")
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_adapter_latebase_min_audit_v1")
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_adapter_latebase_min_analysis_v1")
PROTOCOL_REPORT = (
    ROOT / "reports"
    / "regime_adapter_latebase_min_protocol_2026-07-26.md")

CANONICAL_BOOTSTRAP_NAME = "latebase_canonical_bootstrap.json"
CANONICAL_MANIFEST_NAME = "canonical_manifest.json"
BRANCH_BOOTSTRAP_NAME = "latebase_branch_bootstrap.json"
BUNDLE_MANIFEST_NAME = "bundle_manifest.json"
AUDIT_MANIFEST_NAME = "audit_manifest.json"

CANONICAL_SCHEMA = "bapr.regime-adapter-latebase-canonical.v1"
BUNDLE_SCHEMA = "bapr.regime-adapter-latebase-bundle.v1"
AUDIT_SCHEMA = "bapr.regime-adapter-latebase-audit.v1"
ANALYSIS_SCHEMA = "bapr.regime-adapter-latebase-analysis.v1"


def require_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in TRAINING_SEEDS:
        raise ValueError(f"unknown late-base seed {seed}")
    return seed


def require_mode(mode: int) -> int:
    return source.require_mode(mode)


def source_bundle_dir(seed: int) -> Path:
    return source.robust_bundle_dir(require_seed(seed))


def canonical_dir(seed: int) -> Path:
    return CANONICAL_ROOT / f"seed_{require_seed(seed)}"


def canonical_manifest(seed: int) -> Path:
    return canonical_dir(seed) / CANONICAL_MANIFEST_NAME


def run_dir(seed: int, mode: int) -> Path:
    return (
        RUN_ROOT / f"seed_{require_seed(seed)}"
        / f"mode_{require_mode(mode)}")


def adapter_bundle_dir(seed: int, delta: float, mode: int) -> Path:
    if float(delta) != DELTA:
        raise ValueError(f"late-base delta is frozen at {DELTA}")
    return (
        BUNDLE_ROOT / f"seed_{require_seed(seed)}"
        / f"mode_{require_mode(mode)}")


def bundle_manifest(seed: int, mode: int) -> Path:
    return (
        adapter_bundle_dir(seed, DELTA, mode)
        / BUNDLE_MANIFEST_NAME)


def audit_dir(seed: int, event_seed: int) -> Path:
    seed = require_seed(seed)
    event_seed = int(event_seed)
    if event_seed not in EVENT_SEEDS:
        raise ValueError(f"unknown late-base event seed {event_seed}")
    return AUDIT_ROOT / f"seed_{seed}" / f"event_{event_seed}"


def audit_manifest(seed: int, event_seed: int) -> Path:
    return audit_dir(seed, event_seed) / AUDIT_MANIFEST_NAME


def analysis_json() -> Path:
    return ANALYSIS_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return ANALYSIS_ROOT / "analysis.md"


def canonical_identity(seed: int) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "env": ENV,
        "family": FAMILY,
        "training_seed": require_seed(seed),
        "source_next_iteration": SOURCE_NEXT_ITERATION,
        "source_total_steps": SOURCE_TOTAL_STEPS,
        "critic_target_mode": "min",
        "freeze_alpha": True,
        "canonical_initialization": True,
    }


def identity(seed: int, mode: int) -> dict[str, Any]:
    return {
        **canonical_identity(seed),
        "mode": require_mode(mode),
        "residual_delta": DELTA,
        "final_next_iteration": FINAL_NEXT_ITERATION,
        "diagnostic_only": True,
        "compute_matched": False,
    }


def expected_checkpoint() -> dict[str, int | str]:
    return {
        "iteration": FINAL_NEXT_ITERATION - 1,
        "next_iteration": FINAL_NEXT_ITERATION,
        "total_steps": FINAL_TOTAL_STEPS,
        "update_count": FINAL_UPDATE_COUNT,
        "algo": "bapr_regime",
    }


def required_canonical_paths(seed: int) -> tuple[Path, ...]:
    directory = canonical_dir(seed)
    return (
        directory / CANONICAL_MANIFEST_NAME,
        directory / "checkpoints" / "params.pkl",
        directory / "checkpoints" / "train_state.pkl",
        directory / "checkpoints" / "replay_buffer.npz",
        directory / "checkpoints" / CANONICAL_BOOTSTRAP_NAME,
    )


def required_bundle_paths(seed: int) -> tuple[Path, ...]:
    return tuple(
        path
        for mode in MODES
        for path in (
            bundle_manifest(seed, mode),
            adapter_bundle_dir(seed, DELTA, mode)
            / "checkpoints" / "params.pkl",
            adapter_bundle_dir(seed, DELTA, mode)
            / "checkpoints" / "train_state.pkl",
            adapter_bundle_dir(seed, DELTA, mode)
            / "checkpoints" / BRANCH_BOOTSTRAP_NAME,
            adapter_bundle_dir(seed, DELTA, mode)
            / "logs" / "protocol_signature.json",
        )
    )


read_json = source.read_json
write_json_atomic = source.write_json_atomic
file_record = source.file_record
