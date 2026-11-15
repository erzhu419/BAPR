"""Protocol for the compute-unmatched equal-per-controller adapter bound."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from jax_experiments.analysis import regime_adapter_confirmation as confirmation
from jax_experiments.analysis import regime_adapter_fork as source


ROOT = Path(os.environ.get(
    "BAPR_WORKSPACE_ROOT", Path(__file__).resolve().parents[2])).resolve()
PROTOCOL_VERSION = "v1"
ENV = source.ENV
FAMILY = source.FAMILY
MODES = source.MODES
DELTA = confirmation.DELTA
TRAINING_SEEDS = confirmation.HOLDOUT_SEEDS
EVENT_SEEDS = (77100, 77200, 77300, 77400, 77500)
DWELL_STEPS = source.DWELL_STEPS
EPISODES_PER_TASK = source.EPISODES_PER_TASK
SWITCHING_EPISODES = source.SWITCHING_EPISODES

SOURCE_NEXT_ITERATION = source.SOURCE_NEXT_ITERATION
SOURCE_TOTAL_STEPS = source.SOURCE_TOTAL_STEPS
SOURCE_UPDATE_COUNT = source.SOURCE_UPDATE_COUNT
FINAL_NEXT_ITERATION = source.ROBUST_FINAL_NEXT_ITERATION
FINAL_TOTAL_STEPS = source.ROBUST_FINAL_TOTAL_STEPS
FINAL_UPDATE_COUNT = source.ROBUST_FINAL_UPDATE_COUNT
ADDITIONAL_ITERS = FINAL_NEXT_ITERATION - SOURCE_NEXT_ITERATION
ADDITIONAL_STEPS = FINAL_TOTAL_STEPS - SOURCE_TOTAL_STEPS
PER_CONTROLLER_POST_FORK_STEPS = (
    FINAL_TOTAL_STEPS - source.SOURCE_TOTAL_STEPS)
BANK_AGGREGATE_POST_FORK_STEPS = (
    len(MODES) * PER_CONTROLLER_POST_FORK_STEPS)
BANK_AGGREGATE_TOTAL_STEPS = (
    source.SOURCE_TOTAL_STEPS + BANK_AGGREGATE_POST_FORK_STEPS)

RUN_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_adapter_equal_controller_v1")
BUNDLE_ROOT = (
    ROOT / "jax_experiments"
    / "eval_bundles_regime_adapter_equal_controller_v1")
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_adapter_equal_controller_audit_v1")
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_adapter_equal_controller_analysis_v1")
PROTOCOL_REPORT = (
    ROOT / "reports"
    / "regime_adapter_equal_controller_protocol_2026-07-26.md")

EXTENSION_BOOTSTRAP_NAME = "equal_controller_extension.json"
BUNDLE_MANIFEST_NAME = "bundle_manifest.json"
BUNDLE_SCHEMA = "bapr.regime-adapter-equal-controller-bundle.v1"
AUDIT_MANIFEST_NAME = "audit_manifest.json"
AUDIT_SCHEMA = "bapr.regime-adapter-equal-controller-audit.v1"
ANALYSIS_SCHEMA = "bapr.regime-adapter-equal-controller-analysis.v1"


def require_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in TRAINING_SEEDS:
        raise ValueError(f"unknown equal-controller seed {seed}")
    return seed


def require_mode(mode: int) -> int:
    return source.require_mode(mode)


def source_bundle_dir(seed: int, mode: int) -> Path:
    require_mode(mode)
    return source.source_bundle_dir(require_seed(seed))


def robust_bundle_dir(seed: int) -> Path:
    return source.robust_bundle_dir(require_seed(seed))


def run_dir(seed: int, mode: int) -> Path:
    return (
        RUN_ROOT / f"seed_{require_seed(seed)}"
        / f"mode_{require_mode(mode)}")


def adapter_bundle_dir(
        seed: int, delta: float, mode: int) -> Path:
    if float(delta) != DELTA:
        raise ValueError(f"equal-controller delta is frozen at {DELTA}")
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
        raise ValueError(f"unknown equal-controller event seed {event_seed}")
    return AUDIT_ROOT / f"seed_{seed}" / f"event_{event_seed}"


def audit_manifest(seed: int, event_seed: int) -> Path:
    return audit_dir(seed, event_seed) / AUDIT_MANIFEST_NAME


def analysis_json() -> Path:
    return ANALYSIS_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return ANALYSIS_ROOT / "analysis.md"


def identity(seed: int, mode: int) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "env": ENV,
        "family": FAMILY,
        "training_seed": require_seed(seed),
        "mode": require_mode(mode),
        "residual_delta": DELTA,
        "source_next_iteration": SOURCE_NEXT_ITERATION,
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
            / "checkpoints" / source.BOOTSTRAP_NAME,
            adapter_bundle_dir(seed, DELTA, mode)
            / "checkpoints" / EXTENSION_BOOTSTRAP_NAME,
            adapter_bundle_dir(seed, DELTA, mode)
            / "logs" / "protocol_signature.json",
        )
    )


read_json = source.read_json
write_json_atomic = source.write_json_atomic
file_record = source.file_record
