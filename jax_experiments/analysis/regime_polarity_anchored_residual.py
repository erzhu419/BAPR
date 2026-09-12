"""Fresh-seed protocol for the robust-anchored polarity controller."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_final_confirmation as frozen,
)


ROOT = Path(os.environ.get(
    "BAPR_WORKSPACE_ROOT", Path(__file__).resolve().parents[2])).resolve()
PROTOCOL_VERSION = "v1"
ENV = frozen.ENV
ENVS = (ENV,)
FAMILY = frozen.FAMILY
MODES = frozen.MODES
ROLES = ("robust",)
BRANCH_ROLES = ("robust_continue", "anchored")

# Development-only seeds, sealed before this controller is trained. They are
# disjoint from all earlier 8/16/24, 101-523, and 607-1031 policy splits.
TRAINING_SEEDS = (1103, 1213, 1301)
CALIBRATION_EVENT_SEEDS = (96_001, 96_002)
AUDIT_EVENT_SEEDS = (96_101, 96_102, 96_103)
TEST_EVENT_SEEDS = AUDIT_EVENT_SEEDS

SOURCE_NEXT_ITERATION = 1400
SOURCE_FINAL_ITERATION = SOURCE_NEXT_ITERATION - 1
BRANCH_EXTRA_ITERS = 700
BRANCH_FINAL_NEXT_ITERATION = SOURCE_NEXT_ITERATION + BRANCH_EXTRA_ITERS
BRANCH_FINAL_ITERATION = BRANCH_FINAL_NEXT_ITERATION - 1
SAMPLES_PER_ITER = frozen.SAMPLES_PER_ITER
UPDATES_PER_ITER = frozen.UPDATES_PER_ITER
SOURCE_TOTAL_STEPS = SOURCE_NEXT_ITERATION * SAMPLES_PER_ITER
SOURCE_UPDATE_COUNT = SOURCE_NEXT_ITERATION * UPDATES_PER_ITER
BRANCH_TOTAL_STEPS = BRANCH_FINAL_NEXT_ITERATION * SAMPLES_PER_ITER
BRANCH_UPDATE_COUNT = BRANCH_FINAL_NEXT_ITERATION * UPDATES_PER_ITER
MAX_ITERS = SOURCE_NEXT_ITERATION
FINAL_ITERATION = SOURCE_FINAL_ITERATION
FINAL_TOTAL_STEPS = SOURCE_TOTAL_STEPS
FINAL_UPDATE_COUNT = SOURCE_UPDATE_COUNT
DWELL_STEPS = frozen.DWELL_STEPS
MAX_EPISODE_STEPS = frozen.MAX_EPISODE_STEPS
EPISODES_PER_TASK = frozen.EPISODES_PER_TASK
SWITCHING_EPISODES = frozen.SWITCHING_EPISODES

RESIDUAL_DELTA = 0.5
CONFIDENCE_THRESHOLD = 0.85
CALIBRATION_GAIN_MARGIN = 0.02
MAX_TERMINATION_GAP = 0.02
MIN_BASE_PRESERVATION = 0.95
MIN_ORACLE_RELATIVE_GAIN = 0.05
MIN_LEARNED_RELATIVE_GAIN = 0.03
MIN_SEED_WINS = 2
MIN_MODE_ACCURACY = frozen.MIN_MODE_ACCURACY
MAX_BRIER_SCORE = frozen.MAX_BRIER_SCORE
MAX_MEDIAN_SWITCH_DELAY = frozen.MAX_MEDIAN_SWITCH_DELAY
MAX_P90_SWITCH_DELAY = frozen.MAX_P90_SWITCH_DELAY

RUN_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_anchored_residual_v1")
BUNDLE_ROOT = (
    ROOT / "jax_experiments"
    / "eval_bundles_regime_polarity_anchored_residual_v1")
CALIBRATION_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_anchored_calibration_v1")
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_anchored_audit_v1")
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_anchored_analysis_v1")
PROTOCOL_REPORT = (
    ROOT / "reports"
    / "regime_polarity_anchored_residual_protocol_2026-07-29.md")

SOURCE_BUNDLE_SCHEMA = "bapr.regime-polarity-anchored-source.v1"
BUNDLE_SCHEMA = SOURCE_BUNDLE_SCHEMA
BRANCH_BUNDLE_SCHEMA = "bapr.regime-polarity-anchored-branch.v1"
CALIBRATION_SCHEMA = "bapr.regime-polarity-anchored-calibration.v1"
AUDIT_SCHEMA = "bapr.regime-polarity-anchored-audit.v1"
EVENT_SCHEMA = "bapr.regime-polarity-anchored-event.v1"
ANALYSIS_SCHEMA = "bapr.regime-polarity-anchored-analysis.v1"
BOOTSTRAP_NAME = "anchored_bootstrap.json"

MODEL_MANIFEST = frozen.MODEL_MANIFEST
MODEL_PATH = frozen.MODEL_PATH
FROZEN_MODEL_MANIFEST_RECORD = frozen.FROZEN_MODEL_MANIFEST_RECORD
FROZEN_MODEL_PARAMETER_RECORD = frozen.FROZEN_MODEL_PARAMETER_RECORD
posterior_metrics = frozen.posterior_metrics


def validate_frozen_estimator() -> None:
    frozen.validate_frozen_estimator()


def require_env(env: str) -> str:
    env = str(env)
    if env not in ENVS:
        raise ValueError(f"unknown anchored environment {env!r}")
    return env


def env_slug(env: str) -> str:
    return require_env(env).replace("-v2", "")


def require_role(role: str) -> str:
    role = str(role)
    if role not in ROLES:
        raise ValueError(f"unknown anchored source role {role!r}")
    return role


def require_branch_role(role: str) -> str:
    role = str(role)
    if role not in BRANCH_ROLES:
        raise ValueError(f"unknown anchored branch role {role!r}")
    return role


def require_training_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in TRAINING_SEEDS:
        raise ValueError(f"unknown anchored training seed {seed}")
    return seed


def require_mode(mode: int) -> int:
    mode = int(mode)
    if mode not in MODES:
        raise ValueError(f"unknown anchored mode {mode}")
    return mode


def require_calibration_event_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in CALIBRATION_EVENT_SEEDS:
        raise ValueError(f"unknown anchored calibration event seed {seed}")
    return seed


def require_event_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in AUDIT_EVENT_SEEDS:
        raise ValueError(f"unknown anchored audit event seed {seed}")
    return seed


def run_dir(env: str, role: str, seed: int) -> Path:
    return (
        RUN_ROOT / env_slug(env) / "source" / require_role(role)
        / f"seed_{require_training_seed(seed)}")


def bundle_dir(env: str, role: str, seed: int) -> Path:
    return (
        BUNDLE_ROOT / env_slug(env) / "source" / require_role(role)
        / f"seed_{require_training_seed(seed)}")


def bundle_manifest(env: str, role: str, seed: int) -> Path:
    return bundle_dir(env, role, seed) / "bundle_manifest.json"


def bundle_required_paths(
    env: str,
    role: str,
    seed: int,
) -> tuple[Path, ...]:
    directory = bundle_dir(env, role, seed)
    return (
        directory / "bundle_manifest.json",
        directory / "checkpoints" / "params.pkl",
        directory / "checkpoints" / "train_state.pkl",
        directory / "logs" / "protocol_signature.json",
    )


def branch_run_dir(role: str, seed: int) -> Path:
    return (
        RUN_ROOT / env_slug(ENV) / "branches"
        / require_branch_role(role)
        / f"seed_{require_training_seed(seed)}")


def branch_bundle_dir(role: str, seed: int) -> Path:
    return (
        BUNDLE_ROOT / env_slug(ENV) / "branches"
        / require_branch_role(role)
        / f"seed_{require_training_seed(seed)}")


def branch_manifest(role: str, seed: int) -> Path:
    return branch_bundle_dir(role, seed) / "bundle_manifest.json"


def branch_required_paths(role: str, seed: int) -> tuple[Path, ...]:
    directory = branch_bundle_dir(role, seed)
    return (
        directory / "bundle_manifest.json",
        directory / "checkpoints" / "params.pkl",
        directory / "checkpoints" / "train_state.pkl",
        directory / "checkpoints" / BOOTSTRAP_NAME,
        directory / "logs" / "protocol_signature.json",
    )


def calibration_dir(seed: int) -> Path:
    return CALIBRATION_ROOT / f"seed_{require_training_seed(seed)}"


def calibration_manifest(seed: int) -> Path:
    return calibration_dir(seed) / "calibration_manifest.json"


def audit_dir(seed: int) -> Path:
    return AUDIT_ROOT / f"seed_{require_training_seed(seed)}"


def audit_manifest(seed: int) -> Path:
    return audit_dir(seed) / "audit_manifest.json"


def analysis_json() -> Path:
    return ANALYSIS_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return ANALYSIS_ROOT / "analysis.md"


def identity(env: str, role: str, seed: int) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "env": require_env(env),
        "family": FAMILY,
        "role": require_role(role),
        "training_seed": require_training_seed(seed),
        "algo": "regime_sac",
        "benchmark_role": "fresh_robust_anchor_source",
    }


def branch_identity(role: str, seed: int) -> dict[str, Any]:
    role = require_branch_role(role)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "env": ENV,
        "family": FAMILY,
        "role": role,
        "training_seed": require_training_seed(seed),
        "algo": (
            "regime_sac" if role == "robust_continue"
            else "anchored_regime_sac"
        ),
        "benchmark_role": "paired_robust_anchored_development",
    }


def expected_branch_checkpoint(role: str) -> dict[str, Any]:
    role = require_branch_role(role)
    return {
        "iteration": BRANCH_FINAL_ITERATION,
        "next_iteration": BRANCH_FINAL_NEXT_ITERATION,
        "total_steps": BRANCH_TOTAL_STEPS,
        "update_count": BRANCH_UPDATE_COUNT,
        "algo": (
            "regime_sac" if role == "robust_continue"
            else "anchored_regime_sac"
        ),
    }


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "size": path.stat().st_size}


def checkpoint_record(directory: Path) -> dict[str, Any]:
    return frozen.checkpoint_record(directory)


def _hash_arrays(values: Iterable[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, value in values:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(json.dumps({
            "name": name,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        }, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.view(np.uint8).tobytes())
    return digest.hexdigest()


def source_policy_sha256(policy) -> str:
    values: list[tuple[str, Any]] = []
    for index, layer in enumerate(policy.layers):
        values.extend([
            (f"layers.{index}.kernel", layer.kernel.value),
            (f"layers.{index}.bias", layer.bias.value),
        ])
    values.extend([
        ("mean_head.kernel", policy.mean_head.kernel.value),
        ("mean_head.bias", policy.mean_head.bias.value),
        ("log_std_head.kernel", policy.log_std_head.kernel.value),
        ("log_std_head.bias", policy.log_std_head.bias.value),
    ])
    return _hash_arrays(values)


def anchored_policy_hashes(policy) -> dict[str, str]:
    base: list[tuple[str, Any]] = []
    residual: list[tuple[str, Any]] = []
    for index, layer in enumerate(policy.base_layers):
        base.extend([
            (f"base_layers.{index}.kernel", layer.kernel.value),
            (f"base_layers.{index}.bias", layer.bias.value),
        ])
    base.extend([
        ("base_mean.kernel", policy.base_mean.kernel.value),
        ("base_mean.bias", policy.base_mean.bias.value),
        ("base_log_std.kernel", policy.base_log_std.kernel.value),
        ("base_log_std.bias", policy.base_log_std.bias.value),
    ])
    for index, layer in enumerate(policy.residual_layers):
        residual.extend([
            (f"residual_layers.{index}.kernel", layer.kernel.value),
            (f"residual_layers.{index}.bias", layer.bias.value),
        ])
    residual.extend([
        ("residual_mean.kernel", policy.residual_mean.kernel.value),
        ("residual_mean.bias", policy.residual_mean.bias.value),
    ])
    return {
        "base": _hash_arrays(base),
        "residual": _hash_arrays(residual),
    }


def ensemble_critic_sha256(critic) -> str:
    values: list[tuple[str, Any]] = []
    for index, layer in enumerate(critic.layers):
        values.extend([
            (f"layers.{index}.kernel", layer.kernel.value),
            (f"layers.{index}.bias", layer.bias.value),
        ])
    return _hash_arrays(values)


def anchored_critic_hashes(critic) -> dict[str, str]:
    return {
        "base": ensemble_critic_sha256(critic.base_critic),
        "adaptive": ensemble_critic_sha256(critic.adaptive_critic),
    }
