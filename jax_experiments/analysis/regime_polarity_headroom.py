"""Sealed oracle-headroom screen for persistent actuator-polarity modes."""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

from jax_experiments.analysis import regime_control_headroom as common
from jax_experiments.common.checkpoint import (
    _patch_flax_variablestate_unpickle,
)


ROOT = Path(os.environ.get(
    "BAPR_WORKSPACE_ROOT", Path(__file__).resolve().parents[2])).resolve()
PROTOCOL_VERSION = "v1"
FAMILY = "actuator_polarity"
ENVS = ("HalfCheetah-v2", "Ant-v2", "Hopper-v2", "Walker2d-v2")
ROLES = ("robust", "oracle")
TRAINING_SEEDS = (8, 16, 24)
AUDIT_EVENT_SEEDS = (81100, 81200, 81300)
MODES = (0, 1, 2, 3)

MAX_ITERS = 1400
FINAL_ITERATION = MAX_ITERS - 1
SAMPLES_PER_ITER = 4000
UPDATES_PER_ITER = 250
FINAL_TOTAL_STEPS = MAX_ITERS * SAMPLES_PER_ITER
FINAL_UPDATE_COUNT = MAX_ITERS * UPDATES_PER_ITER
DWELL_STEPS = 250
MAX_EPISODE_STEPS = 1000
EPISODES_PER_TASK = 5
SWITCHING_EPISODES = 5

MIN_RELATIVE_GAIN = 0.15
MAX_TERMINATION_GAP = 0.05
MIN_MODE_WINS = 3
MIN_PASSING_ENVS = 3

RUN_ROOT = (
    ROOT / "jax_experiments" / "results_regime_polarity_headroom_v1")
BUNDLE_ROOT = (
    ROOT / "jax_experiments" / "eval_bundles_regime_polarity_headroom_v1")
AUDIT_ROOT = (
    ROOT / "jax_experiments" / "results_regime_polarity_headroom_audit_v1")
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_headroom_analysis_v1")
PROTOCOL_REPORT = (
    ROOT / "reports" / "regime_polarity_headroom_protocol_2026-07-26.md")

BUNDLE_SCHEMA = "bapr.regime-polarity-headroom-bundle.v1"
AUDIT_SCHEMA = "bapr.regime-polarity-headroom-audit.v1"
ANALYSIS_SCHEMA = "bapr.regime-polarity-headroom-analysis.v1"

sha256_file = common.sha256_file
file_record = common.file_record
read_json = common.read_json
write_json_atomic = common.write_json_atomic
write_text_atomic = common.write_text_atomic


def require_env(env: str) -> str:
    env = str(env)
    if env not in ENVS:
        raise ValueError(f"unknown polarity environment {env!r}")
    return env


def env_slug(env: str) -> str:
    return require_env(env).replace("-v2", "")


def require_role(role: str) -> str:
    role = str(role)
    if role not in ROLES:
        raise ValueError(f"unknown polarity role {role!r}")
    return role


def require_training_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in TRAINING_SEEDS:
        raise ValueError(f"unknown polarity training seed {seed}")
    return seed


def require_event_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in AUDIT_EVENT_SEEDS:
        raise ValueError(f"unknown polarity event seed {seed}")
    return seed


def run_dir(env: str, role: str, seed: int) -> Path:
    return (
        RUN_ROOT / env_slug(env) / require_role(role)
        / f"seed_{require_training_seed(seed)}")


def bundle_dir(env: str, role: str, seed: int) -> Path:
    return (
        BUNDLE_ROOT / env_slug(env) / require_role(role)
        / f"seed_{require_training_seed(seed)}")


def bundle_manifest(env: str, role: str, seed: int) -> Path:
    return bundle_dir(env, role, seed) / "bundle_manifest.json"


def bundle_required_paths(
        env: str, role: str, seed: int) -> tuple[Path, ...]:
    directory = bundle_dir(env, role, seed)
    return (
        directory / "bundle_manifest.json",
        directory / "checkpoints" / "params.pkl",
        directory / "checkpoints" / "train_state.pkl",
        directory / "logs" / "protocol_signature.json",
    )


def audit_dir(env: str, role: str, seed: int) -> Path:
    return (
        AUDIT_ROOT / env_slug(env) / require_role(role)
        / f"seed_{require_training_seed(seed)}")


def audit_event_dir(
        env: str, role: str, seed: int, event_seed: int) -> Path:
    return (
        audit_dir(env, role, seed)
        / f"event_seed_{require_event_seed(event_seed)}")


def audit_manifest(env: str, role: str, seed: int) -> Path:
    return audit_dir(env, role, seed) / "audit_manifest.json"


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
        "benchmark_role": "oracle_headroom_positive_control",
    }


def checkpoint_record(directory: Path) -> dict[str, Any]:
    _patch_flax_variablestate_unpickle()
    checkpoint = directory / "checkpoints"
    with (checkpoint / "train_state.pkl").open("rb") as handle:
        state = pickle.load(handle)
    with (checkpoint / "params.pkl").open("rb") as handle:
        params = pickle.load(handle)
    iteration = int(state["iteration"])
    return {
        "iteration": iteration,
        "next_iteration": iteration + 1,
        "total_steps": int(state["total_steps"]),
        "update_count": int(params["update_count"]),
        "algo": str(state["algo"]),
    }
