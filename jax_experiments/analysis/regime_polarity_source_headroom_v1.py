"""Source-controller oracle ladder for the actuator-polarity benchmark."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from jax_experiments.analysis import regime_polarity_headroom as common


ROOT = Path(os.environ.get(
    "BAPR_WORKSPACE_ROOT", Path(__file__).resolve().parents[2])).resolve()
PROTOCOL_VERSION = "v1-independent-source-controller-headroom"
ENV = "HalfCheetah-v2"
ENVS = (ENV,)
FAMILY = "actuator_polarity"
MODES = (0, 1, 2, 3)
ROLES = (
    "robust_sac",
    "escp",
    "specialist_0",
    "specialist_1",
    "specialist_2",
    "specialist_3",
)
TRAINING_SEEDS = (4021, 4049)
EVENT_SEEDS = (121_001, 121_013, 122_003)

MAX_ITERS = 1400
FINAL_ITERATION = MAX_ITERS - 1
SAMPLES_PER_ITER = 4000
UPDATES_PER_ITER = 250
START_TRAIN_STEPS = 4000
FINAL_TOTAL_STEPS = MAX_ITERS * SAMPLES_PER_ITER
FINAL_UPDATE_COUNT = MAX_ITERS * UPDATES_PER_ITER
DWELL_STEPS = 250
MAX_EPISODE_STEPS = 1000
EPISODES_PER_TASK = 5
SWITCHING_EPISODES = 5

MIN_DYNAMIC_GAIN = 0.10
MIN_DIAGONAL_MODES = 3

RUN_ROOT = (
    ROOT / "jax_experiments" / "results_regime_polarity_source_headroom_v1")
BUNDLE_ROOT = (
    ROOT / "jax_experiments" / "eval_bundles_regime_polarity_source_headroom_v1")
AUDIT_ROOT = (
    ROOT / "jax_experiments" / "results_regime_polarity_source_headroom_audit_v1")
ANALYSIS_ROOT = (
    ROOT / "jax_experiments" / "results_regime_polarity_source_headroom_analysis_v1")
REPORT = (
    ROOT / "reports" / "regime_polarity_source_headroom_v1_2026-08-03.md")

BUNDLE_SCHEMA = "bapr.regime-polarity-source-controller-bundle.v1"
AUDIT_SCHEMA = "bapr.regime-polarity-source-controller-audit.v1"
ANALYSIS_SCHEMA = "bapr.regime-polarity-source-controller-analysis.v1"

file_record = common.file_record
read_json = common.read_json
write_json_atomic = common.write_json_atomic
write_text_atomic = common.write_text_atomic
checkpoint_record = common.checkpoint_record


def require_env(env: str) -> str:
    env = str(env)
    if env not in ENVS:
        raise ValueError(f"unknown source-headroom environment {env!r}")
    return env


def env_slug(env: str) -> str:
    return require_env(env).removesuffix("-v2")


def require_role(role: str) -> str:
    role = str(role)
    if role not in ROLES:
        raise ValueError(f"unknown source-headroom role {role!r}")
    return role


def role_algo(role: str) -> str:
    role = require_role(role)
    return "escp" if role == "escp" else "sac"


def role_fixed_mode(role: str) -> int:
    role = require_role(role)
    return int(role.removeprefix("specialist_")) if role.startswith(
        "specialist_") else -1


def role_ensemble_size(role: str) -> int:
    return 10 if require_role(role) == "escp" else 2


def require_training_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in TRAINING_SEEDS:
        raise ValueError(f"unknown source-headroom training seed {seed}")
    return seed


def require_event_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in EVENT_SEEDS:
        raise ValueError(f"unknown source-headroom event seed {seed}")
    return seed


def run_dir(role: str, seed: int) -> Path:
    return RUN_ROOT / require_role(role) / f"seed_{require_training_seed(seed)}"


def bundle_dir(role: str, seed: int) -> Path:
    return (
        BUNDLE_ROOT / require_role(role)
        / f"seed_{require_training_seed(seed)}")


def bundle_manifest(role: str, seed: int) -> Path:
    return bundle_dir(role, seed) / "bundle_manifest.json"


def bundle_required_paths(role: str, seed: int) -> tuple[Path, ...]:
    directory = bundle_dir(role, seed)
    return (
        directory / "bundle_manifest.json",
        directory / "checkpoints" / "params.pkl",
        directory / "checkpoints" / "train_state.pkl",
        directory / "logs" / "protocol_signature.json",
    )


def audit_dir(seed: int) -> Path:
    return AUDIT_ROOT / f"seed_{require_training_seed(seed)}"


def audit_manifest(seed: int) -> Path:
    return audit_dir(seed) / "audit_manifest.json"


def audit_event_result(seed: int, event_seed: int) -> Path:
    return (
        audit_dir(seed) / f"event_seed_{require_event_seed(event_seed)}"
        / "results.json")


def analysis_json() -> Path:
    return ANALYSIS_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return ANALYSIS_ROOT / "analysis.md"


def identity(role: str, seed: int) -> dict[str, Any]:
    role = require_role(role)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "independent_source_controller_headroom",
        "env": ENV,
        "family": FAMILY,
        "role": role,
        "algo": role_algo(role),
        "fixed_mode": role_fixed_mode(role),
        "training_seed": require_training_seed(seed),
    }


def expected_checkpoint(role: str) -> dict[str, Any]:
    return {
        "iteration": FINAL_ITERATION,
        "next_iteration": MAX_ITERS,
        "total_steps": FINAL_TOTAL_STEPS,
        "update_count": FINAL_UPDATE_COUNT,
        "algo": role_algo(role),
    }
