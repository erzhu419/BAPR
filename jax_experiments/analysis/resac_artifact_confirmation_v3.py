"""Registered five-seed confirmation of the released RE-SAC B0 artifact."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from jax_experiments.analysis import regime_polarity_headroom as common


ROOT = Path(os.environ.get(
    "BAPR_WORKSPACE_ROOT", Path(__file__).resolve().parents[2])).resolve()
PROTOCOL_VERSION = "v3-released-artifact-b0-eight-million-steps"
ENVS = ("HalfCheetah-v2", "Ant-v2")
ROLES = ("sac", "resac")
TRAINING_SEEDS = (4109, 4127, 4153, 4177, 4201)
EVENT_SEEDS = (130_001, 130_011, 130_027)

MAX_ITERS = 2000
FINAL_ITERATION = MAX_ITERS - 1
START_TRAIN_STEPS = 10_000
INITIAL_RANDOM_STEPS = 0
SAMPLES_PER_ITER = 4000
UPDATES_PER_ITER = 250
FINAL_TOTAL_STEPS = MAX_ITERS * SAMPLES_PER_ITER
FIRST_TRAIN_ITERATION = 2
FINAL_UPDATE_COUNT = (
    (MAX_ITERS - FIRST_TRAIN_ITERATION) * UPDATES_PER_ITER)
MAX_EPISODE_STEPS = 1000
TASK_NUM = 40
TEST_TASK_NUM = 40

AUDIT_TASKS = 40
AUDIT_EPISODES_PER_TASK = 3
AUDIT_SWITCHING_EPISODES = 5
SWITCHING_PERIOD_STEPS = 500

LR = 3e-4
HIDDEN_DIM = 256
ENSEMBLE_SIZE = {"sac": 2, "resac": {"HalfCheetah-v2": 5, "Ant-v2": 10}}
RESAC_BETA_START = -2.0
RESAC_BETA_END = {"HalfCheetah-v2": 0.0, "Ant-v2": -2.0}
RESAC_BETA_WARMUP = 0.2
RESAC_INDEPENDENT_RATIO = 0.75
RESAC_ANCHOR = {"HalfCheetah-v2": 0.001, "Ant-v2": 0.01}
RESAC_EMA_TAU = 0.005

RUN_ROOT = ROOT / "jax_experiments" / "results_resac_artifact_confirmation_v3"
BUNDLE_ROOT = (
    ROOT / "jax_experiments" / "eval_bundles_resac_artifact_confirmation_v3")
AUDIT_ROOT = (
    ROOT / "jax_experiments" / "results_resac_artifact_confirmation_audit_v3")
ANALYSIS_ROOT = (
    ROOT / "jax_experiments" / "results_resac_artifact_confirmation_analysis_v3")
REPORT = ROOT / "reports" / "resac_artifact_confirmation_2026-08-08.md"

BUNDLE_SCHEMA = "bapr.resac-artifact-confirmation-bundle.v3"
AUDIT_SCHEMA = "bapr.resac-artifact-confirmation-audit.v3"
ANALYSIS_SCHEMA = "bapr.resac-artifact-confirmation-analysis.v3"

file_record = common.file_record
read_json = common.read_json
write_json_atomic = common.write_json_atomic
write_text_atomic = common.write_text_atomic
checkpoint_record = common.checkpoint_record


def require_env(env: str) -> str:
    env = str(env)
    if env not in ENVS:
        raise ValueError(f"unknown artifact-confirmation environment {env!r}")
    return env


def env_slug(env: str) -> str:
    return require_env(env).removesuffix("-v2")


def require_role(role: str) -> str:
    role = str(role)
    if role not in ROLES:
        raise ValueError(f"unknown artifact-confirmation role {role!r}")
    return role


def require_training_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in TRAINING_SEEDS:
        raise ValueError(f"unknown artifact-confirmation seed {seed}")
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


def bundle_required_paths(env: str, role: str, seed: int) -> tuple[Path, ...]:
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


def audit_manifest(env: str, role: str, seed: int) -> Path:
    return audit_dir(env, role, seed) / "audit_manifest.json"


def audit_event_dir(env: str, role: str, seed: int, event_seed: int) -> Path:
    if int(event_seed) not in EVENT_SEEDS:
        raise ValueError(f"unknown artifact-confirmation event seed {event_seed}")
    return audit_dir(env, role, seed) / f"event_seed_{int(event_seed)}"


def analysis_json() -> Path:
    return ANALYSIS_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return ANALYSIS_ROOT / "analysis.md"


def identity(env: str, role: str, seed: int) -> dict[str, Any]:
    role = require_role(role)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "released_artifact_b0_confirmation",
        "env": require_env(env),
        "role": role,
        "training_seed": require_training_seed(seed),
        "implementation": (
            "released_artifact_b0_controls" if role == "resac"
            else "released_artifact_sac_control"),
        "environment": {
            "env_type": "continuous",
            "varying_params": ["gravity"],
            "task_scale_distribution": "exp",
            "log_scale_limit": 3.0,
            "changing_period": 20_000,
            "changing_interval": 4_000,
        },
    }


def expected_checkpoint(role: str) -> dict[str, Any]:
    return {
        "iteration": FINAL_ITERATION,
        "next_iteration": MAX_ITERS,
        "total_steps": FINAL_TOTAL_STEPS,
        "update_count": FINAL_UPDATE_COUNT,
        "algo": require_role(role),
    }
