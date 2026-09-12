"""Preregistered JAX smoke for paper-aligned RE-SAC training semantics."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from jax_experiments.analysis import regime_polarity_headroom as common


ROOT = Path(os.environ.get(
    "BAPR_WORKSPACE_ROOT", Path(__file__).resolve().parents[2])).resolve()
PROTOCOL_VERSION = "v1-paper-hyperparameter-aligned-jax-smoke"
ENVS = ("HalfCheetah-v2", "Ant-v2")
ROLES = ("sac", "escp", "resac")
TRAINING_SEEDS = (4001, 4013)
EVENT_SEEDS = (120_001, 120_011, 120_017)

MAX_ITERS = 1000
FINAL_ITERATION = MAX_ITERS - 1
INITIAL_RANDOM_STEPS = 10_000
SAMPLES_PER_ITER = 1000
UPDATES_PER_ITER = 1000
FINAL_TOTAL_STEPS = INITIAL_RANDOM_STEPS + MAX_ITERS * SAMPLES_PER_ITER
FINAL_UPDATE_COUNT = MAX_ITERS * UPDATES_PER_ITER
MAX_EPISODE_STEPS = 1000
TASK_NUM = 40
TEST_TASK_NUM = 40
AUDIT_TASKS = 8
AUDIT_EPISODES_PER_TASK = 3
AUDIT_SWITCHING_EPISODES = 5
SWITCHING_PERIOD_STEPS = 500

LR = 3e-4
CLIP_NORM = 1.0
ENSEMBLE_SIZE = {"sac": 2, "escp": 2, "resac": 10}
RESAC_BETA = -2.0
RESAC_BETA_OOD = 0.01
RESAC_WEIGHT_REG = 0.01
RESAC_BETA_BC = 0.001
RESAC_CRITIC_ACTOR_RATIO = 2

RUN_ROOT = (
    ROOT / "jax_experiments" / "results_resac_paper_fidelity_smoke_v1")
BUNDLE_ROOT = (
    ROOT / "jax_experiments" / "eval_bundles_resac_paper_fidelity_smoke_v1")
AUDIT_ROOT = (
    ROOT / "jax_experiments" / "results_resac_paper_fidelity_smoke_audit_v1")
ANALYSIS_ROOT = (
    ROOT / "jax_experiments" / "results_resac_paper_fidelity_smoke_analysis_v1")
REPORT = ROOT / "reports" / "resac_paper_fidelity_smoke_v1_2026-08-03.md"

BUNDLE_SCHEMA = "bapr.resac-paper-fidelity-bundle.v1"
AUDIT_SCHEMA = "bapr.resac-paper-fidelity-audit.v1"
ANALYSIS_SCHEMA = "bapr.resac-paper-fidelity-analysis.v1"

file_record = common.file_record
read_json = common.read_json
write_json_atomic = common.write_json_atomic
write_text_atomic = common.write_text_atomic
checkpoint_record = common.checkpoint_record


def require_env(env: str) -> str:
    env = str(env)
    if env not in ENVS:
        raise ValueError(f"unknown fidelity-smoke environment {env!r}")
    return env


def env_slug(env: str) -> str:
    return require_env(env).removesuffix("-v2")


def require_role(role: str) -> str:
    role = str(role)
    if role not in ROLES:
        raise ValueError(f"unknown fidelity-smoke role {role!r}")
    return role


def require_training_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in TRAINING_SEEDS:
        raise ValueError(f"unknown fidelity-smoke training seed {seed}")
    return seed


def require_event_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in EVENT_SEEDS:
        raise ValueError(f"unknown fidelity-smoke event seed {seed}")
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
    return audit_dir(env, role, seed) / f"event_seed_{require_event_seed(event_seed)}"


def analysis_json() -> Path:
    return ANALYSIS_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return ANALYSIS_ROOT / "analysis.md"


def identity(env: str, role: str, seed: int) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "paper_hyperparameter_aligned_jax_smoke",
        "env": require_env(env),
        "role": require_role(role),
        "training_seed": require_training_seed(seed),
        "environment": {
            "env_type": "continuous",
            "varying_params": ["gravity"],
            "task_scale_distribution": "pow1p5",
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
