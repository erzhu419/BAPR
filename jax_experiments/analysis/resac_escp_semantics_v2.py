"""Registered protocol for RE-SAC/ESCP numerical-semantics diagnostics."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from jax_experiments.analysis import regime_polarity_headroom as common
from jax_experiments.analysis import resac_paper_fidelity_smoke_v1 as legacy


ROOT = Path(os.environ.get(
    "BAPR_WORKSPACE_ROOT", Path(__file__).resolve().parents[2])).resolve()
PROTOCOL_VERSION = "v2-artifact-b0-and-stable-escp"
ENVS = ("HalfCheetah-v2", "Ant-v2")
ROLES = ("sac", "escp", "resac")
TRAIN_ROLES = ("escp", "resac")
TRAINING_SEEDS = (4001, 4013)
PROBE_SEEDS = (4001,)
EVENT_SEEDS = legacy.EVENT_SEEDS

# Keep the v1 diagnostic budget so its already completed SAC controllers are
# an exact environment-step/update-budget reference.  This is deliberately not
# the 8M-step confirmatory budget from the RE-SAC manuscript.
MAX_ITERS = legacy.MAX_ITERS
FINAL_ITERATION = legacy.FINAL_ITERATION
INITIAL_RANDOM_STEPS = legacy.INITIAL_RANDOM_STEPS
SAMPLES_PER_ITER = legacy.SAMPLES_PER_ITER
UPDATES_PER_ITER = legacy.UPDATES_PER_ITER
FINAL_TOTAL_STEPS = legacy.FINAL_TOTAL_STEPS
FINAL_UPDATE_COUNT = legacy.FINAL_UPDATE_COUNT
MAX_EPISODE_STEPS = legacy.MAX_EPISODE_STEPS
TASK_NUM = legacy.TASK_NUM
TEST_TASK_NUM = legacy.TEST_TASK_NUM
AUDIT_TASKS = legacy.AUDIT_TASKS
AUDIT_EPISODES_PER_TASK = legacy.AUDIT_EPISODES_PER_TASK
AUDIT_SWITCHING_EPISODES = legacy.AUDIT_SWITCHING_EPISODES
SWITCHING_PERIOD_STEPS = legacy.SWITCHING_PERIOD_STEPS

LR = 3e-4
HIDDEN_DIM = 256
RESAC_ENSEMBLE_SIZE = {"HalfCheetah-v2": 5, "Ant-v2": 10}
RESAC_BETA_START = -2.0
RESAC_BETA_END = {"HalfCheetah-v2": 0.0, "Ant-v2": -2.0}
RESAC_BETA_WARMUP = 0.2
RESAC_INDEPENDENT_RATIO = 0.75
RESAC_ANCHOR = {"HalfCheetah-v2": 0.001, "Ant-v2": 0.01}
RESAC_EMA_TAU = 0.005

ESCP_ENSEMBLE_SIZE = 2
ESCP_CONTEXT_MIN_STEPS = 100_000
ESCP_CONTEXT_MIN_TASKS = TASK_NUM // 2
ESCP_ALPHA_MAX = 1.0
ESCP_CLIP_NORM = 1.0

PROBE_MAX_ITERS = 25
PROBE_SAVE_INTERVAL = 1

RUN_ROOT = ROOT / "jax_experiments" / "results_resac_escp_semantics_v2"
BUNDLE_ROOT = (
    ROOT / "jax_experiments" / "eval_bundles_resac_escp_semantics_v2")
AUDIT_ROOT = (
    ROOT / "jax_experiments" / "results_resac_escp_semantics_audit_v2")
PROBE_ROOT = (
    ROOT / "jax_experiments" / "results_escp_first_nonfinite_probe_v1")
ANALYSIS_ROOT = (
    ROOT / "jax_experiments" / "results_resac_escp_semantics_analysis_v2")
REPORT = ROOT / "reports" / "resac_escp_numerical_semantics_2026-08-08.md"

BUNDLE_SCHEMA = "bapr.resac-escp-semantics-bundle.v2"
AUDIT_SCHEMA = "bapr.resac-escp-semantics-audit.v2"
PROBE_SCHEMA = "bapr.escp-first-nonfinite-probe.v1"
ANALYSIS_SCHEMA = "bapr.resac-escp-semantics-analysis.v2"

file_record = common.file_record
read_json = common.read_json
write_json_atomic = common.write_json_atomic
write_text_atomic = common.write_text_atomic
checkpoint_record = common.checkpoint_record


def require_env(env: str) -> str:
    env = str(env)
    if env not in ENVS:
        raise ValueError(f"unknown semantics environment {env!r}")
    return env


def env_slug(env: str) -> str:
    return require_env(env).removesuffix("-v2")


def require_role(role: str) -> str:
    role = str(role)
    if role not in ROLES:
        raise ValueError(f"unknown semantics role {role!r}")
    return role


def require_train_role(role: str) -> str:
    role = require_role(role)
    if role not in TRAIN_ROLES:
        raise ValueError(f"role {role!r} reuses the sealed v1 SAC bundle")
    return role


def require_training_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in TRAINING_SEEDS:
        raise ValueError(f"unknown semantics training seed {seed}")
    return seed


def run_dir(env: str, role: str, seed: int) -> Path:
    return (
        RUN_ROOT / env_slug(env) / require_train_role(role)
        / f"seed_{require_training_seed(seed)}")


def bundle_dir(env: str, role: str, seed: int) -> Path:
    role = require_role(role)
    seed = require_training_seed(seed)
    if role == "sac":
        return legacy.bundle_dir(require_env(env), role, seed)
    return BUNDLE_ROOT / env_slug(env) / role / f"seed_{seed}"


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
        raise ValueError(f"unknown event seed {event_seed}")
    return audit_dir(env, role, seed) / f"event_seed_{int(event_seed)}"


def probe_dir(env: str, seed: int) -> Path:
    seed = int(seed)
    if seed not in PROBE_SEEDS:
        raise ValueError(f"unknown probe seed {seed}")
    return PROBE_ROOT / env_slug(env) / f"seed_{seed}"


def probe_result(env: str, seed: int) -> Path:
    return probe_dir(env, seed) / "probe_result.json"


def analysis_json() -> Path:
    return ANALYSIS_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return ANALYSIS_ROOT / "analysis.md"


def identity(env: str, role: str, seed: int) -> dict[str, Any]:
    role = require_role(role)
    implementation = {
        "sac": "sealed_v1_budget_reference",
        "escp": "state_only_stable_core_approximation",
        "resac": "released_artifact_b0_controls",
    }[role]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "numerical_semantics_diagnostic",
        "env": require_env(env),
        "role": role,
        "training_seed": require_training_seed(seed),
        "implementation": implementation,
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
