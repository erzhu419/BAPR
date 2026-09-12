"""Preregistered five-seed confirmation for HalfCheetah polarity headroom."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import regime_polarity_headroom as exploratory


ROOT = exploratory.ROOT
PROTOCOL_VERSION = "v1"
FAMILY = exploratory.FAMILY
ENVS = ("HalfCheetah-v2",)
ROLES = exploratory.ROLES

# These seeds were sealed before any confirmation run was submitted. They are
# disjoint from the exploratory 8/16/24 runs and are never initialized from
# an exploratory checkpoint.
TRAINING_SEEDS = (101, 211, 307, 419, 523)
AUDIT_EVENT_SEEDS = (91_001, 91_002, 91_003)
MODES = exploratory.MODES

MAX_ITERS = exploratory.MAX_ITERS
FINAL_ITERATION = exploratory.FINAL_ITERATION
SAMPLES_PER_ITER = exploratory.SAMPLES_PER_ITER
UPDATES_PER_ITER = exploratory.UPDATES_PER_ITER
FINAL_TOTAL_STEPS = exploratory.FINAL_TOTAL_STEPS
FINAL_UPDATE_COUNT = exploratory.FINAL_UPDATE_COUNT
DWELL_STEPS = exploratory.DWELL_STEPS
MAX_EPISODE_STEPS = exploratory.MAX_EPISODE_STEPS
EPISODES_PER_TASK = exploratory.EPISODES_PER_TASK
SWITCHING_EPISODES = exploratory.SWITCHING_EPISODES

MIN_RELATIVE_GAIN = 0.10
MAX_TERMINATION_GAP = 0.05
MIN_MODE_WINS = 3
MIN_PASSING_ENVS = 1

# The robust arm has the same context-input width as the oracle arm but is
# deliberately given the all-zero vector. RegimeSAC records that as -1 in a
# switching trace, rather than pretending it received the physics mode.
ROBUST_TRACE_CONTEXT_MODE_ID = -1

RUN_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_confirmation_v1")
BUNDLE_ROOT = (
    ROOT / "jax_experiments"
    / "eval_bundles_regime_polarity_confirmation_v1")
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_confirmation_audit_v1")
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_confirmation_analysis_v1")
PROTOCOL_REPORT = (
    ROOT / "reports"
    / "regime_polarity_confirmation_protocol_2026-07-27.md")

BUNDLE_SCHEMA = "bapr.regime-polarity-confirmation-bundle.v1"
AUDIT_SCHEMA = "bapr.regime-polarity-confirmation-audit.v1"
ANALYSIS_SCHEMA = "bapr.regime-polarity-confirmation-analysis.v1"

sha256_file = exploratory.sha256_file
file_record = exploratory.file_record
read_json = exploratory.read_json
write_json_atomic = exploratory.write_json_atomic
write_text_atomic = exploratory.write_text_atomic
checkpoint_record = exploratory.checkpoint_record


def require_env(env: str) -> str:
    env = str(env)
    if env not in ENVS:
        raise ValueError(
            f"unknown polarity confirmation environment {env!r}")
    return env


def env_slug(env: str) -> str:
    return require_env(env).replace("-v2", "")


def require_role(role: str) -> str:
    role = str(role)
    if role not in ROLES:
        raise ValueError(f"unknown polarity confirmation role {role!r}")
    return role


def require_training_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in TRAINING_SEEDS:
        raise ValueError(f"unknown polarity confirmation seed {seed}")
    return seed


def require_event_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in AUDIT_EVENT_SEEDS:
        raise ValueError(
            f"unknown polarity confirmation event seed {seed}")
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
        "benchmark_role": "fresh_five_seed_oracle_confirmation",
    }
