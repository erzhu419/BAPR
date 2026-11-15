"""Independent final confirmation of the frozen expected-action estimator."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import regime_polarity_confirmation as base
from jax_experiments.analysis import (
    regime_polarity_expected_action_system_id as estimator_protocol,
)


ROOT = base.ROOT
PROTOCOL_VERSION = "v5-final"
ENV = "HalfCheetah-v2"
ENVS = (ENV,)
FAMILY = base.FAMILY
ROLES = base.ROLES
MODES = base.MODES

# Frozen before any final-confirmation controller is trained.
TRAINING_SEEDS = (607, 719, 823, 929, 1031)
TEST_CONTROLLER_SEEDS = TRAINING_SEEDS
AUDIT_EVENT_SEEDS = (95_001, 95_002, 95_003)
TEST_EVENT_SEEDS = AUDIT_EVENT_SEEDS

# These identify the frozen estimator development split, not final policy
# seeds. The shared analyzer records them for provenance.
TRAIN_CONTROLLER_SEEDS = estimator_protocol.TRAIN_CONTROLLER_SEEDS
VALIDATION_CONTROLLER_SEEDS = estimator_protocol.VALIDATION_CONTROLLER_SEEDS

MAX_ITERS = base.MAX_ITERS
FINAL_ITERATION = base.FINAL_ITERATION
SAMPLES_PER_ITER = base.SAMPLES_PER_ITER
UPDATES_PER_ITER = base.UPDATES_PER_ITER
FINAL_TOTAL_STEPS = base.FINAL_TOTAL_STEPS
FINAL_UPDATE_COUNT = base.FINAL_UPDATE_COUNT
DWELL_STEPS = base.DWELL_STEPS
MAX_EPISODE_STEPS = base.MAX_EPISODE_STEPS
EPISODES_PER_TASK = base.EPISODES_PER_TASK
SWITCHING_EPISODES = base.SWITCHING_EPISODES
ROBUST_TRACE_CONTEXT_MODE_ID = base.ROBUST_TRACE_CONTEXT_MODE_ID

MIN_MODE_ACCURACY = estimator_protocol.MIN_MODE_ACCURACY
MAX_MEDIAN_SWITCH_DELAY = estimator_protocol.MAX_MEDIAN_SWITCH_DELAY
MAX_P90_SWITCH_DELAY = estimator_protocol.MAX_P90_SWITCH_DELAY
MAX_BRIER_SCORE = estimator_protocol.MAX_BRIER_SCORE
MIN_HEADROOM_RECOVERY = estimator_protocol.MIN_HEADROOM_RECOVERY
MIN_POLICY_SEED_WINS = estimator_protocol.MIN_POLICY_SEED_WINS
MAX_TERMINATION_GAP = estimator_protocol.MAX_TERMINATION_GAP

RUN_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_final_confirmation_v5")
BUNDLE_ROOT = (
    ROOT / "jax_experiments"
    / "eval_bundles_regime_polarity_final_confirmation_v5")
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_final_confirmation_audit_v5")
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_final_confirmation_analysis_v5")
PROTOCOL_REPORT = (
    ROOT / "reports"
    / "regime_polarity_final_confirmation_protocol_2026-07-28.md")

BUNDLE_SCHEMA = "bapr.regime-polarity-final-confirmation-bundle.v5"
AUDIT_SCHEMA = "bapr.regime-polarity-final-confirmation-audit.v5"
ANALYSIS_SCHEMA = "bapr.regime-polarity-final-confirmation-analysis.v5"

MODEL_MANIFEST = estimator_protocol.MODEL_MANIFEST
MODEL_PATH = estimator_protocol.MODEL_PATH
FROZEN_MODEL_MANIFEST_RECORD = {
    "sha256": "fc4b28662ac1f38081aab02723ebcbea2fa47fbe37eea27bb4e1a564b8d571d0",
    "size": 56089,
}
FROZEN_MODEL_PARAMETER_RECORD = {
    "sha256": "c1ff9021a3d3684116464c421c852eb16cd7ab286b2fd53bcd764d9a50393af1",
    "size": 2712127,
}

sha256_file = base.sha256_file
file_record = base.file_record
read_json = base.read_json
write_json_atomic = base.write_json_atomic
write_text_atomic = base.write_text_atomic
checkpoint_record = base.checkpoint_record
posterior_update = estimator_protocol.posterior_update
posterior_metrics = estimator_protocol.posterior_metrics


def validate_frozen_estimator() -> None:
    if (not MODEL_MANIFEST.is_file()
            or file_record(MODEL_MANIFEST)
            != FROZEN_MODEL_MANIFEST_RECORD
            or not MODEL_PATH.is_file()
            or file_record(MODEL_PATH)
            != FROZEN_MODEL_PARAMETER_RECORD):
        raise ValueError("frozen v4 estimator artifact changed")


def require_env(env: str) -> str:
    env = str(env)
    if env not in ENVS:
        raise ValueError(f"unknown final-confirmation environment {env!r}")
    return env


def env_slug(env: str) -> str:
    return require_env(env).replace("-v2", "")


def require_role(role: str) -> str:
    role = str(role)
    if role not in ROLES:
        raise ValueError(f"unknown final-confirmation role {role!r}")
    return role


def require_training_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in TRAINING_SEEDS:
        raise ValueError(f"unknown final-confirmation seed {seed}")
    return seed


def require_event_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in AUDIT_EVENT_SEEDS:
        raise ValueError(f"unknown final-confirmation event seed {seed}")
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
        "benchmark_role": "independent_frozen_bapr_final_confirmation",
    }
