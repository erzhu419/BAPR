"""Specialist-trajectory successor to expected-action system ID v4."""
from __future__ import annotations

from pathlib import Path

from jax_experiments.analysis import (
    regime_polarity_expected_action_system_id as parent,
)
from jax_experiments.analysis import (
    regime_polarity_source_headroom_v1 as source,
)


ROOT = parent.ROOT
PROTOCOL_VERSION = "v5-specialist-trajectory-expected-action-system-id"
ENV = parent.ENV
FAMILY = parent.FAMILY
MODES = parent.MODES

TRAIN_SOURCE_SEEDS = (4021,)
VALIDATION_SOURCE_SEEDS = (4049,)
AUDIT_SOURCE_SEEDS = source.TRAINING_SEEDS
TRAIN_EVENT_SEEDS = (154_001, 154_013)
VALIDATION_EVENT_SEEDS = (154_101, 154_113)
AUDIT_EVENT_SEEDS = (154_211, 154_223, 154_237)

STATIONARY_ARMS = (
    "robust_sac",
    "specialist_0",
    "specialist_1",
    "specialist_2",
    "specialist_3",
)
SWITCHING_ARMS = (
    *STATIONARY_ARMS,
    "dynamic_oracle",
    "sticky_confirm3_v4",
)

DWELL_STEPS = parent.DWELL_STEPS
MAX_EPISODE_STEPS = parent.MAX_EPISODE_STEPS
STATIONARY_STEPS = 1_000
SWITCHING_EPISODES = 3
TRAIN_UPDATES = 1_500
BATCH_SIZE = parent.BATCH_SIZE

MODEL_CONFIG = dict(parent.MODEL_CONFIG)
MODEL_CONFIG["learning_rate"] = 1e-4

MODEL_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_specialist_expected_action_model_v5"
)
MODEL_PATH = MODEL_ROOT / "inverse_params.npz"
MODEL_MANIFEST = MODEL_ROOT / "model_manifest.json"
CHECKPOINT_ROOT = MODEL_ROOT / "checkpoints"
TRAIN_STATE_JSON = CHECKPOINT_ROOT / "train_state.json"
TRAIN_STATE_NPZ = CHECKPOINT_ROOT / "train_state.npz"
TRAIN_STATE_PKL = CHECKPOINT_ROOT / "train_state.pkl"
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_specialist_expected_action_audit_v5"
)
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_specialist_expected_action_analysis_v5"
)
REPORT = (
    ROOT / "reports"
    / "regime_polarity_specialist_expected_action_v5_2026-08-29.md"
)

MODEL_SCHEMA = "bapr.regime-polarity-specialist-expected-action-model.v5"
EVENT_SCHEMA = "bapr.regime-polarity-specialist-expected-action-event.v5"
AUDIT_SCHEMA = "bapr.regime-polarity-specialist-expected-action-audit.v5"
ANALYSIS_SCHEMA = "bapr.regime-polarity-specialist-expected-action-analysis.v5"

FilterConfig = parent.FilterConfig
filter_candidates = parent.filter_candidates
mode_gain_vectors = parent.mode_gain_vectors
posterior_update = parent.posterior_update
file_record = parent.file_record
read_json = parent.read_json
write_json_atomic = parent.write_json_atomic
write_text_atomic = parent.write_text_atomic
save_parameter_state = parent.save_parameter_state
load_parameter_state = parent.load_parameter_state

PRIMARY_ARM = "posterior_sticky_confirm3"
ARMS = (
    "robust_sac",
    "dynamic_oracle",
    "posterior_map_no_gate",
    PRIMARY_ARM,
)
MIN_GAIN = 0.10
MIN_ORACLE_RECOVERY = 0.70
MIN_SWITCHING_RETURN = 2_200.0


def require_source_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in AUDIT_SOURCE_SEEDS:
        raise ValueError(f"unknown specialist estimator source seed {seed}")
    return seed


def require_event_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in AUDIT_EVENT_SEEDS:
        raise ValueError(f"unknown specialist estimator event seed {seed}")
    return seed


def audit_dir(seed: int) -> Path:
    return AUDIT_ROOT / f"seed_{require_source_seed(seed)}"


def audit_manifest(seed: int) -> Path:
    return audit_dir(seed) / "audit_manifest.json"


def event_result(seed: int, event_seed: int) -> Path:
    return (
        audit_dir(seed)
        / f"event_seed_{require_event_seed(event_seed)}"
        / "results.json"
    )


def analysis_json() -> Path:
    return ANALYSIS_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return ANALYSIS_ROOT / "analysis.md"


def estimator_records() -> dict:
    return {
        "manifest": file_record(MODEL_MANIFEST),
        "parameters": file_record(MODEL_PATH),
    }
