"""No-realized-action supervision ablation for inverse system ID."""
from __future__ import annotations

from pathlib import Path

from jax_experiments.analysis import (
    regime_polarity_inverse_system_id as v3,
)


ROOT = v3.ROOT
PROTOCOL_VERSION = "v4"
ENV = v3.ENV
FAMILY = v3.FAMILY
MODES = v3.MODES
ROLES = v3.ROLES

TRAIN_CONTROLLER_SEEDS = v3.TRAIN_CONTROLLER_SEEDS
VALIDATION_CONTROLLER_SEEDS = v3.VALIDATION_CONTROLLER_SEEDS
TEST_CONTROLLER_SEEDS = v3.TEST_CONTROLLER_SEEDS
TRAIN_EVENT_SEEDS = v3.TRAIN_EVENT_SEEDS
VALIDATION_EVENT_SEEDS = v3.VALIDATION_EVENT_SEEDS
TEST_EVENT_SEEDS = v3.TEST_EVENT_SEEDS

DWELL_STEPS = v3.DWELL_STEPS
MAX_EPISODE_STEPS = v3.MAX_EPISODE_STEPS
STATIONARY_STEPS = v3.STATIONARY_STEPS
SWITCHING_EPISODES = v3.SWITCHING_EPISODES
TRAIN_UPDATES = v3.TRAIN_UPDATES
BATCH_SIZE = v3.BATCH_SIZE

MODEL_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_expected_action_system_id_model_v4")
MODEL_PATH = MODEL_ROOT / "inverse_params.npz"
MODEL_MANIFEST = MODEL_ROOT / "model_manifest.json"
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_expected_action_system_id_audit_v4")
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_expected_action_system_id_analysis_v4")
PROTOCOL_REPORT = (
    ROOT / "reports"
    / "regime_polarity_expected_action_system_id_protocol_2026-07-28.md")

MODEL_SCHEMA = "bapr.regime-polarity-expected-action-system-id-model.v4"
AUDIT_SCHEMA = "bapr.regime-polarity-expected-action-system-id-audit.v4"
ANALYSIS_SCHEMA = "bapr.regime-polarity-expected-action-system-id-analysis.v4"
MODEL_CONFIG = v3.MODEL_CONFIG

MIN_MODE_ACCURACY = v3.MIN_MODE_ACCURACY
MAX_MEDIAN_SWITCH_DELAY = v3.MAX_MEDIAN_SWITCH_DELAY
MAX_P90_SWITCH_DELAY = v3.MAX_P90_SWITCH_DELAY
MAX_BRIER_SCORE = v3.MAX_BRIER_SCORE
MIN_HEADROOM_RECOVERY = v3.MIN_HEADROOM_RECOVERY
MIN_POLICY_SEED_WINS = v3.MIN_POLICY_SEED_WINS
MAX_TERMINATION_GAP = v3.MAX_TERMINATION_GAP

FilterConfig = v3.FilterConfig
filter_candidates = v3.filter_candidates
mode_gain_vectors = v3.mode_gain_vectors
posterior_update = v3.posterior_update
causal_posteriors = v3.causal_posteriors
posterior_metrics = v3.posterior_metrics
sha256_file = v3.sha256_file
file_record = v3.file_record
read_json = v3.read_json
write_json_atomic = v3.write_json_atomic
write_text_atomic = v3.write_text_atomic
save_parameter_state = v3.save_parameter_state
load_parameter_state = v3.load_parameter_state


def audit_dir(seed: int) -> Path:
    seed = int(seed)
    if seed not in TEST_CONTROLLER_SEEDS:
        raise ValueError(f"unknown expected-action audit seed {seed}")
    return AUDIT_ROOT / f"seed_{seed}"


def audit_manifest(seed: int) -> Path:
    return audit_dir(seed) / "audit_manifest.json"


def analysis_json() -> Path:
    return ANALYSIS_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return ANALYSIS_ROOT / "analysis.md"
