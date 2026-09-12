"""Sealed protocol for executed-action inverse system identification."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from jax_experiments.analysis import regime_polarity_confirmation as confirmation
from jax_experiments.analysis import regime_polarity_posterior as v1
from jax_experiments.envs.stochastic_mode_env import MODE_FAMILIES


ROOT = confirmation.ROOT
PROTOCOL_VERSION = "v3"
ENV = "HalfCheetah-v2"
FAMILY = confirmation.FAMILY
MODES = confirmation.MODES
ROLES = confirmation.ROLES

TRAIN_CONTROLLER_SEEDS = (8, 16)
VALIDATION_CONTROLLER_SEEDS = (24,)
TEST_CONTROLLER_SEEDS = confirmation.TRAINING_SEEDS
TRAIN_EVENT_SEEDS = (92_001, 92_002)
VALIDATION_EVENT_SEEDS = (93_001, 93_002)
TEST_EVENT_SEEDS = (94_001, 94_002, 94_003)

DWELL_STEPS = confirmation.DWELL_STEPS
MAX_EPISODE_STEPS = confirmation.MAX_EPISODE_STEPS
STATIONARY_STEPS = 2_000
SWITCHING_EPISODES = 4
TRAIN_UPDATES = 3_000
BATCH_SIZE = 256

MODEL_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_inverse_system_id_model_v3")
MODEL_PATH = MODEL_ROOT / "inverse_params.npz"
MODEL_MANIFEST = MODEL_ROOT / "model_manifest.json"
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_inverse_system_id_audit_v3")
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_inverse_system_id_analysis_v3")
PROTOCOL_REPORT = (
    ROOT / "reports"
    / "regime_polarity_inverse_system_id_protocol_2026-07-28.md")

MODEL_SCHEMA = "bapr.regime-polarity-inverse-system-id-model.v3"
AUDIT_SCHEMA = "bapr.regime-polarity-inverse-system-id-audit.v3"
ANALYSIS_SCHEMA = "bapr.regime-polarity-inverse-system-id-analysis.v3"

MODEL_CONFIG = {
    "hidden_dim": 256,
    "ensemble_size": 5,
    "n_layers": 3,
    "obs_scale": 5.0,
    "delta_scale": 0.1,
    "learning_rate": 3e-4,
    "weight_decay": 1e-5,
    "variance_floor": 1e-4,
    "variance_ceiling": 0.25,
}

# Frozen before training; identical to v1/v2.
MIN_MODE_ACCURACY = v1.MIN_MODE_ACCURACY
MAX_MEDIAN_SWITCH_DELAY = v1.MAX_MEDIAN_SWITCH_DELAY
MAX_P90_SWITCH_DELAY = v1.MAX_P90_SWITCH_DELAY
MAX_BRIER_SCORE = v1.MAX_BRIER_SCORE
MIN_HEADROOM_RECOVERY = v1.MIN_HEADROOM_RECOVERY
MIN_POLICY_SEED_WINS = v1.MIN_POLICY_SEED_WINS
MAX_TERMINATION_GAP = v1.MAX_TERMINATION_GAP

FilterConfig = v1.FilterConfig
posterior_update = v1.posterior_update
causal_posteriors = v1.causal_posteriors
posterior_metrics = v1.posterior_metrics
sha256_file = v1.sha256_file
file_record = v1.file_record
read_json = v1.read_json
write_json_atomic = v1.write_json_atomic
write_text_atomic = v1.write_text_atomic
save_parameter_state = v1.save_parameter_state
load_parameter_state = v1.load_parameter_state


def filter_candidates() -> tuple[FilterConfig, ...]:
    return tuple(
        FilterConfig(hazard, evidence, decay)
        for hazard in (0.002, 0.004, 0.008)
        for evidence in (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
        for decay in (0.98, 1.0)
    )


def mode_gain_vectors(act_dim: int) -> np.ndarray:
    """Return the preregistered polarity transform for every mode."""
    act_dim = int(act_dim)
    if act_dim <= 0:
        raise ValueError("action dimension must be positive")
    indices = np.arange(act_dim)
    output = []
    for profile in MODE_FAMILIES[FAMILY]:
        nominal = float(profile["action_gain"])
        impaired = float(profile["impaired_gain"])
        gain = np.full((act_dim,), nominal, dtype=np.float32)
        pattern = profile["action_gain_pattern"]
        if pattern == "low_half":
            selected = indices < act_dim // 2
        elif pattern == "high_half":
            selected = indices >= act_dim // 2
        elif pattern == "even":
            selected = indices % 2 == 0
        elif pattern == "odd":
            selected = indices % 2 == 1
        else:
            raise ValueError(f"unsupported polarity pattern {pattern!r}")
        gain[selected] = impaired
        output.append(gain)
    return np.stack(output)


def audit_dir(seed: int) -> Path:
    seed = int(seed)
    if seed not in TEST_CONTROLLER_SEEDS:
        raise ValueError(f"unknown inverse-system-ID test seed {seed}")
    return AUDIT_ROOT / f"seed_{seed}"


def audit_manifest(seed: int) -> Path:
    return audit_dir(seed) / "audit_manifest.json"


def analysis_json() -> Path:
    return ANALYSIS_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return ANALYSIS_ROOT / "analysis.md"
