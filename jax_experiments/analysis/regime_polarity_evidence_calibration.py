"""Sealed v2 evidence-calibration protocol for actuator polarity."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from jax_experiments.analysis import regime_polarity_posterior as v1


ROOT = v1.ROOT
PROTOCOL_VERSION = "v2-calibrated-evidence"
ENV = v1.ENV
FAMILY = v1.FAMILY
MODES = v1.MODES
ROLES = v1.ROLES

TRAIN_CONTROLLER_SEEDS = v1.TRAIN_CONTROLLER_SEEDS
VALIDATION_CONTROLLER_SEEDS = v1.VALIDATION_CONTROLLER_SEEDS
TEST_CONTROLLER_SEEDS = v1.TEST_CONTROLLER_SEEDS
TRAIN_EVENT_SEEDS = v1.TRAIN_EVENT_SEEDS
VALIDATION_EVENT_SEEDS = v1.VALIDATION_EVENT_SEEDS
TEST_EVENT_SEEDS = v1.TEST_EVENT_SEEDS

DWELL_STEPS = v1.DWELL_STEPS
MAX_EPISODE_STEPS = v1.MAX_EPISODE_STEPS
STATIONARY_STEPS = v1.STATIONARY_STEPS
SWITCHING_EPISODES = v1.SWITCHING_EPISODES

MIN_MODE_ACCURACY = v1.MIN_MODE_ACCURACY
MAX_MEDIAN_SWITCH_DELAY = v1.MAX_MEDIAN_SWITCH_DELAY
MAX_P90_SWITCH_DELAY = v1.MAX_P90_SWITCH_DELAY
MAX_BRIER_SCORE = v1.MAX_BRIER_SCORE
MIN_HEADROOM_RECOVERY = v1.MIN_HEADROOM_RECOVERY
MIN_POLICY_SEED_WINS = v1.MIN_POLICY_SEED_WINS
MAX_TERMINATION_GAP = v1.MAX_TERMINATION_GAP

MODEL_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_evidence_calibration_v2")
MODEL_PATH = MODEL_ROOT / "calibrator_params.npz"
MODEL_MANIFEST = MODEL_ROOT / "calibrator_manifest.json"
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_calibrated_audit_v2")
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_calibrated_analysis_v2")
PROTOCOL_REPORT = (
    ROOT / "reports"
    / "regime_polarity_evidence_calibration_protocol_2026-07-28.md")

MODEL_SCHEMA = "bapr.regime-polarity-evidence-calibrator.v2"
AUDIT_SCHEMA = "bapr.regime-polarity-calibrated-audit.v2"
ANALYSIS_SCHEMA = "bapr.regime-polarity-calibrated-analysis.v2"


@dataclass(frozen=True)
class CalibratorConfig:
    """Causal temporal feature and affine calibration controls."""

    ema_alpha: float
    ridge: float
    temperature: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CalibratorConfig":
        return cls(**{
            name: value[name] for name in cls.__dataclass_fields__
        })

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def candidate_configs() -> tuple[CalibratorConfig, ...]:
    return tuple(
        CalibratorConfig(alpha, ridge, temperature)
        for alpha in (0.0, 0.5, 0.8, 0.9, 0.95)
        for ridge in (1e-3, 1e-2, 1e-1, 1.0)
        for temperature in (0.1, 0.25, 0.5, 1.0)
    )


def centered_evidence(log_likelihood: np.ndarray) -> np.ndarray:
    evidence = np.asarray(log_likelihood, dtype=np.float64)
    if evidence.shape[-1] != len(MODES):
        raise ValueError("evidence must have one column per mode")
    return evidence - np.mean(evidence, axis=-1, keepdims=True)


def temporal_features(
    log_likelihood: np.ndarray,
    ema_alpha: float,
) -> np.ndarray:
    """Build causal instantaneous plus exponentially averaged evidence."""
    instant = centered_evidence(log_likelihood)
    if instant.ndim != 2:
        raise ValueError("sequence evidence must have shape [time, modes]")
    alpha = float(ema_alpha)
    if not 0.0 <= alpha < 1.0:
        raise ValueError("ema_alpha must be in [0, 1)")
    ema = np.zeros((len(MODES),), dtype=np.float64)
    output = []
    for row in instant:
        ema = alpha * ema + (1.0 - alpha) * row
        output.append(np.concatenate([row, ema]))
    return np.asarray(output, dtype=np.float64)


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = values - np.max(values, axis=-1, keepdims=True)
    probabilities = np.exp(values)
    return probabilities / np.sum(
        probabilities, axis=-1, keepdims=True)


class CausalEvidenceCalibrator:
    """Apply a frozen affine map to causal temporal likelihood features."""

    def __init__(
        self,
        config: CalibratorConfig,
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        weights: np.ndarray,
        bias: np.ndarray,
    ):
        self.config = config
        self.feature_mean = np.asarray(feature_mean, dtype=np.float64)
        self.feature_scale = np.asarray(feature_scale, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.bias = np.asarray(bias, dtype=np.float64)
        feature_dim = 2 * len(MODES)
        if self.feature_mean.shape != (feature_dim,):
            raise ValueError("wrong calibrator feature mean shape")
        if self.feature_scale.shape != (feature_dim,):
            raise ValueError("wrong calibrator feature scale shape")
        if self.weights.shape != (feature_dim, len(MODES)):
            raise ValueError("wrong calibrator weight shape")
        if self.bias.shape != (len(MODES),):
            raise ValueError("wrong calibrator bias shape")
        if np.any(self.feature_scale <= 0.0):
            raise ValueError("calibrator feature scale must be positive")

    def initial_state(self):
        return (
            np.zeros((len(MODES),), dtype=np.float64),
            np.full(
                (len(MODES),),
                1.0 / len(MODES),
                dtype=np.float64,
            ),
        )

    @staticmethod
    def probabilities(state):
        return np.asarray(state[1], dtype=np.float64)

    def update_from_evidence(self, state, log_likelihood):
        instant = centered_evidence(
            np.asarray(log_likelihood, dtype=np.float64))
        ema = (
            self.config.ema_alpha * np.asarray(state[0], dtype=np.float64)
            + (1.0 - self.config.ema_alpha) * instant
        )
        feature = np.concatenate([instant, ema])
        standardized = (
            feature - self.feature_mean) / self.feature_scale
        logits = (
            standardized @ self.weights + self.bias
        ) / self.config.temperature
        posterior = _softmax(logits)
        return ema, posterior

    def action_posteriors(self, log_likelihood: np.ndarray) -> np.ndarray:
        state = self.initial_state()
        before = []
        for row in np.asarray(log_likelihood):
            before.append(self.probabilities(state).copy())
            state = self.update_from_evidence(state, row)
        return np.asarray(before)


def save_calibrator(
    path: Path,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                feature_mean=np.asarray(feature_mean, dtype=np.float64),
                feature_scale=np.asarray(feature_scale, dtype=np.float64),
                weights=np.asarray(weights, dtype=np.float64),
                bias=np.asarray(bias, dtype=np.float64),
            )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_calibrator() -> tuple[
    CausalEvidenceCalibrator, dict[str, Any]
]:
    manifest = read_json(MODEL_MANIFEST)
    if (manifest.get("schema") != MODEL_SCHEMA
            or manifest.get("status") != "complete"
            or manifest.get("env") != ENV
            or manifest.get("family") != FAMILY
            or manifest.get("parameter_file") != file_record(MODEL_PATH)
            or manifest.get("source_model_manifest")
            != file_record(v1.MODEL_MANIFEST)
            or manifest.get("source_model_parameters")
            != file_record(v1.MODEL_PATH)):
        raise ValueError("invalid or stale evidence calibrator")
    with np.load(MODEL_PATH, allow_pickle=False) as payload:
        calibrator = CausalEvidenceCalibrator(
            CalibratorConfig.from_dict(manifest["calibrator_config"]),
            payload["feature_mean"],
            payload["feature_scale"],
            payload["weights"],
            payload["bias"],
        )
    return calibrator, manifest


def audit_dir(seed: int) -> Path:
    seed = int(seed)
    if seed not in TEST_CONTROLLER_SEEDS:
        raise ValueError(f"unknown calibrated-posterior test seed {seed}")
    return AUDIT_ROOT / f"seed_{seed}"


def audit_manifest(seed: int) -> Path:
    return audit_dir(seed) / "audit_manifest.json"


def analysis_json() -> Path:
    return ANALYSIS_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return ANALYSIS_ROOT / "analysis.md"


file_record = v1.file_record
read_json = v1.read_json
write_json_atomic = v1.write_json_atomic
write_text_atomic = v1.write_text_atomic
