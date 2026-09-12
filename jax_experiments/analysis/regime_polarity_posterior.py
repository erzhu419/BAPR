"""Frozen-controller posterior screen for the polarity benchmark."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jax_experiments.analysis import regime_polarity_confirmation as confirmation
from jax_experiments.analysis import regime_polarity_headroom as exploratory


ROOT = confirmation.ROOT
PROTOCOL_VERSION = "v1"
ENV = "HalfCheetah-v2"
FAMILY = confirmation.FAMILY
MODES = confirmation.MODES
ROLES = confirmation.ROLES

# Exploratory controllers provide estimator development data. None of the five
# sealed confirmation policy seeds is exposed during model fitting or filter
# selection.
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
TRAIN_UPDATES = 1_500
CONTEXT_LENGTH = 64

MODEL_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_posterior_model_v1")
MODEL_PATH = MODEL_ROOT / "posterior_params.npz"
MODEL_MANIFEST = MODEL_ROOT / "model_manifest.json"
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_posterior_audit_v1")
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_posterior_analysis_v1")
PROTOCOL_REPORT = (
    ROOT / "reports"
    / "regime_polarity_posterior_protocol_2026-07-28.md")

MODEL_SCHEMA = "bapr.regime-polarity-posterior-model.v1"
AUDIT_SCHEMA = "bapr.regime-polarity-posterior-audit.v1"
ANALYSIS_SCHEMA = "bapr.regime-polarity-posterior-analysis.v1"

MODEL_CONFIG = {
    "hidden_dim": 128,
    "ensemble_size": 5,
    "variance_model": "mode_empirical",
    "variance_floor": 1e-4,
    "variance_ceiling": 0.5,
    "fixed_variance": 0.02,
    "reward_scale": 10.0,
    "delta_scale": 1.0,
    "mean_loss_weight": 1.0,
    "variance_ema": 0.05,
    "learning_rate": 3e-4,
}

# Test gates are fixed before estimator training.
MIN_MODE_ACCURACY = 0.85
MAX_MEDIAN_SWITCH_DELAY = 25.0
MAX_P90_SWITCH_DELAY = 50.0
MAX_BRIER_SCORE = 0.25
MIN_HEADROOM_RECOVERY = 0.50
MIN_POLICY_SEED_WINS = 4
MAX_TERMINATION_GAP = 0.05


@dataclass(frozen=True)
class FilterConfig:
    """Sticky-HMM parameters selected only on exploratory validation data."""

    hazard_rate: float
    evidence_scale: float
    posterior_decay: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FilterConfig":
        return cls(**{
            name: value[name] for name in cls.__dataclass_fields__
        })

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def filter_candidates() -> tuple[FilterConfig, ...]:
    return tuple(
        FilterConfig(hazard, evidence, decay)
        for hazard in (0.002, 0.004, 0.008)
        for evidence in (0.25, 0.5, 1.0, 2.0)
        for decay in (0.98, 1.0)
    )


def posterior_update(
    posterior: np.ndarray,
    log_likelihood: np.ndarray,
    config: FilterConfig,
) -> np.ndarray:
    """Apply one stable sticky-HMM Bayes update."""
    posterior = np.asarray(posterior, dtype=np.float64)
    evidence = np.asarray(log_likelihood, dtype=np.float64)
    if posterior.shape != (len(MODES),) or evidence.shape != posterior.shape:
        raise ValueError("posterior and evidence must have one value per mode")
    if not np.all(np.isfinite(evidence)):
        raise ValueError("non-finite mode evidence")
    switch = config.hazard_rate / float(len(MODES) - 1)
    prior = (
        (1.0 - config.hazard_rate) * posterior
        + switch * (1.0 - posterior)
    )
    logits = (
        config.posterior_decay * np.log(np.clip(prior, 1e-12, 1.0))
        + config.evidence_scale * (evidence - np.max(evidence))
    )
    logits -= np.max(logits)
    updated = np.exp(logits)
    return updated / np.sum(updated)


def causal_posteriors(
    log_likelihoods: np.ndarray,
    config: FilterConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return action-time and post-transition posteriors.

    Row ``t`` in the action-time array uses only transitions before ``t``.
    """
    evidence = np.asarray(log_likelihoods, dtype=np.float64)
    if evidence.ndim != 2 or evidence.shape[1] != len(MODES):
        raise ValueError("evidence must have shape [time, modes]")
    current = np.full(
        (len(MODES),), 1.0 / len(MODES), dtype=np.float64)
    before = []
    after = []
    for row in evidence:
        before.append(current.copy())
        current = posterior_update(current, row, config)
        after.append(current.copy())
    return np.asarray(before), np.asarray(after)


def _switch_points(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32)
    return np.flatnonzero(labels[1:] != labels[:-1]) + 1


def posterior_metrics(
    action_posteriors: np.ndarray,
    labels: np.ndarray,
    burnin: int = 25,
    stability: int = 8,
) -> dict[str, Any]:
    """Measure causal mode inference, calibration, and switch latency."""
    posterior = np.asarray(action_posteriors, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    if posterior.shape != (len(labels), len(MODES)):
        raise ValueError("posterior/label shape mismatch")
    if len(labels) == 0:
        raise ValueError("posterior trace is empty")
    one_hot = np.eye(len(MODES), dtype=np.float64)[labels]
    predicted = np.argmax(posterior, axis=-1)
    valid = np.ones((len(labels),), dtype=bool)
    segment_starts = np.concatenate([[0], _switch_points(labels)])
    for start in segment_starts:
        valid[start:min(len(valid), start + int(burnin))] = False
    if not np.any(valid):
        valid[:] = True

    true_probability = posterior[np.arange(len(labels)), labels]
    brier = np.sum(np.square(posterior - one_hot), axis=-1)
    confidence = np.max(posterior, axis=-1)
    correct = predicted == labels
    calibration_gap = []
    for lower in np.linspace(0.0, 0.9, 10):
        selected = valid & (confidence >= lower) & (confidence < lower + 0.1)
        if np.any(selected):
            calibration_gap.append(
                (np.sum(selected) / np.sum(valid)) * abs(
                    float(np.mean(confidence[selected]))
                    - float(np.mean(correct[selected]))
                )
            )

    delays = []
    points = _switch_points(labels)
    for index, start in enumerate(points):
        end = int(points[index + 1]) if index + 1 < len(points) else len(labels)
        wanted = int(labels[start])
        delay = end - int(start)
        for candidate in range(int(start), max(int(start), end - stability + 1)):
            if np.all(predicted[candidate:candidate + stability] == wanted):
                delay = candidate - int(start)
                break
        delays.append(int(delay))
    return {
        "mode_accuracy": float(np.mean(correct[valid])),
        "mean_true_probability": float(np.mean(true_probability[valid])),
        "negative_log_likelihood": float(np.mean(
            -np.log(np.clip(true_probability[valid], 1e-12, 1.0)))),
        "brier_score": float(np.mean(brier[valid])),
        "expected_calibration_error": float(np.sum(calibration_gap)),
        "mean_normalized_entropy": float(np.mean(
            -np.sum(
                posterior[valid]
                * np.log(np.clip(posterior[valid], 1e-12, 1.0)),
                axis=-1,
            ) / np.log(float(len(MODES))))),
        "switch_count": int(len(delays)),
        "switch_delays": delays,
        "median_switch_delay": (
            float(np.median(delays)) if delays else 0.0),
        "p90_switch_delay": (
            float(np.percentile(delays, 90)) if delays else 0.0),
    }


def audit_dir(seed: int) -> Path:
    seed = int(seed)
    if seed not in TEST_CONTROLLER_SEEDS:
        raise ValueError(f"unknown posterior test seed {seed}")
    return AUDIT_ROOT / f"seed_{seed}"


def audit_manifest(seed: int) -> Path:
    return audit_dir(seed) / "audit_manifest.json"


def analysis_json() -> Path:
    return ANALYSIS_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return ANALYSIS_ROOT / "analysis.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "size": path.stat().st_size}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_parameter_state(path: Path, state: Any) -> list[dict[str, Any]]:
    leaves = [np.asarray(value) for value in jax.tree.leaves(state)]
    arrays = {
        f"leaf_{index:05d}": value for index, value in enumerate(leaves)
    }
    metadata = [
        {
            "key": key,
            "shape": list(arrays[key].shape),
            "dtype": str(arrays[key].dtype),
        }
        for key in sorted(arrays)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return metadata


def load_parameter_state(
    path: Path,
    template: Any,
    metadata: list[dict[str, Any]],
) -> Any:
    leaves, treedef = jax.tree.flatten(template)
    with np.load(path, allow_pickle=False) as archive:
        keys = sorted(archive.files)
        if len(keys) != len(leaves) or len(keys) != len(metadata):
            raise ValueError("posterior parameter leaf count changed")
        restored = []
        for index, (key, expected, template_value) in enumerate(
                zip(keys, metadata, leaves)):
            value = np.asarray(archive[key])
            if (key != f"leaf_{index:05d}"
                    or key != expected.get("key")
                    or list(value.shape) != expected.get("shape")
                    or str(value.dtype) != expected.get("dtype")
                    or value.shape != tuple(template_value.shape)):
                raise ValueError(f"posterior parameter mismatch at {key}")
            restored.append(jnp.asarray(value, dtype=template_value.dtype))
    return jax.tree.unflatten(treedef, restored)
