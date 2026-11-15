"""Protocol for a causal gate over frozen slow and fast mode filters."""
from __future__ import annotations

from pathlib import Path

from jax_experiments.analysis import bapr_v3_sequence_router_dual as dual


ROOT = dual.ROOT
FAMILY = dual.FAMILY
ENV = dual.ENV
TRAIN_EVENT_SEEDS = dual.TRAIN_EVENT_SEEDS
VALIDATION_EVENT_SEEDS = dual.VALIDATION_EVENT_SEEDS
HOLDOUT_EVENT_SEEDS = dual.HOLDOUT_EVENT_SEEDS
emission = dual.emission
utility = dual.utility

MODEL_ROOT = (
    ROOT / "jax_experiments"
    / "results_bapr_v3_structured_channel_sequence_router_gate_v1"
)
DATASET_PATH = MODEL_ROOT / "prepared_evidence.npz"
MODEL_PATH = MODEL_ROOT / "gate_params.npz"
MANIFEST_PATH = MODEL_ROOT / "gate_manifest.json"
TRAIN_STATE_META_PATH = MODEL_ROOT / "train_state.json"
SCHEMA = "bapr.v3-sequence-router-causal-gate.v1"
TRAIN_STATE_SCHEMA = "bapr.v3-sequence-router-causal-gate-state.v1"

FEATURE_NAMES = (
    *(f"slow_p{mode}" for mode in range(4)),
    *(f"fast_p{mode}" for mode in range(4)),
    "slow_confidence", "slow_margin", "slow_entropy",
    "fast_confidence", "fast_margin", "fast_entropy",
    *(f"absolute_pdiff{mode}" for mode in range(4)),
    "symmetric_kl", "controller_disagreement", "normalized_log_count",
)
MODEL_CONFIG = {"input_dim": len(FEATURE_NAMES), "hidden_dim": 32}
TRAINING_SEEDS = (0, 1, 2, 3, 4)
THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


def configure() -> None:
    dual.configure()


def file_record(path: Path):
    return dual.file_record(path)


def write_json_atomic(path: Path, payload):
    dual.write_json_atomic(path, payload)
