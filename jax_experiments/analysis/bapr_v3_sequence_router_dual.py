"""Protocol for a privileged two-timescale sequence-router oracle ladder."""
from __future__ import annotations

from pathlib import Path

from jax_experiments.analysis import bapr_v3_sequence_router_v2 as slow


ROOT = slow.ROOT
FAMILY = slow.FAMILY
ENV = slow.ENV
TRAIN_EVENT_SEEDS = slow.TRAIN_EVENT_SEEDS
VALIDATION_EVENT_SEEDS = slow.VALIDATION_EVENT_SEEDS
HOLDOUT_EVENT_SEEDS = slow.HOLDOUT_EVENT_SEEDS
MODEL_CONFIG = slow.MODEL_CONFIG
emission = slow.emission
utility = slow.utility

MODEL_ROOT = (
    ROOT / "jax_experiments"
    / "results_bapr_v3_structured_channel_sequence_router_dual_oracle_v1"
)
DATASET_PATH = MODEL_ROOT / "prepared_evidence.npz"
SLOW_MODEL_PATH = MODEL_ROOT / "slow_router_params.npz"
FAST_MODEL_PATH = MODEL_ROOT / "fast_router_params.npz"
MANIFEST_PATH = MODEL_ROOT / "dual_oracle_manifest.json"
TRAIN_STATE_META_PATH = MODEL_ROOT / "train_state.json"
SCHEMA = "bapr.v3-sequence-router-dual-oracle.v1"
TRAIN_STATE_SCHEMA = "bapr.v3-sequence-router-dual-oracle-state.v1"

WINDOWS = (8, 16, 24, 32, 48, 64, 96)
FAST_VARIANTS = {
    "fast_s25w4": {
        "context_length": 128,
        "switch_fraction": 0.25,
        "switch_weight": 4.0,
        "switch_span": 32,
        "learning_rate": 2e-4,
        "updates": 3000,
    },
    "fast_s50w6": {
        "context_length": 128,
        "switch_fraction": 0.50,
        "switch_weight": 6.0,
        "switch_span": 32,
        "learning_rate": 2e-4,
        "updates": 3000,
    },
    "fast_s75w8": {
        "context_length": 128,
        "switch_fraction": 0.75,
        "switch_weight": 8.0,
        "switch_span": 32,
        "learning_rate": 2e-4,
        "updates": 3000,
    },
}


def configure() -> None:
    slow.configure()


def file_record(path: Path):
    return slow.file_record(path)


def write_json_atomic(path: Path, payload):
    slow.write_json_atomic(path, payload)
