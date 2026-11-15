"""Protocol for switch-centered causal sequence-router training."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jax_experiments.analysis import bapr_v3_sequence_router as v1


ROOT = v1.ROOT
FAMILY = v1.FAMILY
ENV = v1.ENV
TRAIN_EVENT_SEEDS = v1.TRAIN_EVENT_SEEDS
VALIDATION_EVENT_SEEDS = v1.VALIDATION_EVENT_SEEDS
HOLDOUT_EVENT_SEEDS = v1.HOLDOUT_EVENT_SEEDS
MODEL_CONFIG = v1.MODEL_CONFIG
FULL_CYCLE_DWELL_STEPS = v1.FULL_CYCLE_DWELL_STEPS
FULL_CYCLE_SEQUENCES = v1.FULL_CYCLE_SEQUENCES
emission = v1.emission
utility = v1.utility

MODEL_ROOT = (
    ROOT / "jax_experiments"
    / "results_bapr_v3_structured_channel_sequence_router_v2"
)
MODEL_PATH = MODEL_ROOT / "sequence_router_params.npz"
MANIFEST_PATH = MODEL_ROOT / "sequence_router_manifest.json"
TRAIN_STATE_META_PATH = MODEL_ROOT / "train_state.json"
DATASET_PATH = MODEL_ROOT / "prepared_evidence.npz"
SCHEMA = "bapr.v3-sequence-router.v2"
TRAIN_STATE_SCHEMA = "bapr.v3-sequence-router-curriculum-state.v2"

VARIANTS = {
    "uniform_es": {
        "context_length": 128,
        "switch_fraction": 0.0,
        "switch_weight": 3.0,
        "switch_span": 64,
        "learning_rate": 3e-4,
        "updates": 2500,
    },
    "s50w6c128": {
        "context_length": 128,
        "switch_fraction": 0.50,
        "switch_weight": 6.0,
        "switch_span": 32,
        "learning_rate": 2e-4,
        "updates": 3000,
    },
    "s75w8c128": {
        "context_length": 128,
        "switch_fraction": 0.75,
        "switch_weight": 8.0,
        "switch_span": 32,
        "learning_rate": 2e-4,
        "updates": 3000,
    },
    "s50w8c64": {
        "context_length": 64,
        "switch_fraction": 0.50,
        "switch_weight": 8.0,
        "switch_span": 32,
        "learning_rate": 2e-4,
        "updates": 3000,
    },
}


def configure() -> None:
    v1.configure()


def file_record(path: Path) -> dict[str, Any]:
    return v1.file_record(path)


def write_json_atomic(path: Path, payload: object) -> None:
    v1.write_json_atomic(path, payload)


def load_manifest() -> dict[str, Any]:
    configure()
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    table = utility.load_utility_table()
    emission_manifest = emission.load_manifest()
    if (payload.get("schema") != SCHEMA
            or payload.get("status") != "complete"
            or payload.get("family") != FAMILY
            or payload.get("env") != ENV
            or payload.get("model_config") != MODEL_CONFIG
            or payload.get("variants") != VARIANTS
            or tuple(payload.get("train_event_seeds", ()))
            != TRAIN_EVENT_SEEDS
            or tuple(payload.get("validation_event_seeds", ()))
            != VALIDATION_EVENT_SEEDS
            or tuple(payload.get("holdout_event_seeds", ()))
            != HOLDOUT_EVENT_SEEDS
            or payload.get("parameter_file") != file_record(MODEL_PATH)
            or payload.get("emission_manifest_file")
            != file_record(emission.MANIFEST_PATH)
            or payload.get("emission_parameter_file")
            != file_record(emission.MODEL_PATH)
            or payload.get("utility_table_file")
            != file_record(utility.TABLE_PATH)
            or payload.get("oracle_controller_map")
            != table["oracle_controller_map"]):
        raise ValueError("invalid or stale switch-centered sequence manifest")
    return payload
