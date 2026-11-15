"""Protocol for a causal sequence filter over the frozen BAPR-v3 emissions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jax_experiments.analysis import bapr_v3_learned_control_router as emission
from jax_experiments.analysis import bapr_v3_utility_aware_router as utility


ROOT = utility.ROOT
FAMILY = utility.FAMILY
ENV = utility.ENV
TRAIN_EVENT_SEEDS = (8100, 8200)
VALIDATION_EVENT_SEEDS = (9100, 9200)
HOLDOUT_EVENT_SEEDS = utility.HOLDOUT_EVENT_SEEDS

MODEL_ROOT = (
    ROOT / "jax_experiments"
    / "results_bapr_v3_structured_channel_sequence_router_v1"
)
MODEL_PATH = MODEL_ROOT / "sequence_router_params.npz"
MANIFEST_PATH = MODEL_ROOT / "sequence_router_manifest.json"
TRAIN_STATE_META_PATH = MODEL_ROOT / "train_state.json"

SCHEMA = "bapr.v3-sequence-router.v1"
MODEL_CONFIG = {
    "num_modes": 4,
    "hidden_dim": 64,
    "evidence_clip": 6.0,
}
FULL_CYCLE_DWELL_STEPS = 250
FULL_CYCLE_SEQUENCES = (
    (0, 1, 2, 3),
    (3, 2, 1, 0),
    (0, 2, 1, 3),
    (1, 3, 0, 2),
    (2, 0, 3, 1),
)


def configure() -> None:
    utility.configure()


def file_record(path: Path) -> dict[str, Any]:
    return emission.file_record(path)


def write_json_atomic(path: Path, payload: object) -> None:
    emission.write_json_atomic(path, payload)


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
        raise ValueError("invalid or stale causal sequence-router manifest")
    return payload
