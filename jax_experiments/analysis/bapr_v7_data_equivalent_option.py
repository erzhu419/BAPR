"""Frozen protocol for the BAPR-v7 unique-data-equivalent option screen."""
from __future__ import annotations

import pickle

from jax_experiments.analysis import bapr_v6_balanced_option as v6
from jax_experiments.configs.default import Config


ROOT = v6.ROOT
SCHEMA = "bapr.v7-data-equivalent-option-capacity.v1"
# V7 changes only the data/replay budget; the controller implementation is v6.
ALGO_NAME = v6.ALGO_NAME
TRAINING_SEED = v6.TRAINING_SEED
FINAL_NEXT_ITERATION = 5600
SAMPLES_PER_ITER = 4000
FINAL_TOTAL_STEPS = FINAL_NEXT_ITERATION * SAMPLES_PER_ITER
RUN_ROOT = ROOT / "jax_experiments" / "results_bapr_v7_data_equivalent_option_v1"
STATUS_ROOT = (
    ROOT / "jax_experiments" / "results_bapr_v7_data_equivalent_option_status_v1")
AUDIT_ROOT = (
    ROOT / "jax_experiments" / "results_bapr_v7_data_equivalent_option_audit_v1")
ANALYSIS_ROOT = (
    ROOT / "jax_experiments" / "results_bapr_v7_data_equivalent_option_analysis_v1")
BOOTSTRAP_MODEL = v6.BOOTSTRAP_MODEL
BOOTSTRAP_MANIFEST = v6.BOOTSTRAP_MANIFEST
DEVELOPMENT_EVENT_SEEDS = v6.DEVELOPMENT_EVENT_SEEDS
CONTEXT_SOURCES = v6.CONTEXT_SOURCES
AUDIT_SCHEMA = "bapr.v7-data-equivalent-option-audit-group.v1"
ANALYSIS_SCHEMA = "bapr.v7-data-equivalent-option-analysis.v1"


def run_name(profile: str) -> str:
    if profile not in ("smoke", "formal"):
        raise ValueError(f"unsupported BAPR-v7 profile {profile!r}")
    return f"data_equivalent_option_{profile}_seed_{TRAINING_SEED}"


def run_dir(profile: str):
    return RUN_ROOT / run_name(profile)


def checkpoint_dir(profile: str):
    return run_dir(profile) / "checkpoints"


def status_path(profile: str):
    return STATUS_ROOT / profile / "complete.json"


def audit_group_path(event_seed: int):
    if int(event_seed) not in DEVELOPMENT_EVENT_SEEDS:
        raise ValueError(f"unsupported BAPR-v7 event seed {event_seed}")
    return AUDIT_ROOT / f"event_seed_{int(event_seed)}" / "group.json"


def configure(profile: str) -> Config:
    # Use the formal architecture for smoke as well so its observed VRAM is a
    # valid calibration sample for the file-gated formal/audit tasks.
    config = v6.configure("formal")
    config.save_root = str(RUN_ROOT)
    config.run_name = run_name(profile)
    config.samples_per_iter = SAMPLES_PER_ITER
    config.replay_size = 4_000_000
    config.max_iters = FINAL_NEXT_ITERATION if profile == "formal" else 4
    # Across 5600 iterations this gives every option the same total replay
    # draws as v6 and a standalone specialist, while exposing it to four times
    # as many unique switching transitions.
    config.updates_per_iter = 250 if profile == "formal" else 4
    config.save_interval = 200 if profile == "formal" else 1
    config.log_interval = 100 if profile == "formal" else 10_000
    config.eval_episodes = 2 if profile == "formal" else 1
    config.resume = True
    return config


def checkpoint_summary(profile: str):
    path = checkpoint_dir(profile) / "train_state.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"missing BAPR-v7 checkpoint state: {path}")
    with path.open("rb") as handle:
        state = pickle.load(handle)
    return {
        "algo": str(state.get("algo")),
        "saved_iteration": int(state["iteration"]),
        "next_iteration": int(state["iteration"]) + 1,
        "total_steps": int(state["total_steps"]),
    }


sha256_file = v6.sha256_file
file_record = v6.file_record
write_json_atomic = v6.write_json_atomic
