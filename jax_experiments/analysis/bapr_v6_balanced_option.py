"""Frozen protocol for the BAPR-v6 optimizer-equivalent option screen."""
from __future__ import annotations

import pickle
from pathlib import Path

from jax_experiments.analysis import bapr_v5_hard_option as v5
from jax_experiments.configs.default import Config


ROOT = v5.ROOT
SCHEMA = "bapr.v6-balanced-option-capacity.v1"
ALGO_NAME = "bapr_v6"
TRAINING_SEED = v5.TRAINING_SEED
FINAL_NEXT_ITERATION = v5.FINAL_NEXT_ITERATION
FINAL_TOTAL_STEPS = v5.FINAL_TOTAL_STEPS
RUN_ROOT = ROOT / "jax_experiments" / "results_bapr_v6_balanced_option_v1"
STATUS_ROOT = (
    ROOT / "jax_experiments" / "results_bapr_v6_balanced_option_status_v1")
AUDIT_ROOT = (
    ROOT / "jax_experiments" / "results_bapr_v6_balanced_option_audit_v1")
ANALYSIS_ROOT = (
    ROOT / "jax_experiments" / "results_bapr_v6_balanced_option_analysis_v1")
BOOTSTRAP_MODEL = v5.BOOTSTRAP_MODEL
BOOTSTRAP_MANIFEST = v5.BOOTSTRAP_MANIFEST
DEVELOPMENT_EVENT_SEEDS = v5.DEVELOPMENT_EVENT_SEEDS
CONTEXT_SOURCES = v5.CONTEXT_SOURCES
AUDIT_SCHEMA = "bapr.v6-balanced-option-audit-group.v1"
ANALYSIS_SCHEMA = "bapr.v6-balanced-option-analysis.v1"


def run_name(profile: str) -> str:
    if profile not in ("smoke", "formal"):
        raise ValueError(f"unsupported BAPR-v6 profile {profile!r}")
    return f"balanced_option_{profile}_seed_{TRAINING_SEED}"


def run_dir(profile: str) -> Path:
    return RUN_ROOT / run_name(profile)


def checkpoint_dir(profile: str) -> Path:
    return run_dir(profile) / "checkpoints"


def status_path(profile: str) -> Path:
    return STATUS_ROOT / profile / "complete.json"


def audit_group_path(event_seed: int) -> Path:
    if int(event_seed) not in DEVELOPMENT_EVENT_SEEDS:
        raise ValueError(f"unsupported BAPR-v6 event seed {event_seed}")
    return AUDIT_ROOT / f"event_seed_{int(event_seed)}" / "group.json"


def configure(profile: str) -> Config:
    config = v5.configure(profile)
    config.algo = ALGO_NAME
    config.save_root = str(RUN_ROOT)
    config.run_name = run_name(profile)
    # Four times as many updates let each physical-mode option receive the
    # same number of replay draws as one fixed-mode specialist. The v6 batch
    # itself contains equal robust/option example counts per update.
    config.updates_per_iter = 1000 if profile == "formal" else 4
    return config


def checkpoint_summary(profile: str) -> dict[str, int | str]:
    path = checkpoint_dir(profile) / "train_state.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"missing BAPR-v6 checkpoint state: {path}")
    with path.open("rb") as handle:
        state = pickle.load(handle)
    return {
        "algo": str(state.get("algo")),
        "saved_iteration": int(state["iteration"]),
        "next_iteration": int(state["iteration"]) + 1,
        "total_steps": int(state["total_steps"]),
    }


sha256_file = v5.sha256_file
file_record = v5.file_record
write_json_atomic = v5.write_json_atomic
