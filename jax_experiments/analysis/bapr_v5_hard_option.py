"""Frozen protocol for the BAPR-v5 isolated hard-option capacity screen."""
from __future__ import annotations

import pickle
from pathlib import Path

from jax_experiments.analysis import bapr_v4_persistent_option as v4
from jax_experiments.configs.default import Config


ROOT = v4.ROOT
SCHEMA = "bapr.v5-hard-option-capacity.v1"
ALGO_NAME = "bapr_v5"
TRAINING_SEED = v4.TRAINING_SEED
FINAL_NEXT_ITERATION = v4.FINAL_NEXT_ITERATION
FINAL_TOTAL_STEPS = v4.FINAL_TOTAL_STEPS
RUN_ROOT = ROOT / "jax_experiments" / "results_bapr_v5_hard_option_v1"
STATUS_ROOT = (
    ROOT / "jax_experiments" / "results_bapr_v5_hard_option_status_v1")
AUDIT_ROOT = (
    ROOT / "jax_experiments" / "results_bapr_v5_hard_option_audit_v1")
ANALYSIS_ROOT = (
    ROOT / "jax_experiments" / "results_bapr_v5_hard_option_analysis_v1")
BOOTSTRAP_ROOT = (
    ROOT / "jax_experiments" / "analysis" / "protocol_snapshots"
    / "v5_context_bootstrap")
BOOTSTRAP_MODEL = BOOTSTRAP_ROOT / "router_params.npz"
BOOTSTRAP_MANIFEST = BOOTSTRAP_ROOT / "router_manifest.json"
DEVELOPMENT_EVENT_SEEDS = v4.DEVELOPMENT_EVENT_SEEDS
CONTEXT_SOURCES = v4.CONTEXT_SOURCES
AUDIT_SCHEMA = "bapr.v5-hard-option-audit-group.v1"
ANALYSIS_SCHEMA = "bapr.v5-hard-option-analysis.v1"


def run_name(profile: str) -> str:
    if profile not in ("smoke", "formal"):
        raise ValueError(f"unsupported BAPR-v5 profile {profile!r}")
    return f"hard_option_{profile}_seed_{TRAINING_SEED}"


def run_dir(profile: str) -> Path:
    return RUN_ROOT / run_name(profile)


def checkpoint_dir(profile: str) -> Path:
    return run_dir(profile) / "checkpoints"


def status_path(profile: str) -> Path:
    return STATUS_ROOT / profile / "complete.json"


def audit_group_path(event_seed: int) -> Path:
    if int(event_seed) not in DEVELOPMENT_EVENT_SEEDS:
        raise ValueError(f"unsupported BAPR-v5 event seed {event_seed}")
    return AUDIT_ROOT / f"event_seed_{int(event_seed)}" / "group.json"


def configure(profile: str) -> Config:
    config = v4.configure(profile)
    config.algo = ALGO_NAME
    config.save_root = str(RUN_ROOT)
    config.run_name = run_name(profile)
    # Every sampled transition is explicitly relabelled as robust and as its
    # true physical mode. The auxiliary zero-context loss would duplicate the
    # robust objective a second time and is therefore disabled.
    config.bapr_v2_base_aux_weight = 0.0
    if profile == "formal":
        config.bapr_v4_context_bootstrap_model = str(BOOTSTRAP_MODEL)
        config.bapr_v4_context_bootstrap_manifest = str(BOOTSTRAP_MANIFEST)
    # Balance robust and persistent-option behavior trajectories. Replay
    # relabelling supplies both optimization contexts from either behavior.
    config.bapr_v4_training_source_period = 2
    config.bapr_v4_training_robust_slots = 1
    return config


def checkpoint_summary(profile: str) -> dict[str, int | str]:
    path = checkpoint_dir(profile) / "train_state.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"missing BAPR-v5 checkpoint state: {path}")
    with path.open("rb") as handle:
        state = pickle.load(handle)
    return {
        "algo": str(state.get("algo")),
        "saved_iteration": int(state["iteration"]),
        "next_iteration": int(state["iteration"]) + 1,
        "total_steps": int(state["total_steps"]),
    }


sha256_file = v4.sha256_file
file_record = v4.file_record
write_json_atomic = v4.write_json_atomic
