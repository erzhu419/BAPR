"""Frozen protocol for the BAPR-v4 persistent-option capacity screen."""
from __future__ import annotations

import json
import hashlib
import os
import pickle
from pathlib import Path

from jax_experiments.configs.default import Config


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "bapr.v4-persistent-option-capacity.v1"
ALGO_NAME = "bapr_v4"
TRAINING_SEED = 8
FINAL_NEXT_ITERATION = 1400
FINAL_TOTAL_STEPS = 5_600_000
RUN_ROOT = (
    ROOT / "jax_experiments" / "results_bapr_v4_persistent_option_v1")
STATUS_ROOT = (
    ROOT / "jax_experiments"
    / "results_bapr_v4_persistent_option_status_v1")
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_bapr_v4_persistent_option_audit_v1")
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_bapr_v4_persistent_option_analysis_v1")
BOOTSTRAP_ROOT = (
    ROOT / "jax_experiments"
    / "results_bapr_v3_structured_channel_learned_router_v1")
BOOTSTRAP_MODEL = BOOTSTRAP_ROOT / "router_params.npz"
BOOTSTRAP_MANIFEST = BOOTSTRAP_ROOT / "router_manifest.json"
DEVELOPMENT_EVENT_SEEDS = (6100, 6200)
CONTEXT_SOURCES = ("robust", "oracle_persistent", "learned_persistent")
AUDIT_SCHEMA = "bapr.v4-persistent-option-audit-group.v1"
ANALYSIS_SCHEMA = "bapr.v4-persistent-option-analysis.v1"


def run_name(profile: str) -> str:
    if profile not in ("smoke", "formal"):
        raise ValueError(f"unsupported BAPR-v4 profile {profile!r}")
    return f"persistent_option_{profile}_seed_{TRAINING_SEED}"


def run_dir(profile: str) -> Path:
    return RUN_ROOT / run_name(profile)


def checkpoint_dir(profile: str) -> Path:
    return run_dir(profile) / "checkpoints"


def status_path(profile: str) -> Path:
    return STATUS_ROOT / profile / "complete.json"


def audit_group_path(event_seed: int) -> Path:
    if int(event_seed) not in DEVELOPMENT_EVENT_SEEDS:
        raise ValueError(f"unsupported BAPR-v4 event seed {event_seed}")
    return AUDIT_ROOT / f"event_seed_{int(event_seed)}" / "group.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, int | str]:
    return {"sha256": sha256_file(path), "size": path.stat().st_size}


def configure(profile: str) -> Config:
    formal = profile == "formal"
    if not formal and profile != "smoke":
        raise ValueError(f"unsupported BAPR-v4 profile {profile!r}")
    config = Config()
    config.algo = "bapr_v4"
    config.env_name = "HalfCheetah-v2"
    config.env_type = "stochastic_mode"
    config.stochastic_mode_family = "structured_channel"
    config.stochastic_mode_dwell_steps = 250 if formal else 32
    config.stochastic_mode_dwell_distribution = "fixed"
    config.stochastic_mode_fixed_id = -1
    config.brax_backend = "spring"
    config.task_num = 4
    config.test_task_num = 4
    config.reserved_test_task_num = 0
    config.seed = TRAINING_SEED

    config.max_iters = FINAL_NEXT_ITERATION if formal else 4
    config.samples_per_iter = 4000 if formal else 128
    config.updates_per_iter = 250 if formal else 2
    config.start_train_steps = 10_000 if formal else 128
    config.batch_size = 256 if formal else 32
    config.replay_size = 1_000_000 if formal else 4096
    config.hidden_dim = 256 if formal else 32
    config.ensemble_size = 10 if formal else 2
    config.max_episode_steps = 1000 if formal else 128

    config.bapr_v2_mode = "supervised"
    config.bapr_v2_latent_dim = 4
    config.bapr_v2_policy_context_source = "stored"
    config.bapr_v2_training_schedule = "joint"
    config.bapr_v2_context_hidden_dim = 128 if formal else 32
    config.bapr_v2_context_length = 64 if formal else 16
    config.bapr_v2_context_chunks = 8 if formal else 2
    config.bapr_v2_context_burnin = 16 if formal else 4
    config.bapr_v2_min_history = 8 if formal else 2
    config.bapr_v2_context_dropout = 0.0
    config.bapr_v2_base_aux_weight = 0.5
    config.bapr_v2_action_deviation_weight = 0.0
    config.bapr_v2_advantage_gate = False
    config.bapr_v2_actor_objective = "mean"
    config.bapr_v2_beta_ood = 0.0
    config.bapr_v2_reg_weight = 0.0

    config.bapr_v3_likelihood = "probabilistic" if formal else "point"
    config.bapr_v3_context_ensemble_size = 5 if formal else 2
    config.bapr_v3_hazard_rate = 0.005
    config.bapr_v3_evidence_scale = 0.25
    config.bapr_v3_variance_model = (
        "mode_empirical" if formal else "legacy_state")
    config.bapr_v3_variance_floor = 1e-4
    config.bapr_v3_variance_ceiling = 0.5
    config.bapr_v3_variance_ema = 0.05
    config.bapr_v3_instant_classifier_weight = 1.0
    config.bapr_v3_eval_context_ladder = True

    config.bapr_v4_option_hold_steps = 64 if formal else 8
    config.bapr_v4_option_confidence_threshold = 0.80
    config.bapr_v4_option_margin_threshold = 0.02
    config.bapr_v4_option_hysteresis_margin = 0.02
    config.bapr_v4_posterior_decay = 1.0
    config.bapr_v4_cusum_threshold = 4.0
    config.bapr_v4_cusum_drift = 0.25
    config.bapr_v4_training_source_period = 4
    config.bapr_v4_training_robust_slots = 1
    config.bapr_v4_context_bootstrap_model = (
        str(BOOTSTRAP_MODEL) if formal else "")
    config.bapr_v4_context_bootstrap_manifest = (
        str(BOOTSTRAP_MANIFEST) if formal else "")

    config.context_warmup_iters = 0
    config.eval_protocol = "full"
    config.eval_switching_period_steps = 250 if formal else 32
    config.eval_switching_episodes = 1
    config.eval_episodes = 2 if formal else 1
    config.log_interval = 100 if formal else 10_000
    config.save_interval = 50 if formal else 1
    config.save_root = str(RUN_ROOT)
    config.run_name = run_name(profile)
    config.resume = True
    config.resume_boundary_audit = False
    return config


def checkpoint_summary(profile: str) -> dict[str, int | str]:
    path = checkpoint_dir(profile) / "train_state.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"missing BAPR-v4 checkpoint state: {path}")
    with path.open("rb") as handle:
        state = pickle.load(handle)
    return {
        "algo": str(state.get("algo")),
        "saved_iteration": int(state["iteration"]),
        "next_iteration": int(state["iteration"]) + 1,
        "total_steps": int(state["total_steps"]),
    }


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
