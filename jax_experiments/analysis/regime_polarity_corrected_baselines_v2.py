"""Registered corrected-baseline comparison for frozen polarity BAPR."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_fallback_final_comparison_v1 as frozen,
)


ROOT = Path(os.environ.get(
    "BAPR_WORKSPACE_ROOT", Path(__file__).resolve().parents[2])).resolve()
PROTOCOL_VERSION = "v2-recurrent-escp-released-b0-corrected-baselines"
ENV = frozen.ENV
FAMILY = frozen.FAMILY
MODES = frozen.MODES
TRAINING_SEEDS = frozen.TRAINING_SEEDS
EVENT_SEEDS = frozen.FINAL_EVENT_SEEDS
REUSED_METHODS = ("bapr", "sac")
TRAINED_METHODS = ("escp_recurrent", "resac_b0")
METHODS = (*REUSED_METHODS, *TRAINED_METHODS)

MAX_ITERS = frozen.MAX_ITERS
FINAL_ITERATION = MAX_ITERS - 1
SAMPLES_PER_ITER = frozen.SAMPLES_PER_ITER
UPDATES_PER_ITER = frozen.UPDATES_PER_ITER
FINAL_TOTAL_STEPS = frozen.FINAL_TOTAL_STEPS
FINAL_UPDATE_COUNT = frozen.FINAL_UPDATE_COUNT
START_TRAIN_STEPS = frozen.START_TRAIN_STEPS
MAX_EPISODE_STEPS = frozen.MAX_EPISODE_STEPS
DWELL_STEPS = frozen.DWELL_STEPS
AUDIT_EPISODES_PER_TASK = frozen.AUDIT_EPISODES_PER_TASK
AUDIT_SWITCHING_EPISODES = frozen.AUDIT_SWITCHING_EPISODES

ESCP_CONFIG = {
    "ensemble_size": 2,
    "ep_dim": 2,
    "history_length": 16,
    "policy_lr": 3e-4,
    "critic_lr": 1e-3,
    "context_lr": 3e-4,
    "alpha_lr": 1e-2,
    "target_entropy_ratio": 1.5,
    "bottleneck_sigma": 1e-2,
    "prototype_tau": 0.995,
    "context_min_steps": 100_000,
    "context_min_tasks": 2,
    "rbf_radius": 80.0,
    "consistency_weight": 50.0,
    "diversity_weight": 0.025,
    "clip_norm": 0.0,
}
RESAC_CONFIG = {
    "ensemble_size": 5,
    "lr": 3e-4,
    "beta_start": -2.0,
    "beta_end": 0.0,
    "beta_warmup": 0.2,
    "independent_ratio": 0.75,
    "anchor_lambda": 0.001,
    "ema_tau": 0.005,
}

RUN_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_corrected_baseline_runs_v2")
BUNDLE_ROOT = (
    ROOT / "jax_experiments"
    / "eval_bundles_regime_polarity_corrected_baselines_v2")
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_corrected_baseline_audit_v2")
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_corrected_baseline_analysis_v2")
REGISTRATION_ROOT = (
    ROOT / "jax_experiments" / "deployments"
    / "regime_polarity_corrected_baselines_v2")
REGISTRATION_PATH = REGISTRATION_ROOT / "registration.json"
REPORT = (
    ROOT / "reports"
    / "regime_polarity_corrected_baselines_v2_2026-08-09.md")

BUNDLE_SCHEMA = "bapr.regime-polarity-corrected-baseline-bundle.v2"
AUDIT_SCHEMA = "bapr.regime-polarity-corrected-baseline-audit.v2"
ANALYSIS_SCHEMA = "bapr.regime-polarity-corrected-baseline-analysis.v2"
REGISTRATION_SCHEMA = (
    "bapr.regime-polarity-corrected-baseline-registration.v2")

file_record = frozen.file_record
read_json = frozen.read_json
write_json_atomic = frozen.write_json_atomic
write_text_atomic = frozen.write_text_atomic
checkpoint_record = frozen.checkpoint_record

# Immutable result records from the original final audit. These are reused as
# observations only; the old mutable source tree is not trusted at runtime.
REUSED_RESULT_RECORDS = {
    ("bapr", 2009): {"sha256": "3fb58a41a561c6ba0d515ae8fc5f192bfd0c28f289435b9e386b7acf1ff70210", "size": 10908},
    ("bapr", 2113): {"sha256": "958ca9f0442608087e3c00eb7f75bc81bf081e108935816814cc4b19006fb0f6", "size": 10983},
    ("bapr", 2213): {"sha256": "dd6aa3b1456712a7cd4c63b950e576833c1fbd534395727d5f91cad9403bd786", "size": 10858},
    ("bapr", 2311): {"sha256": "cb643fea82ff47d82b0393ac2a49e0ce5ebdc0a493c4289c598ae461e73f45df", "size": 10902},
    ("bapr", 2417): {"sha256": "11ec68363117871011b80f1644459e9e42bfca7643f0e2b31ad7b4698ba49fa5", "size": 10916},
    ("sac", 2009): {"sha256": "d84de7ecb7852925dcf7f6fcc0880730e0a42942e7eeb7615de763c6e0fd0b1c", "size": 9441},
    ("sac", 2113): {"sha256": "5bd7123e90933e243807d0086c67be67dd33b4ff339efc6ca4a0a9e8904e477d", "size": 9470},
    ("sac", 2213): {"sha256": "8ca2b538777e4713b3892cc8195a8295ad512a2231835200facf4f70fe91b517", "size": 9432},
    ("sac", 2311): {"sha256": "b91e066d263fb998c62e55cdd293c1e04b1c02d8f17ba020a95666aef15ce7c5", "size": 9411},
    ("sac", 2417): {"sha256": "9ba21cd251b0189c6d189ac35681074c66c41c9b4b42c4873c28c9fefb0b39a1", "size": 9392},
}


def require_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in TRAINING_SEEDS:
        raise ValueError(f"unknown corrected-baseline seed {seed}")
    return seed


def require_trained_method(method: str) -> str:
    method = str(method)
    if method not in TRAINED_METHODS:
        raise ValueError(f"unknown trained corrected baseline {method!r}")
    return method


def require_method(method: str) -> str:
    method = str(method)
    if method not in METHODS:
        raise ValueError(f"unknown corrected comparison method {method!r}")
    return method


def algo_for(method: str) -> str:
    method = require_trained_method(method)
    return "escp" if method == "escp_recurrent" else "resac"


def run_dir(method: str, seed: int) -> Path:
    return RUN_ROOT / require_trained_method(method) / f"seed_{require_seed(seed)}"


def bundle_dir(method: str, seed: int) -> Path:
    return BUNDLE_ROOT / require_trained_method(method) / f"seed_{require_seed(seed)}"


def bundle_manifest(method: str, seed: int) -> Path:
    return bundle_dir(method, seed) / "bundle_manifest.json"


def bundle_required_paths(method: str, seed: int) -> tuple[Path, ...]:
    directory = bundle_dir(method, seed)
    return (
        directory / "bundle_manifest.json",
        directory / "checkpoints" / "params.pkl",
        directory / "checkpoints" / "train_state.pkl",
        directory / "logs" / "protocol_signature.json",
    )


def audit_dir(method: str, seed: int) -> Path:
    return AUDIT_ROOT / require_trained_method(method) / f"seed_{require_seed(seed)}"


def audit_manifest(method: str, seed: int) -> Path:
    return audit_dir(method, seed) / "audit_manifest.json"


def reused_result_path(method: str, seed: int) -> Path:
    if method not in REUSED_METHODS:
        raise ValueError(f"{method!r} is not a reused method")
    return frozen.audit_dir(method, require_seed(seed)) / "results.json"


def validate_reused_result(method: str, seed: int) -> Path:
    path = reused_result_path(method, seed)
    expected = REUSED_RESULT_RECORDS[(method, require_seed(seed))]
    if not path.is_file() or file_record(path) != expected:
        raise ValueError(f"frozen reused audit changed: {path}")
    payload = read_json(path)
    if (payload.get("status") != "complete"
            or payload.get("identity", {}).get("method") != method
            or payload.get("identity", {}).get("seed") != int(seed)
            or payload.get("identity", {}).get("event_seeds")
            != list(EVENT_SEEDS)):
        raise ValueError(f"invalid reused audit identity: {path}")
    return path


def identity(method: str, seed: int) -> dict[str, Any]:
    method = require_trained_method(method)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "corrected_same_budget_baseline",
        "method": method,
        "algo": algo_for(method),
        "training_seed": require_seed(seed),
        "env": ENV,
        "family": FAMILY,
        "max_iters": MAX_ITERS,
        "total_steps": FINAL_TOTAL_STEPS,
        "update_count": FINAL_UPDATE_COUNT,
        "controller_budget_match": True,
    }


def expected_checkpoint(method: str) -> dict[str, Any]:
    return {
        "iteration": FINAL_ITERATION,
        "next_iteration": MAX_ITERS,
        "total_steps": FINAL_TOTAL_STEPS,
        "update_count": FINAL_UPDATE_COUNT,
        "algo": algo_for(method),
    }


def audit_identity(method: str, seed: int) -> dict[str, Any]:
    return {
        **identity(method, seed),
        "event_seeds": list(EVENT_SEEDS),
        "strict_horizon": MAX_EPISODE_STEPS,
        "dwell_steps": DWELL_STEPS,
        "stationary_episodes_per_mode": AUDIT_EPISODES_PER_TASK,
        "switching_episodes": AUDIT_SWITCHING_EPISODES,
        "evaluation_policy": "deterministic_mean",
    }


def analysis_json() -> Path:
    return ANALYSIS_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return ANALYSIS_ROOT / "analysis.md"


def all_new_audit_manifests() -> tuple[Path, ...]:
    return tuple(
        audit_manifest(method, seed)
        for method in TRAINED_METHODS
        for seed in TRAINING_SEEDS)


def registration_source_paths() -> tuple[Path, ...]:
    return (
        ROOT / "jax_experiments/analysis/regime_polarity_corrected_baselines_v2.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_corrected_baseline_v2.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_corrected_audit_v2.py",
        ROOT / "jax_experiments/analysis/analyze_regime_polarity_corrected_baselines_v2.py",
        ROOT / "jax_experiments/algos/escp.py",
        ROOT / "jax_experiments/algos/resac.py",
        ROOT / "jax_experiments/networks/escp_recurrent.py",
        ROOT / "jax_experiments/common/replay_buffer.py",
        ROOT / "jax_experiments/common/checkpoint.py",
        ROOT / "jax_experiments/envs/brax_env.py",
        ROOT / "jax_experiments/configs/default.py",
        ROOT / "jax_experiments/train.py",
        ROOT / "scripts/submit_regime_polarity_corrected_baselines_v2.py",
    )


def registration_payload(*, validate_reused: bool = True) -> dict[str, Any]:
    reused = {}
    for method in REUSED_METHODS:
        for seed in TRAINING_SEEDS:
            path = (
                validate_reused_result(method, seed)
                if validate_reused else reused_result_path(method, seed))
            reused[f"{method}/seed_{seed}"] = {
                "path": str(path.relative_to(ROOT)),
                **REUSED_RESULT_RECORDS[(method, seed)],
            }
    return {
        "schema": REGISTRATION_SCHEMA,
        "status": "registered",
        "protocol_version": PROTOCOL_VERSION,
        "created_before_new_training": True,
        "environment": {
            "env": ENV,
            "family": FAMILY,
            "modes": list(MODES),
            "dwell_steps": DWELL_STEPS,
        },
        "budget": {
            "max_iters": MAX_ITERS,
            "samples_per_iter": SAMPLES_PER_ITER,
            "updates_per_iter": UPDATES_PER_ITER,
            "total_steps": FINAL_TOTAL_STEPS,
            "update_count": FINAL_UPDATE_COUNT,
        },
        "training_seeds": list(TRAINING_SEEDS),
        "event_seeds": list(EVENT_SEEDS),
        "escp_config": ESCP_CONFIG,
        "resac_config": RESAC_CONFIG,
        "reused_results": reused,
        "source_files": {
            str(path.relative_to(ROOT)): file_record(path)
            for path in registration_source_paths()
        },
        "decision_rule": (
            "compare frozen BAPR against the strongest of unchanged SAC, "
            "recurrent ESCP, and released-B0 RE-SAC on identical event streams"
        ),
    }


def create_registration() -> dict[str, Any]:
    payload = registration_payload(validate_reused=True)
    if REGISTRATION_PATH.is_file():
        existing = read_json(REGISTRATION_PATH)
        if existing != payload:
            raise ValueError("corrected-baseline registration already changed")
        return existing
    REGISTRATION_ROOT.mkdir(parents=True, exist_ok=True)
    write_json_atomic(REGISTRATION_PATH, payload)
    return payload


def validate_registration() -> dict[str, Any]:
    if not REGISTRATION_PATH.is_file():
        raise FileNotFoundError(
            f"missing corrected-baseline registration: {REGISTRATION_PATH}")
    payload = read_json(REGISTRATION_PATH)
    if payload != registration_payload(validate_reused=False):
        raise ValueError("corrected-baseline registration changed")
    return payload
