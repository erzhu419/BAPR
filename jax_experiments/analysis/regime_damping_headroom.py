"""Registered oracle-headroom screen for persistent joint-damping modes."""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

from jax_experiments.analysis import regime_control_headroom as common
from jax_experiments.common.checkpoint import (
    _patch_flax_variablestate_unpickle,
)


ROOT = Path(os.environ.get(
    "BAPR_WORKSPACE_ROOT", Path(__file__).resolve().parents[2])).resolve()
PROTOCOL_VERSION = "v1-untouched-persistent-damping-headroom"
FAMILY = "joint_damping_fault"
ENVS = ("HalfCheetah-v2", "Ant-v2", "Hopper-v2", "Walker2d-v2")
ROLES = ("robust", "oracle")
TRAINING_SEEDS = (32_011, 32_117, 32_233)
AUDIT_EVENT_SEEDS = (132_011, 132_117, 132_233)
MODES = (0, 1, 2, 3)

MAX_ITERS = 1400
FINAL_ITERATION = MAX_ITERS - 1
SAMPLES_PER_ITER = 4000
UPDATES_PER_ITER = 250
FINAL_TOTAL_STEPS = MAX_ITERS * SAMPLES_PER_ITER
FINAL_UPDATE_COUNT = MAX_ITERS * UPDATES_PER_ITER
DWELL_STEPS = 250
MAX_EPISODE_STEPS = 1000
EPISODES_PER_TASK = 5
SWITCHING_EPISODES = 5

MIN_RELATIVE_GAIN = 0.10
MAX_TERMINATION_GAP = 0.05
MIN_MODE_WINS = 3
MIN_PASSING_ENVS = 3

RUN_ROOT = ROOT / "jax_experiments/results_regime_damping_headroom_v1"
BUNDLE_ROOT = ROOT / "jax_experiments/eval_bundles_regime_damping_headroom_v1"
AUDIT_ROOT = ROOT / "jax_experiments/results_regime_damping_headroom_audit_v1"
ANALYSIS_ROOT = (
    ROOT / "jax_experiments/results_regime_damping_headroom_analysis_v1")
REGISTRATION_ROOT = (
    ROOT / "jax_experiments/deployments/regime_damping_headroom_v1")
REGISTRATION_PATH = REGISTRATION_ROOT / "registration.json"
PROTOCOL_REPORT = (
    ROOT / "reports/regime_damping_headroom_protocol_2026-08-09.md")

BUNDLE_SCHEMA = "bapr.regime-damping-headroom-bundle.v1"
AUDIT_SCHEMA = "bapr.regime-damping-headroom-audit.v1"
ANALYSIS_SCHEMA = "bapr.regime-damping-headroom-analysis.v1"
REGISTRATION_SCHEMA = "bapr.regime-damping-headroom-registration.v1"

sha256_file = common.sha256_file
file_record = common.file_record
read_json = common.read_json
write_json_atomic = common.write_json_atomic
write_text_atomic = common.write_text_atomic


def require_env(env: str) -> str:
    env = str(env)
    if env not in ENVS:
        raise ValueError(f"unknown damping environment {env!r}")
    return env


def env_slug(env: str) -> str:
    return require_env(env).replace("-v2", "")


def require_role(role: str) -> str:
    role = str(role)
    if role not in ROLES:
        raise ValueError(f"unknown damping role {role!r}")
    return role


def require_training_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in TRAINING_SEEDS:
        raise ValueError(f"unknown damping training seed {seed}")
    return seed


def require_event_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in AUDIT_EVENT_SEEDS:
        raise ValueError(f"unknown damping event seed {seed}")
    return seed


def run_dir(env: str, role: str, seed: int) -> Path:
    return (
        RUN_ROOT / env_slug(env) / require_role(role)
        / f"seed_{require_training_seed(seed)}")


def bundle_dir(env: str, role: str, seed: int) -> Path:
    return (
        BUNDLE_ROOT / env_slug(env) / require_role(role)
        / f"seed_{require_training_seed(seed)}")


def bundle_manifest(env: str, role: str, seed: int) -> Path:
    return bundle_dir(env, role, seed) / "bundle_manifest.json"


def bundle_required_paths(
        env: str, role: str, seed: int) -> tuple[Path, ...]:
    directory = bundle_dir(env, role, seed)
    return (
        directory / "bundle_manifest.json",
        directory / "checkpoints/params.pkl",
        directory / "checkpoints/train_state.pkl",
        directory / "logs/protocol_signature.json",
    )


def audit_dir(env: str, role: str, seed: int) -> Path:
    return (
        AUDIT_ROOT / env_slug(env) / require_role(role)
        / f"seed_{require_training_seed(seed)}")


def audit_event_dir(
        env: str, role: str, seed: int, event_seed: int) -> Path:
    return (
        audit_dir(env, role, seed)
        / f"event_seed_{require_event_seed(event_seed)}")


def audit_manifest(env: str, role: str, seed: int) -> Path:
    return audit_dir(env, role, seed) / "audit_manifest.json"


def analysis_json() -> Path:
    return ANALYSIS_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return ANALYSIS_ROOT / "analysis.md"


def registration_source_paths() -> tuple[Path, ...]:
    relative = (
        "jax_experiments/envs/persistent_damping_mode_env.py",
        "jax_experiments/envs/stochastic_mode_env.py",
        "jax_experiments/envs/brax_env.py",
        "jax_experiments/analysis/regime_damping_headroom.py",
        "jax_experiments/analysis/train_regime_damping_headroom_entry.py",
        "jax_experiments/analysis/final_task_sweep_regime_damping.py",
        "jax_experiments/analysis/run_regime_damping_headroom_controller.py",
        "jax_experiments/analysis/run_regime_damping_headroom_audit.py",
        "jax_experiments/analysis/analyze_regime_damping_headroom.py",
        "jax_experiments/analysis/regime_control_headroom.py",
        "jax_experiments/analysis/run_regime_control_headroom_controller.py",
        "jax_experiments/analysis/run_regime_control_headroom_audit.py",
        "jax_experiments/analysis/analyze_regime_control_headroom.py",
        "jax_experiments/analysis/final_task_sweep.py",
        "jax_experiments/train.py",
        "jax_experiments/configs/default.py",
        "jax_experiments/common/checkpoint.py",
        "jax_experiments/common/logging.py",
        "jax_experiments/common/replay_buffer.py",
        "jax_experiments/algos/regime_sac.py",
        "jax_experiments/networks/policy.py",
        "jax_experiments/networks/ensemble_critic.py",
        "scripts/submit_regime_damping_headroom.py",
        "reports/regime_damping_headroom_protocol_2026-08-09.md",
    )
    return tuple((ROOT / path).resolve() for path in relative)


def registration_payload() -> dict[str, Any]:
    paths = registration_source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"damping registration sources missing: {missing}")
    return {
        "schema": REGISTRATION_SCHEMA,
        "status": "frozen",
        "identity": {
            "protocol_version": PROTOCOL_VERSION,
            "family": FAMILY,
            "envs": list(ENVS),
            "roles": list(ROLES),
            "training_seeds": list(TRAINING_SEEDS),
            "audit_event_seeds": list(AUDIT_EVENT_SEEDS),
            "environment_design_frozen_before_training": True,
        },
        "source_records": {
            path.relative_to(ROOT).as_posix(): file_record(path)
            for path in paths
        },
    }


def create_registration() -> dict[str, Any]:
    payload = registration_payload()
    REGISTRATION_ROOT.mkdir(parents=True, exist_ok=True)
    if REGISTRATION_PATH.exists():
        if read_json(REGISTRATION_PATH) != payload:
            raise ValueError("existing damping registration changed")
    else:
        write_json_atomic(REGISTRATION_PATH, payload)
    validate_registration()
    return payload


def validate_registration() -> dict[str, Any]:
    if not REGISTRATION_PATH.is_file():
        raise ValueError("damping headroom protocol is not registered")
    payload = read_json(REGISTRATION_PATH)
    if payload != registration_payload():
        raise ValueError("damping registration or source closure changed")
    return payload


def registration_record() -> dict[str, Any]:
    validate_registration()
    return file_record(REGISTRATION_PATH)


def identity(env: str, role: str, seed: int) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "env": require_env(env),
        "family": FAMILY,
        "role": require_role(role),
        "training_seed": require_training_seed(seed),
        "algo": "regime_sac",
        "benchmark_role": "untouched_oracle_headroom_positive_control",
        "registration": registration_record(),
    }


def checkpoint_record(directory: Path) -> dict[str, Any]:
    _patch_flax_variablestate_unpickle()
    checkpoint = directory / "checkpoints"
    with (checkpoint / "train_state.pkl").open("rb") as handle:
        state = pickle.load(handle)
    with (checkpoint / "params.pkl").open("rb") as handle:
        params = pickle.load(handle)
    iteration = int(state["iteration"])
    return {
        "iteration": iteration,
        "next_iteration": iteration + 1,
        "total_steps": int(state["total_steps"]),
        "update_count": int(params["update_count"]),
        "algo": str(state["algo"]),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(create_registration(), indent=2, sort_keys=True))

