"""Pre-training CLI amendment for the persistent-damping headroom screen."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import regime_damping_headroom as predecessor


ROOT = predecessor.ROOT
PROTOCOL_VERSION = "v2-pretraining-cli-amendment-persistent-damping-headroom"
FAMILY = predecessor.FAMILY
ENVS = predecessor.ENVS
ROLES = predecessor.ROLES
TRAINING_SEEDS = predecessor.TRAINING_SEEDS
AUDIT_EVENT_SEEDS = predecessor.AUDIT_EVENT_SEEDS
MODES = predecessor.MODES

MAX_ITERS = predecessor.MAX_ITERS
FINAL_ITERATION = predecessor.FINAL_ITERATION
SAMPLES_PER_ITER = predecessor.SAMPLES_PER_ITER
UPDATES_PER_ITER = predecessor.UPDATES_PER_ITER
FINAL_TOTAL_STEPS = predecessor.FINAL_TOTAL_STEPS
FINAL_UPDATE_COUNT = predecessor.FINAL_UPDATE_COUNT
DWELL_STEPS = predecessor.DWELL_STEPS
MAX_EPISODE_STEPS = predecessor.MAX_EPISODE_STEPS
EPISODES_PER_TASK = predecessor.EPISODES_PER_TASK
SWITCHING_EPISODES = predecessor.SWITCHING_EPISODES

MIN_RELATIVE_GAIN = predecessor.MIN_RELATIVE_GAIN
MAX_TERMINATION_GAP = predecessor.MAX_TERMINATION_GAP
MIN_MODE_WINS = predecessor.MIN_MODE_WINS
MIN_PASSING_ENVS = predecessor.MIN_PASSING_ENVS

RUN_ROOT = ROOT / "jax_experiments/results_regime_damping_headroom_v2"
BUNDLE_ROOT = ROOT / "jax_experiments/eval_bundles_regime_damping_headroom_v2"
AUDIT_ROOT = ROOT / "jax_experiments/results_regime_damping_headroom_audit_v2"
ANALYSIS_ROOT = (
    ROOT / "jax_experiments/results_regime_damping_headroom_analysis_v2")
REGISTRATION_ROOT = (
    ROOT / "jax_experiments/deployments/regime_damping_headroom_v2")
REGISTRATION_PATH = REGISTRATION_ROOT / "registration.json"
PROTOCOL_REPORT = (
    ROOT / "reports/regime_damping_headroom_v2_amendment_2026-08-09.md")

BUNDLE_SCHEMA = "bapr.regime-damping-headroom-bundle.v2"
AUDIT_SCHEMA = "bapr.regime-damping-headroom-audit.v2"
ANALYSIS_SCHEMA = "bapr.regime-damping-headroom-analysis.v2"
REGISTRATION_SCHEMA = "bapr.regime-damping-headroom-registration.v2"

file_record = predecessor.file_record
read_json = predecessor.read_json
write_json_atomic = predecessor.write_json_atomic
write_text_atomic = predecessor.write_text_atomic
checkpoint_record = predecessor.checkpoint_record
require_env = predecessor.require_env
env_slug = predecessor.env_slug
require_role = predecessor.require_role
require_training_seed = predecessor.require_training_seed
require_event_seed = predecessor.require_event_seed

FAILED_PREDECESSOR_TASKS = tuple(f"t{task}" for task in range(79103, 79127))


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
        "pretraining_amendment_only": True,
    }


def registration_source_paths() -> tuple[Path, ...]:
    new = (
        ROOT / "jax_experiments/analysis/regime_damping_headroom_v2.py",
        ROOT / "jax_experiments/analysis/train_regime_damping_headroom_entry_v2.py",
        ROOT / "jax_experiments/analysis/run_regime_damping_headroom_controller_v2.py",
        ROOT / "jax_experiments/analysis/run_regime_damping_headroom_audit_v2.py",
        ROOT / "jax_experiments/analysis/analyze_regime_damping_headroom_v2.py",
        ROOT / "scripts/submit_regime_damping_headroom_v2.py",
        ROOT / "jax_experiments/tests/test_regime_damping_headroom_v2.py",
        PROTOCOL_REPORT,
        predecessor.REGISTRATION_PATH,
    )
    return tuple(dict.fromkeys(
        path.resolve()
        for path in (*new, *predecessor.registration_source_paths())))


def registration_payload() -> dict[str, Any]:
    paths = registration_source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"damping-v2 registration sources missing: {missing}")
    predecessor.validate_registration()
    predecessor_checkpoints = tuple(predecessor.RUN_ROOT.rglob(
        "checkpoints/train_state.pkl")) if predecessor.RUN_ROOT.exists() else ()
    if predecessor_checkpoints:
        raise ValueError(
            "v1 unexpectedly produced checkpoints; v2 amendment is not valid")
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
        "amendment": {
            "predecessor_registration": file_record(
                predecessor.REGISTRATION_PATH),
            "failed_predecessor_tasks": list(FAILED_PREDECESSOR_TASKS),
            "failure_stage": "argparse_before_environment_or_checkpoint",
            "failure": (
                "joint_damping_fault was omitted from the generic CLI "
                "choice list"),
            "scientific_changes": [],
            "engineering_change": (
                "accept the already registered family in the isolated entry"),
            "predecessor_checkpoint_count": 0,
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
            raise ValueError("existing damping-v2 registration changed")
    else:
        write_json_atomic(REGISTRATION_PATH, payload)
    validate_registration()
    return payload


def validate_registration() -> dict[str, Any]:
    if not REGISTRATION_PATH.is_file():
        raise ValueError("damping-v2 headroom protocol is not registered")
    payload = read_json(REGISTRATION_PATH)
    if payload != registration_payload():
        raise ValueError("damping-v2 registration or source closure changed")
    return payload


def registration_record() -> dict[str, Any]:
    validate_registration()
    return file_record(REGISTRATION_PATH)

