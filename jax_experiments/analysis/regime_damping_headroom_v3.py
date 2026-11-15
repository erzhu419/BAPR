"""Evaluation-only validator amendment for persistent-damping headroom."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import regime_damping_headroom_v2 as predecessor


ROOT = predecessor.ROOT
PROTOCOL_VERSION = "v3-robust-trace-validator-amendment"
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

# final_task_sweep encodes the sealed all-zero robust context with -1. The v2
# validator omitted this sentinel and therefore compared robust traces against
# the dynamic physics mode, rejecting correct evaluations after rollout.
ROBUST_TRACE_CONTEXT_MODE_ID = -1

RUN_ROOT = predecessor.RUN_ROOT
BUNDLE_ROOT = predecessor.BUNDLE_ROOT
AUDIT_ROOT = ROOT / "jax_experiments/results_regime_damping_headroom_audit_v3"
ANALYSIS_ROOT = (
    ROOT / "jax_experiments/results_regime_damping_headroom_analysis_v3")
REGISTRATION_ROOT = (
    ROOT / "jax_experiments/deployments/regime_damping_headroom_v3")
REGISTRATION_PATH = REGISTRATION_ROOT / "registration.json"
PROTOCOL_REPORT = (
    ROOT / "reports/regime_damping_headroom_v3_amendment_2026-08-26.md")

BUNDLE_SCHEMA = predecessor.BUNDLE_SCHEMA
AUDIT_SCHEMA = "bapr.regime-damping-headroom-audit.v3"
ANALYSIS_SCHEMA = "bapr.regime-damping-headroom-analysis.v3"
REGISTRATION_SCHEMA = "bapr.regime-damping-headroom-registration.v3"

file_record = predecessor.file_record
read_json = predecessor.read_json
write_json_atomic = predecessor.write_json_atomic
write_text_atomic = predecessor.write_text_atomic
require_env = predecessor.require_env
env_slug = predecessor.env_slug
require_role = predecessor.require_role
require_training_seed = predecessor.require_training_seed
require_event_seed = predecessor.require_event_seed

FAILED_V2_ROBUST_AUDITS = (
    "t79271", "t79272", "t79273",
    "t79277", "t79278", "t79279",
    "t79283", "t79284", "t79285",
    "t79289", "t79290", "t79291",
)
SUPERSEDED_V2_ANALYSIS_TASK = "t79295"


def run_dir(env: str, role: str, seed: int) -> Path:
    return predecessor.run_dir(env, role, seed)


def bundle_dir(env: str, role: str, seed: int) -> Path:
    return predecessor.bundle_dir(env, role, seed)


def bundle_manifest(env: str, role: str, seed: int) -> Path:
    return predecessor.bundle_manifest(env, role, seed)


def bundle_required_paths(
        env: str, role: str, seed: int) -> tuple[Path, ...]:
    return predecessor.bundle_required_paths(env, role, seed)


def audit_dir(env: str, role: str, seed: int) -> Path:
    env = require_env(env)
    role = require_role(role)
    seed = require_training_seed(seed)
    if role == "oracle":
        return predecessor.audit_dir(env, role, seed)
    return AUDIT_ROOT / env_slug(env) / role / f"seed_{seed}"


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
        "evaluation_contract_amendment_only": True,
    }


def registration_source_paths() -> tuple[Path, ...]:
    new = (
        ROOT / "jax_experiments/analysis/regime_damping_headroom_v3.py",
        ROOT / "jax_experiments/analysis/run_regime_damping_headroom_audit_v3.py",
        ROOT / "jax_experiments/analysis/analyze_regime_damping_headroom_v3.py",
        ROOT / "scripts/submit_regime_damping_headroom_v3.py",
        ROOT / "jax_experiments/tests/test_regime_damping_headroom_v3.py",
        PROTOCOL_REPORT,
        predecessor.REGISTRATION_PATH,
    )
    return tuple(dict.fromkeys(
        path.resolve()
        for path in (*new, *predecessor.registration_source_paths())))


def _verify_manifest_tree(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("status") != "complete":
        raise ValueError(f"predecessor artifact is incomplete: {path}")
    records = payload.get("files") or {}
    if not records:
        raise ValueError(f"predecessor artifact has no file records: {path}")
    for relative, expected in records.items():
        candidate = path.parent / relative
        if not candidate.is_file() or file_record(candidate) != expected:
            raise ValueError(f"predecessor artifact changed: {candidate}")
    return payload


def _predecessor_artifact_records() -> dict[str, dict[str, Any]]:
    records = {}
    for env in ENVS:
        for seed in TRAINING_SEEDS:
            for role in ROLES:
                manifest = predecessor.bundle_manifest(env, role, seed)
                payload = _verify_manifest_tree(manifest)
                if (payload.get("schema") != predecessor.BUNDLE_SCHEMA
                        or payload.get("identity")
                        != predecessor.identity(env, role, seed)):
                    raise ValueError(f"invalid predecessor bundle: {manifest}")
                records[manifest.relative_to(ROOT).as_posix()] = file_record(
                    manifest)
            oracle_manifest = predecessor.audit_manifest(
                env, "oracle", seed)
            payload = _verify_manifest_tree(oracle_manifest)
            expected_identity = {
                **predecessor.identity(env, "oracle", seed),
                "event_seeds": list(AUDIT_EVENT_SEEDS),
                "episodes_per_task": EPISODES_PER_TASK,
                "switching_episodes": SWITCHING_EPISODES,
                "switching_period_steps": DWELL_STEPS,
                "effective_eval_seed_rule": "event_seed",
            }
            if (payload.get("schema") != predecessor.AUDIT_SCHEMA
                    or payload.get("identity") != expected_identity):
                raise ValueError(
                    f"invalid predecessor oracle audit: {oracle_manifest}")
            records[oracle_manifest.relative_to(ROOT).as_posix()] = (
                file_record(oracle_manifest))
    return records


def registration_payload() -> dict[str, Any]:
    paths = registration_source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"damping-v3 registration sources missing: {missing}")
    predecessor.validate_registration()
    robust_manifests = tuple(
        predecessor.audit_manifest(env, "robust", seed)
        for env in ENVS for seed in TRAINING_SEEDS)
    unexpected = [str(path) for path in robust_manifests if path.is_file()]
    if unexpected:
        raise ValueError(
            "v2 robust audits unexpectedly appeared before v3 registration: "
            f"{unexpected}")
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
            "evaluation_only": True,
        },
        "amendment": {
            "predecessor_registration": file_record(
                predecessor.REGISTRATION_PATH),
            "failed_v2_robust_audits": list(FAILED_V2_ROBUST_AUDITS),
            "superseded_v2_analysis_task": SUPERSEDED_V2_ANALYSIS_TASK,
            "failure_stage": "post-rollout switching-trace validation",
            "failure": (
                "v2 omitted ROBUST_TRACE_CONTEXT_MODE_ID=-1 and therefore "
                "validated the sealed all-zero robust context as if it were "
                "the dynamic oracle context"),
            "reproduction": {
                "env": "Hopper-v2",
                "role": "robust",
                "seed": 32011,
                "switching_trace_rows": 5000,
                "observed_action_task_ids": [-1],
                "observed_physics_action_task_ids": [1, 3],
                "v2_error": (
                    "robust action context is misaligned with the physics mode"),
            },
            "scientific_changes": [],
            "engineering_change": (
                "declare the existing -1 robust trace sentinel and rerun only "
                "the failed CPU audits"),
            "producer_retraining": False,
            "oracle_audit_reexecution": False,
        },
        "predecessor_artifact_records": _predecessor_artifact_records(),
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
            raise ValueError("existing damping-v3 registration changed")
    else:
        write_json_atomic(REGISTRATION_PATH, payload)
    validate_registration()
    return payload


def validate_registration() -> dict[str, Any]:
    if not REGISTRATION_PATH.is_file():
        raise ValueError("damping-v3 headroom protocol is not registered")
    payload = read_json(REGISTRATION_PATH)
    if payload != registration_payload():
        raise ValueError("damping-v3 registration or source closure changed")
    return payload


def registration_record() -> dict[str, Any]:
    validate_registration()
    return file_record(REGISTRATION_PATH)
