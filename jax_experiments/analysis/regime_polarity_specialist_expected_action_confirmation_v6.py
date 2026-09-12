"""Fresh policy-bank confirmation for the frozen v5 specialist router."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_source_headroom_v1 as source_parent,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_system_id_v5 as frozen_v5,
)


ROOT = source_parent.ROOT
PROTOCOL_VERSION = "v6-fresh-policy-bank-confirmation"
ENV = source_parent.ENV
FAMILY = source_parent.FAMILY
MODES = source_parent.MODES

# These controller and event seeds were absent from the repository when this
# protocol was frozen. They must not be used to refit or select frozen v5.
TRAINING_SEEDS = (5003, 5021, 5039, 5051, 5077)
EVENT_SEEDS = (155_301, 155_317, 155_333)
ROLES = (
    "robust_sac",
    "specialist_0",
    "specialist_1",
    "specialist_2",
    "specialist_3",
)

MAX_ITERS = source_parent.MAX_ITERS
FINAL_ITERATION = source_parent.FINAL_ITERATION
SAMPLES_PER_ITER = source_parent.SAMPLES_PER_ITER
UPDATES_PER_ITER = source_parent.UPDATES_PER_ITER
START_TRAIN_STEPS = source_parent.START_TRAIN_STEPS
FINAL_TOTAL_STEPS = source_parent.FINAL_TOTAL_STEPS
FINAL_UPDATE_COUNT = source_parent.FINAL_UPDATE_COUNT
DWELL_STEPS = source_parent.DWELL_STEPS
MAX_EPISODE_STEPS = source_parent.MAX_EPISODE_STEPS
SWITCHING_EPISODES = source_parent.SWITCHING_EPISODES

PRIMARY_ARM = frozen_v5.PRIMARY_ARM
ARMS = frozen_v5.ARMS
MIN_GAIN = frozen_v5.MIN_GAIN
MIN_ORACLE_RECOVERY = frozen_v5.MIN_ORACLE_RECOVERY
MIN_SWITCHING_RETURN = frozen_v5.MIN_SWITCHING_RETURN

RUN_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_specialist_expected_action_confirmation_v6"
    / "controllers"
)
BUNDLE_ROOT = (
    ROOT / "jax_experiments"
    / "eval_bundles_regime_polarity_specialist_expected_action_confirmation_v6"
)
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_specialist_expected_action_confirmation_audit_v6"
)
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_specialist_expected_action_confirmation_analysis_v6"
)
REPORT = (
    ROOT / "reports"
    / "regime_polarity_specialist_expected_action_confirmation_v6_2026-08-29.md"
)

BUNDLE_SCHEMA = (
    "bapr.regime-polarity-specialist-expected-action-controller-bundle.v6"
)
EVENT_SCHEMA = (
    "bapr.regime-polarity-specialist-expected-action-confirmation-event.v6"
)
AUDIT_SCHEMA = (
    "bapr.regime-polarity-specialist-expected-action-confirmation-audit.v6"
)
ANALYSIS_SCHEMA = (
    "bapr.regime-polarity-specialist-expected-action-confirmation-analysis.v6"
)

file_record = source_parent.file_record
read_json = source_parent.read_json
write_json_atomic = source_parent.write_json_atomic
write_text_atomic = source_parent.write_text_atomic
checkpoint_record = source_parent.checkpoint_record


def require_role(role: str) -> str:
    role = str(role)
    if role not in ROLES:
        raise ValueError(f"unknown confirmation controller role {role!r}")
    return role


def role_fixed_mode(role: str) -> int:
    role = require_role(role)
    return int(role.removeprefix("specialist_")) if role.startswith(
        "specialist_") else -1


def require_training_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in TRAINING_SEEDS:
        raise ValueError(f"unknown confirmation training seed {seed}")
    return seed


def require_event_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in EVENT_SEEDS:
        raise ValueError(f"unknown confirmation event seed {seed}")
    return seed


def run_dir(role: str, seed: int) -> Path:
    return RUN_ROOT / require_role(role) / f"seed_{require_training_seed(seed)}"


def bundle_dir(role: str, seed: int) -> Path:
    return (
        BUNDLE_ROOT / require_role(role)
        / f"seed_{require_training_seed(seed)}"
    )


def bundle_manifest(role: str, seed: int) -> Path:
    return bundle_dir(role, seed) / "bundle_manifest.json"


def bundle_required_paths(role: str, seed: int) -> tuple[Path, ...]:
    directory = bundle_dir(role, seed)
    return (
        directory / "bundle_manifest.json",
        directory / "checkpoints" / "params.pkl",
        directory / "checkpoints" / "train_state.pkl",
        directory / "logs" / "protocol_signature.json",
    )


def audit_dir(seed: int) -> Path:
    return AUDIT_ROOT / f"seed_{require_training_seed(seed)}"


def audit_manifest(seed: int) -> Path:
    return audit_dir(seed) / "audit_manifest.json"


def event_result(seed: int, event_seed: int) -> Path:
    return (
        audit_dir(seed)
        / f"event_seed_{require_event_seed(event_seed)}"
        / "results.json"
    )


def analysis_json() -> Path:
    return ANALYSIS_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return ANALYSIS_ROOT / "analysis.md"


def identity(role: str, seed: int) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "fresh_specialist_policy_bank_confirmation",
        "env": ENV,
        "family": FAMILY,
        "role": require_role(role),
        "algo": "sac",
        "fixed_mode": role_fixed_mode(role),
        "training_seed": require_training_seed(seed),
    }


def expected_checkpoint(role: str) -> dict[str, Any]:
    require_role(role)
    return {
        "iteration": FINAL_ITERATION,
        "next_iteration": MAX_ITERS,
        "total_steps": FINAL_TOTAL_STEPS,
        "update_count": FINAL_UPDATE_COUNT,
        "algo": "sac",
    }


def source_records(seed: int) -> dict[str, Any]:
    seed = require_training_seed(seed)
    return {
        role: file_record(bundle_manifest(role, seed))
        for role in ROLES
    }


def estimator_records() -> dict[str, Any]:
    return frozen_v5.estimator_records()
