"""Conservative frozen-residual development protocol.

The equal-budget robust controller and source checkpoint are immutable inputs.
Only a bounded residual is trained. Candidate actor updates are accepted only
when their target-critic lower-bound advantage preserves every represented
mode.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_frozen_anchor as v2,
)


ROOT = v2.ROOT
PROTOCOL_VERSION = "v3-conservative-residual"
ENV = v2.ENV
ENVS = v2.ENVS
FAMILY = v2.FAMILY
MODES = v2.MODES
TRAINING_SEEDS = v2.TRAINING_SEEDS
CALIBRATION_EVENT_SEEDS = (96_401, 96_402)
AUDIT_EVENT_SEEDS = (96_501, 96_502, 96_503)
TEST_EVENT_SEEDS = AUDIT_EVENT_SEEDS

SOURCE_NEXT_ITERATION = v2.SOURCE_NEXT_ITERATION
SOURCE_FINAL_ITERATION = v2.SOURCE_FINAL_ITERATION
SOURCE_TOTAL_STEPS = v2.SOURCE_TOTAL_STEPS
SOURCE_UPDATE_COUNT = v2.SOURCE_UPDATE_COUNT
BRANCH_EXTRA_ITERS = v2.BRANCH_EXTRA_ITERS
BRANCH_FINAL_NEXT_ITERATION = v2.BRANCH_FINAL_NEXT_ITERATION
BRANCH_FINAL_ITERATION = v2.BRANCH_FINAL_ITERATION
SAMPLES_PER_ITER = v2.SAMPLES_PER_ITER
UPDATES_PER_ITER = v2.UPDATES_PER_ITER
BRANCH_TOTAL_STEPS = v2.BRANCH_TOTAL_STEPS
BRANCH_UPDATE_COUNT = v2.BRANCH_UPDATE_COUNT
DWELL_STEPS = v2.DWELL_STEPS
MAX_EPISODE_STEPS = v2.MAX_EPISODE_STEPS
EPISODES_PER_TASK = v2.EPISODES_PER_TASK
SWITCHING_EPISODES = v2.SWITCHING_EPISODES

VARIANTS = ("strict_small", "trust_small", "trust_tight")
BRANCH_ROLES = ("robust_long",) + VARIANTS
_COMMON_CONSTRAINT = {
    "algo": "frozen_anchored_regime_sac",
    "train_advantage_constraint": True,
    "train_advantage_lcb_scale": 1.0,
    "train_advantage_margin": 0.0,
    "train_advantage_temperature": 0.01,
    "train_advantage_weight": 1.0,
    "train_update_filter": True,
}
VARIANT_CONFIGS = {
    "strict_small": {
        **_COMMON_CONSTRAINT,
        "residual_delta": 0.15,
        "action_deviation_weight": 0.05,
        "train_update_tolerance": 0.0,
        "train_update_floor": 0.0,
        "description": (
            "cap 0.15 residual; reject any represented-mode LCB regression"),
    },
    "trust_small": {
        **_COMMON_CONSTRAINT,
        "residual_delta": 0.15,
        "action_deviation_weight": 0.05,
        "train_update_tolerance": 0.005,
        "train_update_floor": -0.01,
        "description": (
            "cap 0.15 residual; 0.5% update tolerance and -1% LCB floor"),
    },
    "trust_tight": {
        **_COMMON_CONSTRAINT,
        "residual_delta": 0.075,
        "action_deviation_weight": 0.05,
        "train_update_tolerance": 0.005,
        "train_update_floor": -0.01,
        "description": (
            "cap 0.075 residual; 0.5% update tolerance and -1% LCB floor"),
    },
}

CONFIDENCE_THRESHOLD = v2.CONFIDENCE_THRESHOLD
CALIBRATION_GAIN_MARGIN = v2.CALIBRATION_GAIN_MARGIN
MAX_TERMINATION_GAP = v2.MAX_TERMINATION_GAP
MIN_BASE_PRESERVATION = v2.MIN_BASE_PRESERVATION
MIN_ORACLE_RELATIVE_GAIN = v2.MIN_ORACLE_RELATIVE_GAIN
MIN_LEARNED_RELATIVE_GAIN = v2.MIN_LEARNED_RELATIVE_GAIN
MIN_SEED_WINS = v2.MIN_SEED_WINS
MIN_MODE_ACCURACY = v2.MIN_MODE_ACCURACY
MAX_BRIER_SCORE = v2.MAX_BRIER_SCORE
MAX_MEDIAN_SWITCH_DELAY = v2.MAX_MEDIAN_SWITCH_DELAY
MAX_P90_SWITCH_DELAY = v2.MAX_P90_SWITCH_DELAY

RUN_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_conservative_residual_v3")
BUNDLE_ROOT = (
    ROOT / "jax_experiments"
    / "eval_bundles_regime_polarity_conservative_residual_v3")
CALIBRATION_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_conservative_residual_calibration_v3")
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_conservative_residual_audit_v3")
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_conservative_residual_analysis_v3")
PROTOCOL_REPORT = (
    ROOT / "reports"
    / "regime_polarity_conservative_residual_protocol_2026-07-30.md")

BRANCH_BUNDLE_SCHEMA = "bapr.regime-polarity-conservative-residual-branch.v3"
CALIBRATION_SCHEMA = (
    "bapr.regime-polarity-conservative-residual-calibration.v3")
AUDIT_SCHEMA = "bapr.regime-polarity-conservative-residual-audit.v3"
EVENT_SCHEMA = "bapr.regime-polarity-conservative-residual-event.v3"
ANALYSIS_SCHEMA = "bapr.regime-polarity-conservative-residual-analysis.v3"
BOOTSTRAP_SCHEMA = (
    "bapr.regime-polarity-conservative-residual-bootstrap.v3")
BOOTSTRAP_NAME = "conservative_residual_bootstrap.json"

MODEL_MANIFEST = v2.MODEL_MANIFEST
MODEL_PATH = v2.MODEL_PATH
FROZEN_MODEL_MANIFEST_RECORD = v2.FROZEN_MODEL_MANIFEST_RECORD
FROZEN_MODEL_PARAMETER_RECORD = v2.FROZEN_MODEL_PARAMETER_RECORD
posterior_metrics = v2.posterior_metrics

read_json = v2.read_json
write_json_atomic = v2.write_json_atomic
write_text_atomic = v2.write_text_atomic
sha256_file = v2.sha256_file
file_record = v2.file_record
checkpoint_record = v2.checkpoint_record
source_policy_sha256 = v2.source_policy_sha256
ensemble_critic_sha256 = v2.ensemble_critic_sha256
anchored_policy_hashes = v2.anchored_policy_hashes
mode_policy_hashes = v2.mode_policy_hashes
critic_hashes = v2.critic_hashes


def validate_frozen_estimator() -> None:
    v2.validate_frozen_estimator()


def require_env(env: str) -> str:
    return v2.require_env(env)


def env_slug(env: str) -> str:
    return v2.env_slug(env)


def require_training_seed(seed: int) -> int:
    return v2.require_training_seed(seed)


def require_mode(mode: int) -> int:
    return v2.require_mode(mode)


def require_calibration_event_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in CALIBRATION_EVENT_SEEDS:
        raise ValueError(f"unknown conservative calibration seed {seed}")
    return seed


def require_event_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in AUDIT_EVENT_SEEDS:
        raise ValueError(f"unknown conservative audit seed {seed}")
    return seed


def require_variant(variant: str) -> str:
    variant = str(variant)
    if variant not in VARIANTS:
        raise ValueError(f"unknown conservative residual variant {variant!r}")
    return variant


def require_branch_role(role: str) -> str:
    role = str(role)
    if role not in BRANCH_ROLES:
        raise ValueError(f"unknown conservative residual branch {role!r}")
    return role


def source_bundle_dir(seed: int) -> Path:
    return v2.source_bundle_dir(seed)


def source_manifest(seed: int) -> Path:
    return v2.source_manifest(seed)


def source_required_paths(seed: int) -> tuple[Path, ...]:
    return v2.source_required_paths(seed)


def branch_run_dir(role: str, seed: int) -> Path:
    role = require_branch_role(role)
    if role == "robust_long":
        return v2.branch_run_dir(role, seed)
    return (
        RUN_ROOT / env_slug(ENV) / "branches" / role
        / f"seed_{require_training_seed(seed)}")


def branch_bundle_dir(role: str, seed: int) -> Path:
    role = require_branch_role(role)
    if role == "robust_long":
        return v2.branch_bundle_dir(role, seed)
    return (
        BUNDLE_ROOT / env_slug(ENV) / "branches" / role
        / f"seed_{require_training_seed(seed)}")


def branch_manifest(role: str, seed: int) -> Path:
    role = require_branch_role(role)
    if role == "robust_long":
        return v2.branch_manifest(role, seed)
    return branch_bundle_dir(role, seed) / "bundle_manifest.json"


def branch_required_paths(role: str, seed: int) -> tuple[Path, ...]:
    role = require_branch_role(role)
    if role == "robust_long":
        return v2.branch_required_paths(role, seed)
    directory = branch_bundle_dir(role, seed)
    return (
        directory / "bundle_manifest.json",
        directory / "checkpoints" / "params.pkl",
        directory / "checkpoints" / "train_state.pkl",
        directory / "checkpoints" / BOOTSTRAP_NAME,
        directory / "logs" / "protocol_signature.json",
    )


def calibration_dir(variant: str, seed: int) -> Path:
    return (
        CALIBRATION_ROOT / require_variant(variant)
        / f"seed_{require_training_seed(seed)}")


def calibration_manifest(variant: str, seed: int) -> Path:
    return calibration_dir(variant, seed) / "calibration_manifest.json"


def audit_dir(variant: str, seed: int) -> Path:
    return (
        AUDIT_ROOT / require_variant(variant)
        / f"seed_{require_training_seed(seed)}")


def audit_manifest(variant: str, seed: int) -> Path:
    return audit_dir(variant, seed) / "audit_manifest.json"


def analysis_root(variant: str) -> Path:
    return ANALYSIS_ROOT / require_variant(variant)


def analysis_json(variant: str) -> Path:
    return analysis_root(variant) / "analysis.json"


def analysis_markdown(variant: str) -> Path:
    return analysis_root(variant) / "analysis.md"


def branch_algo(role: str) -> str:
    role = require_branch_role(role)
    if role == "robust_long":
        return v2.branch_algo(role)
    return str(VARIANT_CONFIGS[role]["algo"])


def branch_identity(role: str, seed: int) -> dict[str, Any]:
    role = require_branch_role(role)
    if role == "robust_long":
        return v2.branch_identity(role, seed)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "env": ENV,
        "family": FAMILY,
        "role": role,
        "training_seed": require_training_seed(seed),
        "algo": branch_algo(role),
        "benchmark_role": "conservative_frozen_residual_development",
    }


def expected_branch_checkpoint(role: str) -> dict[str, Any]:
    role = require_branch_role(role)
    if role == "robust_long":
        return v2.expected_branch_checkpoint(role)
    return {
        "iteration": BRANCH_FINAL_ITERATION,
        "next_iteration": BRANCH_FINAL_NEXT_ITERATION,
        "total_steps": BRANCH_TOTAL_STEPS,
        "update_count": BRANCH_UPDATE_COUNT,
        "algo": branch_algo(role),
    }


class VariantView:
    """Bind the generic calibration/audit stack to one conservative variant."""

    BRANCH_ROLES = ("robust_continue", "anchored")

    def __init__(self, variant: str):
        self.variant = require_variant(variant)
        self.PROTOCOL_VERSION = f"{PROTOCOL_VERSION}/{self.variant}"
        self.CALIBRATION_SCHEMA = f"{CALIBRATION_SCHEMA}.{self.variant}"
        self.AUDIT_SCHEMA = f"{AUDIT_SCHEMA}.{self.variant}"
        self.EVENT_SCHEMA = f"{EVENT_SCHEMA}.{self.variant}"
        self.ANALYSIS_SCHEMA = f"{ANALYSIS_SCHEMA}.{self.variant}"

    def __getattr__(self, name):
        return globals()[name]

    def _role(self, role: str) -> str:
        if role == "robust_continue":
            return "robust_long"
        if role == "anchored":
            return self.variant
        raise ValueError(f"unknown audit role {role!r}")

    def require_branch_role(self, role: str) -> str:
        if role not in self.BRANCH_ROLES:
            raise ValueError(f"unknown audit role {role!r}")
        return role

    def branch_bundle_dir(self, role: str, seed: int) -> Path:
        return branch_bundle_dir(self._role(role), seed)

    def branch_manifest(self, role: str, seed: int) -> Path:
        return branch_manifest(self._role(role), seed)

    def branch_required_paths(
        self,
        role: str,
        seed: int,
    ) -> tuple[Path, ...]:
        return branch_required_paths(self._role(role), seed)

    def calibration_dir(self, seed: int) -> Path:
        return calibration_dir(self.variant, seed)

    def calibration_manifest(self, seed: int) -> Path:
        return calibration_manifest(self.variant, seed)

    def audit_dir(self, seed: int) -> Path:
        return audit_dir(self.variant, seed)

    def audit_manifest(self, seed: int) -> Path:
        return audit_manifest(self.variant, seed)

    def analysis_json(self) -> Path:
        return analysis_json(self.variant)

    def analysis_markdown(self) -> Path:
        return analysis_markdown(self.variant)


def variant_view(variant: str) -> VariantView:
    return VariantView(variant)
