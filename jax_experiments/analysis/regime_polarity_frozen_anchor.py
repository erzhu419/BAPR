"""Frozen-anchor development protocol after the v1 preservation failure."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_anchored_residual as v1,
)


ROOT = v1.ROOT
PROTOCOL_VERSION = "v2"
ENV = v1.ENV
ENVS = v1.ENVS
FAMILY = v1.FAMILY
MODES = v1.MODES
TRAINING_SEEDS = v1.TRAINING_SEEDS
CALIBRATION_EVENT_SEEDS = (96_201, 96_202)
AUDIT_EVENT_SEEDS = (96_301, 96_302, 96_303)
TEST_EVENT_SEEDS = AUDIT_EVENT_SEEDS

SOURCE_NEXT_ITERATION = v1.BRANCH_FINAL_NEXT_ITERATION
SOURCE_FINAL_ITERATION = SOURCE_NEXT_ITERATION - 1
SOURCE_TOTAL_STEPS = v1.BRANCH_TOTAL_STEPS
SOURCE_UPDATE_COUNT = v1.BRANCH_UPDATE_COUNT
BRANCH_EXTRA_ITERS = 700
BRANCH_FINAL_NEXT_ITERATION = SOURCE_NEXT_ITERATION + BRANCH_EXTRA_ITERS
BRANCH_FINAL_ITERATION = BRANCH_FINAL_NEXT_ITERATION - 1
SAMPLES_PER_ITER = v1.SAMPLES_PER_ITER
UPDATES_PER_ITER = v1.UPDATES_PER_ITER
BRANCH_TOTAL_STEPS = BRANCH_FINAL_NEXT_ITERATION * SAMPLES_PER_ITER
BRANCH_UPDATE_COUNT = BRANCH_FINAL_NEXT_ITERATION * UPDATES_PER_ITER
DWELL_STEPS = v1.DWELL_STEPS
MAX_EPISODE_STEPS = v1.MAX_EPISODE_STEPS
EPISODES_PER_TASK = v1.EPISODES_PER_TASK
SWITCHING_EPISODES = v1.SWITCHING_EPISODES

VARIANTS = ("shared_small", "shared_wide", "mode_residual")
BRANCH_ROLES = ("robust_long",) + VARIANTS
VARIANT_CONFIGS = {
    "shared_small": {
        "algo": "frozen_anchored_regime_sac",
        "residual_delta": 0.15,
        "action_deviation_weight": 0.01,
        "description": "shared bounded residual, cap 0.15",
    },
    "shared_wide": {
        "algo": "frozen_anchored_regime_sac",
        "residual_delta": 0.50,
        "action_deviation_weight": 0.01,
        "description": "shared bounded residual, cap 0.50",
    },
    "mode_residual": {
        "algo": "frozen_mode_residual_sac",
        "residual_delta": 0.0,
        "action_deviation_weight": 0.01,
        "description": "independent full-capacity mode residual heads",
    },
}

CONFIDENCE_THRESHOLD = v1.CONFIDENCE_THRESHOLD
CALIBRATION_GAIN_MARGIN = v1.CALIBRATION_GAIN_MARGIN
MAX_TERMINATION_GAP = v1.MAX_TERMINATION_GAP
MIN_BASE_PRESERVATION = v1.MIN_BASE_PRESERVATION
MIN_ORACLE_RELATIVE_GAIN = v1.MIN_ORACLE_RELATIVE_GAIN
MIN_LEARNED_RELATIVE_GAIN = v1.MIN_LEARNED_RELATIVE_GAIN
MIN_SEED_WINS = v1.MIN_SEED_WINS
MIN_MODE_ACCURACY = v1.MIN_MODE_ACCURACY
MAX_BRIER_SCORE = v1.MAX_BRIER_SCORE
MAX_MEDIAN_SWITCH_DELAY = v1.MAX_MEDIAN_SWITCH_DELAY
MAX_P90_SWITCH_DELAY = v1.MAX_P90_SWITCH_DELAY

RUN_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_frozen_anchor_v2")
BUNDLE_ROOT = (
    ROOT / "jax_experiments"
    / "eval_bundles_regime_polarity_frozen_anchor_v2")
CALIBRATION_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_frozen_anchor_calibration_v2")
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_frozen_anchor_audit_v2")
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_frozen_anchor_analysis_v2")
PROTOCOL_REPORT = (
    ROOT / "reports"
    / "regime_polarity_frozen_anchor_protocol_2026-07-29.md")

BRANCH_BUNDLE_SCHEMA = "bapr.regime-polarity-frozen-anchor-branch.v2"
CALIBRATION_SCHEMA = "bapr.regime-polarity-frozen-anchor-calibration.v2"
AUDIT_SCHEMA = "bapr.regime-polarity-frozen-anchor-audit.v2"
EVENT_SCHEMA = "bapr.regime-polarity-frozen-anchor-event.v2"
ANALYSIS_SCHEMA = "bapr.regime-polarity-frozen-anchor-analysis.v2"
BOOTSTRAP_SCHEMA = "bapr.regime-polarity-frozen-anchor-bootstrap.v2"
BOOTSTRAP_NAME = "frozen_anchor_bootstrap.json"

MODEL_MANIFEST = v1.MODEL_MANIFEST
MODEL_PATH = v1.MODEL_PATH
FROZEN_MODEL_MANIFEST_RECORD = v1.FROZEN_MODEL_MANIFEST_RECORD
FROZEN_MODEL_PARAMETER_RECORD = v1.FROZEN_MODEL_PARAMETER_RECORD
posterior_metrics = v1.posterior_metrics

read_json = v1.read_json
write_json_atomic = v1.write_json_atomic
write_text_atomic = v1.write_text_atomic
sha256_file = v1.sha256_file
file_record = v1.file_record
checkpoint_record = v1.checkpoint_record
source_policy_sha256 = v1.source_policy_sha256
ensemble_critic_sha256 = v1.ensemble_critic_sha256


def validate_frozen_estimator() -> None:
    v1.validate_frozen_estimator()


def require_env(env: str) -> str:
    return v1.require_env(env)


def env_slug(env: str) -> str:
    return v1.env_slug(env)


def require_training_seed(seed: int) -> int:
    return v1.require_training_seed(seed)


def require_mode(mode: int) -> int:
    return v1.require_mode(mode)


def require_calibration_event_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in CALIBRATION_EVENT_SEEDS:
        raise ValueError(f"unknown frozen-anchor calibration seed {seed}")
    return seed


def require_event_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in AUDIT_EVENT_SEEDS:
        raise ValueError(f"unknown frozen-anchor audit seed {seed}")
    return seed


def require_variant(variant: str) -> str:
    variant = str(variant)
    if variant not in VARIANTS:
        raise ValueError(f"unknown frozen-anchor variant {variant!r}")
    return variant


def require_branch_role(role: str) -> str:
    role = str(role)
    if role not in BRANCH_ROLES:
        raise ValueError(f"unknown frozen-anchor branch {role!r}")
    return role


def source_bundle_dir(seed: int) -> Path:
    return v1.branch_bundle_dir("robust_continue", seed)


def source_manifest(seed: int) -> Path:
    return v1.branch_manifest("robust_continue", seed)


def source_required_paths(seed: int) -> tuple[Path, ...]:
    return v1.branch_required_paths("robust_continue", seed)


def branch_run_dir(role: str, seed: int) -> Path:
    return (
        RUN_ROOT / env_slug(ENV) / "branches"
        / require_branch_role(role)
        / f"seed_{require_training_seed(seed)}")


def branch_bundle_dir(role: str, seed: int) -> Path:
    return (
        BUNDLE_ROOT / env_slug(ENV) / "branches"
        / require_branch_role(role)
        / f"seed_{require_training_seed(seed)}")


def branch_manifest(role: str, seed: int) -> Path:
    return branch_bundle_dir(role, seed) / "bundle_manifest.json"


def branch_required_paths(role: str, seed: int) -> tuple[Path, ...]:
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
        return "regime_sac"
    return str(VARIANT_CONFIGS[role]["algo"])


def branch_identity(role: str, seed: int) -> dict[str, Any]:
    role = require_branch_role(role)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "env": ENV,
        "family": FAMILY,
        "role": role,
        "training_seed": require_training_seed(seed),
        "algo": branch_algo(role),
        "benchmark_role": "frozen_anchor_controller_development",
    }


def expected_branch_checkpoint(role: str) -> dict[str, Any]:
    return {
        "iteration": BRANCH_FINAL_ITERATION,
        "next_iteration": BRANCH_FINAL_NEXT_ITERATION,
        "total_steps": BRANCH_TOTAL_STEPS,
        "update_count": BRANCH_UPDATE_COUNT,
        "algo": branch_algo(role),
    }


def anchored_policy_hashes(policy) -> dict[str, str]:
    return v1.anchored_policy_hashes(policy)


def mode_policy_hashes(policy) -> dict[str, str]:
    base: list[tuple[str, Any]] = []
    adaptive: list[tuple[str, Any]] = []
    for index, layer in enumerate(policy.base_layers):
        base.extend([
            (f"base_layers.{index}.kernel", layer.kernel.value),
            (f"base_layers.{index}.bias", layer.bias.value),
        ])
    base.extend([
        ("base_mean.kernel", policy.base_mean.kernel.value),
        ("base_mean.bias", policy.base_mean.bias.value),
        ("base_log_std.kernel", policy.base_log_std.kernel.value),
        ("base_log_std.bias", policy.base_log_std.bias.value),
    ])
    for index, layer in enumerate(policy.mode_layers):
        adaptive.extend([
            (f"mode_layers.{index}.kernel", layer.kernel.value),
            (f"mode_layers.{index}.bias", layer.bias.value),
        ])
    adaptive.extend([
        ("mode_mean.kernel", policy.mode_mean.kernel.value),
        ("mode_mean.bias", policy.mode_mean.bias.value),
        ("mode_log_std.kernel", policy.mode_log_std.kernel.value),
        ("mode_log_std.bias", policy.mode_log_std.bias.value),
    ])
    return {
        "base": v1._hash_arrays(base),
        "adaptive": v1._hash_arrays(adaptive),
    }


def critic_hashes(critic) -> dict[str, str]:
    if hasattr(critic, "adaptive_critic"):
        return {
            "base": ensemble_critic_sha256(critic.base_critic),
            "adaptive": ensemble_critic_sha256(critic.adaptive_critic),
        }
    adaptive = []
    for index, layer in enumerate(critic.option_layers):
        adaptive.extend([
            (f"option_layers.{index}.kernel", layer.kernel.value),
            (f"option_layers.{index}.bias", layer.bias.value),
        ])
    return {
        "base": ensemble_critic_sha256(critic.base_critic),
        "adaptive": v1._hash_arrays(adaptive),
    }


class VariantView:
    """Bind the generic v1 audit implementation to one v2 variant."""

    BRANCH_ROLES = ("robust_continue", "anchored")

    def __init__(self, variant: str):
        self.variant = require_variant(variant)
        self.PROTOCOL_VERSION = f"{PROTOCOL_VERSION}/{self.variant}"
        self.CALIBRATION_SCHEMA = (
            f"{CALIBRATION_SCHEMA}.{self.variant}")
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
