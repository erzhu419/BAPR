"""Bind the conservative residual protocol to the stable-LCB v4 rerun."""
from __future__ import annotations

from jax_experiments.analysis import (
    regime_polarity_conservative_residual as base,
)


VARIANTS = ("trust_small", "trust_tight")
VARIANT_CONFIGS = {
    variant: dict(base.VARIANT_CONFIGS[variant])
    for variant in VARIANTS
}


def bind():
    """Mutate the v3 protocol module for one isolated v4 process."""
    base.PROTOCOL_VERSION = "v4-stable-lcb-gradient"
    base.VARIANTS = VARIANTS
    base.BRANCH_ROLES = ("robust_long",) + VARIANTS
    base.VARIANT_CONFIGS = VARIANT_CONFIGS
    base.RUN_ROOT = (
        base.ROOT / "jax_experiments"
        / "results_regime_polarity_conservative_residual_stable_v4")
    base.BUNDLE_ROOT = (
        base.ROOT / "jax_experiments"
        / "eval_bundles_regime_polarity_conservative_residual_stable_v4")
    base.CALIBRATION_ROOT = (
        base.ROOT / "jax_experiments"
        / "results_regime_polarity_conservative_residual_stable_calibration_v4")
    base.AUDIT_ROOT = (
        base.ROOT / "jax_experiments"
        / "results_regime_polarity_conservative_residual_stable_audit_v4")
    base.ANALYSIS_ROOT = (
        base.ROOT / "jax_experiments"
        / "results_regime_polarity_conservative_residual_stable_analysis_v4")
    base.PROTOCOL_REPORT = (
        base.ROOT / "reports"
        / "regime_polarity_conservative_residual_stable_protocol_2026-07-30.md")
    base.BRANCH_BUNDLE_SCHEMA = (
        "bapr.regime-polarity-conservative-residual-branch.v4")
    base.CALIBRATION_SCHEMA = (
        "bapr.regime-polarity-conservative-residual-calibration.v4")
    base.AUDIT_SCHEMA = (
        "bapr.regime-polarity-conservative-residual-audit.v4")
    base.EVENT_SCHEMA = (
        "bapr.regime-polarity-conservative-residual-event.v4")
    base.ANALYSIS_SCHEMA = (
        "bapr.regime-polarity-conservative-residual-analysis.v4")
    base.BOOTSTRAP_SCHEMA = (
        "bapr.regime-polarity-conservative-residual-bootstrap.v4")
    base.BOOTSTRAP_NAME = "conservative_residual_stable_bootstrap.json"
    return base
