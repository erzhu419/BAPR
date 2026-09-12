#!/usr/bin/env python3
"""Submit the stable-LCB conservative residual v4 rerun."""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import submit_bapr_v3_budget_matched_fork as scheduler_common

if Path(sys.executable).resolve() != scheduler_common.JAX_PYTHON.resolve():
    os.execv(
        str(scheduler_common.JAX_PYTHON),
        [
            str(scheduler_common.JAX_PYTHON),
            str(Path(__file__).resolve()),
            *sys.argv[1:],
        ],
    )

import submit_regime_polarity_frozen_anchor as base
from jax_experiments.analysis.regime_polarity_conservative_residual_stable import (
    bind as bind_protocol,
)


protocol = bind_protocol()
SIGNATURE_PREFIX = (
    "BAPR/regime-polarity-conservative-residual/stable-lcb-v4")
DISPLAY_NAME = "Stable-LCB conservative residual v4"
RESOURCE_PREFIX = (
    "BAPR/regime-polarity-conservative-residual-stable")
BRANCH_MODULE = (
    "jax_experiments.analysis."
    "run_regime_polarity_conservative_residual_stable_branch")
CALIBRATION_MODULE = (
    "jax_experiments.analysis."
    "calibrate_regime_polarity_conservative_residual_stable")
AUDIT_MODULE = (
    "jax_experiments.analysis."
    "audit_regime_polarity_conservative_residual_stable")
ANALYSIS_MODULE = (
    "jax_experiments.analysis."
    "analyze_regime_polarity_conservative_residual_stable")
TRAIN_BRANCH_ROLES = protocol.VARIANTS
SUBMIT_INTENT_LABEL = "bapr-conservative-stable-v4-submit"
DISPATCH_INTENT_LABEL = "bapr-conservative-stable-v4-dispatch"
GPU_NODES = ["local", "jtl110gpu", "node007"]
CPU_NODES = [f"node00{index}" for index in range(1, 7)]
VRAM_MB = {variant: 2800 for variant in protocol.VARIANTS}
MEMORY_FRACTION = {variant: 0.42 for variant in protocol.VARIANTS}
RAM_MB = {variant: 10240 for variant in protocol.VARIANTS}


def _settings() -> dict[str, object]:
    return {
        "protocol": protocol,
        "SIGNATURE_PREFIX": SIGNATURE_PREFIX,
        "DISPLAY_NAME": DISPLAY_NAME,
        "RESOURCE_PREFIX": RESOURCE_PREFIX,
        "BRANCH_MODULE": BRANCH_MODULE,
        "CALIBRATION_MODULE": CALIBRATION_MODULE,
        "AUDIT_MODULE": AUDIT_MODULE,
        "ANALYSIS_MODULE": ANALYSIS_MODULE,
        "TRAIN_BRANCH_ROLES": TRAIN_BRANCH_ROLES,
        "SUBMIT_INTENT_LABEL": SUBMIT_INTENT_LABEL,
        "DISPATCH_INTENT_LABEL": DISPATCH_INTENT_LABEL,
        "GPU_NODES": GPU_NODES,
        "CPU_NODES": CPU_NODES,
        "VRAM_MB": VRAM_MB,
        "MEMORY_FRACTION": MEMORY_FRACTION,
        "RAM_MB": RAM_MB,
    }


@contextmanager
def _bind():
    settings = _settings()
    previous = {name: getattr(base, name) for name in settings}
    for name, value in settings.items():
        setattr(base, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(base, name, value)


def candidates(phase: str, priority: str):
    with _bind():
        return base.candidates(phase, priority)


if __name__ == "__main__":
    with _bind():
        base.main()
