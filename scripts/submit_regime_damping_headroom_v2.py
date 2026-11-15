#!/usr/bin/env python3
"""Submit the pre-training-amended persistent-damping v2 DAG."""
from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import submit_regime_damping_headroom as base
from jax_experiments.analysis import regime_damping_headroom_v2 as protocol


if (__name__ == "__main__"
        and Path(sys.executable).resolve()
        != base.scheduler_common.JAX_PYTHON.resolve()):
    os.execv(
        str(base.scheduler_common.JAX_PYTHON),
        [str(base.scheduler_common.JAX_PYTHON), str(Path(__file__).resolve()),
         *sys.argv[1:]],
    )


base.protocol = protocol
base.SIGNATURE_PREFIX = "BAPR/regime-damping-headroom/v2"
base.SUBMIT_INTENT_LABEL = "bapr-regime-damping-headroom-v2-submit"
_OLD_GPU_COMMAND = base._gpu_command
_OLD_CPU_COMMAND = base._cpu_command


def _gpu_command(values: list[str]) -> str:
    return _OLD_GPU_COMMAND(values).replace(
        "jax_experiments.analysis.run_regime_damping_headroom_controller",
        "jax_experiments.analysis.run_regime_damping_headroom_controller_v2",
    )


def _cpu_command(module: str, values: list[str], threads: int = 4) -> str:
    replacements = {
        "jax_experiments.analysis.run_regime_damping_headroom_audit":
            "jax_experiments.analysis.run_regime_damping_headroom_audit_v2",
        "jax_experiments.analysis.analyze_regime_damping_headroom":
            "jax_experiments.analysis.analyze_regime_damping_headroom_v2",
    }
    return _OLD_CPU_COMMAND(
        replacements.get(module, module), values, threads=threads)


base._gpu_command = _gpu_command
base._cpu_command = _cpu_command


if __name__ == "__main__":
    base.main()

