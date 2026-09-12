#!/usr/bin/env python3
"""Submit the HalfCheetah structured-channel specialist diagnostic."""
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

import submit_bapr_v3_stochastic_independent_specialists as base


base.SIGNATURE_PREFIX = (
    "BAPR/v3-structured-channel-independent-specialist/v1"
)
base.STATUS_BASE = (
    ROOT
    / "jax_experiments"
    / "results_bapr_v3_structured_channel_independent_specialist_status_v1"
)
base.TARGETS = {
    "structured-halfcheetah": base.Target(
        "structured-halfcheetah",
        "structured_channel",
        "HalfCheetah-v2",
        "jtl311linux",
        "structured_channel",
    ),
}


if __name__ == "__main__":
    base.main()
