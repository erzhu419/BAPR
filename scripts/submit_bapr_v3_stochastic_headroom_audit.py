#!/usr/bin/env python3
"""Submit the 20 strict stochastic-headroom grouped audit tasks."""
from __future__ import annotations

import sys
from pathlib import Path


_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if str(_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORT_ROOT))

import submit_bapr_v3_budget_matched_fork_audit as submitter
from jax_experiments.analysis import (
    analyze_bapr_v3_stochastic_headroom_audit as analysis,
)
from jax_experiments.analysis import run_bapr_v3_stochastic_headroom as protocol


FINALIZE_SIGNATURE_PREFIX = "BAPR/v3-stochastic-headroom/v1-finalize"


def producer_signature(family: str, env: str) -> str:
    return f"{FINALIZE_SIGNATURE_PREFIX}/{family}/{env}/seed0"


def main() -> None:
    submitter.audit.FAMILIES = protocol.FAMILIES
    submitter.PAIR_ROOT = protocol.SAVE_ROOT
    submitter.OUT_ROOT = analysis.RESULTS_ROOT
    submitter.RUNNER_MODULE = (
        "jax_experiments.analysis."
        "run_bapr_v3_stochastic_headroom_audit_group")
    submitter.RUNNER = (
        protocol.ROOT / "jax_experiments" / "analysis"
        / "run_bapr_v3_stochastic_headroom_audit_group.py")
    submitter.producer_signature = producer_signature
    submitter.main()


if __name__ == "__main__":
    main()
