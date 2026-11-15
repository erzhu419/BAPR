#!/usr/bin/env python3
"""Submit the 10 strict structured-channel grouped audit tasks."""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import submit_bapr_v3_budget_matched_fork_audit as submitter
from jax_experiments.analysis import (
    analyze_bapr_v3_structured_channel_headroom_audit as analysis,
)
from jax_experiments.analysis import (
    run_bapr_v3_structured_channel_headroom as protocol,
)


PRODUCER_SIGNATURE_PREFIX = "BAPR/v3-structured-channel-headroom/v2"


def producer_signature(family: str, env: str) -> str:
    if family not in protocol.FAMILIES:
        raise ValueError(f"unsupported family {family!r}")
    return f"{PRODUCER_SIGNATURE_PREFIX}/{env}/seed0"


def main() -> None:
    submitter.audit.FAMILIES = protocol.FAMILIES
    submitter.PAIR_ROOT = protocol.SAVE_ROOT
    submitter.OUT_ROOT = analysis.RESULTS_ROOT
    submitter.RUNNER_MODULE = (
        "jax_experiments.analysis."
        "run_bapr_v3_structured_channel_headroom_audit_group"
    )
    submitter.RUNNER = (
        protocol.ROOT / "jax_experiments" / "analysis"
        / "run_bapr_v3_structured_channel_headroom_audit_group.py"
    )
    submitter.producer_signature = producer_signature
    submitter.main()


if __name__ == "__main__":
    main()
