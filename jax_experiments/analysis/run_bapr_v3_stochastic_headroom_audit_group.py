#!/usr/bin/env python3
"""Run one strict stochastic-headroom six-controller audit group.

The validated v2 grouped runner predates the packet-loss/burst-torque rerun.
Only its argparse family choices need extending: all execution, immutable
source extraction, runtime checks, atomic publication, and provenance checks
remain in the original runner.
"""
from __future__ import annotations

from jax_experiments.analysis import (
    run_bapr_v3_budget_matched_fork_audit_group as grouped,
)
from jax_experiments.analysis import run_bapr_v3_stochastic_headroom as protocol


def main() -> None:
    grouped.audit.FAMILIES = protocol.FAMILIES
    grouped.main()


if __name__ == "__main__":
    main()
