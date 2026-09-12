#!/usr/bin/env python3
"""Analyze the complete strict packet-loss/burst-torque audit matrix."""
from __future__ import annotations

from pathlib import Path

from jax_experiments.analysis import (
    analyze_bapr_v3_budget_matched_fork_audit as audit,
)
from jax_experiments.analysis import run_bapr_v3_stochastic_headroom as protocol


RESULTS_ROOT = (
    protocol.ROOT / "jax_experiments"
    / "results_bapr_v3_stochastic_headroom_audit_v1")


def main() -> None:
    audit.FAMILIES = protocol.FAMILIES
    audit.DEFAULT_PAIR_ROOT = protocol.SAVE_ROOT
    audit.DEFAULT_RESULTS_ROOT = RESULTS_ROOT
    audit.main()


if __name__ == "__main__":
    main()
