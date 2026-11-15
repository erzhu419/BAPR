#!/usr/bin/env python3
"""Analyze the strict structured-channel robust/oracle audit matrix."""
from __future__ import annotations

from jax_experiments.analysis import (
    analyze_bapr_v3_stochastic_headroom_audit_family as family_audit,
)
from jax_experiments.analysis import (
    run_bapr_v3_structured_channel_headroom as protocol,
)


RESULTS_ROOT = (
    protocol.ROOT / "jax_experiments"
    / "results_bapr_v3_structured_channel_headroom_audit_v2"
)


def main() -> None:
    family_audit.protocol = protocol
    family_audit.DEFAULT_RESULTS_ROOT = RESULTS_ROOT
    family_audit.main()


if __name__ == "__main__":
    main()
