#!/usr/bin/env python3
"""Run one six-controller structured-channel audit group."""
from __future__ import annotations

from jax_experiments.analysis import (
    run_bapr_v3_budget_matched_fork_audit_group as grouped,
)
from jax_experiments.analysis import (
    run_bapr_v3_structured_channel_headroom as protocol,
)


def main() -> None:
    grouped.audit.FAMILIES = protocol.FAMILIES
    grouped.main()


if __name__ == "__main__":
    main()
