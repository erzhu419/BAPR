"""Audit V25 constrained-risk Ant specialists on frozen holdouts."""
from __future__ import annotations

import argparse
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_ant_constrained_risk_v25 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_ant_constrained_risk_v25 as producer,
)
from jax_experiments.analysis import (
    run_regime_polarity_ant_switch_recovery_audit_v24 as base,
)


SWITCHING_ARMS = base.SWITCHING_ARMS


def _bind() -> None:
    base.protocol = protocol
    base.producer = producer


def validate_audit(variant: str, seed: int) -> dict[str, Any]:
    _bind()
    return base.validate_audit(variant, seed)


def run(variant: str, seed: int) -> None:
    _bind()
    base.run(variant, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=protocol.VARIANTS, required=True)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for checkpoint-safe execution")
    run(args.variant, args.seed)


if __name__ == "__main__":
    main()
