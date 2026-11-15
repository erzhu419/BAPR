"""Audit one Ant full-state specialist bank on held-out event streams."""
from __future__ import annotations

import argparse

from jax_experiments.analysis import (
    regime_polarity_ant_full_state_headroom_v22 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_ant_full_state_specialist_v22 as producer,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_stability_audit_v19 as base,
)


def _bind() -> None:
    base.protocol = protocol
    base.producer = producer


def validate_audit(seed: int):
    _bind()
    return base.validate_audit(protocol.CONTROL_VARIANT, seed)


def run(seed: int) -> None:
    _bind()
    base.run(protocol.CONTROL_VARIANT, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.seed)


if __name__ == "__main__":
    main()
