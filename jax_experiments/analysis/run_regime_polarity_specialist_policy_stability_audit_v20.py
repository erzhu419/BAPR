"""Audit one v20 policy-stability specialist bank."""
from __future__ import annotations

import argparse

from jax_experiments.analysis import (
    regime_polarity_specialist_policy_stability_v20 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_policy_stability_v20 as producer,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_stability_audit_v19 as base,
)


def _bind() -> None:
    base.protocol = protocol
    base.producer = producer


def validate_audit(variant: str, seed: int):
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
    run(args.variant, args.seed)


if __name__ == "__main__":
    main()
