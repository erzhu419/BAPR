"""Run held-out fallback calibration for one frozen-anchor variant."""
from __future__ import annotations

import argparse

from jax_experiments.analysis import (
    calibrate_regime_polarity_anchored_residual as base,
)
from jax_experiments.analysis import (
    regime_polarity_frozen_anchor as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_frozen_anchor_eval as common,
)


def bind_variant(variant: str):
    view = common.bind_variant(variant)
    base.protocol = view
    base.common = common
    return view


def validate(variant: str, seed: int):
    bind_variant(variant)
    return base.validate(seed)


def run(variant: str, seed: int) -> None:
    bind_variant(variant)
    base.run(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=protocol.VARIANTS, required=True)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for file-gated execution")
    run(args.variant, args.seed)


if __name__ == "__main__":
    main()
