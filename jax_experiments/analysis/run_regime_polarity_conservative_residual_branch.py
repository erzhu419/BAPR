"""Train one conservative frozen-residual controller branch."""
from __future__ import annotations

import argparse

from jax_experiments.analysis import (
    regime_polarity_conservative_residual as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_frozen_anchor_branch as base,
)


def _bind() -> None:
    base.protocol = protocol


def validate_branch(seed: int, role: str, run_dir):
    _bind()
    return base.validate_branch(seed, role, run_dir)


def validate_published(seed: int, role: str):
    _bind()
    return base.validate_published(seed, role)


def publish_bundle(seed: int, role: str, run_dir):
    _bind()
    return base.publish_bundle(seed, role, run_dir)


def run(seed: int, role: str) -> None:
    _bind()
    if role == "robust_long":
        raise ValueError("v3 reuses the completed v2 robust_long bundle")
    base.run(seed, role)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument(
        "--role", choices=protocol.VARIANTS, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for checkpoint-safe execution")
    run(args.seed, args.role)


if __name__ == "__main__":
    main()
