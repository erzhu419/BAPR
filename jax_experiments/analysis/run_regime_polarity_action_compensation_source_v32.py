"""Train one fresh V32 robust source and publish compact controller state."""
from __future__ import annotations

import argparse

from jax_experiments.analysis import (
    regime_polarity_action_compensation_reference_v32 as protocol,
)
from jax_experiments.analysis import run_regime_polarity_robust_source_v19 as base


def _bind() -> None:
    base.protocol = protocol
    base.base.protocol = protocol


def training_command(seed: int):
    _bind()
    return base.training_command(seed)


def expected_config(seed: int):
    _bind()
    return base.expected_config(seed)


def validate_signature(seed: int):
    _bind()
    return base.validate_signature(seed)


def validate_bundle(seed: int):
    _bind()
    return base.validate_bundle(seed)


def publish_bundle(seed: int):
    _bind()
    return base.publish_bundle(seed)


def run(seed: int) -> None:
    _bind()
    base.run(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for checkpoint-safe execution")
    run(args.seed)


if __name__ == "__main__":
    main()
