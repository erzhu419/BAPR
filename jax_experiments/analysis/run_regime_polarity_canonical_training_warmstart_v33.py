"""Train the V33 robust-warm-started reference-mode controller."""
from __future__ import annotations

import argparse

from jax_experiments.analysis import (
    regime_polarity_canonical_training_ablation_v33 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_canonical_training_source_v33 as source_runner,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_policy_stability_v20 as base,
)


def _bind() -> None:
    base.protocol = protocol
    base.source_runner = source_runner
    base.historical.protocol = protocol


def training_command(seed: int, run_dir=None):
    _bind()
    return base.training_command(
        protocol.WARMSTART_RECIPE, seed, protocol.REFERENCE_MODE, run_dir)


def expected_config(seed: int):
    _bind()
    return base.expected_config(
        protocol.WARMSTART_RECIPE, seed, protocol.REFERENCE_MODE)


def validate_bundle(seed: int):
    _bind()
    return base.validate_bundle(
        protocol.WARMSTART_RECIPE, seed, protocol.REFERENCE_MODE)


def run(seed: int) -> None:
    _bind()
    base.run(protocol.WARMSTART_RECIPE, seed, protocol.REFERENCE_MODE)


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
