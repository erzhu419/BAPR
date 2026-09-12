"""Run the confirmed actor-only specialist producer under v17."""
from __future__ import annotations

import argparse

from jax_experiments.analysis import (
    regime_polarity_fresh_bank_estimator_confirmation_v17 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_robust_source_v17 as source_runner,
)
from jax_experiments.analysis import (
    run_regime_polarity_robust_warmstart_specialist_v12 as base,
)


def _bind() -> None:
    base.protocol = protocol
    base.source_runner = source_runner
    base.BOOTSTRAP_SCHEMA = protocol.BOOTSTRAP_SCHEMA


def validate_bundle(
    variant_or_seed: str | int,
    seed_or_mode: int,
    mode: int | None = None,
):
    _bind()
    if mode is None:
        seed = int(variant_or_seed)
        mode = int(seed_or_mode)
    else:
        protocol.require_variant(str(variant_or_seed))
        seed = int(seed_or_mode)
    return base.validate_bundle(seed, mode)


def training_command(seed: int, mode: int) -> list[str]:
    _bind()
    return base.training_command(seed, mode)


def publish_bundle(seed: int, mode: int):
    _bind()
    return base.publish_bundle(seed, mode)


def run(seed: int, mode: int) -> None:
    _bind()
    base.run(seed, mode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--mode", choices=protocol.MODES, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for checkpoint-safe execution")
    run(args.seed, args.mode)


if __name__ == "__main__":
    main()
