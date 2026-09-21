"""Train one equal-interaction V32 SAC, ESCP, or RE-SAC baseline."""
from __future__ import annotations

import argparse

from jax_experiments.analysis import (
    regime_polarity_action_compensation_power_confirmation_v32 as protocol,
)
from jax_experiments.analysis import run_regime_polarity_v5_final_baseline_v18 as base


def _bind() -> None:
    base.protocol = protocol
    base.EVAL_PARAMS_SCHEMA = protocol.EVAL_PARAMS_SCHEMA


def training_command(kind: str, seed: int, slot: int | None = None):
    _bind()
    return base.training_command(kind, seed, slot)


def expected_config(kind: str, seed: int, slot: int | None = None):
    _bind()
    return base.expected_config(kind, seed, slot)


def validate_signature(kind: str, seed: int, slot: int | None = None):
    _bind()
    return base.validate_signature(kind, seed, slot)


def validate_bundle(kind: str, seed: int, slot: int | None = None):
    _bind()
    return base.validate_bundle(kind, seed, slot)


def publish_bundle(kind: str, seed: int, slot: int | None = None):
    _bind()
    return base.publish_bundle(kind, seed, slot)


def run(kind: str, seed: int, slot: int | None = None) -> None:
    _bind()
    base.run(kind, seed, slot)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("sac_replica", *protocol.TRAINED_METHODS), required=True)
    parser.add_argument("--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--slot", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for checkpoint-safe execution")
    run(args.kind, args.seed, args.slot)


if __name__ == "__main__":
    main()
