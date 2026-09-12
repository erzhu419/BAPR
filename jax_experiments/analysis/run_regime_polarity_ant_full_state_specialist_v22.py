"""Train one Ant full-state specialist with the frozen V21 recipe."""
from __future__ import annotations

import argparse

from jax_experiments.analysis import (
    regime_polarity_ant_full_state_headroom_v22 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_ant_robust_source_v22 as source_runner,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_policy_stability_v20 as base,
)


def _bind() -> None:
    base.protocol = protocol
    base.source_runner = source_runner
    base.historical.protocol = protocol


def _load_source(seed: int):
    _bind()
    return base._load_source(seed)


def _target_config(source_config, run_dir, mode: int, variant: str):
    _bind()
    return base._target_config(source_config, run_dir, mode, variant)


def _bootstrap(variant: str, seed: int, mode: int, run_dir=None):
    _bind()
    return base._bootstrap(variant, seed, mode, run_dir)


def training_command(variant: str, seed: int, mode: int, run_dir=None):
    _bind()
    return base.training_command(variant, seed, mode, run_dir)


def expected_config(variant: str, seed: int, mode: int):
    _bind()
    return base.expected_config(variant, seed, mode)


def validate_signature(variant: str, seed: int, mode: int):
    _bind()
    return base.validate_signature(variant, seed, mode)


def validate_bundle(variant: str, seed: int, mode: int):
    _bind()
    return base.validate_bundle(variant, seed, mode)


def publish_bundle(variant: str, seed: int, mode: int):
    _bind()
    return base.publish_bundle(variant, seed, mode)


def run(seed: int, mode: int) -> None:
    _bind()
    base.run(protocol.CONTROL_VARIANT, seed, mode)


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
