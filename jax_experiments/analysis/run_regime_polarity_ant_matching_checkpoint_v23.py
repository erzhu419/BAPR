"""Train one Ant mode-0/1 specialist with correct validation selection."""
from __future__ import annotations

import argparse

from jax_experiments.analysis import (
    regime_polarity_ant_matching_checkpoint_v23 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_ant_full_state_specialist_v22 as frozen,
)
from jax_experiments.analysis import (
    run_regime_polarity_ant_robust_source_v22 as source_runner,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_policy_stability_v20 as base,
)


_BASE_TRAINING_COMMAND = base.training_command


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
    protocol.require_trained_mode(mode)
    _bind()
    return base._bootstrap(variant, seed, mode, run_dir)


def training_command(variant: str, seed: int, mode: int, run_dir=None):
    protocol.require_trained_mode(mode)
    _bind()
    command = _BASE_TRAINING_COMMAND(variant, seed, mode, run_dir)
    module_index = command.index("-m") + 1
    command[module_index] = (
        "jax_experiments.analysis."
        "train_regime_polarity_ant_matching_checkpoint_v23"
    )
    return command


def expected_config(variant: str, seed: int, mode: int):
    protocol.require_trained_mode(mode)
    _bind()
    return base.expected_config(variant, seed, mode)


def validate_signature(variant: str, seed: int, mode: int):
    protocol.require_trained_mode(mode)
    _bind()
    return base.validate_signature(variant, seed, mode)


def validate_bundle(variant: str, seed: int, mode: int):
    mode = protocol.require_mode(mode)
    if mode in protocol.FROZEN_MODES:
        return frozen.validate_bundle(
            frozen.protocol.CONTROL_VARIANT, seed, mode)
    _bind()
    return base.validate_bundle(variant, seed, mode)


def publish_bundle(variant: str, seed: int, mode: int):
    protocol.require_trained_mode(mode)
    _bind()
    return base.publish_bundle(variant, seed, mode)


def run(variant: str, seed: int, mode: int) -> None:
    protocol.require_trained_mode(mode)
    _bind()
    original = base.training_command
    base.training_command = training_command
    try:
        base.run(variant, seed, mode)
    finally:
        base.training_command = original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=protocol.VARIANTS, required=True)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument(
        "--mode", choices=protocol.TRAINED_MODES, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for checkpoint-safe execution")
    run(args.variant, args.seed, args.mode)


if __name__ == "__main__":
    main()
