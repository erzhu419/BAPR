"""Train the frozen-recipe V32 canonical mode-0 reference policy."""
from __future__ import annotations

import argparse

from jax_experiments.analysis import (
    regime_polarity_action_compensation_power_confirmation_v32 as final,
)
from jax_experiments.analysis import (
    regime_polarity_action_compensation_reference_v32 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_action_compensation_source_v32 as source_runner,
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


def run(seed: int) -> None:
    _bind()
    base.run(protocol.SPECIALIST_VARIANT, seed, final.REFERENCE_MODE)


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
