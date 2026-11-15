"""Train one corrected ESCP or RE-SAC baseline for v9."""
from __future__ import annotations

import argparse

from jax_experiments.analysis import (
    regime_polarity_specialist_safe_utility_confirmation_v9 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_corrected_baseline_v2 as runner,
)


def _bind() -> None:
    runner.protocol = protocol


def validate_bundle(method: str, seed: int) -> dict:
    _bind()
    return runner.validate_bundle(method, seed)


def run(method: str, seed: int) -> None:
    protocol.validate_registration()
    _bind()
    runner.run(method, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", choices=protocol.BASELINE_METHODS, required=True)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for checkpoint-safe execution")
    run(args.method, args.seed)


if __name__ == "__main__":
    main()
