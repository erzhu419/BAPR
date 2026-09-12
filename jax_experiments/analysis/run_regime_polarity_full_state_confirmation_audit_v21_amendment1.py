"""Run the V21 audit with the corrected baseline bundle schema binding."""
from __future__ import annotations

import argparse

from jax_experiments.analysis import (
    regime_polarity_full_state_final_confirmation_v21 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_full_state_confirmation_audit_v21 as audit,
)


def run(seed: int) -> None:
    protocol.BUNDLE_SCHEMA = protocol.BASELINE_BUNDLE_SCHEMA
    audit.run(protocol.require_training_seed(seed))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for restart-safe evaluation")
    run(args.seed)


if __name__ == "__main__":
    main()
