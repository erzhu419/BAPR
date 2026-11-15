"""Train one fresh robust/oracle final-confirmation controller."""
from __future__ import annotations

import argparse

from jax_experiments.analysis import (
    regime_polarity_final_confirmation as protocol,
)
from jax_experiments.analysis import (
    run_regime_control_headroom_controller as common,
)


common.protocol = protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=protocol.ENVS, required=True)
    parser.add_argument("--role", choices=protocol.ROLES, required=True)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for checkpoint-safe execution")
    protocol.validate_frozen_estimator()
    common.run(args.env, args.role, args.seed)


if __name__ == "__main__":
    main()
