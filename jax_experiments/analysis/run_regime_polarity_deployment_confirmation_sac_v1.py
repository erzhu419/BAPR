"""Train one fresh-cohort same-budget SAC confirmation baseline."""
from __future__ import annotations

import argparse

from jax_experiments.analysis import (
    regime_polarity_deployment_confirmation_v1 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_fallback_final_baseline_v1 as runner,
)


runner.protocol = protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=protocol.TRAINING_SEEDS,
                        required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for checkpoint-safe execution")
    protocol.validate_mechanism_release()
    runner.run("sac", args.seed)


if __name__ == "__main__":
    main()

