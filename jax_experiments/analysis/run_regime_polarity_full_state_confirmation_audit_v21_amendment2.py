"""Run the V21 audit with both reused V18 APIs restored."""
from __future__ import annotations

import argparse

from jax_experiments.analysis import (
    regime_polarity_full_state_final_confirmation_v21 as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_model_v5 as v5_model,
)
from jax_experiments.analysis import (
    run_regime_polarity_full_state_confirmation_audit_v21_amendment1 as audit,
)


def _bind() -> None:
    protocol.BUNDLE_SCHEMA = protocol.BASELINE_BUNDLE_SCHEMA
    v5_model.MODEL_MANIFEST = v5_model.protocol.MODEL_MANIFEST
    v5_model.MODEL_PATH = v5_model.protocol.MODEL_PATH


def run(seed: int) -> None:
    _bind()
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
