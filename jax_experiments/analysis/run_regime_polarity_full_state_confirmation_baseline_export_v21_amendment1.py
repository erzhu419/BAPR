"""Publish V21 baseline bundles from already-complete checkpoints."""
from __future__ import annotations

import argparse
import json

from jax_experiments.analysis import (
    regime_polarity_full_state_final_confirmation_v21 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_full_state_confirmation_baseline_v21 as trainer,
)


def _bind() -> None:
    # V21 named the baseline schema explicitly; the reused V18 publisher
    # expects the historical generic attribute name.
    protocol.BUNDLE_SCHEMA = protocol.BASELINE_BUNDLE_SCHEMA
    trainer._bind()


def run(kind: str, seed: int, slot: int | None = None) -> None:
    _bind()
    seed = protocol.require_training_seed(seed)
    if kind == "sac_replica":
        if slot is None:
            raise ValueError("SAC replica export requires --slot")
        slot = protocol.require_replica_slot(slot)
    else:
        protocol.require_method(kind)
        if slot is not None:
            raise ValueError("--slot is only valid for SAC replicas")

    protocol.validate_registration()
    payload = trainer.publish_bundle(kind, seed, slot)
    print(
        "V21 BASELINE EXPORT COMPLETE: "
        + json.dumps(payload["identity"], sort_keys=True),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind", choices=("sac_replica", *protocol.TRAINED_METHODS),
        required=True)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--slot", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for checkpoint-safe export")
    run(args.kind, args.seed, args.slot)


if __name__ == "__main__":
    main()
