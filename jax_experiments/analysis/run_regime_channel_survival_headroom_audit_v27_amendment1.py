"""Recover V27 robust audits with the registered zero-context sentinel."""
from __future__ import annotations

import argparse

from jax_experiments.analysis import (
    regime_channel_survival_headroom_v27 as protocol,
)
from jax_experiments.analysis import (
    run_regime_channel_survival_headroom_audit_v27 as original,
)


def run(env: str, seed: int) -> None:
    # final_task_sweep records the sealed all-zero robust context as mode -1.
    protocol.ROBUST_TRACE_CONTEXT_MODE_ID = -1
    original.run(env, "robust", seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=protocol.ENVS, required=True)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for scheduler input staging")
    run(args.env, args.seed)


if __name__ == "__main__":
    main()
