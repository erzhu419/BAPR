#!/usr/bin/env python3
"""Run the structured actuator-channel controller-headroom screen.

This protocol uses four equal-severity persistent actuator masks instead of a
scalar packet-loss or burst-noise severity ladder.  It writes to a fresh result
root and delegates the shared-checkpoint, equal-budget fork invariants to the
validated BAPR-v3 runner.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from jax_experiments.analysis import run_bapr_v3_budget_matched_fork as fork


ROOT = fork.ROOT
SAVE_ROOT = (
    ROOT / "jax_experiments"
    / "results_bapr_v3_structured_channel_headroom_fork_v2"
)
FAMILIES = ("structured_channel",)
ENVS = fork.ENVS


def pair_dir(family: str, env: str, seed: int = 0) -> Path:
    return fork.pair_dir(family, env, seed, SAVE_ROOT)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default=FAMILIES[0], choices=FAMILIES)
    parser.add_argument("--env", required=True, choices=ENVS)
    parser.add_argument("--seed", type=int, default=0, choices=(0,))
    parser.add_argument("--save-root", type=Path, default=SAVE_ROOT)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    forwarded = argparse.Namespace(
        family=args.family,
        env=args.env,
        seed=args.seed,
        save_root=args.save_root,
        policy_variant="direct",
        resume=args.resume,
        finalize_existing=False,
    )
    fork.run_protocol(forwarded)


if __name__ == "__main__":
    main()
