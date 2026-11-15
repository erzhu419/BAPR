#!/usr/bin/env python3
"""Run the preregistered stochastic-regime controller-headroom pairs.

This is a narrow entry point over the validated shared-checkpoint budget-fork
runner.  It deliberately accepts only the packet-loss and burst-torque mode
families and writes to a fresh result root, so the strict rerun cannot resume
or be confused with the earlier exploratory oracle screen.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from jax_experiments.analysis import run_bapr_v3_budget_matched_fork as fork


ROOT = fork.ROOT
SAVE_ROOT = (
    ROOT / "jax_experiments" / "results_bapr_v3_stochastic_headroom_fork_v1")
FAMILIES = ("packet_loss", "burst_torque")
ENVS = fork.ENVS


def pair_dir(family: str, env: str, seed: int = 0) -> Path:
    return fork.pair_dir(family, env, seed, SAVE_ROOT)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=FAMILIES)
    parser.add_argument("--env", required=True, choices=ENVS)
    parser.add_argument("--seed", type=int, default=0, choices=(0,))
    parser.add_argument("--save-root", type=Path, default=SAVE_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--finalize-existing", action="store_true",
        help="Validate completed branches and rebuild completion artifacts "
             "without entering training.")
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
        finalize_existing=args.finalize_existing,
    )
    if args.finalize_existing:
        fork.finalize_existing_protocol(forwarded)
    else:
        fork.run_protocol(forwarded)


if __name__ == "__main__":
    main()
