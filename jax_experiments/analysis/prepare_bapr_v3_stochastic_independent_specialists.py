#!/usr/bin/env python3
"""Bootstrap four independent specialists beside one stochastic source pair."""
from __future__ import annotations

import argparse

from jax_experiments.analysis import (
    bapr_v3_independent_specialists as protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("stochastic_headroom", "structured_channel"),
        default="stochastic_headroom",
    )
    parser.add_argument("--family", required=True)
    parser.add_argument("--env", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.profile == "structured_channel":
        protocol.configure_structured_channel_headroom(args.env)
    else:
        protocol.configure_stochastic_headroom(args.env)
    protocol._require_family(args.family)
    robust = protocol.publish_robust_bundle(args.family)
    bootstraps = [
        protocol.bootstrap_specialist(args.family, mode)
        for mode in protocol.MODES
    ]
    print(
        "STOCHASTIC SPECIALIST BOOTSTRAP COMPLETE: "
        f"family={args.family} env={args.env} "
        f"robust_next_iter={robust['checkpoint']['next_iteration']} "
        f"specialists={len(bootstraps)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
