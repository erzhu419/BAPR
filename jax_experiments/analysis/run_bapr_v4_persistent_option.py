"""Train or resume the BAPR-v4 persistent-option capacity candidate."""
from __future__ import annotations

import argparse
from dataclasses import asdict

from jax_experiments.analysis import bapr_v4_persistent_option as protocol
from jax_experiments.train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "formal"),
                        required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = protocol.configure(args.profile)
    if not args.resume:
        config.resume = False
    train(config)
    checkpoint = protocol.checkpoint_summary(args.profile)
    expected_iteration = (
        protocol.FINAL_NEXT_ITERATION if args.profile == "formal" else 4)
    expected_steps = expected_iteration * config.samples_per_iter
    if (checkpoint["algo"] != "bapr_v4"
            or checkpoint["next_iteration"] != expected_iteration
            or checkpoint["total_steps"] != expected_steps):
        raise RuntimeError(
            f"BAPR-v4 final checkpoint mismatch: {checkpoint}, "
            f"expected_iter={expected_iteration}, "
            f"expected_steps={expected_steps}")
    payload = {
        "schema": protocol.SCHEMA,
        "status": "complete",
        "profile": args.profile,
        "checkpoint": checkpoint,
        "run_dir": str(protocol.run_dir(args.profile)),
        "configuration": asdict(config),
    }
    protocol.write_json_atomic(protocol.status_path(args.profile), payload)
    print(
        f"BAPR-V4 {args.profile.upper()} COMPLETE: "
        f"iter={checkpoint['next_iteration']} "
        f"steps={checkpoint['total_steps']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
