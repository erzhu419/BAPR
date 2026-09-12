"""Train or resume the BAPR-v7 unique-data-equivalent option candidate."""
from __future__ import annotations

import argparse
from dataclasses import asdict

from jax_experiments.analysis import bapr_v7_data_equivalent_option as protocol
from jax_experiments.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "formal"), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = protocol.configure(args.profile)
    if not args.resume:
        config.resume = False
    train(config)
    checkpoint = protocol.checkpoint_summary(args.profile)
    expected_iteration = (
        protocol.FINAL_NEXT_ITERATION if args.profile == "formal" else 4)
    expected_steps = expected_iteration * protocol.SAMPLES_PER_ITER
    if (checkpoint["algo"] != protocol.ALGO_NAME
            or checkpoint["next_iteration"] != expected_iteration
            or checkpoint["total_steps"] != expected_steps):
        raise RuntimeError(
            f"BAPR-v7 final checkpoint mismatch: {checkpoint}, "
            f"expected_iter={expected_iteration}, expected_steps={expected_steps}")
    protocol.write_json_atomic(protocol.status_path(args.profile), {
        "schema": protocol.SCHEMA,
        "status": "complete",
        "profile": args.profile,
        "checkpoint": checkpoint,
        "run_dir": str(protocol.run_dir(args.profile)),
        "configuration": asdict(config),
    })
    print(
        f"BAPR-V7 {args.profile.upper()} COMPLETE: "
        f"iter={checkpoint['next_iteration']} steps={checkpoint['total_steps']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
