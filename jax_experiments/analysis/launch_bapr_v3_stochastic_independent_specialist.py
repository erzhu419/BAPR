#!/usr/bin/env python3
"""Short staging launcher for an immutable stochastic specialist runner."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from jax_experiments.analysis import (
    bapr_v3_independent_specialists as protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("stochastic_headroom", "structured_channel"),
        default="stochastic_headroom",
    )
    parser.add_argument("--family", required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--mode", type=int, choices=protocol.MODES, required=True)
    parser.add_argument("--status-dir", type=Path, required=True)
    parser.add_argument("--target-next-iteration", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.profile == "structured_channel":
        protocol.configure_structured_channel_headroom(args.env)
    else:
        protocol.configure_stochastic_headroom(args.env)
    protocol._require_family(args.family)
    run_dir = args.run_dir.resolve()
    protocol.validate_bootstrap(run_dir, args.family, args.mode)

    parent = run_dir
    destination = run_dir / ".specialist_launch_source"
    temporary = Path(tempfile.mkdtemp(
        prefix=".execution-source-launch.tmp.", dir=parent
    ))
    try:
        protocol.extract_immutable_source(run_dir, temporary)
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    child = [
        sys.executable,
        "-u",
        "-m",
        "jax_experiments.analysis.run_bapr_v3_independent_specialist",
        "--run-dir",
        str(run_dir),
        "--family",
        args.family,
        "--profile",
        args.profile,
        "--env",
        args.env,
        "--mode",
        str(args.mode),
        "--status-dir",
        str(args.status_dir.resolve()),
        "--resume",
    ]
    if args.target_next_iteration is not None:
        child += [
            "--target-next-iteration", str(args.target_next_iteration)
        ]
    if args.smoke:
        child.append("--smoke")
    env = os.environ.copy()
    env["BAPR_WORKSPACE_ROOT"] = str(protocol.ROOT)
    env["PYTHONPATH"] = str(destination)
    os.chdir(destination)
    os.execve(sys.executable, child, env)


if __name__ == "__main__":
    main()
