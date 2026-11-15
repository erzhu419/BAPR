#!/usr/bin/env python3
"""Short outer launcher for immutable independent-specialist audit code."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--launch-dir", type=Path, required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("child_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive = args.archive.resolve()
    launch = args.launch_dir.resolve()
    child_args = list(args.child_args)
    if child_args[:1] == ["--"]:
        child_args.pop(0)

    shutil.rmtree(launch, ignore_errors=True)
    launch.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as source:
        source.extractall(launch)
    os.chdir(launch)

    env = os.environ.copy()
    env["BAPR_WORKSPACE_ROOT"] = str(args.workspace_root.resolve())
    env["PYTHONPATH"] = str(launch)
    if args.cpu_only:
        env["JAX_PLATFORMS"] = "cpu"
        env["CUDA_VISIBLE_DEVICES"] = ""
    child = [sys.executable, "-u", "-m", args.module, *child_args]
    os.execve(sys.executable, child, env)


if __name__ == "__main__":
    main()
