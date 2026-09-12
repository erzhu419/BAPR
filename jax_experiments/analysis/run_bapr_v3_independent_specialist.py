"""Resume one fully independent BAPR-v3 robust controller in a fixed mode."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from jax_experiments.analysis import (
    bapr_v3_independent_specialists as protocol,
)
from jax_experiments.analysis import (
    run_bapr_v3_budget_matched_fork as fork_protocol,
)


def training_values(
    run_dir: Path, family: str, mode: int, target_next_iteration: int,
    boundary_audit: bool,
) -> list[str]:
    values = fork_protocol.common_training_values(
        family, protocol.ENV, protocol.SEED, run_dir.parent,
        run_dir.name, "direct")
    values += [
        "--bapr_v2_base_pretrain_iters", "1400",
        "--bapr_v2_teacher_iters", "0",
        "--max_iters", str(target_next_iteration),
        "--min_resume_iteration", str(protocol.BASE_NEXT_ITERATION),
        "--stochastic_mode_fixed_id", str(mode),
    ]
    if boundary_audit:
        values += [
            "--resume_boundary_audit",
            "--resume_boundary_expected_iteration",
            str(protocol.BASE_NEXT_ITERATION),
            "--resume_boundary_expected_total_steps",
            str(protocol.BASE_TOTAL_STEPS),
            "--resume_boundary_expected_update_count",
            str(protocol.BASE_UPDATE_COUNT),
        ]
    return values


def prepare_execution_source(run_dir: Path) -> Path:
    parent = run_dir / protocol.PROTOCOL_CHECKPOINT_DIR
    destination = parent / "execution_source"
    temporary = Path(tempfile.mkdtemp(
        prefix=".execution-source.tmp.", dir=parent))
    try:
        protocol.extract_immutable_source(run_dir, temporary)
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def run_train(
    run_dir: Path, execution_root: Path, values: list[str],
) -> None:
    command = [
        sys.executable, "-u", "-m", "jax_experiments.train", *values]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(execution_root)
    log_path = run_dir / "logs" / "specialist_train.log"
    print("SPECIALIST TRAIN:", " ".join(command), flush=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"\ncommand={command!r}\n")
        process = subprocess.Popen(
            command, cwd=execution_root, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)


def validate_partial(
    run_dir: Path, mode: int, target_next_iteration: int,
) -> None:
    checkpoint = protocol.checkpoint_info(run_dir)
    if (checkpoint["next_iteration"] != target_next_iteration
            or checkpoint["total_steps"]
            != target_next_iteration * fork_protocol.SAMPLES_PER_ITER):
        raise RuntimeError(
            f"smoke checkpoint mismatch: {checkpoint}, "
            f"target_next_iteration={target_next_iteration}")
    mode_log = np.load(
        run_dir / "logs" / "mode_id.npy", allow_pickle=False)
    if (len(mode_log) != target_next_iteration
            or not np.all(
                mode_log[protocol.BASE_NEXT_ITERATION:] == int(mode))):
        raise RuntimeError("smoke rollout escaped its fixed mode")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument(
        "--profile",
        choices=("legacy_ant", "stochastic_headroom", "structured_channel"),
        default="legacy_ant",
    )
    parser.add_argument("--env")
    parser.add_argument("--mode", type=int, choices=protocol.MODES,
                        required=True)
    parser.add_argument("--target-next-iteration", type=int,
                        default=protocol.FINAL_NEXT_ITERATION)
    parser.add_argument("--status-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.profile == "stochastic_headroom":
        if not args.env:
            raise SystemExit("--env is required for stochastic_headroom")
        protocol.configure_stochastic_headroom(args.env)
    elif args.profile == "structured_channel":
        if not args.env:
            raise SystemExit("--env is required for structured_channel")
        protocol.configure_structured_channel_headroom(args.env)
    elif args.env and args.env != protocol.ENV:
        raise SystemExit(
            f"legacy_ant profile requires env={protocol.ENV}, got {args.env}"
        )
    protocol._require_family(args.family)
    run_dir = args.run_dir.resolve()
    target = int(args.target_next_iteration)
    if not protocol.BASE_NEXT_ITERATION < target <= (
            protocol.FINAL_NEXT_ITERATION):
        raise SystemExit(
            f"target-next-iteration must be in "
            f"[{protocol.BASE_NEXT_ITERATION + 1},"
            f"{protocol.FINAL_NEXT_ITERATION}]")
    if args.smoke != (target < protocol.FINAL_NEXT_ITERATION):
        raise SystemExit(
            "--smoke is required exactly when target-next-iteration is below "
            "the formal final budget")

    protocol.validate_bootstrap(run_dir, args.family, args.mode)
    bootstrap = protocol.read_json(run_dir / protocol.BOOTSTRAP_REL)
    if fork_protocol.sha256_file(Path(__file__).resolve()) != bootstrap.get(
            "runner_sha256"):
        raise RuntimeError("executed specialist runner is not the pinned source")
    checkpoint = protocol.checkpoint_info(run_dir)
    current = int(checkpoint["next_iteration"])
    if current > target:
        raise RuntimeError(
            f"checkpoint already exceeds requested target: {current}>{target}")
    if current < target:
        audit_path = run_dir / "logs" / fork_protocol.BOUNDARY_AUDIT_NAME
        if current == protocol.BASE_NEXT_ITERATION and audit_path.exists():
            stale = run_dir / protocol.PROTOCOL_CHECKPOINT_DIR / (
                f"abandoned_{fork_protocol.BOUNDARY_AUDIT_NAME}")
            if stale.exists():
                raise RuntimeError(
                    "both active and abandoned boundary audits exist")
            os.replace(audit_path, stale)
        execution_root = prepare_execution_source(run_dir)
        try:
            values = training_values(
                run_dir, args.family, args.mode, target,
                boundary_audit=(current == protocol.BASE_NEXT_ITERATION))
            run_train(run_dir, execution_root, values)
        finally:
            shutil.rmtree(execution_root, ignore_errors=True)

    if args.smoke:
        validate_partial(run_dir, args.mode, target)
        if args.status_dir:
            protocol.write_json_atomic(
                args.status_dir.resolve() / "specialist_smoke_summary.json",
                {
                    "schema": (
                        "bapr.v3-stochastic-independent-specialist-smoke.v1"
                    ),
                    "status": "complete",
                    "identity": {
                        "family": args.family,
                        "env": protocol.ENV,
                        "seed": protocol.SEED,
                        "mode": int(args.mode),
                    },
                    "checkpoint": protocol.checkpoint_info(run_dir),
                },
            )
        print(
            f"INDEPENDENT SPECIALIST SMOKE COMPLETE: family={args.family} "
            f"mode={args.mode} next_iter={target}", flush=True)
        return

    bundle = protocol.publish_specialist_bundle(
        run_dir, args.family, args.mode)
    if args.status_dir:
        status_dir = args.status_dir.resolve()
        protocol.write_json_atomic(
            status_dir / "specialist_complete_summary.json",
            {
                "schema": "bapr.v3-stochastic-independent-specialist-status.v1",
                "status": "complete",
                "identity": {
                    "family": args.family,
                    "env": protocol.ENV,
                    "seed": protocol.SEED,
                    "mode": int(args.mode),
                },
                "checkpoint": bundle["checkpoint"],
                "bundle_path": str(protocol.specialist_bundle_dir(
                    args.family, args.mode)),
                "bundle_manifest_sha256": fork_protocol.sha256_file(
                    protocol.specialist_bundle_dir(args.family, args.mode)
                    / protocol.BUNDLE_MANIFEST
                ),
            },
        )
    print(
        "INDEPENDENT SPECIALIST COMPLETE: "
        f"family={args.family} mode={args.mode} "
        f"next_iter={bundle['checkpoint']['next_iteration']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
