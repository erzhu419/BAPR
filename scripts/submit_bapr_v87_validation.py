#!/usr/bin/env python3
"""Submit the fixed robust/oracle/learned V87 validation ladder."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import submit_bapr_v2_final_sweeps as sweeps


ROOT = sweeps.ROOT
TRAIN_ROOT = sweeps.V87_ROOT
BUNDLE_ROOT = ROOT / "jax_experiments" / "eval_bundles_bapr_v87"
OUT_ROOT = ROOT / "jax_experiments" / "results_bapr_v87_validation"
MODES = ("robust", "oracle", "learned")
EXPECTED_RUNS = 12


def completed_runs(expected_iteration: int) -> list[Path]:
    runs = []
    for run_dir in sorted(TRAIN_ROOT.glob("v87*_s0")):
        if sweeps.last_iteration(run_dir) < expected_iteration:
            continue
        if not sweeps.final_checkpoint_ready(run_dir):
            continue
        runs.append(run_dir)
    return runs


def eval_args(cli, mode: str) -> SimpleNamespace:
    allowed_nodes = [
        node for node in sweeps.CPU_NODES
        if node not in set(cli.exclude_node)]
    if not allowed_nodes:
        raise ValueError("--exclude-node removed every CPU evaluation node")
    return SimpleNamespace(
        # Scheduler treats ckpt_dir as writable. Keep one immutable bundle per
        # eval mode so the three concurrent reads never share a checkpoint
        # ownership path.
        bundle_root=cli.bundle_root / mode,
        out_root=cli.out_root,
        signature_prefix=f"BAPR/v87-validation/{mode}",
        python=sweeps.REMOTE_PYTHON,
        episodes_per_task=cli.episodes_per_task,
        switching_episodes=cli.switching_episodes,
        switching_period_steps=500,
        heldout_task_stream="validation",
        detection_window_steps=50,
        bapr_v2_context_source=mode,
        bapr_v2_advantage="checkpoint",
        eval_label=mode,
        max_tasks=None,
        rng_seed=20260711,
        cpu=cli.cpu,
        ram_mb=cli.ram_mb,
        priority=cli.priority,
        allowed_node=allowed_nodes,
        allow_duplicate=False,
        dry_run=cli.dry_run,
        switching_only=False,
        no_skip_complete=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-iteration", type=int, default=1399)
    parser.add_argument("--mode", action="append", choices=MODES)
    parser.add_argument("--episodes-per-task", type=int, default=3)
    parser.add_argument("--switching-episodes", type=int, default=5)
    parser.add_argument("--cpu", type=int, default=8)
    parser.add_argument("--ram-mb", type=int, default=8192)
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="high")
    parser.add_argument("--bundle-root", type=Path, default=BUNDLE_ROOT)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument(
        "--exclude-node", action="append", default=[],
        choices=sweeps.CPU_NODES,
        help="Exclude a temporarily unhealthy CPU node from new retries.")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()
    args.bundle_root = args.bundle_root.expanduser().resolve()
    args.out_root = args.out_root.expanduser().resolve()

    runs = completed_runs(args.expected_iteration)
    if len(runs) != EXPECTED_RUNS and not args.allow_partial:
        raise SystemExit(
            f"V87 validation requires {EXPECTED_RUNS} final checkpoints; "
            f"found {len(runs)}")
    modes = args.mode or list(MODES)
    print(
        f"V87 validation matrix: {len(runs)} runs x {len(modes)} modes "
        f"= {len(runs) * len(modes)} tasks",
        flush=True,
    )

    active = sweeps.active_signatures()
    signatures: set[str] = set()
    for mode in modes:
        mode_args = eval_args(args, mode)
        for run_dir in runs:
            signature = sweeps.submit(run_dir, mode_args, active)
            signatures.add(signature)
            active.add(signature)

    if not args.dispatch or args.dry_run:
        return
    task_ids = sweeps.queued_ids_for_signatures(signatures)
    if not task_ids:
        print("No queued V87 validation tasks need dispatch.")
        return
    command = [
        sys.executable,
        str(sweeps.SCHEDULER),
        "dispatch",
        "--intent-label",
        "bapr-v87-validation-ladder",
        "--intent-ttl",
        "900",
    ]
    for task_id in task_ids:
        command += ["--task-id", task_id]
    print(f"Targeted bulk dispatch: {len(task_ids)} tasks", flush=True)
    subprocess.run(command, check=True, env=sweeps.scheduler_env())


if __name__ == "__main__":
    main()
