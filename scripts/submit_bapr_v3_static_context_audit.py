#!/usr/bin/env python3
"""Submit fixed-context controls for the final inverse BAPR-v3 checkpoint."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import submit_bapr_v3_inverse_audit as inverse_audit


ROOT = inverse_audit.ROOT
SCHEDULER = inverse_audit.SCHEDULER
JAX_PYTHON = inverse_audit.JAX_PYTHON
RUN_ROOT = inverse_audit.RUN_ROOT
OUT_ROOT = ROOT / "jax_experiments" / "results_bapr_v3_static_context_audit_v1"
ENVS = inverse_audit.ENVS
DEFAULT_MODES = (0, 1, 2, 3)
DEFAULT_EVENT_SEEDS = inverse_audit.DEFAULT_EVENT_SEEDS


def output_dir(env: str, mode_id: int, event_seed: int) -> Path:
    short_env = env.removesuffix("-v2")
    return (
        OUT_ROOT / short_env / f"fixed_mode_{mode_id}"
        / f"event_seed_{event_seed}")


def signature(env: str, mode_id: int, event_seed: int) -> str:
    return (
        f"BAPR/v3-static-context/v1/{env}/fixed-mode-{mode_id}/"
        f"event-seed-{event_seed}")


def task_spec(
    env: str,
    mode_id: int,
    event_seed: int,
    args: argparse.Namespace,
) -> dict:
    run_dir = RUN_ROOT / inverse_audit.run_name(env)
    out_dir = output_dir(env, mode_id, event_seed)
    command = (
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.20 "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 JAX_NUM_THREADS=1 "
        "TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 "
        f"{shlex.quote(str(JAX_PYTHON))} -u -m "
        "jax_experiments.analysis.final_task_sweep "
        f"--run-dir {shlex.quote(str(run_dir))} "
        f"--out-dir {shlex.quote(str(out_dir))} "
        f"--episodes-per-task {args.episodes_per_task} "
        f"--switching-episodes {args.switching_episodes} "
        "--switching-period-steps 500 "
        "--heldout-task-stream validation "
        "--detection-window-steps 50 "
        "--bapr-v2-context-source oracle "
        f"--fixed-oracle-mode-id {mode_id} "
        "--bapr-v2-advantage off "
        f"--rng-seed {20260715 + event_seed} "
        f"--eval-seed-offset {event_seed} "
        "--max-tasks 4 --stationary-test-only "
        "--min-checkpoint-next-iter 2600 "
        "--resume-from "
        f"{shlex.quote(str(run_dir / 'checkpoints' / 'train_state.pkl'))}"
    )
    return {
        "description": (
            f"BAPR-v3 static context audit {env} mode {mode_id} "
            f"event seed {event_seed}"),
        "cmd": command,
        "cwd": str(ROOT),
        "signature": signature(env, mode_id, event_seed),
        "project": "BAPR",
        # Reserve two slots per 11GB GPU. The evaluator itself is small, but
        # more than eight concurrent JAX compilations can exhaust pthreads.
        "vram": 5200,
        "ram_mb": 4096,
        "cpu": 2,
        "priority": args.priority,
        "require_node": "node007",
        "ckpt_dir": str(run_dir / "checkpoints"),
        "ckpt_glob": "train_state.pkl",
        "resume_flag": "",
        "result_dir": str(out_dir),
        "local_result_dir": str(out_dir),
        "stage_excludes": ["jax_experiments/eval_bundles*/", "paper/"],
        "allow_initial_resume_scan_error": False,
        "reroute_on_node_down": True,
        "node_down_requeue_s": 900,
    }


def output_complete(path: Path) -> bool:
    return inverse_audit.output_complete(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", action="append", choices=ENVS)
    parser.add_argument("--mode-id", action="append", type=int,
                        choices=DEFAULT_MODES)
    parser.add_argument("--event-seed", action="append", type=int)
    parser.add_argument("--episodes-per-task", type=int, default=5)
    parser.add_argument("--switching-episodes", type=int, default=5)
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="high")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

    envs = args.env or list(ENVS)
    mode_ids = args.mode_id or list(DEFAULT_MODES)
    event_seeds = args.event_seed or list(DEFAULT_EVENT_SEEDS)
    tasks = inverse_audit.common_submit.common.scheduler_tasks()
    existing = {
        str(task.get("signature")): str(task.get("status"))
        for task in tasks if task.get("signature")
    }
    specs = []
    skipped = []
    for env in envs:
        run_dir = RUN_ROOT / inverse_audit.run_name(env)
        iteration = inverse_audit.common_submit.common.last_iteration(run_dir)
        if iteration < 2599:
            raise SystemExit(
                f"{env} final inverse logs incomplete: iter={iteration}")
        for mode_id in mode_ids:
            for event_seed in event_seeds:
                sig = signature(env, mode_id, event_seed)
                out_dir = output_dir(env, mode_id, event_seed)
                if output_complete(out_dir):
                    skipped.append((sig, "complete"))
                elif existing.get(sig) in {
                        "queued", "launching", "running", "done"}:
                    skipped.append((sig, existing[sig]))
                else:
                    specs.append(task_spec(env, mode_id, event_seed, args))

    expected = len(envs) * len(mode_ids) * len(event_seeds)
    print(
        f"BAPR-v3 static context audit: expected={expected} "
        f"submit={len(specs)} skip={len(skipped)}",
        flush=True)
    for spec in specs:
        print(f"  {spec['signature']}", flush=True)
    for sig, reason in skipped:
        print(f"  skip {reason}: {sig}", flush=True)
    if args.dry_run or not specs:
        return
    task_ids = inverse_audit.submit_jsonl(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        inverse_audit.dispatch(task_ids)


if __name__ == "__main__":
    main()
