#!/usr/bin/env python3
"""Submit strict final audits for the BAPR-v3 budget-matched screen."""
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path

import submit_bapr_v3_budget_matched_screen as training


ROOT = training.ROOT
SCHEDULER = training.SCHEDULER
JAX_PYTHON = training.JAX_PYTHON
RUN_ROOT = training.SAVE_ROOT
OUT_ROOT = ROOT / "jax_experiments" / "results_bapr_v3_budget_matched_audit_v1"
ENVS = training.ENVS
FAMILIES = training.FAMILIES
ROLES = ("robust", "oracle_ladder")
DEFAULT_EVENT_SEEDS = (1100, 1200, 1300, 1400, 1500)


def variant_for_role(role: str) -> training.Variant:
    name = "robust_long" if role == "robust" else "oracle_direct"
    return next(variant for variant in training.VARIANTS if variant.name == name)


def run_dir(family: str, role: str, env: str) -> Path:
    variant = variant_for_role(role)
    return RUN_ROOT / training.run_name(family, variant, env, 0)


def output_dir(family: str, role: str, env: str, event_seed: int) -> Path:
    return (
        OUT_ROOT / family / env.removesuffix("-v2") / role
        / f"event_seed_{event_seed}")


def signature(family: str, role: str, env: str, event_seed: int) -> str:
    return (
        f"BAPR/v3-budget-match-audit/v1/{family}/{role}/{env}/"
        f"event-seed-{event_seed}")


def expected_rows(role: str) -> dict[str, int]:
    multiplier = 1 if role == "robust" else 5
    return {
        "summary.csv": 2 * multiplier,
        "task_returns.csv": 4 * multiplier,
        "switching_returns.csv": 5 * multiplier,
        "switching_trace.csv": 5000 * multiplier,
    }


def output_complete(path: Path, role: str) -> bool:
    try:
        for filename, count in expected_rows(role).items():
            with (path / filename).open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != count:
                return False
            if any(
                int(float(row["checkpoint_next_iter"])) != 1400
                or int(float(row["checkpoint_total_steps"])) != 5_600_000
                for row in rows
            ):
                return False
        return True
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return False


def task_spec(
    family: str,
    role: str,
    env: str,
    event_seed: int,
    args: argparse.Namespace,
) -> dict:
    source_run = run_dir(family, role, env)
    destination = output_dir(family, role, env, event_seed)
    context_args = (
        "--bapr-v2-context-source robust"
        if role == "robust"
        else "--bapr-v2-context-source oracle --oracle-context-ladder"
    )
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
        f"--run-dir {shlex.quote(str(source_run))} "
        f"--out-dir {shlex.quote(str(destination))} "
        f"--episodes-per-task {args.episodes_per_task} "
        f"--switching-episodes {args.switching_episodes} "
        "--switching-period-steps 500 "
        "--heldout-task-stream validation "
        "--detection-window-steps 50 "
        f"{context_args} --bapr-v2-advantage off "
        f"--rng-seed {20260716 + event_seed} "
        f"--eval-seed-offset {event_seed} "
        "--max-tasks 4 --stationary-test-only "
        "--min-checkpoint-next-iter 1400 "
        "--resume-from "
        f"{shlex.quote(str(source_run / 'checkpoints' / 'train_state.pkl'))}"
    )
    return {
        "description": (
            f"BAPR-v3 budget-matched final audit {family} {role} "
            f"{env} event seed {event_seed}"),
        "cmd": command,
        "cwd": str(ROOT),
        "signature": signature(family, role, env, event_seed),
        "project": "BAPR",
        "vram": 5200,
        "ram_mb": 4096,
        "cpu": 2,
        "priority": args.priority,
        "ckpt_dir": str(source_run / "checkpoints"),
        "ckpt_glob": "train_state.pkl",
        "resume_flag": "",
        "result_dir": str(destination),
        "local_result_dir": str(destination),
        "stage_excludes": ["jax_experiments/eval_bundles*/", "paper/"],
        "allow_initial_resume_scan_error": False,
        "reroute_on_node_down": True,
        "node_down_requeue_s": 900,
    }


def submit_jsonl(specs: list[dict]) -> list[str]:
    payload = "".join(json.dumps(spec) + "\n" for spec in specs)
    command = [
        sys.executable, str(SCHEDULER), "submit-jsonl",
        "--stdin", "--trusted", "--json",
        "--intent-label", "bapr-v3-budget-match-audit-submit",
        "--intent-ttl", "900",
    ]
    result = subprocess.run(
        command, input=payload, text=True, capture_output=True,
        env=training.common.scheduler_env())
    if result.returncode != 0:
        print((result.stdout or "") + (result.stderr or ""), file=sys.stderr)
        result.check_returncode()
    response = json.loads(result.stdout)
    print(json.dumps(response, indent=2))
    return [str(item["id"]) for item in response.get("submitted", [])]


def dispatch(task_ids: list[str]) -> None:
    if not task_ids:
        return
    command = [
        sys.executable, str(SCHEDULER), "dispatch", "--bulk-window",
        "--intent-label", "bapr-v3-budget-match-audit-dispatch",
        "--intent-ttl", "900",
    ]
    for task_id in task_ids:
        command += ["--task-id", task_id]
    subprocess.run(command, check=True, env=training.common.scheduler_env())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", action="append", choices=ENVS)
    parser.add_argument("--family", action="append", choices=FAMILIES)
    parser.add_argument("--role", action="append", choices=ROLES)
    parser.add_argument("--event-seed", action="append", type=int)
    parser.add_argument("--episodes-per-task", type=int, default=5)
    parser.add_argument("--switching-episodes", type=int, default=5)
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="high")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

    envs = args.env or list(ENVS)
    families = args.family or list(FAMILIES)
    roles = args.role or list(ROLES)
    event_seeds = args.event_seed or list(DEFAULT_EVENT_SEEDS)
    tasks = training.common.scheduler_tasks()
    active = {
        str(task.get("signature")) for task in tasks
        if task.get("signature")
        and task.get("status") in {"queued", "launching", "running"}
    }
    specs = []
    skipped = []
    for family in families:
        for role in roles:
            for env in envs:
                source_run = run_dir(family, role, env)
                if training.common.last_iteration(source_run) < 1399:
                    raise SystemExit(
                        f"training logs incomplete for {source_run.name}")
                for event_seed in event_seeds:
                    sig = signature(family, role, env, event_seed)
                    destination = output_dir(family, role, env, event_seed)
                    if output_complete(destination, role):
                        skipped.append((sig, "complete"))
                    elif sig in active:
                        skipped.append((sig, "active"))
                    else:
                        specs.append(task_spec(
                            family, role, env, event_seed, args))

    expected = len(families) * len(roles) * len(envs) * len(event_seeds)
    if expected > 40:
        raise SystemExit(f"audit expands to {expected}; cap is 40")
    print(
        f"BAPR-v3 budget-matched audit: expected={expected} "
        f"submit={len(specs)} skip={len(skipped)}", flush=True)
    for spec in specs:
        print(f"  {spec['signature']}", flush=True)
    for sig, reason in skipped:
        print(f"  skip {reason}: {sig}", flush=True)
    if args.dry_run or not specs:
        return

    task_ids = submit_jsonl(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        dispatch(task_ids)


if __name__ == "__main__":
    main()
