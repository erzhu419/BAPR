#!/usr/bin/env python3
"""Submit paired multi-event-seed audits for final inverse BAPR-v3 checkpoints."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import submit_bapr_v3_burst_student as common_submit


ROOT = common_submit.ROOT
SCHEDULER = common_submit.SCHEDULER
JAX_PYTHON = common_submit.JAX_PYTHON
RUN_ROOT = common_submit.SAVE_ROOT
OUT_ROOT = ROOT / "jax_experiments" / "results_bapr_v3_inverse_audit_v2"
ENVS = common_submit.ENVS
MODES = ("robust", "oracle", "learned")
DEFAULT_EVENT_SEEDS = (1100, 1200, 1300, 1400, 1500)


def run_name(env: str) -> str:
    return common_submit.run_name(env)


def output_dir(env: str, mode: str, event_seed: int) -> Path:
    short_env = env.removesuffix("-v2")
    return OUT_ROOT / short_env / mode / f"event_seed_{event_seed}"


def signature(env: str, mode: str, event_seed: int) -> str:
    return (
        f"BAPR/v3-inverse-audit/v2/{env}/{mode}/"
        f"event-seed-{event_seed}")


def task_spec(
    env: str,
    mode: str,
    event_seed: int,
    args: argparse.Namespace,
) -> dict:
    run_dir = RUN_ROOT / run_name(env)
    out_dir = output_dir(env, mode, event_seed)
    command = (
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.20 "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 "
        f"{shlex.quote(str(JAX_PYTHON))} -u -m "
        "jax_experiments.analysis.final_task_sweep "
        f"--run-dir {shlex.quote(str(run_dir))} "
        f"--out-dir {shlex.quote(str(out_dir))} "
        f"--episodes-per-task {args.episodes_per_task} "
        f"--switching-episodes {args.switching_episodes} "
        "--switching-period-steps 500 "
        "--heldout-task-stream validation "
        "--detection-window-steps 50 "
        f"--bapr-v2-context-source {mode} "
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
            f"BAPR-v3 inverse final audit {env} {mode} seed {event_seed}"),
        "cmd": command,
        "cwd": str(ROOT),
        "signature": signature(env, mode, event_seed),
        "project": "BAPR",
        "vram": 2600,
        "ram_mb": 4096,
        "cpu": 2,
        "priority": args.priority,
        # The iter-2600 generation currently exists only on node007. Keep the
        # audit checkpoint-local until scheduler freshness migration is proven.
        "require_node": "node007",
        "ckpt_dir": str(run_dir / "checkpoints"),
        "ckpt_glob": "train_state.pkl",
        "resume_flag": "",
        "result_dir": str(out_dir),
        "local_result_dir": str(out_dir),
        "stage_excludes": [
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "allow_initial_resume_scan_error": False,
        "reroute_on_node_down": True,
        "node_down_requeue_s": 900,
    }


def output_complete(path: Path) -> bool:
    return all(
        (path / name).is_file() and (path / name).stat().st_size > 0
        for name in (
            "summary.csv", "task_returns.csv",
            "switching_returns.csv", "switching_trace.csv")
    )


def submit_jsonl(specs: list[dict]) -> list[str]:
    payload = "".join(json.dumps(spec) + "\n" for spec in specs)
    command = [
        sys.executable, str(SCHEDULER), "submit-jsonl",
        "--stdin", "--trusted", "--json",
        "--intent-label", "bapr-v3-inverse-audit-submit",
        "--intent-ttl", "900",
    ]
    result = subprocess.run(
        command, input=payload, text=True, capture_output=True,
        env=common_submit.common.scheduler_env())
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
        "--intent-label", "bapr-v3-inverse-audit-dispatch",
        "--intent-ttl", "900",
    ]
    for task_id in task_ids:
        command += ["--task-id", task_id]
    subprocess.run(command, check=True, env=common_submit.common.scheduler_env())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", action="append", choices=ENVS)
    parser.add_argument("--mode", action="append", choices=MODES)
    parser.add_argument("--event-seed", action="append", type=int)
    parser.add_argument("--episodes-per-task", type=int, default=5)
    parser.add_argument("--switching-episodes", type=int, default=5)
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="high")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

    envs = args.env or list(ENVS)
    modes = args.mode or list(MODES)
    event_seeds = args.event_seed or list(DEFAULT_EVENT_SEEDS)
    tasks = common_submit.common.scheduler_tasks()
    existing = {
        str(task.get("signature")): str(task.get("status"))
        for task in tasks if task.get("signature")
    }
    specs = []
    skipped = []
    for env in envs:
        run_dir = RUN_ROOT / run_name(env)
        iteration = common_submit.common.last_iteration(run_dir)
        if iteration < 2599:
            raise SystemExit(
                f"{env} final inverse logs incomplete: iter={iteration}")
        for mode in modes:
            for event_seed in event_seeds:
                sig = signature(env, mode, event_seed)
                out_dir = output_dir(env, mode, event_seed)
                if output_complete(out_dir):
                    skipped.append((sig, "complete"))
                elif existing.get(sig) in {"queued", "launching", "running", "done"}:
                    skipped.append((sig, existing[sig]))
                else:
                    specs.append(task_spec(env, mode, event_seed, args))

    print(
        f"BAPR-v3 inverse audit: expected="
        f"{len(envs) * len(modes) * len(event_seeds)} "
        f"submit={len(specs)} skip={len(skipped)}",
        flush=True)
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
