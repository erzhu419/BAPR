#!/usr/bin/env python3
"""Submit eval-only final task sweeps for splitfix paper-continuous runs."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = Path("/home/erzhu419/mine_code/scheduleurm/skill/scheduler.py")
JAX_PYTHON = Path("/home/erzhu419/.conda/envs/resac-jax/bin/python")
DEFAULT_RESULTS_ROOT = ROOT / "jax_experiments" / "results_evalfix_splitfix"
DEFAULT_OUT_ROOT = ROOT / "jax_experiments" / "results_evalfix_splitfix_finaleval"
SCHEDULER_QUEUE = Path.home() / ".claude" / "scheduler" / "queue.json"
TERMINAL_STATES = {"done", "failed", "cancelled", "forgotten"}
# Final sweeps consume checkpoint/result directories as inputs. Scheduleurm
# stages code, not arbitrary run artifacts, so default to the local node where
# the collected splitfix runs live. Use --allowed-node explicitly only after
# staging the input run directories to that node.
DEFAULT_ALLOWED_NODES = ["local"]


def scheduler_env() -> dict[str, str]:
    env = os.environ.copy()
    skill_path = str(SCHEDULER.parent)
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        skill_path if not current else f"{skill_path}{os.pathsep}{current}"
    )
    return env


def parse_run(run_dir: Path):
    parts = run_dir.name.split("_")
    if len(parts) < 3 or not parts[-1].startswith("s"):
        raise ValueError(f"unsupported run name: {run_dir.name}")
    return {
        "algo": parts[-3],
        "env": parts[-2],
        "seed": parts[-1].lstrip("s"),
    }


def active_signatures() -> set[str]:
    if not SCHEDULER_QUEUE.exists():
        return set()
    try:
        state = json.loads(SCHEDULER_QUEUE.read_text())
    except Exception:
        return set()
    out = set()
    for task in state.get("tasks", []):
        sig = task.get("signature")
        status = task.get("status")
        if sig and status not in TERMINAL_STATES:
            out.add(sig)
    return out


def is_complete(out_dir: Path) -> bool:
    summary = out_dir / "summary.csv"
    return summary.exists() and summary.stat().st_size > 0


def submit(run_dir: Path, out_root: Path, args, active: set[str]) -> None:
    meta = parse_run(run_dir)
    out_dir = out_root / run_dir.name
    signature = (
        f"{args.signature_prefix.rstrip('/')}/"
        f"{meta['algo']}/{meta['env']}/seed{meta['seed']}"
    )
    if signature in active and not args.no_skip_active:
        print(f"skip active {signature}")
        return
    if is_complete(out_dir) and not args.no_skip_complete:
        print(f"skip complete {out_dir}")
        return

    cmd = (
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        f"XLA_PYTHON_CLIENT_MEM_FRACTION={args.xla_mem_fraction} "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
        f"{shlex.quote(str(JAX_PYTHON))} -u -m "
        "jax_experiments.analysis.final_task_sweep "
        f"--run-dir {shlex.quote(str(run_dir))} "
        f"--out-dir {shlex.quote(str(out_dir))} "
        f"--episodes-per-task {args.episodes_per_task} "
        f"--switching-episodes {args.switching_episodes} "
        f"--switching-period-steps {args.switching_period_steps} "
        f"--rng-seed {args.rng_seed}"
    )
    if args.max_tasks is not None:
        cmd += f" --max-tasks {args.max_tasks}"
    cmd += " && echo DONE"

    command = [
        sys.executable, str(SCHEDULER), "submit",
        "--project", "BAPR",
        "--description", f"BAPR final task sweep: {run_dir.name}",
        "--cmd", cmd,
        "--cwd", str(ROOT),
        "--signature", signature,
        "--vram", str(args.vram),
        "--ram-mb", str(args.ram_mb),
        "--cpu", str(args.cpu),
        "--priority", args.priority,
        "--result-dir", str(out_dir),
        "--local-result-dir", str(out_dir),
        "--allow-remote-large-data",
        "--reroute-on-node-down",
        "--node-down-requeue-s", "900",
        "--allow-duplicate",
    ]
    for node in args.allowed_node:
        command += ["--allowed-node", node]

    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    completed = subprocess.run(
        command, text=True, capture_output=True, env=scheduler_env())
    output = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0:
        low = output.lower()
        if "duplicate" in low or "already queued" in low:
            print(output.strip())
            return
        print(output, file=sys.stderr)
        completed.check_returncode()
    if completed.stdout:
        print(completed.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-glob", default="splitfix_continuous_gravity_*_*_s0")
    parser.add_argument("--signature-prefix",
                        default="BAPR/final-sweep/splitfix-paper-continuous")
    parser.add_argument("--episodes-per-task", type=int, default=3)
    parser.add_argument("--switching-episodes", type=int, default=5)
    parser.add_argument("--switching-period-steps", type=int, default=500)
    parser.add_argument("--rng-seed", type=int, default=20260707)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--priority", choices=["low", "normal", "high"],
                        default="high")
    parser.add_argument("--vram", type=int, default=2000)
    parser.add_argument("--ram-mb", type=int, default=4096)
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--xla-mem-fraction", default="0.25")
    parser.add_argument("--allowed-node", action="append", default=None)
    parser.add_argument("--no-skip-complete", action="store_true")
    parser.add_argument("--no-skip-active", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

    if args.allowed_node is None:
        args.allowed_node = list(DEFAULT_ALLOWED_NODES)
    runs = sorted(args.results_root.glob(args.run_glob))
    runs = [run for run in runs if (run / "checkpoints").exists()]
    active = active_signatures()
    print(f"Submitting final eval sweeps: {len(runs)} candidates", flush=True)
    for run_dir in runs:
        submit(run_dir, args.out_root, args, active)

    if args.dispatch and not args.dry_run:
        command = [sys.executable, str(SCHEDULER), "dispatch"]
        print(shlex.join(command), flush=True)
        subprocess.run(command, check=False, env=scheduler_env())


if __name__ == "__main__":
    main()
