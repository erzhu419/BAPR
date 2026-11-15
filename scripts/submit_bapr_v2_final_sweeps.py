#!/usr/bin/env python3
"""Bundle completed BAPR-v2 checkpoints and submit CPU final sweeps."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = Path("/home/erzhu419/mine_code/scheduleurm/skill/scheduler.py")
QUEUE = Path.home() / ".claude" / "scheduler" / "queue.json"
REMOTE_PYTHON = Path(
    "/home/zhengliang01/scheduleurm_work/conda_envs/"
    "csbapr-gpu-py310/bin/python"
)
PHASE1_ROOT = ROOT / "jax_experiments" / "results_bapr_v2_phase1"
V83_ROOT = ROOT / "jax_experiments" / "results_bapr_v2_oracle_capacity"
V84_ROOT = ROOT / "jax_experiments" / "results_bapr_v84_teacher_student"
V85_ROOT = ROOT / "jax_experiments" / "results_bapr_v85_staged_capacity"
V86_ROOT = ROOT / "jax_experiments" / "results_bapr_v86_adaptive_capacity"
V87_ROOT = ROOT / "jax_experiments" / "results_bapr_v87_constrained_deploy"
BUNDLE_ROOT = ROOT / "jax_experiments" / "eval_bundles_bapr_v2"
OUT_ROOT = ROOT / "jax_experiments" / "results_bapr_v2_final_sweeps"
CPU_NODES = [f"node{index:03d}" for index in range(1, 7)]
TERMINAL = {"done", "failed", "cancelled", "forgotten"}

SOURCE_PATTERNS = [
    # The final mechanism verdict needs the complete robust/oracle/learned
    # ladder, not only the cheap first-task proxy used during training.
    (PHASE1_ROOT, "v82*_s0"),
    (V83_ROOT, "v83*_s0"),
    (V84_ROOT, "v84*_s0"),
    (V85_ROOT, "v85*_s0"),
    (V86_ROOT, "v86*_s0"),
    (V87_ROOT, "v87*_s*"),
]


def scheduler_env() -> dict[str, str]:
    env = os.environ.copy()
    path = str(SCHEDULER.parent)
    env["PYTHONPATH"] = (
        path if not env.get("PYTHONPATH")
        else f"{path}{os.pathsep}{env['PYTHONPATH']}"
    )
    return env


def active_signatures() -> set[str]:
    """Return queued/running/done signatures across hot and archived state."""
    command = [
        sys.executable, str(SCHEDULER), "status", "--all", "--json",
        "--brief", "--readonly",
    ]
    try:
        result = subprocess.run(
            command, text=True, capture_output=True, check=True,
            env=scheduler_env(), timeout=30)
        state = json.loads(result.stdout)
        tasks = state.get("tasks", state) if isinstance(state, dict) else state
    except Exception:
        if not QUEUE.exists():
            return set()
        try:
            state = json.loads(QUEUE.read_text())
            tasks = state.get("tasks", [])
        except Exception:
            return set()
    return {
        task["signature"]
        for task in tasks
        if task.get("signature")
        and task.get("status") not in {"failed", "cancelled", "forgotten"}
    }


def queued_ids_for_signatures(signatures: set[str]) -> list[str]:
    """Resolve the exact queued IDs for one targeted bulk-dispatch pass."""
    if not signatures or not QUEUE.exists():
        return []
    try:
        state = json.loads(QUEUE.read_text())
    except Exception:
        return []
    tasks = state.get("tasks", []) if isinstance(state, dict) else []
    return [
        str(task["id"])
        for task in tasks
        if task.get("status") == "queued"
        and task.get("signature") in signatures
        and task.get("id")
    ]


def last_iteration(run_dir: Path) -> int:
    path = run_dir / "logs" / "iteration.npy"
    if not path.exists():
        return -1
    try:
        import numpy as np

        values = np.load(path)
        return int(values[-1]) if len(values) else -1
    except Exception:
        return -1


def final_checkpoint_ready(run_dir: Path) -> bool:
    return all((run_dir / "checkpoints" / name).is_file() for name in (
        "params.pkl", "train_state.pkl"))


def discover_runs(expected_iteration: int) -> list[Path]:
    runs = []
    for root, pattern in SOURCE_PATTERNS:
        for run_dir in sorted(root.glob(pattern)):
            if last_iteration(run_dir) < expected_iteration:
                continue
            if not final_checkpoint_ready(run_dir):
                continue
            runs.append(run_dir)
    return runs


def prepare_bundle(run_dir: Path, bundle_root: Path = BUNDLE_ROOT) -> Path:
    bundle = bundle_root / run_dir.name
    checkpoint_dir = bundle / "checkpoints"
    log_dir = bundle / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    for name in ("params.pkl", "train_state.pkl"):
        shutil.copy2(run_dir / "checkpoints" / name, checkpoint_dir / name)
    shutil.copy2(
        run_dir / "logs" / "protocol_signature.json",
        log_dir / "protocol_signature.json",
    )
    shutil.copy2(
        run_dir / "logs" / "protocol_signature.json",
        checkpoint_dir / "protocol_signature.json",
    )
    code_root = checkpoint_dir / "code"
    if code_root.exists():
        shutil.rmtree(code_root)
    source_root = ROOT / "jax_experiments"
    for source in source_root.rglob("*.py"):
        relative = source.relative_to(source_root)
        if any(
            part == "__pycache__"
            or part.startswith("results")
            or part.startswith("archive")
            or part.startswith("eval_bundles")
            for part in relative.parts
        ):
            continue
        target = code_root / "jax_experiments" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return bundle


def output_complete(out_dir: Path, switching_only: bool = False) -> bool:
    names = (
        ("summary.csv", "switching_returns.csv", "switching_trace.csv")
        if switching_only
        else ("summary.csv", "task_returns.csv", "switching_trace.csv")
    )
    return all(
        (out_dir / name).is_file() and (out_dir / name).stat().st_size > 0
        for name in names
    )


def submit(run_dir: Path, args, active: set[str]) -> str:
    bundle = args.bundle_root / run_dir.name
    output_name = (
        run_dir.name
        if not args.eval_label
        else f"{run_dir.name}__{args.eval_label}")
    out_dir = args.out_root / output_name
    signature = f"{args.signature_prefix.rstrip('/')}/{output_name}"
    if signature in active:
        print(f"skip active {signature}")
        return signature
    if output_complete(out_dir, args.switching_only) and not args.no_skip_complete:
        print(f"skip complete {out_dir}")
        return signature
    if not args.dry_run:
        bundle = prepare_bundle(run_dir, args.bundle_root)

    bundle_relative = bundle.relative_to(ROOT)
    try:
        out_relative = out_dir.relative_to(ROOT)
        out_shell = f'"$PWD/{out_relative.as_posix()}"'
    except ValueError:
        out_shell = shlex.quote(str(out_dir))

    command_text = (
        f'PYTHONPATH="$PWD/{bundle_relative.as_posix()}/checkpoints/code:$PWD" '
        "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        f"OMP_NUM_THREADS={args.cpu} OPENBLAS_NUM_THREADS={args.cpu} "
        f"MKL_NUM_THREADS={args.cpu} "
        f"{shlex.quote(str(args.python))} -u -m "
        "jax_experiments.analysis.final_task_sweep "
        f'--run-dir "$PWD/{bundle_relative.as_posix()}" '
        f"--out-dir {out_shell} "
        f"--episodes-per-task {args.episodes_per_task} "
        f"--switching-episodes {args.switching_episodes} "
        f"--switching-period-steps {args.switching_period_steps} "
        f"--heldout-task-stream {args.heldout_task_stream} "
        f"--detection-window-steps {args.detection_window_steps} "
        f"--bapr-v2-context-source {args.bapr_v2_context_source} "
        f"--bapr-v2-advantage {args.bapr_v2_advantage} "
        f"--rng-seed {args.rng_seed}"
    )
    if args.max_tasks is not None:
        command_text += f" --max-tasks {args.max_tasks}"
    if args.switching_only:
        command_text += " --switching-only"
    command_text += " && echo DONE"
    command = [
        sys.executable, str(SCHEDULER), "submit",
        "--project", "BAPR",
        "--description", f"BAPR-v2 corrected final sweep {output_name}",
        "--cmd", command_text,
        "--cwd", str(ROOT),
        "--signature", signature,
        "--vram", "0",
        "--ram-mb", str(args.ram_mb),
        "--cpu", str(args.cpu),
        "--priority", args.priority,
        "--ckpt-dir", str(bundle / "checkpoints"),
        "--ckpt-glob", "params.pkl",
        "--resume-flag=--resume-from",
        "--result-dir", str(out_dir),
        "--local-result-dir", str(out_dir),
        "--reroute-on-node-down",
        "--node-down-requeue-s", "900",
    ]
    for node in args.allowed_node:
        command += ["--allowed-node", node]
    if args.allow_duplicate:
        command.append("--allow-duplicate")
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return signature
    result = subprocess.run(
        command, text=True, capture_output=True, env=scheduler_env())
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        if "duplicate" in output.lower() or "already queued" in output.lower():
            print(output.strip())
            return signature
        print(output, file=sys.stderr)
        result.check_returncode()
    if result.stdout:
        print(result.stdout.strip())
    return signature


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-iteration", type=int, default=599)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--bundle-root", type=Path, default=BUNDLE_ROOT)
    parser.add_argument("--signature-prefix", default="BAPR/v83-final-sweep")
    parser.add_argument("--python", type=Path, default=REMOTE_PYTHON)
    parser.add_argument("--episodes-per-task", type=int, default=3)
    parser.add_argument("--switching-episodes", type=int, default=5)
    parser.add_argument(
        "--switching-only", action="store_true",
        help="Run only the fixed-horizon switching stream.")
    parser.add_argument("--switching-period-steps", type=int, default=500)
    parser.add_argument(
        "--heldout-task-stream", choices=("validation", "reserved"),
        default="validation")
    parser.add_argument("--detection-window-steps", type=int, default=50)
    parser.add_argument(
        "--bapr-v2-context-source",
        choices=("checkpoint", "robust", "oracle", "learned"),
        default="checkpoint")
    parser.add_argument(
        "--bapr-v2-advantage", choices=("checkpoint", "on", "off"),
        default="checkpoint")
    parser.add_argument(
        "--eval-label",
        help="Append a filesystem/signature label for one policy-ladder mode.")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--rng-seed", type=int, default=20260710)
    parser.add_argument("--cpu", type=int, default=8)
    parser.add_argument("--ram-mb", type=int, default=8192)
    parser.add_argument(
        "--priority", choices=["low", "normal", "high"], default="high")
    parser.add_argument("--allowed-node", action="append", default=None)
    parser.add_argument("--name", action="append",
                        help="Submit only exact run names (repeatable).")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--no-skip-complete", action="store_true")
    parser.add_argument("--allow-duplicate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()
    args.out_root = args.out_root.expanduser().resolve()
    args.bundle_root = args.bundle_root.expanduser().resolve()
    args.allowed_node = args.allowed_node or list(CPU_NODES)
    if args.eval_label and any(
            char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for char in args.eval_label):
        parser.error("--eval-label may contain only letters, digits, '_' and '-'")

    runs = discover_runs(args.expected_iteration)
    if args.name:
        names = set(args.name)
        runs = [run for run in runs if run.name in names]
    if args.max_runs is not None:
        runs = runs[:max(0, args.max_runs)]
    print(f"Ready final-sweep runs: {len(runs)}", flush=True)
    active = active_signatures()
    signatures = {
        submit(run_dir, args, active)
        for run_dir in runs
    }

    if args.dispatch and not args.dry_run:
        queued_ids = queued_ids_for_signatures(signatures)
        if not queued_ids:
            print("No queued BAPR final-sweep tasks need dispatch.")
            return
        command = [
            sys.executable,
            str(SCHEDULER),
            "dispatch",
            "--intent-label",
            "bapr-final-sweep-batch",
            "--intent-ttl",
            "600",
        ]
        for task_id in queued_ids:
            command += ["--task-id", task_id]
        print(
            f"Targeted bulk dispatch: {len(queued_ids)} BAPR task(s)",
            flush=True,
        )
        subprocess.run(command, check=True, env=scheduler_env())


if __name__ == "__main__":
    main()
