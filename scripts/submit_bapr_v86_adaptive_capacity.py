#!/usr/bin/env python3
"""Submit the BAPR-v86 adaptive-capacity seed0 screen via scheduler."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import submit_bapr_v85_staged_capacity as v85


ROOT = v85.ROOT
SCHEDULER = v85.SCHEDULER
JAX_PYTHON = v85.JAX_PYTHON
QUEUE = v85.QUEUE
SAVE_ROOT = ROOT / "jax_experiments" / "results_bapr_v86_adaptive_capacity"
ENVS = v85.ENVS


@dataclass(frozen=True)
class Variant:
    name: str
    policy_mode: str = "gated_direct"
    residual_delta: float = 0.25
    context_burnin: int = 16
    min_history: int = 32
    warmstart_conditioned: bool = True
    policy_gate_init: float = 0.0
    action_deviation_weight: float = 0.0


VARIANTS = (
    Variant("v86a_gated_direct_d0", action_deviation_weight=0.0),
    Variant("v86b_gated_direct_d1", action_deviation_weight=1.0),
    Variant("v86c_gated_direct_d10", action_deviation_weight=10.0),
)


def training_args(variant: Variant, env: str, args, run_name: str) -> str:
    values = shlex.split(v85.training_args(variant, env, args, run_name))
    save_index = values.index("--save_root") + 1
    values[save_index] = str(SAVE_ROOT)
    values.extend([
        "--bapr_v2_policy_gate_init", str(variant.policy_gate_init),
        "--bapr_v2_action_deviation_weight",
        str(variant.action_deviation_weight),
    ])
    return shlex.join(values)


def submit_task(variant: Variant, env: str, args,
                active: set[str]) -> str | None:
    env_short = env.removesuffix("-v2")
    run_name = f"{variant.name}_{env_short}_s0"
    signature = f"BAPR/v86-adaptive-capacity/{variant.name}/{env}/seed0"
    if v85.last_iteration(SAVE_ROOT / run_name) >= args.max_iters - 1:
        print(f"skip complete {run_name}")
        return None
    if signature in active:
        print(f"skip active {signature}")
        return None

    run_dir = SAVE_ROOT / run_name
    command_text = (
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.28 "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        f"{shlex.quote(str(JAX_PYTHON))} -u -m jax_experiments.train "
        f"{training_args(variant, env, args, run_name)}"
    )
    command = [
        sys.executable, str(SCHEDULER), "submit",
        "--project", "BAPR",
        "--description", f"BAPR-v86 adaptive capacity {run_name}",
        "--cmd", command_text,
        "--cwd", str(ROOT),
        "--signature", signature,
        "--vram", "3400",
        "--ram-mb", "6144",
        "--cpu", "2",
        "--priority", args.priority,
        "--ckpt-dir", str(run_dir / "checkpoints"),
        "--result-dir", str(run_dir / "logs"),
        "--local-result-dir", str(run_dir / "logs"),
        "--allow-remote-large-data",
        "--reroute-on-node-down",
        "--node-down-requeue-s", "900",
    ]
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return signature
    result = subprocess.run(
        command, text=True, capture_output=True, env=v85.scheduler_env())
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        if "duplicate" in output.lower() or "already queued" in output.lower():
            print(output.strip())
            return None
        print(output, file=sys.stderr)
        result.check_returncode()
    if result.stdout:
        print(result.stdout.strip())
    return signature


def queued_ids(signatures: set[str]) -> list[str]:
    if not QUEUE.exists():
        return []
    try:
        tasks = json.loads(QUEUE.read_text()).get("tasks", [])
    except Exception:
        return []
    return [
        str(task["id"]) for task in tasks
        if task.get("status") == "queued"
        and task.get("signature") in signatures
        and task.get("id")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-iters", type=int, default=1200)
    parser.add_argument("--base-iters", type=int, default=600)
    parser.add_argument("--teacher-iters", type=int, default=400)
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="high")
    parser.add_argument("--variant", action="append",
                        choices=[variant.name for variant in VARIANTS])
    parser.add_argument("--env", action="append", choices=ENVS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

    if args.base_iters + args.teacher_iters >= args.max_iters:
        raise SystemExit("max-iters must leave at least one student iteration")
    for path in (SCHEDULER, JAX_PYTHON):
        if not path.exists():
            raise SystemExit(f"missing required path: {path}")

    active = {
        str(task.get("signature")) for task in v85.scheduler_tasks()
        if task.get("signature")
        and task.get("status") not in {"failed", "cancelled", "forgotten"}
    }
    variants = [
        variant for variant in VARIANTS
        if not args.variant or variant.name in args.variant]
    envs = args.env or list(ENVS)
    submitted = set()
    for variant in variants:
        for env in envs:
            signature = submit_task(variant, env, args, active)
            if signature:
                submitted.add(signature)

    if args.dispatch and not args.dry_run:
        task_ids = queued_ids(submitted)
        if not task_ids:
            print("No queued v86 tasks need dispatch.")
            return
        command = [
            sys.executable, str(SCHEDULER), "dispatch",
            "--intent-label", "bapr-v86-seed0-screen",
            "--intent-ttl", "600",
        ]
        for task_id in task_ids:
            command += ["--task-id", task_id]
        subprocess.run(command, check=True, env=v85.scheduler_env())


if __name__ == "__main__":
    main()
