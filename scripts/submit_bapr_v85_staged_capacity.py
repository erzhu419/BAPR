#!/usr/bin/env python3
"""Submit the BAPR-v85 staged teacher-capacity seed0 screen."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = Path("/home/erzhu419/mine_code/scheduleurm/skill/scheduler.py")
JAX_PYTHON = Path("/home/erzhu419/.conda/envs/resac-jax/bin/python")
QUEUE = Path.home() / ".claude" / "scheduler" / "queue.json"
SAVE_ROOT = ROOT / "jax_experiments" / "results_bapr_v85_staged_capacity"
ENVS = ("Ant-v2", "HalfCheetah-v2")


@dataclass(frozen=True)
class Variant:
    name: str
    policy_mode: str
    residual_delta: float
    context_burnin: int
    min_history: int
    warmstart_conditioned: bool = False


VARIANTS = (
    # Intermediate bounded capacity between v84's successful 0.25 and failed 1.0.
    Variant("v85a_residual_r050", "residual", 0.50, 16, 32),
    # Full task-conditioned actor, initialized exactly from the frozen robust base.
    Variant("v85b_direct_warm", "direct", 0.25, 16, 32, True),
    # Same teacher with earlier supervision/confidence for short Ant episodes.
    Variant("v85c_direct_warm_fast", "direct", 0.25, 4, 16, True),
)


def scheduler_env() -> dict[str, str]:
    env = os.environ.copy()
    path = str(SCHEDULER.parent)
    env["PYTHONPATH"] = (
        path if not env.get("PYTHONPATH")
        else f"{path}{os.pathsep}{env['PYTHONPATH']}")
    return env


def scheduler_tasks() -> list[dict]:
    command = [
        sys.executable, str(SCHEDULER), "status", "--all", "--json",
        "--brief", "--readonly",
    ]
    try:
        result = subprocess.run(
            command, text=True, capture_output=True, check=True,
            env=scheduler_env(), timeout=30)
        payload = json.loads(result.stdout)
        return payload.get("tasks", payload) if isinstance(payload, dict) else payload
    except Exception:
        if not QUEUE.exists():
            return []
        try:
            return json.loads(QUEUE.read_text()).get("tasks", [])
        except Exception:
            return []


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


def training_args(variant: Variant, env: str, args, run_name: str) -> str:
    values = [
        "--algo", "bapr_v2",
        "--env", env,
        "--seed", "0",
        "--env_type", "continuous",
        "--task_num", "40",
        "--test_task_num", "40",
        "--changing_period", "20000",
        "--varying_params", "gravity",
        "--log_scale_limit", "3.0",
        "--task_scale_distribution", "pow1p5",
        "--ensemble_size", "5",
        "--hidden_dim", "256",
        "--context_warmup_iters", "0",
        "--bapr_v2_mode", "supervised",
        "--bapr_v2_latent_dim", "4",
        "--bapr_v2_latent_scale_mode", "task_distribution",
        "--bapr_v2_policy_context_source", "oracle_task",
        "--bapr_v2_training_schedule", "teacher_student",
        "--bapr_v2_base_pretrain_iters", str(args.base_iters),
        "--bapr_v2_teacher_iters", str(args.teacher_iters),
        "--bapr_v2_context_hidden_dim", "128",
        "--bapr_v2_context_length", "64",
        "--bapr_v2_context_chunks", "8",
        "--bapr_v2_context_burnin", str(variant.context_burnin),
        "--bapr_v2_min_history", str(variant.min_history),
        "--bapr_v2_context_lr", "0.0003",
        "--bapr_v2_predictive_weight", "0.25",
        "--bapr_v2_supervised_weight", "10.0",
        "--bapr_v2_temporal_weight", "0.01",
        "--bapr_v2_policy_mode", variant.policy_mode,
        "--bapr_v2_residual_delta", str(variant.residual_delta),
        "--bapr_v2_context_dropout", "0.0",
        "--bapr_v2_base_aux_weight", "0.0",
        "--no_bapr_v2_fallback",
        "--bapr_v2_actor_objective", "mean",
        "--bapr_v2_beta_ood", "0.0",
        "--bapr_v2_reg_weight", "0.0",
        "--max_iters", str(args.max_iters),
        "--samples_per_iter", "4000",
        "--updates_per_iter", "250",
        "--log_interval", "20",
        "--eval_episodes", "3",
        "--eval_protocol", "full",
        "--eval_switching_episodes", "2",
        "--eval_switching_period_steps", "500",
        "--save_interval", "50",
        "--save_root", str(SAVE_ROOT),
        "--run_name", run_name,
        "--backend", "spring",
        "--resume",
    ]
    if variant.warmstart_conditioned:
        values.append("--bapr_v2_warmstart_conditioned")
    return shlex.join(values)


def submit_task(variant: Variant, env: str, args,
                active: set[str]) -> str | None:
    env_short = env.removesuffix("-v2")
    run_name = f"{variant.name}_{env_short}_s0"
    signature = f"BAPR/v85-staged-capacity/{variant.name}/{env}/seed0"
    if last_iteration(SAVE_ROOT / run_name) >= args.max_iters - 1:
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
        "--description", f"BAPR-v85 staged capacity {run_name}",
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
        command, text=True, capture_output=True, env=scheduler_env())
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
        str(task.get("signature")) for task in scheduler_tasks()
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
            print("No queued v85 tasks need dispatch.")
            return
        command = [
            sys.executable, str(SCHEDULER), "dispatch",
            "--intent-label", "bapr-v85-seed0-screen",
            "--intent-ttl", "600",
        ]
        for task_id in task_ids:
            command += ["--task-id", task_id]
        subprocess.run(command, check=True, env=scheduler_env())


if __name__ == "__main__":
    main()
