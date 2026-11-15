#!/usr/bin/env python3
"""Submit the BAPR-v84 teacher/student seed0 screen via scheduler."""
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
BUS_PYTHON = Path("/home/erzhu419/miniconda3/envs/csbapr/bin/python")
QUEUE = Path.home() / ".claude" / "scheduler" / "queue.json"
SAVE_ROOT = ROOT / "jax_experiments" / "results_bapr_v84_teacher_student"
BUS_ROOT = ROOT / "bus_experiments_v84_smoke"
ENVS = ("Ant-v2", "HalfCheetah-v2")


@dataclass(frozen=True)
class Variant:
    name: str
    residual_delta: float


VARIANTS = (
    Variant("v84a_teacher_student_r025", 0.25),
    Variant("v84b_teacher_student_r100", 1.0),
)


def scheduler_env() -> dict[str, str]:
    env = os.environ.copy()
    path = str(SCHEDULER.parent)
    env["PYTHONPATH"] = (
        path if not env.get("PYTHONPATH")
        else f"{path}{os.pathsep}{env['PYTHONPATH']}"
    )
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


def env_short(env: str) -> str:
    return env.removesuffix("-v2")


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
        "--bapr_v2_context_burnin", "16",
        "--bapr_v2_context_lr", "0.0003",
        "--bapr_v2_predictive_weight", "0.25",
        "--bapr_v2_supervised_weight", "10.0",
        "--bapr_v2_temporal_weight", "0.01",
        "--bapr_v2_policy_mode", "residual",
        "--bapr_v2_residual_delta", str(variant.residual_delta),
        "--bapr_v2_context_dropout", "0.0",
        "--bapr_v2_base_aux_weight", "0.0",
        "--bapr_v2_advantage_gate",
        "--bapr_v2_advantage_margin", "0.0",
        "--bapr_v2_advantage_lcb_scale", "1.0",
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
    return shlex.join(values)


def submit_spec(command: list[str], *, dry_run: bool) -> None:
    print(shlex.join(command), flush=True)
    if dry_run:
        return
    result = subprocess.run(
        command, text=True, capture_output=True, env=scheduler_env())
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        if "duplicate" in output.lower() or "already queued" in output.lower():
            print(output.strip())
            return
        print(output, file=sys.stderr)
        result.check_returncode()
    if result.stdout:
        print(result.stdout.strip())


def submit_training(variant: Variant, env: str, args,
                    active: set[str]) -> str | None:
    run_name = f"{variant.name}_{env_short(env)}_s0"
    signature = f"BAPR/v84-teacher-student/{variant.name}/{env}/seed0"
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
        "--description", f"BAPR-v84 teacher/student {run_name}",
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
    submit_spec(command, dry_run=args.dry_run)
    return signature


def bus_smoke_complete() -> bool:
    rewards = BUS_ROOT / "logs" / "v84_regsign_s0" / "rewards.npy"
    final_policy = BUS_ROOT / "model" / "v84_regsign_s0" / "final_policy"
    if not rewards.exists() or not final_policy.exists():
        return False
    try:
        import numpy as np

        return len(np.load(rewards)) >= 1 and final_policy.stat().st_size > 0
    except Exception:
        return False


def submit_bus_smoke(args, active: set[str]) -> str | None:
    signature = "BAPR-BUS/v84-regression-smoke/weight-reg-0p01/seed0"
    if bus_smoke_complete():
        print("skip complete v84 bus regularization smoke")
        return None
    if signature in active:
        print(f"skip active {signature}")
        return None
    command_text = (
        "unset LD_LIBRARY_PATH; MPLBACKEND=Agg "
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        f"{shlex.quote(str(BUS_PYTHON))} -u sac_ensemble_bapr.py "
        "--max_episodes 2 --enable_mode_switch "
        "--mode_switch_min 1800 --mode_switch_max 7200 "
        f"--save_root {shlex.quote(str(BUS_ROOT))} "
        "--run_name v84_regsign_s0 --seed 0 --plot_freq 10 "
        "--weight_reg 0.01 --beta 0 --penalty_scale 0 "
        "&& echo DONE"
    )
    command = [
        sys.executable, str(SCHEDULER), "submit",
        "--project", "BAPR-BUS",
        "--description", "BAPR v84 legacy bus regularization sign smoke",
        "--cmd", command_text,
        "--cwd", str(ROOT),
        "--signature", signature,
        "--vram", "2800",
        "--ram-mb", "6144",
        "--cpu", "3",
        "--priority", args.priority,
        "--ckpt-dir", str(BUS_ROOT / "model" / "v84_regsign_s0"),
        "--result-dir", str(BUS_ROOT),
        "--local-result-dir", str(BUS_ROOT),
        "--reroute-on-node-down",
        "--node-down-requeue-s", "900",
    ]
    submit_spec(command, dry_run=args.dry_run)
    return signature


def queued_ids_for_signatures(signatures: set[str]) -> list[str]:
    if not signatures or not QUEUE.exists():
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
    parser.add_argument("--with-bus-smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

    if args.base_iters + args.teacher_iters >= args.max_iters:
        raise SystemExit("max-iters must leave at least one student iteration")
    for path in (SCHEDULER, JAX_PYTHON):
        if not path.exists():
            raise SystemExit(f"missing required path: {path}")
    if args.with_bus_smoke and not BUS_PYTHON.exists():
        raise SystemExit(f"missing bus Python: {BUS_PYTHON}")

    tasks = scheduler_tasks()
    active = {
        str(task.get("signature")) for task in tasks
        if task.get("signature")
        and task.get("status") not in {"failed", "cancelled", "forgotten"}
    }
    variants = [
        variant for variant in VARIANTS
        if not args.variant or variant.name in args.variant
    ]
    envs = args.env or list(ENVS)
    submitted_signatures = set()
    for variant in variants:
        for env in envs:
            signature = submit_training(variant, env, args, active)
            if signature:
                submitted_signatures.add(signature)
    if args.with_bus_smoke:
        signature = submit_bus_smoke(args, active)
        if signature:
            submitted_signatures.add(signature)

    if args.dispatch and not args.dry_run and submitted_signatures:
        task_ids = queued_ids_for_signatures(submitted_signatures)
        if not task_ids:
            print("No queued v84 tasks need dispatch.")
            return
        command = [
            sys.executable, str(SCHEDULER), "dispatch",
            "--intent-label", "bapr-v84-seed0-screen",
            "--intent-ttl", "600",
        ]
        for task_id in task_ids:
            command += ["--task-id", task_id]
        subprocess.run(command, check=True, env=scheduler_env())


if __name__ == "__main__":
    main()
