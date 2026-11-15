#!/usr/bin/env python3
"""Submit the preregistered BAPR-v87 constrained-deployment screen."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import submit_bapr_v85_staged_capacity as common


ROOT = common.ROOT
SCHEDULER = common.SCHEDULER
JAX_PYTHON = common.JAX_PYTHON
QUEUE = common.QUEUE
SAVE_ROOT = ROOT / "jax_experiments" / "results_bapr_v87_constrained_deploy"
ENVS = ("Ant-v2", "HalfCheetah-v2", "Hopper-v2", "Walker2d-v2")
TASK_SEED_SALT = 870_000


@dataclass(frozen=True)
class Variant:
    name: str
    paired_episodes: int
    gate_supervision_weight: float
    unsafe_deviation_weight: float


VARIANTS = (
    Variant("v87a_switch_deploy", 0, 0.0, 0.0),
    Variant("v87b_paired_safe", 3, 2.0, 5.0),
    Variant("v87c_paired_strict", 3, 5.0, 20.0),
)


def training_args(variant: Variant, env: str, seed: int,
                  args, run_name: str) -> str:
    values = [
        "--algo", "bapr_v2",
        "--env", env,
        "--seed", str(seed),
        "--env_type", "continuous",
        "--task_num", "40",
        "--test_task_num", "40",
        "--reserved_test_task_num", "40",
        "--task_seed_salt", str(TASK_SEED_SALT),
        "--changing_period", "500",
        "--changing_interval", "500",
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
        "--bapr_v2_training_schedule", "constrained_deploy",
        "--bapr_v2_base_pretrain_iters", str(args.base_iters),
        "--bapr_v2_teacher_iters", str(args.teacher_iters),
        "--bapr_v2_student_iters", str(args.student_iters),
        "--bapr_v2_context_hidden_dim", "128",
        "--bapr_v2_context_length", "64",
        "--bapr_v2_context_chunks", "16",
        "--bapr_v2_context_burnin", "16",
        "--bapr_v2_min_history", "16",
        "--bapr_v2_context_lr", "0.0003",
        "--bapr_v2_predictive_weight", "0.25",
        "--bapr_v2_supervised_weight", "10.0",
        "--bapr_v2_temporal_weight", "0.01",
        "--bapr_v2_policy_mode", "gated_direct",
        "--bapr_v2_policy_gate_init", "6.0",
        "--bapr_v2_switch_rollout_steps", "500",
        "--bapr_v2_paired_calibration_episodes",
        str(variant.paired_episodes),
        "--bapr_v2_paired_gain_margin", "0.02",
        "--bapr_v2_paired_gain_temperature", "0.05",
        "--bapr_v2_paired_risk_tolerance", "0.0",
        "--bapr_v2_paired_risk_temperature", "0.10",
        "--bapr_v2_paired_return_scale", "100.0",
        "--bapr_v2_gate_supervision_weight",
        str(variant.gate_supervision_weight),
        "--bapr_v2_unsafe_deviation_weight",
        str(variant.unsafe_deviation_weight),
        "--bapr_v2_context_dropout", "0.0",
        "--bapr_v2_base_aux_weight", "0.0",
        "--no_bapr_v2_fallback",
        "--bapr_v2_actor_objective", "mean",
        "--bapr_v2_beta_ood", "0.0",
        "--bapr_v2_reg_weight", "0.0",
        "--bapr_v2_warmstart_conditioned",
        "--bapr_v2_freeze_gate_in_teacher",
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


def submit_task(variant: Variant, env: str, seed: int, args,
                active: set[str]) -> str | None:
    env_short = env.removesuffix("-v2")
    run_name = f"{variant.name}_{env_short}_s{seed}"
    signature = f"BAPR/v87-constrained/{variant.name}/{env}/seed{seed}"
    run_dir = SAVE_ROOT / run_name
    if common.last_iteration(run_dir) >= args.max_iters - 1:
        print(f"skip complete {run_name}")
        return None
    if signature in active:
        print(f"skip active {signature}")
        return None

    command_text = (
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        f"{shlex.quote(str(JAX_PYTHON))} -u -m jax_experiments.train "
        f"{training_args(variant, env, seed, args, run_name)}"
    )
    command = [
        sys.executable, str(SCHEDULER), "submit",
        "--project", "BAPR",
        "--description", f"BAPR-v87 constrained deploy {run_name}",
        "--cmd", command_text,
        "--cwd", str(ROOT),
        "--signature", signature,
        "--vram", "3600",
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
        command, text=True, capture_output=True, env=common.scheduler_env())
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
    parser.add_argument("--max-iters", type=int, default=1400)
    parser.add_argument("--base-iters", type=int, default=600)
    parser.add_argument("--teacher-iters", type=int, default=400)
    parser.add_argument("--student-iters", type=int, default=200)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--variant", action="append",
                        choices=[variant.name for variant in VARIANTS])
    parser.add_argument("--env", action="append", choices=ENVS)
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="high")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

    stage_total = args.base_iters + args.teacher_iters + args.student_iters
    if stage_total >= args.max_iters:
        raise SystemExit("max-iters must leave at least one deployment iteration")
    for path in (SCHEDULER, JAX_PYTHON):
        if not path.exists():
            raise SystemExit(f"missing required path: {path}")

    active = {
        str(task.get("signature")) for task in common.scheduler_tasks()
        if task.get("signature")
        and task.get("status") not in {"failed", "cancelled", "forgotten"}
    }
    variants = [
        variant for variant in VARIANTS
        if not args.variant or variant.name in args.variant]
    envs = args.env or list(ENVS)
    seeds = args.seed or [0]
    submitted = set()
    for variant in variants:
        for env in envs:
            for seed in seeds:
                signature = submit_task(variant, env, seed, args, active)
                if signature:
                    submitted.add(signature)

    if args.dispatch and not args.dry_run:
        task_ids = queued_ids(submitted)
        if not task_ids:
            print("No queued v87 tasks need dispatch.")
            return
        command = [
            sys.executable, str(SCHEDULER), "dispatch",
            "--intent-label", "bapr-v87-preregistered-screen",
            "--intent-ttl", "600",
        ]
        for task_id in task_ids:
            command += ["--task-id", task_id]
        subprocess.run(command, check=True, env=common.scheduler_env())


if __name__ == "__main__":
    main()
