#!/usr/bin/env python3
"""Continue promoted burst-torque teachers with learned context only."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import submit_bapr_v85_staged_capacity as common


ROOT = common.ROOT
SCHEDULER = common.SCHEDULER
JAX_PYTHON = common.JAX_PYTHON
SAVE_ROOT = ROOT / "jax_experiments" / "results_bapr_v3_oracle_headroom"
ENVS = ("Ant-v2", "HalfCheetah-v2")


def run_name(env: str) -> str:
    return f"oracle_v1_burst_torque_{env.removesuffix('-v2')}_s0"


def training_values(env: str, args: argparse.Namespace) -> list[str]:
    name = run_name(env)
    return [
        "--algo", "bapr_v3",
        "--env", env,
        "--seed", "0",
        "--env_type", "stochastic_mode",
        "--stochastic_mode_family", "burst_torque",
        "--stochastic_mode_dwell_steps", str(args.dwell_steps),
        "--stochastic_mode_dwell_distribution", "fixed",
        "--task_num", "4",
        "--test_task_num", "4",
        "--ensemble_size", "5",
        "--hidden_dim", "256",
        "--context_warmup_iters", "0",
        "--bapr_v2_mode", "supervised",
        "--bapr_v2_latent_dim", "4",
        "--bapr_v2_policy_context_source", "oracle_task",
        "--bapr_v2_training_schedule", "teacher_student",
        "--bapr_v2_base_pretrain_iters", str(args.base_iters),
        "--bapr_v2_teacher_iters", str(args.teacher_iters),
        "--bapr_v2_student_iters", "0",
        "--bapr_v2_context_hidden_dim", "64",
        "--bapr_v2_context_length", "64",
        "--bapr_v2_context_chunks", "16",
        "--bapr_v2_context_burnin", "16",
        "--bapr_v2_min_history", "16",
        "--bapr_v2_context_lr", "0.0003",
        "--bapr_v2_predictive_weight", "1.0",
        "--bapr_v2_supervised_weight", "1.0",
        "--bapr_v2_temporal_weight", "0.01",
        "--bapr_v2_reward_scale", "10.0",
        "--bapr_v2_delta_scale", "1.0",
        "--bapr_v2_policy_mode", "direct",
        "--bapr_v2_policy_gate_init", "6.0",
        "--bapr_v2_switch_rollout_steps", str(args.dwell_steps),
        "--bapr_v2_paired_calibration_episodes", "0",
        "--bapr_v2_context_dropout", "0.0",
        "--bapr_v2_base_aux_weight", "0.0",
        "--bapr_v2_actor_objective", "mean",
        "--bapr_v2_beta_ood", "0.0",
        "--bapr_v2_reg_weight", "0.0",
        "--bapr_v2_warmstart_conditioned",
        "--bapr_v2_freeze_gate_in_teacher",
        "--bapr_v3_likelihood", "probabilistic",
        "--bapr_v3_variance_model", "mode_shared_empirical",
        "--bapr_v3_context_ensemble_size", "5",
        "--bapr_v3_hazard_rate", "0.002",
        "--bapr_v3_evidence_scale", "1.0",
        "--bapr_v3_evidence_clip", "4.0",
        "--bapr_v3_fixed_variance", "0.02",
        "--bapr_v3_variance_floor", "0.0001",
        "--bapr_v3_variance_ceiling", "0.5",
        "--bapr_v3_variance_ema", "0.05",
        "--bapr_v3_mean_loss_weight", "1.0",
        "--bapr_v3_variance_loss_weight", "0.0",
        "--bapr_v3_variance_prior_weight", "0.0",
        "--bapr_v3_instant_classifier_weight", "0.0",
        "--bapr_v3_surprise_threshold", "2.0",
        "--bapr_v3_surprise_scale", "1.0",
        "--bapr_v3_freeze_teacher_after_teacher",
        "--bapr_v3_reset_context_on_resume",
        "--bapr_v3_eval_context_ladder",
        "--max_iters", str(args.max_iters),
        "--samples_per_iter", "4000",
        "--updates_per_iter", "250",
        "--log_interval", "20",
        "--eval_episodes", "3",
        "--eval_protocol", "full",
        "--eval_switching_episodes", "2",
        "--eval_switching_period_steps", str(args.dwell_steps),
        "--save_interval", "50",
        "--save_root", str(SAVE_ROOT),
        "--run_name", name,
        "--backend", "spring",
        "--resume",
    ]


def task_spec(env: str, args: argparse.Namespace) -> dict:
    name = run_name(env)
    run_dir = SAVE_ROOT / name
    command = (
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.20 "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 "
        f"{shlex.quote(str(JAX_PYTHON))} -u -m jax_experiments.train "
        f"{shlex.join(training_values(env, args))}"
    )
    return {
        "description": f"BAPR-v3 burst shared-empirical student {name}",
        "cmd": command,
        "cwd": str(ROOT),
        "signature": (
            f"BAPR/v3-burst-student/shared-empirical/v1/{env}/seed0"),
        "project": "BAPR",
        "vram": 2600,
        "ram_mb": 4096,
        "cpu": 2,
        "priority": args.priority,
        # Reuse the promoted teacher path so scheduler can locate and migrate
        # the exact iter-1399 checkpoint before the learned-context continuation.
        "ckpt_dir": str(run_dir / "checkpoints"),
        "result_dir": str(run_dir / "logs"),
        "local_result_dir": str(run_dir / "logs"),
        "stage_excludes": [
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "allow_remote_large_data": True,
        "allow_initial_resume_scan_error": False,
        "reroute_on_node_down": True,
        "node_down_requeue_s": 900,
    }


def submit_jsonl(specs: list[dict]) -> list[str]:
    payload = "".join(json.dumps(spec) + "\n" for spec in specs)
    command = [
        sys.executable, str(SCHEDULER), "submit-jsonl",
        "--stdin", "--trusted", "--json",
        "--intent-label", "bapr-v3-burst-student-submit",
        "--intent-ttl", "900",
    ]
    result = subprocess.run(
        command, input=payload, text=True, capture_output=True,
        env=common.scheduler_env())
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
        "--intent-label", "bapr-v3-burst-student-dispatch",
        "--intent-ttl", "900",
    ]
    for task_id in task_ids:
        command += ["--task-id", task_id]
    subprocess.run(command, check=True, env=common.scheduler_env())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-iters", type=int, default=2000)
    parser.add_argument("--base-iters", type=int, default=700)
    parser.add_argument("--teacher-iters", type=int, default=700)
    parser.add_argument("--dwell-steps", type=int, default=500)
    parser.add_argument("--env", action="append", choices=ENVS)
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="high")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

    teacher_end = args.base_iters + args.teacher_iters
    if args.max_iters <= teacher_end:
        raise SystemExit("max-iters must include a learned-context student stage")
    for path in (SCHEDULER, JAX_PYTHON):
        if not Path(path).exists():
            raise SystemExit(f"missing required path: {path}")

    envs = args.env or list(ENVS)
    active = {
        str(task.get("signature")) for task in common.scheduler_tasks()
        if task.get("signature")
        and task.get("status") in {"queued", "launching", "running"}
    }
    specs = []
    skipped = []
    for env in envs:
        spec = task_spec(env, args)
        run_dir = Path(spec["ckpt_dir"]).parent
        iteration = common.last_iteration(run_dir)
        if iteration >= args.max_iters - 1:
            skipped.append((spec["signature"], "complete"))
        elif iteration < teacher_end - 1:
            raise SystemExit(
                f"{env} teacher checkpoint incomplete: iter={iteration}, "
                f"need >= {teacher_end - 1}")
        elif spec["signature"] in active:
            skipped.append((spec["signature"], "active"))
        else:
            specs.append(spec)

    print(
        f"BAPR-v3 burst student: expected={len(envs)} "
        f"submit={len(specs)} skip={len(skipped)}", flush=True)
    for spec in specs:
        print(f"  {spec['signature']}", flush=True)
    for signature, reason in skipped:
        print(f"  skip {reason}: {signature}", flush=True)
    if args.dry_run or not specs:
        return

    task_ids = submit_jsonl(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        dispatch(task_ids)


if __name__ == "__main__":
    main()
