#!/usr/bin/env python3
"""Submit the preregistered BAPR-v3 stochastic-regime seed0 screen."""
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
SAVE_ROOT = ROOT / "jax_experiments" / "results_bapr_v3_stochastic_screen"
ENVS = ("Ant-v2", "HalfCheetah-v2")
FAMILIES = ("deterministic_mean", "variance_only", "mean_variance")


@dataclass(frozen=True)
class Method:
    name: str
    algo: str
    vram_mb: int
    context_mode: str | None = None
    likelihood: str | None = None


METHODS = (
    Method("sac_e5", "sac", 2200),
    Method("resac_reg001", "resac", 2400),
    Method("escp_e4", "escp", 2400),
    Method("v3_point", "bapr_v3", 2600,
           context_mode="supervised", likelihood="point"),
    Method("v3_prob", "bapr_v3", 2600,
           context_mode="supervised", likelihood="probabilistic"),
    Method("v3_oracle", "bapr_v3", 2600,
           context_mode="oracle", likelihood="probabilistic"),
)


def training_values(method: Method, family: str, env: str, seed: int,
                    args: argparse.Namespace, run_name: str) -> list[str]:
    values = [
        "--algo", method.algo,
        "--env", env,
        "--seed", str(seed),
        "--env_type", "stochastic_mode",
        "--stochastic_mode_family", family,
        "--stochastic_mode_dwell_steps", str(args.dwell_steps),
        "--stochastic_mode_dwell_distribution", "fixed",
        "--task_num", "4",
        "--test_task_num", "4",
        "--ensemble_size", "5",
        "--hidden_dim", "256",
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
        "--run_name", run_name,
        "--backend", "spring",
        "--resume",
    ]

    if method.algo == "resac":
        values += [
            "--beta", "-2.0",
            "--weight_reg", "0.01",
            "--beta_ood", "0.01",
        ]
    elif method.algo == "escp":
        values += [
            "--beta", "-2.0",
            "--ep_dim", "4",
            "--rmdm_max_tasks", "4",
            "--context_warmup_iters", "50",
        ]
    elif method.algo == "bapr_v3":
        values += [
            "--context_warmup_iters", "0",
            "--bapr_v2_mode", str(method.context_mode),
            "--bapr_v2_latent_dim", "4",
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
            "--bapr_v2_predictive_weight", "1.0",
            "--bapr_v2_supervised_weight", "1.0",
            "--bapr_v2_temporal_weight", "0.01",
            "--bapr_v2_reward_scale", "10.0",
            "--bapr_v2_delta_scale", "1.0",
            "--bapr_v2_policy_mode", "gated_direct",
            "--bapr_v2_policy_gate_init", "6.0",
            "--bapr_v2_switch_rollout_steps", str(args.dwell_steps),
            "--bapr_v2_paired_calibration_episodes", "3",
            "--bapr_v2_paired_gain_margin", "0.02",
            "--bapr_v2_paired_gain_temperature", "0.05",
            "--bapr_v2_paired_risk_tolerance", "0.0",
            "--bapr_v2_paired_risk_temperature", "0.10",
            "--bapr_v2_paired_return_scale", "100.0",
            "--bapr_v2_gate_supervision_weight", "2.0",
            "--bapr_v2_unsafe_deviation_weight", "5.0",
            "--bapr_v2_context_dropout", "0.0",
            "--bapr_v2_base_aux_weight", "0.0",
            "--bapr_v2_actor_objective", "mean",
            "--bapr_v2_beta_ood", "0.0",
            "--bapr_v2_reg_weight", "0.0",
            "--bapr_v2_advantage_gate",
            "--bapr_v2_advantage_margin", "0.0",
            "--bapr_v2_advantage_lcb_scale", "1.0",
            "--bapr_v2_warmstart_conditioned",
            "--bapr_v2_freeze_gate_in_teacher",
            "--bapr_v3_likelihood", str(method.likelihood),
            "--bapr_v3_context_ensemble_size", "5",
            "--bapr_v3_hazard_rate", "0.002",
            "--bapr_v3_evidence_scale", "4.0",
            "--bapr_v3_fixed_variance", "0.02",
            "--bapr_v3_logvar_min", "-6.0",
            "--bapr_v3_logvar_max", "1.0",
            "--bapr_v3_surprise_threshold", "2.0",
            "--bapr_v3_surprise_scale", "1.0",
        ]
    return values


def task_spec(method: Method, family: str, env: str, seed: int,
              args: argparse.Namespace) -> dict:
    env_short = env.removesuffix("-v2")
    run_name = f"{family}_{method.name}_{env_short}_s{seed}"
    run_dir = SAVE_ROOT / run_name
    command = (
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.22 "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 "
        f"{shlex.quote(str(JAX_PYTHON))} -u -m jax_experiments.train "
        f"{shlex.join(training_values(method, family, env, seed, args, run_name))}"
    )
    return {
        "description": f"BAPR-v3 stochastic screen {run_name}",
        "cmd": command,
        "cwd": str(ROOT),
        "signature": (
            f"BAPR/v3-stochastic-screen/{family}/{method.name}/"
            f"{env}/seed{seed}"
        ),
        "project": "BAPR",
        "vram": method.vram_mb,
        "ram_mb": 4096,
        "cpu": 2,
        "priority": args.priority,
        "ckpt_dir": str(run_dir / "checkpoints"),
        "result_dir": str(run_dir / "logs"),
        "local_result_dir": str(run_dir / "logs"),
        "stage_excludes": [
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "allow_remote_large_data": True,
        "reroute_on_node_down": True,
        "node_down_requeue_s": 900,
    }


def submit_jsonl(specs: list[dict], args: argparse.Namespace) -> list[str]:
    if not specs:
        return []
    payload = "".join(json.dumps(spec) + "\n" for spec in specs)
    command = [
        sys.executable, str(SCHEDULER), "submit-jsonl",
        "--stdin", "--trusted", "--json",
        "--intent-label", "bapr-v3-stochastic-screen-submit",
        "--intent-ttl", "900",
    ]
    result = subprocess.run(
        command, input=payload, text=True, capture_output=True,
        env=common.scheduler_env())
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        result.check_returncode()
    response = json.loads(result.stdout)
    task_ids = [str(item["id"]) for item in response.get("submitted", [])]
    print(json.dumps(response, indent=2))
    return task_ids


def dispatch(task_ids: list[str]) -> None:
    if not task_ids:
        return
    command = [
        sys.executable, str(SCHEDULER), "dispatch",
        "--bulk-window",
        "--intent-label", "bapr-v3-stochastic-screen-dispatch",
        "--intent-ttl", "900",
    ]
    for task_id in task_ids:
        command += ["--task-id", task_id]
    subprocess.run(command, check=True, env=common.scheduler_env())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-iters", type=int, default=1400)
    parser.add_argument("--base-iters", type=int, default=600)
    parser.add_argument("--teacher-iters", type=int, default=400)
    parser.add_argument("--student-iters", type=int, default=200)
    parser.add_argument("--dwell-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--method", action="append",
                        choices=[method.name for method in METHODS])
    parser.add_argument("--family", action="append", choices=FAMILIES)
    parser.add_argument("--env", action="append", choices=ENVS)
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="high")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

    stage_total = args.base_iters + args.teacher_iters + args.student_iters
    if stage_total >= args.max_iters:
        raise SystemExit(
            "max-iters must leave at least one learned deployment iteration")
    for path in (SCHEDULER, JAX_PYTHON):
        if not Path(path).exists():
            raise SystemExit(f"missing required path: {path}")

    methods = [
        method for method in METHODS
        if not args.method or method.name in args.method]
    families = args.family or list(FAMILIES)
    envs = args.env or list(ENVS)
    seeds = args.seed or [0]
    expected = len(methods) * len(families) * len(envs) * len(seeds)
    if expected > 36:
        raise SystemExit(
            f"screen expands to {expected} tasks; the preregistered cap is 36")

    active = {
        str(task.get("signature")) for task in common.scheduler_tasks()
        if task.get("signature")
        and task.get("status") in {"queued", "launching", "running"}
    }
    specs = []
    skipped = []
    for family in families:
        for method in methods:
            for env in envs:
                for seed in seeds:
                    spec = task_spec(method, family, env, seed, args)
                    run_dir = Path(spec["ckpt_dir"]).parent
                    if common.last_iteration(run_dir) >= args.max_iters - 1:
                        skipped.append((spec["signature"], "complete"))
                    elif spec["signature"] in active:
                        skipped.append((spec["signature"], "active"))
                    else:
                        specs.append(spec)

    print(
        f"BAPR-v3 matrix: expected={expected} submit={len(specs)} "
        f"skip={len(skipped)}", flush=True)
    for spec in specs:
        print(
            f"  {spec['signature']} vram={spec['vram']}MB", flush=True)
    for signature, reason in skipped:
        print(f"  skip {reason}: {signature}", flush=True)

    if args.dry_run:
        return
    task_ids = submit_jsonl(specs, args)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        dispatch(task_ids)


if __name__ == "__main__":
    main()
