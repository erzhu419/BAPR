#!/usr/bin/env python3
"""Submit the original unpaired BAPR-v3 budget screen for exploration only.

The two variants below are independent scheduler jobs.  They do not share an
iter-699 checkpoint or a single runtime, so their comparison is not a causal
budget-matched mechanism test.  Use ``submit_bapr_v3_budget_matched_fork.py``
for the corrected shared-checkpoint/common-restart protocol.
"""
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
SAVE_ROOT = ROOT / "jax_experiments" / "results_bapr_v3_budget_matched_v1"
ENVS = ("Ant-v2", "HalfCheetah-v2")
FAMILIES = ("deterministic_mean", "mean_variance")


@dataclass(frozen=True)
class Variant:
    name: str
    base_iters: int
    teacher_iters: int


VARIANTS = (
    # One context-free actor receives the full controller-update budget.
    Variant("robust_long", base_iters=1400, teacher_iters=0),
    # This independent job switches its own direct branch at iter 700; it is
    # not paired to robust_long's iter-699 checkpoint.
    Variant("oracle_direct", base_iters=700, teacher_iters=700),
)


def run_name(family: str, variant: Variant, env: str, seed: int) -> str:
    short_env = env.removesuffix("-v2")
    return f"budget_v1_{family}_{variant.name}_{short_env}_s{seed}"


def training_values(
    family: str,
    variant: Variant,
    env: str,
    seed: int,
    args: argparse.Namespace,
    name: str,
) -> list[str]:
    return [
        "--algo", "bapr_v3",
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
        "--context_warmup_iters", "0",
        "--bapr_v2_mode", "oracle",
        "--bapr_v2_latent_dim", "4",
        "--bapr_v2_policy_context_source", "oracle_task",
        "--bapr_v2_training_schedule", "teacher_student",
        "--bapr_v2_base_pretrain_iters", str(variant.base_iters),
        "--bapr_v2_teacher_iters", str(variant.teacher_iters),
        "--bapr_v2_student_iters", "0",
        "--bapr_v2_context_hidden_dim", "64",
        "--bapr_v2_context_length", "64",
        "--bapr_v2_context_chunks", "8",
        "--bapr_v2_context_burnin", "16",
        "--bapr_v2_min_history", "16",
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
        "--bapr_v3_likelihood", "point",
        "--bapr_v3_variance_model", "mode_calibrated",
        "--bapr_v3_context_ensemble_size", "2",
        "--bapr_v3_instant_classifier_weight", "0.0",
        "--bapr_v3_freeze_teacher_after_teacher",
        "--max_iters", str(args.max_iters),
        "--samples_per_iter", "4000",
        "--updates_per_iter", "250",
        "--log_interval", str(args.log_interval),
        "--eval_episodes", "2",
        "--eval_protocol", "stationary",
        "--save_interval", "50",
        "--save_root", str(SAVE_ROOT),
        "--run_name", name,
        "--backend", "spring",
        "--resume",
    ]


def task_spec(
    family: str,
    variant: Variant,
    env: str,
    seed: int,
    args: argparse.Namespace,
) -> dict:
    name = run_name(family, variant, env, seed)
    run_dir = SAVE_ROOT / name
    command = (
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.24 "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 JAX_NUM_THREADS=1 "
        "TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 "
        f"{shlex.quote(str(JAX_PYTHON))} -u -m jax_experiments.train "
        f"{shlex.join(training_values(family, variant, env, seed, args, name))}"
    )
    return {
        "description": f"BAPR-v3 budget-matched mechanism screen {name}",
        "cmd": command,
        "cwd": str(ROOT),
        "signature": (
            f"BAPR/v3-budget-match/v1/{family}/{variant.name}/{env}/seed{seed}"),
        "project": "BAPR",
        "vram": 3000,
        "ram_mb": 4096,
        "cpu": 2,
        "priority": args.priority,
        "ckpt_dir": str(run_dir / "checkpoints"),
        "result_dir": str(run_dir / "logs"),
        "local_result_dir": str(run_dir / "logs"),
        "stage_excludes": ["jax_experiments/eval_bundles*/", "paper/"],
        "allow_remote_large_data": True,
        "allow_initial_resume_scan_error": True,
        "reroute_on_node_down": True,
        "node_down_requeue_s": 900,
    }


def submit_jsonl(specs: list[dict]) -> list[str]:
    payload = "".join(json.dumps(spec) + "\n" for spec in specs)
    command = [
        sys.executable, str(SCHEDULER), "submit-jsonl",
        "--stdin", "--trusted", "--json",
        "--intent-label", "bapr-v3-budget-match-submit",
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
        "--intent-label", "bapr-v3-budget-match-dispatch",
        "--intent-ttl", "900",
    ]
    for task_id in task_ids:
        command += ["--task-id", task_id]
    subprocess.run(command, check=True, env=common.scheduler_env())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-iters", type=int, default=1400)
    parser.add_argument("--dwell-steps", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--family", action="append", choices=FAMILIES)
    parser.add_argument("--variant", action="append",
                        choices=tuple(variant.name for variant in VARIANTS))
    parser.add_argument("--env", action="append", choices=ENVS)
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="high")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument(
        "--allow-unpaired-exploratory", action="store_true",
        help="Acknowledge that independent jobs are exploratory and cannot "
             "support a causal controller-budget conclusion.")
    args = parser.parse_args()

    if args.max_iters != 1400:
        raise SystemExit("v1 preregistration fixes the total budget at 1400 iterations")
    for path in (SCHEDULER, JAX_PYTHON):
        if not Path(path).exists():
            raise SystemExit(f"missing required path: {path}")

    families = args.family or list(FAMILIES)
    envs = args.env or list(ENVS)
    seeds = args.seed or [0]
    requested_variants = set(args.variant or [v.name for v in VARIANTS])
    variants = [v for v in VARIANTS if v.name in requested_variants]
    expected = len(families) * len(variants) * len(envs) * len(seeds)
    if expected > 8:
        raise SystemExit(f"budget-matched seed0 screen expands to {expected}; cap is 8")

    active = {
        str(task.get("signature")) for task in common.scheduler_tasks()
        if task.get("signature")
        and task.get("status") in {"queued", "launching", "running"}
    }
    specs = []
    skipped = []
    for family in families:
        for variant in variants:
            for env in envs:
                for seed in seeds:
                    spec = task_spec(family, variant, env, seed, args)
                    run_dir = Path(spec["ckpt_dir"]).parent
                    if common.last_iteration(run_dir) >= args.max_iters - 1:
                        skipped.append((spec["signature"], "complete"))
                    elif spec["signature"] in active:
                        skipped.append((spec["signature"], "active"))
                    else:
                        specs.append(spec)

    print(
        f"BAPR-v3 unpaired exploratory screen: expected={expected} "
        f"submit={len(specs)} skip={len(skipped)}", flush=True)
    for spec in specs:
        print(f"  {spec['signature']}", flush=True)
    for signature, reason in skipped:
        print(f"  skip {reason}: {signature}", flush=True)
    if args.dry_run or not specs:
        return
    if not args.allow_unpaired_exploratory:
        raise SystemExit(
            "refusing to submit independent arms as a causal budget-matched "
            "screen; use submit_bapr_v3_budget_matched_fork.py, or pass "
            "--allow-unpaired-exploratory only for descriptive runs")

    task_ids = submit_jsonl(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        dispatch(task_ids)


if __name__ == "__main__":
    main()
