#!/usr/bin/env python3
"""Submit the 36-task BAPR-v2 seed-0 mechanism screen via scheduler."""
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
SAVE_ROOT = ROOT / "jax_experiments" / "results_bapr_v2_phase1"
QUEUE = Path.home() / ".claude" / "scheduler" / "queue.json"

ENVS = ["Ant-v2", "HalfCheetah-v2", "Hopper-v2", "Walker2d-v2"]
TERMINAL = {"done", "failed", "cancelled", "forgotten"}


@dataclass(frozen=True)
class Variant:
    name: str
    flags: tuple[str, ...]


VARIANTS = [
    Variant("v82a_robust_mean", (
        "--bapr_v2_mode", "robust",
        "--bapr_v2_actor_objective", "mean")),
    Variant("v82b_robust_lcb", (
        "--bapr_v2_mode", "robust",
        "--bapr_v2_actor_objective", "lcb", "--beta", "-1.0")),
    Variant("v82c_oracle_r025", (
        "--bapr_v2_mode", "oracle",
        "--bapr_v2_residual_delta", "0.25",
        "--bapr_v2_context_dropout", "0.0")),
    Variant("v82d_oracle_r100", (
        "--bapr_v2_mode", "oracle",
        "--bapr_v2_residual_delta", "1.0",
        "--bapr_v2_context_dropout", "0.0")),
    Variant("v82e_supervised_fb", (
        "--bapr_v2_mode", "supervised",
        "--bapr_v2_residual_delta", "0.25")),
    Variant("v82f_supervised_nofb", (
        "--bapr_v2_mode", "supervised",
        "--bapr_v2_residual_delta", "0.25",
        "--no_bapr_v2_fallback")),
    Variant("v82g_hybrid_fb", (
        "--bapr_v2_mode", "hybrid",
        "--bapr_v2_residual_delta", "0.25")),
    Variant("v82h_hybrid_nofb", (
        "--bapr_v2_mode", "hybrid",
        "--bapr_v2_residual_delta", "0.25",
        "--no_bapr_v2_fallback")),
    Variant("v82i_supervised_h128", (
        "--bapr_v2_mode", "supervised",
        "--bapr_v2_context_length", "128",
        "--bapr_v2_context_burnin", "32",
        "--bapr_v2_min_history", "64",
        "--bapr_v2_residual_delta", "0.25")),
]


def scheduler_env() -> dict[str, str]:
    env = os.environ.copy()
    skill_path = str(SCHEDULER.parent)
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        skill_path if not current else f"{skill_path}{os.pathsep}{current}")
    return env


def env_short(env: str) -> str:
    return env.replace("-v2", "")


def active_signatures() -> set[str]:
    if not QUEUE.exists():
        return set()
    try:
        state = json.loads(QUEUE.read_text())
    except Exception:
        return set()
    return {
        task["signature"]
        for task in state.get("tasks", [])
        if task.get("signature")
        and task.get("status") not in TERMINAL
    }


def complete(run_name: str, max_iters: int) -> bool:
    iteration_path = SAVE_ROOT / run_name / "logs" / "iteration.npy"
    if not iteration_path.exists():
        return False
    try:
        import numpy as np

        values = np.load(iteration_path)
        return bool(len(values) and int(values[-1]) >= max_iters - 1)
    except Exception:
        return False


def train_args(variant: Variant, env: str, max_iters: int,
               run_name: str) -> str:
    common = [
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
        "--bapr_v2_latent_dim", "4",
        "--bapr_v2_context_hidden_dim", "64",
        "--bapr_v2_context_length", "64",
        "--bapr_v2_context_chunks", "8",
        "--bapr_v2_context_burnin", "16",
        "--bapr_v2_context_lr", "0.0003",
        "--bapr_v2_predictive_weight", "1.0",
        "--bapr_v2_supervised_weight", "2.0",
        "--bapr_v2_hybrid_supervised_weight", "0.2",
        "--bapr_v2_temporal_weight", "0.01",
        "--bapr_v2_min_history", "32",
        "--bapr_v2_context_dropout", "0.2",
        "--bapr_v2_base_aux_weight", "0.25",
        "--bapr_v2_beta_ood", "0.0",
        "--bapr_v2_reg_weight", "0.0",
        "--max_iters", str(max_iters),
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
    return shlex.join(common + list(variant.flags))


def submit(variant: Variant, env: str, args, active: set[str]) -> None:
    run_name = f"{variant.name}_{env_short(env)}_s0"
    signature = f"BAPR/v82-phase1/{variant.name}/{env}/seed0"
    if not args.no_skip_complete and complete(run_name, args.max_iters):
        print(f"skip complete {run_name}")
        return
    if not args.no_skip_active and signature in active:
        print(f"skip active {signature}")
        return

    result_dir = SAVE_ROOT / run_name
    command_text = (
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.24 "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        f"{shlex.quote(str(JAX_PYTHON))} -u -m jax_experiments.train "
        f"{train_args(variant, env, args.max_iters, run_name)}"
    )
    command = [
        sys.executable, str(SCHEDULER), "submit",
        "--project", "BAPR",
        "--description", f"BAPR-v2 phase1 {run_name}",
        "--cmd", command_text,
        "--cwd", str(ROOT),
        "--signature", signature,
        "--vram", "2600",
        "--ram-mb", "4096",
        "--cpu", "2",
        "--priority", args.priority,
        "--ckpt-dir", str(result_dir / "checkpoints"),
        "--result-dir", str(result_dir / "logs"),
        "--local-result-dir", str(result_dir / "logs"),
        "--allow-remote-large-data",
        "--reroute-on-node-down",
        "--node-down-requeue-s", "900",
    ]
    if args.allow_duplicate:
        command.append("--allow-duplicate")
    if args.allow_initial_resume_scan_error:
        command.append("--allow-initial-resume-scan-error")
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    result = subprocess.run(
        command, text=True, capture_output=True, env=scheduler_env())
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        lowered = output.lower()
        if "duplicate" in lowered or "already queued" in lowered:
            print(output.strip())
            return
        print(output, file=sys.stderr)
        result.check_returncode()
    if result.stdout:
        print(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-iters", type=int, default=600)
    parser.add_argument(
        "--priority", choices=["low", "normal", "high"], default="high")
    parser.add_argument("--variant", action="append",
                        choices=[variant.name for variant in VARIANTS])
    parser.add_argument("--env", action="append", choices=ENVS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument(
        "--allow-duplicate", action="store_true",
        help="Submit even when a terminal task already has the same signature.")
    parser.add_argument(
        "--allow-initial-resume-scan-error", action="store_true",
        help=(
            "Allow dispatch after a failed initial checkpoint scan. Use only "
            "after manually verifying every unscanned node."))
    parser.add_argument("--no-skip-complete", action="store_true")
    parser.add_argument("--no-skip-active", action="store_true")
    args = parser.parse_args()

    if not JAX_PYTHON.exists():
        raise SystemExit(f"missing Python environment: {JAX_PYTHON}")
    if not SCHEDULER.exists():
        raise SystemExit(f"missing scheduler: {SCHEDULER}")
    variants = [
        variant for variant in VARIANTS
        if args.variant is None or variant.name in args.variant]
    envs = args.env or ENVS
    active = active_signatures()
    for variant in variants:
        for env in envs:
            submit(variant, env, args, active)

    if args.dispatch and not args.dry_run:
        subprocess.run(
            [sys.executable, str(SCHEDULER), "dispatch"],
            check=True, env=scheduler_env())


if __name__ == "__main__":
    main()
