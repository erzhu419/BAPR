#!/usr/bin/env python3
"""Submit v81 context-representation probes through scheduler.

The current ESCP/BAPR mild-continuous checkpoints show collapsed context
embeddings.  This small seed-0 probe tests whether fixing the RMDM task cap and
strengthening diversity is enough before spending full 2000-iter runs.
"""
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
SAVE_ROOT = ROOT / "jax_experiments" / "results_v81_context"
QUEUE = Path.home() / ".claude" / "scheduler" / "queue.json"

ENVS = ["Ant-v2", "HalfCheetah-v2", "Hopper-v2", "Walker2d-v2"]
ALGOS = ["escp", "bapr"]
TERMINAL = {"done", "failed", "cancelled", "forgotten"}


@dataclass(frozen=True)
class Variant:
    name: str
    diversity: float
    consistency: float
    rbf_radius: float = 2.0


VARIANTS = [
    Variant("v81a_rmdm64_div025", diversity=0.25, consistency=50.0),
    Variant("v81b_rmdm64_div1_cons5", diversity=1.0, consistency=5.0),
]


def env_short(env: str) -> str:
    return env.replace("-v2", "")


def env_vram_mb(env: str) -> int:
    if env == "Ant-v2":
        return 3800
    if env == "HalfCheetah-v2":
        return 3200
    return 2800


def scheduler_env() -> dict[str, str]:
    env = os.environ.copy()
    skill_path = str(SCHEDULER.parent)
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = skill_path if not current else f"{skill_path}{os.pathsep}{current}"
    return env


def active_signatures() -> set[str]:
    if not QUEUE.exists():
        return set()
    try:
        state = json.loads(QUEUE.read_text())
    except Exception:
        return set()
    out = set()
    for task in state.get("tasks", []):
        sig = task.get("signature")
        if sig and task.get("status") not in TERMINAL:
            out.add(sig)
    return out


def complete(run_name: str, max_iters: int) -> bool:
    path = SAVE_ROOT / run_name / "logs" / "iteration.npy"
    if not path.exists():
        return False
    try:
        import numpy as np

        values = np.load(path)
        return bool(len(values) and int(values[-1]) >= max_iters - 1)
    except Exception:
        return False


def bapr_v45_args() -> str:
    return (
        "--algo bapr --bapr_adaptation_mode gate --actor_objective mean "
        "--penalty_scale 0.5 --critic_target_mode independent "
        "--weight_reg 0.003 --beta_ood 0.003 "
        "--bapr_recent_frac_cap 0.15 --bapr_recent_frac_floor 0.03 "
        "--bapr_recent_disagreement_gate "
        "--bapr_recent_qstd_threshold 5.0 "
        "--bapr_recent_qstd_ratio_threshold 0.02 "
        "--bapr_reg_disagreement_gate --bapr_reg_warmup_iters 0 "
        "--bapr_reg_max_iters 70 --bapr_reg_latch --bapr_reg_require_both "
        "--bapr_reg_qstd_threshold 5.0 "
        "--bapr_reg_qstd_ratio_threshold 0.02 "
        "--bapr_reg_emergency_gate --bapr_reg_emergency_scale 0.2 "
        "--bapr_reg_emergency_qstd_threshold 8.0 "
        "--bapr_reg_emergency_qstd_ratio_threshold 0.025 "
        "--no_belief_conditioning"
    )


def algo_args(algo: str) -> str:
    if algo == "bapr":
        return bapr_v45_args()
    return f"--algo {algo}"


def train_args(variant: Variant, algo: str, env: str, max_iters: int,
               run_name: str) -> str:
    return (
        f"{algo_args(algo)} --env {env} --seed 0 "
        "--env_type continuous --task_num 40 --test_task_num 40 "
        "--changing_period 20000 --varying_params gravity "
        "--log_scale_limit 3.0 --task_scale_distribution pow1p5 "
        f"--rmdm_max_tasks 64 --diversity_loss_weight {variant.diversity} "
        f"--consistency_loss_weight {variant.consistency} "
        f"--rbf_radius {variant.rbf_radius} "
        f"--max_iters {max_iters} --samples_per_iter 4000 "
        "--updates_per_iter 250 --log_interval 10 --eval_episodes 5 "
        "--eval_protocol full --eval_switching_episodes 1 "
        "--eval_switching_period_steps 500 --save_interval 50 "
        f"--save_root {shlex.quote(str(SAVE_ROOT))} "
        f"--run_name {shlex.quote(run_name)} --backend spring --resume"
    )


def submit(variant: Variant, algo: str, env: str, args: argparse.Namespace) -> None:
    run_name = f"{variant.name}_{algo}_{env_short(env)}_s0"
    signature = f"BAPR/v81-context/{variant.name}/{algo}/{env}/seed0"
    if not args.no_skip_complete and complete(run_name, args.max_iters):
        print(f"skip complete {run_name}")
        return
    if not args.no_skip_active and signature in active_signatures():
        print(f"skip active {signature}")
        return

    result_dir = SAVE_ROOT / run_name
    cmd = (
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
        f"{shlex.quote(str(JAX_PYTHON))} -u -m jax_experiments.train "
        f"{train_args(variant, algo, env, args.max_iters, run_name)}"
    )
    command = [
        sys.executable, str(SCHEDULER), "submit",
        "--project", "BAPR",
        "--description", f"BAPR v81 context probe {run_name}",
        "--cmd", cmd,
        "--cwd", str(ROOT),
        "--signature", signature,
        "--vram", str(env_vram_mb(env)),
        "--ram-mb", "8192",
        "--cpu", "2",
        "--priority", args.priority,
        "--ckpt-dir", str(result_dir / "checkpoints"),
        "--result-dir", str(result_dir),
        "--local-result-dir", str(result_dir),
        "--allow-remote-large-data",
        "--reroute-on-node-down",
        "--node-down-requeue-s", "900",
    ]
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
    parser.add_argument("--max-iters", type=int, default=500)
    parser.add_argument("--priority", choices=["low", "normal", "high"], default="high")
    parser.add_argument("--variant", choices=[v.name for v in VARIANTS], action="append")
    parser.add_argument("--algo", choices=ALGOS, action="append")
    parser.add_argument("--env", choices=ENVS, action="append")
    parser.add_argument("--no-skip-complete", action="store_true")
    parser.add_argument("--no-skip-active", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

    if not JAX_PYTHON.exists():
        raise SystemExit(f"missing Python environment: {JAX_PYTHON}")
    if not SCHEDULER.exists():
        raise SystemExit(f"missing scheduler: {SCHEDULER}")

    variants = [v for v in VARIANTS if args.variant is None or v.name in args.variant]
    algos = args.algo or ALGOS
    envs = args.env or ENVS
    for variant in variants:
        for env in envs:
            for algo in algos:
                submit(variant, algo, env, args)

    if args.dispatch and not args.dry_run:
        subprocess.run(
            [sys.executable, str(SCHEDULER), "dispatch"],
            check=True, env=scheduler_env())


if __name__ == "__main__":
    main()
