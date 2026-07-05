#!/usr/bin/env python3
"""Submit the v78 BAPR soft-controller screening batch.

Round 1 is deliberately broad but bounded:
  - 6 soft-controller variants on Ant seeds 0-4.
  - 2 representative variants on HalfCheetah/Hopper/Walker2d seed 0 as
    cross-environment canaries.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = Path("/home/erzhu419/.claude/skills/scheduler/scheduler.py")
JAX_PYTHON = Path("/home/erzhu419/.conda/envs/resac-jax/bin/python")
SAVE_ROOT = ROOT / "jax_experiments" / "results_paper"

BASE_FLAGS = [
    "--algo", "bapr",
    "--bapr_adaptation_mode", "gate",
    "--actor_objective", "mean",
    "--penalty_scale", "0.5",
    "--critic_target_mode", "independent",
    "--weight_reg", "0.003",
    "--beta_ood", "0.003",
    "--bapr_gate_warmup_iters", "10",
    "--bapr_surprise_threshold", "0.1",
    "--bapr_gate_max", "0.35",
    "--bapr_recent_frac_cap", "0.15",
    "--bapr_recent_frac_floor", "0.03",
    "--bapr_recent_disagreement_gate",
    "--bapr_recent_qstd_threshold", "5.0",
    "--bapr_recent_qstd_ratio_threshold", "0.02",
    "--bapr_reg_disagreement_gate",
    "--bapr_reg_warmup_iters", "0",
    "--bapr_reg_max_iters", "70",
    "--bapr_reg_latch",
    "--bapr_reg_require_both",
    "--bapr_reg_qstd_threshold", "5.0",
    "--bapr_reg_qstd_ratio_threshold", "0.02",
    "--bapr_reg_emergency_gate",
    "--bapr_reg_emergency_scale", "0.2",
    "--bapr_reg_emergency_qstd_threshold", "8.0",
    "--bapr_reg_emergency_qstd_ratio_threshold", "0.025",
    "--bapr_reg_latched_scale", "1.0",
]

COMMON_TRAIN_FLAGS = [
    "--max_iters", "800",
    "--log_interval", "5",
    "--eval_episodes", "5",
    "--save_interval", "100",
    "--save_root", str(SAVE_ROOT),
    "--backend", "spring",
    "--env_type", "discrete_mode",
    "--mean_dwell_iters", "60",
    "--resume",
]

ENV_PREFIX = [
    "env",
    "PYTHONPATH=.",
    "JAX_PLATFORMS=cuda",
    "XLA_PYTHON_CLIENT_PREALLOCATE=false",
    "XLA_PYTHON_CLIENT_MEM_FRACTION=0.22",
    "OMP_NUM_THREADS=1",
    "OPENBLAS_NUM_THREADS=1",
    "MKL_NUM_THREADS=1",
    "NUMEXPR_NUM_THREADS=1",
]

STAGE_EXCLUDES = [
    ".git",
    "__pycache__",
    "bapr_experiments",
    "bus_experiments_paper_v2",
    "jax_experiments/archive_jtl110gpu_server",
    "jax_experiments/archive_jtl110gpu_server_old",
    "jax_experiments/results_paper",
    "jax_experiments/results_jmlr",
    "jax_experiments/results_ablation",
    "jax_experiments/results_ablation_v2",
    "paper",
]


@dataclass(frozen=True)
class Variant:
    name: str
    flags: list[str] = field(default_factory=list)


def soft_flags(
    *,
    drop: float,
    ramp: float,
    low_ratio: float,
    decay: float,
    reg: float,
    actor: float,
    recent_add: float = 0.0,
    recent_mult: float = 1.0,
    lcb: float = 1.0,
    weight_reg: float | None = None,
    actor_recover: int = 0,
    reg_recover: int = 0,
) -> list[str]:
    flags = [
        "--bapr_controller_mode", "soft_recovery",
        "--bapr_controller_warmup_iters", "100",
        "--bapr_controller_min_peak", "100",
        "--bapr_controller_drop_frac", str(drop),
        "--bapr_controller_drop_ramp", str(ramp),
        "--bapr_controller_soft_decay", str(decay),
        "--bapr_controller_soft_min_signal", "0.02",
        "--bapr_controller_low_qstd_ratio", str(low_ratio),
        "--bapr_controller_low_qstd_threshold", "0",
        "--bapr_controller_reg_multiplier", str(reg),
        "--bapr_controller_recent_multiplier", str(recent_mult),
        "--bapr_controller_recent_add_frac", str(recent_add),
        "--bapr_controller_lcb_multiplier", str(lcb),
        "--bapr_controller_actor_update_multiplier", str(actor),
    ]
    if actor_recover:
        flags += ["--bapr_controller_actor_recover_iters", str(actor_recover)]
    if reg_recover:
        flags += ["--bapr_controller_reg_recover_iters", str(reg_recover)]
    if weight_reg is not None:
        flags += ["--weight_reg", str(weight_reg)]
    return flags


VARIANTS = [
    Variant(
        "v78a_soft_mild",
        soft_flags(drop=0.45, ramp=0.55, low_ratio=0.02, decay=0.92,
                   reg=0.70, actor=0.70),
    ),
    Variant(
        "v78b_soft_regsafe",
        soft_flags(drop=0.45, ramp=0.55, low_ratio=0.02, decay=0.92,
                   reg=0.85, actor=0.70),
    ),
    Variant(
        "v78c_soft_actorlight",
        soft_flags(drop=0.45, ramp=0.55, low_ratio=0.02, decay=0.92,
                   reg=0.70, actor=0.85),
    ),
    Variant(
        "v78d_soft_recent02",
        soft_flags(drop=0.45, ramp=0.55, low_ratio=0.02, decay=0.92,
                   reg=0.70, actor=0.75, recent_add=0.02),
    ),
    Variant(
        "v78e_soft_broad",
        soft_flags(drop=0.50, ramp=0.60, low_ratio=0.0, decay=0.90,
                   reg=0.80, actor=0.80),
    ),
    Variant(
        "v78f_soft_wreg001",
        soft_flags(drop=0.45, ramp=0.55, low_ratio=0.02, decay=0.92,
                   reg=0.80, actor=0.75, weight_reg=0.01),
    ),
]


CANARY_VARIANTS = [VARIANTS[0], VARIANTS[2]]
ANT_SEEDS = [0, 1, 2, 3, 4]
CANARY_ENVS = ["HalfCheetah-v2", "Hopper-v2", "Walker2d-v2"]


def env_short(env: str) -> str:
    return env.replace("-v2", "")


def task_vram(env: str) -> int:
    return 1800 if env == "Ant-v2" else 1600


def build_cmd(variant: Variant, env: str, seed: int, run_name: str) -> str:
    args = (
        ENV_PREFIX
        + [str(JAX_PYTHON), "-u", "-m", "jax_experiments.train"]
        + COMMON_TRAIN_FLAGS
        + ["--env", env, "--seed", str(seed), "--run_name", run_name]
        + BASE_FLAGS
        + variant.flags
    )
    return shlex.join(args)


def build_jobs() -> list[tuple[Variant, str, int, str]]:
    jobs: list[tuple[Variant, str, int, str]] = []
    for variant in VARIANTS:
        for seed in ANT_SEEDS:
            env = "Ant-v2"
            run_name = f"{variant.name}_bapr_{env_short(env)}_dw60_s{seed}"
            jobs.append((variant, env, seed, run_name))
    for variant in CANARY_VARIANTS:
        for env in CANARY_ENVS:
            seed = 0
            run_name = f"{variant.name}_canary_bapr_{env_short(env)}_dw60_s{seed}"
            jobs.append((variant, env, seed, run_name))
    return jobs


def is_complete(run_name: str) -> bool:
    done = SAVE_ROOT / run_name / "done.ok"
    if done.exists():
        return True
    iterations = SAVE_ROOT / run_name / "logs" / "iteration.npy"
    if not iterations.exists():
        return False
    try:
        import numpy as np

        values = np.load(iterations)
        return bool(len(values) and int(values[-1]) >= 799)
    except Exception:
        return False


def submit_job(variant: Variant, env: str, seed: int, run_name: str,
               *, dry_run: bool, priority: str, allow_duplicate: bool) -> None:
    cmd = build_cmd(variant, env, seed, run_name)
    run_dir = SAVE_ROOT / run_name
    logs_dir = run_dir / "logs"
    ckpt_dir = run_dir / "checkpoints"
    sched_cmd = [
        "python3", str(SCHEDULER), "submit",
        "--description", f"BAPR v78 soft controller: {run_name}",
        "--cmd", cmd,
        "--cwd", str(ROOT),
        "--signature", f"BAPR/v78-soft/{variant.name}/{env}/seed{seed}",
        "--vram", str(task_vram(env)),
        "--ram-mb", "8192",
        "--cpu", "1",
        "--priority", priority,
        "--ckpt-dir", str(ckpt_dir),
        "--allow-initial-resume-scan-error",
        "--result-dir", str(logs_dir),
        "--local-result-dir", str(logs_dir),
        "--reroute-on-node-down",
        "--node-down-requeue-s", "300",
    ]
    for pattern in STAGE_EXCLUDES:
        sched_cmd += ["--stage-exclude", pattern]
    if allow_duplicate:
        sched_cmd.append("--allow-duplicate")

    print(shlex.join(sched_cmd))
    if dry_run:
        return
    subprocess.run(sched_cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-duplicate", action="store_true")
    parser.add_argument("--priority", choices=["low", "normal", "high"],
                        default="high")
    args = parser.parse_args()

    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    submitted = 0
    skipped = 0
    for variant, env, seed, run_name in build_jobs():
        if is_complete(run_name) and not args.allow_duplicate:
            print(f"SKIP complete: {run_name}")
            skipped += 1
            continue
        submit_job(variant, env, seed, run_name, dry_run=args.dry_run,
                   priority=args.priority,
                   allow_duplicate=args.allow_duplicate)
        submitted += 1
    print(f"submitted={submitted} skipped={skipped} dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
