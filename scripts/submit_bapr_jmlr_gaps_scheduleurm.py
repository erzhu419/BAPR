#!/usr/bin/env python3
"""Submit BAPR experiment gaps relative to the RE-SAC JMLR protocol.

The script deliberately emits one scheduler task per seed/run. Training jobs
request VRAM, so scheduleurm routes them only to GPU-capable nodes/local.
CPU-only evaluation jobs, when added, should use explicit --wait-for-file
dependencies on checkpoint files.
"""
from __future__ import annotations

import argparse
import fnmatch
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = Path("/home/erzhu419/mine_code/scheduleurm/skill/scheduler.py")
JAX_PYTHON = Path("/home/erzhu419/.conda/envs/resac-jax/bin/python")

JMLR_SEEDS = [8, 16, 24, 32, 40]
PAPER_SEEDS = [0, 1, 2, 3, 4]

ENVS = ["HalfCheetah-v2", "Hopper-v2", "Walker2d-v2", "Ant-v2"]
NEGATIVE_ENVS = ["Hopper-v2", "Walker2d-v2"]
ALGOS = ["bapr", "escp", "resac", "sac"]


def env_short(env: str) -> str:
    return env.replace("-v2", "")


def env_vram_mb(env: str) -> int:
    if env == "Ant-v2":
        return 3800
    if env == "HalfCheetah-v2":
        return 3200
    return 2800


@dataclass(frozen=True)
class Job:
    name: str
    signature: str
    args: str
    save_root: Path
    max_iters: int
    description: str
    vram: int
    ram: int = 8192
    cpu: int = 2
    xla_mem_fraction: str = "0.35"


def is_complete(save_root: Path, run_name: str, max_iters: int) -> bool:
    iter_file = save_root / run_name / "logs" / "iteration.npy"
    if not iter_file.exists():
        return False
    try:
        import numpy as np

        values = np.load(iter_file)
        return bool(len(values) and int(values[-1]) >= max_iters - 1)
    except Exception:
        return False


def bapr_recommended_args() -> str:
    return (
        "--algo bapr --bapr_adaptation_mode gate --actor_objective mean "
        "--penalty_scale 0.5 --critic_target_mode independent"
    )


def algo_args(algo: str) -> str:
    if algo == "bapr":
        return bapr_recommended_args()
    return f"--algo {algo}"


def common_train_args(max_iters: int, save_root: Path, run_name: str) -> str:
    return (
        f"--max_iters {max_iters} --samples_per_iter 4000 "
        "--updates_per_iter 250 --log_interval 5 --eval_episodes 5 "
        "--save_interval 50 --backend spring --resume "
        f"--save_root {shlex.quote(str(save_root))} "
        f"--run_name {shlex.quote(run_name)}"
    )


def discrete_args(algo: str, env: str, seed: int, max_iters: int,
                  save_root: Path, run_name: str) -> str:
    return (
        f"{algo_args(algo)} --env {env} --seed {seed} "
        "--env_type discrete_mode --mean_dwell_iters 60 "
        f"{common_train_args(max_iters, save_root, run_name)}"
    )


def continuous_args(algo: str, env: str, seed: int, max_iters: int,
                    save_root: Path, run_name: str,
                    varying_param: str = "gravity",
                    log_scale_limit: float = 3.0) -> str:
    return (
        f"{algo_args(algo)} --env {env} --seed {seed} "
        "--env_type continuous --task_num 40 --test_task_num 40 "
        "--changing_period 20000 "
        f"--varying_params {varying_param} --log_scale_limit {log_scale_limit} "
        f"{common_train_args(max_iters, save_root, run_name)}"
    )


def build_discrete_negative_fill(max_iters: int) -> list[Job]:
    save_root = ROOT / "jax_experiments" / "results_paper"
    jobs: list[Job] = []
    for env in NEGATIVE_ENVS:
        for seed in [3, 4]:
            for algo in ALGOS:
                name = f"paper_negative_v2_{algo}_{env_short(env)}_dw60_s{seed}"
                jobs.append(Job(
                    name=name,
                    signature=f"BAPR/paper-negative-v2/{algo}/{env}/seed{seed}",
                    args=discrete_args(algo, env, seed, max_iters, save_root, name),
                    save_root=save_root,
                    max_iters=max_iters,
                    description=f"BAPR discrete negative fill: {name}",
                    vram=env_vram_mb(env),
                ))
    return jobs


def build_continuous_main(max_iters: int, seeds: list[int]) -> list[Job]:
    save_root = ROOT / "jax_experiments" / "results_jmlr"
    jobs: list[Job] = []
    for env in ENVS:
        for seed in seeds:
            for algo in ALGOS:
                name = f"jmlr_continuous_gravity_{algo}_{env_short(env)}_s{seed}"
                jobs.append(Job(
                    name=name,
                    signature=f"BAPR/jmlr/continuous-gravity/{algo}/{env}/seed{seed}",
                    args=continuous_args(algo, env, seed, max_iters, save_root, name),
                    save_root=save_root,
                    max_iters=max_iters,
                    description=f"BAPR JMLR continuous gravity: {name}",
                    vram=env_vram_mb(env),
                ))
    return jobs


def build_dynamics_breadth(max_iters: int, seeds: list[int],
                           log_scale_limit: float) -> list[Job]:
    save_root = ROOT / "jax_experiments" / "results_jmlr"
    varying_params = ["body_mass", "body_inertia", "dof_damping"]
    jobs: list[Job] = []
    for param in varying_params:
        for env in ENVS:
            for seed in seeds:
                for algo in ALGOS:
                    name = f"jmlr_dynamics_{param}_{algo}_{env_short(env)}_s{seed}"
                    jobs.append(Job(
                        name=name,
                        signature=(
                            f"BAPR/jmlr/dynamics-breadth/{param}/"
                            f"{algo}/{env}/seed{seed}"
                        ),
                        args=continuous_args(
                            algo, env, seed, max_iters, save_root, name,
                            varying_param=param,
                            log_scale_limit=log_scale_limit),
                        save_root=save_root,
                        max_iters=max_iters,
                        description=f"BAPR JMLR dynamics breadth: {name}",
                        vram=env_vram_mb(env),
                    ))
    return jobs


def select_jobs(args: argparse.Namespace) -> list[Job]:
    seeds = args.seeds or JMLR_SEEDS
    jobs: list[Job] = []
    for suite in args.suite:
        if suite == "discrete-negative-fill":
            jobs.extend(build_discrete_negative_fill(args.discrete_max_iters))
        elif suite == "continuous-main":
            jobs.extend(build_continuous_main(args.continuous_max_iters, seeds))
        elif suite == "dynamics-breadth":
            jobs.extend(build_dynamics_breadth(
                args.continuous_max_iters, seeds, args.breadth_log_scale))
        else:
            raise ValueError(f"unknown suite: {suite}")
    if not args.no_skip_complete:
        jobs = [
            job for job in jobs
            if not is_complete(job.save_root, job.name, job.max_iters)
        ]
    if args.only_name:
        jobs = [
            job for job in jobs
            if any(fnmatch.fnmatch(job.name, pat) for pat in args.only_name)
        ]
    return jobs


def run_command(command: list[str], dry_run: bool) -> None:
    print(shlex.join(command), flush=True)
    if dry_run:
        return
    completed = subprocess.run(command, text=True, capture_output=True)
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


def submit(job: Job, args: argparse.Namespace) -> None:
    result_dir = job.save_root / job.name
    ckpt_dir = result_dir / "checkpoints"
    cmd = (
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        f"XLA_PYTHON_CLIENT_MEM_FRACTION={job.xla_mem_fraction} "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
        f"{shlex.quote(str(JAX_PYTHON))} -u -m jax_experiments.train "
        f"{job.args}"
    )
    command = [
        sys.executable, str(SCHEDULER), "submit",
        "--project", "BAPR",
        "--description", job.description,
        "--cmd", cmd,
        "--cwd", str(ROOT),
        "--signature", job.signature,
        "--vram", str(job.vram),
        "--ram-mb", str(job.ram),
        "--cpu", str(job.cpu),
        "--priority", args.priority,
        "--ckpt-dir", str(ckpt_dir),
        "--result-dir", str(result_dir),
        "--local-result-dir", str(result_dir),
        "--allow-remote-large-data",
        "--reroute-on-node-down",
        "--node-down-requeue-s", "900",
    ]
    if args.preferred_node:
        command += ["--preferred-node", args.preferred_node]
    run_command(command, args.dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", action="append", required=True,
        choices=["discrete-negative-fill", "continuous-main", "dynamics-breadth"],
        help="Experiment suite to submit; repeatable.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Override JMLR seeds for continuous suites.")
    parser.add_argument("--discrete-max-iters", type=int, default=1500)
    parser.add_argument("--continuous-max-iters", type=int, default=2000)
    parser.add_argument("--breadth-log-scale", type=float, default=3.0)
    parser.add_argument("--priority", choices=["low", "normal", "high"],
                        default="normal")
    parser.add_argument("--preferred-node", default="",
                        help="Optional preferred node; empty lets scheduler route.")
    parser.add_argument("--only-name", action="append", default=[],
                        help="Submit only run names matching this glob; repeatable.")
    parser.add_argument("--no-skip-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

    if not JAX_PYTHON.exists():
        raise SystemExit(f"missing Python environment: {JAX_PYTHON}")

    jobs = select_jobs(args)
    print(f"Submitting BAPR JMLR-gap jobs: {len(jobs)}", flush=True)
    for job in jobs:
        submit(job, args)

    if args.dispatch and not args.dry_run and jobs:
        run_command([sys.executable, str(SCHEDULER), "dispatch"], False)


if __name__ == "__main__":
    main()
