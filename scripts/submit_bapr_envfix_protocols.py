#!/usr/bin/env python3
"""Submit the post-envfix protocol/eval-instrumentation check for BAPR.

Three experiment lines are kept separate on purpose:

1. paper-continuous: RE-SAC/ESCP-style continuous gravity tasks.
2. escp-mild-continuous: ESCP-style 1.5**u gravity scale, seed-0 probe.
3. semantic-discrete: BAPR's discrete semantic modes after the Brax mapping fix.

Runs go to results_evalfix so strict-horizon/full-protocol metrics do not mix
with the earlier results_envfix sanity pass.

The default is a seed-0 first pass so a bad protocol does not consume the whole
cluster overnight. Pass ``--seeds 0 1 2 3 4`` for the full 5-seed rerun.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = Path("/home/erzhu419/mine_code/scheduleurm/skill/scheduler.py")
JAX_PYTHON = Path("/home/erzhu419/.conda/envs/resac-jax/bin/python")

ENVS = ["HalfCheetah-v2", "Hopper-v2", "Walker2d-v2", "Ant-v2"]
ALGOS = ["sac", "escp", "resac", "bapr"]
DEFAULT_SEEDS = [0]
DIRECT_GPU_NODES = ["local", "jtl110gpu", "jtl110gpu2", "jtl311linux", "node007"]
SCHEDULER_QUEUE = Path.home() / ".claude" / "scheduler" / "queue.json"
TERMINAL_STATES = {"done", "failed", "cancelled", "forgotten"}
DEFAULT_SAVE_ROOT = ROOT / "jax_experiments" / "results_evalfix"


def scheduler_env() -> dict[str, str]:
    env = os.environ.copy()
    skill_path = str(SCHEDULER.parent)
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        skill_path if not current else f"{skill_path}{os.pathsep}{current}"
    )
    return env


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


def existing_active_signatures() -> set[str]:
    if not SCHEDULER_QUEUE.exists():
        return set()
    try:
        import json

        state = json.loads(SCHEDULER_QUEUE.read_text())
    except Exception:
        return set()
    active: set[str] = set()
    for task in state.get("tasks", []):
        signature = task.get("signature")
        status = task.get("status")
        if signature and status not in TERMINAL_STATES:
            active.add(signature)
    return active


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


def common_train_args(max_iters: int, save_root: Path, run_name: str) -> str:
    return (
        f"--max_iters {max_iters} --samples_per_iter 4000 "
        "--updates_per_iter 250 --log_interval 5 --eval_episodes 5 "
        "--eval_protocol full --eval_switching_episodes 1 "
        "--eval_switching_period_steps 500 "
        "--save_interval 50 --backend spring --resume "
        f"--save_root {shlex.quote(str(save_root))} "
        f"--run_name {shlex.quote(run_name)}"
    )


def continuous_args(algo: str, env: str, seed: int, max_iters: int,
                    save_root: Path, run_name: str) -> str:
    return (
        f"{algo_args(algo)} --env {env} --seed {seed} "
        "--env_type continuous --task_num 40 --test_task_num 40 "
        "--changing_period 20000 "
        "--varying_params gravity --log_scale_limit 3.0 "
        f"{common_train_args(max_iters, save_root, run_name)}"
    )


def mild_continuous_args(algo: str, env: str, seed: int, max_iters: int,
                         save_root: Path, run_name: str) -> str:
    return (
        f"{algo_args(algo)} --env {env} --seed {seed} "
        "--env_type continuous --task_num 40 --test_task_num 40 "
        "--changing_period 20000 "
        "--varying_params gravity --log_scale_limit 3.0 "
        "--task_scale_distribution pow1p5 "
        f"{common_train_args(max_iters, save_root, run_name)}"
    )


def discrete_args(algo: str, env: str, seed: int, max_iters: int,
                  save_root: Path, run_name: str) -> str:
    return (
        f"{algo_args(algo)} --env {env} --seed {seed} "
        "--env_type discrete_mode --mean_dwell_iters 60 "
        f"{common_train_args(max_iters, save_root, run_name)}"
    )


def build_paper_continuous(max_iters: int, seeds: list[int],
                           save_root: Path, run_prefix: str) -> list[Job]:
    jobs: list[Job] = []
    for env in ENVS:
        for seed in seeds:
            for algo in ALGOS:
                name = (
                    f"{run_prefix}_continuous_gravity_{algo}_"
                    f"{env_short(env)}_s{seed}"
                )
                jobs.append(Job(
                    name=name,
                    signature=(
                        f"BAPR/{run_prefix}/paper-continuous/"
                        f"{algo}/{env}/seed{seed}"
                    ),
                    args=continuous_args(algo, env, seed, max_iters, save_root, name),
                    save_root=save_root,
                    max_iters=max_iters,
                    description=f"BAPR envfix paper-continuous: {name}",
                    vram=env_vram_mb(env),
                ))
    return jobs


def build_escp_mild_continuous(max_iters: int, seeds: list[int],
                               save_root: Path, run_prefix: str) -> list[Job]:
    jobs: list[Job] = []
    for env in ENVS:
        for seed in seeds:
            for algo in ALGOS:
                name = (
                    f"{run_prefix}_escp_mild_gravity_{algo}_"
                    f"{env_short(env)}_s{seed}"
                )
                jobs.append(Job(
                    name=name,
                    signature=(
                        f"BAPR/{run_prefix}/escp-mild-continuous/"
                        f"{algo}/{env}/seed{seed}"
                    ),
                    args=mild_continuous_args(
                        algo, env, seed, max_iters, save_root, name),
                    save_root=save_root,
                    max_iters=max_iters,
                    description=f"BAPR envfix ESCP-mild continuous: {name}",
                    vram=env_vram_mb(env),
                ))
    return jobs


def build_semantic_discrete(max_iters: int, seeds: list[int],
                            save_root: Path, run_prefix: str) -> list[Job]:
    jobs: list[Job] = []
    for env in ENVS:
        for seed in seeds:
            for algo in ALGOS:
                name = (
                    f"{run_prefix}_semantic_discrete_{algo}_"
                    f"{env_short(env)}_dw60_s{seed}"
                )
                jobs.append(Job(
                    name=name,
                    signature=(
                        f"BAPR/{run_prefix}/semantic-discrete/"
                        f"{algo}/{env}/seed{seed}"
                    ),
                    args=discrete_args(algo, env, seed, max_iters, save_root, name),
                    save_root=save_root,
                    max_iters=max_iters,
                    description=f"BAPR envfix semantic-discrete: {name}",
                    vram=env_vram_mb(env),
                ))
    return jobs


def select_jobs(args: argparse.Namespace) -> list[Job]:
    seeds = args.seeds or DEFAULT_SEEDS
    save_root = resolve_save_root(args.save_root)
    jobs: list[Job] = []
    for suite in args.suite:
        if suite == "paper-continuous":
            jobs.extend(build_paper_continuous(
                args.continuous_max_iters, seeds, save_root, args.run_prefix))
        elif suite == "escp-mild-continuous":
            jobs.extend(build_escp_mild_continuous(
                args.continuous_max_iters, seeds, save_root, args.run_prefix))
        elif suite == "semantic-discrete":
            jobs.extend(build_semantic_discrete(
                args.discrete_max_iters, seeds, save_root, args.run_prefix))
        else:
            raise ValueError(f"unknown suite: {suite}")
    if not args.no_skip_complete:
        jobs = [
            job for job in jobs
            if not is_complete(job.save_root, job.name, job.max_iters)
        ]
    if not args.no_skip_active:
        active = existing_active_signatures()
        jobs = [job for job in jobs if job.signature not in active]
    if args.only_name:
        jobs = [
            job for job in jobs
            if any(fnmatch.fnmatch(job.name, pat) for pat in args.only_name)
        ]
    return jobs


def resolve_save_root(value: str | None) -> Path:
    if not value:
        return DEFAULT_SAVE_ROOT
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


def run_command(command: list[str], dry_run: bool) -> None:
    print(shlex.join(command), flush=True)
    if dry_run:
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
    for node in args.allowed_node:
        command += ["--allowed-node", node]
    run_command(command, args.dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", action="append", required=True,
        choices=[
            "paper-continuous",
            "escp-mild-continuous",
            "semantic-discrete",
        ],
        help="Experiment suite to submit; repeatable.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Seeds to run. Default: seed 0 first pass.")
    parser.add_argument("--continuous-max-iters", type=int, default=2000)
    parser.add_argument("--discrete-max-iters", type=int, default=800)
    parser.add_argument("--save-root", default=None,
                        help="Result root; relative paths are under repo root.")
    parser.add_argument("--run-prefix", default="evalfix",
                        help="Prefix for run_name and scheduler signature.")
    parser.add_argument("--priority", choices=["low", "normal", "high"],
                        default="high")
    parser.add_argument("--allowed-node", action="append", default=None,
                        help="Restrict to direct GPU node; repeatable.")
    parser.add_argument("--only-name", action="append", default=[],
                        help="Submit only run names matching this glob; repeatable.")
    parser.add_argument("--no-skip-complete", action="store_true")
    parser.add_argument("--no-skip-active", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

    if not JAX_PYTHON.exists():
        raise SystemExit(f"missing Python environment: {JAX_PYTHON}")
    if not SCHEDULER.exists():
        raise SystemExit(f"missing scheduler: {SCHEDULER}")
    if args.allowed_node is None:
        args.allowed_node = list(DIRECT_GPU_NODES)

    jobs = select_jobs(args)
    print(f"Submitting BAPR envfix jobs: {len(jobs)}", flush=True)
    for job in jobs:
        submit(job, args)

    if args.dispatch and not args.dry_run and jobs:
        run_command([sys.executable, str(SCHEDULER), "dispatch"], False)


if __name__ == "__main__":
    main()
