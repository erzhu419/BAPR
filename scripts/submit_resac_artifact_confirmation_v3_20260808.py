#!/usr/bin/env python3
"""Submit the five-seed, 8M-step RE-SAC artifact confirmation graph."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import submit_bapr_v3_budget_matched_fork as scheduler_common

if Path(sys.executable).resolve() != scheduler_common.JAX_PYTHON.resolve():
    os.execv(
        str(scheduler_common.JAX_PYTHON),
        [str(scheduler_common.JAX_PYTHON), str(Path(__file__).resolve()),
         *sys.argv[1:]],
    )

from jax_experiments.analysis import resac_artifact_confirmation_v3 as protocol


GPU_NODES = ["local", "jtl110gpu", "jtl110gpu2", "jtl311linux", "node007"]
CPU_NODES = [f"node00{index}" for index in range(1, 7)]
ACTIVE_STATUSES = scheduler_common.ACTIVE_STATUSES
PREFIX = "BAPR/resac-artifact-confirmation/v3"


def _gpu_command(
    module: str,
    values: list[str],
    total_units: int,
    memory_fraction: float,
) -> str:
    return (
        f"SCHEDULEURM_ETA_TOTAL_UNITS={total_units} "
        "BAPR_SCHEDULER_RESUME=--resume "
        "JAX_PLATFORMS=cuda "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        f"XLA_PYTHON_CLIENT_MEM_FRACTION={memory_fraction:.2f} "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
        "OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 "
        "NUMEXPR_NUM_THREADS=2 JAX_NUM_THREADS=2 "
        "TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 "
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        f"{shlex.quote(str(scheduler_common.JAX_PYTHON))} -u -m "
        f"{module} {shlex.join(values)} && echo DONE"
    )


def _cpu_command(module: str, values: list[str]) -> str:
    return (
        "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' "
        "XLA_FLAGS='--xla_cpu_multi_thread_eigen=false "
        "intra_op_parallelism_threads=1' "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 JAX_NUM_THREADS=1 "
        "JAX_CPU_ENABLE_ASYNC_DISPATCH=false "
        "TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 "
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        f"{shlex.quote(str(scheduler_common.JAX_PYTHON))} -u -m "
        f"{module} {shlex.join(values)} && echo DONE"
    )


def train_signature(env: str, role: str, seed: int) -> str:
    return (
        f"{PREFIX}/train/{protocol.env_slug(env)}/"
        f"{protocol.require_role(role)}/seed-{seed}")


def audit_signature(env: str, role: str, seed: int) -> str:
    return (
        f"{PREFIX}/audit/{protocol.env_slug(env)}/"
        f"{protocol.require_role(role)}/seed-{seed}")


def _training_resources(env: str, role: str) -> tuple[int, float, str]:
    role = protocol.require_role(role)
    env = protocol.require_env(env)
    if role == "sac":
        return 1500, 0.17, "BAPR/released-sac-k2/utd250-runtime"
    if env == "HalfCheetah-v2":
        return 1750, 0.19, "BAPR/released-resac-b0-k5/utd250-runtime"
    return 1900, 0.21, "BAPR/released-resac-b0-k10/utd250-runtime"


def train_spec(
    env: str,
    role: str,
    seed: int,
    priority: str,
) -> dict[str, object]:
    run_dir = protocol.run_dir(env, role, seed)
    vram, fraction, family = _training_resources(env, role)
    return {
        "description": (
            f"Artifact 8M {protocol.env_slug(env)} {role} seed {seed}"),
        "cmd": _gpu_command(
            "jax_experiments.analysis.run_resac_artifact_confirmation_v3",
            ["--env", env, "--role", role, "--seed", str(seed), "--resume"],
            protocol.MAX_ITERS,
            fraction,
        ),
        "cwd": str(ROOT),
        "signature": train_signature(env, role, seed),
        "project": "BAPR",
        "vram_resource_family": family,
        "vram": vram,
        "allow_gpu_over_one_third": True,
        "ram_mb": 8192,
        "cpu": 2,
        "priority": priority,
        "allowed_nodes": GPU_NODES,
        "ckpt_dir": str(run_dir / "checkpoints"),
        "ckpt_glob": "train_state.pkl",
        "resume_flag": "",
        "resume_managed_by_cmd": True,
        "result_dir": str(protocol.bundle_dir(env, role, seed)),
        "local_result_dir": str(protocol.bundle_dir(env, role, seed)),
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
    }


def audit_spec(
    env: str,
    role: str,
    seed: int,
    priority: str,
) -> dict[str, object]:
    source = protocol.bundle_dir(env, role, seed)
    output = protocol.audit_dir(env, role, seed)
    return {
        "description": (
            f"Artifact audit {protocol.env_slug(env)} {role} seed {seed}"),
        "cmd": _cpu_command(
            "jax_experiments.analysis."
            "run_resac_artifact_confirmation_audit_v3",
            ["--env", env, "--role", role, "--seed", str(seed), "--resume"],
        ),
        "cwd": str(ROOT),
        "signature": audit_signature(env, role, seed),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 12_288,
        "cpu": 32,
        "priority": priority,
        "allowed_nodes": CPU_NODES,
        "result_dir": str(output),
        "local_result_dir": str(output),
        "wait_for_files": [
            str(path) for path in
            protocol.bundle_required_paths(env, role, seed)
        ],
        "stage_input_paths": [str(source)],
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Strict deterministic checkpoint evaluation only; no updates."),
    }


def analysis_spec(priority: str) -> dict[str, object]:
    inputs = [protocol.AUDIT_ROOT]
    inputs.extend(
        protocol.bundle_dir(env, role, seed)
        for env in protocol.ENVS for role in protocol.ROLES
        for seed in protocol.TRAINING_SEEDS)
    return {
        "description": "Aggregate five-seed RE-SAC artifact confirmation",
        "cmd": _cpu_command(
            "jax_experiments.analysis.analyze_resac_artifact_confirmation_v3",
            [],
        ),
        "cwd": str(ROOT),
        "signature": f"{PREFIX}/analysis",
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 8192,
        "cpu": 4,
        "priority": priority,
        "allowed_nodes": CPU_NODES,
        "result_dir": str(protocol.ANALYSIS_ROOT),
        "local_result_dir": str(protocol.ANALYSIS_ROOT),
        "wait_for_files": [
            str(protocol.audit_manifest(env, role, seed))
            for env in protocol.ENVS for role in protocol.ROLES
            for seed in protocol.TRAINING_SEEDS
        ],
        "stage_input_paths": [str(path) for path in inputs],
        "stage_excludes": ["paper/"],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Checkpoint diagnostics and CSV/JSON aggregation only."),
    }


def candidates(phase: str, priority: str):
    rows = []
    if phase in ("all", "training"):
        rows.extend([
            (train_signature(env, role, seed),
             train_spec(env, role, seed, priority),
             protocol.bundle_manifest(env, role, seed))
            for env in protocol.ENVS for role in protocol.ROLES
            for seed in protocol.TRAINING_SEEDS
        ])
    if phase in ("all", "audit"):
        rows.extend([
            (audit_signature(env, role, seed),
             audit_spec(env, role, seed, priority),
             protocol.audit_manifest(env, role, seed))
            for env in protocol.ENVS for role in protocol.ROLES
            for seed in protocol.TRAINING_SEEDS
        ])
    if phase in ("all", "analysis"):
        rows.append((
            f"{PREFIX}/analysis", analysis_spec(priority),
            protocol.analysis_json()))
    return rows


def _submit_jsonl(specs: list[dict[str, object]]) -> list[str]:
    payload = "".join(json.dumps(spec) + "\n" for spec in specs)
    command = [
        sys.executable, str(scheduler_common.SCHEDULER),
        "submit-jsonl", "--stdin", "--trusted", "--json",
        "--intent-label", "bapr-resac-artifact-v3-submit",
        "--intent-ttl", "900",
    ]
    result = subprocess.run(
        command, input=payload, text=True, capture_output=True,
        env=scheduler_common.scheduler_env())
    if result.returncode != 0:
        print((result.stdout or "") + (result.stderr or ""), file=sys.stderr)
        result.check_returncode()
    response = json.loads(result.stdout)
    print(json.dumps(response, indent=2))
    submitted = response.get("submitted", [])
    task_ids = [str(item.get("id", "")) for item in submitted]
    if (len(task_ids) != len(specs)
            or any(not task_id for task_id in task_ids)
            or len(set(task_ids)) != len(task_ids)):
        raise RuntimeError(
            "scheduler submission was not fully accounted: "
            f"requested={len(specs)}, returned={task_ids}")
    return task_ids


def _dispatch(task_ids: list[str]) -> None:
    if not task_ids:
        return
    command = [
        sys.executable, str(scheduler_common.SCHEDULER), "dispatch",
        "--bulk-window",
        "--intent-label", "bapr-resac-artifact-v3-dispatch",
        "--intent-ttl", "900",
    ]
    for task_id in task_ids:
        command += ["--task-id", task_id]
    subprocess.run(command, check=True, env=scheduler_common.scheduler_env())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("all", "training", "audit", "analysis"),
        default="all")
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="high")
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

    known = scheduler_common.scheduler_tasks()
    specs = []
    for signature, spec, output in candidates(args.phase, args.priority):
        tasks = [
            task for task in known
            if str(task.get("signature") or "") == signature]
        active = [
            task for task in tasks
            if str(task.get("status")) in ACTIVE_STATUSES]
        if output.is_file():
            print(f"skip complete-output: {signature}")
        elif active:
            print("skip active: " + ",".join(
                str(task["id"]) for task in active))
        elif tasks and not args.retry_incomplete:
            print("skip terminal-incomplete: " + ",".join(
                str(task["id"]) for task in tasks))
        else:
            specs.append(spec)
    if args.dry_run:
        print(json.dumps(specs, indent=2, sort_keys=True))
        return
    if not specs:
        print("No artifact-confirmation-v3 tasks to submit")
        return
    task_ids = _submit_jsonl(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        _dispatch(task_ids)


if __name__ == "__main__":
    main()
