#!/usr/bin/env python3
"""Submit the fidelity smoke and source-controller headroom graph."""
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

from jax_experiments.analysis import resac_paper_fidelity_smoke_v1 as fidelity
from jax_experiments.analysis import regime_polarity_source_headroom_v1 as headroom


GPU_NODES = ["jtl311linux"]
CPU_NODES = [f"node00{index}" for index in range(1, 7)]
ACTIVE_STATUSES = scheduler_common.ACTIVE_STATUSES
FIDELITY_PREFIX = "BAPR/resac-paper-fidelity-smoke/v1"
HEADROOM_PREFIX = "BAPR/regime-polarity-source-headroom/v1"


def _gpu_command(module: str, values: list[str], total_units: int) -> str:
    return (
        f"SCHEDULEURM_ETA_TOTAL_UNITS={total_units} "
        "BAPR_SCHEDULER_RESUME=--resume "
        "JAX_PLATFORMS=cuda "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.28 "
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


def _common_gpu_spec(
    *, description: str, command: str, signature: str,
    run_dir: Path, bundle_dir: Path, vram: int,
    resource_family: str, priority: str, allow_over_third: bool,
) -> dict[str, object]:
    return {
        "description": description,
        "cmd": command,
        "cwd": str(ROOT),
        "signature": signature,
        "project": "BAPR",
        "vram_resource_family": resource_family,
        "vram": vram,
        "allow_gpu_over_one_third": allow_over_third,
        "ram_mb": 8192,
        "cpu": 2,
        "priority": priority,
        "allowed_nodes": GPU_NODES,
        "ckpt_dir": str(run_dir / "checkpoints"),
        "ckpt_glob": "train_state.pkl",
        "resume_flag": "",
        "resume_managed_by_cmd": True,
        "result_dir": str(bundle_dir),
        "local_result_dir": str(bundle_dir),
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
    }


def fidelity_train_signature(env: str, role: str, seed: int) -> str:
    return (
        f"{FIDELITY_PREFIX}/train/{fidelity.env_slug(env)}/"
        f"{fidelity.require_role(role)}/seed-{seed}")


def fidelity_train_spec(
    env: str, role: str, seed: int, priority: str,
) -> dict[str, object]:
    return _common_gpu_spec(
        description=(
            f"RE-SAC fidelity {fidelity.env_slug(env)} {role} seed {seed}"),
        command=_gpu_command(
            "jax_experiments.analysis.run_resac_paper_fidelity_smoke_v1",
            ["--env", env, "--role", role, "--seed", str(seed), "--resume"],
            fidelity.MAX_ITERS,
        ),
        signature=fidelity_train_signature(env, role, seed),
        run_dir=fidelity.run_dir(env, role, seed),
        bundle_dir=fidelity.bundle_dir(env, role, seed),
        vram=3200,
        resource_family="BAPR/resac-paper-fidelity/utd1-runtime",
        priority=priority,
        allow_over_third=True,
    )


def fidelity_audit_signature(env: str, role: str, seed: int) -> str:
    return (
        f"{FIDELITY_PREFIX}/audit/{fidelity.env_slug(env)}/"
        f"{fidelity.require_role(role)}/seed-{seed}")


def fidelity_audit_spec(
    env: str, role: str, seed: int, priority: str,
) -> dict[str, object]:
    output = fidelity.audit_dir(env, role, seed)
    return {
        "description": (
            f"RE-SAC fidelity audit {fidelity.env_slug(env)} {role} seed {seed}"),
        "cmd": _cpu_command(
            "jax_experiments.analysis.run_resac_paper_fidelity_audit_v1",
            ["--env", env, "--role", role, "--seed", str(seed), "--resume"],
        ),
        "cwd": str(ROOT),
        "signature": fidelity_audit_signature(env, role, seed),
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
            fidelity.bundle_required_paths(env, role, seed)
        ],
        "stage_input_paths": [str(fidelity.bundle_dir(env, role, seed))],
        "stage_excludes": [
            "jax_experiments/results*/", "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Strict deterministic checkpoint evaluation only; no updates."),
    }


def fidelity_analysis_spec(priority: str) -> dict[str, object]:
    return {
        "description": "Aggregate RE-SAC paper-fidelity smoke",
        "cmd": _cpu_command(
            "jax_experiments.analysis.analyze_resac_paper_fidelity_smoke_v1",
            [],
        ),
        "cwd": str(ROOT),
        "signature": f"{FIDELITY_PREFIX}/analysis",
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 2048,
        "cpu": 2,
        "priority": priority,
        "allowed_nodes": CPU_NODES,
        "result_dir": str(fidelity.ANALYSIS_ROOT),
        "local_result_dir": str(fidelity.ANALYSIS_ROOT),
        "wait_for_files": [
            str(fidelity.audit_manifest(env, role, seed))
            for env in fidelity.ENVS for role in fidelity.ROLES
            for seed in fidelity.TRAINING_SEEDS
        ],
        "stage_input_paths": [str(fidelity.AUDIT_ROOT)],
        "stage_excludes": ["paper/"],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": "CSV/JSON aggregation only.",
    }


def headroom_train_signature(role: str, seed: int) -> str:
    return f"{HEADROOM_PREFIX}/train/{headroom.require_role(role)}/seed-{seed}"


def headroom_train_spec(role: str, seed: int, priority: str) -> dict[str, object]:
    return _common_gpu_spec(
        description=f"Polarity source controller {role} seed {seed}",
        command=_gpu_command(
            "jax_experiments.analysis.run_regime_polarity_source_controller_v1",
            ["--role", role, "--seed", str(seed), "--resume"],
            headroom.MAX_ITERS,
        ),
        signature=headroom_train_signature(role, seed),
        run_dir=headroom.run_dir(role, seed),
        bundle_dir=headroom.bundle_dir(role, seed),
        vram=2300,
        resource_family="BAPR/regime-polarity-source-controller/runtime",
        priority=priority,
        allow_over_third=True,
    )


def headroom_audit_signature(seed: int) -> str:
    return f"{HEADROOM_PREFIX}/audit/seed-{headroom.require_training_seed(seed)}"


def headroom_audit_spec(seed: int, priority: str) -> dict[str, object]:
    output = headroom.audit_dir(seed)
    bundle_dirs = [headroom.bundle_dir(role, seed) for role in headroom.ROLES]
    return {
        "description": f"Polarity source-controller audit seed {seed}",
        "cmd": _cpu_command(
            "jax_experiments.analysis."
            "run_regime_polarity_source_headroom_audit_v1",
            ["--seed", str(seed), "--resume"],
        ),
        "cwd": str(ROOT),
        "signature": headroom_audit_signature(seed),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 16_384,
        "cpu": 32,
        "priority": priority,
        "allowed_nodes": CPU_NODES,
        "result_dir": str(output),
        "local_result_dir": str(output),
        "wait_for_files": [
            str(path) for role in headroom.ROLES
            for path in headroom.bundle_required_paths(role, seed)
        ],
        "stage_input_paths": [str(path) for path in bundle_dirs],
        "stage_excludes": [
            "jax_experiments/results*/", "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Checkpoint-only policy by mode evaluation; no updates."),
    }


def headroom_analysis_spec(priority: str) -> dict[str, object]:
    return {
        "description": "Aggregate polarity source-controller headroom",
        "cmd": _cpu_command(
            "jax_experiments.analysis."
            "analyze_regime_polarity_source_headroom_v1",
            [],
        ),
        "cwd": str(ROOT),
        "signature": f"{HEADROOM_PREFIX}/analysis",
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 2048,
        "cpu": 2,
        "priority": priority,
        "allowed_nodes": CPU_NODES,
        "result_dir": str(headroom.ANALYSIS_ROOT),
        "local_result_dir": str(headroom.ANALYSIS_ROOT),
        "wait_for_files": [
            str(headroom.audit_manifest(seed))
            for seed in headroom.TRAINING_SEEDS
        ],
        "stage_input_paths": [str(headroom.AUDIT_ROOT)],
        "stage_excludes": ["paper/"],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": "JSON aggregation only.",
    }


def candidates(track: str, phase: str, priority: str):
    rows = []
    if track in ("all", "fidelity"):
        if phase in ("all", "training"):
            rows.extend([
                (fidelity_train_signature(env, role, seed),
                 fidelity_train_spec(env, role, seed, priority),
                 fidelity.bundle_manifest(env, role, seed))
                for env in fidelity.ENVS for role in fidelity.ROLES
                for seed in fidelity.TRAINING_SEEDS
            ])
        if phase in ("all", "audit"):
            rows.extend([
                (fidelity_audit_signature(env, role, seed),
                 fidelity_audit_spec(env, role, seed, priority),
                 fidelity.audit_manifest(env, role, seed))
                for env in fidelity.ENVS for role in fidelity.ROLES
                for seed in fidelity.TRAINING_SEEDS
            ])
        if phase in ("all", "analysis"):
            rows.append((
                f"{FIDELITY_PREFIX}/analysis",
                fidelity_analysis_spec(priority), fidelity.analysis_json()))
    if track in ("all", "headroom"):
        if phase in ("all", "training"):
            rows.extend([
                (headroom_train_signature(role, seed),
                 headroom_train_spec(role, seed, priority),
                 headroom.bundle_manifest(role, seed))
                for role in headroom.ROLES for seed in headroom.TRAINING_SEEDS
            ])
        if phase in ("all", "audit"):
            rows.extend([
                (headroom_audit_signature(seed),
                 headroom_audit_spec(seed, priority),
                 headroom.audit_manifest(seed))
                for seed in headroom.TRAINING_SEEDS
            ])
        if phase in ("all", "analysis"):
            rows.append((
                f"{HEADROOM_PREFIX}/analysis",
                headroom_analysis_spec(priority), headroom.analysis_json()))
    return rows


def _submit_jsonl(specs: list[dict[str, object]]) -> list[str]:
    payload = "".join(json.dumps(spec) + "\n" for spec in specs)
    command = [
        sys.executable, str(scheduler_common.SCHEDULER),
        "submit-jsonl", "--stdin", "--trusted", "--json",
        "--intent-label", "bapr-next-protocols-20260803-submit",
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
        "--intent-label", "bapr-next-protocols-20260803-dispatch",
        "--intent-ttl", "900",
    ]
    for task_id in task_ids:
        command += ["--task-id", task_id]
    subprocess.run(command, check=True, env=scheduler_common.scheduler_env())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--track", choices=("all", "fidelity", "headroom"), default="all")
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
    for signature, spec, output in candidates(
            args.track, args.phase, args.priority):
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
        print("No next-protocol tasks to submit")
        return
    task_ids = _submit_jsonl(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        _dispatch(task_ids)


if __name__ == "__main__":
    main()
