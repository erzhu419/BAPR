#!/usr/bin/env python3
"""Submit corrected recurrent-ESCP/RE-SAC polarity comparison DAG."""
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

if (__name__ == "__main__"
        and Path(sys.executable).resolve()
        != scheduler_common.JAX_PYTHON.resolve()):
    os.execv(
        str(scheduler_common.JAX_PYTHON),
        [str(scheduler_common.JAX_PYTHON), str(Path(__file__).resolve()),
         *sys.argv[1:]],
    )

from jax_experiments.analysis import (
    regime_polarity_corrected_baselines_v2 as protocol,
)


SIGNATURE_PREFIX = "BAPR/regime-polarity-corrected-baselines/v2"
SUBMIT_INTENT = "bapr-polarity-corrected-v2-submit"
GPU_NODES = [
    "local", "jtl110gpu", "jtl110gpu2", "jtl311linux", "node007"]
CPU_NODES = [f"node00{index}" for index in range(1, 7)]
ACTIVE_STATUSES = scheduler_common.ACTIVE_STATUSES


def _gpu_command(module: str, values: list[str], total_units: int) -> str:
    return (
        f"SCHEDULEURM_ETA_TOTAL_UNITS={int(total_units)} "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 "
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
        "intra_op_parallelism_threads=4' "
        "OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 "
        "NUMEXPR_NUM_THREADS=4 JAX_NUM_THREADS=4 "
        "JAX_CPU_ENABLE_ASYNC_DISPATCH=false "
        "TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2 "
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        f"{shlex.quote(str(scheduler_common.JAX_PYTHON))} -u -m "
        f"{module} {shlex.join(values)} && echo DONE"
    )


def train_signature(method: str, seed: int) -> str:
    return f"{SIGNATURE_PREFIX}/train/{method}/seed-{int(seed)}"


def audit_signature(method: str, seed: int) -> str:
    return f"{SIGNATURE_PREFIX}/audit/{method}/seed-{int(seed)}"


def analysis_signature() -> str:
    return f"{SIGNATURE_PREFIX}/analysis"


def train_spec(method: str, seed: int, priority: str) -> dict:
    method = protocol.require_trained_method(method)
    seed = protocol.require_seed(seed)
    run = protocol.run_dir(method, seed)
    bundle = protocol.bundle_dir(method, seed)
    recurrent = method == "escp_recurrent"
    return {
        "description": f"Corrected polarity {method} seed {seed}",
        "cmd": _gpu_command(
            "jax_experiments.analysis."
            "run_regime_polarity_corrected_baseline_v2",
            ["--method", method, "--seed", str(seed), "--resume"],
            protocol.MAX_ITERS),
        "cwd": str(ROOT),
        "signature": train_signature(method, seed),
        "project": "BAPR",
        "vram_resource_family": (
            "BAPR/corrected-polarity/recurrent-escp"
            if recurrent else "BAPR/corrected-polarity/resac-b0"),
        # First recurrent smoke is architecture-small; 2.6 GB leaves room for
        # three tasks on an 8 GB card without claiming an arbitrary 4/8 GB.
        "vram": 2600 if recurrent else 2300,
        "ram_mb": 8192,
        "cpu": 2,
        "priority": priority,
        "allowed_nodes": GPU_NODES,
        "ckpt_dir": str(run / "checkpoints"),
        "ckpt_glob": "train_state.pkl",
        "resume_flag": "",
        "resume_managed_by_cmd": True,
        "result_dir": str(bundle),
        "local_result_dir": str(bundle),
        "wait_for_files": [str(protocol.REGISTRATION_PATH)],
        "stage_input_paths": [str(protocol.REGISTRATION_ROOT)],
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
    }


def audit_spec(method: str, seed: int, priority: str) -> dict:
    method = protocol.require_trained_method(method)
    seed = protocol.require_seed(seed)
    bundle = protocol.bundle_dir(method, seed)
    output = protocol.audit_dir(method, seed)
    return {
        "description": f"Corrected polarity audit {method} seed {seed}",
        "cmd": _cpu_command(
            "jax_experiments.analysis."
            "run_regime_polarity_corrected_audit_v2",
            ["--method", method, "--seed", str(seed), "--resume"]),
        "cwd": str(ROOT),
        "signature": audit_signature(method, seed),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 12_288,
        "cpu": 32,
        "priority": priority,
        "allowed_nodes": CPU_NODES,
        "result_dir": str(output),
        "local_result_dir": str(output),
        "wait_for_files": [
            str(path)
            for path in protocol.bundle_required_paths(method, seed)
        ],
        "stage_input_paths": [
            str(protocol.REGISTRATION_ROOT), str(bundle)],
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Strict deterministic checkpoint audit; no optimizer updates."),
    }


def analysis_spec(priority: str) -> dict:
    reused = [
        str(protocol.reused_result_path(method, seed))
        for method in protocol.REUSED_METHODS
        for seed in protocol.TRAINING_SEEDS]
    return {
        "description": "Aggregate corrected polarity baseline comparison",
        "cmd": _cpu_command(
            "jax_experiments.analysis."
            "analyze_regime_polarity_corrected_baselines_v2", []),
        "cwd": str(ROOT),
        "signature": analysis_signature(),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 2048,
        "cpu": 2,
        "priority": priority,
        "allowed_nodes": CPU_NODES,
        "result_dir": str(protocol.ANALYSIS_ROOT),
        "local_result_dir": str(protocol.ANALYSIS_ROOT),
        "wait_for_files": [
            *(str(path) for path in protocol.all_new_audit_manifests()),
            *reused,
        ],
        "stage_input_paths": [
            str(protocol.REGISTRATION_ROOT), str(protocol.AUDIT_ROOT),
            *reused,
        ],
        "stage_excludes": ["paper/"],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": "Immutable JSON aggregation only.",
    }


def candidates(phase: str, priority: str):
    rows = []
    if phase in ("all", "train"):
        rows.extend([
            (train_signature(method, seed),
             train_spec(method, seed, priority),
             protocol.bundle_manifest(method, seed))
            for method in protocol.TRAINED_METHODS
            for seed in protocol.TRAINING_SEEDS
        ])
    if phase in ("all", "audit"):
        rows.extend([
            (audit_signature(method, seed),
             audit_spec(method, seed, priority),
             protocol.audit_manifest(method, seed))
            for method in protocol.TRAINED_METHODS
            for seed in protocol.TRAINING_SEEDS
        ])
    if phase in ("all", "analysis"):
        rows.append((
            analysis_signature(), analysis_spec(priority),
            protocol.analysis_json()))
    return rows


def _submit_jsonl(specs: list[dict]) -> list[str]:
    payload = "".join(json.dumps(spec) + "\n" for spec in specs)
    command = [
        sys.executable, str(scheduler_common.SCHEDULER),
        "submit-jsonl", "--stdin", "--trusted", "--json",
        "--intent-label", SUBMIT_INTENT, "--intent-ttl", "900",
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
            "scheduler batch was not fully accounted: "
            f"requested={len(specs)}, returned={task_ids}")
    return task_ids


def _dispatch(task_ids: list[str]) -> None:
    if not task_ids:
        return
    command = [
        sys.executable, str(scheduler_common.SCHEDULER), "dispatch",
        "--bulk-window", "--intent-label",
        "bapr-polarity-corrected-v2-dispatch", "--intent-ttl", "900",
    ]
    for task_id in task_ids:
        command += ["--task-id", task_id]
    subprocess.run(
        command, check=True, env=scheduler_common.scheduler_env())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("all", "train", "audit", "analysis"),
        default="all")
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="high")
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

    if args.register:
        protocol.create_registration()
    protocol.validate_registration()
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
    task_ids = _submit_jsonl(specs)
    if args.dispatch:
        _dispatch(task_ids)


if __name__ == "__main__":
    main()
