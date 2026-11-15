#!/usr/bin/env python3
"""Register and batch-submit the independent v9 confirmation DAG."""
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

if (
    __name__ == "__main__"
    and Path(sys.executable).resolve() != scheduler_common.JAX_PYTHON.resolve()
):
    os.execv(
        str(scheduler_common.JAX_PYTHON),
        [str(scheduler_common.JAX_PYTHON), str(Path(__file__).resolve()),
         *sys.argv[1:]],
    )

from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_system_id_v5 as frozen_v5,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_safe_utility_confirmation_v9 as protocol,
)


SIGNATURE_PREFIX = "BAPR/regime-polarity-safe-utility/v9-independent"
SUBMIT_INTENT = "bapr-safe-utility-v9-submit"
GPU_NODES = [
    "local", "jtl110gpu", "jtl110gpu2", "jtl311linux", "node007"]
CPU_NODES = [f"node00{index}" for index in range(1, 7)]
ACTIVE_STATUSES = scheduler_common.ACTIVE_STATUSES


def _gpu_command(module: str, values: list[str]) -> str:
    return (
        f"SCHEDULEURM_ETA_TOTAL_UNITS={protocol.MAX_ITERS} "
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


def _cpu_command(module: str, values: list[str], threads: int) -> str:
    return (
        "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' "
        "XLA_FLAGS='--xla_cpu_multi_thread_eigen=false "
        f"intra_op_parallelism_threads={threads}' "
        f"OMP_NUM_THREADS={threads} OPENBLAS_NUM_THREADS={threads} "
        f"MKL_NUM_THREADS={threads} NUMEXPR_NUM_THREADS={threads} "
        f"JAX_NUM_THREADS={threads} JAX_CPU_ENABLE_ASYNC_DISPATCH=false "
        f"TF_NUM_INTRAOP_THREADS={threads} TF_NUM_INTEROP_THREADS=2 "
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        f"{shlex.quote(str(scheduler_common.JAX_PYTHON))} -u -m "
        f"{module} {shlex.join(values)} && echo DONE"
    )


def source_signature(role: str, seed: int) -> str:
    return (
        f"{SIGNATURE_PREFIX}/train/{protocol.require_role(role)}/"
        f"seed-{protocol.require_training_seed(seed)}"
    )


def baseline_signature(method: str, seed: int) -> str:
    return (
        f"{SIGNATURE_PREFIX}/train/{protocol.require_trained_method(method)}/"
        f"seed-{protocol.require_training_seed(seed)}"
    )


def source_spec(role: str, seed: int, priority: str) -> dict:
    role = protocol.require_role(role)
    seed = protocol.require_training_seed(seed)
    run = protocol.run_dir(role, seed)
    bundle = protocol.bundle_dir(role, seed)
    return {
        "description": f"V9 fresh policy bank {role} seed {seed}",
        "cmd": _gpu_command(
            "jax_experiments.analysis."
            "run_regime_polarity_specialist_safe_utility_source_v9",
            ["--role", role, "--seed", str(seed), "--resume"],
        ),
        "cwd": str(ROOT),
        "signature": source_signature(role, seed),
        "project": "BAPR",
        "vram_resource_family": (
            "BAPR/regime-polarity-source-controller/runtime"),
        "vram": 2300,
        "allow_gpu_over_one_third": True,
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
        "stage_input_paths": [
            str(protocol.REGISTRATION_ROOT),
            str(frozen_v5.MODEL_ROOT),
        ],
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
    }


def baseline_spec(method: str, seed: int, priority: str) -> dict:
    method = protocol.require_trained_method(method)
    seed = protocol.require_training_seed(seed)
    run = protocol.run_dir(method, seed)
    bundle = protocol.bundle_dir(method, seed)
    recurrent = method == "escp_recurrent"
    return {
        "description": f"V9 paired {method} seed {seed}",
        "cmd": _gpu_command(
            "jax_experiments.analysis."
            "run_regime_polarity_specialist_safe_utility_baseline_v9",
            ["--method", method, "--seed", str(seed), "--resume"],
        ),
        "cwd": str(ROOT),
        "signature": baseline_signature(method, seed),
        "project": "BAPR",
        "vram_resource_family": (
            "BAPR/polarity-corrected/escp-recurrent-runtime"
            if recurrent else "BAPR/polarity-corrected/resac-b0-runtime"
        ),
        "vram": 2600 if recurrent else 2300,
        "allow_gpu_over_one_third": True,
        "ram_mb": 10_240 if recurrent else 8192,
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
        "stage_input_paths": [
            str(protocol.REGISTRATION_ROOT),
            str(frozen_v5.MODEL_ROOT),
        ],
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
    }


def audit_signature(seed: int) -> str:
    return (
        f"{SIGNATURE_PREFIX}/audit/seed-"
        f"{protocol.require_training_seed(seed)}"
    )


def audit_spec(seed: int, priority: str) -> dict:
    seed = protocol.require_training_seed(seed)
    names = (*protocol.ROLES, *protocol.BASELINE_METHODS)
    return {
        "description": f"V9 joint safe-utility audit seed {seed}",
        "cmd": _cpu_command(
            "jax_experiments.analysis."
            "run_regime_polarity_specialist_safe_utility_confirmation_audit_v9",
            ["--seed", str(seed), "--resume"],
            threads=8,
        ),
        "cwd": str(ROOT),
        "signature": audit_signature(seed),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 12_288,
        "cpu": 8,
        "priority": priority,
        "allowed_nodes": CPU_NODES,
        "result_dir": str(protocol.audit_dir(seed)),
        "local_result_dir": str(protocol.audit_dir(seed)),
        "wait_for_files": [
            str(protocol.REGISTRATION_PATH),
            str(frozen_v5.MODEL_MANIFEST),
            str(frozen_v5.MODEL_PATH),
            *(
                str(path)
                for name in names
                for path in protocol.bundle_required_paths(name, seed)
            ),
        ],
        "stage_input_paths": [
            str(protocol.REGISTRATION_ROOT),
            str(frozen_v5.MODEL_ROOT),
            *(str(protocol.bundle_dir(name, seed)) for name in names),
        ],
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Frozen policy calibration and strict holdout evaluation; no updates."
        ),
    }


def analysis_signature() -> str:
    return f"{SIGNATURE_PREFIX}/analysis"


def analysis_spec(priority: str) -> dict:
    return {
        "description": "Aggregate independent v9 safe-utility confirmation",
        "cmd": _cpu_command(
            "jax_experiments.analysis."
            "analyze_regime_polarity_specialist_safe_utility_confirmation_v9",
            [],
            threads=2,
        ),
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
            str(protocol.audit_manifest(seed))
            for seed in protocol.TRAINING_SEEDS
        ],
        "stage_input_paths": [
            str(protocol.AUDIT_ROOT),
            str(protocol.REGISTRATION_ROOT),
            str(frozen_v5.MODEL_ROOT),
        ],
        "stage_excludes": ["paper/"],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": "Immutable JSON aggregation only.",
    }


def candidates(priority: str):
    rows = [
        (
            source_signature(role, seed),
            source_spec(role, seed, priority),
            protocol.bundle_manifest(role, seed),
        )
        for seed in protocol.TRAINING_SEEDS
        for role in protocol.ROLES
    ]
    rows.extend([
        (
            baseline_signature(method, seed),
            baseline_spec(method, seed, priority),
            protocol.bundle_manifest(method, seed),
        )
        for seed in protocol.TRAINING_SEEDS
        for method in protocol.BASELINE_METHODS
    ])
    rows.extend([
        (
            audit_signature(seed),
            audit_spec(seed, priority),
            protocol.audit_manifest(seed),
        )
        for seed in protocol.TRAINING_SEEDS
    ])
    rows.append((
        analysis_signature(),
        analysis_spec(priority),
        protocol.analysis_json(),
    ))
    return rows


def _submit(specs: list[dict]) -> list[str]:
    payload = "".join(json.dumps(spec) + "\n" for spec in specs)
    command = [
        sys.executable,
        str(scheduler_common.SCHEDULER),
        "submit-jsonl",
        "--stdin",
        "--trusted",
        "--json",
        "--intent-label",
        SUBMIT_INTENT,
        "--intent-ttl",
        "900",
    ]
    result = subprocess.run(
        command,
        input=payload,
        text=True,
        capture_output=True,
        env=scheduler_common.scheduler_env(),
    )
    if result.returncode != 0:
        print((result.stdout or "") + (result.stderr or ""), file=sys.stderr)
        result.check_returncode()
    response = json.loads(result.stdout)
    task_ids = [
        str(row.get("id") or "") for row in response.get("submitted", [])]
    if (
        len(task_ids) != len(specs)
        or any(not task_id for task_id in task_ids)
        or len(set(task_ids)) != len(task_ids)
    ):
        raise RuntimeError(
            f"scheduler batch incomplete: requested={len(specs)} ids={task_ids}")
    print(json.dumps(response, indent=2))
    return task_ids


def _dispatch(task_ids: list[str]) -> None:
    if not task_ids:
        return
    command = [
        sys.executable,
        str(scheduler_common.SCHEDULER),
        "dispatch",
        "--bulk-window",
        "--intent-label",
        "bapr-safe-utility-v9-dispatch",
        "--intent-ttl",
        "900",
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
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

    protocol.create_registration()
    rows = candidates(args.priority)
    if args.phase == "train":
        rows = rows[:35]
    elif args.phase == "audit":
        rows = rows[35:40]
    elif args.phase == "analysis":
        rows = rows[40:]

    known = scheduler_common.scheduler_tasks()
    specs = []
    for signature, spec, output in rows:
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
        print("No v9 confirmation tasks to submit")
        return
    task_ids = _submit(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        _dispatch(task_ids)


if __name__ == "__main__":
    main()
