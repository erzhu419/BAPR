#!/usr/bin/env python3
"""Submit the fresh-seed robust-anchored development chain."""
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

from jax_experiments.analysis import (
    regime_polarity_anchored_residual as protocol,
)


SIGNATURE_PREFIX = "BAPR/regime-polarity-anchored-residual/v1"
GPU_NODES = ["local", "jtl110gpu", "jtl110gpu2", "node007"]
CPU_NODES = [f"node00{index}" for index in range(1, 7)]
SOURCE_VRAM_MB = 1800
ANCHORED_VRAM_MB = 2600
ACTIVE_STATUSES = scheduler_common.ACTIVE_STATUSES


def _gpu_command(module: str, values: list[str], fraction: float) -> str:
    return (
        "BAPR_SCHEDULER_RESUME=--resume "
        "JAX_PLATFORMS=cuda "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        f"XLA_PYTHON_CLIENT_MEM_FRACTION={fraction:.2f} "
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


def source_signature(seed: int) -> str:
    return f"{SIGNATURE_PREFIX}/source/seed-{int(seed)}"


def branch_signature(role: str, seed: int) -> str:
    return (
        f"{SIGNATURE_PREFIX}/branch/"
        f"{protocol.require_branch_role(role)}/seed-{int(seed)}")


def calibration_signature(seed: int) -> str:
    return f"{SIGNATURE_PREFIX}/calibration/seed-{int(seed)}"


def audit_signature(seed: int) -> str:
    return f"{SIGNATURE_PREFIX}/audit/seed-{int(seed)}"


def analysis_signature() -> str:
    return f"{SIGNATURE_PREFIX}/analysis"


def source_spec(seed: int, priority: str) -> dict:
    run_dir = protocol.run_dir(protocol.ENV, "robust", seed)
    bundle = protocol.bundle_dir(protocol.ENV, "robust", seed)
    return {
        "description": f"Anchored fresh robust source seed {seed}",
        "cmd": (
            f"SCHEDULEURM_ETA_TOTAL_UNITS={protocol.SOURCE_NEXT_ITERATION} "
            + _gpu_command(
                "jax_experiments.analysis."
                "run_regime_polarity_anchored_source",
                [
                    "--env", protocol.ENV,
                    "--role", "robust",
                    "--seed", str(seed),
                    "--resume",
                ],
                0.25,
            )
        ),
        "cwd": str(ROOT),
        "signature": source_signature(seed),
        "project": "BAPR",
        "vram_resource_family": (
            "BAPR/regime-polarity-headroom/regime-sac/gpu-runtime"),
        "vram": SOURCE_VRAM_MB,
        "ram_mb": 8192,
        "cpu": 2,
        "priority": priority,
        "allowed_nodes": GPU_NODES,
        "ckpt_dir": str(run_dir / "checkpoints"),
        "ckpt_glob": "train_state.pkl",
        "resume_flag": "",
        "resume_managed_by_cmd": True,
        "result_dir": str(bundle),
        "local_result_dir": str(bundle),
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": False,
    }


def branch_spec(role: str, seed: int, priority: str) -> dict:
    role = protocol.require_branch_role(role)
    run_dir = protocol.branch_run_dir(role, seed)
    bundle = protocol.branch_bundle_dir(role, seed)
    source = protocol.bundle_dir(protocol.ENV, "robust", seed)
    anchored = role == "anchored"
    return {
        "description": f"Anchored paired {role} seed {seed}",
        "cmd": (
            f"SCHEDULEURM_ETA_TOTAL_UNITS="
            f"{protocol.BRANCH_FINAL_NEXT_ITERATION} "
            + _gpu_command(
                "jax_experiments.analysis."
                "run_regime_polarity_anchored_branch",
                [
                    "--role", role,
                    "--seed", str(seed),
                    "--resume",
                ],
                0.38 if anchored else 0.25,
            )
        ),
        "cwd": str(ROOT),
        "signature": branch_signature(role, seed),
        "project": "BAPR",
        "vram_resource_family": (
            "BAPR/regime-polarity-anchored-residual/controller-v1"
            if anchored else
            "BAPR/regime-polarity-headroom/regime-sac/gpu-runtime"
        ),
        "vram": ANCHORED_VRAM_MB if anchored else SOURCE_VRAM_MB,
        "ram_mb": 10240 if anchored else 8192,
        "cpu": 2,
        "priority": priority,
        "allowed_nodes": GPU_NODES,
        "ckpt_dir": str(run_dir / "checkpoints"),
        "ckpt_glob": "train_state.pkl",
        "resume_flag": "",
        "resume_managed_by_cmd": True,
        "result_dir": str(bundle),
        "local_result_dir": str(bundle),
        "wait_for_files": [
            str(path)
            for path in protocol.bundle_required_paths(
                protocol.ENV, "robust", seed)
        ],
        "stage_input_paths": [str(source)],
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
    }


def _branch_wait_paths(seed: int) -> list[str]:
    return [
        str(path)
        for role in protocol.BRANCH_ROLES
        for path in protocol.branch_required_paths(role, seed)
    ]


def calibration_spec(seed: int, priority: str) -> dict:
    output = protocol.calibration_dir(seed)
    bundles = [
        protocol.branch_bundle_dir(role, seed)
        for role in protocol.BRANCH_ROLES
    ]
    return {
        "description": f"Calibrate anchored fallback seed {seed}",
        "cmd": _cpu_command(
            "jax_experiments.analysis."
            "calibrate_regime_polarity_anchored_residual",
            ["--seed", str(seed), "--resume"],
            4,
        ),
        "cwd": str(ROOT),
        "signature": calibration_signature(seed),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 8192,
        "cpu": 8,
        "priority": priority,
        "allowed_nodes": CPU_NODES,
        "result_dir": str(output),
        "local_result_dir": str(output),
        "wait_for_files": _branch_wait_paths(seed),
        "stage_input_paths": [str(path) for path in bundles],
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Deterministic held-out policy evaluation only; no updates."),
    }


def audit_spec(seed: int, priority: str) -> dict:
    output = protocol.audit_dir(seed)
    bundles = [
        protocol.branch_bundle_dir(role, seed)
        for role in protocol.BRANCH_ROLES
    ]
    return {
        "description": f"Strict anchored residual audit seed {seed}",
        "cmd": _cpu_command(
            "jax_experiments.analysis."
            "audit_regime_polarity_anchored_residual",
            ["--seed", str(seed), "--resume"],
            4,
        ),
        "cwd": str(ROOT),
        "signature": audit_signature(seed),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 10240,
        "cpu": 8,
        "priority": priority,
        "allowed_nodes": CPU_NODES,
        "result_dir": str(output),
        "local_result_dir": str(output),
        "wait_for_files": [
            str(protocol.calibration_manifest(seed)),
            str(protocol.MODEL_MANIFEST),
            str(protocol.MODEL_PATH),
            *_branch_wait_paths(seed),
        ],
        "stage_input_paths": [
            str(protocol.calibration_dir(seed)),
            str(protocol.MODEL_MANIFEST.parent),
            *[str(path) for path in bundles],
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
            "Frozen deterministic controllers and estimator evaluation; "
            "no parameter updates."),
    }


def analysis_spec(priority: str) -> dict:
    return {
        "description": "Aggregate anchored residual development decision",
        "cmd": _cpu_command(
            "jax_experiments.analysis."
            "analyze_regime_polarity_anchored_residual",
            [],
            2,
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
        "stage_input_paths": [str(protocol.AUDIT_ROOT)],
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": "JSON aggregation only.",
    }


def candidates(phase: str, priority: str):
    rows = []
    if phase in ("all", "source"):
        rows.extend([
            (
                source_signature(seed),
                source_spec(seed, priority),
                protocol.bundle_manifest(protocol.ENV, "robust", seed),
            )
            for seed in protocol.TRAINING_SEEDS
        ])
    if phase in ("all", "branch"):
        rows.extend([
            (
                branch_signature(role, seed),
                branch_spec(role, seed, priority),
                protocol.branch_manifest(role, seed),
            )
            for role in protocol.BRANCH_ROLES
            for seed in protocol.TRAINING_SEEDS
        ])
    if phase in ("all", "calibration"):
        rows.extend([
            (
                calibration_signature(seed),
                calibration_spec(seed, priority),
                protocol.calibration_manifest(seed),
            )
            for seed in protocol.TRAINING_SEEDS
        ])
    if phase in ("all", "audit"):
        rows.extend([
            (
                audit_signature(seed),
                audit_spec(seed, priority),
                protocol.audit_manifest(seed),
            )
            for seed in protocol.TRAINING_SEEDS
        ])
    if phase in ("all", "analysis"):
        rows.append((
            analysis_signature(),
            analysis_spec(priority),
            protocol.analysis_json(),
        ))
    return rows


def _submit_jsonl(specs: list[dict]) -> list[str]:
    payload = "".join(json.dumps(spec) + "\n" for spec in specs)
    command = [
        sys.executable,
        str(scheduler_common.SCHEDULER),
        "submit-jsonl",
        "--stdin",
        "--trusted",
        "--json",
        "--intent-label",
        "bapr-anchored-residual-v1-submit",
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
    print(json.dumps(response, indent=2))
    submitted = response.get("submitted", [])
    task_ids = [str(item.get("id", "")) for item in submitted]
    if (len(task_ids) != len(specs)
            or any(not task_id for task_id in task_ids)
            or len(set(task_ids)) != len(task_ids)):
        raise RuntimeError(
            "scheduler submission was not all-or-verifiably-accounted: "
            f"requested={len(specs)}, returned={task_ids}")
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
        "bapr-anchored-residual-v1-dispatch",
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
        "--phase",
        choices=(
            "all", "source", "branch", "calibration", "audit", "analysis"),
        default="all",
    )
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
            if str(task.get("signature") or "") == signature
        ]
        active = [
            task for task in tasks
            if str(task.get("status")) in ACTIVE_STATUSES
        ]
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
        print("No anchored residual tasks to submit")
        return
    task_ids = _submit_jsonl(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        _dispatch(task_ids)


if __name__ == "__main__":
    main()
