#!/usr/bin/env python3
"""Submit the equal-budget frozen-anchor v2 development matrix."""
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
        [
            str(scheduler_common.JAX_PYTHON),
            str(Path(__file__).resolve()),
            *sys.argv[1:],
        ],
    )

from jax_experiments.analysis import (
    regime_polarity_frozen_anchor as protocol,
)


SIGNATURE_PREFIX = "BAPR/regime-polarity-frozen-anchor/v2"
DISPLAY_NAME = "Frozen anchor v2"
RESOURCE_PREFIX = "BAPR/regime-polarity-frozen-anchor"
BRANCH_MODULE = (
    "jax_experiments.analysis.run_regime_polarity_frozen_anchor_branch")
CALIBRATION_MODULE = (
    "jax_experiments.analysis.calibrate_regime_polarity_frozen_anchor")
AUDIT_MODULE = (
    "jax_experiments.analysis.audit_regime_polarity_frozen_anchor")
ANALYSIS_MODULE = (
    "jax_experiments.analysis.analyze_regime_polarity_frozen_anchor")
TRAIN_BRANCH_ROLES = protocol.BRANCH_ROLES
SUBMIT_INTENT_LABEL = "bapr-frozen-anchor-v2-submit"
DISPATCH_INTENT_LABEL = "bapr-frozen-anchor-v2-dispatch"
GPU_NODES = ["local", "jtl110gpu", "jtl110gpu2", "node007"]
CPU_NODES = [f"node00{index}" for index in range(1, 7)]
VRAM_MB = {
    "robust_long": 1800,
    "shared_small": 2800,
    "shared_wide": 2800,
    "mode_residual": 4200,
}
MEMORY_FRACTION = {
    "robust_long": 0.25,
    "shared_small": 0.42,
    "shared_wide": 0.42,
    "mode_residual": 0.65,
}
RAM_MB = {
    "robust_long": 8192,
    "shared_small": 10240,
    "shared_wide": 10240,
    "mode_residual": 12288,
}
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


def branch_signature(role: str, seed: int) -> str:
    return (
        f"{SIGNATURE_PREFIX}/branch/"
        f"{protocol.require_branch_role(role)}/seed-{int(seed)}")


def calibration_signature(variant: str, seed: int) -> str:
    return (
        f"{SIGNATURE_PREFIX}/{protocol.require_variant(variant)}"
        f"/calibration/seed-{int(seed)}")


def audit_signature(variant: str, seed: int) -> str:
    return (
        f"{SIGNATURE_PREFIX}/{protocol.require_variant(variant)}"
        f"/audit/seed-{int(seed)}")


def analysis_signature(variant: str) -> str:
    return (
        f"{SIGNATURE_PREFIX}/{protocol.require_variant(variant)}/analysis")


def branch_spec(role: str, seed: int, priority: str) -> dict:
    role = protocol.require_branch_role(role)
    run_dir = protocol.branch_run_dir(role, seed)
    bundle = protocol.branch_bundle_dir(role, seed)
    source = protocol.source_bundle_dir(seed)
    return {
        "description": f"{DISPLAY_NAME} {role} seed {seed}",
        "cmd": (
            f"SCHEDULEURM_ETA_TOTAL_UNITS="
            f"{protocol.BRANCH_FINAL_NEXT_ITERATION} "
            + _gpu_command(
                BRANCH_MODULE,
                [
                    "--role",
                    role,
                    "--seed",
                    str(seed),
                    "--resume",
                ],
                MEMORY_FRACTION[role],
            )
        ),
        "cwd": str(ROOT),
        "signature": branch_signature(role, seed),
        "project": "BAPR",
        "vram_resource_family": (
            f"{RESOURCE_PREFIX}/{role}/gpu-runtime"),
        "vram": VRAM_MB[role],
        "ram_mb": RAM_MB[role],
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
            str(path) for path in protocol.source_required_paths(seed)
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


def _pair_wait_paths(variant: str, seed: int) -> list[str]:
    return [
        str(path)
        for role in ("robust_long", protocol.require_variant(variant))
        for path in protocol.branch_required_paths(role, seed)
    ]


def calibration_spec(
    variant: str,
    seed: int,
    priority: str,
) -> dict:
    variant = protocol.require_variant(variant)
    output = protocol.calibration_dir(variant, seed)
    bundles = [
        protocol.branch_bundle_dir(role, seed)
        for role in ("robust_long", variant)
    ]
    return {
        "description": (
            f"Calibrate {DISPLAY_NAME} {variant} seed {seed}"),
        "cmd": _cpu_command(
            CALIBRATION_MODULE,
            [
                "--variant",
                variant,
                "--seed",
                str(seed),
                "--resume",
            ],
            4,
        ),
        "cwd": str(ROOT),
        "signature": calibration_signature(variant, seed),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 8192,
        "cpu": 8,
        "priority": priority,
        "allowed_nodes": CPU_NODES,
        "result_dir": str(output),
        "local_result_dir": str(output),
        "wait_for_files": _pair_wait_paths(variant, seed),
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
            "Held-out deterministic policy evaluation only; no updates."),
    }


def audit_spec(variant: str, seed: int, priority: str) -> dict:
    variant = protocol.require_variant(variant)
    output = protocol.audit_dir(variant, seed)
    bundles = [
        protocol.branch_bundle_dir(role, seed)
        for role in ("robust_long", variant)
    ]
    return {
        "description": f"Audit {DISPLAY_NAME} {variant} seed {seed}",
        "cmd": _cpu_command(
            AUDIT_MODULE,
            [
                "--variant",
                variant,
                "--seed",
                str(seed),
                "--resume",
            ],
            4,
        ),
        "cwd": str(ROOT),
        "signature": audit_signature(variant, seed),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 10240,
        "cpu": 8,
        "priority": priority,
        "allowed_nodes": CPU_NODES,
        "result_dir": str(output),
        "local_result_dir": str(output),
        "wait_for_files": [
            str(protocol.calibration_manifest(variant, seed)),
            str(protocol.MODEL_MANIFEST),
            str(protocol.MODEL_PATH),
            *_pair_wait_paths(variant, seed),
        ],
        "stage_input_paths": [
            str(protocol.calibration_dir(variant, seed)),
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
            "Frozen controllers and estimator evaluation; no updates."),
    }


def analysis_spec(variant: str, priority: str) -> dict:
    variant = protocol.require_variant(variant)
    output = protocol.analysis_root(variant)
    return {
        "description": f"Aggregate {DISPLAY_NAME} {variant}",
        "cmd": _cpu_command(
            ANALYSIS_MODULE,
            ["--variant", variant],
            2,
        ),
        "cwd": str(ROOT),
        "signature": analysis_signature(variant),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 2048,
        "cpu": 2,
        "priority": priority,
        "allowed_nodes": CPU_NODES,
        "result_dir": str(output),
        "local_result_dir": str(output),
        "wait_for_files": [
            str(protocol.audit_manifest(variant, seed))
            for seed in protocol.TRAINING_SEEDS
        ],
        "stage_input_paths": [
            str(protocol.AUDIT_ROOT / variant),
        ],
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
    if phase in ("all", "branch"):
        rows.extend([
            (
                branch_signature(role, seed),
                branch_spec(role, seed, priority),
                protocol.branch_manifest(role, seed),
            )
            for role in TRAIN_BRANCH_ROLES
            for seed in protocol.TRAINING_SEEDS
        ])
    if phase in ("all", "calibration"):
        rows.extend([
            (
                calibration_signature(variant, seed),
                calibration_spec(variant, seed, priority),
                protocol.calibration_manifest(variant, seed),
            )
            for variant in protocol.VARIANTS
            for seed in protocol.TRAINING_SEEDS
        ])
    if phase in ("all", "audit"):
        rows.extend([
            (
                audit_signature(variant, seed),
                audit_spec(variant, seed, priority),
                protocol.audit_manifest(variant, seed),
            )
            for variant in protocol.VARIANTS
            for seed in protocol.TRAINING_SEEDS
        ])
    if phase in ("all", "analysis"):
        rows.extend([
            (
                analysis_signature(variant),
                analysis_spec(variant, priority),
                protocol.analysis_json(variant),
            )
            for variant in protocol.VARIANTS
        ])
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
        SUBMIT_INTENT_LABEL,
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
    if (
        len(task_ids) != len(specs)
        or any(not task_id for task_id in task_ids)
        or len(set(task_ids)) != len(task_ids)
    ):
        raise RuntimeError(
            "scheduler submission was not fully accounted: "
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
        DISPATCH_INTENT_LABEL,
        "--intent-ttl",
        "900",
    ]
    for task_id in task_ids:
        command += ["--task-id", task_id]
    subprocess.run(
        command,
        check=True,
        env=scheduler_common.scheduler_env(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("all", "branch", "calibration", "audit", "analysis"),
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
            task
            for task in known
            if str(task.get("signature") or "") == signature
        ]
        active = [
            task
            for task in tasks
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
        print("No frozen-anchor tasks to submit")
        return
    task_ids = _submit_jsonl(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        _dispatch(task_ids)


if __name__ == "__main__":
    main()
