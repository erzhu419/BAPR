#!/usr/bin/env python3
"""Register and batch-submit the Ant V23 matching-checkpoint DAG."""
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
    regime_polarity_ant_matching_checkpoint_v23 as protocol,
)


SIGNATURE_PREFIX = "BAPR/regime-polarity/v23-ant-matching-checkpoint"
SUBMIT_INTENT = "bapr-v23-ant-matching-checkpoint-submit"
GPU_NODES = ["jtl110gpu", "jtl110gpu2", "jtl311linux", "node007"]
CPU_NODES = ["jtl110cpu", "jtl110cpu2", "node007"]
ACTIVE_STATUSES = scheduler_common.ACTIVE_STATUSES


def _gpu_command(module: str, values: list[str]) -> str:
    return (
        f"SCHEDULEURM_ETA_TOTAL_UNITS={protocol.FINAL_NEXT_ITERATION} "
        "JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.22 "
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
        "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' TMPDIR=/tmp "
        "XLA_FLAGS='--xla_cpu_multi_thread_eigen=false "
        "intra_op_parallelism_threads=16' "
        "OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 "
        "NUMEXPR_NUM_THREADS=16 JAX_NUM_THREADS=16 "
        "TF_NUM_INTRAOP_THREADS=16 TF_NUM_INTEROP_THREADS=2 "
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        f"{shlex.quote(str(scheduler_common.JAX_PYTHON))} -u -m "
        f"{module} {shlex.join(values)} && echo DONE"
    )


def common_stage_inputs() -> list[str]:
    parent = protocol.parent
    return [
        str(protocol.REGISTRATION_ROOT),
        str(ROOT / "jax_experiments" / "analysis"),
        str(ROOT / "jax_experiments" / "algos"),
        str(ROOT / "jax_experiments" / "common"),
        str(ROOT / "jax_experiments" / "envs"),
        str(ROOT / "scripts"),
        str(parent.REGISTRATION_ROOT),
        str(parent.ANALYSIS_ROOT),
        str(parent.SOURCE_BUNDLE_ROOT),
        str(parent.SPECIALIST_BUNDLE_ROOT),
    ]


def specialist_signature(variant: str, seed: int, mode: int) -> str:
    return (
        f"{SIGNATURE_PREFIX}/train/{variant}/seed-{seed}/mode-{mode}"
    )


def specialist_spec(
    variant: str, seed: int, mode: int, priority: str,
) -> dict:
    run = protocol.run_dir(variant, seed, mode)
    bundle = protocol.bundle_dir(variant, seed, mode)
    return {
        "description": (
            f"Ant v23 {variant} specialist seed {seed} mode {mode}"),
        "cmd": _gpu_command(
            "jax_experiments.analysis."
            "run_regime_polarity_ant_matching_checkpoint_v23",
            ["--variant", variant, "--seed", str(seed),
             "--mode", str(mode), "--resume"],
        ),
        "cwd": str(ROOT),
        "signature": specialist_signature(variant, seed, mode),
        "project": "BAPR",
        "vram_resource_family": "BAPR/ant-v22/sac/gpu-runtime",
        "vram": 2700,
        "ram_mb": 6144,
        "cpu": 2,
        "priority": priority,
        "allowed_nodes": GPU_NODES,
        "ckpt_dir": str(run / "checkpoints"),
        "ckpt_glob": "train_state.pkl",
        "resume_flag": "",
        "resume_managed_by_cmd": True,
        "result_dir": str(bundle),
        "local_result_dir": str(bundle),
        "wait_for_files": [
            str(protocol.REGISTRATION_PATH),
            *(str(path) for path in protocol.source_required_paths(seed)),
        ],
        "stage_input_paths": common_stage_inputs(),
        "stage_excludes": ["paper/", "__pycache__/"],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
    }


def audit_signature(variant: str, seed: int) -> str:
    return f"{SIGNATURE_PREFIX}/audit/{variant}/seed-{seed}"


def audit_spec(variant: str, seed: int, priority: str) -> dict:
    required = [str(protocol.REGISTRATION_PATH)]
    for mode in protocol.MODES:
        required.extend(str(path) for path in protocol.bundle_required_paths(
            variant, seed, mode))
    return {
        "description": f"Ant v23 {variant} hybrid-bank audit seed {seed}",
        "cmd": _cpu_command(
            "jax_experiments.analysis."
            "run_regime_polarity_ant_matching_checkpoint_audit_v23",
            ["--variant", variant, "--seed", str(seed), "--resume"],
        ),
        "cwd": str(ROOT),
        "signature": audit_signature(variant, seed),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 16384,
        "cpu": 16,
        "priority": priority,
        "allowed_nodes": CPU_NODES,
        "result_dir": str(protocol.audit_dir(variant, seed)),
        "local_result_dir": str(protocol.audit_dir(variant, seed)),
        "wait_for_files": required,
        "stage_input_paths": [
            *common_stage_inputs(),
            *(str(protocol.bundle_dir(variant, seed, mode))
              for mode in protocol.MODES),
        ],
        "stage_excludes": ["paper/", "__pycache__/"],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": "Deterministic policy evaluation only.",
    }


def analysis_signature() -> str:
    return f"{SIGNATURE_PREFIX}/analysis"


def analysis_spec(priority: str) -> dict:
    return {
        "description": "Aggregate Ant v23 matching-checkpoint decision",
        "cmd": _cpu_command(
            "jax_experiments.analysis."
            "analyze_regime_polarity_ant_matching_checkpoint_v23",
            [],
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
            str(protocol.audit_manifest(variant, seed))
            for variant in protocol.VARIANTS
            for seed in protocol.TRAINING_SEEDS
        ],
        "stage_input_paths": [
            *common_stage_inputs(),
            str(protocol.AUDIT_ROOT),
        ],
        "stage_excludes": ["paper/", "__pycache__/"],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": "JSON aggregation only.",
    }


def candidates(priority: str):
    rows = []
    for variant in protocol.VARIANTS:
        for seed in protocol.TRAINING_SEEDS:
            for mode in protocol.TRAINED_MODES:
                rows.append((
                    specialist_signature(variant, seed, mode),
                    specialist_spec(variant, seed, mode, priority),
                    protocol.bundle_manifest(variant, seed, mode),
                ))
            rows.append((
                audit_signature(variant, seed),
                audit_spec(variant, seed, priority),
                protocol.audit_manifest(variant, seed),
            ))
    rows.append((
        analysis_signature(), analysis_spec(priority), protocol.analysis_json(),
    ))
    return rows


def _submit_jsonl(specs: list[dict]) -> list[str]:
    payload = "".join(json.dumps(spec) + "\n" for spec in specs)
    command = [
        sys.executable, str(scheduler_common.SCHEDULER), "submit-jsonl",
        "--stdin", "--trusted", "--json",
        "--intent-label", SUBMIT_INTENT,
        "--intent-ttl", "900",
    ]
    result = subprocess.run(
        command, input=payload, text=True, capture_output=True,
        env=scheduler_common.scheduler_env())
    if result.returncode != 0:
        print((result.stdout or "") + (result.stderr or ""), file=sys.stderr)
        result.check_returncode()
    response = json.loads(result.stdout)
    submitted = response.get("submitted", [])
    task_ids = [str(item.get("id", "")) for item in submitted]
    if len(task_ids) != len(specs) or any(not task_id for task_id in task_ids):
        raise RuntimeError(
            "scheduler did not account for every Ant v23 task: "
            f"requested={len(specs)} returned={task_ids}")
    print(json.dumps(response, indent=2))
    return task_ids


def _dispatch(task_ids: list[str]) -> None:
    if not task_ids:
        return
    command = [
        sys.executable, str(scheduler_common.SCHEDULER), "dispatch",
        "--bulk-window", "--intent-label", f"{SUBMIT_INTENT}-dispatch",
        "--intent-ttl", "900",
    ]
    for task_id in task_ids:
        command += ["--task-id", task_id]
    subprocess.run(
        command, check=True, env=scheduler_common.scheduler_env())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="high")
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

    protocol.create_registration()
    known = scheduler_common.scheduler_tasks()
    specs = []
    for signature, spec, output in candidates(args.priority):
        matches = [
            task for task in known
            if str(task.get("signature") or "") == signature
        ]
        active = [
            task for task in matches
            if str(task.get("status")) in ACTIVE_STATUSES
        ]
        if output.is_file():
            print(f"skip complete-output: {signature}")
        elif active:
            print("skip active: " + ",".join(
                str(task["id"]) for task in active))
        elif matches and not args.retry_incomplete:
            print("skip terminal-incomplete: " + ",".join(
                str(task["id"]) for task in matches))
        else:
            specs.append(spec)
    if args.dry_run:
        print(json.dumps(specs, indent=2, sort_keys=True))
        return
    if not specs:
        print("No Ant v23 tasks to submit")
        return
    task_ids = _submit_jsonl(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        _dispatch(task_ids)


if __name__ == "__main__":
    main()
