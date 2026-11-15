#!/usr/bin/env python3
"""Submit the late-base min-target adapter diagnostic."""
from __future__ import annotations

import argparse
import json
import os
import shlex
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

import submit_regime_adapter_fork as development_submit
from jax_experiments.analysis import regime_adapter_fork as source_protocol
from jax_experiments.analysis import regime_adapter_latebase_min as protocol


SIGNATURE_PREFIX = "BAPR/regime-adapter-latebase-min/v1"
CPU_NODES = [f"node00{index}" for index in range(1, 7)]
GPU_NODES = ["local", "jtl110gpu", "jtl110gpu2", "node007"]
MEASURED_VRAM_MB = 2300


def prepare_signature(seed: int) -> str:
    return f"{SIGNATURE_PREFIX}/prepare/seed-{seed}"


def training_signature(seed: int, mode: int) -> str:
    return f"{SIGNATURE_PREFIX}/train/seed-{seed}/mode-{mode}"


def audit_signature(seed: int, event_seed: int) -> str:
    return f"{SIGNATURE_PREFIX}/audit/seed-{seed}/event-{event_seed}"


def analysis_signature() -> str:
    return f"{SIGNATURE_PREFIX}/analysis"


def _gpu_command(seed: int, mode: int) -> str:
    values = [
        "--seed", str(seed), "--mode", str(mode), "--resume",
    ]
    return (
        f"SCHEDULEURM_ETA_TOTAL_UNITS={protocol.FINAL_NEXT_ITERATION} "
        "JAX_PLATFORMS=cuda "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.28 "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false "
        "--xla_cpu_multi_thread_eigen=false "
        "intra_op_parallelism_threads=1' "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 JAX_NUM_THREADS=1 "
        "TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 "
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        f"{shlex.quote(str(scheduler_common.JAX_PYTHON))} -u -m "
        "jax_experiments.analysis."
        "run_regime_adapter_latebase_min_branch "
        f"{shlex.join(values)} && echo DONE"
    )


def _source_wait_paths(seed: int) -> list[str]:
    return [
        str(path) for path in source_protocol.required_bundle_paths(
            protocol.source_bundle_dir(seed))
    ]


def prepare_spec(seed: int, priority: str) -> dict:
    output = protocol.canonical_dir(seed)
    return {
        "description": f"Prepare canonical late-base adapter seed {seed}",
        "cmd": development_submit._cpu_command(
            "jax_experiments.analysis."
            "run_regime_adapter_latebase_min_prepare",
            ["--seed", str(seed), "--resume"],
        ),
        "cwd": str(ROOT),
        "signature": prepare_signature(seed),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 8192,
        "cpu": 4,
        "priority": priority,
        "allowed_nodes": CPU_NODES,
        "result_dir": str(output),
        "local_result_dir": str(output),
        "wait_for_files": _source_wait_paths(seed),
        "stage_input_paths": [str(protocol.source_bundle_dir(seed))],
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Checkpoint conversion and exact parameter-copy audit only; "
            "no environment rollout or gradient update."),
    }


def training_spec(seed: int, mode: int, priority: str) -> dict:
    run_dir = protocol.run_dir(seed, mode)
    bundle_dir = protocol.adapter_bundle_dir(
        seed, protocol.DELTA, mode)
    return {
        "description": (
            f"Late-base min-target adapter seed {seed} mode {mode}"),
        "cmd": _gpu_command(seed, mode),
        "cwd": str(ROOT),
        "signature": training_signature(seed, mode),
        "project": "BAPR",
        "vram_resource_family": (
            "BAPR/regime-adapter-latebase-min/branch/gpu"),
        "vram": MEASURED_VRAM_MB,
        "ram_mb": 8192,
        "cpu": 2,
        "priority": priority,
        "allowed_nodes": GPU_NODES,
        "ckpt_dir": str(run_dir),
        "ckpt_glob": "checkpoints/train_state.pkl",
        "resume_flag": "",
        "resume_managed_by_cmd": True,
        "result_dir": str(bundle_dir),
        "local_result_dir": str(bundle_dir),
        "wait_for_files": [
            str(path) for path in protocol.required_canonical_paths(seed)
        ],
        "stage_input_paths": [str(protocol.canonical_dir(seed))],
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
    }


def _audit_wait_paths(seed: int) -> list[str]:
    return [
        *_source_wait_paths(seed),
        *(str(path) for path in protocol.required_bundle_paths(seed)),
    ]


def audit_spec(seed: int, event_seed: int, priority: str) -> dict:
    output = protocol.audit_dir(seed, event_seed)
    stage_inputs = [
        str(protocol.source_bundle_dir(seed)),
        *(str(protocol.adapter_bundle_dir(
            seed, protocol.DELTA, mode)) for mode in protocol.MODES),
    ]
    return {
        "description": (
            f"Late-base strict audit seed {seed} event {event_seed}"),
        "cmd": development_submit._cpu_command(
            "jax_experiments.analysis."
            "run_regime_adapter_latebase_min_audit",
            ["--seed", str(seed), "--event-seed", str(event_seed),
             "--resume"],
        ),
        "cwd": str(ROOT),
        "signature": audit_signature(seed, event_seed),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 12288,
        "cpu": 32,
        "priority": priority,
        "allowed_nodes": CPU_NODES,
        "result_dir": str(output),
        "local_result_dir": str(output),
        "wait_for_files": _audit_wait_paths(seed),
        "stage_input_paths": stage_inputs,
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Checkpoint-only strict-horizon policy evaluation; no updates."),
    }


def analysis_spec(priority: str) -> dict:
    return {
        "description": "Aggregate late-base min-target diagnostic",
        "cmd": development_submit._cpu_command(
            "jax_experiments.analysis."
            "analyze_regime_adapter_latebase_min",
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
            str(protocol.audit_manifest(seed, event_seed))
            for seed in protocol.TRAINING_SEEDS
            for event_seed in protocol.EVENT_SEEDS
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
        "cpu_training_justification": "CSV/JSON aggregation only.",
    }


def candidates(phase: str, priority: str):
    rows = []
    if phase in ("all", "prepare"):
        for seed in protocol.TRAINING_SEEDS:
            rows.append((
                prepare_signature(seed),
                prepare_spec(seed, priority),
                protocol.canonical_manifest(seed),
            ))
    if phase in ("all", "train"):
        for seed in protocol.TRAINING_SEEDS:
            for mode in protocol.MODES:
                rows.append((
                    training_signature(seed, mode),
                    training_spec(seed, mode, priority),
                    protocol.bundle_manifest(seed, mode),
                ))
    if phase in ("all", "audit"):
        for seed in protocol.TRAINING_SEEDS:
            for event_seed in protocol.EVENT_SEEDS:
                rows.append((
                    audit_signature(seed, event_seed),
                    audit_spec(seed, event_seed, priority),
                    protocol.audit_manifest(seed, event_seed),
                ))
    if phase in ("all", "analysis"):
        rows.append((
            analysis_signature(), analysis_spec(priority),
            protocol.analysis_json(),
        ))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("all", "prepare", "train", "audit", "analysis"),
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
            if str(task.get("signature") or "") == signature
        ]
        active = [
            task for task in tasks
            if str(task.get("status")) in scheduler_common.ACTIVE_STATUSES
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
        print("No late-base tasks to submit")
        return
    task_ids = development_submit._submit_jsonl(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        development_submit._dispatch(task_ids)


if __name__ == "__main__":
    main()
