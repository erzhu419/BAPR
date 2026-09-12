#!/usr/bin/env python3
"""Submit the equal-per-controller independent-adapter diagnostic."""
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

import submit_regime_adapter_fork as development_submit
from jax_experiments.analysis import regime_adapter_equal_controller as protocol
from jax_experiments.analysis import regime_adapter_fork as branch_protocol


SIGNATURE_PREFIX = "BAPR/regime-adapter-equal-controller/v1"
CPU_NODES = [f"node00{index}" for index in range(1, 7)]
GPU_NODES = [
    "local", "jtl110gpu", "jtl110gpu2", "node007"]
# The identical adapter architecture peaked at 2.09 GiB in the completed
# confirmation. This is a measured admission margin.
MEASURED_VRAM_MB = 2300


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
        "run_regime_adapter_equal_controller_branch "
        f"{shlex.join(values)} && echo DONE"
    )


def _source_wait_paths(seed: int) -> list[str]:
    directory = branch_protocol.source_bundle_dir(seed)
    return [str(path) for path in (
        directory / branch_protocol.BUNDLE_MANIFEST_NAME,
        directory / "checkpoints" / "params.pkl",
        directory / "checkpoints" / "train_state.pkl",
        directory / "logs" / "protocol_signature.json",
    )]


def training_spec(seed: int, mode: int, priority: str) -> dict:
    source_dir = branch_protocol.source_bundle_dir(seed)
    run_dir = protocol.run_dir(seed, mode)
    bundle_dir = protocol.adapter_bundle_dir(
        seed, protocol.DELTA, mode)
    return {
        "description": (
            f"Equal-controller adapter seed {seed} mode {mode}"),
        "cmd": _gpu_command(seed, mode),
        "cwd": str(ROOT),
        "signature": training_signature(seed, mode),
        "project": "BAPR",
        "vram_resource_family": (
            "BAPR/regime-adapter-equal-controller/branch/gpu"),
        "vram": MEASURED_VRAM_MB,
        "ram_mb": 8192,
        "cpu": 2,
        "priority": priority,
        "allowed_nodes": GPU_NODES,
        # Strict publication validates both checkpoint state and the training
        # logs. Migrate the complete run on retry so protocol_signature.json
        # and mode_id.npy cannot be stranded on the source node.
        "ckpt_dir": str(run_dir),
        "ckpt_glob": "checkpoints/train_state.pkl",
        "resume_flag": "",
        "resume_managed_by_cmd": True,
        "result_dir": str(bundle_dir),
        "local_result_dir": str(bundle_dir),
        "wait_for_files": _source_wait_paths(seed),
        "stage_input_paths": [str(source_dir)],
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
    }


def _audit_wait_paths(seed: int) -> list[str]:
    robust = branch_protocol.robust_bundle_dir(seed)
    return [
        *(str(path) for path in branch_protocol.required_bundle_paths(robust)),
        *(str(path) for path in protocol.required_bundle_paths(seed)),
    ]


def audit_spec(seed: int, event_seed: int, priority: str) -> dict:
    output = protocol.audit_dir(seed, event_seed)
    stage_inputs = [
        str(branch_protocol.robust_bundle_dir(seed)),
        *(str(protocol.adapter_bundle_dir(
            seed, protocol.DELTA, mode)) for mode in protocol.MODES),
    ]
    return {
        "description": (
            f"Equal-controller strict audit seed {seed} event {event_seed}"),
        "cmd": development_submit._cpu_command(
            "jax_experiments.analysis."
            "run_regime_adapter_equal_controller_audit",
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
        "description": "Aggregate equal-controller adapter upper bound",
        "cmd": development_submit._cpu_command(
            "jax_experiments.analysis."
            "analyze_regime_adapter_equal_controller",
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
        "--phase", choices=("all", "train", "audit", "analysis"),
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
        print("No equal-controller tasks to submit")
        return
    task_ids = development_submit._submit_jsonl(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        training_ids = [
            task_id for task_id, spec in zip(task_ids, specs)
            if "/train/" in str(spec["signature"])
        ]
        development_submit._dispatch(training_ids)


if __name__ == "__main__":
    main()
