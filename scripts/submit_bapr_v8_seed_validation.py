#!/usr/bin/env python3
"""Submit the independent-training-seed BAPR-v8 validation chain."""
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

from jax_experiments.analysis import bapr_v8_seed_validation as protocol


SIGNATURE_PREFIX = "BAPR/v8-seed-validation/v1"
COLD_START_VRAM_MB = 2048
CPU_NODES = ["local", *(f"node00{index}" for index in range(1, 7))]


def _bundle_input_paths(seed: int) -> list[str]:
    return [str(path.parent) for path in protocol.bundle_paths_for_seed(seed)]


def _gpu_command(values: list[str], memory_fraction: float = 0.22) -> str:
    return (
        f"SCHEDULEURM_ETA_TOTAL_UNITS={protocol.MAX_ITERS} "
        "BAPR_SCHEDULER_RESUME=--resume "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        f"XLA_PYTHON_CLIENT_MEM_FRACTION={memory_fraction:.2f} "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
        "OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 "
        "NUMEXPR_NUM_THREADS=2 JAX_NUM_THREADS=2 "
        "TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 "
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        f"{shlex.quote(str(scheduler_common.JAX_PYTHON))} -u -m "
        "jax_experiments.analysis.run_bapr_v8_seed_controller "
        f"{shlex.join(values)} && echo DONE"
    )


def _cpu_command(module: str, values: list[str]) -> str:
    return (
        "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' "
        "OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 "
        "NUMEXPR_NUM_THREADS=4 JAX_NUM_THREADS=4 "
        "TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2 "
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        f"{shlex.quote(str(scheduler_common.JAX_PYTHON))} -u -m "
        f"{module} {shlex.join(values)} && echo DONE"
    )


def training_signature(seed: int, role: str,
                       mode: int | None = None) -> str:
    return (
        f"{SIGNATURE_PREFIX}/train/seed-{seed}/"
        f"{protocol.role_name(role, mode)}")


def training_spec(seed: int, role: str, mode: int | None,
                  priority: str) -> dict:
    run_dir = protocol.run_dir(seed, role, mode)
    bundle = protocol.bundle_dir(seed, role, mode)
    values = ["--seed", str(seed), "--role", role, "--resume"]
    if mode is not None:
        values += ["--mode", str(mode)]
    role_name = protocol.role_name(role, mode)
    return {
        "description": (
            f"BAPR-v8 independent training seed {seed}, {role_name}"),
        "cmd": _gpu_command(
            values, memory_fraction=0.26 if role == "escp" else 0.22),
        "cwd": str(ROOT),
        "signature": training_signature(seed, role, mode),
        "project": "BAPR",
        "vram_resource_family": (
            f"BAPR/v8-seed-validation/{role_name}/gpu-runtime"),
        "vram": COLD_START_VRAM_MB,
        "ram_mb": 8192,
        "cpu": 2,
        "priority": priority,
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
        # Fail closed on checkpoint-discovery errors. A missing checkpoint is
        # a valid first launch; an unreadable checkpoint must never become a
        # silent fresh restart after migration.
        "allow_initial_resume_scan_error": False,
    }


def calibration_signature(seed: int) -> str:
    return f"{SIGNATURE_PREFIX}/calibration/seed-{seed}"


def calibration_spec(seed: int, priority: str) -> dict:
    output = protocol.calibration_path(seed)
    return {
        "description": (
            f"Calibrate frozen BAPR-v8 utility map, training seed {seed}"),
        "cmd": _cpu_command(
            "jax_experiments.analysis.run_bapr_v8_seed_audit",
            ["--phase", "calibration", "--seed", str(seed), "--resume"]),
        "cwd": str(ROOT),
        "signature": calibration_signature(seed),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 8192,
        "cpu": 4,
        "priority": priority,
        "allowed_nodes": CPU_NODES,
        "result_dir": str(output.parent),
        "local_result_dir": str(output.parent),
        "wait_for_files": [
            str(path) for path in protocol.bundle_dependencies_for_seed(seed)],
        "stage_input_paths": _bundle_input_paths(seed),
        "stage_excludes": [
            "jax_experiments/results_bapr_v8_seed_validation_runs_v1/",
            "jax_experiments/results_bapr_v8_seed_validation_audit_v1/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Deterministic evaluation/calibration only; no gradient updates."),
    }


def audit_signature(seed: int, event_seed: int) -> str:
    return (
        f"{SIGNATURE_PREFIX}/audit/seed-{seed}/event-seed-{event_seed}")


def audit_spec(seed: int, event_seed: int, priority: str) -> dict:
    output = protocol.audit_path(seed, event_seed)
    return {
        "description": (
            f"Strict BAPR-v8 paired audit, training seed {seed}, "
            f"event seed {event_seed}"),
        "cmd": _cpu_command(
            "jax_experiments.analysis.run_bapr_v8_seed_audit",
            ["--phase", "audit", "--seed", str(seed),
             "--event-seed", str(event_seed), "--resume"]),
        "cwd": str(ROOT),
        "signature": audit_signature(seed, event_seed),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 12288,
        "cpu": 4,
        "priority": priority,
        "allowed_nodes": CPU_NODES,
        "result_dir": str(output.parent),
        "local_result_dir": str(output.parent),
        "wait_for_files": [
            str(protocol.calibration_path(seed)),
            *(str(path) for path in protocol.bundle_dependencies_for_seed(seed)),
        ],
        "stage_input_paths": [
            str(protocol.calibration_path(seed).parent),
            *_bundle_input_paths(seed),
        ],
        "stage_excludes": [
            "jax_experiments/results_bapr_v8_seed_validation_runs_v1/",
            "jax_experiments/results_bapr_v8_seed_validation_audit_v1/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Strict deterministic policy evaluation only; no training."),
    }


def analysis_signature() -> str:
    return f"{SIGNATURE_PREFIX}/analysis"


def analysis_spec(priority: str) -> dict:
    output = protocol.analysis_path()
    return {
        "description": "Aggregate BAPR-v8 independent-training-seed result",
        "cmd": _cpu_command(
            "jax_experiments.analysis.analyze_bapr_v8_seed_validation", []),
        "cwd": str(ROOT),
        "signature": analysis_signature(),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 1024,
        "cpu": 1,
        "priority": priority,
        "allowed_nodes": CPU_NODES,
        "result_dir": str(output.parent),
        "local_result_dir": str(output.parent),
        "wait_for_files": [
            str(protocol.audit_path(seed, event_seed))
            for seed in protocol.TRAINING_SEEDS
            for event_seed in protocol.EVALUATION_EVENT_SEEDS
        ],
        "stage_excludes": [
            "jax_experiments/results_bapr_v8_seed_validation_runs_v1/",
            "jax_experiments/eval_bundles_bapr_v8_seed_validation_v1/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": "JSON aggregation only.",
    }


def candidates(phase: str, priority: str):
    rows = []
    if phase in ("all", "training"):
        for seed in protocol.TRAINING_SEEDS:
            rows.extend([
                (training_signature(seed, role),
                 training_spec(seed, role, None, priority),
                 protocol.bundle_manifest(seed, role))
                for role in ("sac", "escp", "resac")
            ])
            rows.extend([
                (training_signature(seed, "specialist", mode),
                 training_spec(seed, "specialist", mode, priority),
                 protocol.bundle_manifest(seed, "specialist", mode))
                for mode in protocol.MODES
            ])
    if phase in ("all", "calibration"):
        rows.extend([
            (calibration_signature(seed), calibration_spec(seed, priority),
             protocol.calibration_path(seed))
            for seed in protocol.TRAINING_SEEDS
        ])
    if phase in ("all", "audit"):
        rows.extend([
            (audit_signature(seed, event_seed),
             audit_spec(seed, event_seed, priority),
             protocol.audit_path(seed, event_seed))
            for seed in protocol.TRAINING_SEEDS
            for event_seed in protocol.EVALUATION_EVENT_SEEDS
        ])
    if phase in ("all", "analysis"):
        rows.append((analysis_signature(), analysis_spec(priority),
                     protocol.analysis_path()))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("all", "training", "calibration", "audit",
                             "analysis"), default="all")
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="high")
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

    known = scheduler_common.scheduler_tasks()
    specs = []
    for signature, spec, output in candidates(args.phase, args.priority):
        tasks = [task for task in known
                 if str(task.get("signature") or "") == signature]
        active = [task for task in tasks
                  if str(task.get("status")) in scheduler_common.ACTIVE_STATUSES]
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
        print("No BAPR-v8 seed-validation tasks to submit")
        return
    ids = scheduler_common.submit_jsonl(specs)
    print(f"Submitted task ids: {','.join(ids)}", flush=True)
    if args.dispatch:
        scheduler_common.dispatch(ids)


if __name__ == "__main__":
    main()
