#!/usr/bin/env python3
"""Batch-submit the complete BAPR-v5 capacity chain via scheduler."""
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

from jax_experiments.analysis import bapr_v5_hard_option as protocol
from jax_experiments.analysis import bapr_v3_utility_aware_router as utility


SIGNATURE_PREFIX = "BAPR/v5-hard-option/v1"
RESOURCE_FAMILY = "BAPR/v5-hard-option/gpu-runtime"
COLD_START_VRAM_MB = 2048
LINUX_CPU_NODES = ["local"]
LINUX_CPU_PYTHON = "/usr/bin/python3"
JSON_ANALYSIS_RAM_MB = 512


def command(module: str, values: list[str], *, cpu_only: bool = False,
            memory_fraction: float = 0.70,
            eta_total_units: int | None = None) -> str:
    if eta_total_units is not None and int(eta_total_units) <= 0:
        raise ValueError("eta_total_units must be positive")
    resources = (
        "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' " if cpu_only else
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        f"XLA_PYTHON_CLIENT_MEM_FRACTION={memory_fraction:.2f} "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' ")
    eta_marker = (
        f"SCHEDULEURM_ETA_TOTAL_UNITS={int(eta_total_units)} "
        if eta_total_units is not None else ""
    )
    return (
        resources
        + eta_marker
        + "OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 "
        "NUMEXPR_NUM_THREADS=4 JAX_NUM_THREADS=4 "
        "TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2 "
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        f"{shlex.quote(str(scheduler_common.JAX_PYTHON))} -u -m "
        f"{module} {shlex.join(values)} && echo DONE"
    )


def common_gpu_spec(signature: str, description: str, cmd: str,
                    output: Path, checkpoint: Path, *, vram: int,
                    ram_mb: int, priority: str,
                    resource_family: str = RESOURCE_FAMILY):
    return {
        "description": description,
        "cmd": cmd,
        "cwd": str(ROOT),
        "signature": signature,
        "vram_resource_family": resource_family,
        "project": "BAPR",
        "vram": vram,
        "ram_mb": ram_mb,
        "cpu": 4,
        "priority": priority,
        "result_dir": str(output.parent),
        "local_result_dir": str(output.parent),
        "ckpt_dir": str(checkpoint),
        "ckpt_glob": "params.pkl",
        "resume_flag": "",
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
    }


def training_spec(profile: str, priority: str):
    output = protocol.status_path(profile)
    signature = (
        f"{SIGNATURE_PREFIX}/{profile}/seed-{protocol.TRAINING_SEED}")
    spec = common_gpu_spec(
        signature,
        f"BAPR-v5 isolated hard-option {profile} training, seed "
        f"{protocol.TRAINING_SEED}",
        command(
            "jax_experiments.analysis.run_bapr_v5_hard_option",
            ["--profile", profile, "--resume"],
            memory_fraction=0.72 if profile == "formal" else 0.40,
            eta_total_units=(
                protocol.FINAL_NEXT_ITERATION if profile == "formal" else 4)),
        output, protocol.checkpoint_dir(profile),
        vram=COLD_START_VRAM_MB,
        ram_mb=32768 if profile == "formal" else 8192,
        priority=priority,
    )
    if profile == "formal":
        spec["wait_for_files"] = [
            str(protocol.status_path("smoke")),
            str(protocol.BOOTSTRAP_MODEL),
            str(protocol.BOOTSTRAP_MANIFEST),
        ]
    return signature, spec, output


def audit_spec(event_seed: int, priority: str):
    output = protocol.audit_group_path(event_seed)
    signature = f"{SIGNATURE_PREFIX}/audit/event-seed-{event_seed}"
    spec = common_gpu_spec(
        signature,
        f"Strict BAPR-v5 hard-option audit, event seed {event_seed}",
        command(
            "jax_experiments.analysis.run_bapr_v5_hard_option_audit",
            ["--event-seed", str(event_seed), "--resume"],
            memory_fraction=0.45),
        output, protocol.checkpoint_dir("formal"),
        vram=COLD_START_VRAM_MB, ram_mb=24576, priority=priority,
    )
    baseline = utility.audit_group_path(
        "validation", event_seed, "cs4d025c80h8")
    spec["wait_for_files"] = [
        str(protocol.status_path("formal")), str(baseline)]
    return signature, spec, output


def analysis_spec(priority: str):
    output = protocol.ANALYSIS_ROOT / "summary.json"
    signature = f"{SIGNATURE_PREFIX}/analysis"
    spec = {
        "description": "Aggregate BAPR-v5 hard-option capacity screen",
        "cmd": (
            f"{LINUX_CPU_PYTHON} -u -m "
            "jax_experiments.analysis.analyze_bapr_v5_hard_option "
            "&& echo DONE"
        ),
        "cwd": str(ROOT),
        "signature": signature,
        "project": "BAPR",
        "vram": 0,
        "ram_mb": JSON_ANALYSIS_RAM_MB,
        "cpu": 1,
        "priority": priority,
        "allowed_nodes": LINUX_CPU_NODES,
        "result_dir": str(protocol.ANALYSIS_ROOT),
        "local_result_dir": str(protocol.ANALYSIS_ROOT),
        "wait_for_files": [
            str(protocol.audit_group_path(seed))
            for seed in protocol.DEVELOPMENT_EVENT_SEEDS
        ] + [
            str(utility.audit_group_path(
                "validation", seed, "cs4d025c80h8"))
            for seed in protocol.DEVELOPMENT_EVENT_SEEDS
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
    return signature, spec, output


def candidates(phase: str, priority: str):
    rows = []
    if phase in ("all", "training"):
        rows.extend(training_spec(profile, priority)
                    for profile in ("smoke", "formal"))
    if phase in ("all", "audit"):
        rows.extend(audit_spec(seed, priority)
                    for seed in protocol.DEVELOPMENT_EVENT_SEEDS)
    if phase in ("all", "analysis"):
        rows.append(analysis_spec(priority))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("all", "training", "audit", "analysis"),
        default="all")
    parser.add_argument("--priority", choices=("low", "normal", "high"),
                        default="high")
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()
    known = scheduler_common.scheduler_tasks()
    specs = []
    for signature, spec, output in candidates(args.phase, args.priority):
        tasks = [task for task in known
                 if str(task.get("signature") or "") == signature]
        active = [task for task in tasks if str(task.get("status"))
                  in scheduler_common.ACTIVE_STATUSES]
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
        print("No BAPR-v5 tasks to submit")
        return
    task_ids = scheduler_common.submit_jsonl(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        scheduler_common.dispatch(task_ids)


if __name__ == "__main__":
    main()
