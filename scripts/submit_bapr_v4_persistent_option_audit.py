#!/usr/bin/env python3
"""Submit BAPR-v4 strict audits and analysis through scheduler only."""
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

from jax_experiments.analysis import bapr_v4_persistent_option as protocol
from jax_experiments.analysis import (
    bapr_v3_utility_aware_router as utility,
)


SIGNATURE_PREFIX = "BAPR/v4-persistent-option/v1"


def command(module: str, values: list[str], *, cpu_only: bool) -> str:
    resources = (
        "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' " if cpu_only else
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.24 "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' ")
    return (
        resources
        + "OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 "
        "NUMEXPR_NUM_THREADS=2 JAX_NUM_THREADS=2 "
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        f"{shlex.quote(str(scheduler_common.JAX_PYTHON))} -u -m "
        f"{module} {shlex.join(values)} && echo DONE"
    )


def audit_spec(event_seed: int, priority: str):
    output = protocol.audit_group_path(event_seed)
    signature = f"{SIGNATURE_PREFIX}/audit/event-seed-{event_seed}"
    baseline = utility.audit_group_path(
        "validation", event_seed, "cs4d025c80h8")
    spec = {
        "description": (
            f"Strict BAPR-v4 persistent-option audit, event seed {event_seed}"),
        "cmd": command(
            "jax_experiments.analysis."
            "run_bapr_v4_persistent_option_audit",
            ["--event-seed", str(event_seed), "--resume"], cpu_only=False),
        "cwd": str(ROOT),
        "signature": signature,
        "project": "BAPR",
        "vram": 1800,
        "ram_mb": 12288,
        "cpu": 2,
        "priority": priority,
        "require_node": "jtl311linux",
        "result_dir": str(output.parent),
        "local_result_dir": str(output.parent),
        "ckpt_dir": str(protocol.checkpoint_dir("formal")),
        "ckpt_glob": "params.pkl",
        "resume_flag": "",
        "wait_for_files": [
            str(protocol.status_path("formal")), str(baseline)],
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/", "paper/"],
        "reroute_on_node_down": False,
        "allow_initial_resume_scan_error": False,
    }
    return signature, spec, output


def analysis_spec(priority: str):
    output = protocol.ANALYSIS_ROOT / "summary.json"
    signature = f"{SIGNATURE_PREFIX}/analysis"
    waits = [str(protocol.audit_group_path(seed))
             for seed in protocol.DEVELOPMENT_EVENT_SEEDS]
    spec = {
        "description": "Aggregate BAPR-v4 persistent-option capacity screen",
        "cmd": command(
            "jax_experiments.analysis."
            "analyze_bapr_v4_persistent_option", [], cpu_only=True),
        "cwd": str(ROOT),
        "signature": signature,
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 4096,
        "cpu": 1,
        "priority": priority,
        "require_node": "jtl311linux",
        "result_dir": str(protocol.ANALYSIS_ROOT),
        "local_result_dir": str(protocol.ANALYSIS_ROOT),
        "wait_for_files": waits,
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/", "paper/"],
        "reroute_on_node_down": False,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": "JSON aggregation only.",
    }
    return signature, spec, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("audit", "analysis"),
                        required=True)
    parser.add_argument("--priority", choices=("low", "normal", "high"),
                        default="high")
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()
    candidates = (
        [audit_spec(seed, args.priority)
         for seed in protocol.DEVELOPMENT_EVENT_SEEDS]
        if args.phase == "audit" else [analysis_spec(args.priority)])
    known = scheduler_common.scheduler_tasks()
    specs = []
    for signature, spec, output in candidates:
        tasks = [task for task in known
                 if str(task.get("signature") or "") == signature]
        active = [task for task in tasks if str(task.get("status"))
                  in scheduler_common.ACTIVE_STATUSES]
        if output.is_file():
            print(f"skip complete-output: {signature}")
        elif active:
            print("skip active: " + ",".join(str(task["id"])
                                               for task in active))
        elif tasks and not args.retry_incomplete:
            print("skip terminal-incomplete: " + ",".join(
                str(task["id"]) for task in tasks))
        else:
            specs.append(spec)
    if args.dry_run:
        print(json.dumps(specs, indent=2, sort_keys=True))
        return
    if not specs:
        print("No BAPR-v4 audit tasks to submit")
        return
    task_ids = scheduler_common.submit_jsonl(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        scheduler_common.dispatch(task_ids)


if __name__ == "__main__":
    main()
