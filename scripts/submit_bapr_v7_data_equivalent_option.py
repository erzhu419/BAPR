#!/usr/bin/env python3
"""Submit the BAPR-v7 unique-data-equivalent capacity chain via scheduler."""
from __future__ import annotations

import argparse
import json
import os
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

import submit_bapr_v5_hard_option as v5_submit
from jax_experiments.analysis import bapr_v3_utility_aware_router as utility
from jax_experiments.analysis import bapr_v7_data_equivalent_option as protocol


SIGNATURE_PREFIX = "BAPR/v7-data-equivalent-option/v1"
RESOURCE_FAMILY = "BAPR/v7-data-equivalent-option/gpu-runtime"


def training_spec(profile: str, priority: str):
    output = protocol.status_path(profile)
    signature = f"{SIGNATURE_PREFIX}/{profile}/seed-{protocol.TRAINING_SEED}"
    spec = v5_submit.common_gpu_spec(
        signature,
        f"BAPR-v7 data-equivalent option {profile} training, seed "
        f"{protocol.TRAINING_SEED}",
        v5_submit.command(
            "jax_experiments.analysis.run_bapr_v7_data_equivalent_option",
            ["--profile", profile, "--resume"],
            memory_fraction=0.76 if profile == "formal" else 0.50,
            eta_total_units=(
                protocol.FINAL_NEXT_ITERATION if profile == "formal" else 4)),
        output,
        protocol.checkpoint_dir(profile),
        vram=v5_submit.COLD_START_VRAM_MB,
        ram_mb=12288 if profile == "formal" else 8192,
        priority=priority,
        resource_family=RESOURCE_FAMILY,
    )
    spec["wait_for_files"] = [
        str(protocol.BOOTSTRAP_MODEL),
        str(protocol.BOOTSTRAP_MANIFEST),
    ]
    if profile == "formal":
        spec["wait_for_files"].insert(0, str(protocol.status_path("smoke")))
    return signature, spec, output


def audit_spec(event_seed: int, priority: str):
    output = protocol.audit_group_path(event_seed)
    signature = f"{SIGNATURE_PREFIX}/audit/event-seed-{event_seed}"
    spec = v5_submit.common_gpu_spec(
        signature,
        f"Strict BAPR-v7 data-equivalent audit, event seed {event_seed}",
        v5_submit.command(
            "jax_experiments.analysis.run_bapr_v7_data_equivalent_option_audit",
            ["--event-seed", str(event_seed), "--resume"],
            memory_fraction=0.45),
        output,
        protocol.checkpoint_dir("formal"),
        vram=v5_submit.COLD_START_VRAM_MB,
        ram_mb=8192,
        priority=priority,
        resource_family=RESOURCE_FAMILY,
    )
    baseline = utility.audit_group_path(
        "validation", event_seed, "cs4d025c80h8")
    spec["wait_for_files"] = [
        str(protocol.status_path("formal")), str(baseline)]
    return signature, spec, output


def analysis_spec(priority: str):
    output = protocol.ANALYSIS_ROOT / "summary.json"
    signature = f"{SIGNATURE_PREFIX}/analysis"
    baseline_files = [
        utility.audit_group_path("validation", seed, "cs4d025c80h8")
        for seed in protocol.DEVELOPMENT_EVENT_SEEDS
    ]
    spec = {
        "description": "Aggregate BAPR-v7 data-equivalent capacity screen",
        "cmd": (
            f"{v5_submit.LINUX_CPU_PYTHON} -u -m "
            "jax_experiments.analysis.analyze_bapr_v7_data_equivalent_option "
            "&& echo DONE"
        ),
        "cwd": str(ROOT),
        "signature": signature,
        "project": "BAPR",
        "vram": 0,
        "ram_mb": v5_submit.JSON_ANALYSIS_RAM_MB,
        "cpu": 1,
        "priority": priority,
        "allowed_nodes": v5_submit.LINUX_CPU_NODES,
        "result_dir": str(protocol.ANALYSIS_ROOT),
        "local_result_dir": str(protocol.ANALYSIS_ROOT),
        "wait_for_files": [
            str(protocol.audit_group_path(seed))
            for seed in protocol.DEVELOPMENT_EVENT_SEEDS
        ] + [str(path) for path in baseline_files],
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
        active = [task for task in tasks if str(task.get("status"))
                  in scheduler_common.ACTIVE_STATUSES]
        if output.is_file():
            print(f"skip complete-output: {signature}")
        elif active:
            print("skip active: " + ",".join(str(task["id"]) for task in active))
        elif tasks and not args.retry_incomplete:
            print("skip terminal-incomplete: " + ",".join(
                str(task["id"]) for task in tasks))
        else:
            specs.append(spec)
    if args.dry_run:
        print(json.dumps(specs, indent=2, sort_keys=True))
        return
    if not specs:
        print("No BAPR-v7 tasks to submit")
        return
    task_ids = scheduler_common.submit_jsonl(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        scheduler_common.dispatch(task_ids)


if __name__ == "__main__":
    main()
