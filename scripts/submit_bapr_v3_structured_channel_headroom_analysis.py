#!/usr/bin/env python3
"""Submit the node-local strict structured-channel aggregate analysis."""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import submit_bapr_v3_budget_matched_fork_audit as common
from jax_experiments.analysis import (
    analyze_bapr_v3_structured_channel_headroom_audit as analysis,
)
from jax_experiments.analysis import (
    run_bapr_v3_structured_channel_headroom as protocol,
)


FAMILY = protocol.FAMILIES[0]
SIGNATURE = "BAPR/v3-structured-channel-headroom-analysis/v2"
ANALYSIS_DIR = analysis.RESULTS_ROOT / "analysis"
SUMMARY = ANALYSIS_DIR / "summary.json"
REPORT = ANALYSIS_DIR / "report.md"


def task_spec(priority: str) -> dict:
    values = [
        "--family", FAMILY,
        "--pair-root", str(protocol.SAVE_ROOT),
        "--results-root", str(analysis.RESULTS_ROOT),
        "--json-out", str(SUMMARY),
        "--report-out", str(REPORT),
    ]
    command = (
        "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 "
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        f"{shlex.quote(str(common.JAX_PYTHON))} -u -m "
        "jax_experiments.analysis."
        "analyze_bapr_v3_structured_channel_headroom_audit "
        f"{shlex.join(values)} && echo DONE"
    )
    return {
        "description": (
            "Validate and aggregate structured-channel robust/oracle audit; "
            "no rollout or training"
        ),
        "cmd": command,
        "cwd": str(ROOT),
        "signature": SIGNATURE,
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 4096,
        "cpu": 2,
        "priority": priority,
        "require_node": "jtl311linux",
        "ckpt_dir": str(protocol.SAVE_ROOT),
        "ckpt_glob": "pair_checkpoint_complete.pkl",
        "resume_flag": "",
        "skip_resume_scan": True,
        "result_dir": str(ANALYSIS_DIR),
        "local_result_dir": str(ANALYSIS_DIR),
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Validation and statistics only; no environment rollout, "
            "gradient update, or checkpoint mutation."
        ),
        "allow_initial_resume_scan_error": False,
        "reroute_on_node_down": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="high")
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = common.scheduler_tasks_strict()
    matching = [
        task for task in tasks if str(task.get("signature") or "") == SIGNATURE
    ]
    active = [
        task for task in matching
        if str(task.get("status")) in {"queued", "launching", "running"}
    ]
    complete = SUMMARY.is_file() and REPORT.is_file()
    terminal = [task for task in matching if task not in active]
    if complete:
        print("Structured-channel aggregate already complete", flush=True)
        return
    if active:
        print(
            "Structured-channel aggregate already active: "
            + ",".join(str(task["id"]) for task in active),
            flush=True,
        )
        return
    if terminal and not args.retry_incomplete:
        raise SystemExit(
            "terminal aggregate exists without valid local outputs; use "
            "--retry-incomplete after inspecting it")

    spec = task_spec(args.priority)
    print(json.dumps(spec, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return
    task_ids = common.submit_jsonl([spec])
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        common.dispatch(task_ids)


if __name__ == "__main__":
    main()
