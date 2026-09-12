#!/usr/bin/env python3
"""Submit the frozen-config multi-seed regime-adapter confirmation."""
from __future__ import annotations

import argparse
from copy import deepcopy
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

import submit_regime_adapter_fork as development_submit
from jax_experiments.analysis import regime_adapter_confirmation as protocol
from jax_experiments.analysis import regime_adapter_fork as branch_protocol


SIGNATURE_PREFIX = "BAPR/regime-adapter-confirmation/v1"
CPU_NODES = [f"node00{index}" for index in range(1, 7)]
# The completed development screen peaked at 2.09 GiB. This is a measured
# 10% admission margin, not a cold-start 4/8 GiB guess.
MEASURED_VRAM_MB = 2300


def robust_signature(seed: int) -> str:
    return f"{SIGNATURE_PREFIX}/train/seed-{seed}/robust-continue"


def adapter_signature(seed: int, mode: int) -> str:
    return (
        f"{SIGNATURE_PREFIX}/train/seed-{seed}/"
        f"{branch_protocol.delta_slug(protocol.DELTA)}/mode-{mode}")


def audit_signature(seed: int, event_seed: int) -> str:
    return (
        f"{SIGNATURE_PREFIX}/audit/seed-{seed}/event-{event_seed}")


def analysis_signature() -> str:
    return f"{SIGNATURE_PREFIX}/analysis"


def _confirmation_training_spec(spec: dict, signature: str,
                                description: str) -> dict:
    spec = deepcopy(spec)
    spec.update({
        "description": description,
        "signature": signature,
        "vram_resource_family": (
            "BAPR/regime-adapter-confirmation/branch/gpu"),
        "vram": MEASURED_VRAM_MB,
    })
    # Placement remains portable. Checkpoint directories and resume behavior
    # come unchanged from the validated development submitter.
    spec.pop("preferred_node", None)
    spec.pop("require_node", None)
    spec.pop("require_gpu_idx", None)
    spec.pop("allowed_nodes", None)
    return spec


def robust_spec(seed: int, priority: str) -> dict:
    return _confirmation_training_spec(
        development_submit.robust_spec(seed, priority),
        robust_signature(seed),
        f"Regime adapter confirmation robust continuation seed {seed}",
    )


def adapter_spec(seed: int, mode: int, priority: str) -> dict:
    return _confirmation_training_spec(
        development_submit.adapter_spec(
            seed, protocol.DELTA, mode, priority),
        adapter_signature(seed, mode),
        f"Regime adapter confirmation frozen branch seed {seed} mode {mode}",
    )


def audit_spec(seed: int, event_seed: int, priority: str) -> dict:
    output = protocol.audit_dir(seed, event_seed)
    bundle_dirs = protocol.training_bundle_dirs(seed)
    return {
        "description": (
            f"Regime adapter confirmation audit seed {seed} "
            f"event {event_seed}"),
        "cmd": development_submit._cpu_command(
            "jax_experiments.analysis."
            "run_regime_adapter_confirmation_audit",
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
        "wait_for_files": [
            str(path) for path in protocol.required_training_paths(seed)
        ],
        "stage_input_paths": [str(path) for path in bundle_dirs],
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Checkpoint-only sealed policy evaluation; no updates."),
    }


def analysis_spec(priority: str) -> dict:
    return {
        "description": "Aggregate regime adapter multi-seed confirmation",
        "cmd": development_submit._cpu_command(
            "jax_experiments.analysis."
            "analyze_regime_adapter_confirmation",
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
            robust_dir = branch_protocol.robust_bundle_dir(seed)
            rows.append((
                robust_signature(seed), robust_spec(seed, priority),
                branch_protocol.bundle_manifest(robust_dir),
            ))
            for mode in protocol.MODES:
                adapter_dir = branch_protocol.adapter_bundle_dir(
                    seed, protocol.DELTA, mode)
                rows.append((
                    adapter_signature(seed, mode),
                    adapter_spec(seed, mode, priority),
                    branch_protocol.bundle_manifest(adapter_dir),
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
        print("No regime-adapter confirmation tasks to submit")
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
