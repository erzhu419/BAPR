#!/usr/bin/env python3
"""Submit the v4 expected-action system-ID ablation through scheduleurm."""
from __future__ import annotations

import argparse
import json
import os
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

import submit_regime_polarity_inverse_system_id as v3_submit
from jax_experiments.analysis import regime_polarity_confirmation as confirmation
from jax_experiments.analysis import regime_polarity_headroom as exploratory
from jax_experiments.analysis import (
    regime_polarity_expected_action_system_id as protocol,
)


SIGNATURE_PREFIX = "BAPR/regime-polarity-expected-action-system-id/v4"
ESTIMATED_VRAM_MB = 2500
GPU_NODES = ["local", "jtl110gpu", "node007"]
CPU_NODES = [f"node00{index}" for index in range(1, 7)]
ACTIVE_STATUSES = scheduler_common.ACTIVE_STATUSES
_gpu_command = v3_submit._gpu_command
_cpu_command = v3_submit._cpu_command


def training_signature():
    return f"{SIGNATURE_PREFIX}/train"


def training_spec(priority):
    bundles = [
        exploratory.bundle_dir(protocol.ENV, role, seed)
        for seed in (
            *protocol.TRAIN_CONTROLLER_SEEDS,
            *protocol.VALIDATION_CONTROLLER_SEEDS,
        )
        for role in protocol.ROLES
    ]
    return {
        "description": "Fit polarity expected-action inverse system ID",
        "cmd": _gpu_command(
            "jax_experiments.analysis."
            "train_regime_polarity_expected_action_system_id",
            ["--resume"],
        ),
        "cwd": str(ROOT),
        "signature": training_signature(),
        "project": "BAPR",
        "vram_resource_family": (
            "BAPR/regime-polarity/expected-action-system-id/gpu-runtime"),
        "vram": ESTIMATED_VRAM_MB,
        "ram_mb": 12288,
        "cpu": 2,
        "priority": priority,
        "allowed_nodes": GPU_NODES,
        "result_dir": str(protocol.MODEL_ROOT),
        "local_result_dir": str(protocol.MODEL_ROOT),
        "wait_for_files": [
            str(path)
            for seed in (
                *protocol.TRAIN_CONTROLLER_SEEDS,
                *protocol.VALIDATION_CONTROLLER_SEEDS,
            )
            for role in protocol.ROLES
            for path in exploratory.bundle_required_paths(
                protocol.ENV, role, seed)
        ],
        "stage_input_paths": [str(path) for path in bundles],
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
    }


def audit_signature(seed):
    return f"{SIGNATURE_PREFIX}/audit/seed-{int(seed)}"


def audit_spec(seed, priority):
    output = protocol.audit_dir(seed)
    bundles = [
        confirmation.bundle_dir(protocol.ENV, role, seed)
        for role in protocol.ROLES
    ]
    return {
        "description": f"Audit expected-action system ID seed {seed}",
        "cmd": _cpu_command(
            "jax_experiments.analysis."
            "run_regime_polarity_expected_action_system_id_audit",
            ["--seed", str(seed), "--resume"],
        ),
        "cwd": str(ROOT),
        "signature": audit_signature(seed),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 10240,
        "cpu": 8,
        "priority": priority,
        "allowed_nodes": CPU_NODES,
        "result_dir": str(output),
        "local_result_dir": str(output),
        "wait_for_files": [
            str(protocol.MODEL_MANIFEST),
            str(protocol.MODEL_PATH),
            *[
                str(path)
                for role in protocol.ROLES
                for path in confirmation.bundle_required_paths(
                    protocol.ENV, role, seed)
            ],
        ],
        "stage_input_paths": [
            str(protocol.MODEL_ROOT),
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
            "Frozen deterministic controller and expected-action estimator "
            "evaluation; no parameter updates."),
    }


def analysis_signature():
    return f"{SIGNATURE_PREFIX}/analysis"


def analysis_spec(priority):
    return {
        "description": "Aggregate expected-action system-ID ablation",
        "cmd": _cpu_command(
            "jax_experiments.analysis."
            "analyze_regime_polarity_expected_action_system_id",
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
            str(protocol.audit_manifest(seed))
            for seed in protocol.TEST_CONTROLLER_SEEDS
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
        "cpu_training_justification": "JSON aggregation only.",
    }


def candidates(phase, priority):
    rows = []
    if phase in ("all", "training"):
        rows.append((
            training_signature(),
            training_spec(priority),
            protocol.MODEL_MANIFEST,
        ))
    if phase in ("all", "audit"):
        rows.extend([
            (
                audit_signature(seed),
                audit_spec(seed, priority),
                protocol.audit_manifest(seed),
            )
            for seed in protocol.TEST_CONTROLLER_SEEDS
        ])
    if phase in ("all", "analysis"):
        rows.append((
            analysis_signature(),
            analysis_spec(priority),
            protocol.analysis_json(),
        ))
    return rows


def _submit_jsonl(specs):
    payload = "".join(json.dumps(spec) + "\n" for spec in specs)
    command = [
        sys.executable,
        str(scheduler_common.SCHEDULER),
        "submit-jsonl",
        "--stdin",
        "--trusted",
        "--json",
        "--intent-label",
        "bapr-polarity-expected-action-v4-submit",
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
    if (len(task_ids) != len(specs)
            or any(not task_id for task_id in task_ids)
            or len(set(task_ids)) != len(task_ids)):
        raise RuntimeError(
            "scheduler submission was not all-or-verifiably-accounted: "
            f"requested={len(specs)}, returned={task_ids}")
    return task_ids


def _dispatch(task_ids):
    if not task_ids:
        return
    command = [
        sys.executable,
        str(scheduler_common.SCHEDULER),
        "dispatch",
        "--bulk-window",
        "--intent-label",
        "bapr-polarity-expected-action-v4-dispatch",
        "--intent-ttl",
        "900",
    ]
    for task_id in task_ids:
        command += ["--task-id", task_id]
    subprocess.run(
        command, check=True, env=scheduler_common.scheduler_env())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("all", "training", "audit", "analysis"),
        default="all",
    )
    parser.add_argument(
        "--priority",
        choices=("low", "normal", "high"),
        default="high",
    )
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
        print("No expected-action system-ID tasks to submit")
        return
    task_ids = _submit_jsonl(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        _dispatch(task_ids)


if __name__ == "__main__":
    main()
