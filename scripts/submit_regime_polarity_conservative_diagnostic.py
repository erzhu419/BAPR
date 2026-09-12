#!/usr/bin/env python3
"""Submit the one-iteration conservative residual rejection diagnostic."""
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
        [
            str(scheduler_common.JAX_PYTHON),
            str(Path(__file__).resolve()),
            *sys.argv[1:],
        ],
    )

import submit_regime_polarity_frozen_anchor as scheduler_format
from jax_experiments.analysis import (
    diagnose_regime_polarity_conservative_residual as diagnostic,
)
from jax_experiments.analysis import (
    regime_polarity_conservative_residual as protocol,
)


SIGNATURE_PREFIX = (
    "BAPR/regime-polarity-conservative-residual/diagnostic-v2")
MODULE = (
    "jax_experiments.analysis."
    "diagnose_regime_polarity_conservative_residual")
SEED = 1103
GPU_NODES = ["local", "jtl110gpu", "node007"]
ACTIVE_STATUSES = scheduler_common.ACTIVE_STATUSES


def signature(variant: str) -> str:
    return f"{SIGNATURE_PREFIX}/{protocol.require_variant(variant)}/seed-{SEED}"


def spec(variant: str, priority: str) -> dict:
    variant = protocol.require_variant(variant)
    run_dir = diagnostic.run_dir(variant, SEED)
    output = diagnostic.output_dir(variant, SEED)
    source = protocol.source_bundle_dir(SEED)
    return {
        "description": (
            f"Conservative rejection diagnostic {variant} seed {SEED}"),
        "cmd": (
            f"SCHEDULEURM_ETA_TOTAL_UNITS="
            f"{protocol.SOURCE_NEXT_ITERATION + diagnostic.DIAGNOSTIC_ITERS} "
            + scheduler_format._gpu_command(
                MODULE,
                [
                    "--variant",
                    variant,
                    "--seed",
                    str(SEED),
                    "--resume",
                ],
                0.42,
            )
        ),
        "cwd": str(ROOT),
        "signature": signature(variant),
        "project": "BAPR",
        "vram_resource_family": (
            "BAPR/regime-polarity-conservative-diagnostic/gpu-runtime"),
        "vram": 2800,
        "ram_mb": 10240,
        "cpu": 2,
        "priority": priority,
        "allowed_nodes": GPU_NODES,
        "ckpt_dir": str(run_dir / "checkpoints"),
        "ckpt_glob": "train_state.pkl",
        "resume_flag": "",
        "resume_managed_by_cmd": True,
        "result_dir": str(output),
        "local_result_dir": str(output),
        "wait_for_files": [
            str(path) for path in protocol.source_required_paths(SEED)
        ],
        "stage_input_paths": [str(source)],
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
    }


def candidates(priority: str):
    return [
        (
            signature(variant),
            spec(variant, priority),
            diagnostic.output_path(variant, SEED),
        )
        for variant in protocol.VARIANTS
    ]


def _submit(specs: list[dict]) -> list[str]:
    payload = "".join(json.dumps(item) + "\n" for item in specs)
    command = [
        sys.executable,
        str(scheduler_common.SCHEDULER),
        "submit-jsonl",
        "--stdin",
        "--trusted",
        "--json",
        "--intent-label",
        "bapr-conservative-rejection-diagnostic-v2-submit",
        "--intent-ttl",
        "900",
    ]
    result = subprocess.run(
        command,
        input=payload,
        text=True,
        capture_output=True,
        env=scheduler_common.scheduler_env(),
        check=True,
    )
    response = json.loads(result.stdout)
    print(json.dumps(response, indent=2))
    task_ids = [
        str(item.get("id") or "") for item in response.get("submitted", [])]
    if len(task_ids) != len(specs) or any(not item for item in task_ids):
        raise RuntimeError(
            f"scheduler did not account for every diagnostic: {task_ids}")
    return task_ids


def _dispatch(task_ids: list[str]) -> None:
    command = [
        sys.executable,
        str(scheduler_common.SCHEDULER),
        "dispatch",
        "--bulk-window",
        "--intent-label",
        "bapr-conservative-rejection-diagnostic-v2-dispatch",
        "--intent-ttl",
        "900",
    ]
    for task_id in task_ids:
        command += ["--task-id", task_id]
    subprocess.run(
        command,
        check=True,
        env=scheduler_common.scheduler_env(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="high")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--retry-incomplete", action="store_true")
    args = parser.parse_args()

    known = scheduler_common.scheduler_tasks()
    specs = []
    for task_signature, task_spec, output in candidates(args.priority):
        matches = [
            task for task in known
            if str(task.get("signature") or "") == task_signature
        ]
        active = [
            task for task in matches
            if str(task.get("status")) in ACTIVE_STATUSES
        ]
        if output.is_file():
            print(f"skip complete-output: {task_signature}")
        elif active:
            print("skip active: " + ",".join(
                str(task["id"]) for task in active))
        elif matches and not args.retry_incomplete:
            print("skip terminal-incomplete: " + ",".join(
                str(task["id"]) for task in matches))
        else:
            specs.append(task_spec)
    if args.dry_run:
        print(json.dumps(specs, indent=2, sort_keys=True))
        return
    if not specs:
        print("No conservative rejection diagnostics to submit")
        return
    task_ids = _submit(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        _dispatch(task_ids)


if __name__ == "__main__":
    main()
