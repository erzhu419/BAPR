#!/usr/bin/env python3
"""Register and batch-submit the V32 ten-seed confirmation DAG."""
from __future__ import annotations

import argparse
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

if (
    __name__ == "__main__"
    and Path(sys.executable).resolve() != scheduler_common.JAX_PYTHON.resolve()
):
    os.execv(
        str(scheduler_common.JAX_PYTHON),
        [
            str(scheduler_common.JAX_PYTHON),
            str(Path(__file__).resolve()),
            *sys.argv[1:],
        ],
    )

import submit_regime_polarity_action_compensation_confirmation_v31 as base
from jax_experiments.analysis import (
    regime_polarity_action_compensation_power_confirmation_v32 as protocol,
)


SIGNATURE_PREFIX = "BAPR/regime-polarity/v32-power-confirmation"
SUBMIT_INTENT = "bapr-v32-power-confirmation-submit"
DISPATCH_INTENT = "bapr-v32-power-confirmation-dispatch"
ACTIVE_STATUSES = scheduler_common.ACTIVE_STATUSES
GPU_NODES = base.GPU_NODES
CPU_NODES = base.CPU_NODES

MODULE_REPLACEMENTS = {
    "run_regime_polarity_action_compensation_source_v31":
        "run_regime_polarity_action_compensation_source_v32",
    "run_regime_polarity_action_compensation_reference_v31":
        "run_regime_polarity_action_compensation_reference_v32",
    "run_regime_polarity_action_compensation_baseline_v31":
        "run_regime_polarity_action_compensation_baseline_v32",
    "run_regime_polarity_action_compensation_confirmation_audit_v31":
        "run_regime_polarity_action_compensation_power_audit_v32",
    "analyze_regime_polarity_action_compensation_confirmation_v31":
        "analyze_regime_polarity_action_compensation_power_v32",
}


def _bind() -> None:
    base.protocol = protocol
    base.SIGNATURE_PREFIX = SIGNATURE_PREFIX
    base.SUBMIT_INTENT = SUBMIT_INTENT


def _retarget(spec: dict) -> dict:
    spec = dict(spec)
    command = str(spec["cmd"])
    for old, new in MODULE_REPLACEMENTS.items():
        command = command.replace(old, new)
    spec["cmd"] = command
    spec["description"] = str(spec["description"]).replace("V31", "V32")
    return spec


def candidates(priority: str):
    _bind()
    return [
        (signature, _retarget(spec), output)
        for signature, spec, output in base.candidates(priority)
    ]


def _dispatch(task_ids: list[str]) -> None:
    if not task_ids:
        return
    command = [
        sys.executable,
        str(scheduler_common.SCHEDULER),
        "dispatch",
        "--bulk-window",
        "--intent-label",
        DISPATCH_INTENT,
        "--intent-ttl",
        "900",
    ]
    for task_id in task_ids:
        command += ["--task-id", task_id]
    subprocess.run(command, check=True, env=scheduler_common.scheduler_env())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="high"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

    _bind()
    protocol.create_registration()
    known = scheduler_common.scheduler_tasks()
    specs = []
    skipped_complete = 0
    skipped_active = 0
    for signature, spec, output in candidates(args.priority):
        tasks = [
            task
            for task in known
            if str(task.get("signature") or "") == signature
        ]
        active = [
            task
            for task in tasks
            if str(task.get("status")) in ACTIVE_STATUSES
        ]
        if output.is_file():
            skipped_complete += 1
        elif active:
            skipped_active += 1
        else:
            specs.append(spec)
    if args.dry_run:
        gpu = sum(int(spec["vram"]) > 0 for spec in specs)
        cpu = len(specs) - gpu
        print(
            f"V32 dry-run: submit={len(specs)} gpu={gpu} cpu={cpu} "
            f"complete={skipped_complete} active={skipped_active}",
            flush=True,
        )
        return
    if not specs:
        print("No V32 power-confirmation tasks to submit", flush=True)
        return
    task_ids = base._submit(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        _dispatch(task_ids)


if __name__ == "__main__":
    main()
