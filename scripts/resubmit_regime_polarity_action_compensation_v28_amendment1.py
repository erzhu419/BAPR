#!/usr/bin/env python3
"""Resubmit only the five failed V28 audits with execution amendment 1."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import submit_bapr_v3_budget_matched_fork as scheduler_common

if (
    __name__ == "__main__"
    and Path(sys.executable).resolve()
    != scheduler_common.JAX_PYTHON.resolve()
):
    os.execv(
        str(scheduler_common.JAX_PYTHON),
        [str(scheduler_common.JAX_PYTHON), str(Path(__file__).resolve()),
         *sys.argv[1:]],
    )

import submit_regime_polarity_action_compensation_v28 as base

from jax_experiments.analysis import (
    run_regime_polarity_action_compensation_audit_v28_amendment1 as amendment,
)


SUFFIX = "amendment1"


def amended_audit_spec(seed: int, priority: str) -> dict:
    spec = base.audit_spec(seed, priority)
    old_module = (
        "jax_experiments.analysis."
        "run_regime_polarity_action_compensation_audit_v28"
    )
    new_module = old_module + "_amendment1"
    if old_module not in spec["cmd"]:
        raise ValueError("frozen V28 audit command changed")
    spec["cmd"] = spec["cmd"].replace(old_module, new_module, 1)
    spec["description"] += " amendment1"
    spec["signature"] += f"/{SUFFIX}"
    spec["wait_for_files"].append(str(amendment.AMENDMENT_REGISTRATION))
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="high")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

    amendment.create_amendment_registration()
    known = base.scheduler_common.scheduler_tasks()
    specs = []
    for seed in base.protocol.TRAINING_SEEDS:
        signature = f"{base.audit_signature(seed)}/{SUFFIX}"
        active = [
            task for task in known
            if task.get("signature") == signature
            and task.get("status") in base.ACTIVE_STATUSES
        ]
        if base.protocol.audit_manifest(seed).is_file():
            print(f"skip complete-output: {signature}")
        elif active:
            print("skip active: " + ",".join(
                str(task["id"]) for task in active))
        else:
            specs.append(amended_audit_spec(seed, args.priority))
    if args.dry_run:
        print(f"V28 amendment1 audit task count: {len(specs)}")
        for spec in specs:
            print(spec["signature"], spec["allowed_nodes"])
        return
    if not specs:
        print("No V28 amendment1 audits to submit")
        return
    task_ids = base._submit(specs)
    print(f"Submitted amendment1 audit ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        base._dispatch(task_ids)


if __name__ == "__main__":
    main()
