#!/usr/bin/env python3
"""Resubmit only the five v18 CPU audits with amendment 1."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import submit_regime_polarity_v5_final_comparison_v18 as base

if __name__ == "__main__" and Path(sys.executable).resolve() != base.scheduler_common.JAX_PYTHON.resolve():
    os.execv(
        str(base.scheduler_common.JAX_PYTHON),
        [str(base.scheduler_common.JAX_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
    )

from jax_experiments.analysis import (
    run_regime_polarity_v5_final_comparison_audit_v18_amendment1 as amendment,
)


def amended_audit_spec(seed: int, priority: str) -> dict:
    spec = base.audit_spec(seed, priority)
    old_module = (
        "jax_experiments.analysis."
        "run_regime_polarity_v5_final_comparison_audit_v18"
    )
    new_module = (
        "jax_experiments.analysis."
        "run_regime_polarity_v5_final_comparison_audit_v18_amendment1"
    )
    if old_module not in spec["cmd"]:
        raise ValueError("frozen v18 audit command changed")
    spec["cmd"] = spec["cmd"].replace(old_module, new_module, 1)
    spec["description"] += " amendment1"
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
        signature = base.audit_signature(seed)
        active = [
            task for task in known
            if task.get("signature") == signature
            and task.get("status") in base.ACTIVE_STATUSES
        ]
        if base.protocol.audit_manifest(seed).is_file():
            print(f"skip complete-output: {signature}")
        elif active:
            print("skip active: " + ",".join(str(task["id"]) for task in active))
        else:
            specs.append(amended_audit_spec(seed, args.priority))
    if args.dry_run:
        print(f"V18 amendment1 audit task count: {len(specs)}")
        for spec in specs:
            print(spec["signature"], spec["allowed_nodes"])
        return
    if not specs:
        print("No v18 amendment1 audits to submit")
        return
    task_ids = base._submit(specs)
    print(f"Submitted amendment1 audit ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        base._dispatch(task_ids)


if __name__ == "__main__":
    main()
