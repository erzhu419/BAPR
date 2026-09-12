#!/usr/bin/env python3
"""Submit only the corrected V21 CPU audits for amendment 2."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import submit_regime_polarity_full_state_confirmation_v21 as original

if (
    __name__ == "__main__"
    and Path(sys.executable).resolve()
    != original.scheduler_common.JAX_PYTHON.resolve()
):
    os.execv(
        str(original.scheduler_common.JAX_PYTHON),
        [str(original.scheduler_common.JAX_PYTHON),
         str(Path(__file__).resolve()), *sys.argv[1:]],
    )


protocol = original.protocol
AMENDMENT = protocol.REGISTRATION_ROOT / "amendment2.json"
AUDIT_MODULE = (
    "jax_experiments.analysis."
    "run_regime_polarity_full_state_confirmation_audit_v21_amendment2"
)
SUFFIX = "amendment2"


def audit_signature(seed: int) -> str:
    return f"{original.audit_signature(seed)}/{SUFFIX}"


def audit_spec(seed: int, priority: str) -> dict:
    spec = original.audit_spec(seed, priority)
    spec.update({
        "description": f"V21 audit amendment 2 seed {seed}",
        "cmd": original._cpu_command(
            AUDIT_MODULE, ["--seed", str(seed), "--resume"], threads=16),
        "signature": audit_signature(seed),
    })
    spec["wait_for_files"] = [*spec["wait_for_files"], str(AMENDMENT)]
    spec["stage_input_paths"] = [
        *spec["stage_input_paths"], str(AMENDMENT.parent),
    ]
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="high")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

    if not AMENDMENT.is_file():
        raise FileNotFoundError(AMENDMENT)
    protocol.validate_registration()
    known = original.scheduler_common.scheduler_tasks()
    specs = []
    for seed in protocol.TRAINING_SEEDS:
        signature = audit_signature(seed)
        active = [
            task for task in known
            if str(task.get("signature") or "") == signature
            and str(task.get("status")) in original.ACTIVE_STATUSES
        ]
        if protocol.audit_manifest(seed).is_file():
            try:
                from jax_experiments.analysis import (
                    run_regime_polarity_full_state_confirmation_audit_v21
                    as audit,
                )
                audit.validate_audit(seed)
            except (KeyError, OSError, TypeError, ValueError):
                specs.append(audit_spec(seed, args.priority))
            else:
                print(f"skip complete-output: {signature}")
        elif active:
            print("skip active: " + ",".join(str(task["id"]) for task in active))
        else:
            specs.append(audit_spec(seed, args.priority))
    if args.dry_run:
        print(f"V21 amendment 2 dry-run task count: {len(specs)}")
        return
    if not specs:
        print("No V21 amendment 2 tasks to submit")
        return
    task_ids = original._submit(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        original._dispatch(task_ids)


if __name__ == "__main__":
    main()
