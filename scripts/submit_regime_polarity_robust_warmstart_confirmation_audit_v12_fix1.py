#!/usr/bin/env python3
"""Submit only the five registered v12 audit-fix tasks."""
from __future__ import annotations

import argparse
import json
import os
import shlex
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
        [str(scheduler_common.JAX_PYTHON), str(Path(__file__).resolve()),
         *sys.argv[1:]],
    )

from jax_experiments.analysis import (
    regime_polarity_robust_warmstart_confirmation_v12 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_robust_warmstart_confirmation_audit_v12_fix1 as fix,
)
import submit_regime_polarity_robust_warmstart_confirmation_v12 as parent


SIGNATURE_PREFIX = f"{parent.SIGNATURE_PREFIX}/audit-fix1"
SUBMIT_INTENT = "bapr-v12-audit-fix1-submit"


def signature(seed: int) -> str:
    return f"{SIGNATURE_PREFIX}/seed-{protocol.require_training_seed(seed)}"


def spec(seed: int, priority: str) -> dict:
    seed = protocol.require_training_seed(seed)
    row = parent.audit_spec(seed, priority)
    row["description"] = f"V12 corrected policy-bank audit seed {seed}"
    row["signature"] = signature(seed)
    row["cmd"] = parent._cpu_command(
        "jax_experiments.analysis."
        "run_regime_polarity_robust_warmstart_confirmation_audit_v12_fix1",
        ["--seed", str(seed), "--resume"],
        threads=16,
    )
    row["wait_for_files"] = [
        str(fix.FIX_REGISTRATION_PATH),
        *row["wait_for_files"],
    ]
    return row


def candidates(priority: str):
    return [
        (signature(seed), spec(seed, priority), protocol.audit_manifest(seed))
        for seed in protocol.TRAINING_SEEDS
    ]


def _submit(specs: list[dict]) -> list[str]:
    payload = "".join(json.dumps(row) + "\n" for row in specs)
    command = [
        sys.executable,
        str(scheduler_common.SCHEDULER),
        "submit-jsonl",
        "--stdin",
        "--trusted",
        "--json",
        "--intent-label",
        SUBMIT_INTENT,
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
    task_ids = [
        str(row.get("id") or "") for row in response.get("submitted", [])
    ]
    if len(task_ids) != len(specs) or any(not task_id for task_id in task_ids):
        raise RuntimeError(
            f"scheduler batch incomplete: requested={len(specs)} ids={task_ids}")
    print(json.dumps(response, indent=2))
    return task_ids


def _dispatch(task_ids: list[str]) -> None:
    command = [
        sys.executable,
        str(scheduler_common.SCHEDULER),
        "dispatch",
        "--bulk-window",
        "--intent-label",
        "bapr-v12-audit-fix1-dispatch",
        "--intent-ttl",
        "900",
    ]
    for task_id in task_ids:
        command += ["--task-id", task_id]
    subprocess.run(command, check=True, env=scheduler_common.scheduler_env())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="high")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

    fix.create_fix_registration()
    known = scheduler_common.scheduler_tasks()
    specs = []
    for task_signature, task_spec, output in candidates(args.priority):
        active = [
            task for task in known
            if str(task.get("signature") or "") == task_signature
            and str(task.get("status")) in scheduler_common.ACTIVE_STATUSES
        ]
        if output.is_file():
            print(f"skip complete-output: {task_signature}")
        elif active:
            print("skip active: " + ",".join(str(task["id"]) for task in active))
        else:
            specs.append(task_spec)
    if args.dry_run:
        print(json.dumps(specs, indent=2, sort_keys=True))
        return
    if not specs:
        print("No v12 audit-fix tasks to submit")
        return
    task_ids = _submit(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        _dispatch(task_ids)


if __name__ == "__main__":
    main()
