#!/usr/bin/env python3
"""Submit only missing v19 CPU audits with the registered path amendment."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
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
    regime_polarity_specialist_stability_v19 as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_stability_v19_amendment1 as amendment,
)
import submit_regime_polarity_specialist_stability_v19 as original_submit


SIGNATURE_PREFIX = (
    "BAPR/regime-polarity/v19-specialist-stability/audit-amendment1"
)
SUBMIT_INTENT = "bapr-v19-specialist-stability-audit-amendment1-submit"


def signature(variant: str, seed: int) -> str:
    return (
        f"{SIGNATURE_PREFIX}/{protocol.require_variant(variant)}/seed-"
        f"{protocol.require_training_seed(seed)}"
    )


def spec(variant: str, seed: int, priority: str) -> dict:
    variant = protocol.require_variant(variant)
    seed = protocol.require_training_seed(seed)
    row = deepcopy(original_submit.audit_spec(variant, seed, priority))
    row["description"] = f"V19 amendment1 {variant} audit seed {seed}"
    row["signature"] = signature(variant, seed)
    row["cmd"] = original_submit._cpu_command(
        "jax_experiments.analysis."
        "run_regime_polarity_specialist_stability_audit_v19_amendment1",
        ["--variant", variant, "--seed", str(seed), "--resume"],
        threads=16,
    )
    row["wait_for_files"] = [
        *row["wait_for_files"],
        str(amendment.REGISTRATION_PATH),
        str(amendment.REPORT),
    ]
    row["stage_input_paths"] = list(dict.fromkeys([
        *row["stage_input_paths"],
        str(amendment.REGISTRATION_ROOT),
        str(amendment.REPORT.parent),
    ]))
    return row


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
            f"replacement audit batch incomplete: {task_ids}")
    print(json.dumps(response, indent=2))
    return task_ids


def _dispatch(task_ids: list[str]) -> None:
    command = [
        sys.executable,
        str(scheduler_common.SCHEDULER),
        "dispatch",
        "--bulk-window",
        "--intent-label",
        "bapr-v19-specialist-stability-audit-amendment1-dispatch",
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

    amendment.create_registration()
    known = scheduler_common.scheduler_tasks()
    specs = []
    for variant in protocol.VARIANTS:
        for seed in protocol.TRAINING_SEEDS:
            output = protocol.audit_manifest(variant, seed)
            task_signature = signature(variant, seed)
            active = [
                task for task in known
                if str(task.get("signature") or "") == task_signature
                and str(task.get("status")) in scheduler_common.ACTIVE_STATUSES
            ]
            if output.is_file():
                print(f"skip complete-output: {task_signature}")
            elif active:
                print("skip active: " + ",".join(
                    str(task["id"]) for task in active))
            else:
                specs.append(spec(variant, seed, args.priority))
    if args.dry_run:
        print(json.dumps(specs, indent=2, sort_keys=True))
        return
    if not specs:
        print("No missing v19 audits to submit")
        return
    task_ids = _submit(specs)
    print(f"Submitted replacement audit ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        _dispatch(task_ids)


if __name__ == "__main__":
    main()
