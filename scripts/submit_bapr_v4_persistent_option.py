#!/usr/bin/env python3
"""Submit the BAPR-v4 smoke or formal training through scheduler only."""
from __future__ import annotations

import argparse
import json
import os
import shlex
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

from jax_experiments.analysis import bapr_v4_persistent_option as protocol


SIGNATURE_PREFIX = "BAPR/v4-persistent-option/v1"


def command(profile: str) -> str:
    return (
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.46 "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
        "OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 "
        "NUMEXPR_NUM_THREADS=4 JAX_NUM_THREADS=4 "
        "TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2 "
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        f"{shlex.quote(str(scheduler_common.JAX_PYTHON))} -u -m "
        "jax_experiments.analysis.run_bapr_v4_persistent_option "
        f"--profile {shlex.quote(profile)} --resume && echo DONE"
    )


def task_spec(profile: str, priority: str):
    signature = f"{SIGNATURE_PREFIX}/{profile}/seed-{protocol.TRAINING_SEED}"
    spec = {
        "description": (
            f"BAPR-v4 shared persistent-option {profile} training, "
            f"HalfCheetah structured channel seed {protocol.TRAINING_SEED}"),
        "cmd": command(profile),
        "cwd": str(ROOT),
        "signature": signature,
        "project": "BAPR",
        "vram": 5500 if profile == "formal" else 2200,
        "ram_mb": 24576 if profile == "formal" else 8192,
        "cpu": 4,
        "priority": priority,
        "require_node": "jtl311linux",
        "result_dir": str(protocol.status_path(profile).parent),
        "local_result_dir": str(protocol.status_path(profile).parent),
        "ckpt_dir": str(protocol.checkpoint_dir(profile)),
        "ckpt_glob": "params.pkl",
        "resume_flag": "",
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": False,
        "allow_initial_resume_scan_error": True,
    }
    if profile == "formal":
        spec["wait_for_files"] = [
            str(protocol.BOOTSTRAP_MODEL),
            str(protocol.BOOTSTRAP_MANIFEST),
        ]
    return signature, spec, protocol.status_path(profile)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "formal"),
                        required=True)
    parser.add_argument("--priority", choices=("low", "normal", "high"),
                        default="high")
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()
    signature, spec, output = task_spec(args.profile, args.priority)
    known = scheduler_common.scheduler_tasks()
    tasks = [task for task in known
             if str(task.get("signature") or "") == signature]
    active = [task for task in tasks
              if str(task.get("status")) in scheduler_common.ACTIVE_STATUSES]
    specs = []
    if output.is_file():
        print(f"skip complete-output: {signature}")
    elif active:
        print("skip active: " + ",".join(str(task["id"])
                                           for task in active))
    elif tasks and not args.retry_incomplete:
        print("skip terminal-incomplete: " + ",".join(
            str(task["id"]) for task in tasks))
    else:
        specs.append(spec)
    if args.dry_run:
        print(json.dumps(specs, indent=2, sort_keys=True))
        return
    if not specs:
        print("No BAPR-v4 task to submit")
        return
    task_ids = scheduler_common.submit_jsonl(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        scheduler_common.dispatch(task_ids)


if __name__ == "__main__":
    main()
