#!/usr/bin/env python3
"""Submit the shared-regime BAPR headroom screen through scheduleurm."""
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

from jax_experiments.analysis import bapr_regime_screen as protocol


SIGNATURE_PREFIX = "BAPR/shared-regime-screen/v1"
COLD_START_VRAM_MB = 2048


def signature(role: str) -> str:
    return f"{SIGNATURE_PREFIX}/train/{protocol.require_role(role)}"


def command(role: str) -> str:
    fraction = 0.28 if role.startswith("regime_") else (
        0.26 if role == "escp" else 0.22)
    return (
        f"SCHEDULEURM_ETA_TOTAL_UNITS={protocol.MAX_ITERS} "
        "BAPR_SCHEDULER_RESUME=--resume "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        f"XLA_PYTHON_CLIENT_MEM_FRACTION={fraction:.2f} "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
        "OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 "
        "NUMEXPR_NUM_THREADS=2 JAX_NUM_THREADS=2 "
        "TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 "
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        f"{shlex.quote(str(scheduler_common.JAX_PYTHON))} -u -m "
        "jax_experiments.analysis.run_bapr_regime_screen_controller "
        f"--role {shlex.quote(role)} --resume && echo DONE"
    )


def spec(role: str, priority: str) -> dict:
    run_dir = protocol.run_dir(role)
    bundle = protocol.bundle_dir(role)
    return {
        "description": f"Shared-regime headroom screen: {role}",
        "cmd": command(role),
        "cwd": str(ROOT),
        "signature": signature(role),
        "project": "BAPR",
        "vram_resource_family": (
            f"BAPR/shared-regime-screen/{role}/gpu-runtime"),
        "vram": COLD_START_VRAM_MB,
        "ram_mb": 8192,
        "cpu": 2,
        "priority": priority,
        "ckpt_dir": str(run_dir / "checkpoints"),
        "ckpt_glob": "train_state.pkl",
        "resume_flag": "",
        "resume_managed_by_cmd": True,
        "result_dir": str(bundle),
        "local_result_dir": str(bundle),
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="high")
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

    known = scheduler_common.scheduler_tasks()
    specs = []
    for role in protocol.ROLES:
        task_signature = signature(role)
        tasks = [task for task in known
                 if str(task.get("signature") or "") == task_signature]
        active = [task for task in tasks
                  if str(task.get("status")) in scheduler_common.ACTIVE_STATUSES]
        if protocol.bundle_manifest(role).is_file():
            print(f"skip complete-output: {task_signature}")
        elif active:
            print("skip active: " + ",".join(
                str(task["id"]) for task in active))
        elif tasks and not args.retry_incomplete:
            print("skip terminal-incomplete: " + ",".join(
                str(task["id"]) for task in tasks))
        else:
            specs.append(spec(role, args.priority))

    if args.dry_run:
        print(json.dumps(specs, indent=2, sort_keys=True))
        return
    if not specs:
        print("No shared-regime screen tasks to submit")
        return
    ids = scheduler_common.submit_jsonl(specs)
    print(f"Submitted task ids: {','.join(ids)}", flush=True)
    if args.dispatch:
        scheduler_common.dispatch(ids)


if __name__ == "__main__":
    main()
