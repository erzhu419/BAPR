#!/usr/bin/env python3
"""Submit the causal sequence-router trainer through scheduler only."""
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

from jax_experiments.analysis import bapr_v3_sequence_router as protocol


NODE = "jtl311linux"
SIGNATURE = "BAPR/v3-structured-channel-sequence-router/v1/train"


def train_spec(priority: str):
    command = (
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
        "OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 "
        "NUMEXPR_NUM_THREADS=2 JAX_NUM_THREADS=2 "
        "TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 "
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        f"{shlex.quote(str(scheduler_common.JAX_PYTHON))} -u -m "
        "jax_experiments.analysis.train_bapr_v3_sequence_router "
        "--updates 2500 --batch-size 16 --context-length 128 "
        "--burnin 8 --checkpoint-interval 100 --resume && echo DONE"
    )
    spec = {
        "description": (
            "Train causal recurrent mode filter on frozen emissions and "
            "switch-matched sequences"),
        "cmd": command,
        "cwd": str(ROOT),
        "signature": SIGNATURE,
        "project": "BAPR",
        "vram": 2600,
        "ram_mb": 16384,
        "cpu": 4,
        "priority": priority,
        "require_node": NODE,
        "result_dir": str(protocol.MODEL_ROOT),
        "local_result_dir": str(protocol.MODEL_ROOT),
        "ckpt_dir": str(protocol.MODEL_ROOT),
        "ckpt_glob": "train_state.json",
        "resume_flag": "",
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": False,
        "allow_initial_resume_scan_error": True,
    }
    return SIGNATURE, spec, protocol.MANIFEST_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--priority", choices=("low", "normal", "high"),
                        default="high")
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()
    protocol.configure()
    signature, spec, output = train_spec(args.priority)
    known = scheduler_common.scheduler_tasks()
    active = [task for task in known
              if task.get("signature") == signature
              and str(task.get("status"))
              in scheduler_common.ACTIVE_STATUSES]
    terminal = [task for task in known if task.get("signature") == signature]
    if output.is_file():
        print(f"skip complete-output: {output}")
        return
    if active:
        print("skip active: " + ",".join(str(task["id"])
                                          for task in active))
        return
    if terminal and not args.retry_incomplete:
        print("skip terminal-incomplete: " + ",".join(
            str(task["id"]) for task in terminal))
        return
    if args.dry_run:
        print(json.dumps(spec, indent=2, sort_keys=True))
        return
    task_ids = scheduler_common.submit_jsonl([spec])
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        scheduler_common.dispatch(task_ids)


if __name__ == "__main__":
    main()
