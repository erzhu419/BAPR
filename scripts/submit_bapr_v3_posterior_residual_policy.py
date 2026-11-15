#!/usr/bin/env python3
"""Submit nonlinear posterior residual work through scheduler only."""
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

from jax_experiments.analysis import (
    bapr_v3_learned_control_router as estimator,
)
from jax_experiments.analysis import (
    bapr_v3_posterior_residual_policy as protocol,
)


SIGNATURE_PREFIX = "BAPR/v3-structured-channel/v7/nonlinear-residual"


def command(module: str, values: list[str], *, cpu_only: bool) -> str:
    resources = (
        "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' "
        if cpu_only else
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.24 "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
    )
    return (
        resources
        + "OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 "
        "NUMEXPR_NUM_THREADS=4 JAX_NUM_THREADS=4 "
        "TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2 "
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        f"{shlex.quote(str(scheduler_common.JAX_PYTHON))} -u -m {module} "
        f"{shlex.join(values)} && echo DONE"
    )


def training_spec(priority: str):
    signature = f"{SIGNATURE_PREFIX}/train"
    spec = {
        "description": (
            "Train switch-matched nonlinear posterior residual policy with "
            "frozen robust/controller bank and one DAgger round"),
        "cmd": command(
            "jax_experiments.analysis."
            "train_bapr_v3_posterior_residual_policy",
            ["--resume"], cpu_only=False),
        "cwd": str(ROOT),
        "signature": signature,
        "project": "BAPR",
        "vram": 1800,
        "ram_mb": 12288,
        "cpu": 4,
        "priority": priority,
        "require_node": "jtl311linux",
        "result_dir": str(protocol.MODEL_ROOT),
        "local_result_dir": str(protocol.MODEL_ROOT),
        "ckpt_dir": str(estimator.MODEL_ROOT),
        "ckpt_glob": "router_manifest.json",
        "resume_flag": "",
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": False,
        "allow_initial_resume_scan_error": False,
    }
    return signature, spec, protocol.MANIFEST_PATH


def evaluation_spec(event_seed: int, priority: str):
    output = protocol.audit_group_path(event_seed)
    signature = f"{SIGNATURE_PREFIX}/audit/event-seed-{event_seed}"
    spec = {
        "description": (
            "Strict nonlinear posterior residual audit, "
            f"event seed {event_seed}"),
        "cmd": command(
            "jax_experiments.analysis."
            "run_bapr_v3_posterior_residual_policy_audit",
            ["--event-seed", str(event_seed), "--resume"],
            cpu_only=False),
        "cwd": str(ROOT),
        "signature": signature,
        "project": "BAPR",
        "vram": 600,
        "ram_mb": 8192,
        "cpu": 2,
        "priority": priority,
        "require_node": "jtl311linux",
        "result_dir": str(output.parent),
        "local_result_dir": str(output.parent),
        "ckpt_dir": str(protocol.MODEL_ROOT),
        "ckpt_glob": protocol.MANIFEST_PATH.name,
        "resume_flag": "",
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": False,
        "allow_initial_resume_scan_error": False,
    }
    return signature, spec, output


def analysis_spec(priority: str):
    signature = f"{SIGNATURE_PREFIX}/analysis"
    spec = {
        "description": "Aggregate nonlinear posterior residual dev audit",
        "cmd": command(
            "jax_experiments.analysis."
            "analyze_bapr_v3_posterior_residual_policy",
            [], cpu_only=True),
        "cwd": str(ROOT),
        "signature": signature,
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 4096,
        "cpu": 1,
        "priority": priority,
        "require_node": "jtl311linux",
        "result_dir": str(protocol.ANALYSIS_ROOT),
        "local_result_dir": str(protocol.ANALYSIS_ROOT),
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": False,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Aggregation only; no rollout, policy update, or training."),
    }
    return signature, spec, protocol.ANALYSIS_JSON


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("training", "evaluation", "analysis"),
        required=True)
    parser.add_argument("--priority", choices=("low", "normal", "high"),
                        default="high")
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()
    protocol.configure()
    if args.phase == "training":
        candidates = [training_spec(args.priority)]
    elif args.phase == "evaluation":
        if not protocol.MANIFEST_PATH.is_file():
            raise SystemExit("nonlinear residual model is incomplete")
        protocol.load_manifest()
        candidates = [
            evaluation_spec(seed, args.priority)
            for seed in protocol.DEVELOPMENT_RETURN_EVENT_SEEDS
        ]
    else:
        missing = [
            seed for seed in protocol.DEVELOPMENT_RETURN_EVENT_SEEDS
            if not protocol.audit_group_path(seed).is_file()
        ]
        if missing:
            raise SystemExit(f"nonlinear residual audits incomplete: {missing}")
        candidates = [analysis_spec(args.priority)]

    known = scheduler_common.scheduler_tasks()
    by_signature = {}
    for task in known:
        by_signature.setdefault(str(task.get("signature") or ""), []).append(
            task)
    specs = []
    for signature, spec, output in candidates:
        tasks = by_signature.get(signature, [])
        active = [task for task in tasks if str(task.get("status"))
                  in scheduler_common.ACTIVE_STATUSES]
        if output.is_file():
            print(f"skip complete-output: {signature}")
        elif active:
            print("skip active: " + ",".join(
                str(task["id"]) for task in active))
        elif tasks and not args.retry_incomplete:
            print("skip terminal-incomplete: " + ",".join(
                str(task["id"]) for task in tasks))
        else:
            specs.append(spec)
    if args.dry_run:
        print(json.dumps(specs, indent=2, sort_keys=True))
        return
    if not specs:
        print("No nonlinear residual tasks to submit")
        return
    task_ids = scheduler_common.submit_jsonl(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        scheduler_common.dispatch(task_ids)


if __name__ == "__main__":
    main()
