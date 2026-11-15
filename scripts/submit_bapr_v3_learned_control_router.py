#!/usr/bin/env python3
"""Submit the learned control-equivalence router protocol via scheduler."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import submit_bapr_v3_budget_matched_fork as scheduler_common

if Path(sys.executable).resolve() != scheduler_common.JAX_PYTHON.resolve():
    os.execv(
        str(scheduler_common.JAX_PYTHON),
        [str(scheduler_common.JAX_PYTHON), str(Path(__file__).resolve()),
         *sys.argv[1:]],
    )

from jax_experiments.analysis import bapr_v3_learned_control_router as protocol


JAX_PYTHON = scheduler_common.JAX_PYTHON
NODE = "jtl311linux"
SIGNATURE_PREFIX = "BAPR/v3-structured-channel-learned-router/v1"
ACTIVE_STATUSES = scheduler_common.ACTIVE_STATUSES


def command(module: str, values: list[str], *, cpu_only: bool) -> str:
    resources = (
        "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' "
        if cpu_only else
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.18 "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
    )
    return (
        resources
        + "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 JAX_NUM_THREADS=1 "
        "TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 "
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        f"{shlex.quote(str(JAX_PYTHON))} -u -m {module} "
        f"{shlex.join(values)} && echo DONE"
    )


def common_spec(*, description: str, signature: str, cmd: str,
                output: Path, priority: str, cpu_only: bool) -> dict:
    spec = {
        "description": description,
        "cmd": cmd,
        "cwd": str(ROOT),
        "signature": signature,
        "project": "BAPR",
        "vram": 0 if cpu_only else 1800,
        "ram_mb": 4096 if cpu_only else 8192,
        "cpu": 1 if cpu_only else 2,
        "priority": priority,
        # The five policy bundles are intentionally retained only on this node.
        "require_node": NODE,
        "result_dir": str(output),
        "local_result_dir": str(output),
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": False,
    }
    if cpu_only:
        spec.update({
            "allow_cpu_training": True,
            "cpu_training_justification": (
                "Aggregation only; no rollout, policy update, or estimator "
                "training is executed."),
        })
    return spec


def train_spec(priority: str):
    signature = f"{SIGNATURE_PREFIX}/train"
    spec = common_spec(
        description=(
            "Train causal probabilistic physical-mode estimator and freeze "
            "control-equivalence routing config"),
        signature=signature,
        cmd=command(
            "jax_experiments.analysis.train_bapr_v3_learned_control_router",
            ["--resume"],
            cpu_only=False,
        ),
        output=protocol.MODEL_ROOT,
        priority=priority,
        cpu_only=False,
    )
    spec.update({
        "ckpt_dir": str(protocol.MODEL_ROOT),
        "ckpt_glob": "train_state.json",
        "resume_flag": "",
        "allow_remote_large_data": True,
        "allow_initial_resume_scan_error": True,
    })
    return signature, spec, protocol.MANIFEST_PATH


def finalize_spec(priority: str):
    signature = f"{SIGNATURE_PREFIX}/finalize"
    spec = common_spec(
        description=(
            "Finalize completed estimator checkpoint with vectorized "
            "validation and freeze routing config"),
        signature=signature,
        cmd=command(
            "jax_experiments.analysis."
            "finalize_bapr_v3_learned_control_router",
            [],
            cpu_only=False,
        ),
        output=protocol.MODEL_ROOT,
        priority=priority,
        cpu_only=False,
    )
    spec.update({
        "ckpt_dir": str(protocol.MODEL_ROOT),
        "ckpt_glob": "train_state.json",
        "resume_flag": "",
        "allow_remote_large_data": True,
        "allow_initial_resume_scan_error": True,
    })
    return signature, spec, protocol.MANIFEST_PATH


def audit_spec(event_seed: int, priority: str):
    signature = f"{SIGNATURE_PREFIX}/holdout/event-seed-{event_seed}"
    output = protocol.audit_group_path(event_seed).parent
    spec = common_spec(
        description=(
            "Sealed learned-router holdout with robust fallback, event seed "
            f"{event_seed}"),
        signature=signature,
        cmd=command(
            "jax_experiments.analysis."
            "run_bapr_v3_learned_control_router_audit",
            [
                "--event-seed", str(event_seed),
                "--out-dir", str(output),
                "--resume",
            ],
            cpu_only=False,
        ),
        output=output,
        priority=priority,
        cpu_only=False,
    )
    spec.update({
        "ckpt_dir": str(protocol.MODEL_ROOT),
        "ckpt_glob": "router_manifest.json",
        "resume_flag": "",
        "allow_initial_resume_scan_error": False,
    })
    return signature, spec, output / "group.json"


def analysis_spec(priority: str):
    signature = f"{SIGNATURE_PREFIX}/analysis"
    report = protocol.ANALYSIS_ROOT / "report.md"
    summary = protocol.ANALYSIS_ROOT / "summary.json"
    spec = common_spec(
        description=(
            "Aggregate five sealed learned-router holdouts; no rollout or "
            "training"),
        signature=signature,
        cmd=command(
            "jax_experiments.analysis."
            "analyze_bapr_v3_learned_control_router_audit",
            ["--output", str(report), "--json-output", str(summary)],
            cpu_only=True,
        ),
        output=protocol.ANALYSIS_ROOT,
        priority=priority,
        cpu_only=True,
    )
    spec["allow_initial_resume_scan_error"] = True
    return signature, spec, summary


def known_by_signature():
    result: dict[str, list[dict]] = {}
    for task in scheduler_common.scheduler_tasks():
        signature = str(task.get("signature") or "")
        if signature:
            result.setdefault(signature, []).append(task)
    return result


def should_submit(signature, output, known, retry_incomplete):
    if output.is_file():
        return False, "complete-output"
    tasks = known.get(signature, [])
    active = [task for task in tasks
              if str(task.get("status")) in ACTIVE_STATUSES]
    if active:
        return False, "active:" + ",".join(str(task["id"]) for task in active)
    if tasks and not retry_incomplete:
        return False, "terminal-incomplete:" + ",".join(
            str(task["id"]) for task in tasks)
    return True, "ready"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("train", "finalize", "audit", "analysis"),
        required=True)
    parser.add_argument(
        "--event-seed", type=int, action="append",
        choices=protocol.HOLDOUT_EVENT_SEEDS)
    parser.add_argument("--priority", choices=("low", "normal", "high"),
                        default="high")
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()
    protocol.configure()
    if args.phase == "train":
        protocol.control.load_controller_map()
        candidates = [train_spec(args.priority)]
    elif args.phase == "finalize":
        protocol.control.load_controller_map()
        candidates = [finalize_spec(args.priority)]
    elif args.phase == "audit":
        manifest = protocol.load_manifest()
        if not manifest.get("validation_gate_pass"):
            raise SystemExit(
                "router validation gate failed; refusing sealed holdouts")
        event_seeds = tuple(dict.fromkeys(
            args.event_seed or protocol.HOLDOUT_EVENT_SEEDS))
        candidates = [audit_spec(seed, args.priority)
                      for seed in event_seeds]
    else:
        missing = [seed for seed in protocol.HOLDOUT_EVENT_SEEDS
                   if not protocol.audit_group_path(seed).is_file()]
        if missing:
            raise SystemExit(f"holdout groups incomplete: {missing}")
        candidates = [analysis_spec(args.priority)]

    known = known_by_signature()
    specs = []
    skipped = []
    for signature, spec, output in candidates:
        submit, reason = should_submit(
            signature, output, known, args.retry_incomplete)
        if submit:
            specs.append(spec)
        else:
            skipped.append((signature, reason))
    print(
        f"Learned-router phase={args.phase}: "
        f"submit={len(specs)} skip={len(skipped)}",
        flush=True,
    )
    for spec in specs:
        print(f"  submit {spec['signature']}")
        if args.dry_run:
            print(json.dumps(spec, indent=2, sort_keys=True))
    for signature, reason in skipped:
        print(f"  skip {reason}: {signature}")
    if args.dry_run or not specs:
        return
    task_ids = scheduler_common.submit_jsonl(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        scheduler_common.dispatch(task_ids)


if __name__ == "__main__":
    main()
