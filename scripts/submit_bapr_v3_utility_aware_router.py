#!/usr/bin/env python3
"""Submit utility-aware router calibration and audits via scheduler."""
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

from jax_experiments.analysis import bapr_v3_learned_control_router as estimator
from jax_experiments.analysis import bapr_v3_utility_aware_router as protocol


JAX_PYTHON = scheduler_common.JAX_PYTHON
NODE = "jtl311linux"
SIGNATURE_PREFIX = "BAPR/v3-structured-channel-utility-router/v2"
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
                "Calibration/aggregation only; no rollout, policy update, "
                "or estimator training is executed."),
        })
    return spec


def map_spec(priority: str):
    signature = f"{SIGNATURE_PREFIX}/utility-table"
    spec = common_spec(
        description=(
            "Freeze robust-inclusive controller utility table from "
            "predeclared calibration streams"),
        signature=signature,
        cmd=command(
            "jax_experiments.analysis.bapr_v3_utility_aware_router",
            ["--select"], cpu_only=True),
        output=protocol.TABLE_ROOT,
        priority=priority,
        cpu_only=True,
    )
    spec["allow_initial_resume_scan_error"] = True
    return signature, spec, protocol.TABLE_PATH


def audit_spec(
    role: str,
    event_seed: int,
    priority: str,
    decision_variant: str = protocol.BASELINE_DECISION_VARIANT,
):
    variant_path = (
        "" if decision_variant == protocol.BASELINE_DECISION_VARIANT
        else f"/{decision_variant}")
    signature = (
        f"{SIGNATURE_PREFIX}/{role}{variant_path}/event-seed-{event_seed}")
    output = protocol.audit_group_path(
        role, event_seed, decision_variant).parent
    spec = common_spec(
        description=(
            f"Utility-aware router {role} with slow-pair and full-cycle "
            f"switching, event seed {event_seed}"),
        signature=signature,
        cmd=command(
            "jax_experiments.analysis."
            "run_bapr_v3_utility_aware_router_audit",
            [
                "--role", role,
                "--event-seed", str(event_seed),
                "--decision-variant", decision_variant,
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
        "ckpt_dir": str(estimator.MODEL_ROOT),
        "ckpt_glob": "router_manifest.json",
        "resume_flag": "",
        "allow_initial_resume_scan_error": False,
    })
    return signature, spec, output / "group.json"


def analysis_spec(
    role: str,
    priority: str,
    decision_variant: str = protocol.BASELINE_DECISION_VARIANT,
):
    root = protocol.analysis_root(role, decision_variant)
    variant_path = (
        "" if decision_variant == protocol.BASELINE_DECISION_VARIANT
        else f"/{decision_variant}")
    signature = f"{SIGNATURE_PREFIX}/{role}-analysis{variant_path}"
    report = root / "report.md"
    summary = root / "summary.json"
    spec = common_spec(
        description=(
            f"Aggregate utility-aware router {role}; no rollout or training"),
        signature=signature,
        cmd=command(
            "jax_experiments.analysis."
            "analyze_bapr_v3_utility_aware_router_audit",
            [
                "--role", role,
                "--decision-variant", decision_variant,
                "--output", str(report),
                "--json-output", str(summary),
            ],
            cpu_only=True,
        ),
        output=root,
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


def _validation_passed(decision_variant: str) -> bool:
    path = protocol.analysis_root(
        "validation", decision_variant) / "summary.json"
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return bool(
        payload.get("schema") == protocol.ANALYSIS_SCHEMA
        and payload.get("role") == "validation"
        and payload.get(
            "decision_variant", protocol.BASELINE_DECISION_VARIANT)
        == decision_variant
        and payload.get("gate", {}).get("passed"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("map", "validation", "validation-analysis",
                 "holdout", "analysis"),
        required=True)
    parser.add_argument("--event-seed", type=int, action="append")
    parser.add_argument(
        "--decision-variant", choices=tuple(protocol.DECISION_VARIANTS),
        default=protocol.BASELINE_DECISION_VARIANT)
    parser.add_argument("--priority", choices=("low", "normal", "high"),
                        default="high")
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()
    protocol.configure()

    if args.phase == "map":
        candidates = [map_spec(args.priority)]
    elif args.phase in ("validation", "holdout"):
        protocol.load_utility_table()
        role = args.phase
        if role == "holdout" and not _validation_passed(
                args.decision_variant):
            raise SystemExit(
                "utility-router validation gate has not passed; refusing "
                "sealed holdouts")
        allowed = protocol.event_seeds(role)
        selected = tuple(dict.fromkeys(args.event_seed or allowed))
        if any(seed not in allowed for seed in selected):
            raise SystemExit(f"invalid {role} event seed: {selected}")
        candidates = [audit_spec(
            role, seed, args.priority, args.decision_variant)
                      for seed in selected]
    else:
        role = "validation" if args.phase == "validation-analysis" \
            else "holdout"
        candidates = [analysis_spec(
            role, args.priority, args.decision_variant)]

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
        f"Utility-router phase={args.phase}: "
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
