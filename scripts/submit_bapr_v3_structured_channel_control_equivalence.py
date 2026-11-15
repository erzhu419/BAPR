#!/usr/bin/env python3
"""Submit calibration-frozen structured-channel controller-map audits."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import submit_bapr_v3_budget_matched_fork as scheduler_common

if Path(sys.executable).resolve() != scheduler_common.JAX_PYTHON.resolve():
    os.execv(
        str(scheduler_common.JAX_PYTHON),
        [str(scheduler_common.JAX_PYTHON), str(Path(__file__).resolve()),
         *sys.argv[1:]],
    )

from jax_experiments.analysis import bapr_v3_control_equivalence as protocol


JAX_PYTHON = scheduler_common.JAX_PYTHON
NODE = "jtl311linux"
SIGNATURE_PREFIX = "BAPR/v3-structured-channel-control-equivalence/v1"
ACTIVE_STATUSES = scheduler_common.ACTIVE_STATUSES


def command(module: str, values: list[str], *, cpu_only: bool) -> str:
    resources = (
        "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' "
        if cpu_only else
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.12 "
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


def common_spec(
    *, description: str, signature: str, cmd: str, output: Path,
    priority: str, cpu_only: bool,
) -> dict:
    spec = {
        "description": description,
        "cmd": cmd,
        "cwd": str(ROOT),
        "signature": signature,
        "project": "BAPR",
        "vram": 0 if cpu_only else 500,
        "ram_mb": 4096 if cpu_only else 2048,
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
        "allow_initial_resume_scan_error": True,
        "reroute_on_node_down": False,
    }
    if cpu_only:
        spec.update({
            "allow_cpu_training": True,
            "cpu_training_justification": (
                "Calibration/aggregation only; no rollout, gradient update, "
                "or model training is executed."
            ),
        })
    return spec


def map_spec(priority: str) -> tuple[str, dict, Path]:
    signature = f"{SIGNATURE_PREFIX}/select-map"
    output = protocol.MAPPING_ROOT
    spec = common_spec(
        description=(
            "Freeze structured-channel control-equivalence map from completed "
            "calibration streams; no rollout or training"
        ),
        signature=signature,
        cmd=command(
            "jax_experiments.analysis.bapr_v3_control_equivalence",
            ["--select"],
            cpu_only=True,
        ),
        output=output,
        priority=priority,
        cpu_only=True,
    )
    return signature, spec, protocol.MAPPING_PATH


def audit_spec(event_seed: int, priority: str) -> tuple[str, dict, Path]:
    signature = f"{SIGNATURE_PREFIX}/holdout/event-seed-{event_seed}"
    output = protocol.holdout_group_path(event_seed).parent
    spec = common_spec(
        description=(
            "Untouched holdout audit for frozen structured-channel "
            f"controller map, event seed {event_seed}"
        ),
        signature=signature,
        cmd=command(
            "jax_experiments.analysis."
            "run_bapr_v3_control_equivalence_audit",
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
    return signature, spec, output / "group.json"


def analysis_spec(priority: str) -> tuple[str, dict, Path]:
    signature = f"{SIGNATURE_PREFIX}/analysis"
    output = protocol.ANALYSIS_ROOT
    report = output / "report.md"
    summary = output / "summary.json"
    spec = common_spec(
        description=(
            "Aggregate structured-channel control-equivalence holdouts; "
            "no rollout or training"
        ),
        signature=signature,
        cmd=command(
            "jax_experiments.analysis."
            "analyze_bapr_v3_control_equivalence_audit",
            ["--output", str(report), "--json-output", str(summary)],
            cpu_only=True,
        ),
        output=output,
        priority=priority,
        cpu_only=True,
    )
    return signature, spec, summary


def known_by_signature() -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for task in scheduler_common.scheduler_tasks():
        signature = str(task.get("signature") or "")
        if signature:
            result.setdefault(signature, []).append(task)
    return result


def should_submit(
    signature: str, output: Path, known: dict[str, list[dict]],
    retry_incomplete: bool,
) -> tuple[bool, str]:
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
    parser.add_argument("--phase", choices=("map", "audit", "analysis"),
                        required=True)
    parser.add_argument(
        "--event-seed",
        type=int,
        action="append",
        choices=protocol.HOLDOUT_EVENT_SEEDS,
        help="Limit the audit phase to selected holdout streams.",
    )
    parser.add_argument("--priority", choices=("low", "normal", "high"),
                        default="high")
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()
    protocol.configure()
    if args.phase == "map":
        candidates = [map_spec(args.priority)]
    elif args.phase == "audit":
        event_seeds = tuple(dict.fromkeys(
            args.event_seed or protocol.HOLDOUT_EVENT_SEEDS))
        candidates = [
            audit_spec(seed, args.priority)
            for seed in event_seeds
        ]
    else:
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
        f"Control-equivalence phase={args.phase}: "
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
    ids = scheduler_common.submit_jsonl(specs)
    print(f"Submitted task ids: {','.join(ids)}", flush=True)
    if args.dispatch:
        scheduler_common.dispatch(ids)


if __name__ == "__main__":
    main()
