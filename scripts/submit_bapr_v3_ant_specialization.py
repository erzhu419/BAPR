#!/usr/bin/env python3
"""Submit the BAPR-v3 Ant categorical-controller specialization screen."""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path


_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if str(_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORT_ROOT))

import submit_bapr_v3_budget_matched_fork as common
from jax_experiments.analysis import run_bapr_v3_budget_matched_fork as protocol


ROOT = common.ROOT
JAX_PYTHON = common.JAX_PYTHON
SAVE_BASE = (
    ROOT / "jax_experiments" / "results_bapr_v3_ant_specialization_v1")
FAMILIES = protocol.FAMILIES
VARIANTS = tuple(
    name for name in protocol.POLICY_VARIANTS if name != "direct")
ACTIVE_STATUSES = common.ACTIVE_STATUSES
SIGNATURE_PREFIX = "BAPR/v3-ant-specialization/v1"
MAX_TASKS_PER_INVOCATION = len(FAMILIES) * len(VARIANTS)


def save_root(variant: str) -> Path:
    return SAVE_BASE / variant


def pair_dir(variant: str, family: str) -> Path:
    return protocol.pair_dir(
        family, "Ant-v2", 0, save_root(variant))


def signature(variant: str, family: str) -> str:
    return f"{SIGNATURE_PREFIX}/{variant}/{family}/Ant-v2/seed0"


def output_complete(variant: str, family: str) -> tuple[bool, str]:
    pair = pair_dir(variant, family)
    complete, reason = common.output_complete(pair)
    if not complete:
        return complete, reason
    manifest = protocol.read_json(pair / protocol.PAIR_MANIFEST_REL)
    expected = protocol.pair_identity(family, "Ant-v2", 0, variant)
    if manifest.get("identity") != expected:
        return False, "completed pair has the wrong policy variant identity"
    return True, "complete"


def runner_values(variant: str, family: str) -> list[str]:
    return [
        "--family", family,
        "--env", "Ant-v2",
        "--seed", "0",
        "--save-root", str(save_root(variant)),
        "--policy-variant", variant,
        "--resume",
    ]


def task_spec(variant: str, family: str, priority: str) -> dict:
    pair = pair_dir(variant, family)
    command = (
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.24 "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 JAX_NUM_THREADS=1 "
        "TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 "
        f"{shlex.quote(str(JAX_PYTHON))} -u -m "
        "jax_experiments.analysis.run_bapr_v3_budget_matched_fork "
        f"{shlex.join(runner_values(variant, family))}"
    )
    return {
        "description": (
            f"BAPR-v3 Ant specialization {variant} {family} seed0; "
            "shared checkpoint robust/oracle pair"),
        "cmd": command,
        "cwd": str(ROOT),
        "signature": signature(variant, family),
        "project": "BAPR",
        "vram": 3600,
        "ram_mb": 6144,
        "cpu": 2,
        "priority": priority,
        "ckpt_dir": str(pair),
        "ckpt_glob": protocol.STATE_NAME,
        "resume_flag": "",
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "allow_remote_large_data": True,
        "allow_initial_resume_scan_error": False,
        "reroute_on_node_down": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", action="append", choices=FAMILIES)
    parser.add_argument("--variant", action="append", choices=VARIANTS)
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="high")
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--task-cap", type=int, default=MAX_TASKS_PER_INVOCATION)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    families = tuple(dict.fromkeys(args.family or FAMILIES))
    variants = tuple(dict.fromkeys(args.variant or VARIANTS))
    expected = len(families) * len(variants)
    if not 1 <= args.task_cap <= MAX_TASKS_PER_INVOCATION:
        raise SystemExit(
            f"task-cap must be in [1,{MAX_TASKS_PER_INVOCATION}]")
    if expected > args.task_cap:
        raise SystemExit(
            f"selection expands to {expected} tasks; cap is {args.task_cap}")

    tasks = common.scheduler_tasks()
    by_signature: dict[str, list[dict]] = {}
    for task in tasks:
        value = str(task.get("signature") or "")
        if value:
            by_signature.setdefault(value, []).append(task)

    specs = []
    skipped = []
    for variant in variants:
        for family in families:
            sig = signature(variant, family)
            complete, reason = output_complete(variant, family)
            known = by_signature.get(sig, [])
            active = [
                task for task in known
                if str(task.get("status")) in ACTIVE_STATUSES]
            terminal = [task for task in known if task not in active]
            if complete:
                skipped.append((sig, "complete"))
            elif active:
                ids = ",".join(str(task.get("id", "?")) for task in active)
                skipped.append((sig, f"active:{ids}"))
            elif terminal and not args.retry_incomplete:
                ids = ",".join(str(task.get("id", "?")) for task in terminal)
                skipped.append((
                    sig, f"terminal-incomplete:{ids}; {reason}"))
            else:
                specs.append(task_spec(variant, family, args.priority))

    print(
        f"BAPR-v3 Ant specialization: selected={expected} "
        f"submit={len(specs)} skip={len(skipped)}", flush=True)
    for spec in specs:
        print(f"  submit {spec['signature']}", flush=True)
        if args.dry_run:
            print(json.dumps(spec, indent=2, sort_keys=True))
    for sig, reason in skipped:
        print(f"  skip {reason}: {sig}", flush=True)
    if args.dry_run or not specs:
        return

    task_ids = common.submit_jsonl(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        common.dispatch(task_ids)


if __name__ == "__main__":
    main()
