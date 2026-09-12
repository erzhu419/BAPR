#!/usr/bin/env python3
"""Submit dependency-gated CPU audits for the shared-regime screen."""
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
from jax_experiments.analysis.run_bapr_regime_screen_audit import (
    validate_audit,
)


SIGNATURE_PREFIX = "BAPR/shared-regime-screen/v1"
CPU_NODES = [f"node00{index}" for index in range(1, 7)]


def _cpu_command(module: str, values: list[str]) -> str:
    return (
        "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' "
        "OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 "
        "NUMEXPR_NUM_THREADS=4 JAX_NUM_THREADS=4 "
        "TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2 "
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        f"{shlex.quote(str(scheduler_common.JAX_PYTHON))} -u -m "
        f"{module} {shlex.join(values)} && echo DONE"
    )


def audit_signature(role: str, event_seed: int) -> str:
    return (
        f"{SIGNATURE_PREFIX}/audit/{protocol.require_role(role)}/"
        f"event-seed-{protocol.require_audit_event_seed(event_seed)}")


def audit_spec(role: str, event_seed: int, priority: str) -> dict:
    destination = protocol.audit_dir(role, event_seed)
    return {
        "description": (
            f"Shared-regime strict audit {role}, event seed {event_seed}"),
        "cmd": _cpu_command(
            "jax_experiments.analysis.run_bapr_regime_screen_audit",
            ["--role", role, "--event-seed", str(event_seed), "--resume"]),
        "cwd": str(ROOT),
        "signature": audit_signature(role, event_seed),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 12288,
        "cpu": 4,
        "priority": priority,
        "allowed_nodes": CPU_NODES,
        "result_dir": str(destination),
        "local_result_dir": str(destination),
        "wait_for_files": [
            str(path) for path in protocol.bundle_required_paths(role)],
        "stage_input_paths": [str(protocol.bundle_dir(role))],
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Strict deterministic checkpoint evaluation only; no updates."),
    }


def analysis_signature() -> str:
    return f"{SIGNATURE_PREFIX}/analysis"


def analysis_spec(priority: str) -> dict:
    return {
        "description": "Aggregate shared-regime BAPR headroom gate",
        "cmd": _cpu_command(
            "jax_experiments.analysis.analyze_bapr_regime_screen", []),
        "cwd": str(ROOT),
        "signature": analysis_signature(),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 2048,
        "cpu": 1,
        "priority": priority,
        "allowed_nodes": CPU_NODES,
        "result_dir": str(protocol.AUDIT_ROOT),
        "local_result_dir": str(protocol.AUDIT_ROOT),
        "wait_for_files": [
            str(protocol.audit_manifest(role, event_seed))
            for role in protocol.AUDIT_ROLES
            for event_seed in protocol.AUDIT_EVENT_SEEDS
        ],
        "stage_input_paths": [str(protocol.AUDIT_ROOT)],
        "stage_excludes": [
            "jax_experiments/results_bapr_regime_screen_v1/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": "CSV/JSON aggregation only.",
    }


def _audit_complete(role: str, event_seed: int) -> bool:
    try:
        validate_audit(role, event_seed)
        return True
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
        return False


def _analysis_complete() -> bool:
    try:
        payload = protocol.read_json(protocol.analysis_json())
        return (
            payload.get("schema")
            == "bapr.shared-regime-screen-analysis.v1"
            and payload.get("status") == "complete")
    except (FileNotFoundError, OSError, TypeError, ValueError):
        return False


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
    rows = [
        (
            audit_signature(role, event_seed),
            audit_spec(role, event_seed, args.priority),
            _audit_complete(role, event_seed),
        )
        for role in protocol.AUDIT_ROLES
        for event_seed in protocol.AUDIT_EVENT_SEEDS
    ]
    rows.append((
        analysis_signature(), analysis_spec(args.priority),
        _analysis_complete()))
    for signature, spec, complete in rows:
        tasks = [
            task for task in known
            if str(task.get("signature") or "") == signature]
        active = [
            task for task in tasks
            if str(task.get("status")) in scheduler_common.ACTIVE_STATUSES]
        if complete:
            print(f"skip complete-output: {signature}")
        elif active:
            print("skip active: " + ",".join(
                str(task["id"]) for task in active))
        elif tasks and not args.retry_incomplete:
            print("skip terminal-incomplete: " + ",".join(
                str(task["id"]) for task in tasks))
        else:
            specs.append(spec)

    if len(rows) != 26:
        raise AssertionError(f"unexpected audit chain size {len(rows)}")
    print(
        f"Shared-regime audit chain: total={len(rows)} "
        f"submit={len(specs)}", flush=True)
    if args.dry_run:
        for spec in specs:
            print(json.dumps({
                "signature": spec["signature"],
                "allowed_nodes": spec["allowed_nodes"],
                "wait_for_files": len(spec["wait_for_files"]),
                "vram": spec["vram"],
                "cpu": spec["cpu"],
            }, sort_keys=True))
        return
    if not specs:
        print("No shared-regime audit tasks to submit")
        return
    ids = scheduler_common.submit_jsonl(specs)
    print(f"Submitted task ids: {','.join(ids)}", flush=True)
    if args.dispatch:
        scheduler_common.dispatch(ids)


if __name__ == "__main__":
    main()
