#!/usr/bin/env python3
"""Submit strict per-pair analysis for completed Ant specialization audits."""
from __future__ import annotations

import argparse
import base64
import gzip
import os
import shlex
import sys
from pathlib import Path


_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if str(_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORT_ROOT))

import submit_bapr_v3_ant_specialization_audit as group_submit


common = group_submit.common
audit = group_submit.audit
ROOT = group_submit.ROOT
JAX_PYTHON = group_submit.JAX_PYTHON
PAIR_BASE = group_submit.PAIR_BASE
AUDIT_BASE = group_submit.OUT_BASE
ANALYSIS_BASE = (
    ROOT / "jax_experiments"
    / "results_bapr_v3_ant_specialization_analysis_v1")
SIGNATURE_PREFIX = "BAPR/v3-ant-specialization-analysis/v1"
VARIANTS = group_submit.VARIANTS
FAMILIES = group_submit.FAMILIES
ACTIVE_STATUSES = group_submit.ACTIVE_STATUSES
SOURCE_FILES = (
    "jax_experiments/analysis/"
    "analyze_bapr_v3_budget_matched_fork_audit.py",
    "scripts/analyze_bapr_v3_budget_matched_fork_pair.py",
)


def analysis_dir(variant: str, family: str) -> Path:
    return ANALYSIS_BASE / variant / family / "Ant"


def signature(variant: str, family: str) -> str:
    return f"{SIGNATURE_PREFIX}/{variant}/{family}/Ant-v2/seed0"


def source_sync_values() -> list[str]:
    payloads = {
        relative: base64.b64encode(gzip.compress(
            (ROOT / relative).read_bytes(), mtime=0)).decode("ascii")
        for relative in SOURCE_FILES
    }
    script = (
        "import base64,gzip,os,pathlib\n"
        f"root=pathlib.Path({str(ROOT)!r})\n"
        f"payloads={payloads!r}\n"
        "for relative,payload in payloads.items():\n"
        " path=root/relative\n"
        " path.parent.mkdir(parents=True,exist_ok=True)\n"
        " data=gzip.decompress(base64.b64decode(payload))\n"
        " tmp=path.with_name(path.name+f'.analysis-sync.{os.getpid()}')\n"
        " tmp.write_bytes(data)\n"
        " os.replace(tmp,path)\n"
        "print('ANT SPECIALIZATION ANALYZER SYNC OK')\n"
    )
    return [str(JAX_PYTHON), "-c", script]


def audit_readiness(
    tasks: list[dict], variant: str, family: str,
) -> tuple[bool, str, str | None]:
    pair_ready, pair_reason, producer_node = group_submit.resolve_pair_node(
        tasks, variant, family)
    if not pair_ready:
        return False, pair_reason, None

    missing = []
    for event_seed in audit.DEFAULT_EVENT_SEEDS:
        expected = group_submit.signature(variant, family, event_seed)
        if not any(
            str(task.get("signature") or "") == expected
            and str(task.get("status")) == "done"
            for task in tasks
        ):
            missing.append(event_seed)
    if missing:
        return (
            False,
            f"strict audit groups are not done for event seeds {missing}",
            producer_node,
        )
    return (
        True,
        f"all five strict groups done; {pair_reason}",
        producer_node,
    )


def task_spec(
    variant: str,
    family: str,
    producer_node: str,
    priority: str,
) -> dict:
    pair = group_submit.pair_dir(variant, family)
    pair_root = PAIR_BASE / variant
    results_root = AUDIT_BASE / variant
    destination = analysis_dir(variant, family)
    report = destination / "report.md"
    summary = destination / "summary.json"
    manifest = pair / "provenance" / "pair_manifest.json"
    values = [
        "--pair-root", str(pair_root),
        "--results-root", str(results_root),
        "--family", family,
        "--env", "Ant",
        "--output", str(report),
        "--json-output", str(summary),
    ]
    common_env = (
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 ")
    command = (
        f"{common_env}{shlex.join(source_sync_values())} && "
        f"{common_env}{audit.AUDIT_PAIR_MANIFEST_ENV}="
        f"{shlex.quote(str(manifest))} "
        f"{shlex.quote(str(JAX_PYTHON))} -u "
        f"{shlex.quote(str(ROOT / SOURCE_FILES[1]))} "
        f"{shlex.join(values)}"
    )
    return {
        "description": (
            f"BAPR-v3 Ant specialization strict pair analysis {variant} "
            f"{family}; JSON and Markdown only"),
        "cmd": command,
        "cwd": str(ROOT),
        "signature": signature(variant, family),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 2048,
        "cpu": 1,
        "priority": priority,
        "require_node": producer_node,
        "ckpt_dir": str(pair),
        "ckpt_glob": group_submit.protocol.COMPLETE_SENTINEL_NAME,
        "resume_flag": "",
        "result_dir": str(destination),
        "local_result_dir": str(destination),
        "stage_excludes": [
            "jax_experiments/eval_bundles*/",
            "jax_experiments/results*/",
            "paper/",
        ],
        "allow_remote_large_data": True,
        "allow_initial_resume_scan_error": False,
        "reroute_on_node_down": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", action="append", choices=VARIANTS)
    parser.add_argument("--family", action="append", choices=FAMILIES)
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="high")
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    variants = tuple(dict.fromkeys(args.variant or VARIANTS))
    families = tuple(dict.fromkeys(args.family or FAMILIES))
    tasks = common.scheduler_tasks_strict()
    by_signature: dict[str, list[dict]] = {}
    for task in tasks:
        value = str(task.get("signature") or "")
        if value:
            by_signature.setdefault(value, []).append(task)

    specs = []
    skipped = []
    for variant in variants:
        for family in families:
            ready, reason, node = audit_readiness(tasks, variant, family)
            sig = signature(variant, family)
            known = by_signature.get(sig, [])
            active = [
                task for task in known
                if str(task.get("status")) in ACTIVE_STATUSES]
            done = [
                task for task in known
                if str(task.get("status")) == "done"]
            terminal = [
                task for task in known
                if str(task.get("status")) not in ACTIVE_STATUSES
                and str(task.get("status")) != "done"]
            if active:
                skipped.append((sig, "active"))
            elif done:
                skipped.append((sig, "done"))
            elif not ready or not node:
                skipped.append((sig, f"not-ready: {reason}"))
            elif terminal and not args.retry_incomplete:
                ids = ",".join(str(task.get("id")) for task in terminal)
                skipped.append((sig, f"terminal-incomplete:{ids}"))
            else:
                specs.append(task_spec(
                    variant, family, node, args.priority))

    print(
        f"BAPR-v3 Ant specialization analysis: submit={len(specs)} "
        f"skip={len(skipped)}",
        flush=True,
    )
    for spec in specs:
        print(
            f"  submit {spec['signature']} -> {spec['require_node']}",
            flush=True,
        )
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
