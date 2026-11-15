#!/usr/bin/env python3
"""Submit strict Ant audits for all completed categorical-controller pairs."""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import shlex
import sys
from pathlib import Path


_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if str(_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORT_ROOT))

import submit_bapr_v3_budget_matched_fork_audit as common
from jax_experiments.analysis import (
    analyze_bapr_v3_budget_matched_fork_audit as audit,
)
from jax_experiments.analysis import (
    run_bapr_v3_budget_matched_fork as protocol,
)


ROOT = common.ROOT
JAX_PYTHON = common.JAX_PYTHON
PAIR_BASE = (
    ROOT / "jax_experiments" / "results_bapr_v3_ant_specialization_v1")
OUT_BASE = (
    ROOT / "jax_experiments"
    / "results_bapr_v3_ant_specialization_audit_v1")
SIGNATURE_PREFIX = "BAPR/v3-ant-specialization-audit/v1"
FINALIZE_SIGNATURE_PREFIX = "BAPR/v3-ant-specialization-finalize/v1"
FAMILIES = protocol.FAMILIES
VARIANTS = tuple(
    name for name, config in protocol.POLICY_VARIANTS.items()
    if config.get("policy_mode") == "categorical_expert")
SUPPORTED_NODES = {
    "local", "jtl110gpu", "jtl110gpu2", "jtl311linux", "node007",
}
ACTIVE_STATUSES = {"queued", "launching", "running"}
MAX_TASKS_PER_INVOCATION = (
    len(VARIANTS) * len(FAMILIES) * len(audit.DEFAULT_EVENT_SEEDS))
DEFAULT_AUDIT_VRAM_MB = 4200
NODE007_AUDIT_VRAM_MB = 6000
NODE007_GPU_LOCK = (
    "/tmp/bapr-v3-ant-specialization-audit-"
    "${CUDA_VISIBLE_DEVICES:-unknown}.lock"
)
AUDIT_VALIDATOR_RELATIVE = (
    "jax_experiments/analysis/"
    "analyze_bapr_v3_budget_matched_fork_audit.py")
AUDIT_VALIDATOR_SNAPSHOT_RELATIVE = (
    "jax_experiments/analysis/protocol_snapshots/"
    "analyze_bapr_v3_budget_matched_fork_audit_1779008896.py")
AUDIT_VALIDATOR_SNAPSHOT_SHA256 = (
    "1779008896b1b49d747e85192e8059fa7dcca9a00c513a8175f86bf27877185a")


def pair_dir(variant: str, family: str) -> Path:
    return protocol.pair_dir(
        family, "Ant-v2", 0, PAIR_BASE / variant)


def results_root(variant: str) -> Path:
    return OUT_BASE / variant


def output_dir(variant: str, family: str, event_seed: int) -> Path:
    return audit.group_directory(
        results_root(variant), family, "Ant-v2", event_seed)


def signature(variant: str, family: str, event_seed: int) -> str:
    return (
        f"{SIGNATURE_PREFIX}/{variant}/{family}/Ant-v2/"
        f"event-seed-{event_seed}")


def finalize_signature(variant: str, family: str) -> str:
    return (
        f"{FINALIZE_SIGNATURE_PREFIX}/{variant}/{family}/Ant-v2/seed0")


def resolve_pair_node(
    tasks: list[dict], variant: str, family: str,
) -> tuple[bool, str, str | None]:
    expected = finalize_signature(variant, family)
    evidence = [
        task for task in tasks
        if str(task.get("signature") or "") == expected
        and str(task.get("status")) == "done"
    ]
    if len(evidence) != 1:
        states = sorted(
            f"{task.get('id')}={task.get('status')}"
            for task in tasks
            if str(task.get("signature") or "") == expected)
        return (
            False,
            f"expected one completed finalize-only task, found {states}",
            None,
        )
    node = str(evidence[0].get("node") or "").strip()
    if node not in SUPPORTED_NODES:
        return False, f"unsupported producer node {node!r}", None
    return (
        True,
        f"finalized pair is authoritative on {node} via {evidence[0]['id']}",
        node,
    )


def validator_sync_values() -> list[str]:
    relative = AUDIT_VALIDATOR_RELATIVE
    snapshot = ROOT / AUDIT_VALIDATOR_SNAPSHOT_RELATIVE
    snapshot_bytes = snapshot.read_bytes()
    snapshot_hash = hashlib.sha256(snapshot_bytes).hexdigest()
    if snapshot_hash != AUDIT_VALIDATOR_SNAPSHOT_SHA256:
        raise RuntimeError(
            f"audit validator snapshot hash drift: {snapshot_hash}")
    payload = base64.b64encode(gzip.compress(
        snapshot_bytes, mtime=0)).decode("ascii")
    script = (
        "import base64,gzip,os,pathlib\n"
        f"root=pathlib.Path({str(ROOT)!r})\n"
        f"relative={relative!r}\n"
        f"payload={payload!r}\n"
        "path=root/relative\n"
        "data=gzip.decompress(base64.b64decode(payload))\n"
        "tmp=path.with_name(path.name+f'.audit-validator-sync.{os.getpid()}')\n"
        "tmp.write_bytes(data)\n"
        "os.replace(tmp,path)\n"
        "print('AUDIT VALIDATOR SYNC OK')\n"
    )
    return [str(JAX_PYTHON), "-c", script]


def task_spec(
    variant: str,
    family: str,
    event_seed: int,
    producer_node: str,
    priority: str,
) -> dict:
    pair = pair_dir(variant, family)
    group = output_dir(variant, family, event_seed)
    values = [
        "--pair-dir", str(pair),
        "--results-root", str(results_root(variant)),
        "--family", family,
        "--env", "Ant-v2",
        "--training-seed", "0",
        "--event-seed", str(event_seed),
        "--resume",
    ]
    common_env = (
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.20 "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 JAX_NUM_THREADS=1 "
        "TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 ")
    manifest = pair / "provenance" / "pair_manifest.json"
    validator_prefix = ""
    if producer_node != "local":
        validator_prefix = (
            f"{common_env}{shlex.join(validator_sync_values())} && ")
    audit_command = (
        f"{validator_prefix}{common_env}{audit.AUDIT_PAIR_MANIFEST_ENV}="
        f"{shlex.quote(str(manifest))} "
        f"{shlex.quote(str(JAX_PYTHON))} -u -m "
        "jax_experiments.analysis."
        "run_bapr_v3_budget_matched_fork_audit_group "
        f"{shlex.join(values)}")
    command = audit_command
    if producer_node == "node007":
        # The scheduler may lower an explicit VRAM reservation from history.
        # Keep the actual JAX work at one process per physical GPU regardless
        # of queue-side packing or concurrent dispatch passes.
        command = (
            f"flock --exclusive {NODE007_GPU_LOCK} "
            f"bash -lc {shlex.quote(audit_command)}"
        )
    return {
        "description": (
            f"BAPR-v3 Ant specialization strict audit {variant} {family} "
            f"event seed {event_seed}; six sequential controllers"),
        "cmd": command,
        "cwd": str(ROOT),
        "signature": signature(variant, family, event_seed),
        "project": "BAPR",
        # node007 exhausts host threads when four JAX evaluators share a GPU.
        # Its one-third-VRAM packing rule makes 6000 MB a one-task-per-GPU cap.
        "vram": (
            NODE007_AUDIT_VRAM_MB
            if producer_node == "node007"
            else DEFAULT_AUDIT_VRAM_MB
        ),
        "ram_mb": 4096,
        "cpu": 2,
        "priority": priority,
        "require_node": producer_node,
        "ckpt_dir": str(pair),
        "ckpt_glob": protocol.COMPLETE_SENTINEL_NAME,
        "resume_flag": "",
        "result_dir": str(group),
        "local_result_dir": str(group),
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
    parser.add_argument("--event-seed", action="append", type=int)
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="high")
    parser.add_argument(
        "--task-cap", type=int, default=MAX_TASKS_PER_INVOCATION)
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    variants = tuple(dict.fromkeys(args.variant or VARIANTS))
    families = tuple(dict.fromkeys(args.family or FAMILIES))
    event_seeds = tuple(args.event_seed or audit.DEFAULT_EVENT_SEEDS)
    if (len(event_seeds) != len(set(event_seeds))
            or any(seed not in audit.DEFAULT_EVENT_SEEDS
                   for seed in event_seeds)):
        raise SystemExit(
            f"event seeds must be a distinct subset of "
            f"{audit.DEFAULT_EVENT_SEEDS}")
    selected = len(variants) * len(families) * len(event_seeds)
    if not 1 <= args.task_cap <= MAX_TASKS_PER_INVOCATION:
        raise SystemExit(
            f"task-cap must be in [1,{MAX_TASKS_PER_INVOCATION}]")
    if selected > args.task_cap:
        raise SystemExit(
            f"selection expands to {selected} tasks; cap is {args.task_cap}")

    tasks = common.scheduler_tasks_strict()
    readiness = {
        (variant, family): resolve_pair_node(tasks, variant, family)
        for variant in variants for family in families
    }
    not_ready = [
        (variant, family, reason)
        for (variant, family), (ready, reason, _node) in readiness.items()
        if not ready
    ]
    if not_ready:
        for variant, family, reason in not_ready:
            print(f"  not ready {variant}/{family}: {reason}", flush=True)
        if not args.dry_run:
            raise SystemExit("all selected categorical pairs must be finalized")

    by_signature: dict[str, list[dict]] = {}
    for task in tasks:
        value = str(task.get("signature") or "")
        if value:
            by_signature.setdefault(value, []).append(task)

    specs = []
    skipped = []
    for variant in variants:
        for family in families:
            node = readiness[(variant, family)][2]
            if not node:
                continue
            for event_seed in event_seeds:
                sig = signature(variant, family, event_seed)
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
                    skipped.append((sig, "done; result sync/analysis pending"))
                elif terminal and not args.retry_incomplete:
                    ids = ",".join(str(task.get("id")) for task in terminal)
                    skipped.append((sig, f"terminal-incomplete:{ids}"))
                else:
                    specs.append(task_spec(
                        variant, family, event_seed, node, args.priority))

    print(
        f"BAPR-v3 Ant specialization audit: selected={selected} "
        f"submit={len(specs)} skip={len(skipped)}", flush=True)
    for spec in specs:
        print(
            f"  submit {spec['signature']} -> {spec['require_node']}",
            flush=True)
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
