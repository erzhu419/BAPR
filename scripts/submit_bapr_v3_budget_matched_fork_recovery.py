#!/usr/bin/env python3
"""Submit one source-archive-locked BAPR-v3 fork recovery task."""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import shlex
import sys
from pathlib import Path


_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if str(_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORT_ROOT))

import submit_bapr_v3_budget_matched_fork as submitter
from jax_experiments.analysis import (
    recover_bapr_v3_budget_matched_fork as recovery,
)
from jax_experiments.analysis import run_bapr_v3_budget_matched_fork as protocol


SIGNATURE_PREFIX = "BAPR/v3-budget-match-fork-recovery/v4"
ACTIVE_STATUSES = {"queued", "launching", "running"}
SUPPORTED_NODES = {
    "local", "jtl110gpu", "jtl110gpu2", "jtl311linux", "node007",
}


def signature(family: str, env: str) -> str:
    return f"{SIGNATURE_PREFIX}/{family}/{env}/seed0"


def inline_recovery() -> str:
    source = Path(recovery.__file__).read_bytes()
    payload = base64.b64encode(gzip.compress(source, mtime=0)).decode("ascii")
    return (
        "import base64,gzip;"
        "exec(compile(gzip.decompress(base64.b64decode("
        f"{payload!r})),\"<bapr-v3-fork-recovery>\",\"exec\"))"
    )


def task_spec(args: argparse.Namespace) -> dict:
    pair = protocol.pair_dir(args.family, args.env)
    values = [
        str(submitter.JAX_PYTHON), "-u", "-c", inline_recovery(),
        "--pair-dir", str(pair),
        "--family", args.family,
        "--env", args.env,
        "--seed", "0",
        "--resume",
    ]
    if args.resume_partial:
        values.append("--resume-partial")
    command = (
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.24 "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 JAX_NUM_THREADS=1 "
        "TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 "
        f"{shlex.join(values)}"
    )
    mode = "archive-resume" if args.resume_partial else "finalize-only"
    return {
        "description": (
            f"BAPR-v3 {mode} recovery {args.family} {args.env} seed0"),
        "cmd": command,
        "cwd": str(protocol.ROOT),
        "signature": signature(args.family, args.env),
        "project": "BAPR",
        "vram": 3200 if args.resume_partial else 512,
        "ram_mb": 6144,
        "cpu": 2,
        "priority": "high",
        "require_node": args.node,
        "require_gpu_idx": args.gpu_idx,
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=protocol.FAMILIES)
    parser.add_argument("--env", required=True, choices=protocol.ENVS)
    parser.add_argument("--node", required=True, choices=sorted(SUPPORTED_NODES))
    parser.add_argument("--gpu-idx", required=True, type=int)
    parser.add_argument("--resume-partial", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args(argv)

    sig = signature(args.family, args.env)
    matching = [
        task for task in submitter.scheduler_tasks()
        if str(task.get("signature") or "") == sig
    ]
    active_or_done = [
        task for task in matching
        if str(task.get("status")) in ACTIVE_STATUSES | {"done"}
    ]
    if active_or_done:
        states = ", ".join(
            f"{task.get('id')}={task.get('status')}" for task in active_or_done)
        raise SystemExit(f"REFUSED duplicate recovery {sig}: {states}")

    spec = task_spec(args)
    if args.dry_run:
        print(json.dumps(spec, indent=2, sort_keys=True), flush=True)
        return
    print(
        f"Submit {spec['signature']} on {args.node}:GPU{args.gpu_idx}",
        flush=True)
    task_ids = submitter.submit_jsonl([spec])
    print(f"Submitted task id: {task_ids[0]}", flush=True)
    if args.dispatch:
        submitter.dispatch(task_ids)


if __name__ == "__main__":
    main()
