#!/usr/bin/env python3
"""Finalize one completed BAPR-v3 Ant categorical pair without training."""
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

import submit_bapr_v3_budget_matched_fork as common
from jax_experiments.analysis import run_bapr_v3_budget_matched_fork as protocol


ROOT = common.ROOT
JAX_PYTHON = common.JAX_PYTHON
SAVE_BASE = (
    ROOT / "jax_experiments" / "results_bapr_v3_ant_specialization_v1")
SIGNATURE_PREFIX = "BAPR/v3-ant-specialization-finalize/v1"
ACTIVE_OR_DONE = {"queued", "launching", "running", "done"}
SUPPORTED_NODES = {
    "local", "jtl110gpu", "jtl110gpu2", "jtl311linux", "node007",
}
VARIANTS = tuple(
    name for name, config in protocol.POLICY_VARIANTS.items()
    if config.get("policy_mode") == "categorical_expert")


def save_root(variant: str) -> Path:
    return SAVE_BASE / variant


def pair_dir(variant: str, family: str) -> Path:
    return protocol.pair_dir(
        family, "Ant-v2", 0, save_root(variant))


def signature(variant: str, family: str) -> str:
    return f"{SIGNATURE_PREFIX}/{variant}/{family}/Ant-v2/seed0"


def validator_sync_values() -> list[str]:
    relatives = (
        "jax_experiments/analysis/run_bapr_v3_budget_matched_fork.py",
        "jax_experiments/analysis/analyze_bapr_v3_budget_matched_fork_audit.py",
    )
    payloads = {
        relative: base64.b64encode(gzip.compress(
            (ROOT / relative).read_bytes(), mtime=0)).decode("ascii")
        for relative in relatives
    }
    script = (
        "import base64,gzip,os,pathlib\n"
        f"root=pathlib.Path({str(ROOT)!r})\n"
        f"payloads={payloads!r}\n"
        "for relative,payload in payloads.items():\n"
        " path=root/relative\n"
        " data=gzip.decompress(base64.b64decode(payload))\n"
        " tmp=path.with_name(path.name+'.validator-sync')\n"
        " tmp.write_bytes(data)\n"
        " os.replace(tmp,path)\n"
        "print('VALIDATOR SYNC OK')\n"
    )
    return [str(JAX_PYTHON), "-c", script]


def task_spec(args: argparse.Namespace) -> dict:
    pair = pair_dir(args.variant, args.family)
    values = [
        str(JAX_PYTHON), "-u", "-m",
        "jax_experiments.analysis.run_bapr_v3_budget_matched_fork",
        "--family", args.family,
        "--env", "Ant-v2",
        "--seed", "0",
        "--save-root", str(save_root(args.variant)),
        "--policy-variant", args.variant,
        "--resume",
        "--finalize-existing",
    ]
    command = (
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.08 "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 JAX_NUM_THREADS=1 "
        "TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 "
        f"{shlex.join(validator_sync_values())} && {shlex.join(values)}"
    )
    return {
        "description": (
            f"BAPR-v3 finalize-only Ant specialization {args.variant} "
            f"{args.family} seed0"),
        "cmd": command,
        "cwd": str(ROOT),
        "signature": signature(args.variant, args.family),
        "project": "BAPR",
        "vram": 512,
        "ram_mb": 4096,
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--family", required=True, choices=protocol.FAMILIES)
    parser.add_argument("--node", required=True, choices=sorted(SUPPORTED_NODES))
    parser.add_argument("--gpu-idx", required=True, type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    sig = signature(args.variant, args.family)
    duplicates = [
        task for task in common.scheduler_tasks()
        if str(task.get("signature") or "") == sig
        and str(task.get("status")) in ACTIVE_OR_DONE
    ]
    if duplicates:
        states = ", ".join(
            f"{task.get('id')}={task.get('status')}" for task in duplicates)
        raise SystemExit(f"REFUSED duplicate finalize task {sig}: {states}")

    spec = task_spec(args)
    if args.dry_run:
        print(json.dumps(spec, indent=2, sort_keys=True), flush=True)
        return
    print(
        f"Submit {sig} on {args.node}:GPU{args.gpu_idx}", flush=True)
    task_ids = common.submit_jsonl([spec])
    print(f"Submitted task id: {task_ids[0]}", flush=True)
    if args.dispatch:
        common.dispatch(task_ids)


if __name__ == "__main__":
    main()
