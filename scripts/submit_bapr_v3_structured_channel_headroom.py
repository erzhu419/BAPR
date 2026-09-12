#!/usr/bin/env python3
"""Submit the structured actuator-channel seed0 oracle-headroom screen."""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import submit_bapr_v3_budget_matched_fork as common
from jax_experiments.analysis import run_bapr_v3_budget_matched_fork as fork
from jax_experiments.analysis import (
    run_bapr_v3_structured_channel_headroom as protocol,
)


JAX_PYTHON = common.JAX_PYTHON
ACTIVE_STATUSES = common.ACTIVE_STATUSES
SIGNATURE_PREFIX = "BAPR/v3-structured-channel-headroom/v2"


def signature(env: str) -> str:
    return f"{SIGNATURE_PREFIX}/{env}/seed0"


def output_complete(env: str) -> tuple[bool, str]:
    pair = protocol.pair_dir(protocol.FAMILIES[0], env)
    complete, reason = common.output_complete(pair)
    if not complete:
        return complete, reason
    manifest = fork.read_json(pair / fork.PAIR_MANIFEST_REL)
    expected = fork.pair_identity(
        protocol.FAMILIES[0], env, 0, "direct")
    if manifest.get("identity") != expected:
        return False, "completed pair has the wrong protocol identity"
    return True, "complete"


def task_spec(env: str, priority: str) -> dict:
    family = protocol.FAMILIES[0]
    pair = protocol.pair_dir(family, env)
    values = [
        "--family", family,
        "--env", env,
        "--seed", "0",
        "--save-root", str(protocol.SAVE_ROOT),
        "--resume",
    ]
    command = (
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.24 "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 JAX_NUM_THREADS=1 "
        "TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 "
        f"{shlex.quote(str(JAX_PYTHON))} -u -m "
        "jax_experiments.analysis."
        "run_bapr_v3_structured_channel_headroom "
        f"{shlex.join(values)}"
    )
    return {
        "description": (
            f"BAPR-v3 structured-channel headroom {env} seed0; "
            "equal-budget robust versus true-mode oracle"
        ),
        "cmd": command,
        "cwd": str(ROOT),
        "signature": signature(env),
        "project": "BAPR",
        "vram": 3600,
        "ram_mb": 6144,
        "cpu": 2,
        "priority": priority,
        "ckpt_dir": str(pair),
        "ckpt_glob": fork.STATE_NAME,
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
    parser.add_argument("--env", action="append", choices=protocol.ENVS)
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="high")
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    envs = tuple(dict.fromkeys(args.env or protocol.ENVS))
    by_signature: dict[str, list[dict]] = {}
    for task in common.scheduler_tasks():
        value = str(task.get("signature") or "")
        if value:
            by_signature.setdefault(value, []).append(task)

    specs = []
    skipped = []
    for env in envs:
        sig = signature(env)
        complete, reason = output_complete(env)
        known = by_signature.get(sig, [])
        active = [
            task for task in known
            if str(task.get("status")) in ACTIVE_STATUSES
        ]
        terminal = [task for task in known if task not in active]
        if complete:
            skipped.append((sig, "complete"))
        elif active:
            skipped.append((
                sig,
                "active:" + ",".join(str(task["id"]) for task in active),
            ))
        elif terminal and not args.retry_incomplete:
            skipped.append((
                sig,
                "terminal-incomplete:"
                + ",".join(str(task["id"]) for task in terminal)
                + f"; {reason}",
            ))
        else:
            spec = task_spec(env, args.priority)
            if terminal:
                # The immutable fork state records the exact GPU runtime.  A
                # retry must return to that placement instead of merely finding
                # the checkpoint on the same host and selecting another card.
                previous = max(
                    terminal,
                    key=lambda task: float(task.get("finished_at") or 0.0),
                )
                node = previous.get("last_node") or previous.get("node")
                gpu_idx = previous.get("last_gpu_idx")
                if gpu_idx is None:
                    gpu_idx = previous.get("gpu_idx")
                if node is not None:
                    spec["require_node"] = str(node)
                if gpu_idx is not None:
                    spec["require_gpu_idx"] = int(gpu_idx)
            specs.append(spec)

    print(
        f"BAPR-v3 structured-channel headroom: submit={len(specs)} "
        f"skip={len(skipped)}",
        flush=True,
    )
    for spec in specs:
        print(f"  submit {spec['signature']}", flush=True)
        if args.dry_run:
            print(json.dumps(spec, indent=2, sort_keys=True), flush=True)
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
