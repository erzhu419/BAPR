#!/usr/bin/env python3
"""Submit the strict packet-loss/burst-torque controller-headroom screen."""
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
from jax_experiments.analysis import run_bapr_v3_budget_matched_fork as fork
from jax_experiments.analysis import run_bapr_v3_stochastic_headroom as protocol


ROOT = protocol.ROOT
JAX_PYTHON = common.JAX_PYTHON
FAMILIES = protocol.FAMILIES
ENVS = protocol.ENVS
ACTIVE_STATUSES = common.ACTIVE_STATUSES
SIGNATURE_PREFIX = "BAPR/v3-stochastic-headroom/v1"
MAX_TASKS_PER_INVOCATION = len(FAMILIES) * len(ENVS)
ORIGINAL_PLACEMENTS = {
    ("packet_loss", "Ant-v2"): ("jtl311linux", 0),
    ("packet_loss", "HalfCheetah-v2"): ("jtl311linux", 1),
    ("burst_torque", "Ant-v2"): ("node007", 0),
    ("burst_torque", "HalfCheetah-v2"): ("node007", 1),
}


def signature(family: str, env: str) -> str:
    return f"{SIGNATURE_PREFIX}/{family}/{env}/seed0"


def output_complete(family: str, env: str) -> tuple[bool, str]:
    pair = protocol.pair_dir(family, env)
    complete, reason = common.output_complete(pair)
    if not complete:
        return complete, reason
    manifest = fork.read_json(pair / fork.PAIR_MANIFEST_REL)
    expected = fork.pair_identity(family, env, 0, "direct")
    if manifest.get("identity") != expected:
        return False, "completed pair has the wrong protocol identity"
    return True, "complete"


def runner_values(
    family: str, env: str, *, finalize_existing: bool = False,
) -> list[str]:
    values = [
        "--family", family,
        "--env", env,
        "--seed", "0",
        "--save-root", str(protocol.SAVE_ROOT),
        "--resume",
    ]
    if finalize_existing:
        values.append("--finalize-existing")
    return values


def task_spec(
    family: str, env: str, priority: str, *, finalize_existing: bool = False,
) -> dict:
    pair = protocol.pair_dir(family, env)
    values = runner_values(
        family, env, finalize_existing=finalize_existing)
    command = (
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.24 "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 JAX_NUM_THREADS=1 "
        "TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 "
        f"{shlex.quote(str(JAX_PYTHON))} -u -m "
        "jax_experiments.analysis.run_bapr_v3_stochastic_headroom "
        f"{shlex.join(values)}"
    )
    spec = {
        "description": (
            f"BAPR-v3 strict stochastic headroom {family} {env} seed0; "
            + ("finalize-only validation, no training" if finalize_existing
               else "equal-budget robust versus privileged dynamic controller")),
        "cmd": command,
        "cwd": str(ROOT),
        "signature": (
            f"{SIGNATURE_PREFIX}-finalize/{family}/{env}/seed0"
            if finalize_existing else signature(family, env)),
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
        # The fork manifest intentionally binds a retry to the original
        # runtime. A node failure must wait for that node instead of silently
        # changing the causal pair.
        "reroute_on_node_down": False,
    }
    if finalize_existing:
        node, gpu_idx = ORIGINAL_PLACEMENTS[(family, env)]
        spec.update({
            "require_node": node,
            "require_gpu_idx": gpu_idx,
            "vram": 1800,
        })
    return spec


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", action="append", choices=FAMILIES)
    parser.add_argument("--env", action="append", choices=ENVS)
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="high")
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--task-cap", type=int, default=MAX_TASKS_PER_INVOCATION)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument(
        "--finalize-existing", action="store_true",
        help="Submit validator-only recovery on each pair's original GPU.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    families = tuple(dict.fromkeys(args.family or FAMILIES))
    envs = tuple(dict.fromkeys(args.env or ENVS))
    selected = len(families) * len(envs)
    if not 1 <= args.task_cap <= MAX_TASKS_PER_INVOCATION:
        raise SystemExit(
            f"task-cap must be in [1,{MAX_TASKS_PER_INVOCATION}]")
    if selected > args.task_cap:
        raise SystemExit(
            f"selection expands to {selected} tasks; cap is {args.task_cap}")

    by_signature: dict[str, list[dict]] = {}
    for task in common.scheduler_tasks():
        value = str(task.get("signature") or "")
        if value:
            by_signature.setdefault(value, []).append(task)

    specs = []
    skipped = []
    for family in families:
        for env in envs:
            sig = (
                f"{SIGNATURE_PREFIX}-finalize/{family}/{env}/seed0"
                if args.finalize_existing else signature(family, env))
            complete, reason = output_complete(family, env)
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
                specs.append(task_spec(
                    family, env, args.priority,
                    finalize_existing=args.finalize_existing))

    print(
        f"BAPR-v3 stochastic headroom: selected={selected} "
        f"submit={len(specs)} skip={len(skipped)}", flush=True)
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
