#!/usr/bin/env python3
"""Bootstrap and submit fully independent Ant mode specialists."""
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

from jax_experiments.analysis import (
    bapr_v3_independent_specialists as protocol,
)


JAX_PYTHON = scheduler_common.JAX_PYTHON
SIGNATURE_PREFIX = "BAPR/v3-independent-specialist/v2"
ACTIVE_STATUSES = scheduler_common.ACTIVE_STATUSES


def signature(family: str, mode: int) -> str:
    return f"{SIGNATURE_PREFIX}/{family}/Ant-v2/mode-{mode}/seed0"


def immutable_launch_command(run_dir: Path, values: list[str]) -> str:
    archive = run_dir / protocol.SOURCE_ARCHIVE_REL
    launch_root = run_dir / ".specialist_launch_source"
    child_argv = [
        str(JAX_PYTHON), "-u", "-m",
        "jax_experiments.analysis.run_bapr_v3_independent_specialist",
        *values,
    ]
    launcher = (
        "import os,pathlib,shutil,tarfile\n"
        f"archive=pathlib.Path({str(archive)!r})\n"
        f"launch=pathlib.Path({str(launch_root)!r})\n"
        "shutil.rmtree(launch,ignore_errors=True)\n"
        "launch.mkdir(parents=True)\n"
        "with tarfile.open(archive,'r:gz') as source:\n"
        " source.extractall(launch)\n"
        "os.chdir(launch)\n"
        "env=os.environ.copy()\n"
        f"env['BAPR_WORKSPACE_ROOT']={str(ROOT)!r}\n"
        "env['PYTHONPATH']=str(launch)\n"
        f"os.execve({str(JAX_PYTHON)!r},{child_argv!r},env)\n"
    )
    return (
        "BAPR_SCHEDULER_RESUME=--resume "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.24 "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 JAX_NUM_THREADS=1 "
        "TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 "
        f"{shlex.quote(str(JAX_PYTHON))} -c {shlex.quote(launcher)}"
    )


def task_spec(family: str, mode: int, priority: str) -> dict:
    run_dir = protocol.specialist_run_dir(family, mode)
    bundle = protocol.specialist_bundle_dir(family, mode)
    values = [
        "--run-dir", str(run_dir),
        "--family", family,
        "--mode", str(mode),
        "--resume",
    ]
    command = immutable_launch_command(run_dir, values)
    return {
        "description": (
            f"BAPR-v3 fully independent Ant specialist {family} mode {mode}; "
            "policy/critic/target/alpha/replay independent"),
        "cmd": command,
        "cwd": str(ROOT),
        "signature": signature(family, mode),
        "project": "BAPR",
        "vram": 3600,
        "ram_mb": 6144,
        "cpu": 2,
        "priority": priority,
        # The iter-699 provenance manifest covers both checkpoints and the
        # logger snapshot.  Stage the complete run directory so a remote
        # resume can validate and reconstruct both before training starts.
        "ckpt_dir": str(run_dir),
        "ckpt_glob": "checkpoints/train_state.pkl",
        "resume_flag": "",
        "resume_managed_by_cmd": True,
        "result_dir": str(bundle),
        "local_result_dir": str(bundle),
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "allow_initial_resume_scan_error": False,
        "reroute_on_node_down": True,
    }


def output_complete(family: str, mode: int) -> tuple[bool, str]:
    try:
        protocol.validate_bundle(
            protocol.specialist_bundle_dir(family, mode), family,
            "specialist", mode)
    except (OSError, ValueError, EOFError) as exc:
        return False, str(exc)
    return True, "complete"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", action="append", choices=protocol.FAMILIES)
    parser.add_argument("--mode", action="append", type=int,
                        choices=protocol.MODES)
    parser.add_argument("--priority", choices=("low", "normal", "high"),
                        default="high")
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    families = tuple(dict.fromkeys(args.family or protocol.FAMILIES))
    modes = tuple(dict.fromkeys(args.mode or protocol.MODES))
    for family in families:
        protocol.publish_robust_bundle(family)
        for mode in modes:
            protocol.bootstrap_specialist(family, mode)

    tasks = scheduler_common.scheduler_tasks()
    by_signature: dict[str, list[dict]] = {}
    for task in tasks:
        value = str(task.get("signature") or "")
        if value:
            by_signature.setdefault(value, []).append(task)

    specs = []
    skipped = []
    for family in families:
        for mode in modes:
            sig = signature(family, mode)
            complete, reason = output_complete(family, mode)
            known = by_signature.get(sig, [])
            active = [
                task for task in known
                if str(task.get("status")) in ACTIVE_STATUSES]
            terminal = [task for task in known if task not in active]
            if complete:
                skipped.append((sig, "complete"))
            elif active:
                skipped.append((sig, "active:" + ",".join(
                    str(task.get("id")) for task in active)))
            elif terminal and not args.retry_incomplete:
                skipped.append((
                    sig, "terminal-incomplete:" + ",".join(
                        str(task.get("id")) for task in terminal)
                    + f"; {reason}"))
            else:
                specs.append(task_spec(family, mode, args.priority))

    print(
        f"Independent Ant specialists: selected={len(families) * len(modes)} "
        f"submit={len(specs)} skip={len(skipped)}", flush=True)
    for spec in specs:
        print(f"  submit {spec['signature']}", flush=True)
        if args.dry_run:
            print(json.dumps(spec, indent=2, sort_keys=True))
    for sig, reason in skipped:
        print(f"  skip {reason}: {sig}", flush=True)
    if args.dry_run or not specs:
        return
    ids = scheduler_common.submit_jsonl(specs)
    print(f"Submitted task ids: {','.join(ids)}", flush=True)
    if args.dispatch:
        scheduler_common.dispatch(ids)


if __name__ == "__main__":
    main()
