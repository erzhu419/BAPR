#!/usr/bin/env python3
"""Submit the two positive-headroom stochastic specialist diagnostics."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import submit_bapr_v3_budget_matched_fork as scheduler_common

if Path(sys.executable).resolve() != scheduler_common.JAX_PYTHON.resolve():
    os.execv(
        str(scheduler_common.JAX_PYTHON),
        [
            str(scheduler_common.JAX_PYTHON),
            str(Path(__file__).resolve()),
            *sys.argv[1:],
        ],
    )

from jax_experiments.analysis import (
    bapr_v3_independent_specialists as protocol,
)


JAX_PYTHON = scheduler_common.JAX_PYTHON
SIGNATURE_PREFIX = "BAPR/v3-stochastic-independent-specialist/v1"
STATUS_BASE = (
    ROOT / "jax_experiments"
    / "results_bapr_v3_stochastic_independent_specialist_status_v1"
)
ACTIVE_STATUSES = scheduler_common.ACTIVE_STATUSES


@dataclass(frozen=True)
class Target:
    name: str
    family: str
    env: str
    data_node: str
    profile: str = "stochastic_headroom"


TARGETS = {
    "packet-ant": Target(
        "packet-ant", "packet_loss", "Ant-v2", "jtl311linux"
    ),
    "burst-halfcheetah": Target(
        "burst-halfcheetah",
        "burst_torque",
        "HalfCheetah-v2",
        "node007",
    ),
}


def configure(target: Target) -> None:
    if target.profile == "structured_channel":
        protocol.configure_structured_channel_headroom(target.env)
    elif target.profile == "stochastic_headroom":
        protocol.configure_stochastic_headroom(target.env)
    else:
        raise ValueError(f"unsupported specialist profile {target.profile!r}")


def env_short(target: Target) -> str:
    return target.env.removesuffix("-v2")


def bootstrap_signature(target: Target) -> str:
    return (
        f"{SIGNATURE_PREFIX}/bootstrap/{target.family}/{target.env}/seed0"
    )


def train_signature(target: Target, mode: int) -> str:
    return (
        f"{SIGNATURE_PREFIX}/train/{target.family}/{target.env}/"
        f"mode-{mode}/seed0"
    )


def smoke_signature(target: Target) -> str:
    return (
        f"{SIGNATURE_PREFIX}/smoke/{target.family}/{target.env}/"
        "mode-0/seed0"
    )


def status_dir(target: Target, mode: int) -> Path:
    return (
        STATUS_BASE / target.family / env_short(target)
        / f"specialist_mode_{mode}"
    )


def bootstrap_spec(target: Target, priority: str) -> dict:
    configure(target)
    pair = protocol.source_pair(target.family)
    command = (
        f"PYTHONPATH={shlex.quote(str(ROOT))} JAX_PLATFORMS=cpu "
        "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
        "MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "
        f"{shlex.quote(str(JAX_PYTHON))} -u -m "
        "jax_experiments.analysis."
        "prepare_bapr_v3_stochastic_independent_specialists "
        f"--profile {shlex.quote(target.profile)} "
        f"--family {shlex.quote(target.family)} "
        f"--env {shlex.quote(target.env)} && echo DONE"
    )
    return {
        "description": (
            f"Bootstrap independent specialists beside {target.family}/"
            f"{target.env} source pair; no training"
        ),
        "cmd": command,
        "cwd": str(ROOT),
        "signature": bootstrap_signature(target),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 4096,
        "cpu": 2,
        "priority": priority,
        "require_node": target.data_node,
        "ckpt_dir": str(pair),
        "ckpt_glob": "pair_checkpoint_complete.pkl",
        "resume_flag": "",
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Bootstrap only: validate and copy completed checkpoints; no "
            "rollout, gradient update, or model training is executed."
        ),
        "allow_initial_resume_scan_error": False,
        "reroute_on_node_down": False,
    }


def immutable_train_command(
    target: Target, run_dir: Path, mode: int, compact_status: Path,
    *, target_next_iteration: int | None = None, smoke: bool = False,
) -> str:
    values = [
        "--run-dir",
        str(run_dir),
        "--family",
        target.family,
        "--profile",
        target.profile,
        "--env",
        target.env,
        "--mode",
        str(mode),
        "--status-dir",
        str(compact_status),
        "--resume",
    ]
    if target_next_iteration is not None:
        values += [
            "--target-next-iteration", str(target_next_iteration)
        ]
    if smoke:
        values.append("--smoke")
    return (
        "BAPR_SCHEDULER_RESUME=--resume "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.20 "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 JAX_NUM_THREADS=1 "
        "TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 "
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        f"{shlex.quote(str(JAX_PYTHON))} -u -m "
        "jax_experiments.analysis."
        "launch_bapr_v3_stochastic_independent_specialist "
        f"{shlex.join(values)}"
    )


def train_spec(target: Target, mode: int, priority: str) -> dict:
    configure(target)
    run_dir = protocol.specialist_run_dir(target.family, mode)
    compact_status = status_dir(target, mode)
    return {
        "description": (
            f"Independent {target.family}/{target.env} specialist mode {mode}; "
            "policy/critic/target/alpha/replay independent"
        ),
        "cmd": immutable_train_command(
            target, run_dir, mode, compact_status
        ),
        "cwd": str(ROOT),
        "signature": train_signature(target, mode),
        "project": "BAPR",
        "vram": 1500,
        "ram_mb": 4096,
        "cpu": 2,
        "priority": priority,
        # All five bundles must remain together for the strict paired audit.
        # Both data nodes can run their four specialists concurrently.
        "require_node": target.data_node,
        "ckpt_dir": str(run_dir),
        "ckpt_glob": "checkpoints/train_state.pkl",
        "resume_flag": "",
        "resume_managed_by_cmd": True,
        "result_dir": str(compact_status),
        "local_result_dir": str(compact_status),
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "allow_initial_resume_scan_error": False,
        "reroute_on_node_down": False,
    }


def smoke_spec(target: Target, priority: str) -> dict:
    configure(target)
    mode = 0
    run_dir = protocol.specialist_run_dir(target.family, mode)
    compact_status = status_dir(target, mode) / "smoke"
    return {
        "description": (
            f"Two-iteration independent specialist smoke for "
            f"{target.family}/{target.env} mode 0"
        ),
        "cmd": immutable_train_command(
            target,
            run_dir,
            mode,
            compact_status,
            target_next_iteration=protocol.BASE_NEXT_ITERATION + 2,
            smoke=True,
        ),
        "cwd": str(ROOT),
        "signature": smoke_signature(target),
        "project": "BAPR",
        "vram": 1500,
        "ram_mb": 4096,
        "cpu": 2,
        "priority": priority,
        "require_node": target.data_node,
        "ckpt_dir": str(run_dir),
        "ckpt_glob": "checkpoints/train_state.pkl",
        "resume_flag": "",
        "resume_managed_by_cmd": True,
        "result_dir": str(compact_status),
        "local_result_dir": str(compact_status),
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "allow_initial_resume_scan_error": False,
        "reroute_on_node_down": False,
    }


def _known_by_signature() -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for task in scheduler_common.scheduler_tasks():
        signature = str(task.get("signature") or "")
        if signature:
            result.setdefault(signature, []).append(task)
    return result


def _active(known: dict[str, list[dict]], signature: str) -> list[dict]:
    return [
        task
        for task in known.get(signature, [])
        if str(task.get("status")) in ACTIVE_STATUSES
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", action="append", choices=tuple(TARGETS)
    )
    parser.add_argument(
        "--phase", choices=("bootstrap", "smoke", "train"), required=True
    )
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="high"
    )
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = [TARGETS[name] for name in dict.fromkeys(
        args.target or TARGETS
    )]
    known = _known_by_signature()
    specs = []
    skipped = []
    for target in targets:
        if args.phase == "bootstrap":
            candidates = [(
                bootstrap_signature(target),
                bootstrap_spec(target, args.priority),
                None,
            )]
        elif args.phase == "smoke":
            marker = status_dir(target, 0) / "smoke" / (
                "specialist_smoke_summary.json"
            )
            candidates = [(
                smoke_signature(target),
                smoke_spec(target, args.priority),
                marker,
            )]
        else:
            candidates = []
            for mode in protocol.MODES:
                marker = status_dir(target, mode) / (
                    "specialist_complete_summary.json"
                )
                candidates.append((
                    train_signature(target, mode),
                    train_spec(target, mode, args.priority),
                    marker,
                ))
        for signature, spec, marker in candidates:
            active = _active(known, signature)
            prior = known.get(signature, [])
            if marker is not None and marker.is_file():
                skipped.append((signature, "complete"))
            elif active:
                skipped.append((
                    signature,
                    "active:" + ",".join(str(task["id"]) for task in active),
                ))
            elif prior and not args.retry_incomplete:
                skipped.append((
                    signature,
                    "terminal-incomplete:" + ",".join(
                        str(task["id"]) for task in prior
                    ),
                ))
            else:
                specs.append(spec)

    print(
        f"Stochastic independent specialists phase={args.phase}: "
        f"submit={len(specs)} skip={len(skipped)}",
        flush=True,
    )
    for spec in specs:
        print(f"  submit {spec['signature']}", flush=True)
        if args.dry_run:
            print(json.dumps(spec, indent=2, sort_keys=True))
    for signature, reason in skipped:
        print(f"  skip {reason}: {signature}", flush=True)
    if args.dry_run or not specs:
        return
    ids = scheduler_common.submit_jsonl(specs)
    print(f"Submitted task ids: {','.join(ids)}", flush=True)
    if args.dispatch:
        scheduler_common.dispatch(ids)


if __name__ == "__main__":
    main()
