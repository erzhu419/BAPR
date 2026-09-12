#!/usr/bin/env python3
"""Submit strict audits for the two stochastic independent-specialist ladders."""
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
    analyze_bapr_v3_independent_specialists as analyzer,
)
from jax_experiments.analysis import (
    bapr_v3_independent_specialists as protocol,
)


JAX_PYTHON = scheduler_common.JAX_PYTHON
EVENT_SEEDS = analyzer.EVENT_SEEDS
SIGNATURE_PREFIX = (
    "BAPR/v3-stochastic-independent-specialist-audit/v1"
)
STATUS_BASE = (
    ROOT / "jax_experiments"
    / "results_bapr_v3_stochastic_independent_specialist_audit_status_v1"
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


def prep_signature(target: Target) -> str:
    return (
        f"{SIGNATURE_PREFIX}/prep/{target.family}/{target.env}/seed0"
    )


def audit_signature(target: Target, event_seed: int) -> str:
    return (
        f"{SIGNATURE_PREFIX}/group/{target.family}/{target.env}/"
        f"event-seed-{event_seed}"
    )


def analysis_signature(target: Target) -> str:
    return (
        f"{SIGNATURE_PREFIX}/analysis/{target.family}/{target.env}/seed0"
    )


def prep_status_dir(target: Target) -> Path:
    return STATUS_BASE / target.family / env_short(target) / "prep"


def audit_output_dir(target: Target, event_seed: int) -> Path:
    configure(target)
    return (
        protocol.AUDIT_BASE / target.family / env_short(target)
        / f"event_seed_{event_seed}"
    )


def analysis_output_dir(target: Target) -> Path:
    configure(target)
    return protocol.ANALYSIS_BASE / target.family / env_short(target)


def bundle_manifests(target: Target) -> list[Path]:
    configure(target)
    root = protocol.family_bundle_root(target.family)
    return [
        root / "robust" / protocol.BUNDLE_MANIFEST,
        *(
            root / f"specialist_mode_{mode}" / protocol.BUNDLE_MANIFEST
            for mode in protocol.MODES
        ),
    ]


def immutable_launch_command(
    target: Target,
    launch_name: str,
    module: str,
    values: list[str],
    *,
    cpu_only: bool,
    scheduler_resume: bool = False,
) -> str:
    configure(target)
    archive = protocol.family_audit_source_archive(target.family)
    launch_root = Path("/tmp") / (
        f"bapr-v3-{target.profile}-independent-{launch_name}"
    )
    launcher_values = [
        "--archive", str(archive),
        "--launch-dir", str(launch_root),
        "--module", module,
        "--workspace-root", str(ROOT),
    ]
    if cpu_only:
        launcher_values.append("--cpu-only")
    launcher_values += ["--", *values]
    resource_env = (
        "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' "
        if cpu_only
        else (
            "XLA_PYTHON_CLIENT_PREALLOCATE=false "
            "XLA_PYTHON_CLIENT_MEM_FRACTION=0.28 "
            "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
        )
    )
    command = (
        ("BAPR_SCHEDULER_RESUME=--resume " if scheduler_resume else "")
        + resource_env
        + "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 JAX_NUM_THREADS=1 "
        "TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 "
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        f"{shlex.quote(str(JAX_PYTHON))} -u -m "
        "jax_experiments.analysis."
        "launch_bapr_v3_independent_specialist_audit "
        f"{shlex.join(launcher_values)}"
    )
    return command + " && echo DONE"


def prep_task_spec(target: Target, priority: str) -> dict:
    configure(target)
    status_dir = prep_status_dir(target)
    values = [
        "--profile", target.profile,
        "--family", target.family,
        "--env", target.env,
        "--status-dir", str(status_dir),
    ]
    command = (
        f"PYTHONPATH={shlex.quote(str(ROOT))} JAX_PLATFORMS=cpu "
        "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
        "MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "
        f"{shlex.quote(str(JAX_PYTHON))} -u -m "
        "jax_experiments.analysis."
        "prepare_bapr_v3_independent_specialist_audit "
        f"{shlex.join(values)} && echo DONE"
    )
    return {
        "description": (
            f"Validate independent {target.family}/{target.env} bundles and "
            "freeze strict-audit source; no training"
        ),
        "cmd": command,
        "cwd": str(ROOT),
        "signature": prep_signature(target),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 4096,
        "cpu": 2,
        "priority": priority,
        "require_node": target.data_node,
        "result_dir": str(status_dir),
        "local_result_dir": str(status_dir),
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Validation and immutable-source packaging only; no rollout, "
            "gradient update, or model training is executed."
        ),
        "allow_initial_resume_scan_error": False,
        "reroute_on_node_down": False,
    }


def audit_task_spec(
    target: Target, event_seed: int, priority: str,
) -> dict:
    configure(target)
    output = audit_output_dir(target, event_seed)
    root = protocol.family_bundle_root(target.family)
    values = [
        "--profile", target.profile,
        "--family", target.family,
        "--env", target.env,
        "--event-seed", str(event_seed),
        "--out-dir", str(output),
        "--resume",
    ]
    command = immutable_launch_command(
        target,
        f"audit-{target.name}-{event_seed}",
        "jax_experiments.analysis."
        "run_bapr_v3_independent_specialist_audit",
        values,
        cpu_only=False,
    )
    return {
        "description": (
            f"Strict independent-specialist audit {target.family}/"
            f"{target.env}, paired event seed {event_seed}"
        ),
        "cmd": command,
        "cwd": str(ROOT),
        "signature": audit_signature(target, event_seed),
        "project": "BAPR",
        "vram": 4200,
        "ram_mb": 8192,
        "cpu": 2,
        "priority": priority,
        "require_node": target.data_node,
        "result_dir": str(output),
        "local_result_dir": str(output),
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "allow_initial_resume_scan_error": False,
        "reroute_on_node_down": False,
    }


def analysis_task_spec(target: Target, priority: str) -> dict:
    configure(target)
    output = analysis_output_dir(target)
    report = output / "report.md"
    summary = output / "summary.json"
    values = [
        "--profile", target.profile,
        "--family", target.family,
        "--env", target.env,
        "--results-root", str(protocol.AUDIT_BASE),
        "--output", str(report),
        "--json-output", str(summary),
    ]
    command = immutable_launch_command(
        target,
        f"analysis-{target.name}",
        "jax_experiments.analysis."
        "analyze_bapr_v3_independent_specialists",
        values,
        cpu_only=True,
    )
    return {
        "description": (
            f"Analyze independent-specialist gate for {target.family}/"
            f"{target.env}"
        ),
        "cmd": command,
        "cwd": str(ROOT),
        "signature": analysis_signature(target),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 4096,
        "cpu": 1,
        "priority": priority,
        "require_node": target.data_node,
        "result_dir": str(output),
        "local_result_dir": str(output),
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Aggregation and artifact validation only; no rollout, gradient "
            "update, or model training is executed."
        ),
        "allow_initial_resume_scan_error": True,
        "reroute_on_node_down": False,
    }


def _known_by_signature() -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for task in scheduler_common.scheduler_tasks():
        signature = str(task.get("signature") or "")
        if signature:
            result.setdefault(signature, []).append(task)
    return result


def _should_submit(
    signature: str,
    output_file: Path,
    known: dict[str, list[dict]],
    retry_incomplete: bool,
) -> tuple[bool, str]:
    if output_file.is_file():
        return False, "complete-output"
    tasks = known.get(signature, [])
    active = [
        task for task in tasks
        if str(task.get("status")) in ACTIVE_STATUSES
    ]
    if active:
        return False, "active:" + ",".join(
            str(task.get("id")) for task in active
        )
    if tasks and not retry_incomplete:
        return False, "terminal-incomplete:" + ",".join(
            str(task.get("id")) for task in tasks
        )
    return True, "ready"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", action="append", choices=tuple(TARGETS))
    parser.add_argument(
        "--phase",
        choices=("prep", "audit", "analysis"),
        required=True,
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
    targets = [
        TARGETS[name]
        for name in dict.fromkeys(args.target or TARGETS)
    ]
    phases = (args.phase,)
    known = _known_by_signature()
    specs = []
    skipped = []
    for target in targets:
        configure(target)
        candidates: list[tuple[str, dict, Path]] = []
        if "prep" in phases:
            candidates.append((
                prep_signature(target),
                prep_task_spec(target, args.priority),
                prep_status_dir(target) / "audit_ready_summary.json",
            ))
        if "audit" in phases:
            for event_seed in EVENT_SEEDS:
                candidates.append((
                    audit_signature(target, event_seed),
                    audit_task_spec(target, event_seed, args.priority),
                    audit_output_dir(target, event_seed) / "group.json",
                ))
        if "analysis" in phases:
            candidates.append((
                analysis_signature(target),
                analysis_task_spec(target, args.priority),
                analysis_output_dir(target) / "summary.json",
            ))
        for signature, spec, output_file in candidates:
            submit, reason = _should_submit(
                signature,
                output_file,
                known,
                args.retry_incomplete,
            )
            if submit:
                specs.append(spec)
            else:
                skipped.append((signature, reason))

    print(
        f"Stochastic independent-specialist audit phase={args.phase}: "
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
