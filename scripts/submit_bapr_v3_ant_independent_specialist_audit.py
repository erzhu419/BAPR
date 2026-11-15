#!/usr/bin/env python3
"""Queue strict audits and final analysis for independent Ant specialists."""
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
    analyze_bapr_v3_independent_specialists as analyzer,
)
from jax_experiments.analysis import (
    bapr_v3_independent_specialists as protocol,
)
from jax_experiments.analysis import (
    run_bapr_v3_independent_specialist_audit as audit,
)


JAX_PYTHON = scheduler_common.JAX_PYTHON
EVENT_SEEDS = analyzer.EVENT_SEEDS
AUDIT_SIGNATURE_PREFIX = "BAPR/v3-independent-specialist-audit/v4"
ANALYSIS_SIGNATURE_PREFIX = "BAPR/v3-independent-specialist-analysis/v4"
PREP_SIGNATURE_PREFIX = "BAPR/v3-independent-specialist-audit-prep/v4"
ACTIVE_STATUSES = scheduler_common.ACTIVE_STATUSES


def audit_output_dir(family: str, event_seed: int) -> Path:
    return protocol.AUDIT_BASE / family / "Ant" / f"event_seed_{event_seed}"


def analysis_output_dir(family: str) -> Path:
    return protocol.ANALYSIS_BASE / family / "Ant"


def audit_signature(family: str, event_seed: int) -> str:
    return (
        f"{AUDIT_SIGNATURE_PREFIX}/{family}/Ant-v2/"
        f"event-seed-{event_seed}")


def analysis_signature(family: str) -> str:
    return f"{ANALYSIS_SIGNATURE_PREFIX}/{family}/Ant-v2/seed0"


def prep_signature(family: str) -> str:
    return f"{PREP_SIGNATURE_PREFIX}/{family}/Ant-v2/seed0"


def bundle_manifests(family: str) -> list[Path]:
    root = protocol.family_bundle_root(family)
    return [
        root / "robust" / protocol.BUNDLE_MANIFEST,
        *(root / f"specialist_mode_{mode}" / protocol.BUNDLE_MANIFEST
          for mode in protocol.MODES),
    ]


def immutable_launch_command(
    family: str,
    launch_name: str,
    module: str,
    values: list[str],
    *,
    cpu_only: bool,
    scheduler_resume: bool = False,
) -> str:
    archive = protocol.family_audit_source_archive(family)
    launch_root = Path("/tmp") / (
        f"bapr-v3-independent-{launch_name}")
    child_argv = [str(JAX_PYTHON), "-u", "-m", module, *values]
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
        + ("env['JAX_PLATFORMS']='cpu'\n"
           "env['CUDA_VISIBLE_DEVICES']=''\n" if cpu_only else "")
        + f"os.execve({str(JAX_PYTHON)!r},{child_argv!r},env)\n"
    )
    resource_env = (
        "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' " if cpu_only else
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.28 "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' ")
    return (
        ("BAPR_SCHEDULER_RESUME=--resume " if scheduler_resume else "")
        + resource_env
        + "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 JAX_NUM_THREADS=1 "
        "TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 "
        f"{shlex.quote(str(JAX_PYTHON))} -c {shlex.quote(launcher)}"
    )


def audit_task_spec(family: str, event_seed: int, priority: str) -> dict:
    output = audit_output_dir(family, event_seed)
    values = [
        "--family", family,
        "--event-seed", str(event_seed),
        "--out-dir", str(output),
        "--resume",
    ]
    command = immutable_launch_command(
        family, f"audit-{family}-{event_seed}",
        "jax_experiments.analysis."
        "run_bapr_v3_independent_specialist_audit",
        values, cpu_only=True, scheduler_resume=True,
    )
    return {
        "description": (
            f"BAPR-v3 independent Ant specialist strict audit {family} "
            f"event seed {event_seed}"),
        "cmd": command,
        "cwd": str(protocol.family_bundle_root(family)),
        "signature": audit_signature(family, event_seed),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 8192,
        "cpu": 2,
        "priority": priority,
        # This sentinel expands to node001-node006 through scheduleurm's
        # soft HPC CPU pool; the audit must not consume GPU nodes.
        "require_node": "zhengliang-hpc",
        "ckpt_dir": str(protocol.family_bundle_root(family)),
        "ckpt_glob": protocol.AUDIT_READY_MARKER,
        "resume_flag": "",
        "wait_for_files": [str(
            protocol.family_bundle_root(family)
            / protocol.AUDIT_READY_MARKER)],
        "result_dir": str(output),
        "local_result_dir": str(output),
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "allow_initial_resume_scan_error": False,
        "reroute_on_node_down": True,
    }


def analysis_task_spec(family: str, priority: str) -> dict:
    output = analysis_output_dir(family)
    report = output / "report.md"
    summary = output / "summary.json"
    values = [
        "--family", family,
        "--results-root", str(protocol.AUDIT_BASE),
        "--output", str(report),
        "--json-output", str(summary),
    ]
    command = immutable_launch_command(
        family, f"analysis-{family}",
        "jax_experiments.analysis.analyze_bapr_v3_independent_specialists",
        values, cpu_only=True,
    )
    wait_for = [
        str(audit_output_dir(family, seed) / "group.json")
        for seed in EVENT_SEEDS]
    wait_for += [
        str(protocol.family_audit_source_manifest(family)),
        str(protocol.family_audit_source_archive(family)),
    ]
    return {
        "description": (
            f"BAPR-v3 independent Ant specialist final analysis {family}"),
        "cmd": command,
        "cwd": str(ROOT),
        "signature": analysis_signature(family),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 2048,
        "cpu": 1,
        "priority": priority,
        "require_node": "local",
        "ckpt_dir": str(protocol.AUDIT_BASE / family / "Ant"),
        "ckpt_glob": "event_seed_*/group.json",
        "resume_flag": "",
        "wait_for_files": wait_for,
        "result_dir": str(output),
        "local_result_dir": str(output),
        "allow_initial_resume_scan_error": True,
        "reroute_on_node_down": False,
    }


def prep_task_spec(family: str, priority: str) -> dict:
    root = protocol.family_bundle_root(family)
    marker = root / protocol.AUDIT_READY_MARKER
    values = ["--family", family]
    command = (
        f"PYTHONPATH={shlex.quote(str(ROOT))} JAX_PLATFORMS=cpu "
        "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
        "MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "
        f"{shlex.quote(str(JAX_PYTHON))} -u -m "
        "jax_experiments.analysis."
        "prepare_bapr_v3_independent_specialist_audit "
        f"{shlex.join(values)}"
    )
    wait_for = [
        *(str(path) for path in bundle_manifests(family)),
        str(protocol.specialist_bundle_dir(family, 0)
            / protocol.BUNDLE_SOURCE_MANIFEST),
        str(protocol.specialist_bundle_dir(family, 0)
            / protocol.BUNDLE_SOURCE_ARCHIVE),
    ]
    return {
        "description": (
            f"Validate and stage BAPR-v3 independent Ant bundles {family}"),
        "cmd": command,
        "cwd": str(ROOT),
        "signature": prep_signature(family),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 2048,
        "cpu": 1,
        "priority": priority,
        "require_node": "local",
        "wait_for_files": wait_for,
        "result_dir": str(root),
        "local_result_dir": str(root),
        "allow_initial_resume_scan_error": True,
        "reroute_on_node_down": False,
    }


def _known_by_signature() -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for task in scheduler_common.scheduler_tasks():
        value = str(task.get("signature") or "")
        if value:
            result.setdefault(value, []).append(task)
    return result


def _should_submit(
    signature: str, output_file: Path, known: dict[str, list[dict]],
    retry_incomplete: bool,
) -> tuple[bool, str]:
    if output_file.is_file():
        return False, "complete-output"
    tasks = known.get(signature, [])
    active = [
        task for task in tasks
        if str(task.get("status")) in ACTIVE_STATUSES]
    if active:
        return False, "active:" + ",".join(
            str(task.get("id")) for task in active)
    if tasks and not retry_incomplete:
        return False, "terminal-incomplete:" + ",".join(
            str(task.get("id")) for task in tasks)
    return True, "ready"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", action="append", choices=protocol.FAMILIES)
    parser.add_argument("--priority", choices=("low", "normal", "high"),
                        default="high")
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    families = tuple(dict.fromkeys(args.family or protocol.FAMILIES))
    known = _known_by_signature()
    specs = []
    skipped = []
    for family in families:
        sig = prep_signature(family)
        submit, reason = _should_submit(
            sig,
            protocol.family_bundle_root(family)
            / protocol.AUDIT_READY_MARKER,
            known, args.retry_incomplete)
        if submit:
            specs.append(prep_task_spec(family, args.priority))
        else:
            skipped.append((sig, reason))
        for event_seed in EVENT_SEEDS:
            sig = audit_signature(family, event_seed)
            submit, reason = _should_submit(
                sig, audit_output_dir(family, event_seed) / "group.json",
                known, args.retry_incomplete)
            if submit:
                specs.append(audit_task_spec(
                    family, event_seed, args.priority))
            else:
                skipped.append((sig, reason))
        sig = analysis_signature(family)
        submit, reason = _should_submit(
            sig, analysis_output_dir(family) / "summary.json",
            known, args.retry_incomplete)
        if submit:
            specs.append(analysis_task_spec(family, args.priority))
        else:
            skipped.append((sig, reason))

    print(
        f"Independent specialist pipeline: submit={len(specs)} "
        f"skip={len(skipped)}", flush=True)
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
