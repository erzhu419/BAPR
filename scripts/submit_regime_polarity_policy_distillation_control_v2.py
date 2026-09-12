#!/usr/bin/env python3
"""Submit closed-loop policy-compression v2 training and strict audits."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import submit_bapr_v3_budget_matched_fork as scheduler_common

if (__name__ == "__main__"
        and Path(sys.executable).resolve()
        != scheduler_common.JAX_PYTHON.resolve()):
    os.execv(
        str(scheduler_common.JAX_PYTHON),
        [str(scheduler_common.JAX_PYTHON), str(Path(__file__).resolve()),
         *sys.argv[1:]],
    )

from jax_experiments.analysis import (
    regime_polarity_policy_distillation_control_v2 as protocol,
)


SIGNATURE_PREFIX = "BAPR/regime-polarity-policy-distillation-control/v2"
SUBMIT_INTENT_LABEL = "bapr-polarity-policy-compression-v2-submit"
GPU_NODES = ["jtl311linux"]
CPU_NODES = [f"node00{index}" for index in range(1, 7)]
ACTIVE_STATUSES = scheduler_common.ACTIVE_STATUSES


def _gpu_command(module: str, values: list[str]) -> str:
    return (
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.18 "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        f"{shlex.quote(str(scheduler_common.JAX_PYTHON))} -u -m "
        f"{module} {shlex.join(values)} && echo DONE"
    )


def _cpu_command(module: str, values: list[str]) -> str:
    return (
        "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' "
        "XLA_FLAGS='--xla_cpu_multi_thread_eigen=false "
        "intra_op_parallelism_threads=1' "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 JAX_NUM_THREADS=1 "
        "JAX_CPU_ENABLE_ASYNC_DISPATCH=false "
        "TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 "
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        f"{shlex.quote(str(scheduler_common.JAX_PYTHON))} -u -m "
        f"{module} {shlex.join(values)} && echo DONE"
    )


def train_signature(variant: str, student_seed: int) -> str:
    return (
        f"{SIGNATURE_PREFIX}/train/{protocol.require_variant(variant)}/"
        f"seed-{protocol.require_student_seed(student_seed)}")


def audit_signature(variant: str, student_seed: int, event_seed: int) -> str:
    return (
        f"{SIGNATURE_PREFIX}/audit/{protocol.require_variant(variant)}/"
        f"seed-{protocol.require_student_seed(student_seed)}/"
        f"event-{protocol.require_audit_event_seed(event_seed)}")


def analysis_signature() -> str:
    return f"{SIGNATURE_PREFIX}/analysis"


def _source_wait_files(variant: str) -> list[str]:
    return [
        str(protocol.ensemble.final.MODEL_MANIFEST),
        str(protocol.ensemble.final.MODEL_PATH),
        *(str(path) for path in protocol.source_required_paths(variant)),
    ]


def _source_stage_inputs(variant: str) -> list[str]:
    return [
        str(protocol.ensemble.final.MODEL_MANIFEST.parent),
        *(str(path) for path in protocol.source_bundle_dirs(variant)),
    ]


def train_spec(variant: str, student_seed: int, priority: str):
    variant = protocol.require_variant(variant)
    student_seed = protocol.require_student_seed(student_seed)
    output = protocol.model_dir(variant, student_seed)
    return {
        "description": (
            f"Closed-loop policy compression {variant} seed {student_seed}"),
        "cmd": _gpu_command(
            "jax_experiments.analysis."
            "train_regime_polarity_policy_distillation_control_v2",
            ["--variant", variant,
             "--student-seed", str(student_seed), "--resume"],
        ),
        "cwd": str(ROOT),
        "signature": train_signature(variant, student_seed),
        "project": "BAPR",
        "vram": 2400,
        "ram_mb": 12_288,
        "cpu": 2,
        "priority": priority,
        "allowed_nodes": GPU_NODES,
        "result_dir": str(output),
        "local_result_dir": str(output),
        "ckpt_dir": str(protocol.work_dir(variant, student_seed)),
        "wait_for_files": _source_wait_files(variant),
        "stage_input_paths": _source_stage_inputs(variant),
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
    }


def audit_spec(
    variant: str,
    student_seed: int,
    event_seed: int,
    priority: str,
):
    variant = protocol.require_variant(variant)
    student_seed = protocol.require_student_seed(student_seed)
    event_seed = protocol.require_audit_event_seed(event_seed)
    output = protocol.audit_dir(variant, student_seed, event_seed)
    return {
        "description": (
            f"Closed-loop compression audit {variant} seed {student_seed} "
            f"event {event_seed}"),
        "cmd": _cpu_command(
            "jax_experiments.analysis."
            "run_regime_polarity_policy_distillation_control_audit",
            ["--variant", variant,
             "--student-seed", str(student_seed),
             "--event-seed", str(event_seed), "--resume"],
        ),
        "cwd": str(ROOT),
        "signature": audit_signature(variant, student_seed, event_seed),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 8192,
        "cpu": 32,
        "priority": priority,
        "allowed_nodes": CPU_NODES,
        "result_dir": str(output),
        "local_result_dir": str(output),
        "wait_for_files": [
            *_source_wait_files(variant),
            str(protocol.model_manifest(variant, student_seed)),
            str(protocol.model_path(variant, student_seed)),
        ],
        "stage_input_paths": [
            *_source_stage_inputs(variant),
            str(protocol.model_dir(variant, student_seed)),
        ],
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Strict checkpoint-only deterministic policy audit; no updates."),
    }


def analysis_spec(priority: str):
    return {
        "description": "Aggregate closed-loop policy-compression v2 sweep",
        "cmd": _cpu_command(
            "jax_experiments.analysis."
            "analyze_regime_polarity_policy_distillation_control_v2",
            [],
        ),
        "cwd": str(ROOT),
        "signature": analysis_signature(),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 2048,
        "cpu": 2,
        "priority": priority,
        "allowed_nodes": CPU_NODES,
        "result_dir": str(protocol.ANALYSIS_ROOT),
        "local_result_dir": str(protocol.ANALYSIS_ROOT),
        "wait_for_files": [str(path) for path in protocol.all_audit_manifests()],
        "stage_input_paths": [str(protocol.AUDIT_ROOT), str(protocol.MODEL_ROOT)],
        "stage_excludes": ["paper/"],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": "JSON aggregation only.",
    }


def candidates(phase: str, priority: str):
    rows = []
    if phase in ("all", "train"):
        rows.extend([
            (
                train_signature(variant, student_seed),
                train_spec(variant, student_seed, priority),
                protocol.model_manifest(variant, student_seed),
            )
            for variant in protocol.VARIANTS
            for student_seed in protocol.STUDENT_SEEDS
        ])
    if phase in ("all", "audit"):
        rows.extend([
            (
                audit_signature(variant, student_seed, event_seed),
                audit_spec(variant, student_seed, event_seed, priority),
                protocol.audit_manifest(variant, student_seed, event_seed),
            )
            for variant in protocol.VARIANTS
            for student_seed in protocol.STUDENT_SEEDS
            for event_seed in protocol.AUDIT_EVENT_SEEDS
        ])
    if phase in ("all", "analysis"):
        rows.append((analysis_signature(), analysis_spec(priority),
                     protocol.analysis_json()))
    return rows


def _submit_jsonl(specs):
    payload = "".join(json.dumps(spec) + "\n" for spec in specs)
    command = [
        sys.executable,
        str(scheduler_common.SCHEDULER),
        "submit-jsonl",
        "--stdin",
        "--trusted",
        "--json",
        "--intent-label",
        SUBMIT_INTENT_LABEL,
        "--intent-ttl",
        "900",
    ]
    result = subprocess.run(
        command,
        input=payload,
        text=True,
        capture_output=True,
        env=scheduler_common.scheduler_env(),
    )
    if result.returncode != 0:
        print((result.stdout or "") + (result.stderr or ""), file=sys.stderr)
        result.check_returncode()
    response = json.loads(result.stdout)
    print(json.dumps(response, indent=2))
    submitted = response.get("submitted", [])
    task_ids = [str(item.get("id", "")) for item in submitted]
    if (len(task_ids) != len(specs)
            or any(not task_id for task_id in task_ids)
            or len(set(task_ids)) != len(task_ids)):
        raise RuntimeError(
            "scheduler submission was not all-or-verifiably-accounted: "
            f"requested={len(specs)}, returned={task_ids}")
    return task_ids


def _dispatch(task_ids):
    if not task_ids:
        return
    command = [
        sys.executable,
        str(scheduler_common.SCHEDULER),
        "dispatch",
        "--bulk-window",
        "--intent-label",
        "bapr-polarity-policy-compression-v2-dispatch",
        "--intent-ttl",
        "900",
    ]
    for task_id in task_ids:
        command += ["--task-id", task_id]
    subprocess.run(
        command,
        check=True,
        env=scheduler_common.scheduler_env(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("all", "train", "audit", "analysis"),
        default="all")
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="high")
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

    known = scheduler_common.scheduler_tasks()
    specs = []
    for signature, spec, output in candidates(args.phase, args.priority):
        tasks = [
            task for task in known
            if str(task.get("signature") or "") == signature
        ]
        active = [
            task for task in tasks
            if str(task.get("status")) in ACTIVE_STATUSES
        ]
        if output.is_file():
            print(f"skip complete-output: {signature}")
        elif active:
            print("skip active: " + ",".join(
                str(task["id"]) for task in active))
        elif tasks and not args.retry_incomplete:
            print("skip terminal-incomplete: " + ",".join(
                str(task["id"]) for task in tasks))
        else:
            specs.append(spec)
    if args.dry_run:
        print(json.dumps(specs, indent=2, sort_keys=True))
        return
    if not specs:
        print("No closed-loop policy-compression tasks to submit")
        return
    task_ids = _submit_jsonl(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        train_ids = [
            task_id for task_id, spec in zip(task_ids, specs)
            if "/train/" in str(spec["signature"])
        ]
        _dispatch(train_ids)


if __name__ == "__main__":
    main()
