#!/usr/bin/env python3
"""Submit the frozen transient-fallback cross-student audit graph."""
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
    regime_polarity_policy_distillation_transient_fallback_cross_student_v1 as protocol,
)


SIGNATURE_PREFIX = "BAPR/regime-polarity-transient-fallback/cross-student-v1"
SUBMIT_INTENT_LABEL = "bapr-transient-fallback-cross-student-v1-submit"
CPU_NODES = [f"node00{index}" for index in range(1, 7)]
ACTIVE_STATUSES = scheduler_common.ACTIVE_STATUSES


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


def audit_signature(student_seed: int, event_seed: int) -> str:
    return (
        f"{SIGNATURE_PREFIX}/student-{protocol.require_student_seed(student_seed)}"
        f"/event-{protocol.require_event_seed(event_seed)}"
    )


def analysis_signature() -> str:
    return f"{SIGNATURE_PREFIX}/analysis"


def _common_spec(description: str, cmd: str, signature: str, output: Path,
                 priority: str, cpu: int, ram_mb: int) -> dict:
    return {
        "description": description,
        "cmd": cmd,
        "cwd": str(ROOT),
        "signature": signature,
        "project": "BAPR",
        "vram": 0,
        "ram_mb": ram_mb,
        "cpu": cpu,
        "priority": priority,
        "allowed_nodes": CPU_NODES,
        "result_dir": str(output),
        "local_result_dir": str(output),
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Frozen checkpoint-only policy evaluation; no optimizer update."),
    }


def _source_wait_files(student_seed: int) -> list[str]:
    return [
        str(protocol.development.ensemble.final.MODEL_MANIFEST),
        str(protocol.development.ensemble.final.MODEL_PATH),
        *(str(path) for path in protocol.development.source_required_paths(
            protocol.TEACHER_GROUP)),
        str(protocol.development.model_manifest(
            protocol.TEACHER_GROUP, student_seed)),
        str(protocol.development.model_path(
            protocol.TEACHER_GROUP, student_seed)),
        str(protocol.parent.selection_manifest()),
        str(protocol.parent.analysis_json()),
    ]


def _source_stage_inputs(student_seed: int) -> list[str]:
    return [
        str(protocol.development.ensemble.final.MODEL_MANIFEST.parent),
        *(str(path) for path in protocol.development.source_bundle_dirs(
            protocol.TEACHER_GROUP)),
        str(protocol.development.model_dir(
            protocol.TEACHER_GROUP, student_seed)),
        str(protocol.parent.SELECTION_ROOT),
        str(protocol.parent.ANALYSIS_ROOT),
    ]


def audit_spec(student_seed: int, event_seed: int, priority: str) -> dict:
    student_seed = protocol.require_student_seed(student_seed)
    event_seed = protocol.require_event_seed(event_seed)
    spec = _common_spec(
        f"Frozen fallback student {student_seed} event {event_seed}",
        _cpu_command(
            "jax_experiments.analysis."
            "run_regime_polarity_policy_distillation_transient_fallback_cross_student_audit_v1",
            ["--student-seed", str(student_seed),
             "--event-seed", str(event_seed), "--resume"],
        ),
        audit_signature(student_seed, event_seed),
        protocol.audit_dir(student_seed, event_seed),
        priority,
        cpu=32,
        ram_mb=8192,
    )
    spec["wait_for_files"] = _source_wait_files(student_seed)
    spec["stage_input_paths"] = _source_stage_inputs(student_seed)
    spec["stage_excludes"] = [
        "jax_experiments/results*/", "jax_experiments/eval_bundles*/", "paper/",
    ]
    return spec


def analysis_spec(priority: str) -> dict:
    spec = _common_spec(
        "Aggregate frozen transient-fallback cross-student audit",
        _cpu_command(
            "jax_experiments.analysis."
            "analyze_regime_polarity_policy_distillation_transient_fallback_cross_student_v1",
            [],
        ),
        analysis_signature(),
        protocol.ANALYSIS_ROOT,
        priority,
        cpu=2,
        ram_mb=2048,
    )
    spec["wait_for_files"] = [
        str(path) for path in protocol.all_audit_manifests()
    ]
    spec["stage_input_paths"] = [str(protocol.AUDIT_ROOT)]
    return spec


def candidates(priority: str):
    rows = [
        (
            audit_signature(student_seed, event_seed),
            audit_spec(student_seed, event_seed, priority),
            protocol.audit_manifest(student_seed, event_seed),
        )
        for student_seed in protocol.STUDENT_SEEDS
        for event_seed in protocol.EVENT_SEEDS
    ]
    rows.append((
        analysis_signature(), analysis_spec(priority), protocol.analysis_json()))
    return rows


def _submit_jsonl(specs):
    payload = "".join(json.dumps(spec) + "\n" for spec in specs)
    command = [
        sys.executable,
        str(scheduler_common.SCHEDULER),
        "submit-jsonl", "--stdin", "--trusted", "--json",
        "--intent-label", SUBMIT_INTENT_LABEL,
        "--intent-ttl", "900",
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
    task_ids = [
        str(item.get("id", "")) for item in response.get("submitted", [])
    ]
    if (
        len(task_ids) != len(specs)
        or any(not task_id for task_id in task_ids)
        or len(set(task_ids)) != len(task_ids)
    ):
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
        "dispatch", "--bulk-window",
        "--intent-label", "bapr-transient-fallback-cross-student-v1-dispatch",
        "--intent-ttl", "900",
    ]
    for task_id in task_ids:
        command += ["--task-id", task_id]
    subprocess.run(command, check=True, env=scheduler_common.scheduler_env())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="high")
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

    known = scheduler_common.scheduler_tasks()
    specs = []
    for signature, spec, output in candidates(args.priority):
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
        print("No cross-student fallback tasks to submit")
        return
    task_ids = _submit_jsonl(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        audit_ids = [
            task_id for task_id, spec in zip(task_ids, specs)
            if "/student-" in str(spec["signature"])
        ]
        _dispatch(audit_ids)


if __name__ == "__main__":
    main()
