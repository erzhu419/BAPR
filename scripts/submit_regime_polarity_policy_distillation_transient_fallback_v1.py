#!/usr/bin/env python3
"""Submit the two-stage frozen transient-fallback evaluation graph."""
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
    regime_polarity_policy_distillation_transient_fallback_v1 as protocol,
)


SIGNATURE_PREFIX = "BAPR/regime-polarity-transient-fallback/v1"
SUBMIT_INTENT_LABEL = "bapr-polarity-transient-fallback-v1-submit"
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


def screen_signature(seed: int) -> str:
    return f"{SIGNATURE_PREFIX}/screen/event-{protocol.require_screen_event_seed(seed)}"


def selection_signature() -> str:
    return f"{SIGNATURE_PREFIX}/selection"


def audit_signature(seed: int) -> str:
    return f"{SIGNATURE_PREFIX}/audit/event-{protocol.require_audit_event_seed(seed)}"


def analysis_signature() -> str:
    return f"{SIGNATURE_PREFIX}/analysis"


def _source_wait_files() -> list[str]:
    return [
        str(protocol.frozen.development.ensemble.final.MODEL_MANIFEST),
        str(protocol.frozen.development.ensemble.final.MODEL_PATH),
        *(str(path) for path in protocol.source_required_paths(
            protocol.TEACHER_GROUP)),
        str(protocol.model_manifest(
            protocol.TEACHER_GROUP, protocol.STUDENT_SEED)),
        str(protocol.model_path(
            protocol.TEACHER_GROUP, protocol.STUDENT_SEED)),
    ]


def _source_stage_inputs() -> list[str]:
    return [
        str(protocol.frozen.development.ensemble.final.MODEL_MANIFEST.parent),
        *(str(path) for path in protocol.source_bundle_dirs(
            protocol.TEACHER_GROUP)),
        str(protocol.model_dir(
            protocol.TEACHER_GROUP, protocol.STUDENT_SEED)),
    ]


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


def screen_spec(seed: int, priority: str) -> dict:
    seed = protocol.require_screen_event_seed(seed)
    spec = _common_spec(
        f"Transient fallback development screen event {seed}",
        _cpu_command(
            "jax_experiments.analysis."
            "run_regime_polarity_policy_distillation_transient_fallback_screen_v1",
            ["--event-seed", str(seed), "--resume"],
        ),
        screen_signature(seed),
        protocol.screen_dir(seed),
        priority,
        cpu=32,
        ram_mb=8192,
    )
    spec["wait_for_files"] = _source_wait_files()
    spec["stage_input_paths"] = _source_stage_inputs()
    spec["stage_excludes"] = [
        "jax_experiments/results*/", "jax_experiments/eval_bundles*/", "paper/",
    ]
    return spec


def selection_spec(priority: str) -> dict:
    spec = _common_spec(
        "Select causal transient fallback on development split",
        _cpu_command(
            "jax_experiments.analysis."
            "select_regime_polarity_policy_distillation_transient_fallback_v1",
            [],
        ),
        selection_signature(),
        protocol.SELECTION_ROOT,
        priority,
        cpu=2,
        ram_mb=2048,
    )
    spec["wait_for_files"] = [
        str(protocol.screen_manifest(seed))
        for seed in protocol.SCREEN_EVENT_SEEDS
    ]
    spec["stage_input_paths"] = [str(protocol.SCREEN_ROOT)]
    return spec


def audit_spec(seed: int, priority: str) -> dict:
    seed = protocol.require_audit_event_seed(seed)
    spec = _common_spec(
        f"Independent transient fallback audit event {seed}",
        _cpu_command(
            "jax_experiments.analysis."
            "run_regime_polarity_policy_distillation_transient_fallback_audit_v1",
            ["--event-seed", str(seed), "--resume"],
        ),
        audit_signature(seed),
        protocol.audit_dir(seed),
        priority,
        cpu=32,
        ram_mb=8192,
    )
    spec["wait_for_files"] = [
        *_source_wait_files(),
        str(protocol.selection_manifest()),
        str(protocol.selection_json()),
    ]
    spec["stage_input_paths"] = [
        *_source_stage_inputs(), str(protocol.SELECTION_ROOT),
    ]
    spec["stage_excludes"] = [
        "jax_experiments/results*/", "jax_experiments/eval_bundles*/", "paper/",
    ]
    return spec


def analysis_spec(priority: str) -> dict:
    spec = _common_spec(
        "Aggregate independent transient fallback audit",
        _cpu_command(
            "jax_experiments.analysis."
            "analyze_regime_polarity_policy_distillation_transient_fallback_v1",
            [],
        ),
        analysis_signature(),
        protocol.ANALYSIS_ROOT,
        priority,
        cpu=2,
        ram_mb=2048,
    )
    spec["wait_for_files"] = [
        str(protocol.audit_manifest(seed))
        for seed in protocol.AUDIT_EVENT_SEEDS
    ]
    spec["stage_input_paths"] = [
        str(protocol.AUDIT_ROOT), str(protocol.SELECTION_ROOT),
    ]
    return spec


def candidates(priority: str):
    rows = [
        (screen_signature(seed), screen_spec(seed, priority),
         protocol.screen_manifest(seed))
        for seed in protocol.SCREEN_EVENT_SEEDS
    ]
    rows.append((selection_signature(), selection_spec(priority),
                 protocol.selection_manifest()))
    rows.extend([
        (audit_signature(seed), audit_spec(seed, priority),
         protocol.audit_manifest(seed))
        for seed in protocol.AUDIT_EVENT_SEEDS
    ])
    rows.append((analysis_signature(), analysis_spec(priority),
                 protocol.analysis_json()))
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
        "dispatch", "--bulk-window",
        "--intent-label", "bapr-polarity-transient-fallback-v1-dispatch",
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
        print("No transient-fallback tasks to submit")
        return
    task_ids = _submit_jsonl(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        screen_ids = [
            task_id for task_id, spec in zip(task_ids, specs)
            if "/screen/" in str(spec["signature"])
        ]
        _dispatch(screen_ids)


if __name__ == "__main__":
    main()
