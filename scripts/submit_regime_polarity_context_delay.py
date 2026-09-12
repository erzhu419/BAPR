#!/usr/bin/env python3
"""Submit checkpoint-only polarity context-delay audits."""
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

if Path(sys.executable).resolve() != scheduler_common.JAX_PYTHON.resolve():
    os.execv(
        str(scheduler_common.JAX_PYTHON),
        [str(scheduler_common.JAX_PYTHON), str(Path(__file__).resolve()),
         *sys.argv[1:]],
    )

from jax_experiments.analysis import regime_polarity_context_delay as protocol
from jax_experiments.analysis import regime_polarity_headroom as base


SIGNATURE_PREFIX = "BAPR/regime-polarity-context-delay/v1"
CPU_NODES = [f"node00{index}" for index in range(1, 7)]
ACTIVE_STATUSES = scheduler_common.ACTIVE_STATUSES


def _cpu_command(module: str, values: list[str]) -> str:
    return (
        "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' "
        "XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1' "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 JAX_NUM_THREADS=1 "
        "JAX_CPU_ENABLE_ASYNC_DISPATCH=false "
        "TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 "
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        f"{shlex.quote(str(scheduler_common.JAX_PYTHON))} -u -m "
        f"{module} {shlex.join(values)} && echo DONE"
    )


def audit_signature(env: str, seed: int, event_seed: int) -> str:
    return (
        f"{SIGNATURE_PREFIX}/audit/{base.env_slug(protocol.require_env(env))}/"
        f"seed-{base.require_training_seed(seed)}/"
        f"event-{base.require_event_seed(event_seed)}")


def audit_spec(
        env: str, seed: int, event_seed: int,
        priority: str,
) -> dict[str, object]:
    output = protocol.audit_dir(env, seed, event_seed)
    bundle_paths = [
        base.bundle_dir(env, role, seed) for role in base.ROLES
    ]
    return {
        "description": (
            f"Polarity context-delay {base.env_slug(env)} seed {seed} "
            f"event {event_seed}"),
        "cmd": _cpu_command(
            "jax_experiments.analysis."
            "run_regime_polarity_context_delay_audit",
            ["--env", env, "--seed", str(seed),
             "--event-seed", str(event_seed), "--resume"]),
        "cwd": str(ROOT),
        "signature": audit_signature(env, seed, event_seed),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 12288,
        "cpu": 32,
        "priority": priority,
        "allowed_nodes": CPU_NODES,
        "result_dir": str(output),
        "local_result_dir": str(output),
        "wait_for_files": [
            str(path)
            for role in base.ROLES
            for path in base.bundle_required_paths(env, role, seed)
        ],
        "stage_input_paths": [str(path) for path in bundle_paths],
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Checkpoint-only deterministic policy evaluation; no updates."),
    }


def analysis_signature() -> str:
    return f"{SIGNATURE_PREFIX}/analysis"


def analysis_spec(priority: str) -> dict[str, object]:
    return {
        "description": "Aggregate polarity context-delay diagnosis",
        "cmd": _cpu_command(
            "jax_experiments.analysis."
            "analyze_regime_polarity_context_delay",
            []),
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
        "wait_for_files": [
            str(path) for path in protocol.all_audit_manifests()
        ],
        "stage_input_paths": [str(protocol.AUDIT_ROOT)],
        "stage_excludes": [
            "jax_experiments/results_regime_polarity_headroom_v1/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": "CSV/JSON aggregation only.",
    }


def candidates(phase: str, priority: str):
    rows = []
    if phase in ("all", "audit"):
        rows.extend([
            (
                audit_signature(env, seed, event_seed),
                audit_spec(env, seed, event_seed, priority),
                protocol.audit_manifest(env, seed, event_seed),
            )
            for env in protocol.ENVS
            for seed in base.TRAINING_SEEDS
            for event_seed in base.AUDIT_EVENT_SEEDS
        ])
    if phase in ("all", "analysis"):
        rows.append((
            analysis_signature(), analysis_spec(priority),
            protocol.analysis_json()))
    return rows


def _submit_jsonl(specs: list[dict[str, object]]) -> list[str]:
    payload = "".join(json.dumps(spec) + "\n" for spec in specs)
    command = [
        sys.executable, str(scheduler_common.SCHEDULER), "submit-jsonl",
        "--stdin", "--trusted", "--json",
        "--intent-label", "bapr-polarity-context-delay-v1-submit",
        "--intent-ttl", "900",
    ]
    result = subprocess.run(
        command, input=payload, text=True, capture_output=True,
        env=scheduler_common.scheduler_env())
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


def _dispatch(task_ids: list[str]) -> None:
    if not task_ids:
        return
    command = [
        sys.executable, str(scheduler_common.SCHEDULER), "dispatch",
        "--bulk-window",
        "--intent-label", "bapr-polarity-context-delay-v1-dispatch",
        "--intent-ttl", "900",
    ]
    for task_id in task_ids:
        command += ["--task-id", task_id]
    subprocess.run(
        command, check=True, env=scheduler_common.scheduler_env())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("all", "audit", "analysis"), default="all")
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
        print("No polarity context-delay tasks to submit")
        return
    task_ids = _submit_jsonl(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        audit_ids = [
            task_id for task_id, spec in zip(task_ids, specs)
            if spec["signature"] != analysis_signature()
        ]
        _dispatch(audit_ids)


if __name__ == "__main__":
    main()
