#!/usr/bin/env python3
"""Submit specialist-trajectory system ID v5 and frozen-router audits."""
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

if (
    __name__ == "__main__"
    and Path(sys.executable).resolve() != scheduler_common.JAX_PYTHON.resolve()
):
    os.execv(
        str(scheduler_common.JAX_PYTHON),
        [str(scheduler_common.JAX_PYTHON), str(Path(__file__).resolve()),
         *sys.argv[1:]],
    )

from jax_experiments.analysis import (
    regime_polarity_expected_action_system_id as parent_estimator,
)
from jax_experiments.analysis import (
    regime_polarity_source_headroom_v1 as source,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_system_id_v5 as protocol,
)


SIGNATURE_PREFIX = "BAPR/regime-polarity-specialist-expected-action/v5"
GPU_NODES = [
    "local", "jtl110gpu", "jtl110gpu2", "jtl311linux", "node007"]
CPU_NODES = [f"node00{index}" for index in range(1, 7)]
ACTIVE_STATUSES = scheduler_common.ACTIVE_STATUSES


def _gpu_command(module: str, values: list[str]) -> str:
    return (
        f"SCHEDULEURM_ETA_TOTAL_UNITS={protocol.TRAIN_UPDATES} "
        "JAX_PLATFORMS=cuda "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
        "OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 "
        "NUMEXPR_NUM_THREADS=2 JAX_NUM_THREADS=2 "
        "TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 "
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        f"{shlex.quote(str(scheduler_common.JAX_PYTHON))} -u -m "
        f"{module} {shlex.join(values)} && echo DONE"
    )


def _cpu_command(module: str, values: list[str], threads: int = 4) -> str:
    return (
        "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' "
        "XLA_FLAGS='--xla_cpu_multi_thread_eigen=false "
        f"intra_op_parallelism_threads={threads}' "
        f"OMP_NUM_THREADS={threads} OPENBLAS_NUM_THREADS={threads} "
        f"MKL_NUM_THREADS={threads} NUMEXPR_NUM_THREADS={threads} "
        f"JAX_NUM_THREADS={threads} JAX_CPU_ENABLE_ASYNC_DISPATCH=false "
        f"TF_NUM_INTRAOP_THREADS={threads} TF_NUM_INTEROP_THREADS=2 "
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        f"{shlex.quote(str(scheduler_common.JAX_PYTHON))} -u -m "
        f"{module} {shlex.join(values)} && echo DONE"
    )


def training_signature() -> str:
    return f"{SIGNATURE_PREFIX}/train"


def smoke_signature() -> str:
    return f"{SIGNATURE_PREFIX}/smoke"


def _source_bundle_dirs():
    return [
        source.bundle_dir(role, seed)
        for seed in (
            *protocol.TRAIN_SOURCE_SEEDS,
            *protocol.VALIDATION_SOURCE_SEEDS,
        )
        for role in protocol.STATIONARY_ARMS
    ]


def training_spec(priority: str) -> dict:
    bundles = _source_bundle_dirs()
    return {
        "description": "Fit specialist-trajectory expected-action system ID v5",
        "cmd": _gpu_command(
            "jax_experiments.analysis."
            "train_regime_polarity_specialist_expected_action_v5",
            ["--resume"],
        ),
        "cwd": str(ROOT),
        "signature": training_signature(),
        "project": "BAPR",
        "vram_resource_family": (
            "BAPR/regime-polarity/expected-action-system-id/gpu-runtime"),
        "vram": 3200,
        "allow_gpu_over_one_third": True,
        "ram_mb": 12288,
        "cpu": 2,
        "priority": priority,
        "allowed_nodes": GPU_NODES,
        "ckpt_dir": str(protocol.CHECKPOINT_ROOT),
        "ckpt_glob": "train_state.pkl",
        "resume_flag": "",
        "resume_managed_by_cmd": True,
        "result_dir": str(protocol.MODEL_ROOT),
        "local_result_dir": str(protocol.MODEL_ROOT),
        "wait_for_files": [
            str(parent_estimator.MODEL_MANIFEST),
            str(parent_estimator.MODEL_PATH),
            *(
                str(path)
                for seed in (
                    *protocol.TRAIN_SOURCE_SEEDS,
                    *protocol.VALIDATION_SOURCE_SEEDS,
                )
                for role in protocol.STATIONARY_ARMS
                for path in source.bundle_required_paths(role, seed)
            ),
        ],
        "stage_input_paths": [
            str(parent_estimator.MODEL_ROOT),
            *(str(path) for path in bundles),
        ],
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
    }


def smoke_spec(priority: str) -> dict:
    spec = training_spec(priority)
    bundles = [
        source.bundle_dir(role, protocol.TRAIN_SOURCE_SEEDS[0])
        for role in protocol.STATIONARY_ARMS
    ]
    spec.update({
        "description": "Smoke specialist-trajectory expected-action v5",
        "cmd": _gpu_command(
            "jax_experiments.analysis."
            "train_regime_polarity_specialist_expected_action_v5",
            ["--smoke"],
        ),
        "signature": smoke_signature(),
        "wait_for_files": [
            str(parent_estimator.MODEL_MANIFEST),
            str(parent_estimator.MODEL_PATH),
            *(
                str(path)
                for role in protocol.STATIONARY_ARMS
                for path in source.bundle_required_paths(
                    role, protocol.TRAIN_SOURCE_SEEDS[0])
            ),
        ],
        "stage_input_paths": [
            str(parent_estimator.MODEL_ROOT),
            *(str(path) for path in bundles),
        ],
    })
    for key in (
        "ckpt_dir",
        "ckpt_glob",
        "resume_flag",
        "resume_managed_by_cmd",
        "result_dir",
        "local_result_dir",
    ):
        spec.pop(key, None)
    return spec


def audit_signature(seed: int) -> str:
    return f"{SIGNATURE_PREFIX}/audit/seed-{protocol.require_source_seed(seed)}"


def audit_spec(seed: int, priority: str) -> dict:
    seed = protocol.require_source_seed(seed)
    bundles = [source.bundle_dir(role, seed) for role in source.ROLES]
    return {
        "description": f"Audit specialist estimator confirm3 seed {seed}",
        "cmd": _cpu_command(
            "jax_experiments.analysis."
            "run_regime_polarity_specialist_expected_action_audit_v5",
            ["--seed", str(seed), "--resume"],
        ),
        "cwd": str(ROOT),
        "signature": audit_signature(seed),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 8192,
        "cpu": 8,
        "priority": priority,
        "allowed_nodes": CPU_NODES,
        "result_dir": str(protocol.audit_dir(seed)),
        "local_result_dir": str(protocol.audit_dir(seed)),
        "wait_for_files": [
            str(protocol.MODEL_MANIFEST),
            str(protocol.MODEL_PATH),
            *(
                str(path)
                for role in source.ROLES
                for path in source.bundle_required_paths(role, seed)
            ),
        ],
        "stage_input_paths": [
            str(protocol.MODEL_ROOT),
            *(str(path) for path in bundles),
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
            "Frozen-controller and frozen-estimator audit; no updates."
        ),
    }


def analysis_signature() -> str:
    return f"{SIGNATURE_PREFIX}/analysis"


def analysis_spec(priority: str) -> dict:
    return {
        "description": "Aggregate specialist expected-action v5",
        "cmd": _cpu_command(
            "jax_experiments.analysis."
            "analyze_regime_polarity_specialist_expected_action_v5",
            [], threads=2),
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
            str(protocol.audit_manifest(seed))
            for seed in protocol.AUDIT_SOURCE_SEEDS
        ],
        "stage_input_paths": [str(protocol.AUDIT_ROOT)],
        "stage_excludes": ["paper/"],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": "JSON aggregation only.",
    }


def _submit(specs: list[dict]) -> list[str]:
    payload = "".join(json.dumps(spec) + "\n" for spec in specs)
    command = [
        sys.executable, str(scheduler_common.SCHEDULER),
        "submit-jsonl", "--stdin", "--trusted", "--json",
        "--intent-label", "bapr-specialist-expected-action-v5-submit",
        "--intent-ttl", "900",
    ]
    result = subprocess.run(
        command, input=payload, text=True, capture_output=True,
        env=scheduler_common.scheduler_env())
    if result.returncode != 0:
        print((result.stdout or "") + (result.stderr or ""), file=sys.stderr)
        result.check_returncode()
    response = json.loads(result.stdout)
    task_ids = [str(row.get("id") or "")
                for row in response.get("submitted", [])]
    if len(task_ids) != len(specs) or any(not task_id for task_id in task_ids):
        raise RuntimeError(
            f"scheduler batch incomplete: requested={len(specs)} ids={task_ids}")
    print(json.dumps(response, indent=2))
    return task_ids


def _dispatch(task_ids: list[str]) -> None:
    command = [
        sys.executable, str(scheduler_common.SCHEDULER), "dispatch",
        "--bulk-window", "--intent-label",
        "bapr-specialist-expected-action-v5-dispatch",
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
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    known = scheduler_common.scheduler_tasks()
    if args.smoke_only:
        candidates = [(
            smoke_signature(),
            smoke_spec(args.priority),
            ROOT / ".scheduler-markers/specialist-expected-action-v5-smoke",
        )]
    else:
        candidates = [(
            training_signature(), training_spec(args.priority),
            protocol.MODEL_MANIFEST,
        )]
        candidates.extend([
            (audit_signature(seed), audit_spec(seed, args.priority),
             protocol.audit_manifest(seed))
            for seed in protocol.AUDIT_SOURCE_SEEDS
        ])
        candidates.append((
            analysis_signature(), analysis_spec(args.priority),
            protocol.analysis_json()))
    specs = []
    for signature, spec, output in candidates:
        tasks = [task for task in known
                 if str(task.get("signature") or "") == signature]
        active = [task for task in tasks
                  if str(task.get("status")) in ACTIVE_STATUSES]
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
        print("No specialist expected-action tasks to submit")
        return
    task_ids = _submit(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        _dispatch(task_ids)


if __name__ == "__main__":
    main()
