#!/usr/bin/env python3
"""Register and batch-submit the 36-task v18 final comparison DAG."""
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
    regime_polarity_v5_final_comparison_v18 as protocol,
)


SIGNATURE_PREFIX = "BAPR/regime-polarity/v18-v5-final-comparison"
SUBMIT_INTENT = "bapr-v18-v5-final-comparison-submit"
GPU_NODES = ["local", "jtl110gpu", "jtl110gpu2", "jtl311linux", "node007"]
CPU_NODES = [f"node00{index}" for index in range(1, 7)]
ACTIVE_STATUSES = scheduler_common.ACTIVE_STATUSES


def registration_data_files() -> list[str]:
    return [
        str(path)
        for path in protocol.registration_source_paths()
        if path.suffix != ".py"
    ]


def registration_data_input_dirs() -> list[str]:
    return list(dict.fromkeys(
        [str(protocol.REGISTRATION_ROOT)]
        + [str(Path(path).parent) for path in registration_data_files()]
    ))


def _gpu_command(kind: str, seed: int, slot: int | None) -> str:
    values = ["--kind", kind, "--seed", str(seed)]
    if slot is not None:
        values += ["--slot", str(slot)]
    values.append("--resume")
    return (
        f"SCHEDULEURM_ETA_TOTAL_UNITS={protocol.MAX_ITERS} "
        "JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
        "OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 "
        "NUMEXPR_NUM_THREADS=2 JAX_NUM_THREADS=2 "
        "TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 "
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        f"{shlex.quote(str(scheduler_common.JAX_PYTHON))} -u -m "
        "jax_experiments.analysis."
        "run_regime_polarity_v5_final_baseline_v18 "
        f"{shlex.join(values)} && echo DONE"
    )


def _cpu_command(module: str, values: list[str], threads: int) -> str:
    return (
        "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' TMPDIR=/tmp "
        "XLA_FLAGS='--xla_cpu_multi_thread_eigen=false "
        f"intra_op_parallelism_threads={threads}' "
        f"OMP_NUM_THREADS={threads} OPENBLAS_NUM_THREADS={threads} "
        f"MKL_NUM_THREADS={threads} NUMEXPR_NUM_THREADS={threads} "
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        f"{shlex.quote(str(scheduler_common.JAX_PYTHON))} -u -m "
        f"{module} {shlex.join(values)} && echo DONE"
    )


def producer_signature(kind: str, seed: int, slot: int | None = None) -> str:
    suffix = (
        f"/slot-{protocol.require_replica_slot(slot)}"
        if kind == "sac_replica" and slot is not None else "")
    return (
        f"{SIGNATURE_PREFIX}/train/{kind}/seed-"
        f"{protocol.require_training_seed(seed)}{suffix}"
    )


def producer_spec(
    kind: str, seed: int, slot: int | None, priority: str,
) -> dict:
    seed = protocol.require_training_seed(seed)
    if kind == "sac_replica":
        if slot is None:
            raise ValueError("SAC producer needs a slot")
        slot = protocol.require_replica_slot(slot)
        run = protocol.sac_run_dir(seed, slot)
        bundle = protocol.sac_bundle_dir(seed, slot)
        vram, ram_mb = 1200, 3072
        description = f"V18 SAC5 replica seed {seed} slot {slot}"
    else:
        kind = protocol.require_method(kind)
        if slot is not None:
            raise ValueError("non-SAC producer cannot have a slot")
        run = protocol.baseline_run_dir(kind, seed)
        bundle = protocol.baseline_bundle_dir(kind, seed)
        vram = 2600 if kind == "escp_recurrent" else 2300
        ram_mb = 8192 if kind == "escp_recurrent" else 10240
        description = f"V18 {kind} final baseline seed {seed}"
    return {
        "description": description,
        "cmd": _gpu_command(kind, seed, slot),
        "cwd": str(ROOT),
        "signature": producer_signature(kind, seed, slot),
        "project": "BAPR",
        "vram": vram,
        "ram_mb": ram_mb,
        "cpu": 2,
        "priority": priority,
        "allowed_nodes": GPU_NODES,
        "ckpt_dir": str(run / "checkpoints"),
        "ckpt_glob": "train_state.pkl",
        "resume_flag": "",
        "resume_managed_by_cmd": True,
        "result_dir": str(bundle),
        "local_result_dir": str(bundle),
        "wait_for_files": [
            str(protocol.REGISTRATION_PATH),
            *registration_data_files(),
        ],
        "stage_input_paths": registration_data_input_dirs(),
        "stage_excludes": ["paper/"],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
    }


def audit_signature(seed: int) -> str:
    return (
        f"{SIGNATURE_PREFIX}/audit/seed-"
        f"{protocol.require_training_seed(seed)}"
    )


def _frozen_required_paths(seed: int) -> list[str]:
    paths = [str(path) for path in protocol.frozen.source_required_paths(seed)]
    for mode in protocol.MODES:
        paths.extend(str(path) for path in protocol.frozen.bundle_required_paths(
            "actor_only", seed, mode))
    return paths


def _new_required_paths(seed: int) -> list[str]:
    paths = []
    for method in protocol.TRAINED_METHODS:
        paths.extend(str(path) for path in protocol.bundle_required_paths(
            method, seed))
    for slot in protocol.SAC_REPLICA_SLOTS:
        paths.extend(str(path) for path in protocol.bundle_required_paths(
            "sac_replica", seed, slot))
    return paths


def audit_spec(seed: int, priority: str) -> dict:
    seed = protocol.require_training_seed(seed)
    return {
        "description": f"V18 final equal-budget audit seed {seed}",
        "cmd": _cpu_command(
            "jax_experiments.analysis."
            "run_regime_polarity_v5_final_comparison_audit_v18",
            ["--seed", str(seed), "--resume"],
            threads=16,
        ),
        "cwd": str(ROOT),
        "signature": audit_signature(seed),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 16384,
        "cpu": 16,
        "priority": priority,
        "allowed_nodes": CPU_NODES,
        "result_dir": str(protocol.audit_dir(seed)),
        "local_result_dir": str(protocol.audit_dir(seed)),
        "wait_for_files": [
            str(protocol.REGISTRATION_PATH),
            *registration_data_files(),
            *_frozen_required_paths(seed),
            *_new_required_paths(seed),
            str(protocol.v5_model.MODEL_MANIFEST),
            str(protocol.v5_model.MODEL_PATH),
        ],
        "stage_input_paths": [
            *registration_data_input_dirs(),
            str(protocol.frozen.source_bundle(seed)),
            *(
                str(protocol.frozen.bundle_dir("actor_only", seed, mode))
                for mode in protocol.MODES
            ),
            *(
                str(protocol.baseline_bundle_dir(method, seed))
                for method in protocol.TRAINED_METHODS
            ),
            *(
                str(protocol.sac_bundle_dir(seed, slot))
                for slot in protocol.SAC_REPLICA_SLOTS
            ),
            str(protocol.v5_model.MODEL_ROOT),
        ],
        "stage_excludes": ["paper/"],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Frozen deterministic policy evaluation; no parameter updates."
        ),
    }


def analysis_signature() -> str:
    return f"{SIGNATURE_PREFIX}/analysis"


def analysis_spec(priority: str) -> dict:
    return {
        "description": "Aggregate v18 final equal-policy-budget comparison",
        "cmd": _cpu_command(
            "jax_experiments.analysis."
            "analyze_regime_polarity_v5_final_comparison_v18",
            [],
            threads=2,
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
        "wait_for_files": [
            str(protocol.REGISTRATION_PATH),
            *registration_data_files(),
            *(
                str(protocol.audit_manifest(seed))
                for seed in protocol.TRAINING_SEEDS
            ),
        ],
        "stage_input_paths": [
            str(protocol.AUDIT_ROOT),
            *registration_data_input_dirs(),
        ],
        "stage_excludes": ["paper/"],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": "Immutable JSON aggregation only.",
    }


def candidates(priority: str):
    rows = []
    for seed in protocol.TRAINING_SEEDS:
        for slot in protocol.SAC_REPLICA_SLOTS:
            rows.append((
                producer_signature("sac_replica", seed, slot),
                producer_spec("sac_replica", seed, slot, priority),
                protocol.bundle_manifest("sac_replica", seed, slot),
            ))
        for method in protocol.TRAINED_METHODS:
            rows.append((
                producer_signature(method, seed),
                producer_spec(method, seed, None, priority),
                protocol.bundle_manifest(method, seed),
            ))
        rows.append((
            audit_signature(seed),
            audit_spec(seed, priority),
            protocol.audit_manifest(seed),
        ))
    rows.append((
        analysis_signature(),
        analysis_spec(priority),
        protocol.analysis_json(),
    ))
    return rows


def _submit(specs: list[dict]) -> list[str]:
    payload = "".join(json.dumps(spec) + "\n" for spec in specs)
    command = [
        sys.executable,
        str(scheduler_common.SCHEDULER),
        "submit-jsonl",
        "--stdin",
        "--trusted",
        "--json",
        "--intent-label",
        SUBMIT_INTENT,
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
    task_ids = [
        str(row.get("id") or "") for row in response.get("submitted", [])
    ]
    if (
        len(task_ids) != len(specs)
        or any(not task_id for task_id in task_ids)
        or len(set(task_ids)) != len(task_ids)
    ):
        raise RuntimeError(
            f"scheduler batch incomplete: requested={len(specs)} ids={task_ids}")
    print(json.dumps(response, indent=2))
    return task_ids


def _dispatch(task_ids: list[str]) -> None:
    if not task_ids:
        return
    command = [
        sys.executable,
        str(scheduler_common.SCHEDULER),
        "dispatch",
        "--bulk-window",
        "--intent-label",
        "bapr-v18-v5-final-comparison-dispatch",
        "--intent-ttl",
        "900",
    ]
    for task_id in task_ids:
        command += ["--task-id", task_id]
    subprocess.run(command, check=True, env=scheduler_common.scheduler_env())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="high")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

    protocol.create_registration()
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
        else:
            specs.append(spec)
    if args.dry_run:
        print(json.dumps(specs, indent=2, sort_keys=True))
        print(f"V18 dry-run task count: {len(specs)}", flush=True)
        return
    if not specs:
        print("No v18 final comparison tasks to submit")
        return
    task_ids = _submit(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        _dispatch(task_ids)


if __name__ == "__main__":
    main()
