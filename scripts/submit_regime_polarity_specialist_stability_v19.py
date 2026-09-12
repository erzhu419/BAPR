#!/usr/bin/env python3
"""Register and submit the v19 specialist-stability development DAG."""
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
    regime_polarity_specialist_stability_v19 as protocol,
)


SIGNATURE_PREFIX = "BAPR/regime-polarity/v19-specialist-stability"
SUBMIT_INTENT = "bapr-v19-specialist-stability-submit"
GPU_NODES = ["jtl110gpu", "jtl110gpu2", "jtl311linux", "node007"]
CPU_NODES = [f"node00{index}" for index in range(1, 8)]
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


def _gpu_command(module: str, values: list[str], total_units: int) -> str:
    return (
        f"SCHEDULEURM_ETA_TOTAL_UNITS={int(total_units)} "
        "JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.20 "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
        "OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 "
        "NUMEXPR_NUM_THREADS=2 JAX_NUM_THREADS=2 "
        "TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 "
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        f"{shlex.quote(str(scheduler_common.JAX_PYTHON))} -u -m "
        f"{module} {shlex.join(values)} && echo DONE"
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


def source_signature(seed: int) -> str:
    return (
        f"{SIGNATURE_PREFIX}/train/robust-source/seed-"
        f"{protocol.require_training_seed(seed)}"
    )


def source_spec(seed: int, priority: str) -> dict:
    seed = protocol.require_training_seed(seed)
    run_dir = protocol.source_run_dir(seed)
    bundle = protocol.source_bundle(seed)
    return {
        "description": f"V19 fresh robust source seed {seed}",
        "cmd": _gpu_command(
            "jax_experiments.analysis.run_regime_polarity_robust_source_v19",
            ["--seed", str(seed), "--resume"],
            protocol.SOURCE_NEXT_ITERATION,
        ),
        "cwd": str(ROOT),
        "signature": source_signature(seed),
        "project": "BAPR",
        "vram": 1200,
        "ram_mb": 3072,
        "cpu": 2,
        "priority": priority,
        "allowed_nodes": GPU_NODES,
        "ckpt_dir": str(run_dir / "checkpoints"),
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


def specialist_signature(variant: str, seed: int, mode: int) -> str:
    return (
        f"{SIGNATURE_PREFIX}/train/{protocol.require_variant(variant)}/seed-"
        f"{protocol.require_training_seed(seed)}/mode-"
        f"{protocol.require_mode(mode)}"
    )


def specialist_spec(
    variant: str, seed: int, mode: int, priority: str,
) -> dict:
    variant = protocol.require_variant(variant)
    seed = protocol.require_training_seed(seed)
    mode = protocol.require_mode(mode)
    run_dir = protocol.run_dir(variant, seed, mode)
    bundle = protocol.bundle_dir(variant, seed, mode)
    return {
        "description": (
            f"V19 {variant} specialist seed {seed} mode {mode}"
        ),
        "cmd": _gpu_command(
            "jax_experiments.analysis."
            "run_regime_polarity_specialist_stability_v19",
            [
                "--variant", variant,
                "--seed", str(seed),
                "--mode", str(mode),
                "--resume",
            ],
            protocol.FINAL_NEXT_ITERATION,
        ),
        "cwd": str(ROOT),
        "signature": specialist_signature(variant, seed, mode),
        "project": "BAPR",
        "vram": 1200,
        "ram_mb": 3072,
        "cpu": 2,
        "priority": priority,
        "allowed_nodes": GPU_NODES,
        "ckpt_dir": str(run_dir / "checkpoints"),
        "ckpt_glob": "train_state.pkl",
        "resume_flag": "",
        "resume_managed_by_cmd": True,
        "result_dir": str(bundle),
        "local_result_dir": str(bundle),
        "wait_for_files": [
            str(protocol.REGISTRATION_PATH),
            *registration_data_files(),
            *(str(path) for path in protocol.source_required_paths(seed)),
        ],
        "stage_input_paths": [
            *registration_data_input_dirs(),
            str(protocol.source_bundle(seed)),
        ],
        "stage_excludes": ["paper/"],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
    }


def audit_signature(variant: str, seed: int) -> str:
    return (
        f"{SIGNATURE_PREFIX}/audit/{protocol.require_variant(variant)}/seed-"
        f"{protocol.require_training_seed(seed)}"
    )


def audit_spec(variant: str, seed: int, priority: str) -> dict:
    variant = protocol.require_variant(variant)
    seed = protocol.require_training_seed(seed)
    return {
        "description": f"V19 {variant} audit seed {seed}",
        "cmd": _cpu_command(
            "jax_experiments.analysis."
            "run_regime_polarity_specialist_stability_audit_v19",
            ["--variant", variant, "--seed", str(seed), "--resume"],
            threads=16,
        ),
        "cwd": str(ROOT),
        "signature": audit_signature(variant, seed),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 8192,
        "cpu": 16,
        "priority": priority,
        "allowed_nodes": CPU_NODES,
        "result_dir": str(protocol.audit_dir(variant, seed)),
        "local_result_dir": str(protocol.audit_dir(variant, seed)),
        "wait_for_files": [
            str(protocol.REGISTRATION_PATH),
            *registration_data_files(),
            *(str(path) for path in protocol.source_required_paths(seed)),
            *(
                str(path)
                for mode in protocol.MODES
                for path in protocol.bundle_required_paths(
                    variant, seed, mode)
            ),
        ],
        "stage_input_paths": [
            *registration_data_input_dirs(),
            str(protocol.source_bundle(seed)),
            *(
                str(protocol.bundle_dir(variant, seed, mode))
                for mode in protocol.MODES
            ),
        ],
        "stage_excludes": ["paper/"],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Frozen robust and specialist policy evaluation; no updates."
        ),
    }


def analysis_signature() -> str:
    return f"{SIGNATURE_PREFIX}/analysis"


def analysis_spec(priority: str) -> dict:
    return {
        "description": "Aggregate v19 specialist stability screen",
        "cmd": _cpu_command(
            "jax_experiments.analysis."
            "analyze_regime_polarity_specialist_stability_v19",
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
                str(protocol.audit_manifest(variant, seed))
                for variant in protocol.VARIANTS
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
        rows.append((
            source_signature(seed),
            source_spec(seed, priority),
            protocol.source_manifest(seed),
        ))
        for variant in protocol.VARIANTS:
            for mode in protocol.MODES:
                rows.append((
                    specialist_signature(variant, seed, mode),
                    specialist_spec(variant, seed, mode, priority),
                    protocol.bundle_manifest(variant, seed, mode),
                ))
            rows.append((
                audit_signature(variant, seed),
                audit_spec(variant, seed, priority),
                protocol.audit_manifest(variant, seed),
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
        "bapr-v19-specialist-stability-dispatch",
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
        return
    if not specs:
        print("No v19 specialist stability tasks to submit")
        return
    task_ids = _submit(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        _dispatch(task_ids)


if __name__ == "__main__":
    main()
