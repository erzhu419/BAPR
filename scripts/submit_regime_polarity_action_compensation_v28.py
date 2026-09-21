#!/usr/bin/env python3
"""Register and submit the no-training V28 compensation audit DAG."""
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
    regime_polarity_action_compensation_v28 as protocol,
)


SIGNATURE_PREFIX = "BAPR/regime-polarity/v28-action-compensation"
SUBMIT_INTENT = "bapr-v28-action-compensation-submit"
CPU_NODES = [f"node00{index}" for index in range(1, 8)]
ACTIVE_STATUSES = scheduler_common.ACTIVE_STATUSES


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


def registration_data_files() -> list[str]:
    return [
        str(path) for path in protocol.registration_source_paths()
        if path.suffix != ".py"
    ]


def registration_data_input_dirs() -> list[str]:
    return list(dict.fromkeys(
        [str(protocol.REGISTRATION_ROOT)]
        + [str(Path(path).parent) for path in registration_data_files()]
    ))


def structural_signature() -> str:
    return f"{SIGNATURE_PREFIX}/structural"


def structural_spec(priority: str) -> dict:
    return {
        "description": "V28 polarity compensation structural audit",
        "cmd": _cpu_command(
            "jax_experiments.analysis."
            "run_regime_polarity_action_compensation_structural_v28",
            ["--resume"],
            threads=8,
        ),
        "cwd": str(ROOT),
        "signature": structural_signature(),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 8192,
        "cpu": 8,
        "priority": priority,
        "allowed_nodes": CPU_NODES,
        "result_dir": str(protocol.STRUCTURAL_ROOT),
        "local_result_dir": str(protocol.STRUCTURAL_ROOT),
        "wait_for_files": [str(protocol.REGISTRATION_PATH)],
        "stage_input_paths": registration_data_input_dirs(),
        "stage_excludes": ["paper/"],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Deterministic algebra and short environment trajectory audit; "
            "no parameter updates."
        ),
    }


def audit_signature(seed: int) -> str:
    return (
        f"{SIGNATURE_PREFIX}/audit/seed-"
        f"{protocol.require_training_seed(seed)}"
    )


def audit_spec(seed: int, priority: str) -> dict:
    seed = protocol.require_training_seed(seed)
    return {
        "description": f"V28 action compensation audit seed {seed}",
        "cmd": _cpu_command(
            "jax_experiments.analysis."
            "run_regime_polarity_action_compensation_audit_v28",
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
            str(path) for path in protocol.audit_required_paths(seed)
        ],
        "stage_input_paths": [
            *registration_data_input_dirs(),
            str(protocol.STRUCTURAL_ROOT),
            str(protocol.source_bundle(seed)),
            *(
                str(protocol.specialist_bundle_dir(seed, mode))
                for mode in protocol.MODES
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
        "description": "Aggregate V28 action compensation audit",
        "cmd": _cpu_command(
            "jax_experiments.analysis."
            "analyze_regime_polarity_action_compensation_v28",
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
            *(
                str(protocol.audit_manifest(seed))
                for seed in protocol.TRAINING_SEEDS
            ),
        ],
        "stage_input_paths": [
            *registration_data_input_dirs(),
            str(protocol.AUDIT_ROOT),
            str(protocol.STRUCTURAL_ROOT),
        ],
        "stage_excludes": ["paper/"],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": "Immutable JSON aggregation only.",
    }


def candidates(priority: str):
    rows = [(
        structural_signature(), structural_spec(priority),
        protocol.STRUCTURAL_MANIFEST,
    )]
    rows.extend(
        (
            audit_signature(seed), audit_spec(seed, priority),
            protocol.audit_manifest(seed),
        )
        for seed in protocol.TRAINING_SEEDS
    )
    rows.append((
        analysis_signature(), analysis_spec(priority), protocol.analysis_json(),
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
        "bapr-v28-action-compensation-dispatch",
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
        print(f"V28 dry-run task count: {len(specs)}", flush=True)
        return
    if not specs:
        print("No V28 action-compensation tasks to submit")
        return
    task_ids = _submit(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        _dispatch(task_ids)


if __name__ == "__main__":
    main()
