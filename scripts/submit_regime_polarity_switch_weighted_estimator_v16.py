#!/usr/bin/env python3
"""Register and submit v16 switch-weighted estimator development."""
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
    regime_polarity_conflict_fallback_confirmation_v15 as parent,
)
from jax_experiments.analysis import (
    regime_polarity_conflict_fallback_router_v14 as router_parent,
)
from jax_experiments.analysis import (
    regime_polarity_frozen_estimator_transfer_v13 as transfer_parent,
)
from jax_experiments.analysis import (
    regime_polarity_robust_warmstart_confirmation_v12 as policy_parent,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_system_id_v5 as estimator_parent,
)
from jax_experiments.analysis import (
    regime_polarity_switch_weighted_estimator_v16 as protocol,
)


SIGNATURE_PREFIX = "BAPR/regime-polarity/v16-switch-weighted-estimator"
SUBMIT_INTENT = "bapr-v16-switch-weighted-estimator-submit"
GPU_NODES = [
    "local", "jtl110gpu", "jtl110gpu2", "jtl311linux", "node007"]
CPU_NODES = [f"node00{index}" for index in range(1, 7)]


def _gpu_command(module: str, values: list[str]) -> str:
    return (
        f"SCHEDULEURM_ETA_TOTAL_UNITS={protocol.TRAIN_UPDATES} "
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


def registration_data_files() -> list[str]:
    paths = [
        *protocol.registration_source_paths(),
        *parent.registration_source_paths(),
        *transfer_parent.registration_source_paths(),
    ]
    return list(dict.fromkeys(
        str(path) for path in paths if path.suffix != ".py"
    ))


def frozen_input_roots() -> list[str]:
    return [
        str(protocol.REGISTRATION_ROOT),
        str(parent.REGISTRATION_ROOT),
        str(parent.ANALYSIS_ROOT),
        str(parent.AUDIT_ROOT),
        str(router_parent.REGISTRATION_ROOT),
        str(router_parent.ANALYSIS_ROOT),
        str(transfer_parent.REGISTRATION_ROOT),
        str(transfer_parent.ANALYSIS_ROOT),
        str(transfer_parent.AUDIT_ROOT),
        str(policy_parent.REGISTRATION_ROOT),
        str(policy_parent.ANALYSIS_ROOT),
        str(policy_parent.AUDIT_ROOT),
        str(policy_parent.SOURCE_BUNDLE_ROOT),
        str(policy_parent.SPECIALIST_BUNDLE_ROOT),
        str(estimator_parent.MODEL_ROOT),
        str(ROOT / "reports"),
    ]


def _policy_wait_files(seed: int) -> list[str]:
    seed = protocol.require_policy_seed(seed)
    return [
        *(str(path) for path in policy_parent.source_required_paths(seed)),
        *(
            str(path)
            for mode in policy_parent.MODES
            for path in policy_parent.bundle_required_paths(
                "actor_only", seed, mode)
        ),
        str(transfer_parent.audit_manifest(seed)),
        str(parent.audit_manifest(seed)),
    ]


def training_signature() -> str:
    return f"{SIGNATURE_PREFIX}/train"


def training_spec(priority: str) -> dict:
    return {
        "description": "Train v16 switch-weighted expected-action estimator",
        "cmd": _gpu_command(
            "jax_experiments.analysis."
            "train_regime_polarity_switch_weighted_estimator_v16",
            ["--resume"],
        ),
        "cwd": str(ROOT),
        "signature": training_signature(),
        "project": "BAPR",
        "vram_resource_family": (
            "BAPR/regime-polarity/expected-action-system-id/gpu-runtime"),
        "vram": 1024,
        "ram_mb": 4096,
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
            str(protocol.REGISTRATION_PATH),
            *registration_data_files(),
            str(estimator_parent.MODEL_MANIFEST),
            str(estimator_parent.MODEL_PATH),
            *(
                path
                for seed in (
                    *protocol.TRAIN_POLICY_SEEDS,
                    *protocol.VALIDATION_POLICY_SEEDS,
                )
                for path in _policy_wait_files(seed)
            ),
        ],
        "stage_input_paths": frozen_input_roots(),
        "stage_excludes": ["paper/"],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
    }


def audit_signature(seed: int) -> str:
    return f"{SIGNATURE_PREFIX}/audit/seed-{protocol.require_policy_seed(seed)}"


def audit_spec(seed: int, priority: str) -> dict:
    seed = protocol.require_policy_seed(seed)
    return {
        "description": f"Audit v16 switch-weighted estimator seed {seed}",
        "cmd": _cpu_command(
            "jax_experiments.analysis."
            "run_regime_polarity_switch_weighted_estimator_audit_v16",
            ["--seed", str(seed), "--resume"],
            threads=16,
        ),
        "cwd": str(ROOT),
        "signature": audit_signature(seed),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 8192,
        "cpu": 16,
        "priority": priority,
        "allowed_nodes": CPU_NODES,
        "result_dir": str(protocol.audit_dir(seed)),
        "local_result_dir": str(protocol.audit_dir(seed)),
        "wait_for_files": [
            str(protocol.REGISTRATION_PATH),
            *registration_data_files(),
            str(protocol.MODEL_MANIFEST),
            str(protocol.MODEL_PATH),
            *_policy_wait_files(seed),
        ],
        "stage_input_paths": [
            str(protocol.MODEL_ROOT),
            *frozen_input_roots(),
        ],
        "stage_excludes": ["paper/"],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Frozen-controller and frozen-estimator evaluation; no updates."
        ),
    }


def analysis_signature() -> str:
    return f"{SIGNATURE_PREFIX}/analysis"


def analysis_spec(priority: str) -> dict:
    return {
        "description": "Aggregate v16 switch-weighted estimator development",
        "cmd": _cpu_command(
            "jax_experiments.analysis."
            "analyze_regime_polarity_switch_weighted_estimator_v16",
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
                for seed in protocol.AUDIT_POLICY_SEEDS
            ),
        ],
        "stage_input_paths": [
            str(protocol.AUDIT_ROOT),
            *frozen_input_roots(),
        ],
        "stage_excludes": ["paper/"],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": "Immutable JSON aggregation only.",
    }


def candidates(priority: str):
    rows = [(
        training_signature(), training_spec(priority), protocol.MODEL_MANIFEST,
    )]
    rows.extend(
        (
            audit_signature(seed),
            audit_spec(seed, priority),
            protocol.audit_manifest(seed),
        )
        for seed in protocol.AUDIT_POLICY_SEEDS
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
        "bapr-v16-switch-weighted-estimator-dispatch",
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
        active = [
            task for task in known
            if str(task.get("signature") or "") == signature
            and str(task.get("status")) in scheduler_common.ACTIVE_STATUSES
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
        print("No v16 switch-weighted estimator tasks to submit")
        return
    task_ids = _submit(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        _dispatch(task_ids)


if __name__ == "__main__":
    main()
