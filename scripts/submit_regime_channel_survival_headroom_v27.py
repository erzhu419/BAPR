#!/usr/bin/env python3
"""Register and batch-submit the Hopper/Walker V27 headroom DAG."""
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

from jax_experiments.analysis import (
    regime_channel_survival_headroom_v27 as protocol,
)


SIGNATURE_PREFIX = "BAPR/regime-channel/v27-survival-headroom"
SUBMIT_INTENT = "bapr-v27-channel-survival-headroom-submit"
GPU_NODES = ["jtl110gpu", "jtl110gpu2", "jtl311linux", "node007"]
CPU_NODES = [
    "jtl110cpu", "jtl110cpu2", "node001", "node002", "node003",
    "node004", "node005", "node006",
]
ACTIVE_STATUSES = scheduler_common.ACTIVE_STATUSES


def _gpu_command(values: list[str]) -> str:
    return (
        f"SCHEDULEURM_ETA_TOTAL_UNITS={protocol.MAX_ITERS} "
        "JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.32 "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 JAX_NUM_THREADS=1 "
        "TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 "
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        f"{shlex.quote(str(scheduler_common.JAX_PYTHON))} -u -m "
        "jax_experiments.analysis."
        "run_regime_channel_survival_headroom_controller_v27 "
        f"{shlex.join(values)} && echo DONE"
    )


def _cpu_command(module: str, values: list[str]) -> str:
    return (
        "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' TMPDIR=/tmp "
        "XLA_FLAGS='--xla_cpu_multi_thread_eigen=false "
        "intra_op_parallelism_threads=16' "
        "OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 "
        "NUMEXPR_NUM_THREADS=16 JAX_NUM_THREADS=16 "
        "TF_NUM_INTRAOP_THREADS=16 TF_NUM_INTEROP_THREADS=2 "
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        f"{shlex.quote(str(scheduler_common.JAX_PYTHON))} -u -m "
        f"{module} {shlex.join(values)} && echo DONE"
    )


def common_stage_inputs() -> list[str]:
    return [
        str(protocol.REGISTRATION_ROOT),
        str(ROOT / "jax_experiments" / "analysis"),
        str(ROOT / "jax_experiments" / "algos"),
        str(ROOT / "jax_experiments" / "common"),
        str(ROOT / "jax_experiments" / "configs"),
        str(ROOT / "jax_experiments" / "envs"),
        str(ROOT / "jax_experiments" / "networks"),
        str(ROOT / "scripts"),
        str(protocol.PRIOR_HEADROOM_ANALYSIS.parent),
    ]


def train_signature(env: str, role: str, seed: int) -> str:
    return (
        f"{SIGNATURE_PREFIX}/train/{protocol.env_slug(env)}/"
        f"{role}/seed-{seed}"
    )


def train_spec(env: str, role: str, seed: int, priority: str) -> dict:
    run = protocol.run_dir(env, role, seed)
    bundle = protocol.bundle_dir(env, role, seed)
    return {
        "description": (
            f"V27 {protocol.env_slug(env)} {role} headroom seed {seed}"
        ),
        "cmd": _gpu_command([
            "--env", env, "--role", role, "--seed", str(seed), "--resume",
        ]),
        "cwd": str(ROOT),
        "signature": train_signature(env, role, seed),
        "project": "BAPR",
        "vram_resource_family": "BAPR/v27/regime-sac/gpu-runtime",
        "vram": 2600,
        "ram_mb": 6144,
        "cpu": 2,
        "priority": priority,
        "allowed_nodes": GPU_NODES,
        "ckpt_dir": str(run / "checkpoints"),
        "ckpt_glob": "train_state.pkl",
        "resume_flag": "",
        "resume_managed_by_cmd": True,
        "result_dir": str(bundle),
        "local_result_dir": str(bundle),
        "wait_for_files": [str(protocol.REGISTRATION_PATH)],
        "stage_input_paths": common_stage_inputs(),
        "stage_excludes": ["paper/", "__pycache__/"],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
    }


def audit_signature(env: str, role: str, seed: int) -> str:
    return (
        f"{SIGNATURE_PREFIX}/audit/{protocol.env_slug(env)}/"
        f"{role}/seed-{seed}"
    )


def audit_spec(env: str, role: str, seed: int, priority: str) -> dict:
    output = protocol.audit_dir(env, role, seed)
    return {
        "description": (
            f"V27 {protocol.env_slug(env)} {role} strict audit seed {seed}"
        ),
        "cmd": _cpu_command(
            "jax_experiments.analysis."
            "run_regime_channel_survival_headroom_audit_v27",
            ["--env", env, "--role", role, "--seed", str(seed), "--resume"],
        ),
        "cwd": str(ROOT),
        "signature": audit_signature(env, role, seed),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 8192,
        "cpu": 16,
        "priority": priority,
        "allowed_nodes": CPU_NODES,
        "result_dir": str(output),
        "local_result_dir": str(output),
        "wait_for_files": [
            str(protocol.REGISTRATION_PATH),
            *(str(path) for path in protocol.bundle_required_paths(
                env, role, seed)),
        ],
        "stage_input_paths": [
            *common_stage_inputs(), str(protocol.bundle_dir(env, role, seed)),
        ],
        "stage_excludes": ["paper/", "__pycache__/"],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": "Strict policy evaluation only.",
    }


def analysis_signature() -> str:
    return f"{SIGNATURE_PREFIX}/analysis"


def analysis_spec(priority: str) -> dict:
    return {
        "description": "Aggregate V27 Hopper/Walker survival headroom",
        "cmd": _cpu_command(
            "jax_experiments.analysis."
            "analyze_regime_channel_survival_headroom_v27",
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
        "wait_for_files": [
            str(protocol.audit_manifest(env, role, seed))
            for env in protocol.ENVS
            for role in protocol.ROLES
            for seed in protocol.TRAINING_SEEDS
        ],
        "stage_input_paths": [
            *common_stage_inputs(), str(protocol.AUDIT_ROOT),
        ],
        "stage_excludes": ["paper/", "__pycache__/"],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": "JSON and CSV aggregation only.",
    }


def candidates(priority: str):
    rows = []
    for env in protocol.ENVS:
        for role in protocol.ROLES:
            for seed in protocol.TRAINING_SEEDS:
                rows.append((
                    train_signature(env, role, seed),
                    train_spec(env, role, seed, priority),
                    protocol.bundle_manifest(env, role, seed),
                ))
                rows.append((
                    audit_signature(env, role, seed),
                    audit_spec(env, role, seed, priority),
                    protocol.audit_manifest(env, role, seed),
                ))
    rows.append((
        analysis_signature(), analysis_spec(priority), protocol.analysis_json(),
    ))
    return rows


def _submit_jsonl(specs: list[dict]) -> list[str]:
    payload = "".join(json.dumps(spec) + "\n" for spec in specs)
    command = [
        sys.executable, str(scheduler_common.SCHEDULER), "submit-jsonl",
        "--stdin", "--trusted", "--json",
        "--intent-label", SUBMIT_INTENT, "--intent-ttl", "900",
    ]
    result = subprocess.run(
        command, input=payload, text=True, capture_output=True,
        env=scheduler_common.scheduler_env(),
    )
    if result.returncode != 0:
        print((result.stdout or "") + (result.stderr or ""), file=sys.stderr)
        result.check_returncode()
    response = json.loads(result.stdout)
    task_ids = [str(item.get("id", "")) for item in response.get("submitted", [])]
    if len(task_ids) != len(specs) or any(not task_id for task_id in task_ids):
        raise RuntimeError(
            f"scheduler accounted for {len(task_ids)}/{len(specs)} V27 tasks")
    print(json.dumps(response, indent=2))
    return task_ids


def _dispatch(task_ids: list[str]) -> None:
    if not task_ids:
        return
    command = [
        sys.executable, str(scheduler_common.SCHEDULER), "dispatch",
        "--bulk-window", "--intent-label", f"{SUBMIT_INTENT}-dispatch",
        "--intent-ttl", "900",
    ]
    for task_id in task_ids:
        command += ["--task-id", task_id]
    subprocess.run(
        command, check=True, env=scheduler_common.scheduler_env())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="high")
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

    protocol.create_registration()
    known = scheduler_common.scheduler_tasks()
    specs = []
    for signature, spec, output in candidates(args.priority):
        matches = [
            task for task in known
            if str(task.get("signature") or "") == signature
        ]
        active = [
            task for task in matches
            if str(task.get("status")) in ACTIVE_STATUSES
        ]
        if output.is_file():
            print(f"skip complete-output: {signature}")
        elif active:
            print("skip active: " + ",".join(str(task["id"]) for task in active))
        elif matches and not args.retry_incomplete:
            print("skip terminal-incomplete: " + ",".join(
                str(task["id"]) for task in matches))
        else:
            specs.append(spec)
    if args.dry_run:
        print(json.dumps(specs, indent=2, sort_keys=True))
        return
    if not specs:
        print("No V27 tasks to submit")
        return
    task_ids = _submit_jsonl(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        _dispatch(task_ids)


if __name__ == "__main__":
    main()
