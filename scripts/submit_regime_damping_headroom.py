#!/usr/bin/env python3
"""Submit the registered persistent-damping headroom DAG via scheduler."""
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

from jax_experiments.analysis import regime_damping_headroom as protocol


SIGNATURE_PREFIX = "BAPR/regime-damping-headroom/v1"
SUBMIT_INTENT_LABEL = "bapr-regime-damping-headroom-v1-submit"
MEASURED_VRAM_MB = 2300
GPU_NODES = [
    "local", "jtl110gpu", "jtl110gpu2", "jtl311linux", "node007"
]
CPU_NODES = [f"node00{index}" for index in range(1, 7)]
ACTIVE_STATUSES = scheduler_common.ACTIVE_STATUSES


def _gpu_command(values: list[str]) -> str:
    return (
        f"SCHEDULEURM_ETA_TOTAL_UNITS={protocol.MAX_ITERS} "
        "BAPR_SCHEDULER_RESUME=--resume "
        "JAX_PLATFORMS=cuda "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.22 "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
        "OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 "
        "NUMEXPR_NUM_THREADS=2 JAX_NUM_THREADS=2 "
        "TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 "
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        f"{shlex.quote(str(scheduler_common.JAX_PYTHON))} -u -m "
        "jax_experiments.analysis.run_regime_damping_headroom_controller "
        f"{shlex.join(values)} && echo DONE"
    )


def _cpu_command(module: str, values: list[str], threads: int = 4) -> str:
    threads = int(threads)
    return (
        "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' "
        "XLA_FLAGS='--xla_cpu_multi_thread_eigen=false "
        f"intra_op_parallelism_threads={threads}' "
        f"OMP_NUM_THREADS={threads} OPENBLAS_NUM_THREADS={threads} "
        f"MKL_NUM_THREADS={threads} NUMEXPR_NUM_THREADS={threads} "
        f"JAX_NUM_THREADS={threads} "
        "JAX_CPU_ENABLE_ASYNC_DISPATCH=false "
        f"TF_NUM_INTRAOP_THREADS={threads} TF_NUM_INTEROP_THREADS=2 "
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        f"{shlex.quote(str(scheduler_common.JAX_PYTHON))} -u -m "
        f"{module} {shlex.join(values)} && echo DONE"
    )


def training_signature(env: str, role: str, seed: int) -> str:
    return (
        f"{SIGNATURE_PREFIX}/train/{protocol.env_slug(env)}/"
        f"{protocol.require_role(role)}/seed-{protocol.require_training_seed(seed)}"
    )


def audit_signature(env: str, role: str, seed: int) -> str:
    return (
        f"{SIGNATURE_PREFIX}/audit/{protocol.env_slug(env)}/"
        f"{protocol.require_role(role)}/seed-{protocol.require_training_seed(seed)}"
    )


def analysis_signature() -> str:
    return f"{SIGNATURE_PREFIX}/analysis"


def training_spec(env: str, role: str, seed: int, priority: str) -> dict:
    env = protocol.require_env(env)
    role = protocol.require_role(role)
    seed = protocol.require_training_seed(seed)
    run = protocol.run_dir(env, role, seed)
    bundle = protocol.bundle_dir(env, role, seed)
    return {
        "description": (
            f"Damping headroom {protocol.env_slug(env)} {role} seed {seed}"
        ),
        "cmd": _gpu_command([
            "--env", env, "--role", role, "--seed", str(seed), "--resume"
        ]),
        "cwd": str(ROOT),
        "signature": training_signature(env, role, seed),
        "project": "BAPR",
        "vram_resource_family": (
            "BAPR/regime-damping-headroom/regime-sac/gpu-runtime"),
        "vram": MEASURED_VRAM_MB,
        "allow_gpu_over_one_third": True,
        "ram_mb": 8192,
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
        "stage_input_paths": [str(protocol.REGISTRATION_ROOT)],
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": False,
    }


def audit_spec(env: str, role: str, seed: int, priority: str) -> dict:
    env = protocol.require_env(env)
    role = protocol.require_role(role)
    seed = protocol.require_training_seed(seed)
    output = protocol.audit_dir(env, role, seed)
    return {
        "description": (
            f"Damping strict audit {protocol.env_slug(env)} {role} seed {seed}"
        ),
        "cmd": _cpu_command(
            "jax_experiments.analysis.run_regime_damping_headroom_audit",
            ["--env", env, "--role", role, "--seed", str(seed), "--resume"],
        ),
        "cwd": str(ROOT),
        "signature": audit_signature(env, role, seed),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 12_288,
        "cpu": 32,
        "priority": priority,
        "allowed_nodes": CPU_NODES,
        "result_dir": str(output),
        "local_result_dir": str(output),
        "wait_for_files": [
            str(protocol.REGISTRATION_PATH),
            *(str(path) for path in
              protocol.bundle_required_paths(env, role, seed)),
        ],
        "stage_input_paths": [
            str(protocol.REGISTRATION_ROOT),
            str(protocol.bundle_dir(env, role, seed)),
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
            "Strict fixed-checkpoint policy evaluation only; no updates."
        ),
    }


def analysis_spec(priority: str) -> dict:
    return {
        "description": "Aggregate persistent-damping oracle headroom",
        "cmd": _cpu_command(
            "jax_experiments.analysis.analyze_regime_damping_headroom",
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
            *(str(protocol.audit_manifest(env, role, seed))
              for env in protocol.ENVS
              for role in protocol.ROLES
              for seed in protocol.TRAINING_SEEDS),
        ],
        "stage_input_paths": [
            str(protocol.REGISTRATION_ROOT), str(protocol.AUDIT_ROOT)
        ],
        "stage_excludes": ["paper/"],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": "Immutable CSV/JSON aggregation only.",
    }


def candidates(phase: str, priority: str):
    rows = []
    if phase in ("all", "training"):
        rows.extend([
            (
                training_signature(env, role, seed),
                training_spec(env, role, seed, priority),
                protocol.bundle_manifest(env, role, seed),
            )
            for env in protocol.ENVS
            for role in protocol.ROLES
            for seed in protocol.TRAINING_SEEDS
        ])
    if phase in ("all", "audit"):
        rows.extend([
            (
                audit_signature(env, role, seed),
                audit_spec(env, role, seed, priority),
                protocol.audit_manifest(env, role, seed),
            )
            for env in protocol.ENVS
            for role in protocol.ROLES
            for seed in protocol.TRAINING_SEEDS
        ])
    if phase in ("all", "analysis"):
        rows.append((
            analysis_signature(), analysis_spec(priority),
            protocol.analysis_json(),
        ))
    return rows


def _submit_jsonl(specs: list[dict]) -> list[str]:
    payload = "".join(json.dumps(spec) + "\n" for spec in specs)
    command = [
        sys.executable,
        str(scheduler_common.SCHEDULER),
        "submit-jsonl", "--stdin", "--trusted", "--json",
        "--intent-label", SUBMIT_INTENT_LABEL, "--intent-ttl", "900",
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
    if (
        len(task_ids) != len(specs)
        or any(not task_id for task_id in task_ids)
        or len(set(task_ids)) != len(task_ids)
    ):
        raise RuntimeError(
            "scheduler batch was not fully accounted: "
            f"requested={len(specs)}, returned={task_ids}"
        )
    return task_ids


def _dispatch(task_ids: list[str]) -> None:
    if not task_ids:
        return
    command = [
        sys.executable,
        str(scheduler_common.SCHEDULER),
        "dispatch", "--bulk-window",
        "--intent-label", "bapr-regime-damping-headroom-v1-dispatch",
        "--intent-ttl", "900",
    ]
    for task_id in task_ids:
        command += ["--task-id", task_id]
    subprocess.run(
        command, check=True, env=scheduler_common.scheduler_env())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("all", "training", "audit", "analysis"),
        default="all")
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="high")
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

    if not protocol.REGISTRATION_PATH.is_file():
        protocol.create_registration()
    protocol.validate_registration()
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
        print("No persistent-damping headroom tasks to submit")
        return
    task_ids = _submit_jsonl(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        training_ids = [
            task_id for task_id, spec in zip(task_ids, specs)
            if "/train/" in str(spec["signature"])
        ]
        _dispatch(training_ids)


if __name__ == "__main__":
    main()
