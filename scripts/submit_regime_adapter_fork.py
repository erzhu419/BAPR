#!/usr/bin/env python3
"""Submit the frozen-base independent-adapter development screen."""
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

from jax_experiments.analysis import regime_adapter_fork as protocol


SIGNATURE_PREFIX = "BAPR/regime-adapter-fork/v1"
CPU_NODES = [f"node00{index}" for index in range(1, 7)]
COLD_START_VRAM_MB = 2048


def _gpu_command(values: list[str], fraction: float) -> str:
    return (
        f"SCHEDULEURM_ETA_TOTAL_UNITS={values[-1]} "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        f"XLA_PYTHON_CLIENT_MEM_FRACTION={fraction:.2f} "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1' "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 JAX_NUM_THREADS=1 "
        "TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 "
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        f"{shlex.quote(str(scheduler_common.JAX_PYTHON))} -u -m "
        "jax_experiments.analysis.run_regime_adapter_branch "
        f"{shlex.join(values[:-1])} --resume && echo DONE"
    )


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


def robust_signature(seed: int) -> str:
    return f"{SIGNATURE_PREFIX}/train/seed-{seed}/robust-continue"


def adapter_signature(seed: int, delta: float, mode: int) -> str:
    return (
        f"{SIGNATURE_PREFIX}/train/seed-{seed}/"
        f"{protocol.delta_slug(delta)}/mode-{mode}")


def calibration_signature(seed: int, delta: float) -> str:
    return (
        f"{SIGNATURE_PREFIX}/calibration/seed-{seed}/"
        f"{protocol.delta_slug(delta)}")


def audit_signature(seed: int, delta: float, event_seed: int) -> str:
    return (
        f"{SIGNATURE_PREFIX}/audit/seed-{seed}/"
        f"{protocol.delta_slug(delta)}/event-{event_seed}")


def analysis_signature() -> str:
    return f"{SIGNATURE_PREFIX}/analysis"


def _source_wait_paths(seed: int) -> list[str]:
    directory = protocol.source_bundle_dir(seed)
    return [str(path) for path in (
        directory / "bundle_manifest.json",
        directory / "checkpoints" / "params.pkl",
        directory / "checkpoints" / "train_state.pkl",
        directory / "logs" / "protocol_signature.json",
    )]


def _common_training_spec(
        seed: int, signature: str, description: str, command: str,
        run_dir: Path, bundle_dir: Path, priority: str) -> dict:
    source_dir = protocol.source_bundle_dir(seed)
    return {
        "description": description,
        "cmd": command,
        "cwd": str(ROOT),
        "signature": signature,
        "project": "BAPR",
        "vram_resource_family": (
            f"BAPR/regime-adapter-fork/{signature.split('/')[-1]}/gpu"),
        "vram": COLD_START_VRAM_MB,
        "ram_mb": 8192,
        "cpu": 2,
        "priority": priority,
        "ckpt_dir": str(run_dir / "checkpoints"),
        "ckpt_glob": "train_state.pkl",
        "resume_flag": "",
        "resume_managed_by_cmd": True,
        "result_dir": str(bundle_dir),
        "local_result_dir": str(bundle_dir),
        "wait_for_files": _source_wait_paths(seed),
        "stage_input_paths": [str(source_dir)],
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
    }


def robust_spec(seed: int, priority: str) -> dict:
    return _common_training_spec(
        seed, robust_signature(seed),
        f"Regime adapter fork robust continuation seed {seed}",
        _gpu_command([
            "--seed", str(seed), "--role", "robust_continue",
            str(protocol.ROBUST_FINAL_NEXT_ITERATION)], 0.24),
        protocol.robust_run_dir(seed), protocol.robust_bundle_dir(seed),
        priority)


def adapter_spec(seed: int, delta: float, mode: int,
                 priority: str) -> dict:
    return _common_training_spec(
        seed, adapter_signature(seed, delta, mode),
        f"Frozen-base adapter seed {seed} delta {delta} mode {mode}",
        _gpu_command([
            "--seed", str(seed), "--role", "adapter",
            "--delta", str(delta), "--mode", str(mode),
            str(protocol.ADAPTER_FINAL_NEXT_ITERATION)], 0.28),
        protocol.adapter_run_dir(seed, delta, mode),
        protocol.adapter_bundle_dir(seed, delta, mode), priority)


def _branch_wait_paths(seed: int, delta: float) -> list[str]:
    return [
        str(path)
        for directory in protocol.all_training_bundle_dirs(seed, delta)
        for path in protocol.required_bundle_paths(directory)
    ]


def calibration_spec(seed: int, delta: float, priority: str) -> dict:
    output = protocol.calibration_dir(seed, delta)
    return {
        "description": (
            f"Calibrate adapter utility map seed {seed} delta {delta}"),
        "cmd": _cpu_command(
            "jax_experiments.analysis.run_regime_adapter_calibration",
            ["--seed", str(seed), "--delta", str(delta), "--resume"]),
        "cwd": str(ROOT),
        "signature": calibration_signature(seed, delta),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 12288,
        "cpu": 32,
        "priority": priority,
        "allowed_nodes": CPU_NODES,
        "result_dir": str(output),
        "local_result_dir": str(output),
        "wait_for_files": _branch_wait_paths(seed, delta),
        "stage_input_paths": [
            str(directory)
            for directory in protocol.all_training_bundle_dirs(seed, delta)
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
            "Checkpoint-only deterministic policy calibration; no updates."),
    }


def audit_spec(
        seed: int, delta: float, event_seed: int,
        priority: str) -> dict:
    output = protocol.audit_dir(seed, delta, event_seed)
    return {
        "description": (
            f"Sealed adapter audit seed {seed} delta {delta} "
            f"event {event_seed}"),
        "cmd": _cpu_command(
            "jax_experiments.analysis.run_regime_adapter_audit",
            ["--seed", str(seed), "--delta", str(delta),
             "--event-seed", str(event_seed), "--resume"]),
        "cwd": str(ROOT),
        "signature": audit_signature(seed, delta, event_seed),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 12288,
        "cpu": 32,
        "priority": priority,
        "allowed_nodes": CPU_NODES,
        "result_dir": str(output),
        "local_result_dir": str(output),
        "wait_for_files": [
            *_branch_wait_paths(seed, delta),
            str(protocol.calibration_manifest(seed, delta)),
            str(protocol.calibration_dir(seed, delta) / "utility_map.json"),
        ],
        "stage_input_paths": [
            *(str(directory) for directory in
              protocol.all_training_bundle_dirs(seed, delta)),
            str(protocol.calibration_dir(seed, delta)),
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
            "Checkpoint-only strict-horizon policy evaluation; no updates."),
    }


def analysis_spec(priority: str) -> dict:
    return {
        "description": "Aggregate frozen-base independent-adapter screen",
        "cmd": _cpu_command(
            "jax_experiments.analysis.analyze_regime_adapter_fork", []),
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
            str(protocol.audit_manifest(seed, delta, event_seed))
            for seed in protocol.DEVELOPMENT_SEEDS
            for delta in protocol.RESIDUAL_DELTAS
            for event_seed in protocol.AUDIT_EVENT_SEEDS
        ],
        "stage_input_paths": [str(protocol.AUDIT_ROOT)],
        "stage_excludes": [
            "jax_experiments/results*/",
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
    if phase in ("all", "train"):
        for seed in protocol.DEVELOPMENT_SEEDS:
            rows.append((
                robust_signature(seed), robust_spec(seed, priority),
                protocol.bundle_manifest(protocol.robust_bundle_dir(seed))))
            for delta in protocol.RESIDUAL_DELTAS:
                for mode in protocol.MODES:
                    rows.append((
                        adapter_signature(seed, delta, mode),
                        adapter_spec(seed, delta, mode, priority),
                        protocol.bundle_manifest(
                            protocol.adapter_bundle_dir(
                                seed, delta, mode))))
    if phase in ("all", "calibration"):
        for seed in protocol.DEVELOPMENT_SEEDS:
            for delta in protocol.RESIDUAL_DELTAS:
                rows.append((
                    calibration_signature(seed, delta),
                    calibration_spec(seed, delta, priority),
                    protocol.calibration_manifest(seed, delta)))
    if phase in ("all", "audit"):
        for seed in protocol.DEVELOPMENT_SEEDS:
            for delta in protocol.RESIDUAL_DELTAS:
                for event_seed in protocol.AUDIT_EVENT_SEEDS:
                    rows.append((
                        audit_signature(seed, delta, event_seed),
                        audit_spec(seed, delta, event_seed, priority),
                        protocol.audit_manifest(
                            seed, delta, event_seed)))
    if phase in ("all", "analysis"):
        rows.append((
            analysis_signature(), analysis_spec(priority),
            protocol.analysis_json()))
    return rows


def _submit_jsonl(specs: list[dict]) -> list[str]:
    payload = "".join(json.dumps(spec) + "\n" for spec in specs)
    command = [
        sys.executable, str(scheduler_common.SCHEDULER), "submit-jsonl",
        "--stdin", "--trusted", "--json",
        "--intent-label", "bapr-regime-adapter-fork-v1-submit",
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
            "scheduler submission was not all-or-verifiably-accounted")
    return task_ids


def _dispatch(task_ids: list[str]) -> None:
    if not task_ids:
        return
    command = [
        sys.executable, str(scheduler_common.SCHEDULER), "dispatch",
        "--bulk-window",
        "--intent-label", "bapr-regime-adapter-fork-v1-dispatch",
        "--intent-ttl", "900",
    ]
    for task_id in task_ids:
        command += ["--task-id", task_id]
    subprocess.run(
        command, check=True, env=scheduler_common.scheduler_env())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=(
            "all", "train", "calibration", "audit", "analysis"),
        default="all")
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
            if str(task.get("status")) in scheduler_common.ACTIVE_STATUSES
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
        print("No regime-adapter tasks to submit")
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
