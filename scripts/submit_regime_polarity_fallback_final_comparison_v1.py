#!/usr/bin/env python3
"""Submit the frozen causal-fallback final comparison as one scheduler DAG."""
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
    regime_polarity_fallback_final_comparison_v1 as protocol,
)


SIGNATURE_PREFIX = "BAPR/regime-polarity-fallback-final/v1"
SUBMIT_INTENT_LABEL = "bapr-polarity-fallback-final-v1-submit"
GPU_NODES = ["jtl311linux"]
CPU_NODES = [f"node00{index}" for index in range(1, 7)]
ACTIVE_STATUSES = scheduler_common.ACTIVE_STATUSES


def _gpu_command(module: str, values: list[str], total_units: int) -> str:
    return (
        f"SCHEDULEURM_ETA_TOTAL_UNITS={int(total_units)} "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.26 "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
        "OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 "
        "NUMEXPR_NUM_THREADS=2 JAX_NUM_THREADS=2 "
        "TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 "
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        f"{shlex.quote(str(scheduler_common.JAX_PYTHON))} -u -m "
        f"{module} {shlex.join(values)} && echo DONE"
    )


def _cpu_command(module: str, values: list[str]) -> str:
    return (
        "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' "
        "XLA_FLAGS='--xla_cpu_multi_thread_eigen=false "
        "intra_op_parallelism_threads=4' "
        "OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 "
        "NUMEXPR_NUM_THREADS=4 JAX_NUM_THREADS=4 "
        "JAX_CPU_ENABLE_ASYNC_DISPATCH=false "
        "TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2 "
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        f"{shlex.quote(str(scheduler_common.JAX_PYTHON))} -u -m "
        f"{module} {shlex.join(values)} && echo DONE"
    )


def student_signature(seed: int) -> str:
    return f"{SIGNATURE_PREFIX}/train/bapr/seed-{int(seed)}"


def baseline_signature(role: str, seed: int) -> str:
    return f"{SIGNATURE_PREFIX}/train/{role}/seed-{int(seed)}"


def audit_signature(method: str, seed: int) -> str:
    return f"{SIGNATURE_PREFIX}/audit/{method}/seed-{int(seed)}"


def analysis_signature() -> str:
    return f"{SIGNATURE_PREFIX}/analysis"


def _student_source_files() -> list[str]:
    return [
        str(protocol.REGISTRATION_PATH),
        str(protocol.ensemble.final.MODEL_MANIFEST),
        str(protocol.ensemble.final.MODEL_PATH),
        *(str(path) for path in protocol.source_required_paths("mode_heads")),
    ]


def _registration_artifact_sources() -> list[Path]:
    prefixes = (
        "jax_experiments/results",
        "jax_experiments/eval_bundles",
    )
    return [
        path for path in protocol.registration_source_paths()
        if path.relative_to(ROOT).as_posix().startswith(prefixes)
    ]


def _registration_stage_inputs() -> list[str]:
    paths = [
        protocol.REGISTRATION_ROOT,
        *(path.parent for path in _registration_artifact_sources()),
    ]
    return list(dict.fromkeys(str(path) for path in paths))


def _student_stage_inputs() -> list[str]:
    return list(dict.fromkeys([
        *_registration_stage_inputs(),
        str(protocol.ensemble.final.MODEL_MANIFEST.parent),
        *(str(path) for path in protocol.source_bundle_dirs("mode_heads")),
    ]))


def student_spec(seed: int, priority: str) -> dict:
    seed = protocol.require_student_seed(seed)
    output = protocol.model_dir("mode_heads", seed)
    work = protocol.work_dir("mode_heads", seed)
    return {
        "description": f"Final causal-fallback BAPR student seed {seed}",
        "cmd": _gpu_command(
            "jax_experiments.analysis."
            "train_regime_polarity_fallback_final_student_v1",
            ["--variant", "mode_heads", "--student-seed", str(seed),
             "--resume"],
            1 + protocol.DAGGER_ROUNDS,
        ),
        "cwd": str(ROOT),
        "signature": student_signature(seed),
        "project": "BAPR",
        "vram_resource_family": (
            "BAPR/regime-polarity-fallback-final/student-runtime"),
        "vram": 2400,
        "allow_gpu_over_one_third": True,
        "ram_mb": 12_288,
        "cpu": 2,
        "priority": priority,
        "allowed_nodes": GPU_NODES,
        "ckpt_dir": str(work),
        "ckpt_glob": "phase_*/checkpoint_manifest.json",
        "resume_flag": "",
        "resume_managed_by_cmd": True,
        "result_dir": str(output),
        "local_result_dir": str(output),
        "wait_for_files": _student_source_files(),
        "stage_input_paths": _student_stage_inputs(),
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
    }


def baseline_spec(role: str, seed: int, priority: str) -> dict:
    role = protocol.require_baseline_role(role)
    seed = protocol.require_training_seed(seed)
    run = protocol.baseline_run_dir(role, seed)
    bundle = protocol.baseline_bundle_dir(role, seed)
    return {
        "description": f"Final polarity {role} baseline seed {seed}",
        "cmd": _gpu_command(
            "jax_experiments.analysis."
            "run_regime_polarity_fallback_final_baseline_v1",
            ["--role", role, "--seed", str(seed), "--resume"],
            protocol.MAX_ITERS,
        ),
        "cwd": str(ROOT),
        "signature": baseline_signature(role, seed),
        "project": "BAPR",
        "vram_resource_family": (
            "BAPR/regime-polarity-fallback-final/baseline-runtime"),
        "vram": 2300,
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
        "stage_input_paths": _registration_stage_inputs(),
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
    }


def audit_spec(method: str, seed: int, priority: str) -> dict:
    seed = protocol.require_training_seed(seed)
    output = protocol.audit_dir(method, seed)
    if method == "bapr":
        waits = [
            str(protocol.REGISTRATION_PATH),
            str(protocol.model_manifest("mode_heads", seed)),
            str(protocol.model_path("mode_heads", seed)),
            str(protocol.ensemble.final.MODEL_MANIFEST),
            str(protocol.ensemble.final.MODEL_PATH),
            *(str(path) for path in
              protocol.development.ensemble.final.bundle_required_paths(
                  protocol.ENV, "robust", protocol.ROBUST_SEED)),
        ]
        stages = [
            *_registration_stage_inputs(),
            str(protocol.model_dir("mode_heads", seed)),
            str(protocol.ensemble.final.MODEL_MANIFEST.parent),
            str(protocol.development.ensemble.final.bundle_dir(
                protocol.ENV, "robust", protocol.ROBUST_SEED)),
        ]
    else:
        role = protocol.require_baseline_role(method)
        waits = [
            str(protocol.REGISTRATION_PATH),
            *(str(path) for path in
              protocol.baseline_bundle_required_paths(role, seed)),
        ]
        stages = [
            *_registration_stage_inputs(),
            str(protocol.baseline_bundle_dir(role, seed)),
        ]
    return {
        "description": f"Final comparison audit {method} seed {seed}",
        "cmd": _cpu_command(
            "jax_experiments.analysis."
            "run_regime_polarity_fallback_final_audit_v1",
            ["--method", method, "--seed", str(seed), "--resume"],
        ),
        "cwd": str(ROOT),
        "signature": audit_signature(method, seed),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 12_288,
        "cpu": 32,
        "priority": priority,
        "allowed_nodes": CPU_NODES,
        "result_dir": str(output),
        "local_result_dir": str(output),
        "wait_for_files": waits,
        "stage_input_paths": stages,
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Strict deterministic checkpoint audit; no optimizer updates."),
    }


def analysis_spec(priority: str) -> dict:
    return {
        "description": "Aggregate causal-fallback final comparison",
        "cmd": _cpu_command(
            "jax_experiments.analysis."
            "analyze_regime_polarity_fallback_final_comparison_v1",
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
            str(path) for path in protocol.all_audit_manifests()],
        "stage_input_paths": [
            *_registration_stage_inputs(), str(protocol.AUDIT_ROOT)],
        "stage_excludes": ["paper/"],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": "Immutable JSON aggregation only.",
    }


def candidates(phase: str, priority: str):
    rows = []
    if phase in ("all", "train"):
        rows.extend([
            (student_signature(seed), student_spec(seed, priority),
             protocol.model_manifest("mode_heads", seed))
            for seed in protocol.STUDENT_SEEDS
        ])
        rows.extend([
            (baseline_signature(role, seed),
             baseline_spec(role, seed, priority),
             protocol.baseline_bundle_manifest(role, seed))
            for role in protocol.BASELINE_ROLES
            for seed in protocol.TRAINING_SEEDS
        ])
    if phase in ("all", "audit"):
        rows.extend([
            (audit_signature(method, seed),
             audit_spec(method, seed, priority),
             protocol.audit_manifest(method, seed))
            for method in ("bapr", *protocol.BASELINE_ROLES)
            for seed in protocol.TRAINING_SEEDS
        ])
    if phase in ("all", "analysis"):
        rows.append((
            analysis_signature(), analysis_spec(priority),
            protocol.analysis_json()))
    return rows


def _submit_jsonl(specs: list[dict]) -> list[str]:
    payload = "".join(json.dumps(spec) + "\n" for spec in specs)
    command = [
        sys.executable,
        str(scheduler_common.SCHEDULER),
        "submit-jsonl",
        "--stdin",
        "--trusted",
        "--json",
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
    submitted = response.get("submitted", [])
    task_ids = [str(item.get("id", "")) for item in submitted]
    if (len(task_ids) != len(specs)
            or any(not task_id for task_id in task_ids)
            or len(set(task_ids)) != len(task_ids)):
        raise RuntimeError(
            "scheduler batch was not fully accounted: "
            f"requested={len(specs)}, returned={task_ids}")
    return task_ids


def _dispatch(task_ids: list[str]) -> None:
    if not task_ids:
        return
    command = [
        sys.executable,
        str(scheduler_common.SCHEDULER),
        "dispatch",
        "--bulk-window",
        "--intent-label", "bapr-polarity-fallback-final-v1-dispatch",
        "--intent-ttl", "900",
    ]
    for task_id in task_ids:
        command += ["--task-id", task_id]
    subprocess.run(
        command, check=True, env=scheduler_common.scheduler_env())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("all", "train", "audit", "analysis"),
        default="all")
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="high")
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

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
        print("No final comparison tasks to submit")
        return
    task_ids = _submit_jsonl(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        train_ids = [
            task_id for task_id, spec in zip(task_ids, specs)
            if "/train/" in str(spec["signature"])
        ]
        _dispatch(train_ids)


if __name__ == "__main__":
    main()
