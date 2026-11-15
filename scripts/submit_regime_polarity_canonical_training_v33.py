#!/usr/bin/env python3
"""Register and submit the V33 canonical-controller training ablation DAG."""
from __future__ import annotations

import argparse
import os
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
        [str(scheduler_common.JAX_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
    )

import submit_regime_polarity_action_compensation_confirmation_v31 as base
from jax_experiments.analysis import (
    regime_polarity_canonical_training_ablation_v33 as protocol,
)


SIGNATURE_PREFIX = "BAPR/regime-polarity/v33-canonical-training"
SUBMIT_INTENT = "bapr-v33-canonical-training-submit"
DISPATCH_INTENT = "bapr-v33-canonical-training-dispatch"
GPU_NODES = ["jtl110gpu", "jtl110gpu2", "jtl311linux", "node007"]
CPU_NODES = [f"node00{index}" for index in range(1, 8)]
ACTIVE_STATUSES = scheduler_common.ACTIVE_STATUSES


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


def _common_gpu_spec(
    *, description: str, command: str, signature: str, run: Path,
    bundle: Path, priority: str, wait_for_files: list[str],
    stage_input_paths: list[str],
) -> dict:
    return {
        "description": description,
        "cmd": command,
        "cwd": str(ROOT),
        "signature": signature,
        "project": "BAPR",
        "vram": 1200,
        "ram_mb": 3072,
        "cpu": 2,
        "priority": priority,
        "allowed_nodes": GPU_NODES,
        "ckpt_dir": str(run / "checkpoints"),
        "ckpt_glob": "train_state.pkl",
        "resume_flag": "",
        "resume_managed_by_cmd": True,
        "result_dir": str(bundle),
        "local_result_dir": str(bundle),
        "wait_for_files": wait_for_files,
        "stage_input_paths": stage_input_paths,
        "stage_excludes": ["paper/"],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
    }


def source_signature(seed: int) -> str:
    return f"{SIGNATURE_PREFIX}/train/robust-source/seed-{protocol.require_training_seed(seed)}"


def source_spec(seed: int, priority: str) -> dict:
    seed = protocol.require_training_seed(seed)
    return _common_gpu_spec(
        description=f"V33 robust warm-start source seed {seed}",
        command=base._gpu_command(
            "jax_experiments.analysis."
            "run_regime_polarity_canonical_training_source_v33",
            ["--seed", str(seed), "--resume"],
            protocol.SOURCE_NEXT_ITERATION,
            0.20,
        ),
        signature=source_signature(seed),
        run=protocol.source_run_dir(seed),
        bundle=protocol.source_bundle(seed),
        priority=priority,
        wait_for_files=[
            str(protocol.REGISTRATION_PATH), *registration_data_files()],
        stage_input_paths=registration_data_input_dirs(),
    )


def warmstart_signature(seed: int) -> str:
    return f"{SIGNATURE_PREFIX}/train/warmstart-mode0/seed-{protocol.require_training_seed(seed)}"


def warmstart_spec(seed: int, priority: str) -> dict:
    seed = protocol.require_training_seed(seed)
    return _common_gpu_spec(
        description=f"V33 robust-to-mode0 warm-start seed {seed}",
        command=base._gpu_command(
            "jax_experiments.analysis."
            "run_regime_polarity_canonical_training_warmstart_v33",
            ["--seed", str(seed), "--resume"],
            protocol.FINAL_NEXT_ITERATION,
            0.20,
        ),
        signature=warmstart_signature(seed),
        run=protocol.run_dir(
            protocol.WARMSTART_RECIPE, seed, protocol.REFERENCE_MODE),
        bundle=protocol.bundle_dir(
            protocol.WARMSTART_RECIPE, seed, protocol.REFERENCE_MODE),
        priority=priority,
        wait_for_files=[
            str(protocol.REGISTRATION_PATH),
            *registration_data_files(),
            *(str(path) for path in protocol.source_required_paths(seed)),
        ],
        stage_input_paths=[
            *registration_data_input_dirs(), str(protocol.source_bundle(seed))],
    )


def direct_signature(recipe: str, seed: int) -> str:
    recipe = protocol.require_direct_recipe(recipe)
    return f"{SIGNATURE_PREFIX}/train/{recipe}/seed-{protocol.require_training_seed(seed)}"


def direct_spec(recipe: str, seed: int, priority: str) -> dict:
    recipe = protocol.require_direct_recipe(recipe)
    seed = protocol.require_training_seed(seed)
    return _common_gpu_spec(
        description=f"V33 {recipe} seed {seed}",
        command=base._gpu_command(
            "jax_experiments.analysis."
            "run_regime_polarity_canonical_training_direct_v33",
            ["--recipe", recipe, "--seed", str(seed), "--resume"],
            protocol.FINAL_NEXT_ITERATION,
            0.20,
        ),
        signature=direct_signature(recipe, seed),
        run=protocol.direct_run_dir(recipe, seed),
        bundle=protocol.direct_bundle_dir(recipe, seed),
        priority=priority,
        wait_for_files=[
            str(protocol.REGISTRATION_PATH), *registration_data_files()],
        stage_input_paths=registration_data_input_dirs(),
    )


def audit_signature(seed: int) -> str:
    return f"{SIGNATURE_PREFIX}/audit/seed-{protocol.require_training_seed(seed)}"


def audit_spec(seed: int, priority: str) -> dict:
    seed = protocol.require_training_seed(seed)
    return {
        "description": f"V33 canonical training audit seed {seed}",
        "cmd": base._cpu_command(
            "jax_experiments.analysis."
            "run_regime_polarity_canonical_training_audit_v33",
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
            str(path) for path in protocol.audit_required_paths(seed)],
        "stage_input_paths": [
            *registration_data_input_dirs(),
            *(str(protocol.recipe_bundle_dir(recipe, seed))
              for recipe in protocol.RECIPES),
            str(protocol.v5_model.MODEL_ROOT),
        ],
        "stage_excludes": ["paper/"],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Frozen deterministic policy evaluation; no parameter updates."),
    }


def analysis_signature() -> str:
    return f"{SIGNATURE_PREFIX}/analysis"


def analysis_spec(priority: str) -> dict:
    return {
        "description": "Aggregate V33 canonical training ablation",
        "cmd": base._cpu_command(
            "jax_experiments.analysis."
            "analyze_regime_polarity_canonical_training_v33",
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
            *(str(protocol.audit_manifest(seed))
              for seed in protocol.TRAINING_SEEDS),
        ],
        "stage_input_paths": [
            *registration_data_input_dirs(), str(protocol.AUDIT_ROOT)],
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
            source_signature(seed), source_spec(seed, priority),
            protocol.source_manifest(seed)))
        rows.append((
            warmstart_signature(seed), warmstart_spec(seed, priority),
            protocol.bundle_manifest(
                protocol.WARMSTART_RECIPE, seed, protocol.REFERENCE_MODE)))
        for recipe in protocol.DIRECT_RECIPES:
            rows.append((
                direct_signature(recipe, seed),
                direct_spec(recipe, seed, priority),
                protocol.direct_manifest(recipe, seed)))
        rows.append((
            audit_signature(seed), audit_spec(seed, priority),
            protocol.audit_manifest(seed)))
    rows.append((
        analysis_signature(), analysis_spec(priority), protocol.analysis_json()))
    return rows


def _dispatch(task_ids: list[str]) -> None:
    if not task_ids:
        return
    command = [
        sys.executable,
        str(scheduler_common.SCHEDULER),
        "dispatch",
        "--bulk-window",
        "--intent-label", DISPATCH_INTENT,
        "--intent-ttl", "900",
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
    skipped_complete = 0
    skipped_active = 0
    for signature, spec, output in candidates(args.priority):
        matching = [
            task for task in known
            if str(task.get("signature") or "") == signature
        ]
        active = [
            task for task in matching
            if str(task.get("status")) in ACTIVE_STATUSES
        ]
        if output.is_file():
            skipped_complete += 1
        elif active:
            skipped_active += 1
        else:
            specs.append(spec)
    gpu = sum(int(spec["vram"]) > 0 for spec in specs)
    cpu = len(specs) - gpu
    print(
        f"V33 tasks: submit={len(specs)} gpu={gpu} cpu={cpu} "
        f"complete={skipped_complete} active={skipped_active}",
        flush=True,
    )
    if args.dry_run or not specs:
        return
    base.SUBMIT_INTENT = SUBMIT_INTENT
    task_ids = base._submit(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        _dispatch(task_ids)


if __name__ == "__main__":
    main()
