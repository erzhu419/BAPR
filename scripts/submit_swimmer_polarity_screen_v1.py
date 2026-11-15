#!/usr/bin/env python3
"""Submit the preregistered Swimmer oracle-headroom screen via scheduler."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import submit_bapr_v3_budget_matched_fork as common
from jax_experiments.analysis import swimmer_polarity_screen_v1 as protocol


PREFIX = "BAPR/swimmer-polarity/screen-v1"
PREREG = ROOT / "reports/swimmer_polarity_screen_v1_preregistration_2026-09-26.md"
GPU_NODES = ["jtl110gpu", "jtl110gpu2", "jtl311linux", "node007"]
CPU_NODES = [f"node00{index}" for index in range(1, 8)]


def command(mode: str, args: list[str], gpu: bool) -> str:
    prefix = (
        "JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.28 "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
        if gpu else
        "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' "
        "XLA_FLAGS='--xla_cpu_multi_thread_eigen=false "
        "intra_op_parallelism_threads=8' "
    )
    return (
        ("SCHEDULEURM_ETA_TOTAL_UNITS=2100 " if gpu else "")
        + prefix
        + "OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 "
        + f"PYTHONPATH={shlex.quote(str(ROOT))} "
        + f"{shlex.quote(str(common.JAX_PYTHON))} -u -m "
        + "jax_experiments.analysis.swimmer_polarity_screen_v1 "
        + shlex.join([mode, *args]) + " && echo DONE"
    )


def candidates(priority: str):
    rows = []
    for seed in protocol.SEEDS:
        for arm in protocol.ARMS:
            bundle = protocol.bundle_dir(arm, seed)
            run = protocol.run_dir(arm, seed)
            signature = f"{PREFIX}/train/{arm}/seed-{seed}"
            rows.append((signature, {
                "description": f"Swimmer polarity {arm} seed {seed}",
                "cmd": command("train", ["--arm", arm, "--seed", str(seed), "--resume"], True),
                "cwd": str(ROOT), "signature": signature, "project": "BAPR",
                "vram": 2000, "ram_mb": 4096, "cpu": 2,
                "priority": priority, "allowed_nodes": GPU_NODES,
                "ckpt_dir": str(run / "checkpoints"),
                "ckpt_glob": "train_state.pkl", "resume_flag": "",
                "resume_managed_by_cmd": True,
                "result_dir": str(bundle), "local_result_dir": str(bundle),
                "wait_for_files": [str(PREREG)],
                "stage_excludes": ["paper/"],
                "reroute_on_node_down": True,
                "allow_initial_resume_scan_error": True,
            }, bundle / "bundle.json"))
        audit = protocol.AUDIT_ROOT / f"seed_{seed}" / "audit.json"
        signature = f"{PREFIX}/audit/seed-{seed}"
        bundles = [protocol.bundle_dir(arm, seed) for arm in protocol.ARMS]
        rows.append((signature, {
            "description": f"Swimmer polarity oracle screen audit seed {seed}",
            "cmd": command("audit", ["--seed", str(seed), "--resume"], False),
            "cwd": str(ROOT), "signature": signature, "project": "BAPR",
            "vram": 0, "ram_mb": 8192, "cpu": 8,
            "priority": priority, "allowed_nodes": CPU_NODES,
            "result_dir": str(audit.parent), "local_result_dir": str(audit.parent),
            "wait_for_files": [str(bundle / "bundle.json") for bundle in bundles],
            "stage_input_paths": [str(bundle) for bundle in bundles],
            "stage_excludes": ["paper/"],
            "reroute_on_node_down": True,
            "allow_initial_resume_scan_error": True,
            "allow_cpu_training": True,
            "cpu_training_justification": "Frozen policy evaluation; no updates.",
        }, audit))
    signature = f"{PREFIX}/analysis"
    rows.append((signature, {
        "description": "Aggregate independent Swimmer oracle-headroom screen",
        "cmd": command("analyze", [], False),
        "cwd": str(ROOT), "signature": signature, "project": "BAPR",
        "vram": 0, "ram_mb": 2048, "cpu": 2,
        "priority": priority, "allowed_nodes": CPU_NODES,
        "result_dir": str(protocol.ANALYSIS_PATH.parent),
        "local_result_dir": str(protocol.ANALYSIS_PATH.parent),
        "wait_for_files": [
            str(protocol.AUDIT_ROOT / f"seed_{seed}" / "audit.json")
            for seed in protocol.SEEDS
        ],
        "stage_input_paths": [str(protocol.AUDIT_ROOT)],
        "stage_excludes": ["paper/"],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": "JSON aggregation only.",
    }, protocol.ANALYSIS_PATH))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--priority", choices=("low", "normal", "high"), default="normal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()
    known = common.scheduler_tasks()
    selected = [
        spec for signature, spec, output in candidates(args.priority)
        if not output.is_file() and not any(
            row.get("signature") == signature
            and row.get("status") in common.ACTIVE_STATUSES
            for row in known
        )
    ]
    if args.dry_run:
        print(f"submit={len(selected)} gpu={sum(spec['vram'] > 0 for spec in selected)}")
        for spec in selected:
            print(spec["signature"], spec["allowed_nodes"])
        return
    if not selected:
        print("No new Swimmer screen tasks")
        return
    response = subprocess.run(
        [sys.executable, str(common.SCHEDULER), "submit-jsonl", "--stdin",
         "--trusted", "--json", "--intent-label", "bapr-swimmer-screen-v1-submit",
         "--intent-ttl", "900"],
        input="".join(json.dumps(spec) + "\n" for spec in selected),
        text=True, capture_output=True, env=common.scheduler_env(), check=True,
    )
    payload = json.loads(response.stdout)
    ids = [str(row["id"]) for row in payload["submitted"]]
    if len(ids) != len(selected):
        raise RuntimeError(f"incomplete scheduler submission: {ids}")
    print("submitted:", ",".join(ids))
    if args.dispatch:
        subprocess.run(
            [sys.executable, str(common.SCHEDULER), "dispatch", "--bulk-window",
             "--intent-label", "bapr-swimmer-screen-v1-dispatch", "--intent-ttl",
             "900", *[item for task_id in ids for item in ("--task-id", task_id)]],
            check=True, env=common.scheduler_env(),
        )


if __name__ == "__main__":
    main()
