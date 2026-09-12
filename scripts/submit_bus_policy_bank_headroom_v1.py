#!/usr/bin/env python3
"""Register and batch-submit the bus frozen policy-bank headroom DAG."""
from __future__ import annotations

import argparse
import json
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
from bus_experiments import policy_bank_headroom_v1 as protocol


BUS_PYTHON = Path("/home/erzhu419/anaconda3/bin/python")
SIGNATURE_PREFIX = "BAPR/bus-policy-bank/headroom-v1"
SUBMIT_INTENT = "bapr-bus-policy-bank-headroom-v1-submit"
GPU_NODES = ["jtl110gpu", "jtl110gpu2", "jtl311linux", "node007"]
CPU_NODES = ["jtl110cpu", "jtl110cpu2"]
ACTIVE_STATUSES = scheduler_common.ACTIVE_STATUSES


def _command(module: str, values: list[str], *, units: int | None = None,
             cpu_only: bool = False) -> str:
    prefixes = []
    if units is not None:
        prefixes.append(f"SCHEDULEURM_ETA_TOTAL_UNITS={int(units)}")
    if cpu_only:
        prefixes.extend([
            "CUDA_VISIBLE_DEVICES=''", "OMP_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1",
            "NUMEXPR_NUM_THREADS=1",
        ])
    else:
        prefixes.extend([
            "OMP_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2",
            "MKL_NUM_THREADS=2", "NUMEXPR_NUM_THREADS=2",
        ])
    prefixes.append(f"PYTHONPATH={shlex.quote(str(ROOT))}")
    return (
        " ".join(prefixes) + " " + shlex.quote(str(BUS_PYTHON))
        + " -u -m " + module + " " + shlex.join(values) + " && echo DONE"
    )


def registration_inputs() -> list[str]:
    # Source code is handled by the normal cwd staging pass.  Only the generated
    # immutable registration needs explicit launch-input staging.
    return [str(protocol.REGISTRATION_ROOT)]


def source_signature(seed: int) -> str:
    return f"{SIGNATURE_PREFIX}/train/robust-source/seed-{seed}"


def source_spec(seed: int, priority: str) -> dict:
    run_dir = protocol.source_run_dir(seed)
    bundle_dir = protocol.source_bundle_dir(seed)
    return {
        "description": f"Bus policy-bank robust source seed {seed}",
        "cmd": _command(
            "bus_experiments.run_policy_bank_train_v1",
            ["--role", "robust_source", "--seed", str(seed), "--resume"],
            units=protocol.SOURCE_EPISODES),
        "cwd": str(ROOT),
        "signature": source_signature(seed),
        "project": "BAPR",
        "vram_resource_family": "BAPR/bus-policy-bank-v1/torch-runtime",
        "vram": 1800,
        "ram_mb": 4096,
        "cpu": 2,
        "priority": priority,
        "allowed_nodes": GPU_NODES,
        "ckpt_dir": str(run_dir / "checkpoints"),
        "ckpt_glob": "latest.json",
        "resume_flag": "",
        "resume_managed_by_cmd": True,
        "result_dir": str(bundle_dir),
        "local_result_dir": str(bundle_dir),
        "wait_for_files": [str(protocol.REGISTRATION_PATH)],
        "stage_input_paths": registration_inputs(),
        "stage_excludes": [
            "paper/", "bus_experiments_paper_v2/",
            "bus_experiments_bapr_bus_gate/"],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
    }


def specialist_signature(seed: int, mode: str) -> str:
    return f"{SIGNATURE_PREFIX}/train/specialist/seed-{seed}/mode-{mode}"


def specialist_spec(seed: int, mode: str, priority: str) -> dict:
    run_dir = protocol.specialist_run_dir(seed, mode)
    bundle_dir = protocol.specialist_bundle_dir(seed, mode)
    return {
        "description": f"Bus policy-bank specialist seed {seed} mode {mode}",
        "cmd": _command(
            "bus_experiments.run_policy_bank_train_v1",
            ["--role", "specialist", "--seed", str(seed), "--mode", mode,
             "--resume"], units=protocol.SPECIALIST_EPISODES),
        "cwd": str(ROOT),
        "signature": specialist_signature(seed, mode),
        "project": "BAPR",
        "vram_resource_family": "BAPR/bus-policy-bank-v1/torch-runtime",
        "vram": 1800,
        "ram_mb": 4096,
        "cpu": 2,
        "priority": priority,
        "allowed_nodes": GPU_NODES,
        "ckpt_dir": str(run_dir / "checkpoints"),
        "ckpt_glob": "latest.json",
        "resume_flag": "",
        "resume_managed_by_cmd": True,
        "result_dir": str(bundle_dir),
        "local_result_dir": str(bundle_dir),
        "wait_for_files": [
            str(protocol.REGISTRATION_PATH),
            *(str(path) for path in protocol.source_required_paths(seed)),
        ],
        "stage_input_paths": [
            *registration_inputs(), str(protocol.source_bundle_dir(seed))],
        "stage_excludes": [
            "paper/", "bus_experiments_paper_v2/",
            "bus_experiments_bapr_bus_gate/"],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
    }


def audit_signature(seed: int) -> str:
    return f"{SIGNATURE_PREFIX}/audit/seed-{seed}"


def audit_spec(seed: int, priority: str) -> dict:
    required = [str(path) for path in protocol.source_required_paths(seed)]
    for mode in protocol.MODES:
        required.extend(
            str(path) for path in protocol.specialist_required_paths(seed, mode))
    return {
        "description": f"Bus policy-bank paired headroom audit seed {seed}",
        "cmd": _command(
            "bus_experiments.run_policy_bank_audit_v1",
            ["--seed", str(seed), "--workers", "16"], cpu_only=True),
        "cwd": str(ROOT),
        "signature": audit_signature(seed),
        "project": "BAPR",
        "vram": 0,
        "ram_mb": 24576,
        "cpu": 16,
        "priority": priority,
        "allowed_nodes": CPU_NODES,
        "result_dir": str(protocol.audit_dir(seed)),
        "local_result_dir": str(protocol.audit_dir(seed)),
        "wait_for_files": [str(protocol.REGISTRATION_PATH), *required],
        "stage_input_paths": [
            *registration_inputs(),
            str(protocol.source_bundle_dir(seed)),
            *(str(protocol.specialist_bundle_dir(seed, mode))
              for mode in protocol.MODES),
        ],
        "stage_excludes": ["paper/"],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": "Paired deterministic simulation only.",
    }


def analysis_signature() -> str:
    return f"{SIGNATURE_PREFIX}/analysis"


def analysis_spec(priority: str) -> dict:
    return {
        "description": "Aggregate bus policy-bank headroom decision",
        "cmd": _command(
            "bus_experiments.analyze_policy_bank_headroom_v1", [],
            cpu_only=True),
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
            str(protocol.audit_manifest(seed))
            for seed in protocol.TRAINING_SEEDS
        ],
        "stage_input_paths": [
            *registration_inputs(), str(protocol.AUDIT_ROOT)],
        "stage_excludes": ["paper/"],
        "reroute_on_node_down": True,
        "allow_initial_resume_scan_error": True,
        "allow_cpu_training": True,
        "cpu_training_justification": "JSON aggregation only.",
    }


def candidates(priority: str):
    rows = []
    for seed in protocol.TRAINING_SEEDS:
        rows.append((
            source_signature(seed), source_spec(seed, priority),
            protocol.source_manifest(seed)))
        for mode in protocol.MODES:
            rows.append((
                specialist_signature(seed, mode),
                specialist_spec(seed, mode, priority),
                protocol.specialist_manifest(seed, mode)))
        rows.append((
            audit_signature(seed), audit_spec(seed, priority),
            protocol.audit_manifest(seed)))
    rows.append((
        analysis_signature(), analysis_spec(priority), protocol.analysis_json()))
    return rows


def _submit_jsonl(specs: list[dict]) -> list[str]:
    payload = "".join(json.dumps(spec) + "\n" for spec in specs)
    command = [
        sys.executable, str(scheduler_common.SCHEDULER), "submit-jsonl",
        "--stdin", "--trusted", "--json", "--intent-label", SUBMIT_INTENT,
        "--intent-ttl", "900",
    ]
    result = subprocess.run(
        command, input=payload, text=True, capture_output=True,
        env=scheduler_common.scheduler_env())
    if result.returncode != 0:
        print((result.stdout or "") + (result.stderr or ""), file=sys.stderr)
        result.check_returncode()
    response = json.loads(result.stdout)
    submitted = response.get("submitted", [])
    task_ids = [str(item.get("id", "")) for item in submitted]
    if len(task_ids) != len(specs) or any(not task_id for task_id in task_ids):
        raise RuntimeError(
            "scheduler did not account for every bus task: "
            f"requested={len(specs)} returned={task_ids}")
    print(json.dumps(response, indent=2))
    return task_ids


def _dispatch(task_ids: list[str]) -> None:
    command = [
        sys.executable, str(scheduler_common.SCHEDULER), "dispatch",
        "--bulk-window", "--intent-label", f"{SUBMIT_INTENT}-dispatch",
        "--intent-ttl", "900",
    ]
    for task_id in task_ids:
        command.extend(["--task-id", task_id])
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

    if not BUS_PYTHON.is_file():
        raise FileNotFoundError(f"missing bus Python runtime: {BUS_PYTHON}")
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
            print("skip active: " + ",".join(
                str(task["id"]) for task in active))
        elif matches and not args.retry_incomplete:
            print("skip terminal-incomplete: " + ",".join(
                str(task["id"]) for task in matches))
        else:
            specs.append(spec)
    if args.dry_run:
        print(json.dumps(specs, indent=2, sort_keys=True))
        return
    if not specs:
        print("No bus policy-bank tasks to submit")
        return
    task_ids = _submit_jsonl(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        _dispatch(task_ids)


if __name__ == "__main__":
    main()
