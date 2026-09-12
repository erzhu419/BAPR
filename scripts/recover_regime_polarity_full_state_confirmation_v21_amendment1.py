#!/usr/bin/env python3
"""Submit export-only recovery and corrected audits for V21 amendment 1."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import submit_regime_polarity_full_state_confirmation_v21 as original

if (
    __name__ == "__main__"
    and Path(sys.executable).resolve()
    != original.scheduler_common.JAX_PYTHON.resolve()
):
    os.execv(
        str(original.scheduler_common.JAX_PYTHON),
        [str(original.scheduler_common.JAX_PYTHON),
         str(Path(__file__).resolve()), *sys.argv[1:]],
    )


protocol = original.protocol
AMENDMENT = protocol.REGISTRATION_ROOT / "amendment1.json"
EXPORT_MODULE = (
    "jax_experiments.analysis."
    "run_regime_polarity_full_state_confirmation_baseline_export_"
    "v21_amendment1"
)
AUDIT_MODULE = (
    "jax_experiments.analysis."
    "run_regime_polarity_full_state_confirmation_audit_v21_amendment1"
)
RECOVERY_SUFFIX = "amendment1"


def scheduler_tasks_with_resume_locations() -> list[dict]:
    tasks = {
        str(task.get("id")): task
        for task in original.scheduler_common.scheduler_tasks()
    }
    queue_payload = json.loads(
        original.scheduler_common.QUEUE.read_text(encoding="utf-8"))
    for task in queue_payload["tasks"]:
        tasks[str(task.get("id"))] = task
    return list(tasks.values())


def export_signature(kind: str, seed: int, slot: int | None) -> str:
    return f"{original.baseline_signature(kind, seed, slot)}/export-{RECOVERY_SUFFIX}"


def export_spec(
    kind: str, seed: int, slot: int | None, priority: str,
    checkpoint_node: str,
) -> dict:
    if checkpoint_node not in original.GPU_NODES:
        raise ValueError(f"invalid checkpoint node {checkpoint_node!r}")
    spec = original.baseline_spec(kind, seed, slot, priority)
    values = ["--kind", kind, "--seed", str(seed)]
    if slot is not None:
        values += ["--slot", str(slot)]
    values.append("--resume")
    fraction = 0.20 if kind == "sac_replica" else 0.30
    spec.update({
        "description": (
            f"V21 export amendment 1 {kind} seed {seed}"
            + (f" slot {slot}" if slot is not None else "")
        ),
        "cmd": original._gpu_command(EXPORT_MODULE, values, 1, fraction),
        "signature": export_signature(kind, seed, slot),
        "allowed_nodes": [checkpoint_node],
        "preferred_node": checkpoint_node,
        "require_node": checkpoint_node,
        "skip_resume_scan": True,
    })
    for key in (
        "ckpt_dir", "ckpt_glob", "resume_flag", "resume_managed_by_cmd",
        "allow_initial_resume_scan_error",
    ):
        spec.pop(key, None)
    spec["wait_for_files"] = [*spec["wait_for_files"], str(AMENDMENT)]
    spec["stage_input_paths"] = [
        *spec["stage_input_paths"], str(AMENDMENT.parent),
    ]
    return spec


def audit_signature(seed: int) -> str:
    return f"{original.audit_signature(seed)}/{RECOVERY_SUFFIX}"


def audit_spec(seed: int, priority: str) -> dict:
    spec = original.audit_spec(seed, priority)
    spec.update({
        "description": f"V21 audit amendment 1 seed {seed}",
        "cmd": original._cpu_command(
            AUDIT_MODULE, ["--seed", str(seed), "--resume"], threads=16),
        "signature": audit_signature(seed),
    })
    spec["wait_for_files"] = [*spec["wait_for_files"], str(AMENDMENT)]
    spec["stage_input_paths"] = [
        *spec["stage_input_paths"], str(AMENDMENT.parent),
    ]
    return spec


def _checkpoint_node(
    tasks: list[dict], kind: str, seed: int, slot: int | None,
) -> str:
    signature = original.baseline_signature(kind, seed, slot)
    locations = [
        location
        for task in tasks
        if (
            str(task.get("signature") or "") == signature
            or str(task.get("signature") or "").startswith(
                signature + "/export-")
        )
        for location in (task.get("resume_locations") or [])
        if location.get("node") in original.GPU_NODES
    ]
    if locations:
        return str(max(
            locations,
            key=lambda location: (
                float(location.get("mtime") or 0.0),
                int(location.get("size") or 0),
            ),
        )["node"])

    completed = []
    for task in tasks:
        if (
            str(task.get("signature") or "") != signature
            and not str(task.get("signature") or "").startswith(
                signature + "/export-")
        ) or task.get("node") not in original.GPU_NODES:
            continue
        progress = str(task.get("last_progress_line") or "")
        if "iter=1400" in progress or "Training complete" in progress:
            completed.append(task)
    if completed:
        return str(max(
            completed,
            key=lambda task: float(task.get("finished_at") or 0.0),
        )["node"])
    raise RuntimeError(f"no completed checkpoint evidence for {signature}")


def candidates(priority: str, known: list[dict]):
    rows = []
    for seed in protocol.TRAINING_SEEDS:
        for slot in protocol.SAC_REPLICA_SLOTS:
            rows.append((
                export_signature("sac_replica", seed, slot),
                export_spec(
                    "sac_replica", seed, slot, priority,
                    _checkpoint_node(
                        known, "sac_replica", seed, slot)),
                protocol.bundle_manifest("sac_replica", seed, slot),
            ))
        for kind in protocol.TRAINED_METHODS:
            rows.append((
                export_signature(kind, seed, None),
                export_spec(
                    kind, seed, None, priority,
                    _checkpoint_node(known, kind, seed, None)),
                protocol.bundle_manifest(kind, seed),
            ))
        rows.append((
            audit_signature(seed),
            audit_spec(seed, priority),
            protocol.audit_manifest(seed),
        ))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="high")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

    if not AMENDMENT.is_file():
        raise FileNotFoundError(AMENDMENT)
    protocol.validate_registration()
    known = scheduler_tasks_with_resume_locations()
    specs = []
    for signature, spec, output in candidates(args.priority, known):
        active = [
            task for task in known
            if str(task.get("signature") or "") == signature
            and str(task.get("status")) in original.ACTIVE_STATUSES
        ]
        if output.is_file():
            print(f"skip complete-output: {signature}")
        elif active:
            print("skip active: " + ",".join(str(task["id"]) for task in active))
        else:
            specs.append(spec)
    if args.dry_run:
        print(f"V21 amendment 1 dry-run task count: {len(specs)}")
        return
    if not specs:
        print("No V21 amendment 1 tasks to submit")
        return
    task_ids = original._submit(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        original._dispatch(task_ids)


if __name__ == "__main__":
    main()
