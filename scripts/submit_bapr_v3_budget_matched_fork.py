#!/usr/bin/env python3
"""Submit four fail-closed BAPR-v3 shared-checkpoint pair tasks.

There is one scheduler task per (family, environment), never one task per
branch.  The paired runner owns all three sequential train subprocesses on the
single scheduler-assigned runtime/GPU.  This script only submits, optionally
dispatches, and never uses Slurm directly or adopts/manual-copies checkpoints.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if str(_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORT_ROOT))

import run_bapr_v3_budget_matched_fork as protocol


ROOT = protocol.ROOT
SAVE_ROOT = protocol.SAVE_ROOT
FAMILIES = protocol.FAMILIES
ENVS = protocol.ENVS
SCHEDULER = Path("/home/erzhu419/mine_code/scheduleurm/skill/scheduler.py")
JAX_PYTHON = Path("/home/erzhu419/.conda/envs/resac-jax/bin/python")
QUEUE = Path.home() / ".claude" / "scheduler" / "queue.json"
MAX_TASKS_PER_INVOCATION = 4
SIGNATURE_PREFIX = "BAPR/v3-budget-match-fork/v2"
ACTIVE_STATUSES = {"queued", "launching", "running"}


def scheduler_env() -> dict[str, str]:
    env = os.environ.copy()
    scheduler_path = str(SCHEDULER.parent)
    env["PYTHONPATH"] = (
        scheduler_path if not env.get("PYTHONPATH")
        else f"{scheduler_path}{os.pathsep}{env['PYTHONPATH']}")
    return env


def scheduler_tasks() -> list[dict]:
    command = [
        sys.executable, str(SCHEDULER), "status", "--all", "--json",
        "--brief", "--readonly",
    ]
    try:
        result = subprocess.run(
            command, text=True, capture_output=True, check=True,
            env=scheduler_env(), timeout=30)
        payload = json.loads(result.stdout)
        tasks = payload.get("tasks", payload) if isinstance(payload, dict) else payload
        if not isinstance(tasks, list):
            raise RuntimeError(
                f"readonly scheduler status has non-list tasks: {type(tasks).__name__}")
        return tasks
    except Exception as status_exc:
        if not QUEUE.exists():
            raise RuntimeError(
                "cannot prove scheduler duplicate state: readonly status "
                f"failed ({status_exc}) and queue file is missing") from status_exc
        try:
            payload = json.loads(QUEUE.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or "tasks" not in payload:
                raise RuntimeError("queue fallback has no tasks field")
            tasks = payload["tasks"]
            if not isinstance(tasks, list):
                raise RuntimeError(
                    f"queue fallback has non-list tasks: {type(tasks).__name__}")
            return tasks
        except Exception as queue_exc:
            raise RuntimeError(
                "cannot prove scheduler duplicate state: readonly status and "
                f"queue fallback both failed (status={status_exc}; "
                f"queue={queue_exc})") from queue_exc


def signature(family: str, env: str, seed: int = 0) -> str:
    return f"{SIGNATURE_PREFIX}/{family}/{env}/seed{seed}"


def output_complete(pair: Path) -> tuple[bool, str]:
    try:
        validated = protocol.validate_complete_artifacts(pair)
        manifest = validated["manifest"]
        runtime = manifest["runtime"]
        runtime_without_hash = dict(runtime)
        runtime_sha = runtime_without_hash.pop("sha256")
        if protocol.canonical_sha256(runtime_without_hash) != runtime_sha:
            return False, "runtime hash mismatch"
        source = manifest["source"]
        if protocol.canonical_sha256(source["files"]) != source["sha256"]:
            return False, "source hash mismatch"
        archive = pair / source["archive"]["path"]
        protocol.validate_source_archive(
            archive,
            {"files": source["files"]},
            source["archive"]["sha256"])
        base_sha = manifest["shared_base"]["snapshot_sha256"]
        for branch in ("robust_long", "oracle_direct"):
            fork = manifest["forks"][branch]
            if (fork["source_snapshot_sha256"] != base_sha
                    or fork["start_iteration"] != protocol.BASE_NEXT_ITERATION
                    or fork["start_total_steps"] != protocol.BASE_TOTAL_STEPS
                    or fork["runtime_sha256"] != runtime_sha
                    or fork["source_sha256"] != source["sha256"]):
                return False, f"invalid fork provenance: {branch}"
            boundary = manifest["resume_boundary"][branch]
            boundary_path = pair / branch / "logs" / protocol.BOUNDARY_AUDIT_NAME
            if (not boundary_path.is_file()
                    or protocol.sha256_file(boundary_path)
                    != boundary["file_sha256"]):
                return False, f"invalid boundary hash: {branch}"
            for relative, record in manifest["final"][branch]["files"].items():
                path = pair / branch / relative
                if (not path.is_file()
                        or protocol.sha256_file(path) != record["sha256"]
                        or path.stat().st_size != record["size"]):
                    return False, f"invalid final file hash: {branch}/{relative}"
        resume = manifest["resume_boundary"]
        validation = resume.get("physical_rollout_validation")
        if validation == "categorical_policy_equivalence":
            if resume.get("physical_rollout_accepted") is not True:
                return False, "categorical physical rollout proof was rejected"
        elif (not resume.get("physical_rollout_equal")
                or not resume.get("field_sha256_equal")
                or not resume.get("physical_rollout_sha256")):
            return False, "paired physical rollout proof is incomplete"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, "complete"


def runner_values(family: str, env: str) -> list[str]:
    return [
        "--family", family,
        "--env", env,
        "--seed", "0",
        "--save-root", str(SAVE_ROOT),
        "--resume",
    ]


def task_spec(family: str, env: str, args: argparse.Namespace) -> dict:
    pair = protocol.pair_dir(family, env)
    command = (
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.24 "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 JAX_NUM_THREADS=1 "
        "TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 "
        f"{shlex.quote(str(JAX_PYTHON))} -u -m "
        "jax_experiments.analysis.run_bapr_v3_budget_matched_fork "
        f"{shlex.join(runner_values(family, env))}"
    )
    return {
        "description": (
            "BAPR-v3 exact shared-checkpoint/common-restart budget pair "
            f"{family} {env} seed0"),
        "cmd": command,
        "cwd": str(ROOT),
        "signature": signature(family, env),
        "project": "BAPR",
        "vram": 3200,
        "ram_mb": 6144,
        "cpu": 2,
        "priority": args.priority,
        # The marker is written immediately and atomically after the pair owns
        # its runtime/source identity.  Scheduleurm stages the entire pair root.
        "ckpt_dir": str(pair),
        "ckpt_glob": protocol.STATE_NAME,
        "resume_flag": "",
        "result_dir": str(pair),
        "local_result_dir": str(pair),
        "stage_excludes": [
            "jax_experiments/results*/",
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "allow_remote_large_data": True,
        # A retry must never launch fresh merely because checkpoint discovery
        # failed; the runner cannot compare identities if no state was staged.
        "allow_initial_resume_scan_error": False,
        # A different node/runtime is a protocol violation; the runner also
        # independently fails closed if a retry identity changes.
        "reroute_on_node_down": False,
    }


def submit_jsonl(specs: list[dict]) -> list[str]:
    payload = "".join(json.dumps(spec) + "\n" for spec in specs)
    command = [
        sys.executable, str(SCHEDULER), "submit-jsonl",
        "--stdin", "--trusted", "--json",
        "--intent-label", "bapr-v3-budget-fork-v2-submit",
        "--intent-ttl", "900",
    ]
    result = subprocess.run(
        command, input=payload, text=True, capture_output=True,
        env=scheduler_env())
    if result.returncode != 0:
        print((result.stdout or "") + (result.stderr or ""), file=sys.stderr)
        result.check_returncode()
    response = json.loads(result.stdout)
    print(json.dumps(response, indent=2))
    submitted = response.get("submitted", [])
    if not isinstance(submitted, list):
        raise RuntimeError("scheduler response has no submitted task list")
    task_ids = [str(item.get("id", "")) for item in submitted]
    if (len(task_ids) != len(specs) or any(not task_id for task_id in task_ids)
            or len(set(task_ids)) != len(task_ids)):
        raise RuntimeError(
            "scheduler submission was not all-or-verifiably-accounted: "
            f"requested={len(specs)}, returned_ids={task_ids!r}; refusing dispatch")
    return task_ids


def dispatch(task_ids: list[str]) -> None:
    if not task_ids:
        return
    command = [
        sys.executable, str(SCHEDULER), "dispatch", "--bulk-window",
        "--intent-label", "bapr-v3-budget-fork-v2-dispatch",
        "--intent-ttl", "900",
    ]
    for task_id in task_ids:
        command += ["--task-id", task_id]
    subprocess.run(command, check=True, env=scheduler_env())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", action="append", choices=FAMILIES)
    parser.add_argument("--env", action="append", choices=ENVS)
    parser.add_argument("--task-cap", type=int, default=MAX_TASKS_PER_INVOCATION)
    parser.add_argument(
        "--retry-incomplete", action="store_true",
        help="Allow a new task only for a terminal scheduler record whose "
             "local pair is not valid/complete.")
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="high")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args(argv)

    if not 1 <= args.task_cap <= MAX_TASKS_PER_INVOCATION:
        raise SystemExit(
            f"task-cap must be in [1,{MAX_TASKS_PER_INVOCATION}]")
    for path in (SCHEDULER, JAX_PYTHON):
        if not path.exists():
            raise SystemExit(f"missing required path: {path}")
    families = list(dict.fromkeys(args.family or FAMILIES))
    envs = list(dict.fromkeys(args.env or ENVS))
    expected = len(families) * len(envs)
    if expected > args.task_cap:
        raise SystemExit(
            f"selection expands to {expected} pair tasks; cap is {args.task_cap}")

    by_signature: dict[str, list[dict]] = {}
    try:
        known_tasks = scheduler_tasks()
    except RuntimeError as exc:
        raise SystemExit(f"REFUSED: {exc}") from exc
    for task in known_tasks:
        if task.get("signature"):
            by_signature.setdefault(str(task["signature"]), []).append(task)

    specs = []
    skipped = []
    for family in families:
        for env in envs:
            sig = signature(family, env)
            pair = protocol.pair_dir(family, env)
            complete, reason = output_complete(pair)
            existing = by_signature.get(sig, [])
            active = [
                task for task in existing
                if str(task.get("status")) in ACTIVE_STATUSES]
            terminal = [task for task in existing if task not in active]
            if complete:
                skipped.append((sig, "complete"))
            elif active:
                ids = ",".join(str(task.get("id", "?")) for task in active)
                skipped.append((sig, f"active:{ids}"))
            elif terminal and not args.retry_incomplete:
                ids = ",".join(str(task.get("id", "?")) for task in terminal)
                skipped.append((sig, f"terminal-incomplete:{ids}; {reason}"))
            else:
                specs.append(task_spec(family, env, args))

    print(
        f"BAPR-v3 fork-v2 pairs: selected={expected} submit={len(specs)} "
        f"skip={len(skipped)}", flush=True)
    for spec in specs:
        print(f"  submit {spec['signature']}", flush=True)
        if args.dry_run:
            print(json.dumps(spec, indent=2, sort_keys=True), flush=True)
    for sig, reason in skipped:
        print(f"  skip {reason}: {sig}", flush=True)
    if args.dry_run or not specs:
        return

    task_ids = submit_jsonl(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        dispatch(task_ids)


if __name__ == "__main__":
    main()
