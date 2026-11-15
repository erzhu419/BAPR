#!/usr/bin/env python3
"""Submit grouped fixed-context audits for the causal BAPR-v3 v2 pairs.

The unit of scheduling is one family/environment/event-seed group.  Each of
the 20 jobs runs all six controller evaluations sequentially, so the 120 CSV
outputs never compare evaluator jobs from different runtimes or GPUs.
"""
from __future__ import annotations

import argparse
import json
import platform
import shlex
import subprocess
import sys
from pathlib import Path


_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if str(_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORT_ROOT))

import submit_bapr_v85_staged_capacity as common
from jax_experiments.analysis import (
    analyze_bapr_v3_budget_matched_fork_audit as audit,
)


ROOT = common.ROOT
SCHEDULER = common.SCHEDULER
JAX_PYTHON = common.JAX_PYTHON
RUNNER_MODULE = (
    "jax_experiments.analysis.run_bapr_v3_budget_matched_fork_audit_group")
RUNNER = (
    ROOT / "jax_experiments" / "analysis"
    / "run_bapr_v3_budget_matched_fork_audit_group.py")
PAIR_ROOT = audit.DEFAULT_PAIR_ROOT
OUT_ROOT = audit.DEFAULT_RESULTS_ROOT
QUEUE = Path.home() / ".claude" / "scheduler" / "queue.json"
MAX_TASKS_PER_INVOCATION = 20
RECOVERY_SIGNATURE_PREFIX = "BAPR/v3-budget-match-fork-recovery/v4"


def signature(family: str, env: str, event_seed: int) -> str:
    return (
        f"BAPR/v3-budget-match-fork-audit/v2/{family}/{env}/"
        f"event-seed-{event_seed}")


def producer_signature(family: str, env: str) -> str:
    return f"BAPR/v3-budget-match-fork/v2/{family}/{env}/seed0"


def recovery_signature(family: str, env: str) -> str:
    return f"{RECOVERY_SIGNATURE_PREFIX}/{family}/{env}/seed0"


def pair_dir(family: str, env: str) -> Path:
    return audit.pair_directory(PAIR_ROOT, family, env, 0)


def output_dir(family: str, env: str, event_seed: int) -> Path:
    return audit.group_directory(OUT_ROOT, family, env, event_seed)


def producer_ready_local(family: str, env: str) -> tuple[bool, str]:
    directory = pair_dir(family, env)
    try:
        audit.validate_pair_provenance(directory, family, env, 0)
    except (OSError, ValueError) as exc:
        return False, str(exc)
    return True, "causal pair complete"


def producer_scheduler_node(family: str, env: str) -> str | None:
    manifest_path = pair_dir(family, env) / "provenance" / "pair_manifest.json"
    try:
        manifest = audit.read_json(manifest_path)
        host = str(audit.require_dict(
            manifest.get("runtime"), "pair runtime").get("host") or "")
    except ValueError:
        return None
    normalized = host.strip().lower()
    known = {
        "jtl110gpu": "jtl110gpu",
        "jtl110gpu2": "jtl110gpu2",
        "jtl311linux": "jtl311linux",
        "node007": "node007",
    }
    if normalized in known:
        return known[normalized]
    if normalized == platform.node().strip().lower():
        return "local"
    if normalized not in known:
        raise ValueError(
            f"producer host {host!r} has no fail-closed scheduler-node mapping")
    raise AssertionError("unreachable producer host mapping")


def resolve_producer(
    tasks: list[dict], family: str, env: str,
) -> tuple[bool, str, str | None]:
    """Locate a complete producer without copying its checkpoints locally.

    A locally available pair receives the full provenance validation here. A
    remote-only pair is located by its unique successful scheduler producer;
    the grouped evaluator remains fail-closed and repeats the full pair
    validation before and after evaluating all six controllers.
    """
    local_ready, local_reason = producer_ready_local(family, env)
    if local_ready:
        try:
            node = producer_scheduler_node(family, env)
        except ValueError as exc:
            return False, str(exc), None
        if node is None:
            return False, "validated local pair has no scheduler node", None
        return True, local_reason, node

    expected = producer_signature(family, env)
    producer_tasks = [
        task for task in tasks
        if str(task.get("signature") or "") == expected
    ]
    completed_producers = [
        task for task in producer_tasks if str(task.get("status")) == "done"
    ]
    if len(completed_producers) > 1:
        states = sorted(str(task.get("status")) for task in producer_tasks)
        return (
            False,
            f"local pair unavailable ({local_reason}); found multiple completed "
            f"producer tasks states={states}",
            None,
        )
    evidence = completed_producers
    evidence_kind = "producer"
    if not evidence:
        recovered = recovery_signature(family, env)
        recovery_tasks = [
            task for task in tasks
            if str(task.get("signature") or "") == recovered
        ]
        completed_recoveries = [
            task for task in recovery_tasks
            if str(task.get("status")) == "done"
        ]
        if len(completed_recoveries) != 1:
            producer_states = sorted(
                str(task.get("status")) for task in producer_tasks)
            recovery_states = sorted(
                str(task.get("status")) for task in recovery_tasks)
            return (
                False,
                f"local pair unavailable ({local_reason}); expected one completed "
                f"producer or v4 recovery, producer_states={producer_states}, "
                f"recovery_states={recovery_states}",
                None,
            )
        evidence = completed_recoveries
        evidence_kind = "source-archive recovery"

    node = str(evidence[0].get("node") or "").strip()
    if node not in {
        "local", "jtl110gpu", "jtl110gpu2", "jtl311linux", "node007",
    }:
        return False, f"completed {evidence_kind} has unsupported node {node!r}", None
    return (
        True,
        f"remote pair complete via {evidence_kind} on {node}; "
        "grouped runner will revalidate it",
        node,
    )


def output_complete(family: str, env: str, event_seed: int) -> bool:
    try:
        audit.validate_group_bundle(
            output_dir(family, env, event_seed), pair_dir(family, env),
            family, env, 0, event_seed)
    except (OSError, ValueError):
        return False
    return True


def task_spec(
    family: str,
    env: str,
    event_seed: int,
    producer_node: str,
    args: argparse.Namespace,
) -> dict:
    source_dir = pair_dir(family, env)
    result_dir = output_dir(family, env, event_seed)
    values = [
        "--pair-dir", str(source_dir),
        "--results-root", str(OUT_ROOT),
        "--family", family,
        "--env", env,
        "--training-seed", "0",
        "--event-seed", str(event_seed),
        "--resume",
    ]
    command = (
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.20 "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 JAX_NUM_THREADS=1 "
        "TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 "
        f"{shlex.quote(str(JAX_PYTHON))} -u -m {RUNNER_MODULE} "
        f"{shlex.join(values)}"
    )
    spec = {
        "description": (
            f"BAPR-v3 causal fork audit {family} {env} "
            f"event seed {event_seed}; six sequential controllers"),
        "cmd": command,
        "cwd": str(ROOT),
        "signature": signature(family, env, event_seed),
        "project": "BAPR",
        # Stay below the scheduler's 1/3+grace admission line on a free 12GB
        # GPU. The estimated post-launch allocation then admits one compiler
        # per GPU and blocks another until the live-memory probe replaces it.
        "vram": 4200,
        "ram_mb": 4096,
        "cpu": 2,
        "priority": args.priority,
        # The complete marker lives at pair root.  Staging therefore copies
        # both final branch checkpoints and the authoritative provenance in
        # one operation before the grouped evaluator starts.
        "ckpt_dir": str(source_dir),
        "ckpt_glob": "pair_checkpoint_complete.pkl",
        "resume_flag": "",
        "result_dir": str(result_dir),
        "local_result_dir": str(result_dir),
        "stage_excludes": [
            "jax_experiments/eval_bundles*/",
            "jax_experiments/results*/",
            "paper/",
        ],
        "allow_remote_large_data": True,
        "allow_initial_resume_scan_error": False,
        "reroute_on_node_down": False,
        "require_node": producer_node,
    }
    return spec


def scheduler_tasks_strict() -> list[dict]:
    command = [
        sys.executable, str(SCHEDULER), "status", "--all", "--json",
        "--brief", "--readonly",
    ]
    try:
        result = subprocess.run(
            command, text=True, capture_output=True, check=True,
            env=common.scheduler_env(), timeout=30)
        payload = json.loads(result.stdout)
        tasks = payload.get("tasks", payload) if isinstance(payload, dict) else payload
        if not isinstance(tasks, list):
            raise RuntimeError(
                f"readonly scheduler status has non-list tasks: "
                f"{type(tasks).__name__}")
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


def submit_jsonl(specs: list[dict]) -> list[str]:
    signatures = [str(spec.get("signature") or "") for spec in specs]
    if (any(not value for value in signatures)
            or len(set(signatures)) != len(signatures)):
        raise RuntimeError(
            f"submission specs have missing/duplicate signatures: {signatures!r}")
    payload = "".join(json.dumps(spec) + "\n" for spec in specs)
    command = [
        sys.executable, str(SCHEDULER), "submit-jsonl",
        "--stdin", "--trusted", "--json",
        "--intent-label", "bapr-v3-fork-audit-submit",
        "--intent-ttl", "900",
    ]
    result = subprocess.run(
        command, input=payload, text=True, capture_output=True,
        env=common.scheduler_env())
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
            f"requested={len(specs)}, returned_ids={task_ids!r}; "
            "refusing dispatch")
    return task_ids


def dispatch(task_ids: list[str]) -> None:
    if not task_ids:
        return
    command = [
        sys.executable, str(SCHEDULER), "dispatch", "--bulk-window",
        "--intent-label", "bapr-v3-fork-audit-dispatch",
        "--intent-ttl", "900",
    ]
    for task_id in task_ids:
        command += ["--task-id", task_id]
    subprocess.run(command, check=True, env=common.scheduler_env())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", action="append", choices=audit.FAMILIES)
    parser.add_argument("--env", action="append", choices=audit.FULL_ENVS)
    parser.add_argument("--event-seed", action="append", type=int)
    parser.add_argument("--task-cap", type=int, default=MAX_TASKS_PER_INVOCATION)
    parser.add_argument(
        "--retry-incomplete", action="store_true",
        help="Resubmit a terminal group only when strict local validation fails.")
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="high")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.task_cap <= MAX_TASKS_PER_INVOCATION:
        raise SystemExit(
            f"task-cap must be in [1,{MAX_TASKS_PER_INVOCATION}]")
    for path in (SCHEDULER, JAX_PYTHON, RUNNER):
        if not Path(path).exists():
            raise SystemExit(f"missing required path: {path}")

    families = tuple(dict.fromkeys(args.family or audit.FAMILIES))
    envs = tuple(dict.fromkeys(args.env or audit.FULL_ENVS))
    event_seeds = tuple(args.event_seed or audit.DEFAULT_EVENT_SEEDS)
    if (len(event_seeds) != len(set(event_seeds))
            or any(seed not in audit.DEFAULT_EVENT_SEEDS for seed in event_seeds)):
        raise SystemExit(
            f"event seeds must be a distinct subset of the preregistered "
            f"values {audit.DEFAULT_EVENT_SEEDS}")

    try:
        tasks = scheduler_tasks_strict()
    except RuntimeError as exc:
        raise SystemExit(f"REFUSED: {exc}") from exc
    readiness = {
        (family, env): resolve_producer(tasks, family, env)
        for family in families for env in envs
    }
    not_ready = [
        (family, env, reason)
        for (family, env), (ready, reason, _node) in readiness.items()
        if not ready
    ]
    for family, env, reason in not_ready:
        print(
            f"  producer not ready: {family}/{env}: {reason}", flush=True)
    if not_ready and not args.dry_run:
        raise SystemExit(
            "all selected v2 shared-fork producer pairs must pass strict "
            "provenance validation before audit submission")

    active_statuses = {"queued", "launching", "running"}
    statuses_by_signature: dict[str, set[str]] = {}
    for task in tasks:
        if task.get("signature"):
            statuses_by_signature.setdefault(
                str(task["signature"]), set()).add(str(task.get("status")))
    active_counts = {}
    for task in tasks:
        sig = str(task.get("signature") or "")
        if sig and str(task.get("status")) in active_statuses:
            active_counts[sig] = active_counts.get(sig, 0) + 1
    duplicate_active = {
        sig: count for sig, count in active_counts.items() if count > 1
        and sig.startswith("BAPR/v3-budget-match-fork-audit/v2/")
    }
    if duplicate_active:
        raise SystemExit(
            f"REFUSED: duplicate active audit signatures: {duplicate_active}")

    specs = []
    skipped = []
    for family in families:
        for env in envs:
            for event_seed in event_seeds:
                sig = signature(family, env, event_seed)
                statuses = statuses_by_signature.get(sig, set())
                if output_complete(family, env, event_seed):
                    skipped.append((sig, "complete and provenance-valid"))
                elif statuses & active_statuses:
                    skipped.append((sig, "active"))
                elif "done" in statuses and not args.retry_incomplete:
                    skipped.append((sig, "done; awaiting valid result sync"))
                else:
                    producer_node = readiness[(family, env)][2]
                    if not producer_node:
                        continue
                    specs.append(task_spec(
                        family, env, event_seed, producer_node, args))

    expected_groups = len(families) * len(envs) * len(event_seeds)
    print(
        f"BAPR-v3 grouped causal audit: groups={expected_groups} "
        f"outputs={expected_groups * 6} submit={len(specs)} "
        f"skip={len(skipped)}",
        flush=True)
    for spec in specs:
        print(f"  {spec['signature']}", flush=True)
    for sig, reason in skipped:
        print(f"  skip {reason}: {sig}", flush=True)
    if len(specs) > args.task_cap:
        raise SystemExit(
            f"selected {len(specs)} groups exceeds task-cap={args.task_cap}; "
            "use family/env filters")
    if args.dry_run or not specs:
        return

    task_ids = submit_jsonl(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        dispatch(task_ids)


if __name__ == "__main__":
    main()
