#!/usr/bin/env python3
"""Continue burst-torque teachers with an inverse-dynamics mode estimator."""
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import submit_bapr_v3_burst_student as previous


ROOT = previous.ROOT
SCHEDULER = previous.SCHEDULER
JAX_PYTHON = previous.JAX_PYTHON
SAVE_ROOT = previous.SAVE_ROOT
ENVS = previous.ENVS


def _set_option(values: list[str], option: str, value: str) -> None:
    index = values.index(option)
    values[index + 1] = value


def training_values(env: str, args: argparse.Namespace) -> list[str]:
    values = previous.training_values(env, args)
    _set_option(values, "--bapr_v2_context_hidden_dim", "128")
    _set_option(values, "--bapr_v3_variance_model", "inverse_empirical")
    insert_at = values.index("--bapr_v3_reset_context_on_resume")
    values[insert_at:insert_at] = [
        "--bapr_v3_estimator_rollout_source", "robust",
        "--min_resume_iteration", "2000",
    ]
    return values


def task_spec(env: str, args: argparse.Namespace) -> dict:
    name = previous.run_name(env)
    run_dir = SAVE_ROOT / name
    command = (
        f"PYTHONPATH={shlex.quote(str(ROOT))} "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.20 "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 "
        f"{shlex.quote(str(JAX_PYTHON))} -u -m jax_experiments.train "
        f"{shlex.join(training_values(env, args))}"
    )
    return {
        "description": f"BAPR-v3 burst inverse-empirical student {name}",
        "cmd": command,
        "cwd": str(ROOT),
        "signature": f"BAPR/v3-burst-student/inverse-empirical/v1/{env}/seed0",
        "project": "BAPR",
        "vram": 2600,
        "ram_mb": 4096,
        "cpu": 2,
        "priority": args.priority,
        "ckpt_dir": str(run_dir / "checkpoints"),
        "result_dir": str(run_dir / "logs"),
        "local_result_dir": str(run_dir / "logs"),
        "stage_excludes": [
            "jax_experiments/eval_bundles*/",
            "paper/",
        ],
        "allow_remote_large_data": True,
        "allow_initial_resume_scan_error": False,
        "reroute_on_node_down": True,
        "node_down_requeue_s": 900,
    }


def archive_protocol_signature(run_dir: Path) -> None:
    source = run_dir / "logs" / "protocol_signature.json"
    target = run_dir / "logs" / "protocol_signature_shared_empirical.json"
    if source.exists() and not target.exists():
        shutil.copy2(source, target)


def submit_jsonl(specs: list[dict]) -> list[str]:
    payload = "".join(json.dumps(spec) + "\n" for spec in specs)
    command = [
        sys.executable, str(SCHEDULER), "submit-jsonl",
        "--stdin", "--trusted", "--json",
        "--intent-label", "bapr-v3-inverse-student-submit",
        "--intent-ttl", "900",
    ]
    result = subprocess.run(
        command, input=payload, text=True, capture_output=True,
        env=previous.common.scheduler_env())
    if result.returncode != 0:
        print((result.stdout or "") + (result.stderr or ""), file=sys.stderr)
        result.check_returncode()
    response = json.loads(result.stdout)
    print(json.dumps(response, indent=2))
    return [str(item["id"]) for item in response.get("submitted", [])]


def dispatch(task_ids: list[str]) -> None:
    if not task_ids:
        return
    command = [
        sys.executable, str(SCHEDULER), "dispatch", "--bulk-window",
        "--intent-label", "bapr-v3-inverse-student-dispatch",
        "--intent-ttl", "900",
    ]
    for task_id in task_ids:
        command += ["--task-id", task_id]
    subprocess.run(command, check=True, env=previous.common.scheduler_env())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-iters", type=int, default=2600)
    parser.add_argument("--base-iters", type=int, default=700)
    parser.add_argument("--teacher-iters", type=int, default=700)
    parser.add_argument("--dwell-steps", type=int, default=500)
    parser.add_argument("--env", action="append", choices=ENVS)
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="high")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

    student_start = args.base_iters + args.teacher_iters
    if args.max_iters <= 2000:
        raise SystemExit("max-iters must continue beyond the completed iter 1999")
    for path in (SCHEDULER, JAX_PYTHON):
        if not Path(path).exists():
            raise SystemExit(f"missing required path: {path}")

    envs = args.env or list(ENVS)
    active = {
        str(task.get("signature")) for task in previous.common.scheduler_tasks()
        if task.get("signature")
        and task.get("status") in {"queued", "launching", "running"}
    }
    specs = []
    skipped = []
    for env in envs:
        spec = task_spec(env, args)
        run_dir = Path(spec["ckpt_dir"]).parent
        iteration = previous.common.last_iteration(run_dir)
        if iteration >= args.max_iters - 1:
            skipped.append((spec["signature"], "complete"))
        elif iteration < max(student_start - 1, 1999):
            raise SystemExit(
                f"{env} source checkpoint incomplete: iter={iteration}, "
                f"need >= {max(student_start - 1, 1999)}")
        elif spec["signature"] in active:
            skipped.append((spec["signature"], "active"))
        else:
            archive_protocol_signature(run_dir)
            specs.append(spec)

    print(
        f"BAPR-v3 inverse student: expected={len(envs)} "
        f"submit={len(specs)} skip={len(skipped)}", flush=True)
    for spec in specs:
        print(f"  {spec['signature']}", flush=True)
    for signature, reason in skipped:
        print(f"  skip {reason}: {signature}", flush=True)
    if args.dry_run or not specs:
        return

    task_ids = submit_jsonl(specs)
    print(f"Submitted task ids: {','.join(task_ids)}", flush=True)
    if args.dispatch:
        dispatch(task_ids)


if __name__ == "__main__":
    main()
