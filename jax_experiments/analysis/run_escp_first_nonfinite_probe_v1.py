"""Record the first non-finite update of the legacy JAX ESCP path."""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys

from jax_experiments.analysis import resac_escp_semantics_v2 as protocol


NONFINITE_RE = re.compile(
    r"ESCP non-finite update at global_update=(?P<global_update>\d+), "
    r"scan_index=(?P<scan_index>\d+), iter=(?P<iteration>\d+), "
    r"target_mode=(?P<target_mode>[^,]+), actor_mode=(?P<actor_mode>[^,]+), "
    r"details=(?P<details>.*)")


def _command(env: str, seed: int) -> list[str]:
    directory = protocol.probe_dir(env, seed) / "run"
    return [
        sys.executable, "-u", "-m", "jax_experiments.train",
        "--algo", "escp",
        "--env", protocol.require_env(env),
        "--seed", str(seed),
        "--max_iters", str(protocol.PROBE_MAX_ITERS),
        "--save_root", str(directory.parent),
        "--run_name", directory.name,
        "--env_type", "continuous",
        "--varying_params", "gravity",
        "--task_scale_distribution", "pow1p5",
        "--log_scale_limit", "3.0",
        "--changing_period", "20000",
        "--changing_interval", "4000",
        "--task_num", str(protocol.TASK_NUM),
        "--test_task_num", str(protocol.TEST_TASK_NUM),
        "--samples_per_iter", str(protocol.SAMPLES_PER_ITER),
        "--updates_per_iter", str(protocol.UPDATES_PER_ITER),
        "--start_train_steps", str(protocol.INITIAL_RANDOM_STEPS),
        "--initial_random_steps", str(protocol.INITIAL_RANDOM_STEPS),
        "--ensemble_size", str(protocol.ESCP_ENSEMBLE_SIZE),
        "--hidden_dim", str(protocol.HIDDEN_DIM),
        "--lr", str(protocol.LR),
        "--max_episode_steps", str(protocol.MAX_EPISODE_STEPS),
        "--backend", "spring",
        "--eval_protocol", "stationary",
        "--log_interval", "1",
        "--eval_episodes", "1",
        "--save_interval", str(protocol.PROBE_SAVE_INTERVAL),
        "--escp_target_mode", "independent",
        "--escp_actor_mode", "lcb",
        "--escp_context_min_steps", "0",
        "--escp_context_min_tasks", "0",
        "--escp_alpha_max", "-1",
        "--context_warmup_iters", "50",
        "--resume",
    ]


def validate(env: str, seed: int) -> dict:
    payload = protocol.read_json(protocol.probe_result(env, seed))
    if (payload.get("schema") != protocol.PROBE_SCHEMA
            or payload.get("status") not in {
                "first_nonfinite_captured", "finite_through_limit"}
            or payload.get("env") != protocol.require_env(env)
            or int(payload.get("seed")) != int(seed)
            or int(payload.get("max_iters")) != protocol.PROBE_MAX_ITERS):
        raise ValueError("invalid ESCP first-nonfinite probe result")
    return payload


def run(env: str, seed: int) -> None:
    env = protocol.require_env(env)
    seed = int(seed)
    destination = protocol.probe_dir(env, seed)
    result_path = protocol.probe_result(env, seed)
    if result_path.is_file():
        validate(env, seed)
        print(f"ESCP NONFINITE PROBE ALREADY COMPLETE: {result_path}")
        return
    destination.mkdir(parents=True, exist_ok=True)
    command = _command(env, seed)
    print("ESCP NONFINITE PROBE:", " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(protocol.ROOT)
    completed = subprocess.run(
        command,
        cwd=protocol.ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = completed.stdout or ""
    output_path = destination / "subprocess.log"
    output_path.write_text(output, encoding="utf-8")
    match = NONFINITE_RE.search(output)
    if match is None:
        print(output[-8000:], flush=True)
    else:
        # The child traceback is the expected probe payload. Keep it in the
        # artifact, but do not emit crash keywords that make the scheduler
        # classify this successful capture as a failed task.
        print(
            "ESCP EXPECTED NONFINITE CAPTURE: "
            f"global_update={match.group('global_update')}, "
            f"iter={match.group('iteration')}, "
            f"target_mode={match.group('target_mode')}, "
            f"actor_mode={match.group('actor_mode')}",
            flush=True,
        )
    if match is None and completed.returncode != 0:
        raise RuntimeError(
            "legacy ESCP failed without the finite-guard marker; see "
            f"{output_path}")
    if match is not None:
        if completed.returncode == 0:
            raise RuntimeError("finite-guard marker appeared on a successful run")
        status = "first_nonfinite_captured"
        first_nonfinite = {
            "global_update": int(match.group("global_update")),
            "scan_index": int(match.group("scan_index")),
            "iteration": int(match.group("iteration")),
            "target_mode": match.group("target_mode"),
            "actor_mode": match.group("actor_mode"),
            "details": match.group("details"),
        }
    else:
        status = "finite_through_limit"
        first_nonfinite = None
    protocol.write_json_atomic(result_path, {
        "schema": protocol.PROBE_SCHEMA,
        "status": status,
        "env": env,
        "seed": seed,
        "max_iters": protocol.PROBE_MAX_ITERS,
        "updates_per_iter": protocol.UPDATES_PER_ITER,
        "returncode": completed.returncode,
        "first_nonfinite": first_nonfinite,
        "command": command,
        "subprocess_log": protocol.file_record(output_path),
    })
    checkpoint_dir = destination / "run" / "checkpoints"
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    validate(env, seed)
    print(f"ESCP NONFINITE PROBE COMPLETE: {result_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=protocol.ENVS, required=True)
    parser.add_argument("--seed", choices=protocol.PROBE_SEEDS, type=int,
                        required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for scheduler execution")
    run(args.env, args.seed)


if __name__ == "__main__":
    main()
