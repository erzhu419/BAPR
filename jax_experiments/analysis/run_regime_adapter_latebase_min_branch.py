"""Train and publish one late-base, min-target fixed-mode adapter."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from jax_experiments.analysis import regime_adapter_fork as source_protocol
from jax_experiments.analysis import regime_adapter_latebase_min as protocol
from jax_experiments.analysis import run_regime_adapter_branch as source_runner
from jax_experiments.analysis import (
    run_regime_adapter_latebase_min_prepare as prepare,
)
from jax_experiments.common.checkpoint import (
    _patch_flax_variablestate_unpickle,
)


REQUIRED_BUNDLE_FILES = {
    "checkpoints/params.pkl",
    "checkpoints/train_state.pkl",
    "checkpoints/" + protocol.BRANCH_BOOTSTRAP_NAME,
    "logs/protocol_signature.json",
}


def _checkpoint_record(directory: Path) -> dict:
    _patch_flax_variablestate_unpickle()
    return source_protocol.source.checkpoint_record(directory)


def _replace_option(command: list[str], option: str, value: int) -> None:
    try:
        index = command.index(option)
    except ValueError as error:
        raise ValueError(f"training command lacks {option}") from error
    command[index + 1] = str(value)


def _training_command(seed: int, mode: int, run_dir: Path) -> list[str]:
    command = source_runner._training_command(
        seed, "adapter", run_dir, protocol.DELTA, mode)
    _replace_option(command, "--max_iters", protocol.FINAL_NEXT_ITERATION)
    _replace_option(
        command, "--min_resume_iteration", protocol.SOURCE_NEXT_ITERATION)
    command.extend([
        "--bapr_v2_critic_target_mode", "min",
        "--bapr_v2_freeze_alpha",
    ])
    return command


def _training_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(protocol.ROOT)
    xla_flags = environment.get("XLA_FLAGS", "").split()
    for flag in (
            "--xla_gpu_enable_triton_gemm=false",
            "--xla_cpu_multi_thread_eigen=false",
            "intra_op_parallelism_threads=1"):
        if flag not in xla_flags:
            xla_flags.append(flag)
    environment["XLA_FLAGS"] = " ".join(xla_flags)
    for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            "JAX_NUM_THREADS", "TF_NUM_INTRAOP_THREADS",
            "TF_NUM_INTEROP_THREADS"):
        environment[name] = "1"
    environment["JAX_CPU_ENABLE_ASYNC_DISPATCH"] = "false"
    return environment


def _expected_steps_at_resume(next_iteration: int) -> int:
    next_iteration = int(next_iteration)
    if not (
            protocol.SOURCE_NEXT_ITERATION
            <= next_iteration
            <= protocol.FINAL_NEXT_ITERATION):
        raise ValueError(
            f"resume iteration is outside the late-base branch: "
            f"{next_iteration}")
    return (
        protocol.SOURCE_TOTAL_STEPS
        + (next_iteration - protocol.SOURCE_NEXT_ITERATION)
        * protocol.SAMPLES_PER_ITER
    )


def _replay_is_readable(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            return bool(payload.files)
    except (EOFError, OSError, ValueError):
        return False


def _bootstrap(seed: int, mode: int, run_dir: Path) -> dict:
    seed = protocol.require_seed(seed)
    mode = protocol.require_mode(mode)
    canonical = prepare.validate_canonical(seed)
    path = run_dir / "checkpoints" / protocol.BRANCH_BOOTSTRAP_NAME
    expected_identity = protocol.identity(seed, mode)
    if path.is_file():
        payload = protocol.read_json(path)
        if (payload.get("schema")
                != "bapr.regime-adapter-latebase-branch-bootstrap.v1"
                or payload.get("identity") != expected_identity
                or payload.get("canonical_manifest")
                != protocol.file_record(protocol.canonical_manifest(seed))):
            raise ValueError(f"invalid late-base branch bootstrap: {path}")
        return payload

    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(protocol.canonical_dir(seed), run_dir)
    payload = {
        "schema": "bapr.regime-adapter-latebase-branch-bootstrap.v1",
        "status": "complete",
        "identity": expected_identity,
        "canonical_manifest": protocol.file_record(
            protocol.canonical_manifest(seed)),
        "source_bundle_manifest": canonical["source_bundle_manifest"],
        "initial_components": canonical["initial_components"],
        "empty_replay_at_late_base": True,
        "optimizer_states_reset_at_late_base": True,
        "canonical_clone": True,
    }
    protocol.write_json_atomic(path, payload)
    checkpoint = _checkpoint_record(run_dir)
    if checkpoint != {
            "iteration": protocol.SOURCE_NEXT_ITERATION - 1,
            "next_iteration": protocol.SOURCE_NEXT_ITERATION,
            "total_steps": protocol.SOURCE_TOTAL_STEPS,
            "update_count": protocol.SOURCE_UPDATE_COUNT,
            "algo": "bapr_regime"}:
        shutil.rmtree(run_dir)
        raise ValueError("canonical clone has the wrong checkpoint boundary")
    return payload


def _restart_corrupt_incomplete_branch(
        seed: int, mode: int, run_dir: Path, checkpoint: dict) -> dict:
    replay = run_dir / "checkpoints" / "replay_buffer.npz"
    if checkpoint["next_iteration"] >= protocol.FINAL_NEXT_ITERATION:
        return checkpoint
    if _replay_is_readable(replay):
        return checkpoint
    discarded = dict(checkpoint)
    shutil.rmtree(run_dir)
    _bootstrap(seed, mode, run_dir)
    current = _checkpoint_record(run_dir)
    print(
        "LATE-BASE RECOVERY: discarded incomplete checkpoint "
        f"iter={discarded['next_iteration']} because replay was corrupt; "
        f"restarted from iter={current['next_iteration']}",
        flush=True)
    return current


def validate_branch(seed: int, mode: int, run_dir: Path) -> dict:
    seed = protocol.require_seed(seed)
    mode = protocol.require_mode(mode)
    bootstrap = protocol.read_json(
        run_dir / "checkpoints" / protocol.BRANCH_BOOTSTRAP_NAME)
    canonical = prepare.validate_canonical(seed)
    if (bootstrap.get("schema")
            != "bapr.regime-adapter-latebase-branch-bootstrap.v1"
            or bootstrap.get("identity") != protocol.identity(seed, mode)
            or bootstrap.get("canonical_manifest")
            != protocol.file_record(protocol.canonical_manifest(seed))
            or bootstrap.get("initial_components")
            != canonical["initial_components"]):
        raise ValueError(f"invalid late-base branch provenance: {run_dir}")

    config, env, agent, next_iteration, total_steps, temporary = (
        source_runner._load_final_agent(run_dir))
    try:
        checkpoint = _checkpoint_record(run_dir)
        if checkpoint != protocol.expected_checkpoint():
            raise ValueError(
                f"late-base branch has wrong final budget: {checkpoint}")
        if (next_iteration != checkpoint["next_iteration"]
                or total_steps != checkpoint["total_steps"]):
            raise ValueError("loaded branch state disagrees with checkpoint")
        signature = protocol.read_json(
            run_dir / "logs" / "protocol_signature.json")
        signature_config = signature.get("config") or {}
        resume_iteration = int(signature.get("start_iteration", -1))
        expected_resume_steps = _expected_steps_at_resume(resume_iteration)
        if (signature.get("checkpoint_loaded") is not True
                or int(signature.get("total_steps_at_start", -1))
                != expected_resume_steps
                or signature_config.get("start_train_steps") != 0
                or signature_config.get("max_iters")
                != protocol.FINAL_NEXT_ITERATION):
            raise ValueError("late-base resume boundary changed")
        if (config.algo != "bapr_regime"
                or int(config.stochastic_mode_fixed_id) != mode
                or float(config.bapr_v2_residual_delta) != protocol.DELTA
                or config.bapr_v2_critic_target_mode != "min"
                or config.bapr_v2_freeze_alpha is not True):
            raise ValueError("late-base adapter config changed")

        initial = canonical["initial_components"]
        final_components = {
            "frozen_base": source_protocol.base_policy_sha256(agent.policy),
            "residual": source_protocol.residual_policy_sha256(agent.policy),
            "critic": source_protocol.critic_sha256(agent.critic),
            "target_critic": source_protocol.critic_sha256(
                agent.target_critic),
            "log_alpha": prepare._array_record(agent.log_alpha),
        }
        if final_components["frozen_base"] != initial["frozen_base"]:
            raise ValueError("late-base frozen actor changed")
        if final_components["residual"] == initial["residual"]:
            raise ValueError("late-base residual did not update")
        if final_components["critic"] == initial["critic"]:
            raise ValueError("late-base critic did not update")
        if final_components["log_alpha"] != initial["log_alpha"]:
            raise ValueError("late-base entropy temperature changed")

        mode_log = np.load(
            run_dir / "logs" / "mode_id.npy", allow_pickle=False)
        if (len(mode_log) != protocol.FINAL_NEXT_ITERATION
                or not np.all(
                    mode_log[protocol.SOURCE_NEXT_ITERATION:] == mode)):
            raise ValueError("late-base rollout escaped its fixed mode")
        return {
            "identity": protocol.identity(seed, mode),
            "checkpoint": checkpoint,
            "bootstrap": bootstrap,
            "protocol_signature": signature,
            "final_components": final_components,
        }
    finally:
        temporary.cleanup()
        if hasattr(env, "close"):
            env.close()


def validate_published_bundle(
        seed: int, role: str = "adapter",
        delta: float = protocol.DELTA, mode: int | None = None) -> dict:
    if role != "adapter" or mode is None:
        raise ValueError("late-base bundles contain fixed-mode adapters only")
    seed = protocol.require_seed(seed)
    mode = protocol.require_mode(mode)
    directory = protocol.adapter_bundle_dir(seed, delta, mode)
    payload = protocol.read_json(
        directory / protocol.BUNDLE_MANIFEST_NAME)
    if (payload.get("schema") != protocol.BUNDLE_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("identity") != protocol.identity(seed, mode)
            or set(payload.get("files") or {}) != REQUIRED_BUNDLE_FILES
            or payload.get("checkpoint") != protocol.expected_checkpoint()
            or payload.get("canonical_manifest")
            != protocol.file_record(protocol.canonical_manifest(seed))):
        raise ValueError(f"invalid late-base adapter bundle: {directory}")
    for relative, expected in payload["files"].items():
        path = directory / relative
        if not path.is_file() or protocol.file_record(path) != expected:
            raise ValueError(f"late-base bundle file changed: {path}")
    return payload


def publish_bundle(seed: int, mode: int, run_dir: Path) -> dict:
    validation = validate_branch(seed, mode, run_dir)
    destination = protocol.adapter_bundle_dir(
        seed, protocol.DELTA, mode)
    manifest = destination / protocol.BUNDLE_MANIFEST_NAME
    if manifest.is_file():
        return validate_published_bundle(
            seed, "adapter", protocol.DELTA, mode)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp.", dir=destination.parent))
    try:
        files = (
            Path("checkpoints") / "params.pkl",
            Path("checkpoints") / "train_state.pkl",
            Path("checkpoints") / protocol.BRANCH_BOOTSTRAP_NAME,
            Path("logs") / "protocol_signature.json",
        )
        records = {}
        for relative in files:
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(run_dir / relative, target)
            records[relative.as_posix()] = protocol.file_record(target)
        payload = {
            "schema": protocol.BUNDLE_SCHEMA,
            "status": "complete",
            "identity": protocol.identity(seed, mode),
            "checkpoint": validation["checkpoint"],
            "canonical_manifest": protocol.file_record(
                protocol.canonical_manifest(seed)),
            "budget_semantics": {
                "shared_late_base_steps": protocol.SOURCE_TOTAL_STEPS,
                "per_controller_post_fork_steps": (
                    protocol.PER_CONTROLLER_POST_FORK_STEPS),
                "adapter_bank_aggregate_post_fork_steps": (
                    protocol.BANK_AGGREGATE_POST_FORK_STEPS),
                "compute_matched_to_robust": False,
            },
            "components": validation["final_components"],
            "files": records,
        }
        protocol.write_json_atomic(
            temporary / protocol.BUNDLE_MANIFEST_NAME, payload)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return validate_published_bundle(
        seed, "adapter", protocol.DELTA, mode)


def run(seed: int, mode: int) -> None:
    seed = protocol.require_seed(seed)
    mode = protocol.require_mode(mode)
    run_dir = protocol.run_dir(seed, mode)
    destination = protocol.adapter_bundle_dir(
        seed, protocol.DELTA, mode)
    if (destination / protocol.BUNDLE_MANIFEST_NAME).is_file():
        validate_published_bundle(
            seed, "adapter", protocol.DELTA, mode)
        print(f"LATE-BASE BRANCH ALREADY COMPLETE: {destination}")
        return

    _bootstrap(seed, mode, run_dir)
    checkpoint = _checkpoint_record(run_dir)
    checkpoint = _restart_corrupt_incomplete_branch(
        seed, mode, run_dir, checkpoint)
    if checkpoint["next_iteration"] < protocol.FINAL_NEXT_ITERATION:
        command = _training_command(seed, mode, run_dir)
        print("LATE-BASE TRAIN:", " ".join(command), flush=True)
        subprocess.run(
            command, cwd=protocol.ROOT,
            env=_training_environment(), check=True)
    payload = publish_bundle(seed, mode, run_dir)
    replay = run_dir / "checkpoints" / "replay_buffer.npz"
    if replay.is_file():
        replay.unlink()
    print("LATE-BASE BRANCH COMPLETE: " + json.dumps(
        payload["identity"], sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, choices=protocol.TRAINING_SEEDS, required=True)
    parser.add_argument(
        "--mode", type=int, choices=protocol.MODES, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for checkpoint-safe execution")
    run(args.seed, args.mode)


if __name__ == "__main__":
    main()
