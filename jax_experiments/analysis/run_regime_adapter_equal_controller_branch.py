"""Train one fixed-mode adapter with the full robust-controller budget."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from jax_experiments.analysis import regime_adapter_equal_controller as protocol
from jax_experiments.analysis import regime_adapter_fork as branch_protocol
from jax_experiments.analysis import run_regime_adapter_branch as branch


REQUIRED_BUNDLE_FILES = {
    "checkpoints/params.pkl",
    "checkpoints/train_state.pkl",
    "checkpoints/" + branch_protocol.BOOTSTRAP_NAME,
    "checkpoints/" + protocol.EXTENSION_BOOTSTRAP_NAME,
    "logs/protocol_signature.json",
}


def _source_manifest(seed: int) -> Path:
    return protocol.source_bundle_dir(seed, protocol.MODES[0]) / (
        branch_protocol.BUNDLE_MANIFEST_NAME)


def _bootstrap(seed: int, mode: int, run_dir: Path) -> dict:
    """Create a fresh source-boundary branch with one empty replay."""
    seed = protocol.require_seed(seed)
    mode = protocol.require_mode(mode)
    original = branch._bootstrap(
        seed, "adapter", run_dir, protocol.DELTA, mode)
    path = run_dir / "checkpoints" / protocol.EXTENSION_BOOTSTRAP_NAME
    expected_identity = protocol.identity(seed, mode)
    if path.is_file():
        payload = protocol.read_json(path)
        if (payload.get("schema")
                != "bapr.regime-adapter-equal-controller-bootstrap.v1"
                or payload.get("identity") != expected_identity):
            raise ValueError(f"invalid existing equal-controller bootstrap: {path}")
        return payload

    source_manifest = _source_manifest(seed)
    payload = {
        "schema": "bapr.regime-adapter-equal-controller-bootstrap.v1",
        "status": "complete",
        "identity": expected_identity,
        "semantics": (
            "Fresh common-controller branch trained continuously from iter "
            "1400 to iter 2100. Replay and optimizer states are reset exactly "
            "once at the common fork boundary; there is no iter-1575 resume."
        ),
        "source_bundle_manifest": protocol.file_record(source_manifest),
        "source_checkpoint": {
            "next_iteration": protocol.SOURCE_NEXT_ITERATION,
            "total_steps": protocol.SOURCE_TOTAL_STEPS,
            "update_count": protocol.SOURCE_UPDATE_COUNT,
        },
        "original_bootstrap": protocol.file_record(
            run_dir / "checkpoints" / branch_protocol.BOOTSTRAP_NAME),
        "empty_replay_at_source_boundary": (
            original.get("empty_replay") is True),
        "optimizer_states_reset_at_source_boundary": (
            original.get("optimizer_states_reset") is True),
        "continuous_post_fork_training": True,
    }
    if (payload["empty_replay_at_source_boundary"] is not True
            or payload["optimizer_states_reset_at_source_boundary"] is not True):
        raise ValueError("source branch did not use the frozen fork semantics")
    protocol.write_json_atomic(path, payload)
    return payload


def _replace_option(command: list[str], option: str, value: int) -> None:
    try:
        index = command.index(option)
    except ValueError as error:
        raise ValueError(f"training command lacks {option}") from error
    command[index + 1] = str(value)


def _training_command(seed: int, mode: int, run_dir: Path) -> list[str]:
    command = branch._training_command(
        seed, "adapter", run_dir, protocol.DELTA, mode)
    _replace_option(command, "--max_iters", protocol.FINAL_NEXT_ITERATION)
    _replace_option(
        command, "--min_resume_iteration", protocol.SOURCE_NEXT_ITERATION)
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
            f"resume iteration is outside the sealed branch: {next_iteration}")
    return (
        protocol.SOURCE_TOTAL_STEPS
        + (next_iteration - protocol.SOURCE_NEXT_ITERATION)
        * branch_protocol.SAMPLES_PER_ITER
    )


def _replay_is_readable(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            return bool(payload.files)
    except (EOFError, OSError, ValueError):
        return False


def _restart_corrupt_incomplete_branch(
        seed: int, mode: int, run_dir: Path, checkpoint: dict) -> dict:
    """Restart at the common source instead of resetting replay mid-branch."""
    replay = run_dir / "checkpoints" / "replay_buffer.npz"
    if checkpoint["next_iteration"] >= protocol.FINAL_NEXT_ITERATION:
        return checkpoint
    if _replay_is_readable(replay):
        return checkpoint

    discarded = dict(checkpoint)
    shutil.rmtree(run_dir)
    extension = _bootstrap(seed, mode, run_dir)
    extension["recovery"] = {
        "reason": "missing_or_corrupt_incomplete_replay",
        "discarded_checkpoint": discarded,
        "action": "clean_restart_from_common_source_boundary",
        "mid_branch_empty_replay_resume": False,
    }
    protocol.write_json_atomic(
        run_dir / "checkpoints" / protocol.EXTENSION_BOOTSTRAP_NAME,
        extension)
    current = branch_protocol.source.checkpoint_record(run_dir)
    if current["next_iteration"] != protocol.SOURCE_NEXT_ITERATION:
        raise RuntimeError("clean replay recovery did not restore the source")
    print(
        "EQUAL-CONTROLLER RECOVERY: discarded incomplete checkpoint "
        f"iter={discarded['next_iteration']} because replay was corrupt; "
        f"restarted from iter={current['next_iteration']}",
        flush=True,
    )
    return current


def validate_branch(seed: int, mode: int, run_dir: Path) -> dict:
    seed = protocol.require_seed(seed)
    mode = protocol.require_mode(mode)
    original_path = (
        run_dir / "checkpoints" / branch_protocol.BOOTSTRAP_NAME)
    extension_path = (
        run_dir / "checkpoints" / protocol.EXTENSION_BOOTSTRAP_NAME)
    original = protocol.read_json(original_path)
    extension = protocol.read_json(extension_path)
    if (original.get("schema") != "bapr.regime-adapter-bootstrap.v1"
            or original.get("identity")
            != branch_protocol.identity(
                seed, "adapter", protocol.DELTA, mode)
            or original.get("empty_replay") is not True
            or original.get("optimizer_states_reset") is not True):
        raise ValueError(f"invalid original adapter bootstrap: {run_dir}")
    if (extension.get("schema")
            != "bapr.regime-adapter-equal-controller-bootstrap.v1"
            or extension.get("identity") != protocol.identity(seed, mode)
            or extension.get("source_bundle_manifest")
            != protocol.file_record(_source_manifest(seed))
            or extension.get("original_bootstrap")
            != protocol.file_record(original_path)
            or extension.get("continuous_post_fork_training") is not True):
        raise ValueError(f"invalid equal-controller bootstrap: {run_dir}")

    config, env, agent, next_iteration, total_steps, temporary = (
        branch._load_final_agent(run_dir))
    try:
        checkpoint = branch_protocol.source.checkpoint_record(run_dir)
        if checkpoint != protocol.expected_checkpoint():
            raise ValueError(
                f"equal-controller branch has wrong final budget: {checkpoint}")
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
            raise ValueError(
                "equal-controller branch has an invalid checkpoint-resume "
                "boundary")
        if (config.algo != "bapr_regime"
                or int(config.stochastic_mode_fixed_id) != mode
                or float(config.bapr_v2_residual_delta) != protocol.DELTA):
            raise ValueError("equal-controller adapter config changed")

        final_components = {
            "frozen_base": branch_protocol.base_policy_sha256(agent.policy),
            "residual": branch_protocol.residual_policy_sha256(agent.policy),
            "critic": branch_protocol.critic_sha256(agent.critic),
            "target_critic": branch_protocol.critic_sha256(
                agent.target_critic),
        }
        initial = original["component_hashes"]
        if final_components["frozen_base"] != initial["frozen_base"]:
            raise ValueError("frozen robust actor changed during adaptation")
        if final_components["residual"] == initial["initial_residual"]:
            raise ValueError("equal-controller residual did not update")
        if final_components["critic"] == initial["initial_critic"]:
            raise ValueError("equal-controller critic did not update")

        mode_log = np.load(
            run_dir / "logs" / "mode_id.npy", allow_pickle=False)
        if (len(mode_log) != protocol.FINAL_NEXT_ITERATION
                or not np.all(
                    mode_log[protocol.SOURCE_NEXT_ITERATION:] == mode)):
            raise ValueError(
                "equal-controller rollout escaped its fixed mode")
        return {
            "identity": protocol.identity(seed, mode),
            "checkpoint": checkpoint,
            "bootstrap": extension,
            "protocol_signature": signature,
            "last_resume_boundary": {
                "next_iteration": resume_iteration,
                "total_steps": expected_resume_steps,
            },
            "final_components": final_components,
        }
    finally:
        temporary.cleanup()
        if hasattr(env, "close"):
            env.close()


def validate_published_bundle(
        seed: int, role: str = "adapter",
        delta: float = protocol.DELTA, mode: int | None = None) -> dict:
    if role != "adapter":
        raise ValueError("equal-controller bundles contain adapters only")
    if float(delta) != protocol.DELTA or mode is None:
        raise ValueError("equal-controller bundle identity changed")
    seed = protocol.require_seed(seed)
    mode = protocol.require_mode(mode)
    directory = protocol.adapter_bundle_dir(seed, delta, mode)
    payload = protocol.read_json(
        directory / protocol.BUNDLE_MANIFEST_NAME)
    if (payload.get("schema") != protocol.BUNDLE_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("identity") != protocol.identity(seed, mode)
            or set(payload.get("files") or {}) != REQUIRED_BUNDLE_FILES
            or payload.get("checkpoint") != protocol.expected_checkpoint()):
        raise ValueError(f"invalid equal-controller bundle: {directory}")
    for relative, expected in payload["files"].items():
        path = directory / relative
        if not path.is_file() or protocol.file_record(path) != expected:
            raise ValueError(f"equal-controller bundle file changed: {path}")
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
            Path("checkpoints") / branch_protocol.BOOTSTRAP_NAME,
            Path("checkpoints") / protocol.EXTENSION_BOOTSTRAP_NAME,
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
            "budget_semantics": {
                "shared_pretrain_steps": protocol.SOURCE_TOTAL_STEPS,
                "per_controller_post_fork_steps": (
                    protocol.PER_CONTROLLER_POST_FORK_STEPS),
                "adapter_bank_aggregate_post_fork_steps": (
                    protocol.BANK_AGGREGATE_POST_FORK_STEPS),
                "adapter_bank_aggregate_total_steps": (
                    protocol.BANK_AGGREGATE_TOTAL_STEPS),
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
        print(f"EQUAL-CONTROLLER BRANCH ALREADY COMPLETE: {destination}")
        return

    _bootstrap(seed, mode, run_dir)
    expected = protocol.expected_checkpoint()
    current = branch_protocol.source.checkpoint_record(run_dir)
    if current["next_iteration"] > expected["next_iteration"]:
        raise ValueError(
            f"checkpoint is newer than the sealed protocol: {current}")
    if current["next_iteration"] < expected["next_iteration"]:
        current = _restart_corrupt_incomplete_branch(
            seed, mode, run_dir, current)
        command = _training_command(seed, mode, run_dir)
        print("EQUAL-CONTROLLER TRAIN:", " ".join(command), flush=True)
        subprocess.run(
            command, cwd=protocol.ROOT,
            env=_training_environment(), check=True)
    payload = publish_bundle(seed, mode, run_dir)
    replay = run_dir / "checkpoints" / "replay_buffer.npz"
    if replay.is_file():
        replay.unlink()
    print("EQUAL-CONTROLLER BRANCH COMPLETE: " + json.dumps(
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
