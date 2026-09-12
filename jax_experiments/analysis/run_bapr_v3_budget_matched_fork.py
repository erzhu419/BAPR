#!/usr/bin/env python3
"""Run one BAPR-v3 budget-matched pair from one shared checkpoint.

This is intentionally a single scheduler payload.  It trains ``shared_base``
through iter 699, atomically copies that complete run to two branch directories,
then resumes ``robust_long`` and ``oracle_direct`` in separate, sequential train
subprocesses.  The fork is a shared-checkpoint/common-restart protocol, not an
exact continuation: the training checkpoint does not contain live environment
or all process RNG state.

The runner is restartable only on the same host, Python runtime, visible GPU,
and source snapshot.  A retry with any identity mismatch fails closed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import shutil
import subprocess
import sys
import tarfile
import uuid
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
SAVE_ROOT = (
    ROOT / "jax_experiments" / "results_bapr_v3_budget_matched_fork_v2")
FAMILIES = ("deterministic_mean", "mean_variance")
STOCHASTIC_EVENT_FAMILIES = (
    "packet_loss", "burst_torque", "structured_channel")
SUPPORTED_FAMILIES = FAMILIES + STOCHASTIC_EVENT_FAMILIES
ENVS = ("Ant-v2", "HalfCheetah-v2")
RUN_NAMES = ("shared_base", "robust_long", "oracle_direct")
POLICY_VARIANTS = {
    "direct": {
        "policy_mode": "direct",
        "num_experts": None,
        "action_deviation_weight": 0.0,
        "actor_objective": "mean",
    },
    "cat_mean": {
        "policy_mode": "categorical_expert",
        "num_experts": 4,
        "action_deviation_weight": 0.0,
        "actor_objective": "mean",
    },
    "cat_lcb": {
        "policy_mode": "categorical_expert",
        "num_experts": 4,
        "action_deviation_weight": 0.0,
        "actor_objective": "lcb",
    },
    "cat_anchor_0p1": {
        "policy_mode": "categorical_expert",
        "num_experts": 4,
        "action_deviation_weight": 0.1,
        "actor_objective": "mean",
    },
    "cat_anchor_1p0": {
        "policy_mode": "categorical_expert",
        "num_experts": 4,
        "action_deviation_weight": 1.0,
        "actor_objective": "mean",
    },
}

SCHEMA = "bapr.v3-budget-matched-fork.v2"
SEMANTICS = "shared-checkpoint/common-restart; not exact continuation"
BASE_FINAL_ITERATION = 699
BASE_NEXT_ITERATION = 700
BASE_TOTAL_STEPS = 2_800_000
BASE_UPDATE_COUNT = 174_500
FINAL_ITERATION = 1399
FINAL_NEXT_ITERATION = 1400
FINAL_TOTAL_STEPS = 5_600_000
FINAL_UPDATE_COUNT = 349_500
SAMPLES_PER_ITER = 4000
UPDATES_PER_ITER = 250

STATE_NAME = "protocol_checkpoint.pkl"
COMPLETE_SENTINEL_NAME = "pair_checkpoint_complete.pkl"
PAIR_MANIFEST_REL = Path("provenance") / "pair_manifest.json"
SOURCE_MANIFEST_REL = Path("provenance") / "source_manifest.json"
SOURCE_ARCHIVE_REL = Path("provenance") / "source_snapshot.tar.gz"
BOUNDARY_AUDIT_NAME = "resume_boundary_audit.json"

CHECKPOINT_FILES = (
    "checkpoints/params.pkl",
    "checkpoints/train_state.pkl",
    "checkpoints/replay_buffer.npz",
)
CONFIG_DIFF_ALLOWLIST = {
    # Run/path and stopping/resume controls.
    "run_name",
    "max_iters",
    "min_resume_iteration",
    # The preregistered treatment assignment.
    "bapr_v2_base_pretrain_iters",
    "bapr_v2_teacher_iters",
    # Audit-only instrumentation; it does not alter an update.
    "resume_boundary_audit",
    "resume_boundary_expected_iteration",
    "resume_boundary_expected_total_steps",
    "resume_boundary_expected_update_count",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def pair_name(family: str, env: str, seed: int = 0) -> str:
    return f"budget_fork_v2_{family}_{env.removesuffix('-v2')}_s{seed}"


def pair_dir(family: str, env: str, seed: int = 0,
             save_root: Path = SAVE_ROOT) -> Path:
    return save_root / pair_name(family, env, seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with tmp.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        fsync_dir(path.parent)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def atomic_write_pickle(path: Path, value: Any) -> None:
    atomic_write_bytes(path, pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def read_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def source_paths() -> list[Path]:
    paths = [Path(__file__).resolve()]
    source_root = ROOT / "jax_experiments"
    for path in source_root.rglob("*.py"):
        relative = path.relative_to(source_root)
        if any(
            part.startswith("results") or part.startswith("eval_bundles")
            or part == "__pycache__"
            for part in relative.parts
        ):
            continue
        paths.append(path.resolve())
    return sorted(set(paths), key=lambda path: path.relative_to(ROOT).as_posix())


def build_file_manifest(paths: Iterable[Path], base: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"source file disappeared: {path}")
        relative = path.relative_to(base).as_posix()
        files[relative] = {
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }
    return {"files": files, "sha256": canonical_sha256(files)}


def current_source_manifest() -> dict[str, Any]:
    result = build_file_manifest(source_paths(), ROOT)
    result.update({
        "schema": "bapr.source-snapshot.v1",
        "root": str(ROOT),
    })
    return result


def create_source_archive(path: Path, manifest: dict[str, Any]) -> None:
    """Archive the exact Python files listed in the immutable manifest."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with tarfile.open(tmp, "w:gz") as archive:
            for relative in sorted(manifest["files"]):
                archive.add(ROOT / relative, arcname=relative, recursive=False)
        with tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        fsync_dir(path.parent)
    finally:
        if tmp.exists():
            tmp.unlink()


def validate_source_archive(path: Path, manifest: dict[str, Any],
                            expected_sha256: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing source archive: {path}")
    actual_sha256 = sha256_file(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise RuntimeError("source archive file hash changed on re-entry")
    expected_files = manifest["files"]
    required_members = {
        "jax_experiments/analysis/run_bapr_v3_budget_matched_fork.py",
        "jax_experiments/analysis/analyze_bapr_v3_budget_matched_fork_audit.py",
        "jax_experiments/analysis/analyze_bapr_v3_budget_matched_audit.py",
        "jax_experiments/train.py",
        "jax_experiments/configs/default.py",
        "jax_experiments/algos/bapr_v2.py",
        "jax_experiments/algos/bapr_v3.py",
        "jax_experiments/common/checkpoint.py",
    }
    missing_required = sorted(required_members - set(expected_files))
    if missing_required:
        raise RuntimeError(
            f"source manifest omits required runtime files: {missing_required}")
    with tarfile.open(path, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        names = [member.name for member in members]
        if names != sorted(expected_files):
            raise RuntimeError("source archive members differ from source manifest")
        for member in members:
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError(f"cannot read source archive member {member.name}")
            digest = hashlib.sha256()
            size = 0
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
            expected = expected_files[member.name]
            if digest.hexdigest() != expected["sha256"] or size != expected["size"]:
                raise RuntimeError(
                    f"source archive content mismatch: {member.name}")
    return {"sha256": actual_sha256, "size": path.stat().st_size}


def selected_gpu_identity() -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    command = [
        "nvidia-smi", "--query-gpu=index,uuid,name,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command, text=True, capture_output=True, timeout=15, check=True)
        rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception as exc:
        rows = []
        error = f"{type(exc).__name__}: {exc}"
    else:
        error = ""
    tokens = {token.strip() for token in visible.split(",") if token.strip()}
    selected = []
    for row in rows:
        columns = [item.strip() for item in row.split(",", 3)]
        if not tokens or any(
            token == columns[0] or (len(columns) > 1 and token == columns[1])
            for token in tokens
        ):
            selected.append(row)
    return {
        "cuda_visible_devices": visible,
        "selected_nvidia_smi_rows": selected,
        "nvidia_smi_error": error,
    }


def package_versions() -> dict[str, str]:
    versions = {}
    for distribution in ("jax", "jaxlib", "flax", "optax", "numpy", "brax"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def jax_runtime_identity() -> dict[str, Any]:
    """Probe JAX in a disposable child so the runner does not retain GPU RAM."""
    code = (
        "import json,jax; "
        "print('__BAPR_JAX_ID__'+json.dumps({"
        "'backend':jax.default_backend(),"
        "'devices':[{'platform':d.platform,'kind':d.device_kind,"
        "'id':int(d.id)} for d in jax.devices()]},sort_keys=True))")
    try:
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT, env=os.environ.copy(),
            text=True, capture_output=True, timeout=60, check=True)
        line = next(
            item for item in reversed(result.stdout.splitlines())
            if item.startswith("__BAPR_JAX_ID__"))
        return json.loads(line.removeprefix("__BAPR_JAX_ID__"))
    except Exception as exc:
        raise RuntimeError(f"cannot probe JAX runtime identity: {exc}") from exc


def runtime_identity() -> dict[str, Any]:
    gpu = selected_gpu_identity()
    jax_identity = jax_runtime_identity()
    devices = jax_identity.get("devices") or []
    if not gpu.get("cuda_visible_devices"):
        raise RuntimeError(
            "paired GPU protocol requires scheduler-assigned "
            "CUDA_VISIBLE_DEVICES")
    if gpu.get("nvidia_smi_error") or len(
            gpu.get("selected_nvidia_smi_rows") or []) != 1:
        raise RuntimeError(
            "cannot prove one scheduler-assigned physical GPU: "
            f"{gpu!r}")
    if jax_identity.get("backend") != "gpu" or len(devices) != 1:
        raise RuntimeError(
            "paired protocol requires exactly one visible JAX GPU device: "
            f"{jax_identity!r}")
    identity = {
        "host": platform.node(),
        "python_executable": sys.executable,
        "python_realpath": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "jax": jax_identity,
        "gpu": gpu,
    }
    identity["sha256"] = canonical_sha256(identity)
    return identity


def stable_runtime_identity(identity: dict[str, Any]) -> dict[str, Any]:
    gpu = identity.get("gpu", {})
    return {
        "host": identity.get("host"),
        "python_executable": identity.get("python_executable"),
        "python_realpath": identity.get("python_realpath"),
        "python_version": identity.get("python_version"),
        "platform": identity.get("platform"),
        "packages": identity.get("packages"),
        "jax": identity.get("jax"),
        "cuda_visible_devices": gpu.get("cuda_visible_devices"),
        "selected_nvidia_smi_rows": gpu.get("selected_nvidia_smi_rows"),
    }


def assert_source_unchanged(expected: dict[str, Any]) -> None:
    current = current_source_manifest()
    if current["sha256"] != expected["sha256"] or current["files"] != expected["files"]:
        raise RuntimeError(
            "source snapshot changed during/re-entering the paired protocol; "
            "refusing to mix sources")


def assert_runtime_unchanged(expected: dict[str, Any]) -> None:
    current = runtime_identity()
    if stable_runtime_identity(current) != stable_runtime_identity(expected):
        raise RuntimeError(
            "runtime/node/GPU identity changed during/re-entering the paired "
            "protocol; refusing to mix runs\n"
            f"expected={stable_runtime_identity(expected)!r}\n"
            f"current={stable_runtime_identity(current)!r}")


def snapshot_paths(run_dir: Path) -> list[Path]:
    paths = [run_dir / relative for relative in CHECKPOINT_FILES]
    log_dir = run_dir / "logs"
    if not log_dir.is_dir():
        raise RuntimeError(f"missing log directory: {log_dir}")
    paths.extend(path for path in log_dir.rglob("*") if path.is_file())
    unique = sorted(set(paths), key=lambda path: path.relative_to(run_dir).as_posix())
    missing = [str(path) for path in unique if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing snapshot files: {missing}")
    return unique


def run_snapshot_manifest(run_dir: Path) -> dict[str, Any]:
    result = build_file_manifest(snapshot_paths(run_dir), run_dir)
    result["run_dir"] = str(run_dir)
    return result


def checkpoint_info(run_dir: Path) -> dict[str, Any]:
    state = read_pickle(run_dir / "checkpoints" / "train_state.pkl")
    if not isinstance(state, dict):
        raise RuntimeError(f"invalid train state in {run_dir}")
    iteration = int(state["iteration"])
    total_steps = int(state["total_steps"])
    params_path = run_dir / "checkpoints" / "params.pkl"
    update_count = None
    update_count_error = ""
    try:
        params = read_pickle(params_path)
        if isinstance(params, dict) and "update_count" in params:
            update_count = int(params["update_count"])
    except Exception as exc:
        update_count_error = f"{type(exc).__name__}: {exc}"
    try:
        import numpy as np

        with np.load(run_dir / "checkpoints" / "replay_buffer.npz") as replay:
            replay_size = int(replay["size"])
            replay_ptr = int(replay["ptr"])
    except Exception as exc:
        raise RuntimeError(f"cannot inspect replay checkpoint in {run_dir}: {exc}") from exc
    return {
        "iteration": iteration,
        "next_iteration": iteration + 1,
        "total_steps": total_steps,
        "update_count": update_count,
        "update_count_error": update_count_error,
        "replay_size": replay_size,
        "replay_ptr": replay_ptr,
        "logger_key_count": len(state.get("logger_data", {})),
    }


def validate_checkpoint(run_dir: Path, *, expected_iteration: int,
                        expected_steps: int,
                        expected_update_count: int) -> dict[str, Any]:
    info = checkpoint_info(run_dir)
    expected = {
        "iteration": expected_iteration,
        "next_iteration": expected_iteration + 1,
        "total_steps": expected_steps,
    }
    for key, value in expected.items():
        if info[key] != value:
            raise RuntimeError(
                f"{run_dir.name} {key}={info[key]}, expected {value}")
    if info["update_count"] is None:
        raise RuntimeError(
            f"{run_dir.name} checkpoint does not expose update_count: "
            f"{info['update_count_error']}")
    if info["update_count"] != expected_update_count:
        raise RuntimeError(
            f"{run_dir.name} update_count={info['update_count']}, "
            f"expected {expected_update_count}")
    return info


def validate_final_logs(run_dir: Path) -> dict[str, Any]:
    import numpy as np

    iteration = np.load(run_dir / "logs" / "iteration.npy")
    total_steps = np.load(run_dir / "logs" / "total_steps.npy")
    if len(iteration) != FINAL_NEXT_ITERATION or int(iteration[-1]) != FINAL_ITERATION:
        raise RuntimeError(f"invalid final iteration log in {run_dir}")
    if len(total_steps) != FINAL_NEXT_ITERATION or int(total_steps[-1]) != FINAL_TOTAL_STEPS:
        raise RuntimeError(f"invalid final total_steps log in {run_dir}")
    return {
        "iteration_rows": len(iteration),
        "iteration_last": int(iteration[-1]),
        "total_steps_rows": len(total_steps),
        "total_steps_last": int(total_steps[-1]),
    }


def validate_complete_artifacts(pair: Path) -> dict[str, Any]:
    manifest_path = pair / PAIR_MANIFEST_REL
    sentinel_path = pair / COMPLETE_SENTINEL_NAME
    manifest = read_json(manifest_path)
    sentinel = read_pickle(sentinel_path)
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
        raise RuntimeError("authoritative pair manifest is not complete")
    if (not isinstance(sentinel, dict)
            or sentinel.get("schema") != "bapr.pair-checkpoint-complete.v1"
            or sentinel.get("status") != "complete"):
        raise RuntimeError("pair completion sentinel is invalid")
    expected_sentinel_keys = {
        "schema", "status", "pair_manifest", "pair_manifest_sha256",
        "shared_base_snapshot_sha256", "final_next_iteration",
        "final_total_steps", "completed_at",
    }
    if set(sentinel) != expected_sentinel_keys:
        raise RuntimeError("pair completion sentinel has an invalid schema")
    if sentinel.get("pair_manifest") != str(PAIR_MANIFEST_REL):
        raise RuntimeError("pair completion sentinel points at the wrong manifest")
    manifest_sha = sha256_file(manifest_path)
    if sentinel.get("pair_manifest_sha256") != manifest_sha:
        raise RuntimeError("pair completion sentinel has a stale manifest hash")
    shared = manifest.get("shared_base") or {}
    if (sentinel.get("shared_base_snapshot_sha256")
            != shared.get("snapshot_sha256")):
        raise RuntimeError("pair completion sentinel has the wrong shared snapshot")
    if (type(sentinel.get("final_next_iteration")) is not int
            or sentinel["final_next_iteration"] != FINAL_NEXT_ITERATION
            or type(sentinel.get("final_total_steps")) is not int
            or sentinel["final_total_steps"] != FINAL_TOTAL_STEPS):
        raise RuntimeError("pair completion sentinel has the wrong final budget")
    if (not isinstance(manifest.get("completed_at"), str)
            or not manifest["completed_at"]
            or sentinel.get("completed_at") != manifest["completed_at"]):
        raise RuntimeError("pair completion sentinel has the wrong completion time")

    identity = manifest.get("identity")
    if (not isinstance(identity, dict)
            or identity.get("family") not in SUPPORTED_FAMILIES
            or identity.get("env") not in ENVS
            or identity.get("seed") != 0
            or pair.name != pair_name(
                identity["family"], identity["env"], identity["seed"])):
        raise RuntimeError("authoritative pair manifest has an invalid identity")

    # Keep the producer's completion check exactly as strict as the downstream
    # causal analyzer.  This validates the source archive member-by-member,
    # runtime digest/single-GPU identity, shared snapshot and fork manifests,
    # boundary canaries/physical equality, config contract, and final files.
    try:
        from jax_experiments.analysis import (
            analyze_bapr_v3_budget_matched_fork_audit as audit_validator,
        )

        audited = audit_validator.validate_pair_provenance(
            pair, identity["family"], identity["env"], identity["seed"])
    except Exception as exc:
        raise RuntimeError(
            f"deep pair provenance validation failed: {exc}") from exc
    if audited != manifest:
        raise RuntimeError("deep validator returned a different pair manifest")

    runtime = manifest.get("runtime") or {}
    runtime_without_hash = dict(runtime)
    runtime_sha = runtime_without_hash.pop("sha256", None)
    if runtime_sha != canonical_sha256(runtime_without_hash):
        raise RuntimeError("pair runtime digest is invalid")
    source = manifest.get("source") or {}
    source_files = source.get("files") or {}
    source_sha = canonical_sha256(source_files)
    if (source.get("manifest_path") != str(SOURCE_MANIFEST_REL)
            or source.get("sha256") != source_sha
            or source.get("snapshot_sha256") != source_sha):
        raise RuntimeError("pair source provenance is invalid")

    shared_files = shared.get("files") or {}
    for relative in CHECKPOINT_FILES:
        if relative not in shared_files:
            raise RuntimeError(f"shared-base manifest omits {relative}")
    if canonical_sha256(shared_files) != shared.get("snapshot_sha256"):
        raise RuntimeError("shared-base snapshot hash is invalid")
    for relative, record in shared_files.items():
        path = pair / "shared_base" / relative
        if (not path.is_file() or sha256_file(path) != record.get("sha256")
                or path.stat().st_size != record.get("size")):
            raise RuntimeError(f"shared-base file validation failed: {relative}")
    required_final_files = set(CHECKPOINT_FILES) | {
        "logs/protocol_signature.json",
        f"logs/{BOUNDARY_AUDIT_NAME}",
    }
    forks = manifest.get("forks") or {}
    if set(forks) != {"robust_long", "oracle_direct"}:
        raise RuntimeError("pair manifest has invalid fork names")
    for branch in ("robust_long", "oracle_direct"):
        fork = forks.get(branch) or {}
        expected_fork = {
            "run_name": branch,
            "source_snapshot_sha256": shared.get("snapshot_sha256"),
            "source_files_verified": len(shared_files),
            "source_checkpoint_next_iteration": BASE_NEXT_ITERATION,
            "source_total_steps": BASE_TOTAL_STEPS,
            "start_iteration": BASE_NEXT_ITERATION,
            "start_total_steps": BASE_TOTAL_STEPS,
            "runtime_sha256": runtime_sha,
            "source_sha256": source_sha,
        }
        if fork != expected_fork:
            raise RuntimeError(f"pair manifest has invalid fork provenance: {branch}")
        branch_final = ((manifest.get("final") or {}).get(branch) or {})
        checkpoint = branch_final.get("checkpoint", {})
        if (checkpoint.get("iteration") != FINAL_ITERATION
                or checkpoint.get("next_iteration") != FINAL_NEXT_ITERATION
                or checkpoint.get("total_steps") != FINAL_TOTAL_STEPS
                or checkpoint.get("update_count") != FINAL_UPDATE_COUNT):
            raise RuntimeError(f"pair manifest has an invalid final {branch}")
        files = branch_final.get("files") or {}
        if set(files) != required_final_files:
            raise RuntimeError(f"pair manifest has incomplete final files: {branch}")
        for relative, record in files.items():
            path = pair / branch / relative
            if (not path.is_file() or sha256_file(path) != record.get("sha256")
                    or path.stat().st_size != record.get("size")):
                raise RuntimeError(
                    f"final file validation failed: {branch}/{relative}")
    base_checkpoint = validate_checkpoint(
        pair / "shared_base", expected_iteration=BASE_FINAL_ITERATION,
        expected_steps=BASE_TOTAL_STEPS,
        expected_update_count=BASE_UPDATE_COUNT)
    if (base_checkpoint != shared.get("checkpoint")
            or base_checkpoint.get("replay_size") != 1_000_000
            or base_checkpoint.get("replay_ptr") != 800_000):
        raise RuntimeError("live shared-base checkpoint differs from its manifest")
    base_signature = validate_base_protocol_signature(
        pair / "shared_base", runtime)
    resume = manifest.get("resume_boundary") or {}
    boundaries: dict[str, dict[str, Any]] = {}
    for branch in ("robust_long", "oracle_direct"):
        run_dir = pair / branch
        final = manifest["final"][branch]
        checkpoint = validate_checkpoint(
            run_dir, expected_iteration=FINAL_ITERATION,
            expected_steps=FINAL_TOTAL_STEPS,
            expected_update_count=FINAL_UPDATE_COUNT)
        recorded_checkpoint = dict(final.get("checkpoint") or {})
        saved_iteration = recorded_checkpoint.pop("saved_iteration", None)
        if (recorded_checkpoint != checkpoint
                or saved_iteration != FINAL_ITERATION
                or checkpoint.get("replay_size") != 1_000_000
                or checkpoint.get("replay_ptr") != 600_000):
            raise RuntimeError(f"live final checkpoint differs: {branch}")
        if final.get("logs") != validate_final_logs(run_dir):
            raise RuntimeError(f"live final logs differ: {branch}")
        signature = validate_protocol_signature(
            run_dir, runtime, expected_start=BASE_NEXT_ITERATION,
            expected_steps_at_start=BASE_TOTAL_STEPS, expected_loaded=True)
        if (final.get("protocol_signature_sha256")
                != sha256_file(run_dir / "logs" / "protocol_signature.json")):
            raise RuntimeError(f"live protocol signature differs: {branch}")
        expected_diff = validate_config_diff(base_signature, signature, branch)
        if (manifest.get("config_diffs") or {}).get(branch) != expected_diff:
            raise RuntimeError(f"recorded config diff is invalid: {branch}")
        prefix = validate_logger_prefix(pair / "shared_base", run_dir)
        if final.get("logger_prefix") != prefix:
            raise RuntimeError(f"recorded logger prefix is invalid: {branch}")
        boundary = validate_boundary_audit(run_dir, branch, base_checkpoint)
        if (resume.get(branch) != boundary
                or final.get("boundary") != boundary):
            raise RuntimeError(f"recorded boundary audit differs: {branch}")
        boundaries[branch] = boundary
    expected_proof = boundary_rollout_proof(
        manifest.get("identity") or {}, boundaries["robust_long"],
        boundaries["oracle_direct"])
    legacy_keys = {
        "physical_rollout_equal", "physical_rollout_sha256",
        "field_sha256_equal",
    }
    for key in legacy_keys:
        if resume.get(key) != expected_proof[key]:
            raise RuntimeError("paired physical rollout summary is invalid")
    # Existing byte-exact v2 manifests predate the richer proof fields.  New
    # manifests record them, while categorical acceptance requires all fields.
    has_rich_proof = "physical_rollout_validation" in resume
    if (expected_proof["physical_rollout_validation"]
            == "categorical_policy_equivalence" or has_rich_proof):
        recorded_proof = {
            key: value for key, value in resume.items()
            if key not in ("robust_long", "oracle_direct")
        }
        if recorded_proof != expected_proof:
            raise RuntimeError("paired physical rollout proof is inconsistent")
    return {"manifest": manifest, "sentinel": sentinel, "sha256": manifest_sha}


def atomic_copy_run(source: Path, destination: Path,
                    expected_manifest: dict[str, Any]) -> dict[str, Any]:
    if destination.exists():
        actual = run_snapshot_manifest(destination)
        if (actual["sha256"] != expected_manifest["sha256"]
                or actual["files"] != expected_manifest["files"]):
            raise RuntimeError(
                f"existing fork destination is not the exact shared snapshot: "
                f"{destination}")
        return actual
    tmp = destination.with_name(
        f".{destination.name}.fork-tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        shutil.copytree(source, tmp, copy_function=shutil.copy2)
        actual = run_snapshot_manifest(tmp)
        if (actual["sha256"] != expected_manifest["sha256"]
                or actual["files"] != expected_manifest["files"]):
            raise RuntimeError(f"atomic fork verification failed for {destination}")
        tmp.rename(destination)
        fsync_dir(destination.parent)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    return run_snapshot_manifest(destination)


def common_training_values(family: str, env: str, seed: int,
                           save_root: Path, run_name: str,
                           policy_variant: str = "direct") -> list[str]:
    variant = POLICY_VARIANTS[policy_variant]
    values = [
        "--algo", "bapr_v3",
        "--env", env,
        "--seed", str(seed),
        "--env_type", "stochastic_mode",
        "--stochastic_mode_family", family,
        "--stochastic_mode_dwell_steps", "500",
        "--stochastic_mode_dwell_distribution", "fixed",
        "--task_num", "4",
        "--test_task_num", "4",
        "--ensemble_size", "5",
        "--hidden_dim", "256",
        "--context_warmup_iters", "0",
        "--bapr_v2_mode", "oracle",
        "--bapr_v2_latent_dim", "4",
        "--bapr_v2_policy_context_source", "oracle_task",
        "--bapr_v2_training_schedule", "teacher_student",
        "--bapr_v2_student_iters", "0",
        "--bapr_v2_context_hidden_dim", "64",
        "--bapr_v2_context_length", "64",
        "--bapr_v2_context_chunks", "8",
        "--bapr_v2_context_burnin", "16",
        "--bapr_v2_min_history", "16",
        "--bapr_v2_policy_mode", str(variant["policy_mode"]),
        "--bapr_v2_policy_gate_init", "6.0",
        "--bapr_v2_switch_rollout_steps", "500",
        "--bapr_v2_paired_calibration_episodes", "0",
        "--bapr_v2_context_dropout", "0.0",
        "--bapr_v2_base_aux_weight", "0.0",
        "--bapr_v2_actor_objective", str(variant["actor_objective"]),
        "--bapr_v2_beta_ood", "0.0",
        "--bapr_v2_reg_weight", "0.0",
        "--bapr_v2_warmstart_conditioned",
        "--bapr_v2_freeze_gate_in_teacher",
        "--bapr_v3_likelihood", "point",
        "--bapr_v3_variance_model", "mode_calibrated",
        "--bapr_v3_context_ensemble_size", "2",
        "--bapr_v3_instant_classifier_weight", "0.0",
        "--bapr_v3_freeze_teacher_after_teacher",
        "--samples_per_iter", str(SAMPLES_PER_ITER),
        "--updates_per_iter", str(UPDATES_PER_ITER),
        "--log_interval", "50",
        "--eval_episodes", "2",
        "--eval_protocol", "stationary",
        "--save_interval", "50",
        "--save_root", str(save_root),
        "--run_name", run_name,
        "--backend", "spring",
        "--resume",
    ]
    if variant["num_experts"] is not None:
        values += [
            "--bapr_v2_num_experts", str(variant["num_experts"]),
            "--bapr_v2_action_deviation_weight",
            str(variant["action_deviation_weight"]),
        ]
    return values


def training_command(family: str, env: str, seed: int, pair: Path,
                     run_name: str, *, max_iters: int, base_iters: int,
                     teacher_iters: int, boundary_audit: bool = False,
                     policy_variant: str = "direct") -> list[str]:
    values = common_training_values(
        family, env, seed, pair, run_name, policy_variant)
    values += [
        "--bapr_v2_base_pretrain_iters", str(base_iters),
        "--bapr_v2_teacher_iters", str(teacher_iters),
        "--max_iters", str(max_iters),
    ]
    if run_name != "shared_base":
        values += ["--min_resume_iteration", str(BASE_NEXT_ITERATION)]
    if boundary_audit:
        values += [
            "--resume_boundary_audit",
            "--resume_boundary_expected_iteration", str(BASE_NEXT_ITERATION),
            "--resume_boundary_expected_total_steps", str(BASE_TOTAL_STEPS),
            "--resume_boundary_expected_update_count", str(BASE_UPDATE_COUNT),
        ]
    return [sys.executable, "-u", "-m", "jax_experiments.train", *values]


def run_train(command: list[str], log_path: Path, source: dict[str, Any],
              runtime: dict[str, Any]) -> None:
    assert_runtime_unchanged(runtime)
    assert_source_unchanged(source)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("TRAIN SUBPROCESS:", " ".join(command), flush=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n[{utc_now()}] command={command!r}\n")
        process = subprocess.Popen(
            command, cwd=ROOT, env=os.environ.copy(), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)
    assert_runtime_unchanged(runtime)
    assert_source_unchanged(source)


def protocol_signature(run_dir: Path) -> dict[str, Any]:
    return read_json(run_dir / "logs" / "protocol_signature.json")


def validate_protocol_signature(run_dir: Path, runtime: dict[str, Any],
                                expected_start: int,
                                expected_steps_at_start: int,
                                expected_loaded: bool) -> dict[str, Any]:
    signature = protocol_signature(run_dir)
    checks = {
        "host": runtime["host"],
        "python": runtime["python_executable"],
        "start_iteration": expected_start,
        "total_steps_at_start": expected_steps_at_start,
        "checkpoint_loaded": expected_loaded,
    }
    for key, value in checks.items():
        if signature.get(key) != value:
            raise RuntimeError(
                f"{run_dir.name} protocol {key}={signature.get(key)!r}, "
                f"expected {value!r}")
    return signature


def validate_base_protocol_signature(
        run_dir: Path, runtime: dict[str, Any]) -> dict[str, Any]:
    """The immutable shared base must be one uninterrupted iter0-699 run."""
    return validate_protocol_signature(
        run_dir, runtime, expected_start=0,
        expected_steps_at_start=0, expected_loaded=False)


def validate_config_diff(base_signature: dict[str, Any],
                         branch_signature: dict[str, Any],
                         branch: str) -> dict[str, Any]:
    base = base_signature.get("config")
    target = branch_signature.get("config")
    if not isinstance(base, dict) or not isinstance(target, dict):
        raise RuntimeError("protocol signatures do not contain config objects")
    changed = {
        key: {"base": base.get(key), "branch": target.get(key)}
        for key in sorted(set(base) | set(target))
        if base.get(key) != target.get(key)
    }
    unexpected = sorted(set(changed) - CONFIG_DIFF_ALLOWLIST)
    if unexpected:
        raise RuntimeError(
            f"{branch} has non-allowlisted config changes: {unexpected}")
    expected_schedule = (
        (1400, 0) if branch == "robust_long" else (700, 700))
    if (target.get("bapr_v2_base_pretrain_iters"),
            target.get("bapr_v2_teacher_iters")) != expected_schedule:
        raise RuntimeError(f"{branch} has the wrong training schedule")
    return {
        "allowlist": sorted(CONFIG_DIFF_ALLOWLIST),
        "changed": changed,
        "unexpected": unexpected,
    }


def _equal_values(left: Any, right: Any) -> bool:
    try:
        import numpy as np

        return bool(np.array_equal(
            np.asarray(left), np.asarray(right), equal_nan=True))
    except (TypeError, ValueError):
        return repr(left) == repr(right)


def validate_logger_prefix(base_dir: Path, branch_dir: Path) -> dict[str, Any]:
    base_state = read_pickle(base_dir / "checkpoints" / "train_state.pkl")
    branch_state = read_pickle(branch_dir / "checkpoints" / "train_state.pkl")
    base_logger = base_state.get("logger_data", {})
    branch_logger = branch_state.get("logger_data", {})
    if not isinstance(base_logger, dict) or not isinstance(branch_logger, dict):
        raise RuntimeError("logger_data is not a dictionary")
    for key, values in base_logger.items():
        other = branch_logger.get(key)
        if other is None or len(other) < len(values):
            raise RuntimeError(f"{branch_dir.name} logger prefix missing key {key}")
        for index, value in enumerate(values):
            if not _equal_values(value, other[index]):
                raise RuntimeError(
                    f"{branch_dir.name} logger prefix differs at {key}[{index}]")

    import numpy as np

    npy_checked = 0
    for base_path in sorted((base_dir / "logs").glob("*.npy")):
        branch_path = branch_dir / "logs" / base_path.name
        if not branch_path.is_file():
            raise RuntimeError(f"missing branch log {branch_path}")
        base_array = np.load(base_path, allow_pickle=True)
        branch_array = np.load(branch_path, allow_pickle=True)
        if branch_array.shape[:1] < base_array.shape[:1]:
            raise RuntimeError(f"short branch log {branch_path}")
        if not np.array_equal(
                base_array, branch_array[:len(base_array)], equal_nan=True):
            raise RuntimeError(f"npy logger prefix differs: {branch_path}")
        npy_checked += 1
    diagnostic = "python_runtime_diagnostics.log"
    base_diag = (base_dir / "logs" / diagnostic).read_bytes()
    branch_diag = (branch_dir / "logs" / diagnostic).read_bytes()
    if not branch_diag.startswith(base_diag):
        raise RuntimeError(f"diagnostic log prefix differs: {branch_dir}")
    return {
        "logger_keys_checked": len(base_logger),
        "logger_prefix_pickle_sha256": hashlib.sha256(
            pickle.dumps(base_logger, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest(),
        "npy_files_checked": npy_checked,
        "diagnostic_prefix_sha256": hashlib.sha256(base_diag).hexdigest(),
    }


def validate_boundary_audit(run_dir: Path, branch: str,
                            base_checkpoint: dict[str, Any]) -> dict[str, Any]:
    path = run_dir / "logs" / BOUNDARY_AUDIT_NAME
    audit = read_json(path)
    expected = {
        "schema": "bapr.resume-boundary-audit.v1",
        "run_name": branch,
        "seed": 0,
        "iteration": BASE_NEXT_ITERATION,
        "total_steps_before_rollout": BASE_TOTAL_STEPS,
        "replay_size_before_rollout": base_checkpoint["replay_size"],
        "agent_update_count_before_rollout": BASE_UPDATE_COUNT,
        "training_stage": "robust" if branch == "robust_long" else "teacher",
        "rollout_context_source": 0 if branch == "robust_long" else 1,
        "controller_update_flags": (
            [True, False, True] if branch == "robust_long"
            else [False, True, True]),
        "train_policy_gate": False,
        "conditioned_warmstarted": branch == "oracle_direct",
    }
    for key, value in expected.items():
        if audit.get(key) != value:
            raise RuntimeError(
                f"{branch} boundary {key}={audit.get(key)!r}, expected {value!r}")
    if audit.get("semantics") != SEMANTICS:
        raise RuntimeError(f"{branch} boundary audit overclaims continuation")
    physical = audit.get("physical_rollout")
    if not isinstance(physical, dict) or not physical.get("sha256"):
        raise RuntimeError(f"{branch} boundary audit lacks physical rollout hashes")
    if int(physical.get("transitions", -1)) != SAMPLES_PER_ITER:
        raise RuntimeError(f"{branch} boundary rollout has wrong length")
    required_fields = ["obs", "act", "rew", "next_obs", "done", "task_id"]

    def valid_sha256(value: Any) -> bool:
        return (isinstance(value, str) and len(value) == 64
                and all(character in "0123456789abcdef" for character in value))

    if (physical.get("fields") != required_fields
            or physical.get("finite") is not True
            or physical.get("excludes_context_by_design") is not True
            or not valid_sha256(physical.get("sha256"))):
        raise RuntimeError(f"{branch} physical rollout metadata is invalid")
    field_hashes = physical.get("field_sha256")
    shapes = physical.get("validated_shapes")
    if (not isinstance(field_hashes, dict)
            or set(field_hashes) != set(required_fields)
            or not all(valid_sha256(field_hashes[name]) for name in required_fields)
            or not isinstance(shapes, dict)
            or set(shapes) != set(required_fields)):
        raise RuntimeError(f"{branch} physical rollout fields are invalid")
    if any(
            not isinstance(shapes[name], list) or not shapes[name]
            or int(shapes[name][0]) != SAMPLES_PER_ITER
            for name in required_fields):
        raise RuntimeError(f"{branch} physical rollout shapes are invalid")
    if (len(shapes["obs"]) != 2 or int(shapes["obs"][1]) <= 0
            or shapes["next_obs"] != shapes["obs"]
            or len(shapes["act"]) != 2 or int(shapes["act"][1]) <= 0
            or shapes["rew"] != [SAMPLES_PER_ITER, 1]
            or shapes["done"] != [SAMPLES_PER_ITER, 1]
            or shapes["task_id"] != [SAMPLES_PER_ITER]):
        raise RuntimeError(f"{branch} physical rollout shape contract is invalid")
    equivalence = audit.get("policy_equivalence") or {}
    tolerance = float(equivalence.get("tolerance", 1e-6))
    mean_difference = float(
        equivalence.get("max_abs_mean_diff", float("inf")))
    log_std_difference = float(
        equivalence.get("max_abs_log_std_diff", float("inf")))
    expected_contexts = (
        ["robust_zero"] if branch == "robust_long"
        else [f"oracle_task_{index}" for index in range(4)])
    if (tolerance != 1e-6
            or not equivalence.get("pass") or not equivalence.get("finite")
            or mean_difference > min(tolerance, 1e-6)
            or log_std_difference > min(tolerance, 1e-6)
            or equivalence.get("tested_rollout_context_source")
            != expected["rollout_context_source"]
            or equivalence.get("tested_contexts") != expected_contexts
            or int(equivalence.get("observations", -1)) != 32
            or int(equivalence.get("task_latents", -1)) != 4):
        raise RuntimeError(f"{branch} boundary policy equivalence failed")
    result = dict(audit)
    result["file"] = str(path.relative_to(run_dir.parent))
    result["file_sha256"] = sha256_file(path)
    return result


def boundary_rollout_proof(identity: dict[str, Any],
                           robust_boundary: dict[str, Any],
                           oracle_boundary: dict[str, Any]) -> dict[str, Any]:
    """Validate the common-restart rollout evidence for one controller pair.

    The original direct actor follows the same arithmetic graph in both arms,
    so its full rollout remains byte-exact.  A categorical oracle evaluates a
    one-hot expert reduction while the robust arm evaluates the base path.
    Even with exactly copied parameters, long Ant trajectories can diverge
    after floating-point roundoff.  That case is accepted only with exact
    pre-rollout policy canaries and an identical mode schedule.
    """
    robust_physical = robust_boundary["physical_rollout"]
    oracle_physical = oracle_boundary["physical_rollout"]
    rollout_equal = (
        robust_physical.get("sha256") == oracle_physical.get("sha256"))
    fields_equal = (
        robust_physical.get("field_sha256")
        == oracle_physical.get("field_sha256"))
    task_hash = (robust_physical.get("field_sha256") or {}).get("task_id")
    task_schedule_equal = bool(
        task_hash
        and task_hash
        == (oracle_physical.get("field_sha256") or {}).get("task_id"))

    common = {
        "physical_rollout_equal": bool(rollout_equal),
        # Retained for compatibility.  In the categorical proof this is the
        # robust reference hash, not a claim that both trajectories match.
        "physical_rollout_sha256": robust_physical["sha256"],
        "field_sha256_equal": bool(fields_equal),
    }
    if rollout_equal and fields_equal:
        return {
            **common,
            "physical_rollout_validation": "byte_exact",
            "physical_rollout_accepted": True,
            "robust_physical_rollout_sha256": robust_physical["sha256"],
            "oracle_physical_rollout_sha256": oracle_physical["sha256"],
            "task_schedule_equal": task_schedule_equal,
            "task_schedule_sha256": task_hash,
        }

    variant_name = str(identity.get("policy_variant") or "direct")
    variant = POLICY_VARIANTS.get(variant_name)
    if not variant or variant.get("policy_mode") != "categorical_expert":
        raise RuntimeError(
            "PROTOCOL_INTEGRITY: the two common-restart physical rollouts "
            "differ at iter 700")
    if not task_schedule_equal:
        raise RuntimeError(
            "PROTOCOL_INTEGRITY: categorical common-restart mode schedules "
            "differ at iter 700")

    canary_max = {"mean": 0.0, "log_std": 0.0}
    for branch, boundary in (
            ("robust_long", robust_boundary),
            ("oracle_direct", oracle_boundary)):
        equivalence = boundary.get("policy_equivalence") or {}
        mean_difference = float(
            equivalence.get("max_abs_mean_diff", float("inf")))
        log_std_difference = float(
            equivalence.get("max_abs_log_std_diff", float("inf")))
        if (equivalence.get("pass") is not True
                or equivalence.get("finite") is not True
                or mean_difference != 0.0
                or log_std_difference != 0.0):
            raise RuntimeError(
                "PROTOCOL_INTEGRITY: categorical boundary policy canary "
                f"is not exactly equivalent for {branch}")
        canary_max["mean"] = max(canary_max["mean"], mean_difference)
        canary_max["log_std"] = max(
            canary_max["log_std"], log_std_difference)
    if (robust_boundary.get("conditioned_warmstarted") is not False
            or oracle_boundary.get("conditioned_warmstarted") is not True):
        raise RuntimeError(
            "PROTOCOL_INTEGRITY: categorical boundary warm-start evidence "
            "is invalid")

    return {
        **common,
        "physical_rollout_validation": "categorical_policy_equivalence",
        "physical_rollout_accepted": True,
        "robust_physical_rollout_sha256": robust_physical["sha256"],
        "oracle_physical_rollout_sha256": oracle_physical["sha256"],
        "task_schedule_equal": True,
        "task_schedule_sha256": task_hash,
        "policy_canary_exact": True,
        "policy_canary_max_abs_mean_diff": canary_max["mean"],
        "policy_canary_max_abs_log_std_diff": canary_max["log_std"],
        "explanation": (
            "The robust base and each one-hot categorical expert are exactly "
            "equal in the pre-rollout policy canary. Their mode schedule is "
            "identical; the long physical trajectories diverge because the "
            "base and categorical reduction use different floating-point "
            "graphs."),
    }


def branch_checkpoint_iteration(run_dir: Path) -> int:
    path = run_dir / "checkpoints" / "train_state.pkl"
    if not path.is_file():
        return -1
    try:
        return int(read_pickle(path)["iteration"])
    except Exception as exc:
        raise RuntimeError(f"cannot inspect branch checkpoint {path}: {exc}") from exc


def discard_protocol_branch(run_dir: Path) -> None:
    """Remove only a protocol-owned partial branch before an exact re-fork."""
    if not run_dir.exists():
        return
    discarded = run_dir.with_name(
        f".{run_dir.name}.discarded-{os.getpid()}-{uuid.uuid4().hex}")
    run_dir.rename(discarded)
    fsync_dir(run_dir.parent)
    shutil.rmtree(discarded)


def update_state(state_path: Path, state: dict[str, Any], phase: str,
                 **extra: Any) -> None:
    state = dict(state)
    state.update(extra)
    state["phase"] = phase
    state["updated_at"] = utc_now()
    atomic_write_pickle(state_path, state)


def pair_identity(family: str, env: str, seed: int,
                  policy_variant: str) -> dict[str, Any]:
    identity = {"family": family, "env": env, "seed": seed}
    if policy_variant != "direct":
        identity["policy_variant"] = policy_variant
    return identity


def initialize_or_validate_pair(pair: Path, family: str, env: str,
                                seed: int,
                                policy_variant: str = "direct") -> dict[str, Any]:
    state_path = pair / STATE_NAME
    if state_path.exists():
        state = read_pickle(state_path)
        if not isinstance(state, dict) or state.get("schema") != SCHEMA:
            raise RuntimeError(f"invalid protocol state: {state_path}")
        expected_identity = pair_identity(
            family, env, seed, policy_variant)
        if state.get("identity") != expected_identity:
            raise RuntimeError("pair identity does not match existing protocol state")
        assert_runtime_unchanged(state["runtime"])
        assert_source_unchanged(state["source"])
        return state

    allowed = {".protocol.lock"}
    unexpected = [path.name for path in pair.iterdir() if path.name not in allowed]
    if unexpected:
        raise RuntimeError(
            "refusing to adopt a pre-existing pair without protocol state: "
            f"{unexpected}")
    source = current_source_manifest()
    runtime = runtime_identity()
    state = {
        "schema": SCHEMA,
        "semantics": SEMANTICS,
        "identity": pair_identity(family, env, seed, policy_variant),
        "runtime": runtime,
        "source": source,
        "phase": "initialized",
        "created_at": utc_now(),
        "updated_at": utc_now(),
    }
    atomic_write_pickle(state_path, state)
    return state


def ensure_source_provenance(pair: Path, state: dict[str, Any]) -> dict[str, Any]:
    provenance = pair / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    source_manifest_path = pair / SOURCE_MANIFEST_REL
    if source_manifest_path.exists():
        recorded = read_json(source_manifest_path)
        if (recorded.get("sha256") != state["source"]["sha256"]
                or recorded.get("files") != state["source"]["files"]):
            raise RuntimeError("persisted source manifest does not match pair state")
    else:
        atomic_write_json(source_manifest_path, state["source"])
    archive_path = pair / SOURCE_ARCHIVE_REL
    create_source_archive(archive_path, state["source"])
    expected_archive_sha = (
        (state.get("source_archive") or {}).get("sha256"))
    archive_info = validate_source_archive(
        archive_path, state["source"], expected_archive_sha)
    state = dict(state)
    state["source_archive"] = {
        "path": str(SOURCE_ARCHIVE_REL),
        **archive_info,
    }
    atomic_write_pickle(pair / STATE_NAME, state)
    return state


def ensure_base(pair: Path, family: str, env: str, seed: int,
                state: dict[str, Any],
                policy_variant: str = "direct") -> tuple[dict[str, Any], dict[str, Any]]:
    base_dir = pair / "shared_base"
    manifest_path = pair / "provenance" / "shared_base_snapshot_manifest.json"
    pinned_snapshot = state.get("shared_base_snapshot_sha256")
    pinned_manifest = state.get("shared_base_manifest_sha256")
    if (pinned_snapshot is None) != (pinned_manifest is None):
        raise RuntimeError("protocol state has an incomplete shared-base pin")
    iteration = branch_checkpoint_iteration(base_dir)
    if iteration > BASE_FINAL_ITERATION:
        raise RuntimeError("shared base advanced beyond iter 699")
    if ((pinned_snapshot is not None or manifest_path.exists())
            and iteration != BASE_FINAL_ITERATION):
        raise RuntimeError(
            "immutable shared base is missing or no longer at iter 699")
    if iteration != BASE_FINAL_ITERATION:
        # The live environment/RNG state is not in the checkpoint.  A partial
        # base resume would therefore alter the supposedly common source.
        # Restart base training from iter0 instead of silently continuing it.
        if base_dir.exists():
            discard_protocol_branch(base_dir)
        command = training_command(
            family, env, seed, pair, "shared_base", max_iters=700,
            base_iters=1400, teacher_iters=0,
            policy_variant=policy_variant)
        run_train(
            command, pair / "provenance" / "shared_base_train.log",
            state["source"], state["runtime"])
    checkpoint = validate_checkpoint(
        base_dir, expected_iteration=BASE_FINAL_ITERATION,
        expected_steps=BASE_TOTAL_STEPS,
        expected_update_count=BASE_UPDATE_COUNT)
    base_signature = validate_base_protocol_signature(base_dir, state["runtime"])
    snapshot = run_snapshot_manifest(base_dir)
    if pinned_snapshot is not None and pinned_snapshot != snapshot["sha256"]:
        raise RuntimeError("immutable shared-base snapshot changed on re-entry")
    candidate_manifest = {
        "schema": "bapr.shared-base-snapshot.v1",
        "created_at": utc_now(),
        "checkpoint": checkpoint,
        "protocol_start": {
            "checkpoint_loaded": bool(base_signature["checkpoint_loaded"]),
            "start_iteration": int(base_signature["start_iteration"]),
            "total_steps_at_start": int(base_signature["total_steps_at_start"]),
        },
        "snapshot_sha256": snapshot["sha256"],
        "files": snapshot["files"],
    }
    if manifest_path.exists():
        base_manifest = read_json(manifest_path)
        for key in ("schema", "checkpoint", "protocol_start",
                    "snapshot_sha256", "files"):
            if base_manifest.get(key) != candidate_manifest[key]:
                raise RuntimeError(
                    f"immutable shared-base manifest changed at {key}")
        if (pinned_manifest is not None
                and canonical_sha256(base_manifest) != pinned_manifest):
            raise RuntimeError("immutable shared-base manifest hash changed")
    else:
        if pinned_snapshot is not None:
            raise RuntimeError("pinned shared-base manifest is missing")
        base_manifest = candidate_manifest
        atomic_write_json(manifest_path, base_manifest)
    manifest_sha = canonical_sha256(base_manifest)
    if pinned_snapshot is None:
        update_state(
            pair / STATE_NAME, state, "base_complete",
            shared_base_snapshot_sha256=snapshot["sha256"],
            shared_base_manifest_sha256=manifest_sha)
    elif pinned_manifest != manifest_sha:
        raise RuntimeError("immutable shared-base manifest changed on re-entry")
    return checkpoint, base_manifest


def ensure_forks(pair: Path, base_manifest: dict[str, Any],
                 state: dict[str, Any]) -> dict[str, Any]:
    source_manifest = {
        "sha256": base_manifest["snapshot_sha256"],
        "files": base_manifest["files"],
    }
    fork_manifest_path = pair / "provenance" / "fork_manifest.json"
    recorded = read_json(fork_manifest_path) if fork_manifest_path.exists() else None
    if recorded is not None and (
            recorded.get("shared_base_snapshot_sha256")
            != base_manifest["snapshot_sha256"]):
        raise RuntimeError("recorded fork source differs from shared_base")
    forks: dict[str, Any] = {}
    for branch in ("robust_long", "oracle_direct"):
        destination = pair / branch
        iteration = branch_checkpoint_iteration(destination)
        if iteration == BASE_FINAL_ITERATION:
            try:
                copied = atomic_copy_run(
                    pair / "shared_base", destination, source_manifest)
            except RuntimeError:
                # A killed iter-700 attempt can leave the immutable audit/log
                # beside the still-iter699 checkpoint.  Never reuse it.
                discard_protocol_branch(destination)
                copied = atomic_copy_run(
                    pair / "shared_base", destination, source_manifest)
        elif iteration < BASE_FINAL_ITERATION:
            if destination.exists():
                discard_protocol_branch(destination)
            copied = atomic_copy_run(
                pair / "shared_base", destination, source_manifest)
        else:
            if recorded is None:
                raise RuntimeError(
                    f"{branch} advanced without an atomic-fork manifest")
            prior = (recorded.get("forks") or {}).get(branch, {})
            if (prior.get("source_snapshot_sha256")
                    != base_manifest["snapshot_sha256"]):
                raise RuntimeError(
                    f"{branch} does not prove the shared fork source")
            copied = source_manifest
        forks[branch] = {
            "run_name": branch,
            "source_snapshot_sha256": copied["sha256"],
            "source_files_verified": len(copied["files"]),
            "source_checkpoint_next_iteration": BASE_NEXT_ITERATION,
            "source_total_steps": BASE_TOTAL_STEPS,
            "start_iteration": BASE_NEXT_ITERATION,
            "start_total_steps": BASE_TOTAL_STEPS,
            "runtime_sha256": state["runtime"]["sha256"],
            "source_sha256": state["source"]["sha256"],
        }
    if len({value["source_snapshot_sha256"] for value in forks.values()}) != 1:
        raise RuntimeError("fork source digests are not identical")
    atomic_write_json(fork_manifest_path, {
        "schema": "bapr.atomic-fork.v1",
        "semantics": SEMANTICS,
        "shared_base_snapshot_sha256": base_manifest["snapshot_sha256"],
        "forks": forks,
    })
    update_state(
        pair / STATE_NAME, state, "fork_complete",
        shared_base_snapshot_sha256=base_manifest["snapshot_sha256"])
    return forks


def ensure_branch(pair: Path, family: str, env: str, seed: int,
                  branch: str, state: dict[str, Any],
                  base_checkpoint: dict[str, Any],
                  policy_variant: str = "direct") -> dict[str, Any]:
    run_dir = pair / branch
    iteration = branch_checkpoint_iteration(run_dir)
    audit_path = run_dir / "logs" / BOUNDARY_AUDIT_NAME
    if iteration < BASE_FINAL_ITERATION:
        raise RuntimeError(f"{branch} is older than the shared fork source")
    if BASE_FINAL_ITERATION < iteration < FINAL_ITERATION:
        # Branch-local resumes would reconstruct a different live environment
        # state and could overwrite/miss the iter-700 canary.  Throw the partial
        # branch away and restart it from the immutable shared checkpoint.
        discard_protocol_branch(run_dir)
        source_manifest = {
            "sha256": state["shared_base_snapshot_sha256"],
            "files": read_json(
                pair / "provenance" / "shared_base_snapshot_manifest.json")[
                    "files"],
        }
        atomic_copy_run(pair / "shared_base", run_dir, source_manifest)
        iteration = BASE_FINAL_ITERATION
        audit_path = run_dir / "logs" / BOUNDARY_AUDIT_NAME
    if iteration == BASE_FINAL_ITERATION and audit_path.exists():
        # The immutable artifact means a killed attempt already crossed the
        # boundary even if its next checkpoint was not saved.
        discard_protocol_branch(run_dir)
        source_manifest = {
            "sha256": state["shared_base_snapshot_sha256"],
            "files": read_json(
                pair / "provenance" / "shared_base_snapshot_manifest.json")[
                    "files"],
        }
        atomic_copy_run(pair / "shared_base", run_dir, source_manifest)
        iteration = BASE_FINAL_ITERATION
        audit_path = run_dir / "logs" / BOUNDARY_AUDIT_NAME
    if iteration > FINAL_ITERATION:
        raise RuntimeError(f"{branch} advanced beyond the preregistered budget")
    if iteration < FINAL_ITERATION:
        if branch == "robust_long":
            base_iters, teacher_iters = 1400, 0
        else:
            base_iters, teacher_iters = 700, 700
        command = training_command(
            family, env, seed, pair, branch, max_iters=1400,
            base_iters=base_iters, teacher_iters=teacher_iters,
            boundary_audit=(iteration == BASE_FINAL_ITERATION),
            policy_variant=policy_variant)
        run_train(
            command, pair / "provenance" / f"{branch}_train.log",
            state["source"], state["runtime"])
    checkpoint = validate_checkpoint(
        run_dir, expected_iteration=FINAL_ITERATION,
        expected_steps=FINAL_TOTAL_STEPS,
        expected_update_count=FINAL_UPDATE_COUNT)
    logs = validate_final_logs(run_dir)
    signature = validate_protocol_signature(
        run_dir, state["runtime"], expected_start=BASE_NEXT_ITERATION,
        expected_steps_at_start=BASE_TOTAL_STEPS, expected_loaded=True)
    prefix = validate_logger_prefix(pair / "shared_base", run_dir)
    boundary = validate_boundary_audit(run_dir, branch, base_checkpoint)
    audit_copy = pair / "provenance" / f"{branch}_{BOUNDARY_AUDIT_NAME}"
    atomic_write_bytes(audit_copy, audit_path.read_bytes())
    if sha256_file(audit_copy) != boundary["file_sha256"]:
        raise RuntimeError(f"failed to preserve {branch} boundary audit")
    update_state(pair / STATE_NAME, state, f"{branch}_complete")
    final_files = {}
    for relative in (*CHECKPOINT_FILES, "logs/protocol_signature.json",
                     f"logs/{BOUNDARY_AUDIT_NAME}"):
        path = run_dir / relative
        final_files[relative] = {
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }
    return {
        "checkpoint": checkpoint,
        "files": final_files,
        "logs": logs,
        "protocol_signature_sha256": sha256_file(
            run_dir / "logs" / "protocol_signature.json"),
        "protocol_signature": signature,
        "logger_prefix": prefix,
        "boundary": boundary,
        "boundary_copy": {
            "path": str(audit_copy.relative_to(pair)),
            "sha256": sha256_file(audit_copy),
        },
    }


def finalize_pair(pair: Path, state: dict[str, Any],
                  base_checkpoint: dict[str, Any],
                  base_manifest: dict[str, Any], forks: dict[str, Any],
                  robust: dict[str, Any], oracle: dict[str, Any],
                  finalization: dict[str, Any] | None = None) -> dict[str, Any]:
    robust_boundary = robust["boundary"]
    oracle_boundary = oracle["boundary"]
    rollout_proof = boundary_rollout_proof(
        state["identity"], robust_boundary, oracle_boundary)

    base_signature = protocol_signature(pair / "shared_base")
    config_diffs = {
        "robust_long": validate_config_diff(
            base_signature, robust["protocol_signature"], "robust_long"),
        "oracle_direct": validate_config_diff(
            base_signature, oracle["protocol_signature"], "oracle_direct"),
    }
    # Avoid duplicating the full config/tasks in the authoritative manifest.
    robust = dict(robust)
    oracle = dict(oracle)
    robust.pop("protocol_signature")
    oracle.pop("protocol_signature")
    robust["checkpoint"]["saved_iteration"] = robust["checkpoint"]["iteration"]
    oracle["checkpoint"]["saved_iteration"] = oracle["checkpoint"]["iteration"]
    manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "semantics": SEMANTICS,
        "continuation_claim": False,
        "limitation": (
            "Both branches restore the same full training checkpoint and use "
            "the same seeded process environment. Live environment and all "
            "process RNG state are not checkpointed, so this is a controlled "
            "common restart rather than an exact trajectory continuation."),
        "identity": state["identity"],
        "completed_at": utc_now(),
        "runtime": state["runtime"],
        "source": {
            "manifest_path": str(SOURCE_MANIFEST_REL),
            "sha256": state["source"]["sha256"],
            "snapshot_sha256": state["source"]["sha256"],
            "files": state["source"]["files"],
            "archive": state["source_archive"],
        },
        "shared_base": {
            "run_name": "shared_base",
            "checkpoint": base_checkpoint,
            "snapshot_sha256": base_manifest["snapshot_sha256"],
            "files": base_manifest["files"],
            "manifest_sha256": canonical_sha256(base_manifest),
        },
        "forks": forks,
        "config_diffs": config_diffs,
        "resume_boundary": {
            "robust_long": robust_boundary,
            "oracle_direct": oracle_boundary,
            **rollout_proof,
        },
        "final": {
            "robust_long": robust,
            "oracle_direct": oracle,
        },
    }
    if finalization is not None:
        manifest["finalization"] = finalization
    manifest_path = pair / PAIR_MANIFEST_REL
    atomic_write_json(manifest_path, manifest)
    manifest_file_sha = sha256_file(manifest_path)
    sentinel = {
        "schema": "bapr.pair-checkpoint-complete.v1",
        "status": "complete",
        "pair_manifest": str(PAIR_MANIFEST_REL),
        "pair_manifest_sha256": manifest_file_sha,
        "shared_base_snapshot_sha256": base_manifest["snapshot_sha256"],
        "final_next_iteration": FINAL_NEXT_ITERATION,
        "final_total_steps": FINAL_TOTAL_STEPS,
        "completed_at": manifest["completed_at"],
    }
    atomic_write_pickle(pair / COMPLETE_SENTINEL_NAME, sentinel)
    try:
        validated = validate_complete_artifacts(pair)
    except Exception:
        # Completion is authoritative only after deep validation.  Do not
        # strand a syntactically complete manifest/sentinel pair when the
        # fail-closed validator rejects it.
        for path in (pair / COMPLETE_SENTINEL_NAME, manifest_path):
            if path.exists():
                path.unlink()
        fsync_dir(pair)
        raise
    update_state(
        pair / STATE_NAME, state, "complete",
        pair_manifest_sha256=validated["sha256"])
    return manifest


def completed_pair_on_reentry(
        pair: Path, state: dict[str, Any]) -> dict[str, Any] | None:
    """Return a valid completed pair without rewriting authoritative artifacts."""
    manifest_exists = (pair / PAIR_MANIFEST_REL).is_file()
    sentinel_exists = (pair / COMPLETE_SENTINEL_NAME).is_file()
    state_claims_complete = (
        state.get("phase") == "complete"
        or state.get("pair_manifest_sha256") is not None)
    if not (manifest_exists or sentinel_exists or state_claims_complete):
        return None
    if not manifest_exists or not sentinel_exists:
        raise RuntimeError(
            "partial/corrupt completion artifacts exist; refusing to rewrite them")
    validated = validate_complete_artifacts(pair)
    manifest = validated["manifest"]
    source = manifest.get("source") or {}
    shared = manifest.get("shared_base") or {}
    if (manifest.get("identity") != state.get("identity")
            or manifest.get("runtime") != state.get("runtime")
            or source.get("sha256") != (state.get("source") or {}).get("sha256")
            or source.get("files") != (state.get("source") or {}).get("files")
            or shared.get("snapshot_sha256")
            != state.get("shared_base_snapshot_sha256")
            or shared.get("manifest_sha256")
            != state.get("shared_base_manifest_sha256")):
        raise RuntimeError("complete artifacts differ from immutable pair state")
    recorded_sha = state.get("pair_manifest_sha256")
    if recorded_sha is not None and recorded_sha != validated["sha256"]:
        raise RuntimeError("complete manifest hash differs from protocol state")
    if state.get("phase") == "complete":
        if recorded_sha is None:
            raise RuntimeError("complete protocol state has no manifest hash")
    else:
        # Crash-safe adoption window: pair manifest and sentinel were both
        # committed, but the final state update did not reach disk.
        update_state(
            pair / STATE_NAME, state, "complete",
            pair_manifest_sha256=validated["sha256"])
    return manifest


def finalize_existing_protocol(args: argparse.Namespace) -> dict[str, Any]:
    """Finalize an explicitly authorized completed pair without training.

    This validator-overlay path covers either categorical pairs whose immutable
    source used the obsolete byte-exact rollout check, or direct stochastic
    event pairs whose immutable source omitted their family names from the
    final identity allowlist. It validates the source archive, original
    runtime, shared fork, final checkpoints, logs, and boundary proof before
    writing completion artifacts. It never enters a training subprocess.
    """
    if not args.resume:
        raise RuntimeError("--finalize-existing requires --resume")
    variant = POLICY_VARIANTS[args.policy_variant]
    categorical = variant.get("policy_mode") == "categorical_expert"
    stochastic_identity_overlay = (
        args.policy_variant == "direct"
        and args.family in STOCHASTIC_EVENT_FAMILIES)
    if not (categorical or stochastic_identity_overlay):
        raise RuntimeError(
            "--finalize-existing is restricted to categorical expert pairs "
            "or direct packet-loss/burst-torque pairs")
    pair = pair_dir(args.family, args.env, args.seed, args.save_root)
    pair.mkdir(parents=True, exist_ok=True)
    lock_path = pair / ".protocol.lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another runner owns pair lock: {pair}") from exc

        state = read_pickle(pair / STATE_NAME)
        expected_identity = pair_identity(
            args.family, args.env, args.seed, args.policy_variant)
        if (not isinstance(state, dict) or state.get("schema") != SCHEMA
                or state.get("identity") != expected_identity):
            raise RuntimeError(
                "PROTOCOL_INTEGRITY: finalize-only pair state/identity is invalid")
        assert_runtime_unchanged(state["runtime"])

        manifest_exists = (pair / PAIR_MANIFEST_REL).is_file()
        sentinel_exists = (pair / COMPLETE_SENTINEL_NAME).is_file()
        if manifest_exists or sentinel_exists:
            if not (manifest_exists and sentinel_exists):
                raise RuntimeError(
                    "PROTOCOL_INTEGRITY: partial completion artifacts exist")
            complete = completed_pair_on_reentry(pair, state)
            assert complete is not None
            proof = complete["resume_boundary"]
            print(
                f"PAIR ALREADY COMPLETE: {pair} finalization=FINALIZE-ONLY "
                f"validation={proof.get('physical_rollout_validation', 'byte_exact')}",
                flush=True)
            return complete

        persisted_source = read_json(pair / SOURCE_MANIFEST_REL)
        if persisted_source != state.get("source"):
            raise RuntimeError(
                "PROTOCOL_INTEGRITY: source manifest differs from pair state")
        archive_state = state.get("source_archive") or {}
        if archive_state.get("path") != str(SOURCE_ARCHIVE_REL):
            raise RuntimeError(
                "PROTOCOL_INTEGRITY: source archive identity is invalid")
        validate_source_archive(
            pair / SOURCE_ARCHIVE_REL, state["source"],
            archive_state.get("sha256"))

        required_iterations = {
            "shared_base": BASE_FINAL_ITERATION,
            "robust_long": FINAL_ITERATION,
            "oracle_direct": FINAL_ITERATION,
        }
        for run_name, expected_iteration in required_iterations.items():
            actual = branch_checkpoint_iteration(pair / run_name)
            if actual != expected_iteration:
                raise RuntimeError(
                    "PROTOCOL_INTEGRITY: finalize-only refuses training; "
                    f"{run_name} is iter {actual}, expected {expected_iteration}")

        base_checkpoint, base_manifest = ensure_base(
            pair, args.family, args.env, args.seed, state,
            args.policy_variant)
        state = read_pickle(pair / STATE_NAME)
        forks = ensure_forks(pair, base_manifest, state)
        state = read_pickle(pair / STATE_NAME)
        robust = ensure_branch(
            pair, args.family, args.env, args.seed, "robust_long", state,
            base_checkpoint, args.policy_variant)
        state = read_pickle(pair / STATE_NAME)
        oracle = ensure_branch(
            pair, args.family, args.env, args.seed, "oracle_direct", state,
            base_checkpoint, args.policy_variant)
        state = read_pickle(pair / STATE_NAME)
        finalizer_path = Path(__file__).resolve()
        finalization_mode = (
            "posthoc-categorical-boundary-validator-v1" if categorical
            else "posthoc-stochastic-family-identity-validator-v1")
        finalization = {
            "mode": finalization_mode,
            "training_reentered": False,
            "original_training_source_sha256": state["source"]["sha256"],
            "original_source_archive_sha256": archive_state["sha256"],
            "validator_file": str(finalizer_path.relative_to(ROOT)),
            "validator_file_sha256": sha256_file(finalizer_path),
            "reason": (
                "Replace the obsolete byte-exact long-rollout requirement "
                "with the fail-closed categorical policy-equivalence proof."
                if categorical else
                "Accept the preregistered packet-loss/burst-torque family "
                "identity after validating the immutable completed pair."),
        }
        manifest = finalize_pair(
            pair, state, base_checkpoint, base_manifest, forks, robust, oracle,
            finalization=finalization)
        proof = manifest["resume_boundary"]
        print(
            f"PAIR COMPLETE: {pair} finalization=FINALIZE-ONLY "
            f"validation={proof['physical_rollout_validation']} "
            f"robust_sha256={proof['robust_physical_rollout_sha256']} "
            f"oracle_sha256={proof['oracle_physical_rollout_sha256']}",
            flush=True)
        return manifest


def run_protocol(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed != 0:
        raise RuntimeError("v2 preregistration fixes seed=0")
    pair = pair_dir(args.family, args.env, args.seed, args.save_root)
    pair.mkdir(parents=True, exist_ok=True)
    lock_path = pair / ".protocol.lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another runner owns pair lock: {pair}") from exc
        state = initialize_or_validate_pair(
            pair, args.family, args.env, args.seed, args.policy_variant)
        complete = completed_pair_on_reentry(pair, state)
        if complete is not None:
            proof = complete["resume_boundary"]
            print(
                f"PAIR ALREADY COMPLETE: {pair} "
                f"physical_validation="
                f"{proof.get('physical_rollout_validation', 'byte_exact')} "
                f"physical_sha256={proof['physical_rollout_sha256']}",
                flush=True)
            return complete

        state = ensure_source_provenance(pair, state)
        assert_runtime_unchanged(state["runtime"])
        assert_source_unchanged(state["source"])

        base_checkpoint, base_manifest = ensure_base(
            pair, args.family, args.env, args.seed, state,
            args.policy_variant)
        state = read_pickle(pair / STATE_NAME)
        forks = ensure_forks(pair, base_manifest, state)
        state = read_pickle(pair / STATE_NAME)
        robust = ensure_branch(
            pair, args.family, args.env, args.seed, "robust_long", state,
            base_checkpoint, args.policy_variant)
        state = read_pickle(pair / STATE_NAME)
        oracle = ensure_branch(
            pair, args.family, args.env, args.seed, "oracle_direct", state,
            base_checkpoint, args.policy_variant)
        state = read_pickle(pair / STATE_NAME)
        manifest = finalize_pair(
            pair, state, base_checkpoint, base_manifest, forks, robust, oracle)
        proof = manifest["resume_boundary"]
        print(
            f"PAIR COMPLETE: {pair} "
            f"physical_validation={proof['physical_rollout_validation']} "
            f"physical_sha256={proof['physical_rollout_sha256']}",
            flush=True)
        return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=FAMILIES)
    parser.add_argument("--env", required=True, choices=ENVS)
    parser.add_argument("--seed", type=int, default=0, choices=(0,))
    parser.add_argument("--save-root", type=Path, default=SAVE_ROOT)
    parser.add_argument(
        "--policy-variant", choices=tuple(POLICY_VARIANTS), default="direct")
    parser.add_argument(
        "--resume", action="store_true",
        help="Declare restart-safe orchestration to the scheduler. The runner "
             "always validates existing protocol state before continuing.")
    parser.add_argument(
        "--finalize-existing", action="store_true",
        help="Validate and finalize already-complete categorical branches "
             "without entering any training subprocess.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.finalize_existing:
        finalize_existing_protocol(args)
    else:
        run_protocol(args)


if __name__ == "__main__":
    main()
