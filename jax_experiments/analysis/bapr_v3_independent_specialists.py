"""Protocol helpers for fully independent fixed-mode specialists."""
from __future__ import annotations

import hashlib
import functools
import json
import os
import pickle
import shutil
import tarfile
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from jax_experiments.analysis import (
    run_bapr_v3_budget_matched_fork as fork_protocol,
)


ROOT = Path(os.environ.get(
    "BAPR_WORKSPACE_ROOT", Path(__file__).resolve().parents[2])).resolve()
LEGACY_FAMILIES = ("deterministic_mean", "mean_variance")
STOCHASTIC_HEADROOM_FAMILIES = ("packet_loss", "burst_torque")
STOCHASTIC_HEADROOM_ENVS = ("Ant-v2", "HalfCheetah-v2")
STRUCTURED_CHANNEL_FAMILIES = ("structured_channel",)
STRUCTURED_CHANNEL_ENVS = ("HalfCheetah-v2",)
FAMILIES = LEGACY_FAMILIES
MODES = tuple(range(4))
ENV = "Ant-v2"
SEED = 0
BASE_NEXT_ITERATION = 700
BASE_TOTAL_STEPS = 2_800_000
BASE_UPDATE_COUNT = 174_500
FINAL_ITERATION = 1399
FINAL_NEXT_ITERATION = 1400
FINAL_TOTAL_STEPS = 5_600_000
FINAL_UPDATE_COUNT = 349_500

SOURCE_PAIR_BASE = (
    ROOT / "jax_experiments" / "results_bapr_v3_budget_matched_fork_v2")
RUN_BASE = (
    ROOT / "jax_experiments"
    / "results_bapr_v3_independent_specialists_v2")
BUNDLE_BASE = (
    ROOT / "jax_experiments"
    / "eval_bundles_bapr_v3_independent_specialists_v2")
AUDIT_BASE = (
    ROOT / "jax_experiments"
    / "results_bapr_v3_independent_specialist_audit_v2")
ANALYSIS_BASE = (
    ROOT / "jax_experiments"
    / "results_bapr_v3_independent_specialist_analysis_v2")

BOOTSTRAP_SCHEMA = "bapr.v3-independent-specialist-bootstrap.v1"
COMPLETE_SCHEMA = "bapr.v3-independent-specialist-complete.v1"
BUNDLE_SCHEMA = "bapr.v3-independent-specialist-bundle.v1"
PROTOCOL_CHECKPOINT_DIR = Path("checkpoints") / "specialist_protocol"
BOOTSTRAP_REL = PROTOCOL_CHECKPOINT_DIR / "specialist_bootstrap.json"
SOURCE_MANIFEST_REL = PROTOCOL_CHECKPOINT_DIR / "source_manifest.json"
SOURCE_ARCHIVE_REL = PROTOCOL_CHECKPOINT_DIR / "source_snapshot.tar.gz"
COMPLETE_REL = Path("specialist_complete.json")
BUNDLE_MANIFEST = "bundle_manifest.json"
BUNDLE_SOURCE_DIR = Path("protocol_source")
BUNDLE_SOURCE_MANIFEST = BUNDLE_SOURCE_DIR / "source_manifest.json"
BUNDLE_SOURCE_ARCHIVE = BUNDLE_SOURCE_DIR / "source_snapshot.tar.gz"
AUDIT_SOURCE_DIR = Path("audit_source")
AUDIT_SOURCE_MANIFEST = AUDIT_SOURCE_DIR / "source_manifest.json"
AUDIT_SOURCE_ARCHIVE = AUDIT_SOURCE_DIR / "source_snapshot.tar.gz"
AUDIT_READY_MARKER = "audit_ready.pkl"
AUDIT_READY_SCHEMA = "bapr.v3-independent-specialist-audit-ready.v2"
COMPONENT_HASH_SCHEMA = "bapr.canonical-component-hash.v1"
RUNNER_REL = (
    "jax_experiments/analysis/run_bapr_v3_independent_specialist.py")


def configure_stochastic_headroom(env: str) -> None:
    """Select the equal-budget packet-loss/burst-torque source pairs."""
    global FAMILIES, ENV, SOURCE_PAIR_BASE, RUN_BASE, BUNDLE_BASE
    global AUDIT_BASE, ANALYSIS_BASE
    if env not in STOCHASTIC_HEADROOM_ENVS:
        raise ValueError(
            f"stochastic specialist env must be in "
            f"{STOCHASTIC_HEADROOM_ENVS}, got {env!r}"
        )
    FAMILIES = STOCHASTIC_HEADROOM_FAMILIES
    ENV = env
    SOURCE_PAIR_BASE = (
        ROOT / "jax_experiments"
        / "results_bapr_v3_stochastic_headroom_fork_v1"
    )
    RUN_BASE = (
        ROOT / "jax_experiments"
        / "results_bapr_v3_stochastic_independent_specialists_v1"
    )
    BUNDLE_BASE = (
        ROOT / "jax_experiments"
        / "eval_bundles_bapr_v3_stochastic_independent_specialists_v1"
    )
    AUDIT_BASE = (
        ROOT / "jax_experiments"
        / "results_bapr_v3_stochastic_independent_specialist_audit_v1"
    )
    ANALYSIS_BASE = (
        ROOT / "jax_experiments"
        / "results_bapr_v3_stochastic_independent_specialist_analysis_v1"
    )
    _validate_source_pair.cache_clear()


def configure_structured_channel_headroom(env: str) -> None:
    """Select the strict structured-channel HalfCheetah source pair."""
    global FAMILIES, ENV, SOURCE_PAIR_BASE, RUN_BASE, BUNDLE_BASE
    global AUDIT_BASE, ANALYSIS_BASE
    if env not in STRUCTURED_CHANNEL_ENVS:
        raise ValueError(
            f"structured-channel specialist env must be in "
            f"{STRUCTURED_CHANNEL_ENVS}, got {env!r}"
        )
    FAMILIES = STRUCTURED_CHANNEL_FAMILIES
    ENV = env
    SOURCE_PAIR_BASE = (
        ROOT / "jax_experiments"
        / "results_bapr_v3_structured_channel_headroom_fork_v2"
    )
    RUN_BASE = (
        ROOT / "jax_experiments"
        / "results_bapr_v3_structured_channel_independent_specialists_v1"
    )
    BUNDLE_BASE = (
        ROOT / "jax_experiments"
        / "eval_bundles_bapr_v3_structured_channel_independent_specialists_v1"
    )
    AUDIT_BASE = (
        ROOT / "jax_experiments"
        / "results_bapr_v3_structured_channel_independent_specialist_audit_v1"
    )
    ANALYSIS_BASE = (
        ROOT / "jax_experiments"
        / "results_bapr_v3_structured_channel_independent_specialist_analysis_v1"
    )
    _validate_source_pair.cache_clear()


def env_short() -> str:
    return ENV.removesuffix("-v2")


def source_pair(family: str) -> Path:
    _require_family(family)
    return fork_protocol.pair_dir(
        family, ENV, SEED, SOURCE_PAIR_BASE)


def family_run_root(family: str) -> Path:
    _require_family(family)
    return RUN_BASE / family / env_short()


def specialist_run_dir(family: str, mode: int) -> Path:
    _require_mode(mode)
    return family_run_root(family) / f"specialist_mode_{mode}"


def family_bundle_root(family: str) -> Path:
    _require_family(family)
    return BUNDLE_BASE / family / env_short()


def robust_bundle_dir(family: str) -> Path:
    return family_bundle_root(family) / "robust"


def specialist_bundle_dir(family: str, mode: int) -> Path:
    _require_mode(mode)
    return family_bundle_root(family) / f"specialist_mode_{mode}"


def _require_family(family: str) -> None:
    if family not in FAMILIES:
        raise ValueError(f"unsupported family {family!r}")


def _require_mode(mode: int) -> None:
    if int(mode) not in MODES:
        raise ValueError(f"mode must be in {MODES}, got {mode}")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "sha256": fork_protocol.sha256_file(path),
        "size": path.stat().st_size,
    }


def _hash_value(digest: Any, value: Any) -> None:
    if isinstance(value, Mapping):
        digest.update(b"mapping{")
        for key in sorted(value, key=lambda item: repr(item)):
            _hash_value(digest, key)
            _hash_value(digest, value[key])
        digest.update(b"}")
        return
    raw_value = getattr(value, "get_raw_value", None)
    if callable(raw_value):
        digest.update(b"variable{")
        _hash_value(digest, raw_value())
        digest.update(b"}")
        return
    if isinstance(value, (list, tuple)):
        value_type = type(value)
        type_name = f"{value_type.__module__}.{value_type.__qualname__}"
        digest.update(type_name.encode("utf-8") + b"[")
        for item in value:
            _hash_value(digest, item)
        digest.update(b"]")
        return
    if (isinstance(value, np.ndarray)
            or (hasattr(value, "__array__")
                and hasattr(value, "dtype")
                and hasattr(value, "shape"))):
        array = np.ascontiguousarray(value)
        digest.update(b"array:")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes())
        return
    if isinstance(value, np.generic):
        _hash_value(digest, np.asarray(value))
        return
    digest.update(type(value).__name__.encode("ascii"))
    digest.update(b":")
    digest.update(repr(value).encode("utf-8"))


def value_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _hash_value(digest, value)
    return digest.hexdigest()


def checkpoint_component_hashes(params_path: Path) -> dict[str, str]:
    ensure_checkpoint_compatibility()
    with params_path.open("rb") as handle:
        params = pickle.load(handle)
    required = (
        "policy", "critic", "target_critic", "log_alpha",
        "policy_opt_state", "critic_opt_state", "alpha_opt_state",
    )
    missing = [key for key in required if key not in params]
    if missing:
        raise ValueError(f"checkpoint omits components {missing}: {params_path}")
    return {key: value_sha256(params[key]) for key in required}


def ensure_checkpoint_compatibility() -> None:
    """Install the process-local shim required by legacy Flax checkpoints."""
    from jax_experiments.common.checkpoint import (
        _patch_flax_variablestate_unpickle,
    )

    _patch_flax_variablestate_unpickle()


def checkpoint_info(run_dir: Path) -> dict[str, Any]:
    ensure_checkpoint_compatibility()
    return fork_protocol.checkpoint_info(run_dir)


def _source_snapshot_record(run_dir: Path) -> dict[str, Any]:
    manifest = fork_protocol.run_snapshot_manifest(run_dir)
    return {
        "sha256": manifest["sha256"],
        "files": manifest["files"],
    }


@functools.lru_cache(maxsize=len(FAMILIES))
def _validate_source_pair(family: str) -> tuple[Path, dict[str, Any]]:
    ensure_checkpoint_compatibility()
    pair = source_pair(family)
    validated = fork_protocol.validate_complete_artifacts(pair)
    manifest = validated["manifest"]
    identity = manifest.get("identity") or {}
    if identity != {"family": family, "env": ENV, "seed": SEED}:
        raise ValueError(f"source pair identity mismatch: {identity!r}")
    return pair, manifest


def _immutable_source_record(run_dir: Path) -> dict[str, Any]:
    source_manifest = fork_protocol.current_source_manifest()
    manifest_path = run_dir / SOURCE_MANIFEST_REL
    archive_path = run_dir / SOURCE_ARCHIVE_REL
    write_json_atomic(manifest_path, source_manifest)
    fork_protocol.create_source_archive(archive_path, source_manifest)
    archive = fork_protocol.validate_source_archive(
        archive_path, source_manifest)
    return {
        "manifest": str(SOURCE_MANIFEST_REL),
        "manifest_sha256": fork_protocol.sha256_file(manifest_path),
        "source_sha256": source_manifest["sha256"],
        "archive": str(SOURCE_ARCHIVE_REL),
        "archive_sha256": archive["sha256"],
        "archive_size": archive["size"],
    }


def bootstrap_specialist(family: str, mode: int) -> dict[str, Any]:
    """Create one immutable iter-699 specialist fork on the local host."""
    _require_family(family)
    _require_mode(mode)
    run_dir = specialist_run_dir(family, mode)
    bootstrap_path = run_dir / BOOTSTRAP_REL
    if bootstrap_path.is_file():
        return validate_bootstrap(run_dir, family, mode)
    if run_dir.exists():
        raise RuntimeError(
            f"partial specialist directory lacks bootstrap manifest: {run_dir}")

    pair, pair_manifest = _validate_source_pair(family)
    shared = pair / "shared_base"
    shared_record = _source_snapshot_record(shared)
    fork_protocol.atomic_copy_run(shared, run_dir, shared_record)
    source_record = _immutable_source_record(run_dir)
    source_pair_manifest = pair / fork_protocol.PAIR_MANIFEST_REL
    payload = {
        "schema": BOOTSTRAP_SCHEMA,
        "status": "ready",
        "identity": {
            "family": family,
            "env": ENV,
            "seed": SEED,
            "mode": int(mode),
        },
        "source_pair": {
            "name": pair.name,
            "manifest_schema": pair_manifest["schema"],
            "manifest_sha256": fork_protocol.sha256_file(
                source_pair_manifest),
            "shared_snapshot_sha256": shared_record["sha256"],
        },
        "bootstrap_checkpoint": checkpoint_info(run_dir),
        "bootstrap_components": checkpoint_component_hashes(
            run_dir / "checkpoints" / "params.pkl"),
        "shared_snapshot": shared_record,
        "immutable_source": source_record,
        "runner_sha256": fork_protocol.sha256_file(ROOT / RUNNER_REL),
    }
    write_json_atomic(bootstrap_path, payload)
    return validate_bootstrap(run_dir, family, mode)


def validate_bootstrap(
    run_dir: Path, family: str, mode: int,
) -> dict[str, Any]:
    payload = read_json(run_dir / BOOTSTRAP_REL)
    expected_identity = {
        "family": family, "env": ENV, "seed": SEED, "mode": int(mode)}
    if (payload.get("schema") != BOOTSTRAP_SCHEMA
            or payload.get("status") != "ready"
            or payload.get("identity") != expected_identity):
        raise ValueError(f"invalid specialist bootstrap: {run_dir}")
    source = payload.get("immutable_source") or {}
    source_manifest_path = run_dir / str(source.get("manifest", ""))
    source_archive_path = run_dir / str(source.get("archive", ""))
    source_manifest = read_json(source_manifest_path)
    if (fork_protocol.sha256_file(source_manifest_path)
            != source.get("manifest_sha256")
            or source_manifest.get("sha256") != source.get("source_sha256")):
        raise ValueError("immutable specialist source manifest changed")
    if ((source_manifest.get("files") or {}).get(RUNNER_REL, {}).get(
            "sha256") != payload.get("runner_sha256")):
        raise ValueError("immutable source contains the wrong specialist runner")
    archive = fork_protocol.validate_source_archive(
        source_archive_path, source_manifest,
        expected_sha256=source.get("archive_sha256"))
    if archive["size"] != source.get("archive_size"):
        raise ValueError("immutable specialist source archive size changed")

    checkpoint = checkpoint_info(run_dir)
    if checkpoint["next_iteration"] == BASE_NEXT_ITERATION:
        if (checkpoint != payload.get("bootstrap_checkpoint")
                or _source_snapshot_record(run_dir)
                != payload.get("shared_snapshot")):
            raise ValueError("iter-699 specialist fork changed before training")
    elif not BASE_NEXT_ITERATION < checkpoint["next_iteration"] <= (
            FINAL_NEXT_ITERATION):
        raise ValueError(f"specialist checkpoint is outside budget: {checkpoint}")
    return payload


def extract_immutable_source(run_dir: Path, destination: Path) -> None:
    bootstrap = read_json(run_dir / BOOTSTRAP_REL)
    source = bootstrap["immutable_source"]
    manifest = read_json(run_dir / source["manifest"])
    archive_path = run_dir / source["archive"]
    fork_protocol.validate_source_archive(
        archive_path, manifest, expected_sha256=source["archive_sha256"])
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe source member: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"cannot read source member: {member.name}")
            data = stream.read()
            expected = manifest["files"][member.name]
            if (len(data) != expected["size"]
                    or hashlib.sha256(data).hexdigest()
                    != expected["sha256"]):
                raise ValueError(f"source member changed: {member.name}")
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)


def validate_specialist(
    run_dir: Path, family: str, mode: int,
) -> dict[str, Any]:
    bootstrap = validate_bootstrap(run_dir, family, mode)
    ensure_checkpoint_compatibility()
    checkpoint = fork_protocol.validate_checkpoint(
        run_dir, expected_iteration=FINAL_ITERATION,
        expected_steps=FINAL_TOTAL_STEPS,
        expected_update_count=FINAL_UPDATE_COUNT)
    logs = fork_protocol.validate_final_logs(run_dir)
    signature_path = run_dir / "logs" / "protocol_signature.json"
    signature = read_json(signature_path)
    config = signature.get("config") or {}
    expected_config = {
        "algo": "bapr_v3",
        "env_name": ENV,
        "seed": SEED,
        "env_type": "stochastic_mode",
        "stochastic_mode_family": family,
        "stochastic_mode_fixed_id": int(mode),
        "bapr_v2_base_pretrain_iters": 1400,
        "bapr_v2_teacher_iters": 0,
        "max_iters": FINAL_NEXT_ITERATION,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise ValueError(
                f"{run_dir.name} config {key}={config.get(key)!r}, "
                f"expected {expected!r}")
    start = int(signature.get("start_iteration", -1))
    total_at_start = int(signature.get("total_steps_at_start", -1))
    if (not signature.get("checkpoint_loaded")
            or not BASE_NEXT_ITERATION <= start < FINAL_NEXT_ITERATION
            or total_at_start != start * fork_protocol.SAMPLES_PER_ITER):
        raise ValueError("specialist did not resume from a valid fork checkpoint")

    mode_log = np.load(run_dir / "logs" / "mode_id.npy", allow_pickle=False)
    if (len(mode_log) != FINAL_NEXT_ITERATION
            or not np.all(mode_log[BASE_NEXT_ITERATION:] == int(mode))):
        values, counts = np.unique(
            mode_log[BASE_NEXT_ITERATION:], return_counts=True)
        raise ValueError(
            f"specialist mode log escaped mode {mode}: "
            f"{dict(zip(values.tolist(), counts.tolist()))}")

    components = checkpoint_component_hashes(
        run_dir / "checkpoints" / "params.pkl")
    unchanged = sorted(
        key for key, value in components.items()
        if value == bootstrap["bootstrap_components"].get(key))
    if unchanged:
        raise ValueError(
            f"specialist components did not update independently: {unchanged}")
    return {
        "checkpoint": checkpoint,
        "logs": logs,
        "protocol_signature": file_record(signature_path),
        "mode_log": {
            "rows": len(mode_log),
            "specialist_rows": len(mode_log) - BASE_NEXT_ITERATION,
            "unique_after_fork": [int(value) for value in np.unique(
                mode_log[BASE_NEXT_ITERATION:])],
        },
        "components": components,
        "source_shared_snapshot_sha256": bootstrap[
            "source_pair"]["shared_snapshot_sha256"],
    }


def _publish_bundle(
    run_dir: Path,
    destination: Path,
    *,
    family: str,
    role: str,
    mode: int | None,
    source_record: dict[str, Any],
    source_files: tuple[tuple[Path, Path], ...] = (),
) -> dict[str, Any]:
    if (destination / BUNDLE_MANIFEST).is_file():
        return validate_bundle(destination, family, role, mode)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp.", dir=destination.parent))
    try:
        files = (
            Path("checkpoints") / "params.pkl",
            Path("checkpoints") / "train_state.pkl",
            Path("logs") / "protocol_signature.json",
        )
        records = {}
        for relative in files:
            source = run_dir / relative
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            records[relative.as_posix()] = file_record(target)
        for source, relative in source_files:
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            records[relative.as_posix()] = file_record(target)
        payload = {
            "schema": BUNDLE_SCHEMA,
            "status": "complete",
            "identity": {
                "family": family,
                "env": ENV,
                "seed": SEED,
                "role": role,
                "mode": mode,
            },
            "checkpoint": checkpoint_info(run_dir),
            "component_hash_schema": COMPONENT_HASH_SCHEMA,
            "components": checkpoint_component_hashes(
                run_dir / "checkpoints" / "params.pkl"),
            "files": records,
            "source": source_record,
        }
        write_json_atomic(temporary / BUNDLE_MANIFEST, payload)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return validate_bundle(destination, family, role, mode)


def publish_specialist_bundle(
    run_dir: Path, family: str, mode: int,
) -> dict[str, Any]:
    complete = validate_specialist(run_dir, family, mode)
    complete_payload = {
        "schema": COMPLETE_SCHEMA,
        "status": "complete",
        "identity": {
            "family": family, "env": ENV, "seed": SEED,
            "mode": int(mode)},
        **complete,
    }
    write_json_atomic(run_dir / COMPLETE_REL, complete_payload)
    bootstrap = read_json(run_dir / BOOTSTRAP_REL)
    immutable = bootstrap["immutable_source"]
    return _publish_bundle(
        run_dir, specialist_bundle_dir(family, mode), family=family,
        role="specialist", mode=int(mode), source_record={
            "specialist_complete": file_record(run_dir / COMPLETE_REL),
            "shared_snapshot_sha256": complete[
                "source_shared_snapshot_sha256"],
            "source_sha256": immutable["source_sha256"],
            "source_archive_sha256": immutable["archive_sha256"],
            "source_archive_size": immutable["archive_size"],
            "bundle_source_manifest": BUNDLE_SOURCE_MANIFEST.as_posix(),
            "bundle_source_archive": BUNDLE_SOURCE_ARCHIVE.as_posix(),
        }, source_files=(
            (run_dir / immutable["manifest"], BUNDLE_SOURCE_MANIFEST),
            (run_dir / immutable["archive"], BUNDLE_SOURCE_ARCHIVE),
        ))


def publish_robust_bundle(family: str) -> dict[str, Any]:
    pair, manifest = _validate_source_pair(family)
    run_dir = pair / "robust_long"
    ensure_checkpoint_compatibility()
    fork_protocol.validate_checkpoint(
        run_dir, expected_iteration=FINAL_ITERATION,
        expected_steps=FINAL_TOTAL_STEPS,
        expected_update_count=FINAL_UPDATE_COUNT)
    fork_protocol.validate_final_logs(run_dir)
    return _publish_bundle(
        run_dir, robust_bundle_dir(family), family=family,
        role="robust", mode=None, source_record={
            "pair_manifest": file_record(
                pair / fork_protocol.PAIR_MANIFEST_REL),
            "pair_schema": manifest["schema"],
            "shared_snapshot_sha256": manifest[
                "shared_base"]["snapshot_sha256"],
        })


def validate_bundle(
    directory: Path, family: str, role: str, mode: int | None,
) -> dict[str, Any]:
    payload = read_json(directory / BUNDLE_MANIFEST)
    expected_identity = {
        "family": family, "env": ENV, "seed": SEED,
        "role": role, "mode": mode}
    if (payload.get("schema") != BUNDLE_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("identity") != expected_identity):
        raise ValueError(f"invalid specialist bundle identity: {directory}")
    checkpoint = payload.get("checkpoint") or {}
    if (checkpoint.get("iteration") != FINAL_ITERATION
            or checkpoint.get("next_iteration") != FINAL_NEXT_ITERATION
            or checkpoint.get("total_steps") != FINAL_TOTAL_STEPS
            or checkpoint.get("update_count") != FINAL_UPDATE_COUNT):
        raise ValueError(f"invalid specialist bundle budget: {directory}")
    for relative, expected in (payload.get("files") or {}).items():
        path = directory / relative
        if not path.is_file() or file_record(path) != expected:
            raise ValueError(f"specialist bundle file changed: {path}")
    components = checkpoint_component_hashes(
        directory / "checkpoints" / "params.pkl")
    component_schema = payload.get("component_hash_schema")
    if component_schema is None:
        # Legacy manifests hashed nnx.State via repr(), which varies between
        # CPU and GPU processes.  The params file was already verified above
        # by its full-file SHA256, so recompute canonical component hashes for
        # cross-host comparison without weakening artifact integrity.
        payload = dict(payload)
        payload["legacy_components"] = payload.get("components")
        payload["component_hash_schema"] = COMPONENT_HASH_SCHEMA
        payload["components"] = components
    elif component_schema != COMPONENT_HASH_SCHEMA:
        raise ValueError(
            f"unknown specialist component hash schema: {directory}")
    elif components != payload.get("components"):
        raise ValueError(f"specialist bundle components changed: {directory}")
    if role == "specialist":
        source = payload.get("source") or {}
        manifest_relative = source.get("bundle_source_manifest")
        archive_relative = source.get("bundle_source_archive")
        if (manifest_relative != BUNDLE_SOURCE_MANIFEST.as_posix()
                or archive_relative != BUNDLE_SOURCE_ARCHIVE.as_posix()):
            raise ValueError(f"specialist bundle source paths are invalid: {directory}")
        source_manifest = read_json(directory / manifest_relative)
        if source_manifest.get("sha256") != source.get("source_sha256"):
            raise ValueError(f"specialist source manifest changed: {directory}")
        archive = fork_protocol.validate_source_archive(
            directory / archive_relative, source_manifest,
            expected_sha256=source.get("source_archive_sha256"))
        if archive["size"] != source.get("source_archive_size"):
            raise ValueError(f"specialist source archive size changed: {directory}")
    return payload


def validate_family_bundles(family: str) -> dict[str, dict[str, Any]]:
    bundles = {
        "robust": validate_bundle(
            robust_bundle_dir(family), family, "robust", None)
    }
    for mode in MODES:
        bundles[f"specialist_mode_{mode}"] = validate_bundle(
            specialist_bundle_dir(family, mode), family,
            "specialist", mode)
    shared = {
        item["source"]["shared_snapshot_sha256"]
        for item in bundles.values()
    }
    if len(shared) != 1:
        raise ValueError(
            f"family bundles do not share one iter-699 snapshot: {shared}")
    specialist_hashes = [
        bundles[f"specialist_mode_{mode}"]["components"]
        for mode in MODES
    ]
    for component in (
            "policy", "critic", "target_critic", "log_alpha",
            "policy_opt_state", "critic_opt_state", "alpha_opt_state"):
        values = [item[component] for item in specialist_hashes]
        if len(set(values)) != len(MODES):
            raise ValueError(
                f"specialist {component} is not independent across modes")
    source_hashes = {
        bundles[f"specialist_mode_{mode}"]["source"].get("source_sha256")
        for mode in MODES
    }
    if len(source_hashes) != 1 or None in source_hashes:
        raise ValueError(
            f"specialists do not share one immutable source: {source_hashes}")
    # gzip/tar headers include creation metadata, so independently packaged
    # archives may have different file hashes even when every extracted source
    # file is identical. validate_bundle() already checks each archive against
    # the shared source manifest byte-for-byte.
    return bundles


def family_audit_source_manifest(family: str) -> Path:
    return family_bundle_root(family) / AUDIT_SOURCE_MANIFEST


def family_audit_source_archive(family: str) -> Path:
    return family_bundle_root(family) / AUDIT_SOURCE_ARCHIVE


def _validate_family_audit_source(
    family: str, record: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = family_audit_source_manifest(family)
    archive_path = family_audit_source_archive(family)
    if (record.get("manifest") != AUDIT_SOURCE_MANIFEST.as_posix()
            or record.get("archive") != AUDIT_SOURCE_ARCHIVE.as_posix()):
        raise ValueError("independent specialist audit source paths are invalid")
    if (file_record(manifest_path) != record.get("manifest_file")
            or file_record(archive_path) != record.get("archive_file")):
        raise ValueError("independent specialist audit source files changed")
    manifest = read_json(manifest_path)
    if manifest.get("sha256") != record.get("source_sha256"):
        raise ValueError("independent specialist audit source manifest changed")
    archive = fork_protocol.validate_source_archive(
        archive_path, manifest,
        expected_sha256=record["archive_file"]["sha256"])
    if archive["size"] != record["archive_file"]["size"]:
        raise ValueError("independent specialist audit source archive changed")
    return record


def publish_family_audit_source(family: str) -> dict[str, Any]:
    root = family_bundle_root(family)
    destination = root / AUDIT_SOURCE_DIR
    if destination.exists():
        manifest_path = family_audit_source_manifest(family)
        archive_path = family_audit_source_archive(family)
        manifest = read_json(manifest_path)
        record = {
            "manifest": AUDIT_SOURCE_MANIFEST.as_posix(),
            "manifest_file": file_record(manifest_path),
            "archive": AUDIT_SOURCE_ARCHIVE.as_posix(),
            "archive_file": file_record(archive_path),
            "source_sha256": manifest["sha256"],
        }
        return _validate_family_audit_source(family, record)

    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{AUDIT_SOURCE_DIR.name}.tmp.", dir=root))
    try:
        manifest = fork_protocol.current_source_manifest()
        manifest_path = temporary / "source_manifest.json"
        archive_path = temporary / "source_snapshot.tar.gz"
        write_json_atomic(manifest_path, manifest)
        fork_protocol.create_source_archive(archive_path, manifest)
        fork_protocol.validate_source_archive(archive_path, manifest)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    record = {
        "manifest": AUDIT_SOURCE_MANIFEST.as_posix(),
        "manifest_file": file_record(family_audit_source_manifest(family)),
        "archive": AUDIT_SOURCE_ARCHIVE.as_posix(),
        "archive_file": file_record(family_audit_source_archive(family)),
        "source_sha256": read_json(
            family_audit_source_manifest(family))["sha256"],
    }
    return _validate_family_audit_source(family, record)


def publish_family_audit_ready(family: str) -> dict[str, Any]:
    bundles = validate_family_bundles(family)
    root = family_bundle_root(family)
    audit_source = publish_family_audit_source(family)
    payload = {
        "schema": AUDIT_READY_SCHEMA,
        "status": "complete",
        "family": family,
        "env": ENV,
        "seed": SEED,
        "bundle_manifests": {
            name: file_record(root / name / BUNDLE_MANIFEST)
            for name in bundles
        },
        "source_sha256": bundles["specialist_mode_0"]["source"][
            "source_sha256"],
        "source_archive_sha256": bundles["specialist_mode_0"]["source"][
            "source_archive_sha256"],
        "audit_source": audit_source,
    }
    write_json_atomic(root / AUDIT_READY_MARKER, payload)
    return validate_family_audit_ready(family)


def validate_family_audit_ready(family: str) -> dict[str, Any]:
    root = family_bundle_root(family)
    payload = read_json(root / AUDIT_READY_MARKER)
    if (payload.get("schema") != AUDIT_READY_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("family") != family
            or payload.get("env") != ENV
            or payload.get("seed") != SEED):
        raise ValueError(f"invalid family audit-ready marker: {root}")
    bundles = validate_family_bundles(family)
    expected = {
        name: file_record(root / name / BUNDLE_MANIFEST)
        for name in bundles
    }
    if payload.get("bundle_manifests") != expected:
        raise ValueError(f"family bundle set changed after audit preparation: {root}")
    source = bundles["specialist_mode_0"]["source"]
    if (payload.get("source_sha256") != source["source_sha256"]
            or payload.get("source_archive_sha256")
            != source["source_archive_sha256"]):
        raise ValueError(f"family audit source changed: {root}")
    _validate_family_audit_source(family, payload.get("audit_source") or {})
    return payload
