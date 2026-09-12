#!/usr/bin/env python3
"""Validate and summarize the causal v2 BAPR-v3 budget-matched audit.

The validator is fail-closed.  It first proves that all four training pairs
were shared-checkpoint/common-restart forks, then proves that each of the 20
event-stream groups evaluated all six controllers under one runtime and source
snapshot.  Only a complete 120-output tree can reach the numerical gate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import statistics
import sys
import tarfile
from pathlib import Path
from typing import Any

from jax_experiments.analysis import analyze_bapr_v3_budget_matched_audit as v1
from jax_experiments.analysis import run_bapr_v3_budget_matched_fork as protocol


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIR_ROOT = (
    ROOT / "jax_experiments" / "results_bapr_v3_budget_matched_fork_v2")
DEFAULT_RESULTS_ROOT = (
    ROOT / "jax_experiments" / "results_bapr_v3_budget_matched_fork_audit_v2")
FAMILIES = v1.FAMILIES
ENVS = v1.ENVS
FULL_ENVS = tuple(f"{env}-v2" for env in ENVS)
FIXED_MODES = v1.FIXED_MODES
DEFAULT_EVENT_SEEDS = v1.DEFAULT_EVENT_SEEDS
SOURCES = v1.SOURCES
EXPECTED_ROWS = v1.EXPECTED_ROWS
EXPECTED_CHECKPOINT_NEXT_ITER = 1400
EXPECTED_CHECKPOINT_TOTAL_STEPS = 5_600_000
EXPECTED_FINAL_ITERATION = 1399
EXPECTED_FORK_START_ITERATION = 700
EXPECTED_FORK_START_STEPS = 2_800_000
EXPECTED_TRAINING_SEED = 0
PAIR_SCHEMA = "bapr.v3-budget-matched-fork.v2"
GROUP_SCHEMA = "bapr.v3-budget-matched-fork-audit-group.v2"
HEX_DIGITS = frozenset("0123456789abcdef")
AUDIT_PAIR_MANIFEST_ENV = "BAPR_AUDIT_PAIR_MANIFEST"
AUDIT_ORCHESTRATOR_MODULES = (
    "jax_experiments/analysis/"
    "run_bapr_v3_budget_matched_fork_audit_group.py",
    "jax_experiments/analysis/"
    "analyze_bapr_v3_budget_matched_fork_audit.py",
    "jax_experiments/analysis/analyze_bapr_v3_budget_matched_audit.py",
)
AUDIT_VALIDATOR_MODULE = AUDIT_ORCHESTRATOR_MODULES[1]
CATEGORICAL_VALIDATOR_OVERLAY_MODE = (
    "categorical-finalization-validator-overlay-v1")
CATEGORICAL_BOUNDARY_VALIDATOR_SHA256 = frozenset({
    "15cc4259d8d739d68945c1c417e895ad9a9eb6b0ebcdbbeaa00c10b04b0ebcd0",
    "f4baece7ebb379522acd4729023aa7a9daa6dc15f2c7c125808edf4890b970f1",
})


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing JSON file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError as exc:
        raise ValueError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_relative_path(raw: object, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} must be a non-empty relative path")
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != raw:
        raise ValueError(f"unsafe {label}: {raw!r}")
    return path


def _pair_source_files(source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    files = require_dict(source.get("files"), "pair source files")
    if not files:
        raise ValueError("pair source file manifest is empty")
    normalized = {}
    for raw_path, raw_entry in files.items():
        path = _safe_relative_path(raw_path, "pair source path").as_posix()
        entry = require_dict(raw_entry, f"pair source entry {path}")
        digest = require_sha256(entry.get("sha256"), f"pair source {path}")
        try:
            size = int(entry["size"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid pair source size for {path}") from exc
        if size < 0:
            raise ValueError(f"negative pair source size for {path}")
        normalized[path] = {"sha256": digest, "size": size}
    return normalized


def _validate_pair_source_archive(
    pair_dir: Path, source: dict[str, Any], *, extract_to: Path | None = None,
) -> None:
    files = _pair_source_files(source)
    snapshot_hash = require_sha256(
        source.get("snapshot_sha256"), "pair source snapshot hash")
    require_equal(
        canonical_sha256(files), snapshot_hash,
        "pair source file-manifest digest")
    archive = require_dict(source.get("archive"), "pair source archive")
    archive_relative = _safe_relative_path(
        archive.get("path"), "pair source archive path")
    require_equal(
        archive_relative.as_posix(), "provenance/source_snapshot.tar.gz",
        "pair source archive location")
    archive_path = pair_dir / archive_relative
    require_equal(
        file_sha256(archive_path),
        require_sha256(archive.get("sha256"), "pair source archive hash"),
        "pair source archive file hash")
    try:
        expected_size = int(archive["size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid pair source archive size") from exc
    require_equal(archive_path.stat().st_size, expected_size, "source archive size")

    if extract_to is not None:
        if extract_to.exists():
            raise ValueError(f"source extraction target already exists: {extract_to}")
        extract_to.mkdir(parents=True)
    discovered = {}
    try:
        with tarfile.open(archive_path, "r:gz") as archive_handle:
            for member in archive_handle.getmembers():
                if member.isdir():
                    continue
                relative = _safe_relative_path(
                    member.name, "source archive member").as_posix()
                if (not member.isfile() or member.issym() or member.islnk()
                        or relative in discovered):
                    raise ValueError(
                        f"unsafe/duplicate source archive member: {member.name!r}")
                stream = archive_handle.extractfile(member)
                if stream is None:
                    raise ValueError(f"cannot read source archive member {relative}")
                digest = hashlib.sha256()
                size = 0
                destination = extract_to / relative if extract_to else None
                output = None
                try:
                    if destination is not None:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        output = destination.open("xb")
                    while True:
                        block = stream.read(1024 * 1024)
                        if not block:
                            break
                        digest.update(block)
                        size += len(block)
                        if output is not None:
                            output.write(block)
                finally:
                    stream.close()
                    if output is not None:
                        output.close()
                discovered[relative] = {
                    "sha256": digest.hexdigest(), "size": size}
    except (OSError, tarfile.TarError) as exc:
        raise ValueError(f"cannot validate source archive: {exc}") from exc
    require_equal(discovered, files, "source archive contents")


def extract_pair_source_snapshot(
    pair_dir: Path, manifest: dict[str, Any], destination: Path,
) -> None:
    """Validate and extract the immutable producer source for evaluation."""
    source = require_dict(manifest.get("source"), "pair source")
    _validate_pair_source_archive(pair_dir, source, extract_to=destination)


def _audit_orchestrator_hashes(
    source: dict[str, Any],
) -> tuple[dict[str, str], dict[str, str]]:
    """Return producer and live hashes for the three audit modules."""
    files = _pair_source_files(source)
    producer = {}
    live = {}
    for relative in AUDIT_ORCHESTRATOR_MODULES:
        if relative not in files:
            raise ValueError(
                f"producer source snapshot omits audit orchestrator {relative}")
        producer[relative] = files[relative]["sha256"]
        live[relative] = file_sha256(ROOT / relative)
    return producer, live


def _categorical_overlay_authorization(
    source: dict[str, Any], pair_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Validate the narrow post-training categorical manifest exception."""
    require_equal(
        pair_manifest.get("source"), source,
        "overlay pair/source identity")
    identity = require_dict(
        pair_manifest.get("identity"), "overlay pair identity")
    variant = str(identity.get("policy_variant") or "")
    variant_config = protocol.POLICY_VARIANTS.get(variant)
    if (not isinstance(variant_config, dict)
            or variant_config.get("policy_mode") != "categorical_expert"):
        raise ValueError(
            f"validator overlay is restricted to categorical experts, got "
            f"{variant!r}")

    finalization = require_dict(
        pair_manifest.get("finalization"), "overlay finalization")
    require_equal(
        finalization.get("mode"),
        "posthoc-categorical-boundary-validator-v1",
        "overlay finalization mode")
    require_equal(
        finalization.get("training_reentered"), False,
        "overlay training-reentered flag")
    require_equal(
        finalization.get("original_training_source_sha256"),
        source.get("snapshot_sha256"),
        "overlay original training-source hash")
    archive = require_dict(source.get("archive"), "overlay source archive")
    require_equal(
        finalization.get("original_source_archive_sha256"),
        archive.get("sha256"),
        "overlay original source-archive hash")
    validator_relative = (
        "jax_experiments/analysis/run_bapr_v3_budget_matched_fork.py")
    require_equal(
        finalization.get("validator_file"), validator_relative,
        "overlay validator file")
    validator_hash = require_sha256(
        finalization.get("validator_file_sha256"),
        "overlay recorded finalization-validator hash")
    if validator_hash not in CATEGORICAL_BOUNDARY_VALIDATOR_SHA256:
        raise ValueError(
            "unrecognized categorical finalization-validator hash: "
            f"{validator_hash}")

    boundary = require_dict(
        pair_manifest.get("resume_boundary"), "overlay resume boundary")
    require_equal(
        boundary.get("physical_rollout_validation"),
        "categorical_policy_equivalence",
        "overlay boundary proof mode")
    require_equal(
        boundary.get("physical_rollout_accepted"), True,
        "overlay boundary acceptance")
    require_equal(
        boundary.get("task_schedule_equal"), True,
        "overlay task-schedule proof")
    require_equal(
        boundary.get("policy_canary_exact"), True,
        "overlay policy-canary proof")
    return {
        "policy_variant": variant,
        "finalization_mode": finalization["mode"],
        "finalization_validator_sha256": validator_hash,
        "training_reentered": False,
        "resume_boundary_validation": boundary["physical_rollout_validation"],
        "source_snapshot_sha256": source.get("snapshot_sha256"),
    }


def expected_orchestrator_source_provenance(
    source: dict[str, Any], pair_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the exact or narrowly authorized live-orchestrator record."""
    producer, live = _audit_orchestrator_hashes(source)
    if live == producer:
        # Preserve the v2 manifest shape for producer-exact historical groups.
        return producer

    changed = sorted(
        relative for relative in AUDIT_ORCHESTRATOR_MODULES
        if live[relative] != producer[relative])
    require_equal(
        changed, [AUDIT_VALIDATOR_MODULE],
        "categorical validator-overlay changed modules")
    if pair_manifest is None:
        raise ValueError(
            "categorical validator overlay requires the authoritative pair "
            "manifest")
    authorization = _categorical_overlay_authorization(
        source, pair_manifest)
    return {
        "mode": CATEGORICAL_VALIDATOR_OVERLAY_MODE,
        "producer": producer,
        "live": live,
        "changed_files": changed,
        "authorization": authorization,
    }


def validate_live_audit_modules(source: dict[str, Any]) -> dict[str, Any]:
    """Prove the staged orchestrator or its categorical-only validator overlay."""
    producer, live = _audit_orchestrator_hashes(source)
    if live == producer:
        return producer

    raw_manifest = os.environ.get(AUDIT_PAIR_MANIFEST_ENV, "").strip()
    if not raw_manifest:
        raise ValueError(
            f"live audit validator differs from the producer snapshot and "
            f"{AUDIT_PAIR_MANIFEST_ENV} is unset")
    manifest_path = Path(raw_manifest)
    if (not manifest_path.is_absolute()
            or manifest_path.name != "pair_manifest.json"
            or manifest_path.parent.name != "provenance"):
        raise ValueError(
            f"invalid {AUDIT_PAIR_MANIFEST_ENV}: {raw_manifest!r}")
    pair_manifest = read_json(manifest_path.resolve())
    return expected_orchestrator_source_provenance(source, pair_manifest)


def require_dict(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def require_list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ValueError(
            f"wrong {label}: found {actual!r}, expected {expected!r}")


def require_int(actual: object, expected: int, label: str) -> None:
    if isinstance(actual, bool):
        raise ValueError(f"{label} must be the integer {expected}")
    try:
        numeric = float(actual)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be the integer {expected}") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{label} must be the integer {expected}")
    require_equal(int(numeric), expected, label)


def require_sha256(actual: object, label: str) -> str:
    if not isinstance(actual, str):
        raise ValueError(f"{label} must be a SHA-256 string")
    value = actual.lower()
    if len(value) != 64 or any(char not in HEX_DIGITS for char in value):
        raise ValueError(f"{label} is not a valid SHA-256 digest: {actual!r}")
    return value


def pair_directory(
    pair_root: Path, family: str, env: str, seed: int,
) -> Path:
    short_env = env.removesuffix("-v2")
    return pair_root / f"budget_fork_v2_{family}_{short_env}_s{seed}"


def _last_npy_int(path: Path) -> int:
    try:
        import numpy as np

        values = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"cannot read numeric log {path}: {exc}") from exc
    if values.size == 0:
        raise ValueError(f"numeric log is empty: {path}")
    value = float(values.reshape(-1)[-1])
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError(f"numeric log has non-integral final value: {path}")
    return int(value)


def _validate_protocol_pair(
    pair_dir: Path, runtime: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    protocols = {
        branch: read_json(pair_dir / branch / "logs" / "protocol_signature.json")
        for branch in ("robust_long", "oracle_direct")
    }
    expected_treatments = {
        "robust_long": (1400, 0),
        "oracle_direct": (700, 700),
    }
    comparable_configs = {}
    for branch, protocol in protocols.items():
        require_equal(
            protocol.get("host"), runtime.get("host"),
            f"{branch} protocol host")
        require_equal(
            protocol.get("python"), runtime.get("python_executable"),
            f"{branch} protocol Python")
        require_equal(
            protocol.get("checkpoint_loaded"), True,
            f"{branch} protocol checkpoint_loaded")
        require_int(
            protocol.get("start_iteration"), EXPECTED_FORK_START_ITERATION,
            f"{branch} protocol start_iteration")
        require_int(
            protocol.get("total_steps_at_start"), EXPECTED_FORK_START_STEPS,
            f"{branch} protocol total_steps_at_start")
        config = dict(require_dict(
            protocol.get("config"), f"{branch} protocol config"))
        require_equal(config.get("run_name"), branch, f"{branch} run_name")
        require_equal(config.get("algo"), "bapr_v3", f"{branch} algo")
        base_iters, teacher_iters = expected_treatments[branch]
        require_int(
            config.get("bapr_v2_base_pretrain_iters"), base_iters,
            f"{branch} base_pretrain_iters")
        require_int(
            config.get("bapr_v2_teacher_iters"), teacher_iters,
            f"{branch} teacher_iters")
        for key, expected in (
            ("max_iters", 1400),
            ("samples_per_iter", 4000),
            ("updates_per_iter", 250),
            ("seed", 0),
        ):
            require_int(config.get(key), expected, f"{branch} config {key}")
        require_equal(
            config.get("bapr_v2_training_schedule"), "teacher_student",
            f"{branch} training schedule")
        require_equal(
            config.get("bapr_v2_warmstart_conditioned"), True,
            f"{branch} warmstart flag")
        require_equal(
            config.get("bapr_v2_freeze_gate_in_teacher"), True,
            f"{branch} teacher gate freeze")
        require_equal(
            config.get("resume_boundary_audit"), True,
            f"{branch} resume boundary audit flag")
        for treatment_key in (
            "run_name", "bapr_v2_base_pretrain_iters",
            "bapr_v2_teacher_iters"):
            config.pop(treatment_key, None)
        comparable_configs[branch] = config
    require_equal(
        comparable_configs["robust_long"],
        comparable_configs["oracle_direct"],
        "non-treatment branch configs")
    return protocols


def _validate_resume_boundaries(pair_dir: Path) -> dict[str, dict[str, Any]]:
    expected = {
        "robust_long": {
            "stage": "robust", "source": 0,
            "flags": [True, False, True], "warmstarted": False,
        },
        "oracle_direct": {
            "stage": "teacher", "source": 1,
            "flags": [False, True, True], "warmstarted": True,
        },
    }
    audits = {}
    for branch, branch_expected in expected.items():
        path = pair_dir / branch / "logs" / "resume_boundary_audit.json"
        payload = read_json(path)
        audits[branch] = payload
        require_equal(
            payload.get("schema"), "bapr.resume-boundary-audit.v1",
            f"{branch} boundary schema")
        require_equal(
            payload.get("semantics"),
            "shared-checkpoint/common-restart; not exact continuation",
            f"{branch} boundary semantics")
        require_equal(payload.get("run_name"), branch, f"{branch} boundary run")
        require_int(payload.get("seed"), 0, f"{branch} boundary seed")
        require_int(
            payload.get("iteration"), EXPECTED_FORK_START_ITERATION,
            f"{branch} boundary iteration")
        require_int(
            payload.get("total_steps_before_rollout"),
            EXPECTED_FORK_START_STEPS, f"{branch} boundary steps")
        try:
            replay_size = int(payload["replay_size_before_rollout"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid {branch} boundary replay size") from exc
        if replay_size <= 0:
            raise ValueError(f"{branch} boundary replay size must be positive")
        require_int(
            payload.get("agent_update_count_before_rollout"), 174500,
            f"{branch} boundary update count")
        require_equal(
            payload.get("training_stage"), branch_expected["stage"],
            f"{branch} boundary stage")
        require_int(
            payload.get("rollout_context_source"), branch_expected["source"],
            f"{branch} boundary context source")
        require_equal(
            payload.get("controller_update_flags"), branch_expected["flags"],
            f"{branch} boundary controller flags")
        require_equal(
            payload.get("train_policy_gate"), False,
            f"{branch} boundary policy gate")
        require_equal(
            payload.get("conditioned_warmstarted"),
            branch_expected["warmstarted"],
            f"{branch} conditioned warmstart")
        physical = require_dict(
            payload.get("physical_rollout"), f"{branch} physical rollout")
        require_equal(
            physical.get("fields"),
            ["obs", "act", "rew", "next_obs", "done", "task_id"],
            f"{branch} physical rollout fields")
        require_int(
            physical.get("transitions"), 4000,
            f"{branch} physical rollout transitions")
        require_equal(
            physical.get("excludes_context_by_design"), True,
            f"{branch} physical rollout context exclusion")
        require_equal(
            physical.get("finite"), True,
            f"{branch} physical rollout finiteness")
        shapes = require_dict(
            physical.get("validated_shapes"),
            f"{branch} physical validated shapes")
        require_equal(
            set(shapes), {"obs", "act", "rew", "next_obs", "done", "task_id"},
            f"{branch} physical shape fields")
        normalized_shapes = {}
        for field, raw_shape in shapes.items():
            shape = require_list(raw_shape, f"{branch} {field} shape")
            if (not shape or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    or value <= 0 for value in shape)):
                raise ValueError(f"invalid {branch} {field} shape: {shape!r}")
            if shape[0] != 4000:
                raise ValueError(
                    f"{branch} {field} shape does not contain 4000 transitions")
            normalized_shapes[field] = shape
        if (len(normalized_shapes["obs"]) != 2
                or len(normalized_shapes["act"]) != 2
                or normalized_shapes["next_obs"] != normalized_shapes["obs"]
                or len(normalized_shapes["rew"]) not in (1, 2)
                or (len(normalized_shapes["rew"]) == 2
                    and normalized_shapes["rew"][1] != 1)
                or len(normalized_shapes["done"]) not in (1, 2)
                or (len(normalized_shapes["done"]) == 2
                    and normalized_shapes["done"][1] != 1)
                or len(normalized_shapes["task_id"]) != 1):
            raise ValueError(f"invalid {branch} physical rollout shapes")
        require_sha256(
            physical.get("sha256"), f"{branch} physical rollout hash")
        field_hashes = require_dict(
            physical.get("field_sha256"),
            f"{branch} physical rollout field hashes")
        require_equal(
            set(field_hashes),
            {"obs", "act", "rew", "next_obs", "done", "task_id"},
            f"{branch} physical field-hash names")
        for field, digest in field_hashes.items():
            require_sha256(digest, f"{branch} physical {field} hash")

    expected_contexts = {
        "robust_long": ["robust_zero"],
        "oracle_direct": [f"oracle_task_{index}" for index in range(4)],
    }
    for branch, expected_source in (("robust_long", 0), ("oracle_direct", 1)):
        equivalence = require_dict(
            audits[branch].get("policy_equivalence"),
            f"{branch} boundary policy equivalence")
        require_equal(
            equivalence.get("pass"), True,
            f"{branch} policy equivalence pass")
        require_equal(
            equivalence.get("finite"), True,
            f"{branch} policy equivalence finite")
        require_int(
            equivalence.get("observations"), 32,
            f"{branch} policy-equivalence observations")
        require_int(
            equivalence.get("task_latents"), 4,
            f"{branch} policy-equivalence task latents")
        require_int(
            equivalence.get("tested_rollout_context_source"), expected_source,
            f"{branch} tested rollout source")
        require_equal(
            equivalence.get("tested_contexts"), expected_contexts[branch],
            f"{branch} tested policy contexts")
        try:
            max_mean = float(equivalence["max_abs_mean_diff"])
            max_log_std = float(equivalence["max_abs_log_std_diff"])
            tolerance = float(equivalence["tolerance"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid {branch} policy-equivalence tolerance") from exc
        if not (
            math.isfinite(max_mean) and math.isfinite(max_log_std)
            and math.isfinite(tolerance) and tolerance > 0.0
            and max_mean <= tolerance and max_log_std <= tolerance
        ):
            raise ValueError(
                f"{branch} policy distribution is not equivalent to base")

    require_equal(
        audits["robust_long"]["replay_size_before_rollout"],
        audits["oracle_direct"]["replay_size_before_rollout"],
        "paired replay size at restart")
    require_equal(
        audits["robust_long"]["agent_update_count_before_rollout"],
        audits["oracle_direct"]["agent_update_count_before_rollout"],
        "paired update count at restart")

    # Pair-level equivalence depends on the controller architecture and is
    # validated by protocol.boundary_rollout_proof after both individually
    # well-formed boundary audits have been loaded.
    return audits


def _npy_length_and_last(path: Path) -> tuple[int, int]:
    try:
        import numpy as np

        values = np.load(path, allow_pickle=False).reshape(-1)
    except Exception as exc:
        raise ValueError(f"cannot read numeric log {path}: {exc}") from exc
    if values.size == 0:
        raise ValueError(f"numeric log is empty: {path}")
    value = float(values[-1])
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError(f"numeric log has invalid final value: {path}")
    return int(values.size), int(value)


def _validate_final_logs(pair_dir: Path) -> dict[str, dict[str, int]]:
    result = {}
    for branch in ("robust_long", "oracle_direct"):
        log_dir = pair_dir / branch / "logs"
        iteration_rows, iteration_last = _npy_length_and_last(
            log_dir / "iteration.npy")
        steps_rows, steps_last = _npy_length_and_last(
            log_dir / "total_steps.npy")
        require_equal(
            iteration_rows, EXPECTED_CHECKPOINT_NEXT_ITER,
            f"{branch} iteration log rows")
        require_equal(
            iteration_last, EXPECTED_FINAL_ITERATION,
            f"{branch} final iteration")
        require_equal(
            steps_rows, EXPECTED_CHECKPOINT_NEXT_ITER,
            f"{branch} total-steps log rows")
        require_equal(
            steps_last, EXPECTED_CHECKPOINT_TOTAL_STEPS,
            f"{branch} final total steps")
        for filename in ("params.pkl", "train_state.pkl", "replay_buffer.npz"):
            path = pair_dir / branch / "checkpoints" / filename
            if not path.is_file() or path.stat().st_size <= 0:
                raise ValueError(f"missing/empty final checkpoint file: {path}")
        result[branch] = {
            "iteration_rows": iteration_rows,
            "iteration_last": iteration_last,
            "total_steps_rows": steps_rows,
            "total_steps_last": steps_last,
        }
    return result


def validate_pair_provenance(
    pair_dir: Path, family: str, env: str, training_seed: int,
) -> dict[str, Any]:
    """Validate one producer pair and return its authoritative manifest."""
    pair_dir = pair_dir.resolve()
    if training_seed != EXPECTED_TRAINING_SEED:
        raise ValueError("the causal audit requires training seed 0")
    expected_name = pair_directory(
        pair_dir.parent, family, env, training_seed).name
    require_equal(pair_dir.name, expected_name, "pair directory name")
    manifest_path = pair_dir / "provenance" / "pair_manifest.json"
    manifest = read_json(manifest_path)
    complete_marker = pair_dir / "pair_checkpoint_complete.pkl"
    if not complete_marker.is_file():
        raise ValueError(f"missing complete marker in {pair_dir}")
    try:
        with complete_marker.open("rb") as handle:
            sentinel = pickle.load(handle)
    except Exception as exc:
        raise ValueError(f"invalid complete marker {complete_marker}: {exc}") from exc
    sentinel = require_dict(sentinel, "complete marker")
    require_equal(
        sentinel.get("schema"), "bapr.pair-checkpoint-complete.v1",
        "complete marker schema")
    require_equal(sentinel.get("status"), "complete", "complete marker status")
    require_equal(
        sentinel.get("pair_manifest"), "provenance/pair_manifest.json",
        "complete marker manifest path")
    require_equal(
        sentinel.get("pair_manifest_sha256"), file_sha256(manifest_path),
        "complete marker manifest hash")
    require_int(
        sentinel.get("final_next_iteration"), EXPECTED_CHECKPOINT_NEXT_ITER,
        "complete marker final iteration")
    require_int(
        sentinel.get("final_total_steps"), EXPECTED_CHECKPOINT_TOTAL_STEPS,
        "complete marker final steps")

    require_equal(manifest.get("schema"), PAIR_SCHEMA, "pair manifest schema")
    require_equal(manifest.get("status"), "complete", "pair manifest status")
    require_equal(
        manifest.get("semantics"),
        "shared-checkpoint/common-restart; not exact continuation",
        "pair semantics")
    require_equal(
        manifest.get("continuation_claim"), False,
        "pair continuation claim")
    identity = require_dict(manifest.get("identity"), "pair identity")
    require_equal(identity.get("family"), family, "pair family")
    require_equal(identity.get("env"), env, "pair environment")
    require_int(identity.get("seed"), training_seed, "pair seed")

    runtime = require_dict(manifest.get("runtime"), "pair runtime")
    runtime_hash = _validate_runtime_manifest(runtime)
    source = require_dict(manifest.get("source"), "pair source")
    source_files = _pair_source_files(source)
    source_hash = require_sha256(
        source.get("snapshot_sha256"), "pair source snapshot hash")
    require_equal(
        canonical_sha256(source_files), source_hash,
        "pair source snapshot digest")
    require_equal(source.get("sha256"), source_hash, "pair source hash alias")
    require_equal(
        source.get("manifest_path"), "provenance/source_manifest.json",
        "pair source-manifest path")
    source_manifest_path = pair_dir / "provenance" / "source_manifest.json"
    source_manifest = read_json(source_manifest_path)
    require_equal(
        source_manifest.get("schema"), "bapr.source-snapshot.v1",
        "source manifest schema")
    require_equal(source_manifest.get("files"), source_files, "source files")
    require_equal(source_manifest.get("sha256"), source_hash, "source hash")
    _validate_pair_source_archive(pair_dir, source)

    shared = require_dict(manifest.get("shared_base"), "shared_base")
    require_equal(shared.get("run_name"), "shared_base", "shared-base run name")
    shared_hash = require_sha256(
        shared.get("snapshot_sha256"), "shared_base snapshot hash")
    checkpoint = require_dict(
        shared.get("checkpoint"), "shared_base checkpoint")
    require_int(
        checkpoint.get("iteration"), 699,
        "shared_base saved iteration")
    require_int(
        checkpoint.get("next_iteration"), EXPECTED_FORK_START_ITERATION,
        "shared_base next iteration")
    require_int(
        checkpoint.get("total_steps"), EXPECTED_FORK_START_STEPS,
        "shared_base total steps")
    require_int(
        checkpoint.get("update_count"), 174500,
        "shared_base update count")
    require_equal(
        checkpoint.get("update_count_error"), "",
        "shared_base update-count inspection")
    require_int(
        checkpoint.get("replay_size"), 1_000_000,
        "shared_base replay size")
    require_int(
        checkpoint.get("replay_ptr"), 800_000,
        "shared_base replay pointer")
    try:
        logger_key_count = int(checkpoint["logger_key_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid shared-base logger key count") from exc
    if logger_key_count <= 0:
        raise ValueError("shared-base logger key count must be positive")
    files = require_dict(shared.get("files"), "shared_base files")
    for required_file in (
        "checkpoints/params.pkl", "checkpoints/train_state.pkl",
        "checkpoints/replay_buffer.npz", "logs/protocol_signature.json"):
        if required_file not in files:
            raise ValueError(f"shared_base files missing {required_file}")
    normalized_shared_files = {}
    for raw_path, raw_entry in files.items():
        relative = _safe_relative_path(
            raw_path, "shared-base snapshot path").as_posix()
        entry = require_dict(raw_entry, f"shared-base file {relative}")
        digest = require_sha256(
            entry.get("sha256"), f"shared-base file {relative}")
        try:
            size = int(entry["size"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid shared-base file size for {relative}") from exc
        path = pair_dir / "shared_base" / relative
        require_equal(path.stat().st_size, size, f"shared-base {relative} size")
        require_equal(file_sha256(path), digest, f"shared-base {relative} hash")
        normalized_shared_files[relative] = {"sha256": digest, "size": size}
    require_equal(files, normalized_shared_files, "shared-base file manifest")
    require_equal(
        canonical_sha256(normalized_shared_files), shared_hash,
        "shared-base aggregate snapshot hash")
    require_equal(
        sentinel.get("shared_base_snapshot_sha256"), shared_hash,
        "complete-marker shared snapshot hash")

    base_snapshot_path = (
        pair_dir / "provenance" / "shared_base_snapshot_manifest.json")
    base_snapshot_manifest = read_json(base_snapshot_path)
    require_equal(
        base_snapshot_manifest.get("schema"), "bapr.shared-base-snapshot.v1",
        "shared-base snapshot schema")
    require_equal(
        base_snapshot_manifest.get("checkpoint"), checkpoint,
        "shared-base checkpoint manifest")
    require_equal(
        base_snapshot_manifest.get("files"), normalized_shared_files,
        "shared-base snapshot files")
    require_equal(
        base_snapshot_manifest.get("snapshot_sha256"), shared_hash,
        "shared-base snapshot digest")
    require_equal(
        shared.get("manifest_sha256"),
        canonical_sha256(base_snapshot_manifest),
        "shared-base manifest hash")

    forks = require_dict(manifest.get("forks"), "forks")
    require_equal(set(forks), {"robust_long", "oracle_direct"}, "fork names")
    for branch in ("robust_long", "oracle_direct"):
        fork = require_dict(forks.get(branch), f"fork {branch}")
        require_sha256(
            fork.get("source_snapshot_sha256"),
            f"{branch} source snapshot hash")
        require_equal(
            fork.get("source_snapshot_sha256"), shared_hash,
            f"{branch} shared snapshot identity")
        require_equal(fork.get("run_name"), branch, f"{branch} fork run name")
        require_int(
            fork.get("source_checkpoint_next_iteration"),
            EXPECTED_FORK_START_ITERATION,
            f"{branch} fork start iteration")
        require_int(
            fork.get("source_total_steps"), EXPECTED_FORK_START_STEPS,
            f"{branch} fork start steps")
        require_int(
            fork.get("start_iteration"), EXPECTED_FORK_START_ITERATION,
            f"{branch} start-iteration alias")
        require_int(
            fork.get("start_total_steps"), EXPECTED_FORK_START_STEPS,
            f"{branch} start-steps alias")
        require_equal(
            fork.get("runtime_sha256"), runtime_hash,
            f"{branch} fork runtime hash")
        require_equal(
            fork.get("source_sha256"), source_hash,
            f"{branch} fork source hash")
        require_int(
            fork.get("source_files_verified"), len(normalized_shared_files),
            f"{branch} verified source files")
    fork_manifest = read_json(pair_dir / "provenance" / "fork_manifest.json")
    require_equal(
        fork_manifest.get("schema"), "bapr.atomic-fork.v1",
        "fork manifest schema")
    require_equal(
        fork_manifest.get("semantics"), manifest.get("semantics"),
        "fork manifest semantics")
    require_equal(
        fork_manifest.get("shared_base_snapshot_sha256"), shared_hash,
        "fork-manifest shared hash")
    require_equal(fork_manifest.get("forks"), forks, "fork manifest branches")

    boundaries = _validate_resume_boundaries(pair_dir)
    boundary_manifest = require_dict(
        manifest.get("resume_boundary"), "resume_boundary")
    expected_proof = protocol.boundary_rollout_proof(
        require_dict(manifest.get("identity"), "identity"),
        boundaries["robust_long"], boundaries["oracle_direct"])
    legacy_keys = (
        "physical_rollout_equal", "physical_rollout_sha256",
        "field_sha256_equal",
    )
    for key in legacy_keys:
        require_equal(
            boundary_manifest.get(key), expected_proof[key],
            f"manifest physical proof {key}")
    has_rich_proof = "physical_rollout_validation" in boundary_manifest
    if (expected_proof["physical_rollout_validation"]
            == "categorical_policy_equivalence" or has_rich_proof):
        recorded_proof = {
            key: value for key, value in boundary_manifest.items()
            if key not in ("robust_long", "oracle_direct")
        }
        require_equal(
            recorded_proof, expected_proof,
            "manifest categorical physical rollout proof")
    for branch in ("robust_long", "oracle_direct"):
        entry = require_dict(
            boundary_manifest.get(branch), f"resume_boundary {branch}")
        boundary_hash = require_sha256(
            entry.get("file_sha256"), f"{branch} boundary file hash")
        require_equal(
            boundary_hash, file_sha256(
                pair_dir / branch / "logs" / "resume_boundary_audit.json"),
            f"{branch} boundary manifest hash")
        require_equal(
            require_dict(
                entry.get("physical_rollout"),
                f"{branch} manifest physical rollout").get("sha256"),
            boundaries[branch]["physical_rollout"]["sha256"],
            f"{branch} manifest physical hash")

    protocols = _validate_protocol_pair(pair_dir, runtime)
    final_logs = _validate_final_logs(pair_dir)
    config_diffs = require_dict(manifest.get("config_diffs"), "config diffs")
    require_equal(
        set(config_diffs), {"robust_long", "oracle_direct"},
        "config-diff branches")
    expected_allowlist = {
        "run_name", "max_iters", "min_resume_iteration",
        "bapr_v2_base_pretrain_iters", "bapr_v2_teacher_iters",
        "resume_boundary_audit", "resume_boundary_expected_iteration",
        "resume_boundary_expected_total_steps",
        "resume_boundary_expected_update_count",
    }
    for branch in ("robust_long", "oracle_direct"):
        diff = require_dict(config_diffs[branch], f"{branch} config diff")
        require_equal(
            set(require_list(diff.get("allowlist"), f"{branch} allowlist")),
            expected_allowlist, f"{branch} config allowlist")
        require_equal(diff.get("unexpected"), [], f"{branch} unexpected config")
        require_dict(diff.get("changed"), f"{branch} changed config")

    final = require_dict(manifest.get("final"), "final")
    for branch in ("robust_long", "oracle_direct"):
        entry = require_dict(final.get(branch), f"final {branch}")
        checkpoint = require_dict(
            entry.get("checkpoint"), f"final {branch} checkpoint")
        require_int(
            checkpoint.get("iteration"), EXPECTED_FINAL_ITERATION,
            f"final {branch} saved iteration")
        require_int(
            checkpoint.get("saved_iteration"), EXPECTED_FINAL_ITERATION,
            f"final {branch} saved-iteration alias")
        require_int(
            checkpoint.get("next_iteration"), EXPECTED_CHECKPOINT_NEXT_ITER,
            f"final {branch} next iteration")
        require_int(
            checkpoint.get("total_steps"), EXPECTED_CHECKPOINT_TOTAL_STEPS,
            f"final {branch} total steps")
        require_int(
            checkpoint.get("update_count"), 349500,
            f"final {branch} update count")
        require_equal(
            checkpoint.get("update_count_error"), "",
            f"final {branch} update inspection")
        require_int(
            checkpoint.get("replay_size"), 1_000_000,
            f"final {branch} replay size")
        require_int(
            checkpoint.get("replay_ptr"), 600_000,
            f"final {branch} replay pointer")
        require_equal(
            entry.get("logs"), final_logs[branch],
            f"final {branch} log manifest")
        logger_prefix = require_dict(
            entry.get("logger_prefix"), f"final {branch} logger prefix")
        for key in ("logger_keys_checked", "npy_files_checked"):
            try:
                count = int(logger_prefix[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid final {branch} logger-prefix {key}") from exc
            if count <= 0:
                raise ValueError(
                    f"final {branch} logger-prefix {key} must be positive")
        require_sha256(
            logger_prefix.get("logger_prefix_pickle_sha256"),
            f"final {branch} logger pickle-prefix hash")
        require_sha256(
            logger_prefix.get("diagnostic_prefix_sha256"),
            f"final {branch} diagnostic-prefix hash")
        final_files = require_dict(
            entry.get("files"), f"final {branch} files")
        expected_final_files = {
            "checkpoints/params.pkl", "checkpoints/train_state.pkl",
            "checkpoints/replay_buffer.npz", "logs/protocol_signature.json",
            "logs/resume_boundary_audit.json",
        }
        require_equal(
            set(final_files), expected_final_files,
            f"final {branch} file names")
        for raw_relative, raw_record in final_files.items():
            relative = _safe_relative_path(
                raw_relative, f"final {branch} file path")
            record = require_dict(
                raw_record, f"final {branch} file {raw_relative}")
            digest = require_sha256(
                record.get("sha256"),
                f"final {branch} file {raw_relative} hash")
            try:
                size = int(record["size"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid final {branch} file size {raw_relative}") from exc
            path = pair_dir / branch / relative
            require_equal(path.stat().st_size, size, f"{branch}/{raw_relative} size")
            require_equal(file_sha256(path), digest, f"{branch}/{raw_relative} hash")
        protocol_path = pair_dir / branch / "logs" / "protocol_signature.json"
        require_equal(
            entry.get("protocol_signature_sha256"), file_sha256(protocol_path),
            f"final {branch} protocol hash")
        require_equal(
            protocols[branch], read_json(protocol_path),
            f"final {branch} protocol identity")
        boundary_copy = require_dict(
            entry.get("boundary_copy"), f"final {branch} boundary copy")
        copy_relative = _safe_relative_path(
            boundary_copy.get("path"), f"{branch} boundary-copy path")
        expected_copy = f"provenance/{branch}_resume_boundary_audit.json"
        require_equal(
            copy_relative.as_posix(), expected_copy,
            f"{branch} boundary-copy location")
        copy_hash = file_sha256(pair_dir / copy_relative)
        require_equal(
            boundary_copy.get("sha256"), copy_hash,
            f"{branch} boundary-copy hash")
        require_equal(
            copy_hash, boundary_manifest[branch]["file_sha256"],
            f"{branch} immutable/live boundary identity")
    return manifest


def group_directory(
    results_root: Path, family: str, env: str, event_seed: int,
) -> Path:
    return (
        results_root / family / env.removesuffix("-v2")
        / f"event_seed_{event_seed}")


def _validate_source_manifest(source: dict[str, Any]) -> str:
    digest = require_sha256(source.get("sha256"), "group source hash")
    entries = require_list(source.get("files"), "group source files")
    declared_count = source.get("file_count")
    require_int(declared_count, len(entries), "group source file count")
    combined = hashlib.sha256()
    seen = set()
    for index, raw_entry in enumerate(entries):
        entry = require_dict(raw_entry, f"group source file {index}")
        path = entry.get("path")
        if not isinstance(path, str) or not path or path in seen:
            raise ValueError(f"invalid/duplicate group source path {path!r}")
        seen.add(path)
        value = require_sha256(
            entry.get("sha256"), f"group source file {path} hash")
        combined.update(path.encode("utf-8"))
        combined.update(b"\0")
        combined.update(value.encode("ascii"))
        combined.update(b"\n")
    require_equal(combined.hexdigest(), digest, "group source aggregate hash")
    return digest


def _validate_runtime_manifest(runtime: dict[str, Any]) -> str:
    digest = require_sha256(runtime.get("sha256"), "group runtime hash")
    unhashed = dict(runtime)
    unhashed.pop("sha256", None)
    require_equal(canonical_sha256(unhashed), digest, "group runtime digest")
    packages = require_dict(runtime.get("packages"), "group runtime packages")
    for package in ("jax", "jaxlib", "flax", "optax", "brax", "numpy"):
        value = packages.get(package)
        if not isinstance(value, str) or not value:
            raise ValueError(f"group runtime missing package version {package}")
    jax_runtime = require_dict(runtime.get("jax"), "group JAX runtime")
    devices = require_list(jax_runtime.get("devices"), "group JAX devices")
    if len(devices) != 1 or not isinstance(devices[0], dict):
        raise ValueError("group runtime must select exactly one JAX device")
    if jax_runtime.get("backend") != "gpu":
        raise ValueError("group evaluator did not use the GPU JAX backend")
    device = devices[0]
    if device.get("platform") != "gpu" and device.get("platform") != "cuda":
        raise ValueError(f"unexpected JAX device platform: {device!r}")
    if not isinstance(device.get("kind"), str) or not device["kind"]:
        raise ValueError("group JAX device lacks a model/kind")
    gpu = require_dict(runtime.get("gpu"), "group GPU runtime")
    rows = require_list(
        gpu.get("selected_nvidia_smi_rows"), "group selected GPUs")
    if len(rows) != 1 or not isinstance(rows[0], str):
        raise ValueError("group runtime must identify exactly one selected GPU")
    if gpu.get("nvidia_smi_error"):
        raise ValueError(f"group nvidia-smi failed: {gpu['nvidia_smi_error']}")
    return digest


def _gpu_model_driver(runtime: dict[str, Any], label: str) -> tuple[str, str]:
    gpu = require_dict(runtime.get("gpu"), f"{label} GPU runtime")
    rows = require_list(
        gpu.get("selected_nvidia_smi_rows"), f"{label} selected GPUs")
    if len(rows) != 1 or not isinstance(rows[0], str):
        raise ValueError(f"{label} must identify exactly one selected GPU")
    columns = [item.strip() for item in rows[0].split(",", 3)]
    if len(columns) != 4 or not columns[2] or not columns[3]:
        raise ValueError(f"cannot parse {label} GPU identity: {rows[0]!r}")
    return columns[2], columns[3]


def validate_evaluator_runtime(
    pair_runtime: dict[str, Any], evaluator_runtime: dict[str, Any],
) -> None:
    """Require evaluator/trainer runtime equality, allowing only GPU slot ID."""
    _validate_runtime_manifest(pair_runtime)
    _validate_runtime_manifest(evaluator_runtime)
    for key in (
        "host", "python_executable", "python_realpath", "python_version",
        "platform", "packages",
    ):
        require_equal(
            evaluator_runtime.get(key), pair_runtime.get(key),
            f"evaluator/trainer runtime {key}")
    pair_jax = require_dict(pair_runtime.get("jax"), "pair JAX runtime")
    eval_jax = require_dict(
        evaluator_runtime.get("jax"), "evaluator JAX runtime")
    require_equal(
        eval_jax.get("backend"), pair_jax.get("backend"),
        "evaluator/trainer JAX backend")
    pair_devices = require_list(pair_jax.get("devices"), "pair JAX devices")
    eval_devices = require_list(eval_jax.get("devices"), "evaluator JAX devices")
    pair_device = require_dict(pair_devices[0], "pair JAX device")
    eval_device = require_dict(eval_devices[0], "evaluator JAX device")
    require_equal(
        (eval_device.get("platform"), eval_device.get("kind")),
        (pair_device.get("platform"), pair_device.get("kind")),
        "evaluator/trainer JAX device model")
    require_equal(
        _gpu_model_driver(evaluator_runtime, "evaluator"),
        _gpu_model_driver(pair_runtime, "trainer"),
        "evaluator/trainer GPU model and driver")


def validate_evaluator_source(
    pair_source: dict[str, Any], evaluator_source: dict[str, Any],
) -> None:
    """Require every training/evaluation package source file to be identical."""
    pair_files = require_dict(pair_source.get("files"), "pair source files")
    eval_entries = require_list(
        evaluator_source.get("files"), "evaluator source files")
    eval_files = {}
    for raw_entry in eval_entries:
        entry = require_dict(raw_entry, "evaluator source entry")
        path = entry.get("path")
        if not isinstance(path, str) or path in eval_files:
            raise ValueError(f"invalid/duplicate evaluator source path: {path!r}")
        eval_files[path] = require_sha256(
            entry.get("sha256"), f"evaluator source {path}")
    package_paths = sorted(
        path for path in pair_files if path.startswith("jax_experiments/"))
    if not package_paths:
        raise ValueError("pair source manifest contains no jax_experiments files")
    eval_package_paths = sorted(
        path for path in eval_files if path.startswith("jax_experiments/"))
    require_equal(
        eval_package_paths, package_paths,
        "evaluator/trainer package source paths")
    for path in package_paths:
        raw_pair_entry = pair_files[path]
        pair_digest = (
            raw_pair_entry.get("sha256")
            if isinstance(raw_pair_entry, dict) else raw_pair_entry)
        require_equal(
            eval_files[path], require_sha256(
                pair_digest, f"pair source {path}"),
            f"evaluator/trainer source hash {path}")


def _validate_output_metadata(
    rows: list[dict[str, str]], path: Path, family: str, env_short: str,
    source: v1.SourceSpec,
) -> None:
    run_name = (
        "robust_long" if source.directory == "robust" else "oracle_direct")
    expected_oracle_mode = (
        "dynamic" if source.fixed_mode is None else str(source.fixed_mode))
    for row_number, row in enumerate(rows, start=2):
        v1.require_text(row, "algo", "bapr_v3", path, row_number)
        v1.require_text(row, "env", env_short, path, row_number)
        v1.require_text(row, "run_name", run_name, path, row_number)
        v1.require_text(
            row, "eval_context_source", source.context_source, path,
            row_number)
        v1.require_text(
            row, "eval_oracle_mode_id", expected_oracle_mode, path,
            row_number)
        v1.require_text(row, "eval_advantage", "off", path, row_number)
        v1.require_text(
            row, "heldout_task_stream", "validation", path, row_number)
        if v1.exact_int(
            row, "checkpoint_next_iter", path, row_number
        ) != EXPECTED_CHECKPOINT_NEXT_ITER:
            raise ValueError(f"wrong checkpoint iteration in {path}")
        if v1.exact_int(
            row, "checkpoint_total_steps", path, row_number
        ) != EXPECTED_CHECKPOINT_TOTAL_STEPS:
            raise ValueError(f"wrong checkpoint steps in {path}")
        if v1.exact_int(row, "seed", path, row_number) != 0:
            raise ValueError(f"wrong training seed in {path}")


def validate_controller_output(
    directory: Path, family: str, env_short: str, source: v1.SourceSpec,
) -> v1.OutputMetrics:
    outputs = {
        filename: v1.read_rows(directory / filename, expected_count)
        for filename, expected_count in EXPECTED_ROWS.items()
    }
    for filename, rows in outputs.items():
        _validate_output_metadata(
            rows, directory / filename, family, env_short, source)
    stationary, switching = v1.select_summary_metrics(
        outputs["summary.csv"], directory / "summary.csv")
    task_returns = v1.validate_task_returns(
        outputs["task_returns.csv"], directory / "task_returns.csv")
    v1.validate_switching_returns(
        outputs["switching_returns.csv"],
        directory / "switching_returns.csv")
    v1.validate_switching_trace(
        outputs["switching_trace.csv"], directory / "switching_trace.csv")
    return v1.OutputMetrics(stationary, switching, task_returns)


def _validate_controller_command(
    record: dict[str, Any],
    pair_dir: Path,
    source: v1.SourceSpec,
    event_seed: int,
    runtime: dict[str, Any],
) -> None:
    command = require_list(
        record.get("command"), f"{source.directory} evaluator command")
    if any(not isinstance(token, str) for token in command):
        raise ValueError(f"{source.directory} evaluator command has non-text tokens")
    expected_branch = (
        "robust_long" if source.directory == "robust" else "oracle_direct")
    run_dir = pair_dir / expected_branch
    try:
        output_index = command.index("--out-dir") + 1
        output_value = command[output_index]
    except (ValueError, IndexError) as exc:
        raise ValueError(
            f"{source.directory} evaluator command lacks --out-dir") from exc
    if Path(output_value).name != source.directory:
        raise ValueError(
            f"{source.directory} evaluator command has wrong output path")
    expected = [
        str(runtime["python_executable"]), "-u", "-m",
        "jax_experiments.analysis.final_task_sweep",
        "--run-dir", str(run_dir),
        "--out-dir", output_value,
        "--episodes-per-task", "5",
        "--switching-episodes", "5",
        "--switching-period-steps", "500",
        "--heldout-task-stream", "validation",
        "--detection-window-steps", "50",
        "--bapr-v2-context-source", source.context_source,
        "--bapr-v2-advantage", "off",
        "--rng-seed", str(20260715 + event_seed),
        "--eval-seed-offset", str(event_seed),
        "--max-tasks", "4",
        "--stationary-test-only",
        "--min-checkpoint-next-iter", str(EXPECTED_CHECKPOINT_NEXT_ITER),
        "--resume-from", str(run_dir / "checkpoints" / "train_state.pkl"),
    ]
    if source.fixed_mode is not None:
        expected += ["--fixed-oracle-mode-id", str(source.fixed_mode)]
    require_equal(command, expected, f"{source.directory} evaluator command")


def validate_group_bundle(
    group_dir: Path,
    pair_dir: Path,
    family: str,
    env: str,
    training_seed: int,
    event_seed: int,
) -> dict[str, v1.OutputMetrics]:
    """Validate one six-controller output group and return its metrics."""
    manifest_path = group_dir / "provenance" / "group_manifest.json"
    manifest = read_json(manifest_path)
    require_equal(manifest.get("schema"), GROUP_SCHEMA, "group schema")
    require_equal(manifest.get("status"), "complete", "group status")
    require_equal(manifest.get("family"), family, "group family")
    require_equal(manifest.get("env"), env, "group environment")
    require_int(manifest.get("training_seed"), training_seed, "group seed")
    require_int(manifest.get("event_seed"), event_seed, "group event seed")
    require_int(
        manifest.get("rng_seed"), 20260715 + event_seed, "group RNG seed")
    require_int(
        manifest.get("eval_seed_offset"), event_seed,
        "group evaluation seed offset")
    require_equal(manifest.get("pair_name"), pair_dir.name, "group pair name")
    require_equal(
        manifest.get("pair_manifest_schema"), PAIR_SCHEMA,
        "group pair schema")
    require_equal(
        manifest.get("pair_manifest_sha256"),
        file_sha256(pair_dir / "provenance" / "pair_manifest.json"),
        "group producer-manifest hash")

    runtime = require_dict(manifest.get("runtime"), "group runtime")
    source_tree = require_dict(manifest.get("source"), "group source")
    runtime_hash = _validate_runtime_manifest(runtime)
    source_hash = _validate_source_manifest(source_tree)
    pair_manifest = read_json(
        pair_dir / "provenance" / "pair_manifest.json")
    validate_evaluator_runtime(
        require_dict(pair_manifest.get("runtime"), "pair runtime"), runtime)
    validate_evaluator_source(
        require_dict(pair_manifest.get("source"), "pair source"), source_tree)
    pair_source_files = _pair_source_files(require_dict(
        pair_manifest.get("source"), "pair source"))
    pair_source = require_dict(pair_manifest.get("source"), "pair source")
    pair_archive = require_dict(
        pair_source.get("archive"), "pair source archive")
    execution_source = require_dict(
        manifest.get("execution_source"), "group execution source")
    require_equal(
        execution_source.get("mode"), "validated_producer_archive",
        "group execution-source mode")
    require_equal(
        execution_source.get("archive_path"), pair_archive.get("path"),
        "group execution archive path")
    require_equal(
        execution_source.get("archive_sha256"), pair_archive.get("sha256"),
        "group execution archive hash")
    require_equal(
        execution_source.get("source_tree_sha256"), source_hash,
        "group execution source-tree hash")
    orchestrator = require_dict(
        manifest.get("orchestrator_source_sha256"),
        "group orchestrator source hashes")
    if any(relative not in pair_source_files
           for relative in AUDIT_ORCHESTRATOR_MODULES):
        raise ValueError("producer source omits packaged audit orchestrator modules")
    required_orchestrator = expected_orchestrator_source_provenance(
        pair_source, pair_manifest)
    require_equal(
        orchestrator, required_orchestrator,
        "group orchestrator source provenance")
    controllers = require_dict(manifest.get("controllers"), "group controllers")
    require_equal(
        set(controllers), {source.directory for source in SOURCES},
        "group controller names")

    expected_children = {
        source.directory for source in SOURCES} | {"provenance"}
    actual_children = {path.name for path in group_dir.iterdir()}
    require_equal(actual_children, expected_children, "group directory children")
    metrics = {}
    env_short = env.removesuffix("-v2")
    for source in SOURCES:
        record = require_dict(
            controllers[source.directory],
            f"group controller {source.directory}")
        expected_branch = (
            "robust_long" if source.directory == "robust"
            else "oracle_direct")
        require_equal(
            record.get("branch"), expected_branch,
            f"{source.directory} branch")
        require_equal(
            record.get("context_source"), source.context_source,
            f"{source.directory} context source")
        require_equal(
            record.get("fixed_mode"), source.fixed_mode,
            f"{source.directory} fixed mode")
        require_equal(
            record.get("runtime_sha256"), runtime_hash,
            f"{source.directory} runtime identity")
        require_equal(
            record.get("source_tree_sha256"), source_hash,
            f"{source.directory} source identity")
        require_equal(
            record.get("execution_source"), "validated_producer_archive",
            f"{source.directory} execution source")
        _validate_controller_command(
            record, pair_dir, source, event_seed, runtime)
        hashes = require_dict(
            record.get("output_sha256"),
            f"{source.directory} output hashes")
        require_equal(set(hashes), set(EXPECTED_ROWS), "output hash filenames")
        directory = group_dir / source.directory
        for filename in EXPECTED_ROWS:
            require_equal(
                hashes.get(filename), file_sha256(directory / filename),
                f"{source.directory}/{filename} hash")
        metrics[source.directory] = validate_controller_output(
            directory, family, env_short, source)
    return metrics


def _expected_group_directories(
    results_root: Path, event_seeds: tuple[int, ...],
) -> set[Path]:
    return {
        group_directory(results_root, family, f"{env}-v2", event_seed)
        for family in FAMILIES
        for env in ENVS
        for event_seed in event_seeds
    }


def load_complete_audit(
    pair_root: Path,
    results_root: Path,
    event_seeds: tuple[int, ...],
) -> tuple[
    dict[tuple[str, str, str, int], v1.OutputMetrics],
    dict[tuple[str, str], dict[str, Any]],
]:
    expected_groups = _expected_group_directories(results_root, event_seeds)
    if not results_root.is_dir():
        raise ValueError(f"results root does not exist: {results_root}")
    discovered = {
        path for path in results_root.rglob("event_seed_*") if path.is_dir()
    }
    missing = sorted(expected_groups - discovered)
    unexpected = sorted(discovered - expected_groups)
    if missing or unexpected:
        details = []
        if missing:
            details.append(
                f"missing {len(missing)} groups (first: {missing[0]})")
        if unexpected:
            details.append(
                f"unexpected {len(unexpected)} groups (first: {unexpected[0]})")
        raise ValueError("; ".join(details))

    pair_manifests = {}
    outputs = {}
    for family in FAMILIES:
        for env_short in ENVS:
            env = f"{env_short}-v2"
            pair_dir = pair_directory(
                pair_root, family, env, EXPECTED_TRAINING_SEED)
            pair_manifests[(family, env_short)] = validate_pair_provenance(
                pair_dir, family, env, EXPECTED_TRAINING_SEED)
            for event_seed in event_seeds:
                group = validate_group_bundle(
                    group_directory(results_root, family, env, event_seed),
                    pair_dir, family, env, EXPECTED_TRAINING_SEED, event_seed)
                for source, metrics in group.items():
                    outputs[(family, env_short, source, event_seed)] = metrics
    if len(pair_manifests) != 4 or len(outputs) != 120:
        raise ValueError(
            "internal cardinality error: expected four pairs and 120 outputs, "
            f"found {len(pair_manifests)} and {len(outputs)}")
    return outputs, pair_manifests


def render_report(
    outputs: dict[tuple[str, str, str, int], v1.OutputMetrics],
    pair_root: Path,
    results_root: Path,
    event_seeds: tuple[int, ...],
) -> str:
    lines = [
        "# BAPR-v3 v2 shared-fork budget audit",
        "",
        f"Validated **4/4** exact producer pair manifests and **120/120** "
        f"evaluation outputs under `{results_root}`. Every six-controller "
        "event group used one recorded runtime, source snapshot, host/GPU, "
        "and paired event stream.",
        "",
        f"Producer root: `{pair_root}`. Final checkpoints are iteration "
        f"`{EXPECTED_FINAL_ITERATION}` / next iteration "
        f"`{EXPECTED_CHECKPOINT_NEXT_ITER}` / total steps "
        f"`{EXPECTED_CHECKPOINT_TOTAL_STEPS}`. Shared restarts are exactly "
        f"iteration `{EXPECTED_FORK_START_ITERATION}` / steps "
        f"`{EXPECTED_FORK_START_STEPS}`.",
        "",
        "**Causal eligibility: PASS.** The robust and oracle arms share one "
        "bit-identified iteration-699 snapshot, one runtime/source identity, "
        "and an identical first physical rollout at the common restart.",
        "",
        "Gate semantics: paired mean differences must be strictly positive; "
        "95% paired t intervals and stream-wise wins are descriptive.",
        "",
    ]
    pair_results = []
    family_results: dict[str, list[bool]] = {family: [] for family in FAMILIES}
    for family in FAMILIES:
        for env in ENVS:
            pair_lines, pair_pass = v1.render_pair(
                outputs, family, env, event_seeds)
            lines.extend(pair_lines)
            pair_results.append(pair_pass)
            family_results[family].append(pair_pass)

    lines.extend(["## Preregistered decision", ""])
    for family in FAMILIES:
        passed = all(family_results[family])
        lines.append(
            f"- `{family}` passes both environments: "
            f"**{'PASS' if passed else 'FAIL'}**")
    numerical_pass = all(pair_results)
    lines.extend([
        f"- All four family/environment numerical gates pass: "
        f"**{'PASS' if numerical_pass else 'FAIL'}**",
        "- Shared-checkpoint/common-restart provenance gate: **PASS**",
        f"- Learned-latent authorization gate: "
        f"**{'PASS' if numerical_pass else 'FAIL'}**",
        "",
        "## Scope caveat",
        "",
        "The five event seeds are paired evaluation streams for one trained "
        "policy seed (`s0`); they are not five independent training seeds.",
    ])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", type=Path, default=DEFAULT_PAIR_ROOT)
    parser.add_argument(
        "--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--event-seed", action="append", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    event_seeds = tuple(args.event_seed or DEFAULT_EVENT_SEEDS)
    if event_seeds != DEFAULT_EVENT_SEEDS:
        raise SystemExit(
            f"the preregistered ordered event seeds are {DEFAULT_EVENT_SEEDS}")
    try:
        outputs, _ = load_complete_audit(
            args.pair_root.resolve(), args.results_root.resolve(), event_seeds)
    except ValueError as exc:
        raise SystemExit(f"AUDIT INCOMPLETE OR INVALID: {exc}") from exc
    sys.stdout.write(render_report(
        outputs, args.pair_root.resolve(), args.results_root.resolve(),
        event_seeds))


if __name__ == "__main__":
    main()
