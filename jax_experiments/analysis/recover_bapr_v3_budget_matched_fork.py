#!/usr/bin/env python3
"""Recover a BAPR-v3 fork with its immutable source archive.

The default path is finalize-only: it refuses partial training and only
re-enters the original protocol when the shared base and both branches already
have their exact preregistered final checkpoints and complete logs. The
explicit ``--resume-partial`` path may continue an incomplete branch, but only
through the source archive recorded by the original protocol checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


BASE_ITERATION = 699
BASE_STEPS = 2_800_000
FINAL_ITERATION = 1399
FINAL_STEPS = 5_600_000
SOURCE_MANIFEST = Path("provenance/source_manifest.json")
SOURCE_ARCHIVE = Path("provenance/source_snapshot.tar.gz")
RUNNER_MODULE = "jax_experiments.analysis.run_bapr_v3_budget_matched_fork"


def file_sha256(path: Path) -> str:
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


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def read_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def validate_checkpoint(
    run_dir: Path, expected_iteration: int, expected_steps: int,
) -> None:
    checkpoint_dir = run_dir / "checkpoints"
    for filename in ("params.pkl", "train_state.pkl", "replay_buffer.npz"):
        path = checkpoint_dir / filename
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"missing or empty checkpoint: {path}")
    state = read_pickle(checkpoint_dir / "train_state.pkl")
    if not isinstance(state, dict):
        raise RuntimeError(f"invalid train state: {run_dir}")
    identity = (int(state.get("iteration", -1)),
                int(state.get("total_steps", -1)))
    expected = (expected_iteration, expected_steps)
    if identity != expected:
        raise RuntimeError(
            f"{run_dir.name} checkpoint is {identity}, expected {expected}; "
            "recovery is finalize-only and will not train")


def validate_final_logs(run_dir: Path) -> None:
    log_dir = run_dir / "logs"
    iterations = np.asarray(np.load(log_dir / "iteration.npy")).reshape(-1)
    total_steps = np.asarray(np.load(log_dir / "total_steps.npy")).reshape(-1)
    if (len(iterations) != FINAL_ITERATION + 1
            or int(iterations[-1]) != FINAL_ITERATION):
        raise RuntimeError(f"incomplete iteration log: {run_dir}")
    if (len(total_steps) != FINAL_ITERATION + 1
            or int(total_steps[-1]) != FINAL_STEPS):
        raise RuntimeError(f"incomplete total-steps log: {run_dir}")
    for filename in ("protocol_signature.json", "resume_boundary_audit.json"):
        path = log_dir / filename
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"missing final audit input: {path}")


def validate_source_archive(pair_dir: Path) -> tuple[dict[str, Any], Path]:
    state = read_pickle(pair_dir / "protocol_checkpoint.pkl")
    if not isinstance(state, dict):
        raise RuntimeError("invalid protocol checkpoint")
    manifest = read_json(pair_dir / SOURCE_MANIFEST)
    expected_source = state.get("source")
    if not isinstance(expected_source, dict):
        raise RuntimeError("protocol checkpoint has no source identity")
    if (manifest.get("files") != expected_source.get("files")
            or manifest.get("sha256") != expected_source.get("sha256")):
        raise RuntimeError("source manifest differs from protocol checkpoint")
    files = manifest.get("files")
    if not isinstance(files, dict) or canonical_sha256(files) != manifest.get("sha256"):
        raise RuntimeError("source manifest digest is invalid")

    archive_path = pair_dir / SOURCE_ARCHIVE
    archive_state = state.get("source_archive")
    if not isinstance(archive_state, dict):
        raise RuntimeError("protocol checkpoint has no source archive identity")
    if (archive_state.get("path") != SOURCE_ARCHIVE.as_posix()
            or file_sha256(archive_path) != archive_state.get("sha256")):
        raise RuntimeError("source archive differs from protocol checkpoint")

    with tarfile.open(archive_path, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        if [member.name for member in members] != sorted(files):
            raise RuntimeError("source archive members differ from manifest")
        for member in members:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"unsafe archive member: {member.name}")
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"cannot read archive member: {member.name}")
            payload = source.read()
            expected = files[member.name]
            if (len(payload) != int(expected["size"])
                    or hashlib.sha256(payload).hexdigest() != expected["sha256"]):
                raise RuntimeError(f"source archive content mismatch: {member.name}")
    return manifest, archive_path


def extract_source(
    archive_path: Path, manifest: dict[str, Any], destination: Path,
) -> None:
    files = manifest["files"]
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"unsafe archive member: {member.name}")
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"cannot read archive member: {member.name}")
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = source.read()
            expected = files[member.name]
            if (len(payload) != int(expected["size"])
                    or hashlib.sha256(payload).hexdigest() != expected["sha256"]):
                raise RuntimeError(f"source archive content mismatch: {member.name}")
            target.write_bytes(payload)


def validate_finalize_only(pair_dir: Path) -> tuple[dict[str, Any], Path]:
    if (pair_dir / "provenance/pair_manifest.json").exists() != (
            pair_dir / "pair_checkpoint_complete.pkl").exists():
        raise RuntimeError("partial completion artifacts exist; refusing recovery")
    validate_checkpoint(
        pair_dir / "shared_base", BASE_ITERATION, BASE_STEPS)
    for branch in ("robust_long", "oracle_direct"):
        run_dir = pair_dir / branch
        validate_checkpoint(run_dir, FINAL_ITERATION, FINAL_STEPS)
        validate_final_logs(run_dir)
    return validate_source_archive(pair_dir)


def validate_archive_resume(
    pair_dir: Path, family: str, env: str, seed: int,
) -> tuple[dict[str, Any], Path]:
    manifest_path = pair_dir / "provenance/pair_manifest.json"
    sentinel_path = pair_dir / "pair_checkpoint_complete.pkl"
    if manifest_path.exists() != sentinel_path.exists():
        raise RuntimeError("partial completion artifacts exist; refusing recovery")
    if manifest_path.exists():
        return validate_finalize_only(pair_dir)

    state = read_pickle(pair_dir / "protocol_checkpoint.pkl")
    if not isinstance(state, dict):
        raise RuntimeError("invalid protocol checkpoint")
    expected_identity = {"family": family, "env": env, "seed": seed}
    if state.get("identity") != expected_identity:
        raise RuntimeError(
            f"protocol identity {state.get('identity')!r} differs from "
            f"requested {expected_identity!r}")
    validate_checkpoint(
        pair_dir / "shared_base", BASE_ITERATION, BASE_STEPS)
    return validate_source_archive(pair_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-dir", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--resume", action="store_true",
        help="Required scheduler marker.")
    parser.add_argument(
        "--resume-partial", action="store_true",
        help="Allow the archived runner to continue an incomplete branch. "
             "Without this flag recovery remains finalize-only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.resume:
        raise SystemExit("--resume is required")
    pair_dir = args.pair_dir.resolve()
    if args.resume_partial:
        manifest, archive_path = validate_archive_resume(
            pair_dir, args.family, args.env, args.seed)
        precheck = "ARCHIVE-RESUME PRECHECK PASSED"
    else:
        manifest, archive_path = validate_finalize_only(pair_dir)
        precheck = "FINALIZE-ONLY PRECHECK PASSED"
    print(
        f"{precheck}: base={BASE_ITERATION}, "
        f"target_branches={FINAL_ITERATION}, steps={FINAL_STEPS}, "
        f"source={manifest['sha256']}",
        flush=True)

    with tempfile.TemporaryDirectory(prefix="bapr-v3-fork-source-") as tmp:
        source_root = Path(tmp)
        extract_source(archive_path, manifest, source_root)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(source_root)
        command = [
            sys.executable, "-u", "-m", RUNNER_MODULE,
            "--family", args.family,
            "--env", args.env,
            "--seed", str(args.seed),
            "--save-root", str(pair_dir.parent),
            "--resume",
        ]
        subprocess.run(
            command, cwd=source_root, env=environment, check=True)

    validate_finalize_only(pair_dir)
    manifest_path = pair_dir / "provenance/pair_manifest.json"
    sentinel_path = pair_dir / "pair_checkpoint_complete.pkl"
    if not manifest_path.is_file() or not sentinel_path.is_file():
        raise RuntimeError("archived runner exited without completion artifacts")
    sentinel = read_pickle(sentinel_path)
    if (not isinstance(sentinel, dict)
            or sentinel.get("status") != "complete"
            or sentinel.get("final_next_iteration") != FINAL_ITERATION + 1
            or sentinel.get("final_total_steps") != FINAL_STEPS
            or sentinel.get("pair_manifest_sha256") != file_sha256(manifest_path)):
        raise RuntimeError("completion sentinel failed post-recovery validation")
    print(
        f"RECOVERY COMPLETE: {pair_dir} "
        f"manifest_sha256={sentinel['pair_manifest_sha256']}",
        flush=True)


if __name__ == "__main__":
    main()
