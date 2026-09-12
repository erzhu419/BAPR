"""Run one sealed multi-seed confirmation audit for the fixed adapter bank."""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import jax

from jax_experiments.analysis import regime_adapter_confirmation as protocol
from jax_experiments.analysis import regime_adapter_fork as branch_protocol
from jax_experiments.analysis import run_regime_adapter_audit as development
from jax_experiments.analysis.regime_adapter_policy_bank import (
    load_policy_bank,
)


OUTPUT_FILES = development.OUTPUT_FILES
CASES = (
    development.EvaluationCase("robust_continue", "robust"),
    development.EvaluationCase(
        "frozen_base", "bank", (-1,) * len(protocol.MODES)),
    development.EvaluationCase(
        "identity_adapter", "bank", protocol.CONTROLLER_MAP),
    *(development.EvaluationCase(
        f"fixed_adapter_{mode}", "bank",
        (mode,) * len(protocol.MODES)) for mode in protocol.MODES),
)


def _portable_manifest_key(path: str | Path) -> str:
    parts = Path(path).parts
    try:
        index = parts.index("jax_experiments")
    except ValueError as error:
        raise ValueError(
            f"training provenance is outside jax_experiments: {path}"
        ) from error
    return Path(*parts[index:]).as_posix()


def _normalize_training_manifests(records: dict) -> dict:
    normalized = {
        _portable_manifest_key(path): record
        for path, record in records.items()
    }
    if len(normalized) != len(records):
        raise ValueError("duplicate portable training provenance key")
    return normalized


def _current_training_manifests(seed: int) -> dict:
    return {
        _portable_manifest_key(path): protocol.file_record(path)
        for path in protocol.required_training_paths(seed)
        if path.name == branch_protocol.BUNDLE_MANIFEST_NAME
    }


def validate_audit(seed: int, event_seed: int) -> dict:
    seed = protocol.require_training_seed(seed)
    event_seed = protocol.require_event_seed(event_seed)
    destination = protocol.audit_dir(seed, event_seed)
    payload = protocol.read_json(
        destination / protocol.AUDIT_MANIFEST_NAME)
    expected_files = {
        f"{case.label}/{filename}"
        for case in CASES for filename in OUTPUT_FILES
    }
    if (payload.get("schema") != protocol.AUDIT_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("identity") != protocol.identity(seed, event_seed)
            or set(payload.get("files") or {}) != expected_files):
        raise ValueError(f"invalid adapter confirmation audit: {destination}")
    for relative, expected in payload["files"].items():
        path = destination / relative
        if not path.is_file() or protocol.file_record(path) != expected:
            raise ValueError(f"confirmation audit file changed: {path}")
    for case in CASES:
        development._validate_case(
            destination / case.label, seed, protocol.DELTA,
            event_seed, case)
    recorded_bundles = _normalize_training_manifests(
        payload.get("training_bundle_manifests") or {})
    if recorded_bundles != _current_training_manifests(seed):
        raise ValueError("confirmation training provenance changed")
    return payload


def run(seed: int, event_seed: int) -> None:
    seed = protocol.require_training_seed(seed)
    event_seed = protocol.require_event_seed(event_seed)
    destination = protocol.audit_dir(seed, event_seed)
    manifest = destination / protocol.AUDIT_MANIFEST_NAME
    if manifest.is_file():
        validate_audit(seed, event_seed)
        print(f"REGIME ADAPTER CONFIRMATION ALREADY COMPLETE: {destination}")
        return

    robust = development._load_robust(seed)
    bank = load_policy_bank(seed, protocol.DELTA)
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp.", dir=destination.parent))
    try:
        records = {}
        for case in CASES:
            output = temporary / case.label
            print(
                f"REGIME ADAPTER CONFIRMATION seed={seed} "
                f"event={event_seed} case={case.label}", flush=True)
            development._evaluate(
                seed, protocol.DELTA, event_seed, case,
                output, robust, bank)
            for filename in OUTPUT_FILES:
                relative = f"{case.label}/{filename}"
                records[relative] = protocol.file_record(
                    temporary / relative)
            jax.clear_caches()
        payload = {
            "schema": protocol.AUDIT_SCHEMA,
            "status": "complete",
            "identity": protocol.identity(seed, event_seed),
            "routing_rule": (
                "Fixed identity map [0,1,2,3] selected once on development "
                "seed 8; no per-seed or confirmation-stream calibration."),
            "training_bundle_manifests": _current_training_manifests(seed),
            "files": records,
        }
        protocol.write_json_atomic(
            temporary / protocol.AUDIT_MANIFEST_NAME, payload)
        os.replace(temporary, destination)
    finally:
        bank.close()
        robust[4].cleanup()
        if hasattr(robust[3], "close"):
            robust[3].close()
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_audit(seed, event_seed)
    print(f"REGIME ADAPTER CONFIRMATION COMPLETE: {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, choices=protocol.TRAINING_SEEDS, required=True)
    parser.add_argument(
        "--event-seed", type=int, choices=protocol.EVENT_SEEDS,
        required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for scheduler staging")
    run(args.seed, args.event_seed)


if __name__ == "__main__":
    main()
