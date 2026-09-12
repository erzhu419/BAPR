"""Immutable audit-path amendment for the v19 specialist screen."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_specialist_stability_v19 as original,
)


ROOT = original.ROOT
AMENDMENT_VERSION = "v19-amendment1-bundle-signature-lookup"
REGISTRATION_ROOT = original.REGISTRATION_ROOT
REGISTRATION_PATH = REGISTRATION_ROOT / "amendment1.json"
REPORT = (
    ROOT / "reports"
    / "regime_polarity_specialist_stability_v19_amendment1_2026-09-08.md"
)
AUDIT_PROVENANCE_NAME = "amendment1_provenance.json"
REGISTRATION_SCHEMA = "bapr.specialist-stability-amendment-registration.v19a1"


def registration_source_paths() -> tuple[Path, ...]:
    return (
        ROOT / "jax_experiments/analysis/regime_polarity_specialist_stability_v19_amendment1.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_specialist_stability_audit_v19_amendment1.py",
        ROOT / "scripts/resubmit_regime_polarity_specialist_stability_audits_v19_amendment1.py",
        original.REGISTRATION_PATH,
        REPORT,
    )


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def registration_payload() -> dict[str, Any]:
    original.validate_registration()
    paths = tuple(path.resolve() for path in registration_source_paths())
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"v19 amendment sources missing: {missing}")
    return {
        "schema": REGISTRATION_SCHEMA,
        "status": "registered",
        "created_before_replacement_audits": True,
        "amendment_version": AMENDMENT_VERSION,
        "original_registration": original.file_record(
            original.REGISTRATION_PATH),
        "scope": {
            "training_changed": False,
            "evaluation_changed": False,
            "event_streams_changed": False,
            "decision_gate_changed": False,
            "change": (
                "validate the staged specialist protocol signature at its "
                "bundle path instead of the GPU-only training-run path"
            ),
        },
        "source_records": {
            _relative(path): original.file_record(path) for path in paths
        },
    }


def create_registration() -> dict[str, Any]:
    payload = registration_payload()
    REGISTRATION_ROOT.mkdir(parents=True, exist_ok=True)
    if REGISTRATION_PATH.is_file():
        if original.read_json(REGISTRATION_PATH) != payload:
            raise ValueError("existing v19 amendment registration changed")
    else:
        original.write_json_atomic(REGISTRATION_PATH, payload)
    return validate_registration()


def validate_registration() -> dict[str, Any]:
    if not REGISTRATION_PATH.is_file():
        raise FileNotFoundError(
            f"missing v19 amendment registration: {REGISTRATION_PATH}")
    payload = original.read_json(REGISTRATION_PATH)
    if payload != registration_payload():
        raise ValueError("v19 amendment registration or sources changed")
    return payload
