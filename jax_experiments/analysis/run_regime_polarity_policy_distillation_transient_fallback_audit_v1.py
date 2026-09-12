"""Audit the development-selected transient fallback on one sealed event."""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from jax_experiments.analysis import (
    regime_polarity_policy_distillation_control_model as model_lib,
)
from jax_experiments.analysis import (
    regime_polarity_policy_distillation_transient_fallback_v1 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_policy_distillation_transient_fallback_screen_v1 as screen,
)
from jax_experiments.analysis import (
    select_regime_polarity_policy_distillation_transient_fallback_v1 as selection,
)


def _selected_arms() -> tuple[str, ...]:
    selection.validate_selection()
    payload = protocol.read_json(protocol.selection_json())
    selected = protocol.require_config(payload["selected_config"])
    return (*protocol.BASELINE_ARMS, selected.name)


def validate_audit(event_seed: int):
    event_seed = protocol.require_audit_event_seed(event_seed)
    destination = protocol.audit_dir(event_seed)
    identity = protocol.common_identity(
        "transient_fallback_independent_audit", event_seed)
    manifest = protocol.read_json(destination / "audit_manifest.json")
    expected_selection = protocol.file_record(protocol.selection_manifest())
    if (manifest.get("schema") != protocol.AUDIT_SCHEMA
            or manifest.get("status") != "complete"
            or manifest.get("identity") != identity
            or manifest.get("selection_manifest") != expected_selection
            or manifest.get("source_bundles")
            != model_lib.source_records(protocol.TEACHER_GROUP)
            or manifest.get("student_manifest")
            != protocol.FROZEN_STUDENT_MANIFEST_RECORD
            or manifest.get("student_parameters")
            != protocol.FROZEN_STUDENT_PARAMETER_RECORD
            or manifest.get("result_file")
            != protocol.file_record(destination / "results.json")):
        raise ValueError(f"invalid transient-fallback audit {destination}")
    result = protocol.read_json(destination / "results.json")
    screen._validate_result(
        result,
        identity,
        _selected_arms(),
        protocol.AUDIT_EVENT_SCHEMA,
    )
    return manifest


def run(event_seed: int) -> None:
    event_seed = protocol.require_audit_event_seed(event_seed)
    destination = protocol.audit_dir(event_seed)
    if (destination / "audit_manifest.json").is_file():
        try:
            validate_audit(event_seed)
        except (KeyError, OSError, TypeError, ValueError):
            pass
        else:
            print(f"TRANSIENT FALLBACK AUDIT ALREADY COMPLETE: {destination}")
            return

    selection.validate_selection()
    identity = protocol.common_identity(
        "transient_fallback_independent_audit", event_seed)
    result = {
        "schema": protocol.AUDIT_EVENT_SCHEMA,
        "status": "complete",
        "identity": identity,
        "selection_manifest": protocol.file_record(
            protocol.selection_manifest()),
        "switching": screen.evaluate_event(event_seed, _selected_arms()),
    }
    if destination.exists() or destination.is_symlink():
        shutil.rmtree(destination) if destination.is_dir() else destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp.", dir=destination.parent))
    try:
        protocol.write_json_atomic(temporary / "results.json", result)
        manifest = {
            "schema": protocol.AUDIT_SCHEMA,
            "status": "complete",
            "identity": identity,
            "selection_manifest": protocol.file_record(
                protocol.selection_manifest()),
            "source_bundles": model_lib.source_records(protocol.TEACHER_GROUP),
            "frozen_estimator_manifest": protocol.file_record(
                protocol.frozen.development.ensemble.final.MODEL_MANIFEST),
            "frozen_estimator_parameters": protocol.file_record(
                protocol.frozen.development.ensemble.final.MODEL_PATH),
            "student_manifest": protocol.FROZEN_STUDENT_MANIFEST_RECORD,
            "student_parameters": protocol.FROZEN_STUDENT_PARAMETER_RECORD,
            "result_file": protocol.file_record(temporary / "results.json"),
        }
        protocol.write_json_atomic(temporary / "audit_manifest.json", manifest)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_audit(event_seed)
    print(f"TRANSIENT FALLBACK AUDIT COMPLETE: {destination}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event-seed", type=int, choices=protocol.AUDIT_EVENT_SEEDS,
        required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.event_seed)


if __name__ == "__main__":
    main()
