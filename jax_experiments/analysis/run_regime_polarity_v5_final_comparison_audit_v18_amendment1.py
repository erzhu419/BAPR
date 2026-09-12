"""Run the frozen v18 audit with its v5 artifact-path API restored."""
from __future__ import annotations

import argparse
from pathlib import Path

from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_model_v5 as v5_model,
)
from jax_experiments.analysis import (
    regime_polarity_v5_final_comparison_v18 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_v5_final_comparison_audit_v18 as audit,
)


AMENDMENT_REGISTRATION = protocol.REGISTRATION_ROOT / "audit_amendment1.json"
AMENDMENT_SOURCE = Path(__file__).resolve()


def amendment_registration_payload() -> dict:
    return {
        "schema": "bapr.regime-polarity-v5-final-audit-amendment.v18.1",
        "status": "registered",
        "created_after_training_before_successful_audit_retry": True,
        "parent_registration": protocol.file_record(
            protocol.REGISTRATION_PATH),
        "failure": {
            "exception": "AttributeError",
            "missing_attributes": ["MODEL_MANIFEST", "MODEL_PATH"],
            "module": v5_model.__name__,
        },
        "correction": (
            "Expose the already-preregistered v5 system-ID manifest and "
            "parameter paths as process-local loader aliases before invoking "
            "the frozen v18 audit."
        ),
        "scientific_protocol_changes": [],
        "source": protocol.file_record(AMENDMENT_SOURCE),
    }


def create_amendment_registration() -> dict:
    protocol.validate_registration()
    payload = amendment_registration_payload()
    if AMENDMENT_REGISTRATION.is_file():
        if protocol.read_json(AMENDMENT_REGISTRATION) != payload:
            raise ValueError("existing v18 audit amendment registration changed")
    else:
        protocol.write_json_atomic(AMENDMENT_REGISTRATION, payload)
    return payload


def install_v5_artifact_path_aliases() -> None:
    v5_model.MODEL_MANIFEST = v5_model.protocol.MODEL_MANIFEST
    v5_model.MODEL_PATH = v5_model.protocol.MODEL_PATH


def annotate_audit(seed: int) -> None:
    result_path = protocol.audit_result(seed)
    manifest_path = protocol.audit_manifest(seed)
    payload = protocol.read_json(result_path)
    manifest = protocol.read_json(manifest_path)
    amendment = protocol.file_record(AMENDMENT_REGISTRATION)
    payload["execution_amendment"] = amendment
    protocol.write_json_atomic(result_path, payload)
    manifest["execution_amendment"] = amendment
    manifest["audit"] = protocol.file_record(result_path)
    protocol.write_json_atomic(manifest_path, manifest)
    audit.validate_audit(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    create_amendment_registration()
    install_v5_artifact_path_aliases()
    audit.run(args.seed)
    annotate_audit(args.seed)
    print(f"V18 FINAL AUDIT AMENDMENT1 COMPLETE: seed={args.seed}", flush=True)


if __name__ == "__main__":
    main()
