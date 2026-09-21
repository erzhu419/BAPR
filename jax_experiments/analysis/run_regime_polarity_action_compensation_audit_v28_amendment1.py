"""Run the frozen V28 audit with the registered v5 estimator implementation."""
from __future__ import annotations

import argparse
from pathlib import Path

from jax_experiments.analysis import (
    regime_polarity_action_compensation_v28 as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_model_v5 as estimator_model,
)
from jax_experiments.analysis import (
    run_regime_polarity_action_compensation_audit_v28 as audit,
)


AMENDMENT_REGISTRATION = protocol.REGISTRATION_ROOT / "amendment1.json"
AMENDMENT_SOURCE = Path(__file__).resolve()
SUBMIT_SOURCE = (
    protocol.ROOT / "scripts"
    / "resubmit_regime_polarity_action_compensation_v28_amendment1.py"
)
ESTIMATOR_SOURCE = Path(estimator_model.__file__).resolve()


def amendment_registration_payload() -> dict:
    return {
        "schema": "bapr.regime-polarity-action-compensation-amendment.v28.1",
        "status": "registered",
        "created_after_failed_execution_before_successful_policy_audit": True,
        "parent_registration": protocol.file_record(
            protocol.REGISTRATION_PATH),
        "failed_task_ids": [
            "t93039", "t93040", "t93041", "t93042", "t93043",
        ],
        "failure": {
            "exception": "AttributeError",
            "missing_attribute": "make_estimator",
            "incorrect_module": (
                "jax_experiments.analysis."
                "regime_polarity_specialist_expected_action_system_id_v5"
            ),
        },
        "correction": (
            "Bind the frozen V28 audit's estimator symbol to the existing v5 "
            "model-construction module before running the unchanged audit."
        ),
        "discarded_outputs": (
            "All five failed runs exited before writing an audit result or "
            "manifest."
        ),
        "scientific_protocol_changes": [],
        "source_records": {
            "amendment_runner": protocol.file_record(AMENDMENT_SOURCE),
            "resubmission_script": protocol.file_record(SUBMIT_SOURCE),
            "estimator_implementation": protocol.file_record(ESTIMATOR_SOURCE),
        },
    }


def create_amendment_registration() -> dict:
    protocol.validate_registration()
    payload = amendment_registration_payload()
    if AMENDMENT_REGISTRATION.is_file():
        if protocol.read_json(AMENDMENT_REGISTRATION) != payload:
            raise ValueError("existing V28 amendment registration changed")
    else:
        protocol.write_json_atomic(AMENDMENT_REGISTRATION, payload)
    return payload


def install_estimator_module() -> None:
    expected = (
        "jax_experiments.analysis."
        "regime_polarity_specialist_expected_action_system_id_v5"
    )
    current = audit.v5_model.__name__
    if current == estimator_model.__name__:
        return
    if current != expected or hasattr(audit.v5_model, "make_estimator"):
        raise ValueError(f"unexpected frozen V28 estimator binding: {current}")
    audit.v5_model = estimator_model


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
    if not args.resume:
        raise SystemExit("--resume is required for idempotent execution")
    create_amendment_registration()
    install_estimator_module()
    audit.run(args.seed)
    annotate_audit(args.seed)
    print(f"V28 AUDIT AMENDMENT1 COMPLETE: seed={args.seed}", flush=True)


if __name__ == "__main__":
    main()
