"""Run the frozen v12 audit with its producer-validator interface corrected."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_robust_warmstart_confirmation_v12 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_robust_warmstart_confirmation_audit_v12 as audit,
)
from jax_experiments.analysis import (
    run_regime_polarity_robust_warmstart_specialist_v12 as producer,
)


FIX_SCHEMA = "bapr.robust-warmstart-specialist-audit-fix.v12.1"
FIX_REGISTRATION_PATH = protocol.REGISTRATION_ROOT / "audit_fix1_registration.json"
SOURCE_PATH = Path(__file__).resolve()


def registration_payload() -> dict[str, Any]:
    protocol.validate_registration()
    return {
        "schema": FIX_SCHEMA,
        "status": "registered",
        "created_before_fixed_audit": True,
        "protocol_registration": protocol.file_record(
            protocol.REGISTRATION_PATH),
        "source": protocol.file_record(SOURCE_PATH),
        "scope": (
            "audit-only call adapter: validate_bundle(variant, seed, mode) "
            "delegates to the frozen v12 validate_bundle(seed, mode)"
        ),
        "scientific_protocol_changed": False,
        "training_changed": False,
    }


def create_fix_registration() -> dict[str, Any]:
    payload = registration_payload()
    if FIX_REGISTRATION_PATH.is_file():
        if protocol.read_json(FIX_REGISTRATION_PATH) != payload:
            raise ValueError("existing v12 audit-fix registration changed")
    else:
        protocol.write_json_atomic(FIX_REGISTRATION_PATH, payload)
    return validate_fix_registration()


def validate_fix_registration() -> dict[str, Any]:
    if not FIX_REGISTRATION_PATH.is_file():
        raise FileNotFoundError(
            f"missing v12 audit-fix registration: {FIX_REGISTRATION_PATH}")
    payload = protocol.read_json(FIX_REGISTRATION_PATH)
    if payload != registration_payload():
        raise ValueError("v12 audit-fix registration or source changed")
    return payload


def run(seed: int) -> None:
    seed = protocol.require_training_seed(seed)
    protocol.validate_registration()
    validate_fix_registration()
    frozen_validate = producer.validate_bundle

    def compatible_validate_bundle(
        variant: str, requested_seed: int, mode: int,
    ) -> dict[str, Any]:
        protocol.require_variant(variant)
        return frozen_validate(requested_seed, mode)

    producer.validate_bundle = compatible_validate_bundle
    try:
        audit.run(seed)
    finally:
        producer.validate_bundle = frozen_validate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.seed)


if __name__ == "__main__":
    main()
