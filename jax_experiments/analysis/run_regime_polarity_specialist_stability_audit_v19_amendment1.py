"""Run v19 audit using protocol signatures from staged policy bundles."""
from __future__ import annotations

import argparse

from jax_experiments.analysis import (
    regime_polarity_specialist_stability_v19 as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_stability_v19_amendment1 as amendment,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_stability_audit_v19 as audit,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_stability_v19 as producer,
)


def _validate_bundle_signature(
    variant: str, seed: int, mode: int,
):
    variant = protocol.require_variant(variant)
    seed = protocol.require_training_seed(seed)
    mode = protocol.require_mode(mode)
    path = (
        protocol.bundle_dir(variant, seed, mode)
        / "logs" / "protocol_signature.json"
    )
    signature = protocol.read_json(path)
    config = signature.get("config") or {}
    mismatches = {
        key: {"actual": config.get(key), "expected": expected}
        for key, expected in producer.expected_config(
            variant, seed, mode).items()
        if config.get(key) != expected
    }
    if (
        variant != "critic_warmup"
        and config.get("sac_actor_update_after") is not None
    ):
        mismatches["sac_actor_update_after"] = {
            "actual": config.get("sac_actor_update_after"),
            "expected": None,
        }
    for key, expected in {
        "checkpoint_loaded": True,
        "start_iteration": protocol.SOURCE_NEXT_ITERATION,
        "total_steps_at_start": protocol.SOURCE_TOTAL_STEPS,
    }.items():
        if signature.get(key) != expected:
            mismatches[key] = {
                "actual": signature.get(key), "expected": expected}
    if mismatches:
        raise ValueError(
            f"v19 staged specialist signature mismatch: {mismatches}")
    return signature


def run(variant: str, seed: int) -> None:
    variant = protocol.require_variant(variant)
    seed = protocol.require_training_seed(seed)
    amendment.validate_registration()
    producer.validate_signature = _validate_bundle_signature
    audit.run(variant, seed)
    destination = protocol.audit_dir(variant, seed)
    protocol.write_json_atomic(
        destination / amendment.AUDIT_PROVENANCE_NAME,
        {
            "schema": "bapr.specialist-stability-audit-amendment.v19a1",
            "status": "complete",
            "variant": variant,
            "training_seed": seed,
            "amendment_registration": protocol.file_record(
                amendment.REGISTRATION_PATH),
            "audit_manifest": protocol.file_record(
                protocol.audit_manifest(variant, seed)),
            "signature_source": "staged specialist policy bundle",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=protocol.VARIANTS, required=True)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.variant, args.seed)


if __name__ == "__main__":
    main()
