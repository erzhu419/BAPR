"""Audit the paired v19 specialist initialization arms."""
from __future__ import annotations

import argparse
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

from flax import nnx

from jax_experiments.analysis import (
    regime_polarity_specialist_stability_v19 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_robust_warmstart_confirmation_audit_v12
    as switching,
)
from jax_experiments.analysis import (
    run_regime_polarity_robust_warmstart_specialist_audit_v11 as base,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_stability_v19 as producer,
)


def _bind() -> None:
    base.protocol = protocol
    base.producer = producer
    base._bind()
    switching.protocol = protocol
    switching.base = base
    switching.utility_audit.protocol = protocol


def _load_source(seed: int) -> dict[str, Any]:
    config, agent = producer._load_source(seed)
    return {
        "config": config,
        "agent": agent,
        "policy_graphdef": nnx.graphdef(agent.policy),
        "policy_params": nnx.state(agent.policy, nnx.Param),
        "context_graphdef": None,
        "context_params": None,
    }


def _load_controllers(
    variant: str, seed: int,
) -> dict[str, dict[str, Any]]:
    _bind()
    robust = _load_source(seed)
    controllers = {"robust_sac": robust}
    for mode in protocol.MODES:
        controllers[f"specialist_{mode}"] = base._load_specialist(
            variant, seed, mode, robust)
    return controllers


def _identity(variant: str, seed: int) -> dict[str, Any]:
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "benchmark_role": "specialist_training_stability_development_audit",
        "variant": protocol.require_variant(variant),
        "training_seed": protocol.require_training_seed(seed),
        "calibration_event_seeds": list(protocol.CALIBRATION_EVENT_SEEDS),
        "stationary_holdout_event_seeds": list(
            protocol.STATIONARY_HOLDOUT_EVENT_SEEDS),
        "switching_event_seeds": list(protocol.SWITCHING_EVENT_SEEDS),
        "switching_schedules": {
            str(key): list(value)
            for key, value in protocol.SWITCHING_SCHEDULES.items()
        },
    }


def validate_audit(variant: str, seed: int) -> dict[str, Any]:
    variant = protocol.require_variant(variant)
    seed = protocol.require_training_seed(seed)
    manifest = protocol.read_json(protocol.audit_manifest(variant, seed))
    payload = protocol.read_json(protocol.audit_result(variant, seed))
    expected_identity = _identity(variant, seed)
    if (
        manifest.get("schema") != protocol.AUDIT_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("identity") != expected_identity
        or manifest.get("registration")
        != protocol.file_record(protocol.REGISTRATION_PATH)
        or manifest.get("source_bundle_manifest")
        != protocol.file_record(protocol.source_manifest(seed))
        or manifest.get("specialist_bundles")
        != protocol.bundle_records(variant, seed)
        or manifest.get("audit")
        != protocol.file_record(protocol.audit_result(variant, seed))
        or payload.get("schema") != protocol.AUDIT_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("identity") != expected_identity
        or payload.get("source_bundle_manifest")
        != protocol.file_record(protocol.source_manifest(seed))
        or payload.get("specialist_bundles")
        != protocol.bundle_records(variant, seed)
    ):
        raise ValueError("invalid v19 specialist stability audit")
    for split_name, event_seeds in (
        ("calibration_events", protocol.CALIBRATION_EVENT_SEEDS),
        ("stationary_holdout", protocol.STATIONARY_HOLDOUT_EVENT_SEEDS),
    ):
        split = payload.get(split_name) or {}
        if set(split) != {str(value) for value in event_seeds}:
            raise ValueError(f"v19 {split_name} is incomplete")
        for event in split.values():
            for role in ("robust_sac", "matching_specialist"):
                if set(event.get(role) or {}) != {
                    str(mode) for mode in protocol.MODES
                }:
                    raise ValueError("v19 stationary modes are incomplete")
                for row in event[role].values():
                    values = row.get("returns") or []
                    if (
                        len(values) != protocol.EPISODES_PER_TASK
                        or not all(math.isfinite(float(value))
                                   for value in values)
                        or int(row.get("total_actions", -1))
                        != protocol.EPISODES_PER_TASK
                        * protocol.MAX_EPISODE_STEPS
                    ):
                        raise ValueError("invalid v19 stationary rollout")
    switching_rows = payload.get("switching_holdout") or {}
    if set(switching_rows) != {
        str(value) for value in protocol.SWITCHING_EVENT_SEEDS
    }:
        raise ValueError("v19 switching holdout is incomplete")
    event_hashes = set()
    expected_mode_count = (
        protocol.SWITCHING_EPISODES * protocol.MAX_EPISODE_STEPS
        // len(protocol.MODES)
    )
    for event_seed, event in switching_rows.items():
        if set(event) != {
            "robust_sac",
            "dynamic_specialist_oracle",
            "true_mode_safe_utility",
        }:
            raise ValueError("v19 switching arms are incomplete")
        arm_hashes = set()
        for row in event.values():
            values = row.get("returns") or []
            if (
                len(values) != protocol.SWITCHING_EPISODES
                or not all(math.isfinite(float(value)) for value in values)
                or int(row.get("total_actions", -1))
                != protocol.SWITCHING_EPISODES
                * protocol.MAX_EPISODE_STEPS
                or row.get("mode_counts") != {
                    str(mode): expected_mode_count for mode in protocol.MODES
                }
                or row.get("base_schedule")
                != list(protocol.SWITCHING_SCHEDULES[int(event_seed)])
            ):
                raise ValueError("invalid v19 switching rollout")
            arm_hashes.add(str(row.get("mode_trace_sha256") or ""))
        if len(arm_hashes) != 1 or "" in arm_hashes:
            raise ValueError("v19 switching arms used different streams")
        event_hashes.update(arm_hashes)
    if len(event_hashes) != len(protocol.SWITCHING_EVENT_SEEDS):
        raise ValueError("v19 switching event traces are not distinct")
    return manifest


def run(variant: str, seed: int) -> None:
    variant = protocol.require_variant(variant)
    seed = protocol.require_training_seed(seed)
    protocol.validate_registration()
    destination = protocol.audit_dir(variant, seed)
    if protocol.audit_manifest(variant, seed).is_file():
        try:
            validate_audit(variant, seed)
        except (KeyError, OSError, TypeError, ValueError):
            pass
        else:
            print(
                f"V19 AUDIT ALREADY COMPLETE: {variant} seed={seed}",
                flush=True,
            )
            return
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp.", dir=destination.parent))
    try:
        controllers = _load_controllers(variant, seed)
        calibration_events = base._stationary_events(
            controllers, seed, protocol.CALIBRATION_EVENT_SEEDS)
        calibration_matrix = base._aggregate_calibration(calibration_events)
        utility_map = base._utility_map(calibration_matrix)
        payload = {
            "schema": protocol.AUDIT_SCHEMA,
            "status": "complete",
            "identity": _identity(variant, seed),
            "registration": protocol.file_record(protocol.REGISTRATION_PATH),
            "source_bundle_manifest": protocol.file_record(
                protocol.source_manifest(seed)),
            "specialist_bundles": protocol.bundle_records(variant, seed),
            "calibration_events": calibration_events,
            "calibration_matrix": calibration_matrix,
            "utility_map": utility_map,
            "stationary_holdout": base._stationary_events(
                controllers,
                seed,
                protocol.STATIONARY_HOLDOUT_EVENT_SEEDS,
            ),
            "switching_holdout": switching._switching_events(
                controllers, utility_map),
        }
        protocol.write_json_atomic(temporary / "audit.json", payload)
        protocol.write_json_atomic(
            temporary / "audit_manifest.json",
            {
                "schema": protocol.AUDIT_SCHEMA,
                "status": "complete",
                "identity": _identity(variant, seed),
                "registration": protocol.file_record(
                    protocol.REGISTRATION_PATH),
                "source_bundle_manifest": protocol.file_record(
                    protocol.source_manifest(seed)),
                "specialist_bundles": protocol.bundle_records(variant, seed),
                "audit": protocol.file_record(temporary / "audit.json"),
            },
        )
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_audit(variant, seed)
    print(f"V19 AUDIT COMPLETE: {variant} seed={seed}", flush=True)


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
