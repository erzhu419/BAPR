"""Confirm the frozen v14 confirm-3 fallback router on holdout events."""
from __future__ import annotations

import argparse
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_conflict_fallback_confirmation_v15 as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_conflict_fallback_router_v14 as parent,
)
from jax_experiments.analysis import (
    regime_polarity_frozen_estimator_transfer_v13 as transfer_parent,
)
from jax_experiments.analysis import (
    run_regime_polarity_conflict_fallback_router_audit_v14 as parent_audit,
)


def _identity(seed: int, event_seed: int) -> dict[str, Any]:
    event_seed = protocol.require_event_seed(event_seed)
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "benchmark_role": "frozen_conflict_fallback_holdout_confirmation",
        "env": protocol.ENV,
        "family": protocol.FAMILY,
        "training_seed": protocol.require_training_seed(seed),
        "event_seed": event_seed,
        "base_schedule": list(protocol.SWITCHING_SCHEDULES[event_seed]),
        "arms": list(protocol.ARMS),
        "dwell_steps": protocol.DWELL_STEPS,
        "max_episode_steps": protocol.MAX_EPISODE_STEPS,
        "switching_episodes": protocol.SWITCHING_EPISODES,
    }


def _load_stack(seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    seed = protocol.require_training_seed(seed)
    parent.validate_registration()
    parent_audit.validate_audit(seed)
    transfer_parent.validate_registration()
    return parent_audit._load_stack(seed)


def _switching_arm(
    stack: dict[str, Any],
    mapping: dict[str, Any],
    arm: str,
    event_seed: int,
) -> dict[str, Any]:
    """Run the frozen v14 implementation against the v15 protocol constants."""
    previous_protocol = parent_audit.protocol
    parent_audit.protocol = protocol
    try:
        return parent_audit._switching_arm(
            stack, mapping, arm, event_seed)
    finally:
        parent_audit.protocol = previous_protocol


def evaluate(seed: int, event_seed: int) -> dict[str, Any]:
    seed = protocol.require_training_seed(seed)
    event_seed = protocol.require_event_seed(event_seed)
    stack, utility_map = _load_stack(seed)
    return {
        "schema": protocol.EVENT_SCHEMA,
        "status": "complete",
        "identity": _identity(seed, event_seed),
        "registration": protocol.file_record(protocol.REGISTRATION_PATH),
        "parent_registration": protocol.file_record(parent.REGISTRATION_PATH),
        "parent_analysis": protocol.file_record(parent.analysis_json()),
        "parent_audit": protocol.file_record(parent.audit_manifest(seed)),
        "transfer_parent_audit": protocol.file_record(
            transfer_parent.audit_manifest(seed)),
        "policy_bank": protocol.policy_bank_records(seed),
        "estimator": protocol.estimator_records(),
        "utility_map": utility_map,
        "switching": {
            arm: _switching_arm(stack, utility_map, arm, event_seed)
            for arm in protocol.ARMS
        },
    }


def validate_event(payload: dict, seed: int, event_seed: int) -> None:
    parent_payload = protocol.read_json(
        parent.event_result(seed, parent.EVENT_SEEDS[0]))
    if (
        payload.get("schema") != protocol.EVENT_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("identity") != _identity(seed, event_seed)
        or payload.get("registration")
        != protocol.file_record(protocol.REGISTRATION_PATH)
        or payload.get("parent_registration")
        != protocol.file_record(parent.REGISTRATION_PATH)
        or payload.get("parent_analysis")
        != protocol.file_record(parent.analysis_json())
        or payload.get("parent_audit")
        != protocol.file_record(parent.audit_manifest(seed))
        or payload.get("transfer_parent_audit")
        != protocol.file_record(transfer_parent.audit_manifest(seed))
        or payload.get("policy_bank") != protocol.policy_bank_records(seed)
        or payload.get("estimator") != protocol.estimator_records()
        or payload.get("utility_map") != parent_payload["utility_map"]
        or set(payload.get("switching") or {}) != set(protocol.ARMS)
    ):
        raise ValueError("invalid v15 conflict-fallback confirmation event")

    expected_mode_count = (
        protocol.SWITCHING_EPISODES * protocol.MAX_EPISODE_STEPS
        // len(protocol.MODES)
    )
    traces = set()
    for arm in protocol.ARMS:
        row = payload["switching"][arm]
        if (
            len(row.get("returns") or []) != protocol.SWITCHING_EPISODES
            or not all(math.isfinite(float(value)) for value in row["returns"])
            or int(row.get("total_actions", -1))
            != protocol.SWITCHING_EPISODES * protocol.MAX_EPISODE_STEPS
            or row.get("mode_counts") != {
                str(mode): expected_mode_count for mode in protocol.MODES
            }
            or row.get("base_schedule")
            != list(protocol.SWITCHING_SCHEDULES[int(event_seed)])
            or int(row.get("confirmation_steps", -1))
            != protocol.confirm_steps_for_arm(arm)
        ):
            raise ValueError("invalid v15 conflict-fallback confirmation rollout")
        traces.add(str(row.get("mode_trace_sha256") or ""))
    if len(traces) != 1 or "" in traces:
        raise ValueError("v15 arms used different switching streams")
    for arm in (
        "true_mode_safe_utility", "delayed_oracle_4_safe_utility",
    ):
        if payload["switching"][arm]["routing_mode_accuracy"] != 1.0:
            raise ValueError("v15 privileged arm routed a wrong mode")
    for arm in ("posterior_map_safe_utility", protocol.SELECTED_ARM):
        posterior = payload["switching"][arm].get("posterior_metrics") or {}
        if not (
            math.isfinite(float(posterior.get("mode_accuracy", math.nan)))
            and math.isfinite(float(posterior.get("brier_score", math.nan)))
        ):
            raise ValueError("v15 posterior metrics are missing")


def validate_audit(seed: int) -> dict[str, Any]:
    seed = protocol.require_training_seed(seed)
    manifest = protocol.read_json(protocol.audit_manifest(seed))
    expected = {
        str(event_seed): protocol.file_record(
            protocol.event_result(seed, event_seed))
        for event_seed in protocol.EVENT_SEEDS
    }
    if (
        manifest.get("schema") != protocol.AUDIT_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("identity")
        != {
            "protocol_version": protocol.PROTOCOL_VERSION,
            "training_seed": seed,
            "event_seeds": list(protocol.EVENT_SEEDS),
            "switching_schedules": {
                str(key): list(value)
                for key, value in protocol.SWITCHING_SCHEDULES.items()
            },
        }
        or manifest.get("registration")
        != protocol.file_record(protocol.REGISTRATION_PATH)
        or manifest.get("parent_audit")
        != protocol.file_record(parent.audit_manifest(seed))
        or manifest.get("transfer_parent_audit")
        != protocol.file_record(transfer_parent.audit_manifest(seed))
        or manifest.get("policy_bank") != protocol.policy_bank_records(seed)
        or manifest.get("estimator") != protocol.estimator_records()
        or manifest.get("event_files") != expected
    ):
        raise ValueError("invalid v15 conflict-fallback audit manifest")
    traces = set()
    for event_seed in protocol.EVENT_SEEDS:
        payload = protocol.read_json(protocol.event_result(seed, event_seed))
        validate_event(payload, seed, event_seed)
        traces.add(payload["switching"]["robust_sac"]["mode_trace_sha256"])
    if len(traces) != len(protocol.EVENT_SEEDS):
        raise ValueError("v15 event streams are not distinct")
    return manifest


def run(seed: int) -> None:
    seed = protocol.require_training_seed(seed)
    protocol.validate_registration()
    destination = protocol.audit_dir(seed)
    if protocol.audit_manifest(seed).is_file():
        try:
            validate_audit(seed)
        except (KeyError, OSError, TypeError, ValueError):
            pass
        else:
            print(f"V15 CONFLICT FALLBACK AUDIT ALREADY COMPLETE: seed={seed}")
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
        event_files = {}
        for event_seed in protocol.EVENT_SEEDS:
            result = temporary / f"event_seed_{event_seed}" / "results.json"
            protocol.write_json_atomic(result, evaluate(seed, event_seed))
            event_files[str(event_seed)] = protocol.file_record(result)
            print(
                f"v15 conflict fallback seed={seed} event={event_seed} complete",
                flush=True,
            )
        protocol.write_json_atomic(
            temporary / "audit_manifest.json",
            {
                "schema": protocol.AUDIT_SCHEMA,
                "status": "complete",
                "identity": {
                    "protocol_version": protocol.PROTOCOL_VERSION,
                    "training_seed": seed,
                    "event_seeds": list(protocol.EVENT_SEEDS),
                    "switching_schedules": {
                        str(key): list(value)
                        for key, value in protocol.SWITCHING_SCHEDULES.items()
                    },
                },
                "registration": protocol.file_record(
                    protocol.REGISTRATION_PATH),
                "parent_audit": protocol.file_record(
                    parent.audit_manifest(seed)),
                "transfer_parent_audit": protocol.file_record(
                    transfer_parent.audit_manifest(seed)),
                "policy_bank": protocol.policy_bank_records(seed),
                "estimator": protocol.estimator_records(),
                "event_files": event_files,
            },
        )
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_audit(seed)
    print(f"V15 CONFLICT FALLBACK AUDIT COMPLETE: seed={seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.seed)


if __name__ == "__main__":
    main()
