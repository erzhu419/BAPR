"""Calibrate and audit the frozen v9 MAP safe-utility policy bank."""
from __future__ import annotations

import argparse
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_specialist_safe_utility_confirmation_v9 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_confirmation_source_controller_v6 as source_runner,
)
from jax_experiments.analysis import (
    run_regime_polarity_corrected_audit_v2 as baseline_audit,
)
from jax_experiments.analysis import (
    run_regime_polarity_corrected_baseline_v2 as baseline_runner,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_capacity_diagnostic_v7 as capacity_audit,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_expected_action_confirmation_audit_v6 as source_audit,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_router_diagnostic_v2 as diagnostic_audit,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_safe_utility_audit_v8 as utility_audit,
)


def _bind() -> None:
    source_runner.protocol = protocol
    source_audit.protocol = protocol
    capacity_audit.protocol = protocol
    diagnostic_audit.protocol = protocol
    utility_audit.protocol = protocol
    utility_audit.capacity_audit = capacity_audit
    baseline_runner.protocol = protocol
    baseline_audit.protocol = protocol
    baseline_audit.trainer = baseline_runner


def _archived_records(seed: int, key: str, expected_names) -> dict[str, Any]:
    manifest = protocol.read_json(protocol.audit_manifest(seed))
    records = manifest.get(key) or {}
    if set(records) != set(expected_names):
        raise ValueError(f"invalid archived {key}")
    return records


def _expected_source_records(seed: int) -> dict[str, Any]:
    archived = _archived_records(seed, "source_bundles", protocol.ROLES)
    live = {
        role: protocol.file_record(protocol.bundle_manifest(role, seed))
        for role in protocol.ROLES
        if protocol.bundle_manifest(role, seed).is_file()
    }
    for role, record in live.items():
        if record != archived[role]:
            raise ValueError("live controller bundle changed after audit")
    return archived


def _expected_baseline_records(seed: int) -> dict[str, Any]:
    archived = _archived_records(
        seed, "baseline_bundles", protocol.BASELINE_METHODS)
    live = {
        method: protocol.file_record(protocol.bundle_manifest(method, seed))
        for method in protocol.BASELINE_METHODS
        if protocol.bundle_manifest(method, seed).is_file()
    }
    for method, record in live.items():
        if record != archived[method]:
            raise ValueError("live baseline bundle changed after audit")
    return archived


def _calibration_identity(seed: int) -> dict[str, Any]:
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "training_seed": protocol.require_training_seed(seed),
        "event_seeds": list(protocol.CALIBRATION_EVENT_SEEDS),
        "selection_rule": {
            "minimum_relative_gain": protocol.MIN_CALIBRATION_GAIN,
            "required_event_wins": len(protocol.CALIBRATION_EVENT_SEEDS),
            "require_zero_termination": True,
            "candidate_controllers": [
                "robust_sac", "matching_diagonal_specialist"],
        },
    }


def calibrate(seed: int, controllers: dict[str, dict]) -> dict[str, Any]:
    events = []
    for event_seed in protocol.CALIBRATION_EVENT_SEEDS:
        events.append({
            role: capacity_audit._stationary(controller, seed, event_seed)
            for role, controller in controllers.items()
        })
    matrix = utility_audit._aggregate_calibration(events)
    return {
        "schema": protocol.CALIBRATION_SCHEMA,
        "status": "complete",
        "identity": _calibration_identity(seed),
        "registration": protocol.file_record(protocol.REGISTRATION_PATH),
        "source_bundles": protocol.source_records(seed),
        "origin": {"kind": "v9_fresh_stationary_calibration"},
        "matrix": matrix,
        "utility_map": utility_audit._utility_map(matrix),
    }


def _event_identity(seed: int, event_seed: int) -> dict[str, Any]:
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "benchmark_role": "independent_safe_utility_switching_holdout",
        "env": protocol.ENV,
        "family": protocol.FAMILY,
        "training_seed": protocol.require_training_seed(seed),
        "event_seed": protocol.require_holdout_event_seed(event_seed),
        "arms": list(protocol.ARMS),
        "primary_arm": protocol.PRIMARY_ARM,
    }


def evaluate_event(
    seed: int,
    event_seed: int,
    stack: dict[str, Any],
    utility_stack: dict[str, Any],
    baseline_runtimes: dict[str, tuple[Any, Any]],
    calibration_record: dict[str, Any],
    utility_map: dict[str, Any],
) -> dict[str, Any]:
    switching = {
        "robust_sac": diagnostic_audit._switching_arm(
            stack, "robust_sac", event_seed),
        "dynamic_specialist_oracle": diagnostic_audit._switching_arm(
            stack, "dynamic_oracle", event_seed),
        "true_mode_safe_utility": diagnostic_audit._switching_arm(
            utility_stack, "dynamic_oracle", event_seed),
        protocol.PRIMARY_ARM: diagnostic_audit._switching_arm(
            utility_stack, "posterior_map_no_gate", event_seed),
    }
    baseline_stationary = {}
    for method, (config, runtime) in baseline_runtimes.items():
        baseline_stationary[method] = baseline_audit._stationary(
            config, runtime, event_seed)
        switching[method] = baseline_audit._switching(
            config, runtime, event_seed)
    return {
        "schema": protocol.EVENT_SCHEMA,
        "status": "complete",
        "identity": _event_identity(seed, event_seed),
        "registration": protocol.file_record(protocol.REGISTRATION_PATH),
        "source_bundles": protocol.source_records(seed),
        "baseline_bundles": protocol.baseline_records(seed),
        "estimator": protocol.estimator_records(),
        "calibration": calibration_record,
        "utility_map": utility_map,
        "switching": switching,
        "baseline_stationary": baseline_stationary,
    }


def validate_calibration(payload: dict, seed: int) -> None:
    if (
        payload.get("schema") != protocol.CALIBRATION_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("identity") != _calibration_identity(seed)
        or payload.get("registration")
        != protocol.file_record(protocol.REGISTRATION_PATH)
        or payload.get("source_bundles") != _expected_source_records(seed)
        or payload.get("origin")
        != {"kind": "v9_fresh_stationary_calibration"}
        or payload.get("utility_map")
        != utility_audit._utility_map(payload.get("matrix") or {})
    ):
        raise ValueError("invalid v9 safe-utility calibration")


def validate_event(payload: dict, seed: int, event_seed: int) -> None:
    calibration = protocol.read_json(protocol.calibration_result(seed))
    if (
        payload.get("schema") != protocol.EVENT_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("identity") != _event_identity(seed, event_seed)
        or payload.get("registration")
        != protocol.file_record(protocol.REGISTRATION_PATH)
        or payload.get("source_bundles") != _expected_source_records(seed)
        or payload.get("baseline_bundles")
        != _expected_baseline_records(seed)
        or payload.get("estimator") != protocol.estimator_records()
        or payload.get("calibration")
        != protocol.file_record(protocol.calibration_result(seed))
        or payload.get("utility_map") != calibration["utility_map"]
        or set(payload.get("switching") or {}) != set(protocol.ARMS)
        or set(payload.get("baseline_stationary") or {})
        != set(protocol.BASELINE_METHODS)
    ):
        raise ValueError("invalid v9 safe-utility holdout event")

    traces = set()
    for arm in protocol.ARMS:
        row = payload["switching"][arm]
        returns = row.get("returns") or []
        if (
            len(returns) != protocol.SWITCHING_EPISODES
            or not all(math.isfinite(float(value)) for value in returns)
            or int(row.get("total_actions", -1))
            != protocol.SWITCHING_EPISODES * protocol.MAX_EPISODE_STEPS
        ):
            raise ValueError("invalid v9 switching rollout")
        traces.add(str(row.get("mode_trace_sha256") or ""))
    if len(traces) != 1 or "" in traces:
        raise ValueError("v9 methods used different switching streams")
    if payload["switching"]["true_mode_safe_utility"][
        "adaptive_mode_accuracy"] != 1.0:
        raise ValueError("v9 safe oracle used a wrong mode")

    for method in protocol.BASELINE_METHODS:
        rows = payload["baseline_stationary"][method]
        if (
            [row.get("mode") for row in rows] != list(protocol.MODES)
            or any(
                len(row.get("returns") or [])
                != protocol.AUDIT_EPISODES_PER_TASK
                for row in rows
            )
        ):
            raise ValueError("invalid v9 baseline stationary rollout")


def validate_audit(seed: int) -> dict[str, Any]:
    seed = protocol.require_training_seed(seed)
    manifest = protocol.read_json(protocol.audit_manifest(seed))
    event_files = {
        str(event_seed): protocol.file_record(
            protocol.event_result(seed, event_seed))
        for event_seed in protocol.HOLDOUT_EVENT_SEEDS
    }
    if (
        manifest.get("schema") != protocol.AUDIT_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("identity")
        != {
            "protocol_version": protocol.PROTOCOL_VERSION,
            "training_seed": seed,
            "holdout_event_seeds": list(protocol.HOLDOUT_EVENT_SEEDS),
        }
        or manifest.get("registration")
        != protocol.file_record(protocol.REGISTRATION_PATH)
        or manifest.get("source_bundles") != _expected_source_records(seed)
        or manifest.get("baseline_bundles")
        != _expected_baseline_records(seed)
        or manifest.get("estimator") != protocol.estimator_records()
        or manifest.get("calibration")
        != protocol.file_record(protocol.calibration_result(seed))
        or manifest.get("event_files") != event_files
    ):
        raise ValueError("invalid v9 audit manifest")
    validate_calibration(
        protocol.read_json(protocol.calibration_result(seed)), seed)
    for event_seed in protocol.HOLDOUT_EVENT_SEEDS:
        validate_event(
            protocol.read_json(protocol.event_result(seed, event_seed)),
            seed,
            event_seed,
        )
    return manifest


def run(seed: int) -> None:
    seed = protocol.require_training_seed(seed)
    protocol.validate_registration()
    _bind()
    destination = protocol.audit_dir(seed)
    if protocol.audit_manifest(seed).is_file():
        try:
            validate_audit(seed)
        except (KeyError, OSError, TypeError, ValueError):
            pass
        else:
            print(f"V9 AUDIT ALREADY COMPLETE: {destination}")
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
        controllers = {
            role: source_audit._load_controller(role, seed)
            for role in protocol.ROLES
        }
        calibration = calibrate(seed, controllers)
        protocol.write_json_atomic(
            temporary / "calibration.json", calibration)
        calibration_record = protocol.file_record(
            temporary / "calibration.json")
        stack = utility_audit._stack_from_controllers(controllers)
        utility_stack = utility_audit._utility_stack(
            stack, calibration["utility_map"])
        baseline_runtimes = {
            method: baseline_audit._load_runtime(method, seed)
            for method in protocol.BASELINE_METHODS
        }
        event_files = {}
        for event_seed in protocol.HOLDOUT_EVENT_SEEDS:
            result = temporary / f"event_seed_{event_seed}" / "results.json"
            protocol.write_json_atomic(
                result,
                evaluate_event(
                    seed,
                    event_seed,
                    stack,
                    utility_stack,
                    baseline_runtimes,
                    calibration_record,
                    calibration["utility_map"],
                ),
            )
            event_files[str(event_seed)] = protocol.file_record(result)
            print(
                f"v9 audit seed={seed} event={event_seed} complete",
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
                    "holdout_event_seeds": list(protocol.HOLDOUT_EVENT_SEEDS),
                },
                "registration": protocol.file_record(
                    protocol.REGISTRATION_PATH),
                "source_bundles": protocol.source_records(seed),
                "baseline_bundles": protocol.baseline_records(seed),
                "estimator": protocol.estimator_records(),
                "calibration": protocol.file_record(
                    temporary / "calibration.json"),
                "event_files": event_files,
            },
        )
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_audit(seed)
    print(f"V9 SAFE UTILITY AUDIT COMPLETE: seed={seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.seed)


if __name__ == "__main__":
    main()
