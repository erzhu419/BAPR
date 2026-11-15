"""Calibrate robust-inclusive mode utility and audit it on holdout switches."""
from __future__ import annotations

import argparse
import math
import shutil
import statistics
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_specialist_capacity_diagnostic_v7 as capacity,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_confirmation_v6 as source,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_model_v5 as model_lib,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_safe_utility_v8 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_capacity_diagnostic_v7 as capacity_audit,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_router_diagnostic_v2 as diagnostic_audit,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_sticky_router_audit_v3 as sticky_audit,
)
from jax_experiments.analysis.run_regime_polarity_specialist_expected_action_confirmation_audit_v6 import (
    _action_fn,
    _load_controller,
)


def _mean(values) -> float:
    return float(statistics.fmean(values))


def _stack_from_controllers(controllers: dict[str, dict]) -> dict[str, Any]:
    actions = {}
    for role, controller in controllers.items():
        action_fn = _action_fn(controller)

        def action(observation, *, _fn=action_fn, _source=controller):
            return np.asarray(
                _fn(_source["policy_params"], observation),
                dtype=np.float32,
            )

        actions[role] = action
    reference = controllers["robust_sac"]
    return {
        "config": reference["config"],
        "actions": actions,
        "estimator_factory": lambda: model_lib.make_estimator(
            int(reference["agent"].obs_dim),
            int(reference["agent"].act_dim),
        ),
    }


def _load_controllers(seed: int) -> dict[str, dict]:
    return {
        role: _load_controller(role, seed)
        for role in protocol.ROLES
    }


def _aggregate_calibration(events: list[dict]) -> dict[str, Any]:
    matrix = {}
    for role in protocol.ROLES:
        matrix[role] = {}
        for mode in protocol.MODES:
            rows = [event[role][str(mode)] for event in events]
            matrix[role][str(mode)] = {
                "mean": _mean(row["return_mean"] for row in rows),
                "terminated_rate": _mean(
                    row["terminated_rate"] for row in rows),
                "event_returns": {
                    str(event_seed): float(row["return_mean"])
                    for event_seed, row in zip(
                        protocol.CALIBRATION_EVENT_SEEDS, rows)
                },
            }
    return matrix


def _calibration_events_from_capacity(seed: int) -> tuple[list[dict], dict]:
    capacity_audit.validate_audit(seed)
    events = [
        capacity.read_json(capacity.event_result(seed, event_seed))["stationary"]
        for event_seed in protocol.CALIBRATION_EVENT_SEEDS
    ]
    return events, {
        "kind": "frozen_v7_capacity_audit",
        "audit_manifest": capacity.file_record(capacity.audit_manifest(seed)),
    }


def _evaluate_calibration(
    seed: int,
    controllers: dict[str, dict],
) -> tuple[list[dict], dict]:
    events = []
    for event_seed in protocol.CALIBRATION_EVENT_SEEDS:
        events.append({
            role: capacity_audit._stationary(
                controller, seed, event_seed)
            for role, controller in controllers.items()
        })
    return events, {"kind": "v8_in_task_stationary_calibration"}


def _utility_map(matrix: dict[str, Any]) -> dict[str, Any]:
    mapping = {}
    for mode in protocol.MODES:
        robust = matrix["robust_sac"][str(mode)]
        specialist = matrix[f"specialist_{mode}"][str(mode)]
        relative_gain = (
            (specialist["mean"] - robust["mean"]) / abs(robust["mean"])
            if robust["mean"] != 0.0 else float("-inf")
        )
        event_wins = sum(
            specialist["event_returns"][str(event_seed)]
            > robust["event_returns"][str(event_seed)]
            for event_seed in protocol.CALIBRATION_EVENT_SEEDS
        )
        use_specialist = bool(
            relative_gain >= protocol.MIN_CALIBRATION_GAIN
            and event_wins == len(protocol.CALIBRATION_EVENT_SEEDS)
            and specialist["terminated_rate"] == 0.0
        )
        mapping[str(mode)] = {
            "controller": (
                f"specialist_{mode}" if use_specialist else "robust_sac"
            ),
            "relative_gain": relative_gain,
            "event_wins": event_wins,
            "specialist_terminated_rate": specialist["terminated_rate"],
        }
    return mapping


def calibrate(seed: int, controllers: dict[str, dict] | None) -> dict[str, Any]:
    if seed in protocol.CAPACITY_AUDIT_SEEDS:
        events, origin = _calibration_events_from_capacity(seed)
    else:
        if controllers is None:
            raise ValueError("fresh stationary calibration needs controllers")
        events, origin = _evaluate_calibration(seed, controllers)
    matrix = _aggregate_calibration(events)
    return {
        "schema": protocol.CALIBRATION_SCHEMA,
        "status": "complete",
        "identity": {
            "protocol_version": protocol.PROTOCOL_VERSION,
            "training_seed": seed,
            "event_seeds": list(protocol.CALIBRATION_EVENT_SEEDS),
            "selection_rule": {
                "minimum_relative_gain": protocol.MIN_CALIBRATION_GAIN,
                "required_event_wins": len(protocol.CALIBRATION_EVENT_SEEDS),
                "require_zero_termination": True,
                "candidate_controllers": [
                    "robust_sac", "matching_diagonal_specialist"],
            },
        },
        "source_bundles": protocol.source_records(seed),
        "origin": origin,
        "matrix": matrix,
        "utility_map": _utility_map(matrix),
    }


def _utility_stack(stack: dict[str, Any], mapping: dict[str, Any]) -> dict:
    actions = dict(stack["actions"])
    for mode in protocol.MODES:
        selected = mapping[str(mode)]["controller"]
        actions[f"specialist_{mode}"] = actions[selected]
    return {**stack, "actions": actions}


def _identity(seed: int, event_seed: int) -> dict[str, Any]:
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "benchmark_role": "robust_inclusive_specialist_utility_holdout",
        "env": protocol.ENV,
        "family": protocol.FAMILY,
        "training_seed": protocol.require_training_seed(seed),
        "event_seed": protocol.require_holdout_event_seed(event_seed),
        "arms": list(protocol.ARMS),
        "primary_arm": protocol.PRIMARY_ARM,
    }


def evaluate_holdout(
    seed: int,
    event_seed: int,
    stack: dict[str, Any],
    calibration: dict[str, Any],
    calibration_record: dict[str, Any],
) -> dict[str, Any]:
    utility_stack = _utility_stack(stack, calibration["utility_map"])
    switching = {
        "robust_sac": diagnostic_audit._switching_arm(
            stack, "robust_sac", event_seed),
        "dynamic_specialist_oracle": diagnostic_audit._switching_arm(
            stack, "dynamic_oracle", event_seed),
        "true_mode_safe_utility": diagnostic_audit._switching_arm(
            utility_stack, "dynamic_oracle", event_seed),
        "posterior_map_safe_utility": diagnostic_audit._switching_arm(
            utility_stack, "posterior_map_no_gate", event_seed),
    }
    sticky_arm = "posterior_sticky_confirm3_safe_utility"
    if sticky_arm in protocol.ARMS:
        switching[sticky_arm] = sticky_audit._sticky_arm(
            utility_stack, "posterior_sticky_confirm3", event_seed)
    return {
        "schema": protocol.EVENT_SCHEMA,
        "status": "complete",
        "identity": _identity(seed, event_seed),
        "source_bundles": protocol.source_records(seed),
        "estimator": protocol.estimator_records(),
        "calibration": calibration_record,
        "utility_map": calibration["utility_map"],
        "switching": switching,
    }


def validate_calibration(payload: dict, seed: int) -> None:
    if (
        payload.get("schema") != protocol.CALIBRATION_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("identity")
        != {
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
        or payload.get("source_bundles") != protocol.source_records(seed)
        or payload.get("utility_map") != _utility_map(payload.get("matrix") or {})
    ):
        raise ValueError("invalid robust-inclusive utility calibration")
    if seed in protocol.CAPACITY_AUDIT_SEEDS:
        expected_origin = {
            "kind": "frozen_v7_capacity_audit",
            "audit_manifest": capacity.file_record(
                capacity.audit_manifest(seed)),
        }
        if payload.get("origin") != expected_origin:
            raise ValueError("v8 calibration did not use frozen v7 evidence")
    elif payload.get("origin") != {
        "kind": "v8_in_task_stationary_calibration"
    }:
        raise ValueError("v8 calibration has an invalid origin")


def validate_event(payload: dict, seed: int, event_seed: int) -> None:
    calibration = protocol.read_json(protocol.calibration_result(seed))
    if (
        payload.get("schema") != protocol.EVENT_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("identity") != _identity(seed, event_seed)
        or payload.get("source_bundles") != protocol.source_records(seed)
        or payload.get("estimator") != protocol.estimator_records()
        or payload.get("calibration")
        != protocol.file_record(protocol.calibration_result(seed))
        or payload.get("utility_map") != calibration["utility_map"]
        or set(payload.get("switching") or {}) != set(protocol.ARMS)
    ):
        raise ValueError("invalid safe-utility holdout event")
    traces = set()
    for arm in protocol.ARMS:
        row = payload["switching"][arm]
        values = row.get("returns") or []
        if (
            len(values) != protocol.SWITCHING_EPISODES
            or not all(math.isfinite(float(value)) for value in values)
            or int(row.get("total_actions", -1))
            != protocol.SWITCHING_EPISODES * protocol.MAX_EPISODE_STEPS
        ):
            raise ValueError("invalid safe-utility switching rollout")
        traces.add(str(row.get("mode_trace_sha256") or ""))
    if len(traces) != 1 or "" in traces:
        raise ValueError("safe-utility arms used different switching streams")
    if payload["switching"]["true_mode_safe_utility"][
        "adaptive_mode_accuracy"] != 1.0:
        raise ValueError("safe-utility oracle used a wrong mode")


def validate_audit(seed: int) -> dict[str, Any]:
    seed = protocol.require_training_seed(seed)
    manifest = protocol.read_json(protocol.audit_manifest(seed))
    expected_events = {
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
        or manifest.get("source_bundles") != protocol.source_records(seed)
        or manifest.get("estimator") != protocol.estimator_records()
        or manifest.get("calibration")
        != protocol.file_record(protocol.calibration_result(seed))
        or manifest.get("event_files") != expected_events
    ):
        raise ValueError("invalid safe-utility audit manifest")
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
    destination = protocol.audit_dir(seed)
    if protocol.audit_manifest(seed).is_file():
        try:
            validate_audit(seed)
        except (KeyError, OSError, TypeError, ValueError):
            pass
        else:
            print(f"SAFE UTILITY AUDIT ALREADY COMPLETE: {destination}")
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
        controllers = _load_controllers(seed)
        calibration = calibrate(seed, controllers)
        protocol.write_json_atomic(
            temporary / "calibration.json", calibration)
        calibration_record = protocol.file_record(
            temporary / "calibration.json")
        stack = _stack_from_controllers(controllers)
        event_files = {}
        for event_seed in protocol.HOLDOUT_EVENT_SEEDS:
            result = temporary / f"event_seed_{event_seed}" / "results.json"
            protocol.write_json_atomic(
                result,
                evaluate_holdout(
                    seed,
                    event_seed,
                    stack,
                    calibration,
                    calibration_record,
                ),
            )
            event_files[str(event_seed)] = protocol.file_record(result)
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
                "source_bundles": protocol.source_records(seed),
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
    print(f"SAFE UTILITY AUDIT COMPLETE: seed={seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.seed)


if __name__ == "__main__":
    main()
