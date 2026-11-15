"""Utility-aware decision layer for the frozen physical-mode estimator."""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from jax_experiments.analysis import (
    bapr_v3_control_equivalence as control,
)
from jax_experiments.analysis import (
    bapr_v3_learned_control_router as estimator,
)
from jax_experiments.analysis import (
    run_bapr_v3_independent_specialist_audit as specialist_audit,
)


ROOT = control.ROOT
FAMILY = control.FAMILY
ENV = control.ENV
FALLBACK_CONTROLLER = -1
ROBUST_CONTROLLER = 4
SPECIALIST_CONTROLLERS = (0, 1, 2, 3)
ALL_CONTROLLERS = (ROBUST_CONTROLLER, *SPECIALIST_CONTROLLERS)
CONTROLLER_NAMES = {
    ROBUST_CONTROLLER: "robust",
    **{mode: f"fixed_mode_{mode}" for mode in SPECIALIST_CONTROLLERS},
}

VALIDATION_EVENT_SEEDS = (6100, 6200)
HOLDOUT_EVENT_SEEDS = (7100, 7200, 7300, 7400, 7500)
BASELINE_DECISION_VARIANT = "c80h8"
DECISION_VARIANTS = {
    BASELINE_DECISION_VARIANT: {
        "confidence_threshold": 0.80,
        "min_history": 8,
    },
    "c75h4": {
        "confidence_threshold": 0.75,
        "min_history": 4,
    },
    "c70h8": {
        "confidence_threshold": 0.70,
        "min_history": 8,
    },
    "c70h4": {
        "confidence_threshold": 0.70,
        "min_history": 4,
    },
    "h005e50c80h4": {
        "hazard_rate": 0.005,
        "evidence_scale": 0.50,
        "confidence_threshold": 0.80,
        "min_history": 4,
    },
    "h010e50c80h4": {
        "hazard_rate": 0.010,
        "evidence_scale": 0.50,
        "confidence_threshold": 0.80,
        "min_history": 4,
    },
    "h010e100c80h4": {
        "hazard_rate": 0.010,
        "evidence_scale": 1.00,
        "confidence_threshold": 0.80,
        "min_history": 4,
    },
    "h020e50c80h4": {
        "hazard_rate": 0.020,
        "evidence_scale": 0.50,
        "confidence_threshold": 0.80,
        "min_history": 4,
    },
    "d090c80h8": {
        "posterior_decay": 0.90,
        "confidence_threshold": 0.80,
        "min_history": 8,
    },
    "d095c80h8": {
        "posterior_decay": 0.95,
        "confidence_threshold": 0.80,
        "min_history": 8,
    },
    "d0975c80h8": {
        "posterior_decay": 0.975,
        "confidence_threshold": 0.80,
        "min_history": 8,
    },
    "d099c80h8": {
        "posterior_decay": 0.99,
        "confidence_threshold": 0.80,
        "min_history": 8,
    },
    "cp025a25c80h8": {
        "change_reset_threshold": 0.25,
        "change_reset_alpha": 0.25,
        "change_reset_mix": 1.0,
        "confidence_threshold": 0.80,
        "min_history": 8,
    },
    "cp050a25c80h8": {
        "change_reset_threshold": 0.50,
        "change_reset_alpha": 0.25,
        "change_reset_mix": 1.0,
        "confidence_threshold": 0.80,
        "min_history": 8,
    },
    "cp100a25c80h8": {
        "change_reset_threshold": 1.00,
        "change_reset_alpha": 0.25,
        "change_reset_mix": 1.0,
        "confidence_threshold": 0.80,
        "min_history": 8,
    },
    "cs2d025c80h8": {
        "change_cusum_threshold": 2.0,
        "change_cusum_drift": 0.25,
        "change_reset_mix": 1.0,
        "confidence_threshold": 0.80,
        "min_history": 8,
    },
    "cs4d025c80h8": {
        "change_cusum_threshold": 4.0,
        "change_cusum_drift": 0.25,
        "change_reset_mix": 1.0,
        "confidence_threshold": 0.80,
        "min_history": 8,
    },
    "cs2d050c80h8": {
        "change_cusum_threshold": 2.0,
        "change_cusum_drift": 0.50,
        "change_reset_mix": 1.0,
        "confidence_threshold": 0.80,
        "min_history": 8,
    },
}

TABLE_ROOT = (
    ROOT / "jax_experiments"
    / "results_bapr_v3_structured_channel_utility_table_v2"
)
TABLE_PATH = TABLE_ROOT / "utility_table.json"
VALIDATION_ROOT = (
    ROOT / "jax_experiments"
    / "results_bapr_v3_structured_channel_utility_router_validation_v2"
)
HOLDOUT_ROOT = (
    ROOT / "jax_experiments"
    / "results_bapr_v3_structured_channel_utility_router_holdout_v2"
)
VALIDATION_ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_bapr_v3_structured_channel_utility_router_validation_analysis_v2"
)
HOLDOUT_ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_bapr_v3_structured_channel_utility_router_analysis_v2"
)

TABLE_SCHEMA = "bapr.v3-utility-table.v2"
AUDIT_SCHEMA = "bapr.v3-utility-router-audit-group.v2"
ANALYSIS_SCHEMA = "bapr.v3-utility-router-analysis.v2"


def configure() -> None:
    estimator.configure()


def controller_name(controller: int) -> str:
    try:
        return CONTROLLER_NAMES[int(controller)]
    except KeyError as exc:
        raise ValueError(f"unknown controller code {controller}") from exc


def decision_config(
    base: estimator.RouterConfig,
    variant: str,
) -> estimator.RouterConfig:
    try:
        overrides = DECISION_VARIANTS[str(variant)]
    except KeyError as exc:
        raise ValueError(f"unknown decision variant {variant!r}") from exc
    return replace(base, **overrides)


def audit_root(
    role: str,
    decision_variant: str = BASELINE_DECISION_VARIANT,
) -> Path:
    if decision_variant not in DECISION_VARIANTS:
        raise ValueError(f"unknown decision variant {decision_variant!r}")
    if role == "validation":
        base = VALIDATION_ROOT
    elif role == "holdout":
        base = HOLDOUT_ROOT
    else:
        raise ValueError(f"unknown utility-router role {role!r}")
    if decision_variant == BASELINE_DECISION_VARIANT:
        return base
    return base.with_name(base.name.replace("_v2", "_v3")) \
        / decision_variant


def analysis_root(
    role: str,
    decision_variant: str = BASELINE_DECISION_VARIANT,
) -> Path:
    if decision_variant not in DECISION_VARIANTS:
        raise ValueError(f"unknown decision variant {decision_variant!r}")
    if role == "validation":
        base = VALIDATION_ANALYSIS_ROOT
    elif role == "holdout":
        base = HOLDOUT_ANALYSIS_ROOT
    else:
        raise ValueError(f"unknown utility-router role {role!r}")
    if decision_variant == BASELINE_DECISION_VARIANT:
        return base
    return base.with_name(base.name.replace("_v2", "_v3")) \
        / decision_variant


def event_seeds(role: str) -> tuple[int, ...]:
    if role == "validation":
        return VALIDATION_EVENT_SEEDS
    if role == "holdout":
        return HOLDOUT_EVENT_SEEDS
    raise ValueError(f"unknown utility-router role {role!r}")


def audit_group_path(
    role: str,
    event_seed: int,
    decision_variant: str = BASELINE_DECISION_VARIANT,
) -> Path:
    return (
        audit_root(role, decision_variant) / FAMILY / "HalfCheetah"
        / f"event_seed_{int(event_seed)}" / "group.json"
    )


def _controller_values(group: dict[str, Any], physics_mode: int) -> dict[int, float]:
    return {
        controller: float(group["stationary"][controller_name(controller)][
            str(physics_mode)]["mean"])
        for controller in ALL_CONTROLLERS
    }


def _nondominated_controllers(means: dict[int, list[float]]) -> list[int]:
    kept = []
    for controller in ALL_CONTROLLERS:
        values = np.asarray(means[controller], dtype=np.float64)
        dominated = False
        for competitor in ALL_CONTROLLERS:
            if competitor == controller:
                continue
            other = np.asarray(means[competitor], dtype=np.float64)
            if np.all(other >= values) and np.any(other > values):
                dominated = True
                break
        if not dominated:
            kept.append(int(controller))
    return kept


def select_utility_table() -> dict[str, Any]:
    """Freeze controller utility from calibration streams only."""
    configure()
    groups = {}
    records = {}
    bundle_hashes = None
    for seed in control.CALIBRATION_EVENT_SEEDS:
        path = control.calibration_group_path(seed)
        payload = json.loads(path.read_text(encoding="utf-8"))
        specialist_audit.validate_group(payload)
        if (payload.get("family") != FAMILY
                or payload.get("env") != ENV
                or int(payload.get("event_seed", -1)) != seed):
            raise ValueError(f"calibration provenance mismatch: {path}")
        if bundle_hashes is None:
            bundle_hashes = payload.get("bundle_manifest_sha256")
        elif payload.get("bundle_manifest_sha256") != bundle_hashes:
            raise ValueError("calibration groups use different policy bundles")
        groups[seed] = payload
        records[str(seed)] = control.file_record(path)

    rows = {}
    mean_matrix = {
        controller: [] for controller in ALL_CONTROLLERS
    }
    oracle_map = []
    for physics_mode in range(4):
        per_seed = {
            str(seed): _controller_values(group, physics_mode)
            for seed, group in groups.items()
        }
        means = {
            controller: statistics.mean(
                per_seed[str(seed)][controller]
                for seed in control.CALIBRATION_EVENT_SEEDS)
            for controller in ALL_CONTROLLERS
        }
        for controller in ALL_CONTROLLERS:
            mean_matrix[controller].append(float(means[controller]))
        winner = max(
            ALL_CONTROLLERS,
            key=lambda controller: (means[controller], -controller),
        )
        oracle_map.append(int(winner))
        rows[str(physics_mode)] = {
            "mean_returns": {
                str(controller): float(means[controller])
                for controller in ALL_CONTROLLERS
            },
            "mean_advantage_over_robust": {
                str(controller): float(
                    means[controller] - means[ROBUST_CONTROLLER])
                for controller in SPECIALIST_CONTROLLERS
            },
            "per_seed_returns": {
                seed: {
                    str(controller): float(value)
                    for controller, value in values.items()
                }
                for seed, values in per_seed.items()
            },
            "oracle_controller": int(winner),
        }

    nondominated = _nondominated_controllers(mean_matrix)
    if ROBUST_CONTROLLER not in nondominated:
        raise ValueError("robust controller cannot be removed from utility table")
    payload = {
        "schema": TABLE_SCHEMA,
        "status": "complete",
        "family": FAMILY,
        "env": ENV,
        "training_seed": 0,
        "calibration_event_seeds": list(control.CALIBRATION_EVENT_SEEDS),
        "estimator_train_event_seeds": list(estimator.TRAIN_EVENT_SEEDS),
        "estimator_validation_event_seeds": list(
            estimator.VALIDATION_EVENT_SEEDS),
        "utility_validation_event_seeds": list(VALIDATION_EVENT_SEEDS),
        "utility_holdout_event_seeds": list(HOLDOUT_EVENT_SEEDS),
        "controller_codes": {
            str(code): controller_name(code) for code in ALL_CONTROLLERS
        },
        "fallback_controller": FALLBACK_CONTROLLER,
        "robust_controller": ROBUST_CONTROLLER,
        "nondominated_controllers": nondominated,
        "oracle_controller_map": oracle_map,
        "selection_rule": (
            "posterior expected stationary return over calibration-frozen "
            "nondominated controllers; low confidence or nonpositive "
            "specialist advantage executes robust"
        ),
        "calibration_groups": records,
        "bundle_manifest_sha256": bundle_hashes,
        "estimator_manifest_file": estimator.file_record(
            estimator.MANIFEST_PATH),
        "estimator_parameter_file": estimator.file_record(
            estimator.MODEL_PATH),
        "rows": rows,
    }
    estimator.write_json_atomic(TABLE_PATH, payload)
    return payload


def load_utility_table() -> dict[str, Any]:
    payload = json.loads(TABLE_PATH.read_text(encoding="utf-8"))
    if (payload.get("schema") != TABLE_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("family") != FAMILY
            or payload.get("env") != ENV
            or payload.get("robust_controller") != ROBUST_CONTROLLER
            or payload.get("fallback_controller") != FALLBACK_CONTROLLER
            or len(payload.get("oracle_controller_map", [])) != 4):
        raise ValueError(f"invalid utility table: {TABLE_PATH}")
    valid_codes = set(ALL_CONTROLLERS)
    if (not set(payload.get("nondominated_controllers", [])).issubset(
            valid_codes)
            or any(int(value) not in valid_codes
                   for value in payload["oracle_controller_map"])):
        raise ValueError("utility table contains invalid controller codes")
    for seed, expected in payload.get("calibration_groups", {}).items():
        if control.file_record(
                control.calibration_group_path(int(seed))) != expected:
            raise ValueError("calibration group changed after utility freeze")
    if (payload.get("estimator_manifest_file")
            != estimator.file_record(estimator.MANIFEST_PATH)
            or payload.get("estimator_parameter_file")
            != estimator.file_record(estimator.MODEL_PATH)):
        raise ValueError("frozen physical-mode estimator changed")
    return payload


def utility_matrix(table: dict[str, Any]) -> tuple[np.ndarray, tuple[int, ...]]:
    controllers = tuple(int(value) for value in
                        table["nondominated_controllers"])
    matrix = np.asarray([
        [float(table["rows"][str(mode)]["mean_returns"][str(controller)])
         for controller in controllers]
        for mode in range(4)
    ], dtype=np.float64)
    return matrix, controllers


def select_utility_controller(
    posterior: np.ndarray,
    count: int,
    table: dict[str, Any],
    config: estimator.RouterConfig,
) -> tuple[int, dict[str, float]]:
    """Select a controller by posterior expected utility."""
    probabilities = np.asarray(posterior, dtype=np.float64)
    if (probabilities.shape != (4,)
            or not np.all(np.isfinite(probabilities))
            or np.any(probabilities < 0.0)
            or float(np.sum(probabilities)) <= 0.0):
        raise ValueError("posterior must be a finite nonnegative 4-vector")
    probabilities /= np.sum(probabilities)
    ordered = np.sort(probabilities)[::-1]
    confidence = float(ordered[0])
    probability_margin = float(ordered[0] - ordered[1])
    matrix, controllers = utility_matrix(table)
    expected = probabilities @ matrix
    robust_index = controllers.index(ROBUST_CONTROLLER)
    robust_utility = float(expected[robust_index])
    best_index = int(np.argmax(expected))
    best_controller = int(controllers[best_index])
    best_utility = float(expected[best_index])
    best_advantage = best_utility - robust_utility
    eligible = (
        int(count) >= int(config.min_history)
        and confidence >= float(config.confidence_threshold)
        and probability_margin >= float(config.margin_threshold)
    )
    if not eligible:
        selected = FALLBACK_CONTROLLER
    elif (best_controller == ROBUST_CONTROLLER
          or best_advantage <= 0.0):
        selected = ROBUST_CONTROLLER
    else:
        selected = best_controller
    return selected, {
        "confidence": confidence,
        "margin": probability_margin,
        "fallback": float(not eligible),
        "deliberate_robust": float(
            eligible and selected == ROBUST_CONTROLLER),
        "best_expected_utility": best_utility,
        "robust_expected_utility": robust_utility,
        "best_expected_advantage": best_advantage,
    }


def routing_metrics(
    decisions: np.ndarray,
    true_modes: np.ndarray,
    controller_map: Iterable[int],
    stability: int = 8,
    burnin: int = 32,
) -> dict[str, Any]:
    """Score decisions while distinguishing fallback from robust intent."""
    decisions = np.asarray(decisions, dtype=np.int32)
    true_modes = np.asarray(true_modes, dtype=np.int32)
    mapping = np.asarray(tuple(controller_map), dtype=np.int32)
    if decisions.shape != true_modes.shape or decisions.ndim != 1:
        raise ValueError("routing decisions and true modes must be 1-D peers")
    expected = mapping[true_modes]
    action_controller = np.where(
        decisions == FALLBACK_CONTROLLER, ROBUST_CONTROLLER, decisions)
    index = np.arange(len(decisions)) >= int(burnin)
    committed = decisions != FALLBACK_CONTROLLER
    action_correct = action_controller == expected
    committed_rows = index & committed
    coverage = float(np.mean(committed[index])) if np.any(index) else 0.0
    conditional_accuracy = (
        float(np.mean((decisions == expected)[committed_rows]))
        if np.any(committed_rows) else 0.0)
    action_accuracy = (
        float(np.mean(action_correct[index])) if np.any(index) else 0.0)
    wrong_rate = (
        float(np.mean((~action_correct)[index])) if np.any(index) else 0.0)

    switch_points = np.flatnonzero(true_modes[1:] != true_modes[:-1]) + 1
    delays = []
    for switch in switch_points:
        next_switches = switch_points[switch_points > switch]
        end = int(next_switches[0]) if len(next_switches) else len(decisions)
        wanted = int(expected[switch])
        delay = end - switch
        for start in range(switch, max(switch, end - stability + 1)):
            if np.all(action_controller[start:start + stability] == wanted):
                delay = start - switch
                break
        delays.append(int(delay))
    return {
        "coverage": coverage,
        "fallback_rate": 1.0 - coverage,
        "conditional_accuracy": conditional_accuracy,
        "action_accuracy": action_accuracy,
        "wrong_route_rate": wrong_rate,
        "effective_accuracy": action_accuracy,
        "switch_count": int(len(switch_points)),
        "switch_delays": delays,
        "median_switch_delay": (
            float(np.median(delays)) if delays else 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select", action="store_true")
    args = parser.parse_args()
    if not args.select:
        parser.error("--select is required")
    payload = select_utility_table()
    print(
        "UTILITY TABLE COMPLETE: "
        f"map={payload['oracle_controller_map']} output={TABLE_PATH}",
        flush=True,
    )


if __name__ == "__main__":
    main()
