"""Select one causal fallback configuration on development events only."""
from __future__ import annotations

import statistics
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_policy_distillation_transient_fallback_v1 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_policy_distillation_transient_fallback_screen_v1 as screen,
)


TERMINATION_PENALTY = 2_000.0


def _mean(values) -> float:
    values = [float(value) for value in values]
    if not values:
        raise ValueError("cannot average empty fallback values")
    return float(statistics.fmean(values))


def _selection_identity() -> dict[str, Any]:
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "benchmark_role": "transient_fallback_development_selection",
        "development_only": True,
        "selection_event_seeds": list(protocol.SCREEN_EVENT_SEEDS),
        "candidate_configs": [
            config.to_dict() for config in protocol.FALLBACK_CONFIGS
        ],
        "objective": "mean_switching_return_minus_2000x_termination_rate",
        "tie_break": "registered_config_order",
        "audit_event_seeds_sealed": list(protocol.AUDIT_EVENT_SEEDS),
    }


def validate_selection() -> dict[str, Any]:
    protocol.validate_frozen_candidate()
    manifest = protocol.read_json(protocol.selection_manifest())
    selection = protocol.read_json(protocol.selection_json())
    expected_screen_records = {
        str(seed): protocol.file_record(protocol.screen_manifest(seed))
        for seed in protocol.SCREEN_EVENT_SEEDS
    }
    if (manifest.get("schema") != protocol.SELECTION_SCHEMA
            or manifest.get("status") != "complete"
            or manifest.get("identity") != _selection_identity()
            or manifest.get("screen_manifests") != expected_screen_records
            or manifest.get("selection_file")
            != protocol.file_record(protocol.selection_json())):
        raise ValueError("invalid transient-fallback selection manifest")
    if (selection.get("schema") != protocol.SELECTION_SCHEMA
            or selection.get("status") != "complete"
            or selection.get("identity") != _selection_identity()
            or selection.get("selected_config")
            not in protocol.FALLBACK_CONFIG_BY_NAME):
        raise ValueError("invalid transient-fallback selection")
    selected = protocol.require_config(selection["selected_config"])
    if selection.get("selected_config_values") != selected.to_dict():
        raise ValueError("selected fallback values changed")
    return manifest


def run() -> dict[str, Any]:
    protocol.validate_frozen_candidate()
    rows_by_seed = {}
    for seed in protocol.SCREEN_EVENT_SEEDS:
        screen.validate_screen(seed)
        payload = protocol.read_json(protocol.screen_dir(seed) / "results.json")
        rows_by_seed[int(seed)] = {
            row["arm"]: row for row in payload["switching"]
        }

    arm_metrics = {}
    for arm in protocol.SCREEN_ARMS:
        returns = [
            float(value)
            for seed in protocol.SCREEN_EVENT_SEEDS
            for value in rows_by_seed[int(seed)][arm]["returns"]
        ]
        terminations = [
            float(rows_by_seed[int(seed)][arm]["terminated_rate"])
            for seed in protocol.SCREEN_EVENT_SEEDS
        ]
        mean_return = _mean(returns)
        terminated_rate = _mean(terminations)
        arm_metrics[arm] = {
            "return_mean": mean_return,
            "terminated_rate": terminated_rate,
            "selection_score": (
                mean_return - TERMINATION_PENALTY * terminated_rate),
            "event_return_means": {
                str(seed): _mean(rows_by_seed[int(seed)][arm]["returns"])
                for seed in protocol.SCREEN_EVENT_SEEDS
            },
        }
        if arm in protocol.FALLBACK_CONFIG_BY_NAME:
            arm_metrics[arm].update({
                "fallback_action_fraction": _mean(
                    rows_by_seed[int(seed)][arm]["fallback_action_fraction"]
                    for seed in protocol.SCREEN_EVENT_SEEDS
                ),
                "adaptive_wrong_action_fraction": _mean(
                    rows_by_seed[int(seed)][arm][
                        "adaptive_wrong_action_fraction"]
                    for seed in protocol.SCREEN_EVENT_SEEDS
                ),
            })

    order = {config.name: index for index, config in enumerate(
        protocol.FALLBACK_CONFIGS)}
    selected_name = max(
        protocol.FALLBACK_CONFIG_BY_NAME,
        key=lambda name: (arm_metrics[name]["selection_score"], -order[name]),
    )
    selected = protocol.require_config(selected_name)
    identity = _selection_identity()
    payload = {
        "schema": protocol.SELECTION_SCHEMA,
        "status": "complete",
        "identity": identity,
        "selected_config": selected_name,
        "selected_config_values": selected.to_dict(),
        "arm_metrics": arm_metrics,
        "development_delta_vs_learned": (
            arm_metrics[selected_name]["return_mean"]
            - arm_metrics[protocol.LEARNED_ARM]["return_mean"]),
        "development_delta_vs_robust719": (
            arm_metrics[selected_name]["return_mean"]
            - arm_metrics[protocol.ROBUST_ARM]["return_mean"]),
    }

    protocol.SELECTION_ROOT.mkdir(parents=True, exist_ok=True)
    protocol.write_json_atomic(protocol.selection_json(), payload)
    manifest = {
        "schema": protocol.SELECTION_SCHEMA,
        "status": "complete",
        "identity": identity,
        "screen_manifests": {
            str(seed): protocol.file_record(protocol.screen_manifest(seed))
            for seed in protocol.SCREEN_EVENT_SEEDS
        },
        "selection_file": protocol.file_record(protocol.selection_json()),
    }
    protocol.write_json_atomic(protocol.selection_manifest(), manifest)
    validate_selection()
    print(
        f"TRANSIENT FALLBACK SELECTED: {selected_name} "
        f"score={arm_metrics[selected_name]['selection_score']:.1f}",
        flush=True,
    )
    return payload


def main() -> None:
    if protocol.selection_manifest().is_file():
        try:
            validate_selection()
        except (KeyError, OSError, TypeError, ValueError):
            pass
        else:
            print("TRANSIENT FALLBACK SELECTION ALREADY COMPLETE")
            return
    run()


if __name__ == "__main__":
    main()
