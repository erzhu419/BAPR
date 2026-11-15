"""Diagnose the causal sequence router on its independent validation streams."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.analysis import bapr_v3_sequence_router as protocol
from jax_experiments.analysis import bapr_v3_utility_aware_router as utility
from jax_experiments.analysis import (
    run_bapr_v3_independent_specialist_audit as specialist_audit,
)
from jax_experiments.analysis import train_bapr_v3_sequence_router as trainer


SCHEMA = "bapr.v3-sequence-router-diagnostic.v1"
OUTPUT_ROOT = protocol.MODEL_ROOT.with_name(
    "results_bapr_v3_structured_channel_sequence_router_diagnostic_v1")
JSON_PATH = OUTPUT_ROOT / "diagnostic.json"
MARKDOWN_PATH = OUTPUT_ROOT / "diagnostic.md"
TEMPERATURES = (0.50, 0.75, 1.00)
CONFIDENCE_THRESHOLDS = (0.60, 0.70, 0.80)
MARGIN_THRESHOLDS = (0.00, 0.02)
MIN_HISTORIES = (0, 4, 8)
AGE_BINS = ((0, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 1001))


def load_sequence_model(manifest):
    model = trainer.make_sequence_model(seed=20260718)
    template = nnx.state(model, nnx.Param)
    params = protocol.emission.load_parameter_state(
        protocol.MODEL_PATH, template, manifest["parameter_leaves"])
    nnx.update(model, params)
    return model


def posterior_trace(model, evidence, temperature: float = 1.0):
    _, logits = model.sequence(jnp.asarray(evidence))
    return np.asarray(jax.nn.softmax(logits / float(temperature), axis=-1))


def before_action(posteriors):
    return np.concatenate([
        np.full((1, 4), 0.25, dtype=np.float64),
        np.asarray(posteriors[:-1], dtype=np.float64),
    ])


def decision_trace(posteriors, table, config):
    probabilities = before_action(posteriors)
    matrix, controllers = utility.utility_matrix(table)
    controllers_np = np.asarray(controllers, dtype=np.int32)
    expected = probabilities @ matrix
    best_indices = np.argmax(expected, axis=1)
    best_controllers = controllers_np[best_indices]
    robust_index = controllers.index(utility.ROBUST_CONTROLLER)
    advantages = (
        expected[np.arange(len(expected)), best_indices]
        - expected[:, robust_index]
    )
    ordered = np.sort(probabilities, axis=1)
    eligible = (
        (np.arange(len(probabilities)) >= int(config.min_history))
        & (ordered[:, -1] >= float(config.confidence_threshold))
        & ((ordered[:, -1] - ordered[:, -2])
           >= float(config.margin_threshold))
    )
    selected = np.where(
        eligible,
        np.where(
            (best_controllers == utility.ROBUST_CONTROLLER)
            | (advantages <= 0.0),
            utility.ROBUST_CONTROLLER,
            best_controllers,
        ),
        utility.FALLBACK_CONTROLLER,
    )
    return selected.astype(np.int32)


def _aggregate_metrics(rows):
    names = (
        "coverage", "conditional_accuracy", "action_accuracy",
        "wrong_route_rate", "median_switch_delay",
    )
    return {
        name: float(np.mean([float(row[name]) for row in rows]))
        for name in names
    }


def score_config(prepared, posteriors, table, config):
    by_kind = {"stationary": [], "full_cycle": []}
    oracle_map = table["oracle_controller_map"]
    for sequence, posterior in zip(prepared, posteriors):
        decisions = decision_trace(posterior, table, config)
        kind = str(sequence["identity"]["kind"])
        by_kind[kind].append(utility.routing_metrics(
            decisions, sequence["labels"], oracle_map, burnin=32))
    stationary = _aggregate_metrics(by_kind["stationary"])
    full_cycle = _aggregate_metrics(by_kind["full_cycle"])
    gate = {
        "stationary_action_accuracy_at_least_90pct": (
            stationary["action_accuracy"] >= 0.90),
        "stationary_wrong_route_at_most_10pct": (
            stationary["wrong_route_rate"] <= 0.10),
        "full_cycle_action_accuracy_at_least_90pct": (
            full_cycle["action_accuracy"] >= 0.90),
        "full_cycle_wrong_route_at_most_10pct": (
            full_cycle["wrong_route_rate"] <= 0.10),
        "full_cycle_median_delay_below_75": (
            full_cycle["median_switch_delay"] < 75.0),
    }
    gate["passed"] = all(gate.values())
    return {
        "stationary": stationary,
        "full_cycle": full_cycle,
        "gate": gate,
    }


def argmax_ceiling(prepared, posteriors, table):
    rows = {"stationary": [], "full_cycle": []}
    oracle_map = np.asarray(table["oracle_controller_map"], dtype=np.int32)
    for sequence, posterior in zip(prepared, posteriors):
        inferred_modes = np.argmax(before_action(posterior), axis=1)
        decisions = oracle_map[inferred_modes]
        kind = str(sequence["identity"]["kind"])
        rows[kind].append(utility.routing_metrics(
            decisions, sequence["labels"], oracle_map, burnin=32))
    return {kind: _aggregate_metrics(values) for kind, values in rows.items()}


def classifier_diagnostics(prepared, posteriors):
    confusion = {
        "stationary": np.zeros((4, 4), dtype=np.int64),
        "full_cycle": np.zeros((4, 4), dtype=np.int64),
    }
    by_behavior: dict[str, list[float]] = {}
    by_mode: dict[str, dict[str, list[float]]] = {
        "stationary": {str(mode): [] for mode in range(4)},
        "full_cycle": {str(mode): [] for mode in range(4)},
    }
    age_correct = {
        f"{start}-{stop - 1}": [] for start, stop in AGE_BINS
    }
    for sequence, posterior in zip(prepared, posteriors):
        labels = np.asarray(sequence["labels"], dtype=np.int32)
        predictions = np.argmax(before_action(posterior), axis=1)
        index = np.arange(len(labels)) >= 32
        kind = str(sequence["identity"]["kind"])
        for true_mode, predicted_mode in zip(labels[index], predictions[index]):
            confusion[kind][int(true_mode), int(predicted_mode)] += 1
        behavior = str(sequence["identity"]["controller"])
        by_behavior.setdefault(behavior, []).append(float(np.mean(
            predictions[index] == labels[index])))
        for mode in range(4):
            selected = index & (labels == mode)
            if np.any(selected):
                by_mode[kind][str(mode)].append(float(np.mean(
                    predictions[selected] == labels[selected])))
        if kind == "full_cycle":
            age = np.zeros(len(labels), dtype=np.int32)
            for step in range(1, len(labels)):
                age[step] = 0 if labels[step] != labels[step - 1] \
                    else age[step - 1] + 1
            correct = predictions == labels
            for start, stop in AGE_BINS:
                selected = index & (age >= start) & (age < stop)
                if np.any(selected):
                    age_correct[f"{start}-{stop - 1}"].append(float(
                        np.mean(correct[selected])))
    return {
        "confusion": {
            kind: matrix.tolist() for kind, matrix in confusion.items()
        },
        "accuracy_by_behavior": {
            key: float(np.mean(values))
            for key, values in sorted(by_behavior.items())
        },
        "accuracy_by_kind_and_mode": {
            kind: {
                mode: float(np.mean(values)) if values else 0.0
                for mode, values in modes.items()
            }
            for kind, modes in by_mode.items()
        },
        "full_cycle_accuracy_by_segment_age": {
            key: float(np.mean(values)) if values else 0.0
            for key, values in age_correct.items()
        },
    }


def candidate_key(row):
    stationary = row["stationary"]
    full_cycle = row["full_cycle"]
    return (
        int(row["gate"]["passed"]),
        min(stationary["action_accuracy"], full_cycle["action_accuracy"]),
        -(stationary["wrong_route_rate"] + full_cycle["wrong_route_rate"]),
        -full_cycle["median_switch_delay"],
    )


def render_markdown(payload):
    best = payload["best_candidate"]
    ceiling = payload["argmax_ceiling"]
    lines = [
        "# BAPR-v3 sequence-router diagnostic",
        "",
        "Only independent internal validation streams 9100/9200 were used; "
        "sealed utility holdouts 7100-7500 remain unopened.",
        "",
        "| Route | Stationary accuracy / wrong | Full-cycle accuracy / wrong | Delay | Gate |",
        "|---|---:|---:|---:|---|",
        (
            f"| frozen c80h8 | {payload['frozen_result']['stationary']['action_accuracy']:.1%} / "
            f"{payload['frozen_result']['stationary']['wrong_route_rate']:.1%} | "
            f"{payload['frozen_result']['full_cycle']['action_accuracy']:.1%} / "
            f"{payload['frozen_result']['full_cycle']['wrong_route_rate']:.1%} | "
            f"{payload['frozen_result']['full_cycle']['median_switch_delay']:.1f} | fail |"
        ),
        (
            f"| best internal candidate | {best['stationary']['action_accuracy']:.1%} / "
            f"{best['stationary']['wrong_route_rate']:.1%} | "
            f"{best['full_cycle']['action_accuracy']:.1%} / "
            f"{best['full_cycle']['wrong_route_rate']:.1%} | "
            f"{best['full_cycle']['median_switch_delay']:.1f} | "
            f"{'pass' if best['gate']['passed'] else 'fail'} |"
        ),
        (
            f"| argmax/no-fallback ceiling | {ceiling['stationary']['action_accuracy']:.1%} / "
            f"{ceiling['stationary']['wrong_route_rate']:.1%} | "
            f"{ceiling['full_cycle']['action_accuracy']:.1%} / "
            f"{ceiling['full_cycle']['wrong_route_rate']:.1%} | "
            f"{ceiling['full_cycle']['median_switch_delay']:.1f} | diagnostic |"
        ),
        "",
        "Best candidate: `" + json.dumps(best["config"], sort_keys=True) + "`.",
        "",
        "Per-mode action-time physical accuracy: `" + json.dumps(
            payload["classifier"]["accuracy_by_kind_and_mode"],
            sort_keys=True) + "`.",
        "",
        "Full-cycle accuracy by steps since segment start: `" + json.dumps(
            payload["classifier"]["full_cycle_accuracy_by_segment_age"],
            sort_keys=True) + "`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    protocol.configure()
    if JSON_PATH.is_file():
        payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
        if payload.get("schema") != SCHEMA:
            raise ValueError("stale sequence-router diagnostic output")
        print(f"Complete sequence-router diagnostic exists: {JSON_PATH}")
        return
    manifest = protocol.load_manifest()
    table = utility.load_utility_table()
    config, agents, policy_states = specialist_audit._controller_policy_states(
        protocol.FAMILY)
    policy_graphdef = nnx.graphdef(agents["robust"].policy)
    emission_model, _ = trainer.load_frozen_emission_model(
        agents["robust"].obs_dim, agents["robust"].act_dim)
    model = load_sequence_model(manifest)
    print("Collecting independent sequence-router diagnostic streams", flush=True)
    raw = trainer.emission_train.collect_stationary_sequences(
        config, policy_graphdef, policy_states,
        protocol.VALIDATION_EVENT_SEEDS, 1000)
    raw += trainer.collect_full_cycle_sequences(
        config, policy_graphdef, policy_states,
        protocol.VALIDATION_EVENT_SEEDS)
    prepared = trainer.prepare_sequences(emission_model, raw)
    logits_posteriors = {
        temperature: [posterior_trace(model, sequence["evidence"], temperature)
                      for sequence in prepared]
        for temperature in TEMPERATURES
    }
    base_config = protocol.emission.RouterConfig.from_dict(
        manifest["decision_config"])
    candidates = []
    for temperature in TEMPERATURES:
        for confidence in CONFIDENCE_THRESHOLDS:
            for margin in MARGIN_THRESHOLDS:
                for history in MIN_HISTORIES:
                    decision_config = replace(
                        base_config,
                        confidence_threshold=confidence,
                        margin_threshold=margin,
                        min_history=history,
                    )
                    row = score_config(
                        prepared, logits_posteriors[temperature], table,
                        decision_config)
                    row["config"] = {
                        "temperature": temperature,
                        "confidence_threshold": confidence,
                        "margin_threshold": margin,
                        "min_history": history,
                    }
                    candidates.append(row)
    best = max(candidates, key=candidate_key)
    frozen = score_config(
        prepared, logits_posteriors[1.0], table, base_config)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete",
        "family": protocol.FAMILY,
        "env": protocol.ENV,
        "validation_event_seeds": list(protocol.VALIDATION_EVENT_SEEDS),
        "sealed_holdout_event_seeds": list(protocol.HOLDOUT_EVENT_SEEDS),
        "sequence_manifest_file": protocol.file_record(protocol.MANIFEST_PATH),
        "sequence_parameter_file": protocol.file_record(protocol.MODEL_PATH),
        "emission_manifest_file": protocol.file_record(
            protocol.emission.MANIFEST_PATH),
        "utility_table_file": protocol.file_record(utility.TABLE_PATH),
        "candidate_grid": {
            "temperatures": list(TEMPERATURES),
            "confidence_thresholds": list(CONFIDENCE_THRESHOLDS),
            "margin_thresholds": list(MARGIN_THRESHOLDS),
            "min_histories": list(MIN_HISTORIES),
        },
        "frozen_result": frozen,
        "best_candidate": best,
        "argmax_ceiling": argmax_ceiling(
            prepared, logits_posteriors[1.0], table),
        "classifier": classifier_diagnostics(
            prepared, logits_posteriors[1.0]),
        "candidates": sorted(candidates, key=candidate_key, reverse=True),
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MARKDOWN_PATH.write_text(render_markdown(payload), encoding="utf-8")
    protocol.write_json_atomic(JSON_PATH, payload)
    print(
        "SEQUENCE ROUTER DIAGNOSTIC COMPLETE: "
        f"best_gate={'PASS' if best['gate']['passed'] else 'FAIL'} "
        f"output={JSON_PATH}",
        flush=True,
    )


if __name__ == "__main__":
    main()
