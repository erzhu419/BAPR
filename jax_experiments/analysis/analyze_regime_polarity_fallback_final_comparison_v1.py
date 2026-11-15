"""Aggregate the preregistered causal-fallback final comparison."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_fallback_final_comparison_v1 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_fallback_final_audit_v1 as auditor,
)


METHODS = ("bapr", *protocol.BASELINE_ROLES)


def _two_sample_interval(candidate, reference) -> list[float]:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if candidate.shape != reference.shape or candidate.shape != (5,):
        raise ValueError("final comparison requires two five-seed vectors")
    delta = float(np.mean(candidate) - np.mean(reference))
    standard_error = math.sqrt(
        float(np.var(candidate, ddof=1) / len(candidate))
        + float(np.var(reference, ddof=1) / len(reference)))
    margin = protocol.CLUSTER_T_CRITICAL_95 * standard_error
    return [delta - margin, delta + margin]


def _seed_summary(payload: dict[str, Any]) -> dict[str, float]:
    events = payload["events"]
    switching = [float(row["switching"]["return_mean"]) for row in events]
    stationary = [
        float(mode["return_mean"])
        for row in events
        for mode in row["stationary"]
    ]
    switching_termination = [
        float(row["switching"]["terminated_rate"]) for row in events]
    stationary_termination = [
        float(mode["terminated_rate"])
        for row in events
        for mode in row["stationary"]
    ]
    output = {
        "switching_mean": float(np.mean(switching)),
        "stationary_mean": float(np.mean(stationary)),
        "switching_terminated_rate": float(np.mean(switching_termination)),
        "stationary_terminated_rate": float(np.mean(stationary_termination)),
    }
    fallback = [
        float(row["switching"].get("fallback_action_fraction", np.nan))
        for row in events
    ]
    if np.all(np.isfinite(fallback)):
        output["fallback_action_fraction"] = float(np.mean(fallback))
    return output


def _comparison(candidate, reference) -> dict[str, Any]:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    interval = _two_sample_interval(candidate, reference)
    deltas = candidate - reference
    return {
        "candidate_mean": float(np.mean(candidate)),
        "reference_mean": float(np.mean(reference)),
        "mean_delta": float(np.mean(candidate) - np.mean(reference)),
        "conservative_two_sample_95pct_interval": interval,
        "registered_seed_slot_deltas": {
            str(seed): float(value)
            for seed, value in zip(protocol.TRAINING_SEEDS, deltas)
        },
        "registered_seed_slot_wins": int(np.sum(deltas > 0.0)),
    }


def analyze() -> dict[str, Any]:
    protocol.validate_registration()
    payloads: dict[str, dict[int, dict[str, Any]]] = {}
    summaries: dict[str, dict[int, dict[str, float]]] = {}
    for method in METHODS:
        payloads[method] = {}
        summaries[method] = {}
        for seed in protocol.TRAINING_SEEDS:
            auditor.validate_manifest(method, seed)
            path = protocol.audit_dir(method, seed) / "results.json"
            payload = protocol.read_json(path)
            payloads[method][seed] = payload
            summaries[method][seed] = _seed_summary(payload)

    # Pairing is valid at the event-stream level only. Every model must have
    # the exact same physical-mode trace for a given registered event.
    for event_index, event_seed in enumerate(protocol.FINAL_EVENT_SEEDS):
        hashes = {
            payloads[method][seed]["events"][event_index]["switching"]
            ["mode_trace_sha256"]
            for method in METHODS
            for seed in protocol.TRAINING_SEEDS
        }
        if len(hashes) != 1:
            raise ValueError(
                f"final mode streams differ for event {event_seed}: {hashes}")

    method_rows = {}
    for method in METHODS:
        switching = np.asarray([
            summaries[method][seed]["switching_mean"]
            for seed in protocol.TRAINING_SEEDS
        ])
        stationary = np.asarray([
            summaries[method][seed]["stationary_mean"]
            for seed in protocol.TRAINING_SEEDS
        ])
        method_rows[method] = {
            "seed_summaries": {
                str(seed): summaries[method][seed]
                for seed in protocol.TRAINING_SEEDS
            },
            "switching_mean": float(np.mean(switching)),
            "switching_std_over_seeds": float(np.std(switching, ddof=1)),
            "stationary_mean": float(np.mean(stationary)),
            "stationary_std_over_seeds": float(np.std(stationary, ddof=1)),
            "switching_terminated_rate": float(np.mean([
                summaries[method][seed]["switching_terminated_rate"]
                for seed in protocol.TRAINING_SEEDS
            ])),
            "stationary_terminated_rate": float(np.mean([
                summaries[method][seed]["stationary_terminated_rate"]
                for seed in protocol.TRAINING_SEEDS
            ])),
        }
        if method == "bapr":
            method_rows[method]["fallback_action_fraction"] = float(np.mean([
                summaries[method][seed]["fallback_action_fraction"]
                for seed in protocol.TRAINING_SEEDS
            ]))

    bapr_switching = np.asarray([
        summaries["bapr"][seed]["switching_mean"]
        for seed in protocol.TRAINING_SEEDS
    ])
    bapr_stationary = np.asarray([
        summaries["bapr"][seed]["stationary_mean"]
        for seed in protocol.TRAINING_SEEDS
    ])
    comparisons = {}
    for role in protocol.BASELINE_ROLES:
        reference = np.asarray([
            summaries[role][seed]["switching_mean"]
            for seed in protocol.TRAINING_SEEDS
        ])
        comparisons[f"bapr_minus_{role}"] = _comparison(
            bapr_switching, reference)

    strongest_switching = np.asarray([
        max(summaries[role][seed]["switching_mean"]
            for role in protocol.BASELINE_ROLES)
        for seed in protocol.TRAINING_SEEDS
    ])
    strongest_stationary = np.asarray([
        max(summaries[role][seed]["stationary_mean"]
            for role in protocol.BASELINE_ROLES)
        for seed in protocol.TRAINING_SEEDS
    ])
    strongest_termination = np.asarray([
        min(summaries[role][seed]["switching_terminated_rate"]
            for role in protocol.BASELINE_ROLES)
        for seed in protocol.TRAINING_SEEDS
    ])
    primary = _comparison(bapr_switching, strongest_switching)
    primary.update({
        "stationary_retention": float(
            np.mean(bapr_stationary) / max(np.mean(strongest_stationary), 1e-8)),
        "termination_gap": float(
            method_rows["bapr"]["switching_terminated_rate"]
            - np.mean(strongest_termination)),
    })
    primary["pass"] = bool(
        primary["mean_delta"] > 0.0
        and primary["conservative_two_sample_95pct_interval"][0] > 0.0
        and primary["registered_seed_slot_wins"] >= protocol.MIN_SEED_WINS
        and primary["stationary_retention"]
        >= protocol.MIN_STATIONARY_RETENTION
        and primary["termination_gap"] <= protocol.MAX_TERMINATION_GAP
    )
    comparisons["bapr_minus_strongest_baseline"] = primary

    resac_values = np.asarray([
        summaries["resac"][seed]["switching_mean"]
        for seed in protocol.TRAINING_SEEDS
    ])
    resac_reproduction_warning = bool(
        not np.all(np.isfinite(resac_values)) or np.mean(resac_values) <= 0.0)
    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "registration": protocol.FROZEN_REGISTRATION_RECORD,
        "training_seeds": list(protocol.TRAINING_SEEDS),
        "event_seeds": list(protocol.FINAL_EVENT_SEEDS),
        "methods": method_rows,
        "comparisons": comparisons,
        "primary_pass": primary["pass"],
        "resac_reproduction_warning": resac_reproduction_warning,
        "claim_scope": (
            "deployment-performance comparison; BAPR uses a frozen ten-"
            "controller teacher and is not sample-efficiency matched to the "
            "single-controller baselines"
        ),
    }


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Causal-fallback BAPR final comparison",
        "",
        "| method | switching | stationary | switching termination |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        row = payload["methods"][method]
        lines.append(
            f"| {method} | {row['switching_mean']:.1f} +/- "
            f"{row['switching_std_over_seeds']:.1f} | "
            f"{row['stationary_mean']:.1f} +/- "
            f"{row['stationary_std_over_seeds']:.1f} | "
            f"{row['switching_terminated_rate']:.3f} |"
        )
    primary = payload["comparisons"]["bapr_minus_strongest_baseline"]
    interval = primary["conservative_two_sample_95pct_interval"]
    lines += [
        "",
        "## Frozen decision",
        "",
        f"- BAPR minus strongest baseline: {primary['mean_delta']:+.1f}",
        f"- Conservative 95% interval: [{interval[0]:+.1f},{interval[1]:+.1f}]",
        f"- Registered seed-slot wins: {primary['registered_seed_slot_wins']}/5",
        f"- Stationary retention: {primary['stationary_retention']:.1%}",
        f"- Termination gap: {primary['termination_gap']:+.3f}",
        f"- Primary pass: `{payload['primary_pass']}`",
        f"- RE-SAC reproduction warning: "
        f"`{payload['resac_reproduction_warning']}`",
        "",
        payload["claim_scope"],
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    payload = analyze()
    protocol.write_json_atomic(protocol.analysis_json(), payload)
    protocol.write_text_atomic(protocol.analysis_markdown(), markdown(payload))
    print(
        "FINAL COMPARISON COMPLETE: "
        f"primary_pass={payload['primary_pass']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
