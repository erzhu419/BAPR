"""Aggregate the independent transient-fallback audit."""
from __future__ import annotations

import math
import statistics
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_policy_distillation_transient_fallback_v1 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_policy_distillation_transient_fallback_audit_v1 as audit,
)
from jax_experiments.analysis import (
    select_regime_polarity_policy_distillation_transient_fallback_v1 as selection,
)


T_CRITICAL_95 = 2.7764451051977987


def _mean(values) -> float:
    values = [float(value) for value in values]
    if not values:
        raise ValueError("cannot average empty transient-fallback values")
    return float(statistics.fmean(values))


def _comparison(events, candidate: str, reference: str) -> dict[str, Any]:
    event_deltas = {}
    paired = []
    candidate_values = []
    reference_values = []
    for seed in protocol.AUDIT_EVENT_SEEDS:
        left = events[int(seed)][candidate]["returns"]
        right = events[int(seed)][reference]["returns"]
        if len(left) != len(right):
            raise ValueError("unpaired transient-fallback returns")
        deltas = [float(a) - float(b) for a, b in zip(left, right)]
        event_deltas[str(seed)] = _mean(deltas)
        paired.extend(deltas)
        candidate_values.extend(float(value) for value in left)
        reference_values.extend(float(value) for value in right)
    clusters = list(event_deltas.values())
    center = _mean(clusters)
    half_width = (
        T_CRITICAL_95 * statistics.stdev(clusters) / math.sqrt(len(clusters))
    )
    return {
        "candidate_mean": _mean(candidate_values),
        "reference_mean": _mean(reference_values),
        "mean_delta": _mean(paired),
        "event_seed_deltas": event_deltas,
        "event_seed_wins": sum(value > 0.0 for value in clusters),
        "cluster_95pct_t_interval": [center - half_width, center + half_width],
    }


def _positive(row: dict[str, Any], min_wins: int = 4) -> bool:
    return (
        row["mean_delta"] > 0.0
        and row["event_seed_wins"] >= int(min_wins)
        and row["cluster_95pct_t_interval"][0] > 0.0
    )


def _markdown(payload: dict[str, Any]) -> str:
    selected = payload["selected_config"]
    lines = [
        "# Causal transient-fallback audit",
        "",
        f"Development-selected config: `{selected}`.",
        "",
        "| arm | switching return |",
        "|---|---:|",
    ]
    for arm, value in payload["arm_means"].items():
        lines.append(f"| `{arm}` | {value:.1f} |")
    lines.extend([
        "",
        "| comparison | delta | event wins | clustered 95% interval |",
        "|---|---:|---:|---:|",
    ])
    for name, row in payload["comparisons"].items():
        interval = row["cluster_95pct_t_interval"]
        lines.append(
            f"| `{name}` | {row['mean_delta']:+.1f} | "
            f"{row['event_seed_wins']}/5 | "
            f"[{interval[0]:+.1f}, {interval[1]:+.1f}] |"
        )
    lines.extend([
        "",
        "## Decision",
        "",
        f"- Oracle headroom recovery: "
        f"`{100.0 * payload['oracle_headroom_recovery']:.1f}%`.",
        f"- Fallback action fraction: "
        f"`{100.0 * payload['fallback_action_fraction']:.1f}%`.",
        f"- Zero termination: `{payload['gates']['zero_termination']}`.",
        f"- Beats learned posterior: "
        f"`{payload['gates']['beats_learned']}`.",
        f"- Beats robust 719: `{payload['gates']['beats_robust719']}`.",
        f"- Authorize stale/soft-belief training: "
        f"`{payload['authorize_belief_augmentation_training']}`.",
        "",
        payload["recommendation"],
        "",
    ])
    return "\n".join(lines)


def run() -> dict[str, Any]:
    selection.validate_selection()
    selected_payload = protocol.read_json(protocol.selection_json())
    selected = protocol.require_config(
        selected_payload["selected_config"]).name
    events = {}
    for seed in protocol.AUDIT_EVENT_SEEDS:
        audit.validate_audit(seed)
        result = protocol.read_json(protocol.audit_dir(seed) / "results.json")
        events[int(seed)] = {
            row["arm"]: row for row in result["switching"]
        }

    arms = (*protocol.BASELINE_ARMS, selected)
    arm_means = {
        arm: _mean(
            value
            for seed in protocol.AUDIT_EVENT_SEEDS
            for value in events[int(seed)][arm]["returns"]
        )
        for arm in arms
    }
    comparisons = {
        "fallback_minus_learned": _comparison(
            events, selected, protocol.LEARNED_ARM),
        "fallback_minus_robust719": _comparison(
            events, selected, protocol.ROBUST_ARM),
        "fallback_minus_oracle": _comparison(
            events, selected, protocol.ORACLE_ARM),
        "learned_minus_robust719": _comparison(
            events, protocol.LEARNED_ARM, protocol.ROBUST_ARM),
        "oracle_minus_robust719": _comparison(
            events, protocol.ORACLE_ARM, protocol.ROBUST_ARM),
    }
    oracle_headroom = (
        arm_means[protocol.ORACLE_ARM] - arm_means[protocol.ROBUST_ARM])
    recovered = (
        (arm_means[selected] - arm_means[protocol.ROBUST_ARM])
        / oracle_headroom
        if oracle_headroom > 0.0 else float("-inf")
    )
    fallback_fraction = _mean(
        events[int(seed)][selected]["fallback_action_fraction"]
        for seed in protocol.AUDIT_EVENT_SEEDS
    )
    zero_termination = all(
        float(events[int(seed)][selected]["terminated_rate"]) == 0.0
        for seed in protocol.AUDIT_EVENT_SEEDS
    )
    gates = {
        "beats_learned": _positive(comparisons["fallback_minus_learned"]),
        "beats_robust719": _positive(
            comparisons["fallback_minus_robust719"]),
        "recovers_half_oracle_headroom": recovered >= 0.50,
        "fallback_fraction_at_most_20pct": fallback_fraction <= 0.20,
        "zero_termination": zero_termination,
    }
    authorized = all(gates.values())
    recommendation = (
        "The frozen causal fallback clears its independent gate. A new "
        "development-only student may now be trained with stale and soft "
        "belief augmentation while retaining the same explicit robust "
        "fallback; these audit events remain sealed."
        if authorized else
        "The causal fallback does not reliably convert the small oracle "
        "headroom into a robust advantage. Do not train another student on "
        "this benchmark from this branch."
    )
    payload = {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "development_only": True,
        "selected_config": selected,
        "selected_config_values": protocol.require_config(selected).to_dict(),
        "screen_selection_manifest": protocol.file_record(
            protocol.selection_manifest()),
        "audit_event_seeds": list(protocol.AUDIT_EVENT_SEEDS),
        "arm_means": arm_means,
        "comparisons": comparisons,
        "oracle_headroom_recovery": recovered,
        "fallback_action_fraction": fallback_fraction,
        "gates": gates,
        "authorize_belief_augmentation_training": authorized,
        "recommendation": recommendation,
    }
    protocol.ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    protocol.write_json_atomic(protocol.analysis_json(), payload)
    protocol.write_text_atomic(protocol.analysis_markdown(), _markdown(payload))
    print(
        "TRANSIENT FALLBACK ANALYSIS COMPLETE: "
        f"authorized={authorized}",
        flush=True,
    )
    return payload


def main() -> None:
    run()


if __name__ == "__main__":
    main()
