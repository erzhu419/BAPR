"""Aggregate the frozen mode-head context-ablation diagnostic."""
from __future__ import annotations

import math
import statistics
from typing import Any

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_policy_distillation_context_ablation_v1 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_policy_distillation_context_ablation_audit_v1 as audit,
)


T_CRITICAL_95 = 2.7764451051977987


def _mean(values) -> float:
    values = [float(value) for value in values]
    if not values:
        raise ValueError("cannot average an empty sequence")
    return float(statistics.fmean(values))


def _event_results():
    events = {}
    for event_seed in protocol.AUDIT_EVENT_SEEDS:
        audit.validate_audit(event_seed)
        result = protocol.read_json(protocol.audit_dir(
            protocol.TEACHER_GROUP,
            protocol.STUDENT_SEED,
            event_seed,
        ) / "results.json")
        events[int(event_seed)] = {
            "switching": {row["arm"]: row for row in result["switching"]},
            "stationary": {
                (row["arm"], int(row["mode"])): row
                for row in result["stationary"]
            },
        }
    return events


def _arm_values(events, arm: str):
    return {
        event_seed: [
            float(value)
            for value in event["switching"][arm]["returns"]
        ]
        for event_seed, event in events.items()
    }


def _comparison(candidate, reference) -> dict[str, Any]:
    event_deltas = {}
    paired_deltas = []
    candidate_values = []
    reference_values = []
    for event_seed in protocol.AUDIT_EVENT_SEEDS:
        left = candidate[int(event_seed)]
        right = reference[int(event_seed)]
        if len(left) != len(right):
            raise ValueError("unpaired context-ablation return vectors")
        deltas = [float(a) - float(b) for a, b in zip(left, right)]
        event_deltas[str(event_seed)] = _mean(deltas)
        paired_deltas.extend(deltas)
        candidate_values.extend(left)
        reference_values.extend(right)
    clusters = list(event_deltas.values())
    cluster_mean = _mean(clusters)
    cluster_sem = statistics.stdev(clusters) / math.sqrt(len(clusters))
    half_width = T_CRITICAL_95 * cluster_sem
    return {
        "candidate_mean": _mean(candidate_values),
        "reference_mean": _mean(reference_values),
        "mean_delta": _mean(paired_deltas),
        "event_seed_deltas": event_deltas,
        "event_seed_wins": sum(value > 0.0 for value in clusters),
        "episode_wins": sum(value > 0.0 for value in paired_deltas),
        "cluster_count": len(clusters),
        "cluster_95pct_t_interval": [
            cluster_mean - half_width,
            cluster_mean + half_width,
        ],
    }


def _stationary_by_mode(events, arm: str) -> dict[str, float]:
    return {
        str(mode): _mean(
            value
            for event in events.values()
            for value in event["stationary"][(arm, mode)]["returns"]
        )
        for mode in protocol.MODES
    }


def _posterior_summary(events, arm: str) -> dict[str, float]:
    metrics = [
        event["switching"][arm]["posterior_metrics"]
        for event in events.values()
    ]
    return {
        key: _mean(row[key] for row in metrics)
        for key in (
            "mode_accuracy",
            "brier_score",
            "median_switch_delay",
            "p90_switch_delay",
        )
    }


def _comparison_pass(row: dict[str, Any]) -> bool:
    return (
        row["mean_delta"] > 0.0
        and row["cluster_95pct_t_interval"][0] > 0.0
    )


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Frozen mode-head context-ablation diagnostic",
        "",
        "This is a post-confirmation mechanism diagnostic. It cannot select "
        "another model or reopen the failed confirmation.",
        "",
        "| arm | switching return |",
        "|---|---:|",
    ]
    for arm in protocol.ARM_LABELS:
        lines.append(
            f"| `{arm}` | {payload['arm_means'][arm]:.1f} |")
    lines.extend([
        "",
        "## Paired context comparisons",
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
        "## Mechanism",
        "",
        f"- Classification: `{payload['classification']}`.",
        f"- Oracle context beats uniform and every fixed context with "
        f"positive intervals: "
        f"`{payload['gates']['oracle_context_value']}`.",
        f"- Learned posterior beats uniform and every fixed context with "
        f"positive intervals: "
        f"`{payload['gates']['learned_context_value']}`.",
        f"- Matching fixed head is stationary-optimal in "
        f"`{payload['stationary_diagonal_optima']}/4` modes.",
        f"- Learned posterior accuracy: "
        f"`{payload['learned_posterior']['mode_accuracy']:.4f}`; median "
        f"switch delay: "
        f"`{payload['learned_posterior']['median_switch_delay']:.1f}` steps.",
        "",
        payload["recommendation"],
        "",
    ])
    return "\n".join(lines)


def run() -> dict[str, Any]:
    protocol.validate_frozen_candidate()
    events = _event_results()
    values = {
        arm: _arm_values(events, arm) for arm in protocol.ARM_LABELS
    }
    arm_means = {
        arm: _mean(value for rows in values[arm].values() for value in rows)
        for arm in protocol.ARM_LABELS
    }
    learned = values["student_learned"]
    oracle = values["student_oracle"]
    uniform = values["student_uniform"]
    fixed_arms = [f"student_fixed_{mode}" for mode in protocol.MODES]
    comparisons = {
        "learned_minus_robust719": _comparison(
            learned, values[protocol.FIXED_ROBUST_ARM]),
        "oracle_minus_robust719": _comparison(
            oracle, values[protocol.FIXED_ROBUST_ARM]),
        "learned_minus_oracle": _comparison(learned, oracle),
        "learned_minus_uniform": _comparison(learned, uniform),
        "oracle_minus_uniform": _comparison(oracle, uniform),
        "learned_minus_cyclic": _comparison(
            learned, values["student_cyclic"]),
        "learned_minus_shuffled": _comparison(
            learned, values["student_shuffled"]),
        **{
            f"learned_minus_fixed_{mode}": _comparison(
                learned, values[f"student_fixed_{mode}"])
            for mode in protocol.MODES
        },
        **{
            f"oracle_minus_fixed_{mode}": _comparison(
                oracle, values[f"student_fixed_{mode}"])
            for mode in protocol.MODES
        },
    }

    stationary = {
        arm: _stationary_by_mode(events, arm)
        for arm in protocol.STUDENT_CONTEXT_ARMS
        if arm not in ("student_cyclic", "student_shuffled")
    }
    diagonal_optima = 0
    for mode in protocol.MODES:
        oracle_value = stationary["student_oracle"][str(mode)]
        matching = stationary[f"student_fixed_{mode}"][str(mode)]
        if not np.isclose(oracle_value, matching, atol=1e-5, rtol=1e-7):
            raise ValueError(
                f"oracle/fixed context mismatch in stationary mode {mode}")
        fixed_values = {
            fixed_mode: stationary[f"student_fixed_{fixed_mode}"][str(mode)]
            for fixed_mode in protocol.MODES
        }
        if np.isclose(
            fixed_values[mode], max(fixed_values.values()),
            atol=1e-5, rtol=1e-7,
        ):
            diagonal_optima += 1

    oracle_context_value = (
        _comparison_pass(comparisons["oracle_minus_uniform"])
        and all(
            _comparison_pass(comparisons[f"oracle_minus_fixed_{mode}"])
            for mode in protocol.MODES
        )
    )
    learned_context_value = (
        _comparison_pass(comparisons["learned_minus_uniform"])
        and all(
            _comparison_pass(comparisons[f"learned_minus_fixed_{mode}"])
            for mode in protocol.MODES
        )
    )
    if not oracle_context_value:
        classification = "student_not_dynamically_context_specialized"
        recommendation = (
            "The frozen student's strong return is primarily robust-policy "
            "compression; its mode heads do not establish dynamic context "
            "value. Retain the failed confirmation and stop this branch."
        )
    elif learned_context_value:
        classification = "causal_posterior_realizes_context_value"
        recommendation = (
            "The student is genuinely context-dependent, but the prior "
            "confirmation still failed superiority to robust 719. Report the "
            "mechanism result without reopening model selection."
        )
    else:
        classification = "context_specialized_but_not_realized_causally"
        recommendation = (
            "The frozen heads contain dynamic context value, but the causal "
            "posterior path does not realize it reliably. This remains a "
            "mechanism explanation, not permission to tune on confirmation."
        )

    payload = {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "confirmatory": False,
        "selection_forbidden": True,
        "env": protocol.ENV,
        "family": protocol.FAMILY,
        "variant": protocol.TEACHER_GROUP,
        "student_seed": protocol.STUDENT_SEED,
        "event_seeds": list(protocol.AUDIT_EVENT_SEEDS),
        "arm_means": arm_means,
        "comparisons": comparisons,
        "stationary_by_mode": stationary,
        "stationary_diagonal_optima": diagonal_optima,
        "learned_posterior": _posterior_summary(
            events, "student_learned"),
        "gates": {
            "oracle_context_value": oracle_context_value,
            "learned_context_value": learned_context_value,
        },
        "classification": classification,
        "recommendation": recommendation,
    }
    protocol.write_json_atomic(protocol.analysis_json(), payload)
    protocol.write_text_atomic(protocol.analysis_markdown(), _markdown(payload))
    print(f"CONTEXT ABLATION ANALYSIS COMPLETE: {protocol.ANALYSIS_ROOT}")
    return payload


def main() -> None:
    run()


if __name__ == "__main__":
    main()

