"""Aggregate the frozen causal-fallback cross-student audit."""
from __future__ import annotations

import math
import statistics
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_policy_distillation_transient_fallback_cross_student_v1 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_policy_distillation_transient_fallback_cross_student_audit_v1 as audit,
)


T_CRITICAL_95 = 2.7764451051977987


def _mean(values) -> float:
    values = [float(value) for value in values]
    if not values:
        raise ValueError("cannot average empty cross-student values")
    return float(statistics.fmean(values))


def _comparison(events, candidate: str, reference: str) -> dict[str, Any]:
    event_deltas = {}
    paired = []
    candidate_values = []
    reference_values = []
    for event_seed in protocol.EVENT_SEEDS:
        left = events[event_seed][candidate]["returns"]
        right = events[event_seed][reference]["returns"]
        if len(left) != len(right):
            raise ValueError("unpaired cross-student returns")
        deltas = [float(a) - float(b) for a, b in zip(left, right)]
        event_deltas[str(event_seed)] = _mean(deltas)
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


def _positive(row: dict[str, Any]) -> bool:
    return (
        row["mean_delta"] > 0.0
        and row["event_seed_wins"] >= 4
        and row["cluster_95pct_t_interval"][0] > 0.0
    )


def _student_result(student_seed: int) -> dict[str, Any]:
    events = {}
    for event_seed in protocol.EVENT_SEEDS:
        audit.validate_audit(student_seed, event_seed)
        result = protocol.read_json(
            protocol.audit_dir(student_seed, event_seed) / "results.json")
        events[event_seed] = {
            row["arm"]: row for row in result["switching"]
        }

    arm_means = {
        arm: _mean(
            value
            for event_seed in protocol.EVENT_SEEDS
            for value in events[event_seed][arm]["returns"]
        )
        for arm in protocol.ARMS
    }
    comparisons = {
        "fallback_minus_learned": _comparison(
            events, protocol.SELECTED_CONFIG_NAME, protocol.LEARNED_ARM),
        "fallback_minus_robust719": _comparison(
            events, protocol.SELECTED_CONFIG_NAME, protocol.ROBUST_ARM),
        "fallback_minus_oracle": _comparison(
            events, protocol.SELECTED_CONFIG_NAME, protocol.ORACLE_ARM),
        "learned_minus_robust719": _comparison(
            events, protocol.LEARNED_ARM, protocol.ROBUST_ARM),
        "oracle_minus_robust719": _comparison(
            events, protocol.ORACLE_ARM, protocol.ROBUST_ARM),
    }
    oracle_headroom = arm_means[protocol.ORACLE_ARM] - arm_means[protocol.ROBUST_ARM]
    recovery = (
        (arm_means[protocol.SELECTED_CONFIG_NAME] - arm_means[protocol.ROBUST_ARM])
        / oracle_headroom
        if oracle_headroom > 0.0 else float("-inf")
    )
    fallback_fraction = _mean(
        events[event_seed][protocol.SELECTED_CONFIG_NAME]["fallback_action_fraction"]
        for event_seed in protocol.EVENT_SEEDS
    )
    zero_termination = all(
        float(events[event_seed][protocol.SELECTED_CONFIG_NAME]["terminated_rate"])
        == 0.0
        for event_seed in protocol.EVENT_SEEDS
    )
    gates = {
        "beats_learned": _positive(comparisons["fallback_minus_learned"]),
        "beats_robust719": _positive(comparisons["fallback_minus_robust719"]),
        "recovers_half_oracle_headroom": recovery >= 0.50,
        "fallback_fraction_at_most_20pct": fallback_fraction <= 0.20,
        "zero_termination": zero_termination,
    }
    return {
        "student_seed": int(student_seed),
        "arm_means": arm_means,
        "comparisons": comparisons,
        "oracle_headroom_recovery": recovery,
        "fallback_action_fraction": fallback_fraction,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Frozen transient-fallback cross-student audit",
        "",
        f"Frozen config: `{protocol.SELECTED_CONFIG_NAME}`.",
        "",
        "| student seed | robust 719 | learned | oracle | fallback | fallback - robust | fallback - learned | fallback use | pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for seed in protocol.STUDENT_SEEDS:
        row = payload["students"][str(seed)]
        means = row["arm_means"]
        vs_robust = row["comparisons"]["fallback_minus_robust719"]
        vs_learned = row["comparisons"]["fallback_minus_learned"]
        lines.append(
            f"| {seed} | {means[protocol.ROBUST_ARM]:.1f} | "
            f"{means[protocol.LEARNED_ARM]:.1f} | "
            f"{means[protocol.ORACLE_ARM]:.1f} | "
            f"{means[protocol.SELECTED_CONFIG_NAME]:.1f} | "
            f"{vs_robust['mean_delta']:+.1f} ({vs_robust['event_seed_wins']}/5) | "
            f"{vs_learned['mean_delta']:+.1f} ({vs_learned['event_seed_wins']}/5) | "
            f"{100.0 * row['fallback_action_fraction']:.2f}% | "
            f"{row['passed']} |"
        )
    lines.extend([
        "",
        "## Decision",
        "",
        f"- Student passes: `{payload['student_passes']}/3`.",
        f"- Cross-initialization gate: `{payload['cross_student_pass']}`.",
        "",
        payload["recommendation"],
        "",
    ])
    return "\n".join(lines)


def run() -> dict[str, Any]:
    students = {
        str(student_seed): _student_result(student_seed)
        for student_seed in protocol.STUDENT_SEEDS
    }
    passes = sum(row["passed"] for row in students.values())
    cross_student_pass = passes == len(protocol.STUDENT_SEEDS)
    recommendation = (
        "The frozen causal fallback reproduces across every existing mode-head "
        "initialization. Freeze this deployable mechanism; only now consider "
        "whether stale/soft-belief training adds value beyond it on new "
        "development models."
        if cross_student_pass else
        "The fallback effect is initialization-dependent. Do not train a new "
        "belief-augmented student from this branch; retain the passing seeds "
        "as diagnostics and analyze the failed initialization."
    )
    payload = {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "development_only": True,
        "selected_config": protocol.SELECTED_CONFIG_NAME,
        "event_seeds": list(protocol.EVENT_SEEDS),
        "students": students,
        "student_passes": passes,
        "cross_student_pass": cross_student_pass,
        "recommendation": recommendation,
    }
    protocol.ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    protocol.write_json_atomic(protocol.analysis_json(), payload)
    protocol.write_text_atomic(protocol.analysis_markdown(), _markdown(payload))
    print(
        "CROSS-STUDENT FALLBACK ANALYSIS COMPLETE: "
        f"passed={cross_student_pass}",
        flush=True,
    )
    return payload


def main() -> None:
    run()


if __name__ == "__main__":
    main()
