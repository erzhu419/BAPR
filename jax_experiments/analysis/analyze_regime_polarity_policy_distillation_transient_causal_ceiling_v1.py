"""Aggregate the delayed-oracle causal-ceiling diagnostic."""
from __future__ import annotations

import math
import statistics
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_policy_distillation_transient_causal_ceiling_v1 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_policy_distillation_transient_causal_ceiling_audit_v1 as audit,
)


T_CRITICAL_95 = 2.7764451051977987
MIN_REDUCIBLE_FRACTION_OF_ROBUST = 0.02
MIN_STUDENTS_FOR_TRAINING = 2


def _mean(values) -> float:
    values = [float(value) for value in values]
    if not values:
        raise ValueError("cannot average empty causal-ceiling values")
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
            raise ValueError("unpaired causal-ceiling returns")
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


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else float("-inf")


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
        "fallback_minus_robust719": _comparison(
            events, protocol.FALLBACK_ARM, protocol.ROBUST_ARM),
        "delay1_minus_fallback": _comparison(
            events, protocol.delay_arm(1), protocol.FALLBACK_ARM),
        "delay1_minus_robust719": _comparison(
            events, protocol.delay_arm(1), protocol.ROBUST_ARM),
        "oracle_minus_delay1": _comparison(
            events, protocol.ORACLE_ARM, protocol.delay_arm(1)),
        "oracle_minus_robust719": _comparison(
            events, protocol.ORACLE_ARM, protocol.ROBUST_ARM),
    }
    for delay in protocol.DELAY_STEPS[1:]:
        comparisons[f"delay{delay}_minus_robust719"] = _comparison(
            events, protocol.delay_arm(delay), protocol.ROBUST_ARM)

    robust = arm_means[protocol.ROBUST_ARM]
    oracle_headroom = arm_means[protocol.ORACLE_ARM] - robust
    causal_headroom = arm_means[protocol.delay_arm(1)] - robust
    fallback_headroom = arm_means[protocol.FALLBACK_ARM] - robust
    delay1_retention = _safe_ratio(causal_headroom, oracle_headroom)
    fallback_zero_recovery = _safe_ratio(fallback_headroom, oracle_headroom)
    fallback_causal_recovery = _safe_ratio(fallback_headroom, causal_headroom)
    gap = comparisons["delay1_minus_fallback"]
    material_reducible_gap = (
        gap["mean_delta"] / max(abs(robust), 1.0)
        >= MIN_REDUCIBLE_FRACTION_OF_ROBUST
        and gap["event_seed_wins"] >= 4
        and gap["cluster_95pct_t_interval"][0] > 0.0
    )
    zero_termination = all(
        float(events[event_seed][arm]["terminated_rate"]) == 0.0
        for event_seed in protocol.EVENT_SEEDS
        for arm in protocol.ARMS
    )
    return {
        "student_seed": int(student_seed),
        "arm_means": arm_means,
        "comparisons": comparisons,
        "delay1_oracle_headroom_retention": delay1_retention,
        "fallback_zero_delay_headroom_recovery": fallback_zero_recovery,
        "fallback_causal_headroom_recovery": fallback_causal_recovery,
        "material_reducible_post_observation_gap": material_reducible_gap,
        "zero_termination": zero_termination,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Transient delayed-oracle causal-ceiling diagnostic",
        "",
        "| student | robust | fallback | delay 1 | delay 2 | delay 5 | delay 10 | oracle | fallback causal recovery | delay1 - fallback | reducible |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for seed in protocol.STUDENT_SEEDS:
        row = payload["students"][str(seed)]
        means = row["arm_means"]
        gap = row["comparisons"]["delay1_minus_fallback"]
        lines.append(
            f"| {seed} | {means[protocol.ROBUST_ARM]:.1f} | "
            f"{means[protocol.FALLBACK_ARM]:.1f} | "
            f"{means[protocol.delay_arm(1)]:.1f} | "
            f"{means[protocol.delay_arm(2)]:.1f} | "
            f"{means[protocol.delay_arm(5)]:.1f} | "
            f"{means[protocol.delay_arm(10)]:.1f} | "
            f"{means[protocol.ORACLE_ARM]:.1f} | "
            f"{100.0 * row['fallback_causal_headroom_recovery']:.1f}% | "
            f"{gap['mean_delta']:+.1f} ({gap['event_seed_wins']}/5) | "
            f"{row['material_reducible_post_observation_gap']} |"
        )
    lines.extend([
        "",
        "## Decision",
        "",
        f"- Students with material reducible gaps: "
        f"`{payload['students_with_material_reducible_gap']}/3`.",
        f"- Authorize transient training: "
        f"`{payload['authorize_transient_training']}`.",
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
    reducible = sum(
        row["material_reducible_post_observation_gap"]
        for row in students.values()
    )
    zero_termination = all(
        row["zero_termination"] for row in students.values())
    authorized = (
        reducible >= MIN_STUDENTS_FOR_TRAINING and zero_termination)
    recommendation = (
        "A material post-observation gap remains for at least two student "
        "initializations. Register a new development-only transient student "
        "trained on stale/soft contexts while retaining the frozen fallback."
        if authorized else
        "The frozen fallback is already close to the one-transition causal "
        "ceiling, or the remaining gap is not stable across initializations. "
        "Do not train another student; freeze fallback as the deployable "
        "mechanism and report the zero-delay oracle as noncausal headroom."
    )
    payload = {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "development_only": True,
        "event_seeds": list(protocol.EVENT_SEEDS),
        "students": students,
        "minimum_reducible_fraction_of_robust":
            MIN_REDUCIBLE_FRACTION_OF_ROBUST,
        "minimum_students_for_training": MIN_STUDENTS_FOR_TRAINING,
        "students_with_material_reducible_gap": reducible,
        "zero_termination": zero_termination,
        "authorize_transient_training": authorized,
        "recommendation": recommendation,
    }
    protocol.ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    protocol.write_json_atomic(protocol.analysis_json(), payload)
    protocol.write_text_atomic(protocol.analysis_markdown(), _markdown(payload))
    print(
        "CAUSAL CEILING ANALYSIS COMPLETE: "
        f"authorize_training={authorized}",
        flush=True,
    )
    return payload


def main() -> None:
    run()


if __name__ == "__main__":
    main()
