"""Aggregate the frozen five-event policy-distillation confirmation."""
from __future__ import annotations

import math
import statistics
from typing import Any

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_policy_distillation_confirmation as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_policy_distillation_audit as retrospective_audit,
)
from jax_experiments.analysis import (
    run_regime_polarity_policy_distillation_confirmation_audit as audit,
)


def _mean(values) -> float:
    values = [float(value) for value in values]
    if not values:
        raise ValueError("cannot average an empty sequence")
    return float(statistics.fmean(values))


def _event_results():
    rows = {}
    for event_seed in protocol.AUDIT_EVENT_SEEDS:
        audit.validate_audit(event_seed)
        result = protocol.read_json(
            protocol.audit_dir(
                protocol.TEACHER_GROUP,
                protocol.STUDENT_SEED,
                event_seed,
            ) / "results.json")
        rows[int(event_seed)] = {
            "switching": {
                row["arm"]: row for row in result["switching"]
            },
            "stationary": {
                (row["arm"], int(row["mode"])): row
                for row in result["stationary"]
            },
        }
    return rows


def _arm_values(events, arm: str):
    return {
        event_seed: [
            float(value)
            for value in event["switching"][arm]["returns"]
        ]
        for event_seed, event in events.items()
    }


def _population_values(events, arms):
    output = {}
    for event_seed, event in events.items():
        matrix = np.asarray([
            event["switching"][arm]["returns"] for arm in arms
        ], dtype=np.float64)
        output[event_seed] = list(np.mean(matrix, axis=0))
    return output


def _comparison(candidate, reference) -> dict[str, Any]:
    event_deltas = {}
    paired_deltas = []
    candidate_values = []
    reference_values = []
    for event_seed in protocol.AUDIT_EVENT_SEEDS:
        left = candidate[int(event_seed)]
        right = reference[int(event_seed)]
        if len(left) != len(right):
            raise ValueError("unpaired confirmation return vectors")
        deltas = [float(a) - float(b) for a, b in zip(left, right)]
        event_deltas[str(event_seed)] = _mean(deltas)
        paired_deltas.extend(deltas)
        candidate_values.extend(left)
        reference_values.extend(right)

    cluster_values = list(event_deltas.values())
    cluster_mean = _mean(cluster_values)
    cluster_sem = (
        float(statistics.stdev(cluster_values)) / math.sqrt(len(cluster_values))
        if len(cluster_values) > 1 else 0.0
    )
    half_width = protocol.CLUSTER_T_CRITICAL_95 * cluster_sem
    return {
        "candidate_mean": _mean(candidate_values),
        "reference_mean": _mean(reference_values),
        "mean_delta": _mean(paired_deltas),
        "event_seed_deltas": event_deltas,
        "event_seed_wins": sum(value > 0.0 for value in cluster_values),
        "episode_wins": sum(value > 0.0 for value in paired_deltas),
        "n_episodes": len(paired_deltas),
        "cluster_unit": "event_seed_mean",
        "cluster_count": len(cluster_values),
        "cluster_mean_delta": cluster_mean,
        "cluster_sem": cluster_sem,
        "cluster_95pct_t_interval": [
            cluster_mean - half_width,
            cluster_mean + half_width,
        ],
    }


def _stationary_summary(events, arm: str) -> dict[str, Any]:
    by_mode = {
        str(mode): _mean(
            value
            for event in events.values()
            for value in event["stationary"][(arm, mode)]["returns"]
        )
        for mode in protocol.MODES
    }
    return {
        "by_mode": by_mode,
        "worst_mode": min(by_mode.values()),
    }


def _terminated_rate(events, arm: str) -> float:
    return _mean(
        event["switching"][arm]["terminated_rate"]
        for event in events.values()
    )


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


def _markdown(payload: dict[str, Any]) -> str:
    population = payload["student_vs_robust_population"]
    fixed = payload["student_vs_fixed_strongest_robust"]
    recovery = payload["teacher_headroom_recovery"]
    recovery_text = (
        f"{100.0 * recovery:.1f}%"
        if recovery is not None and np.isfinite(recovery)
        else "n/a"
    )
    lines = [
        "# Frozen policy-distillation confirmation",
        "",
        f"The `{payload['frozen_teacher_group']}` student seed "
        f"`{payload['frozen_student_seed']}` was frozen before these five "
        "event seeds were evaluated.",
        "",
        "| event seed | student | robust population | fixed robust 719 | "
        "delta vs population | delta vs fixed |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for event_seed in protocol.AUDIT_EVENT_SEEDS:
        row = payload["event_means"][str(event_seed)]
        lines.append(
            f"| {event_seed} | {row['student']:.1f} | "
            f"{row['robust_population']:.1f} | {row['fixed_robust']:.1f} | "
            f"{row['student_minus_population']:+.1f} | "
            f"{row['student_minus_fixed']:+.1f} |"
        )
    fixed_ci = fixed["cluster_95pct_t_interval"]
    population_ci = population["cluster_95pct_t_interval"]
    lines.extend([
        "",
        "## Aggregate",
        "",
        f"- Student switching return: `{population['candidate_mean']:.1f}`.",
        f"- Robust-population return: `{population['reference_mean']:.1f}`; "
        f"delta `{population['mean_delta']:+.1f}`, event wins "
        f"`{population['event_seed_wins']}/5`, cluster 95% t interval "
        f"`[{population_ci[0]:+.1f}, {population_ci[1]:+.1f}]`.",
        f"- Frozen strongest robust return: `{fixed['reference_mean']:.1f}`; "
        f"delta `{fixed['mean_delta']:+.1f}`, event wins "
        f"`{fixed['event_seed_wins']}/5`, cluster 95% t interval "
        f"`[{fixed_ci[0]:+.1f}, {fixed_ci[1]:+.1f}]`.",
        f"- Learned-teacher headroom recovery: `{recovery_text}`.",
        f"- Student switching termination rate: "
        f"`{payload['student_switching_terminated_rate']:.3f}`.",
        "",
        "## Decision",
        "",
        f"Overall confirmation: `{'pass' if payload['promotion'] else 'fail'}`.",
        "",
        payload["recommendation"],
        "",
    ])
    return "\n".join(lines)


def run() -> dict[str, Any]:
    protocol.validate_frozen_candidate()
    events = _event_results()
    robust_arms = [
        retrospective_audit.robust_label(source_group, seed)
        for source_group, seed in protocol.controller_keys(
            protocol.TEACHER_GROUP)
    ]
    if protocol.FIXED_STRONGEST_ROBUST_ARM not in robust_arms:
        raise ValueError("frozen strongest robust comparator is absent")

    student = _arm_values(events, "student_learned")
    teacher = _arm_values(events, "teacher_learned_median")
    oracle = _arm_values(events, "teacher_oracle_median")
    population = _population_values(events, robust_arms)
    fixed = _arm_values(events, protocol.FIXED_STRONGEST_ROBUST_ARM)

    student_vs_population = _comparison(student, population)
    teacher_vs_population = _comparison(teacher, population)
    student_vs_fixed = _comparison(student, fixed)
    headroom = float(teacher_vs_population["mean_delta"])
    recovery = (
        float(student_vs_population["mean_delta"]) / headroom
        if headroom > 0.0 else None
    )
    terminated_rate = _terminated_rate(events, "student_learned")

    gates = {
        "population_event_wins": (
            student_vs_population["event_seed_wins"]
            >= protocol.MIN_EVENT_WINS_VS_POPULATION),
        "population_cluster_ci_positive": (
            student_vs_population["cluster_95pct_t_interval"][0] > 0.0),
        "fixed_robust_event_wins": (
            student_vs_fixed["event_seed_wins"]
            >= protocol.MIN_EVENT_WINS_VS_FIXED_ROBUST),
        "fixed_robust_cluster_ci_positive": (
            student_vs_fixed["cluster_95pct_t_interval"][0] > 0.0),
        "teacher_headroom_recovery": (
            recovery is not None
            and np.isfinite(recovery)
            and recovery >= protocol.MIN_HEADROOM_RECOVERY),
        "zero_termination": terminated_rate <= protocol.MAX_TERMINATED_RATE,
    }
    promotion = all(gates.values())

    event_means = {}
    for event_seed in protocol.AUDIT_EVENT_SEEDS:
        student_mean = _mean(student[event_seed])
        population_mean = _mean(population[event_seed])
        fixed_mean = _mean(fixed[event_seed])
        event_means[str(event_seed)] = {
            "student": student_mean,
            "robust_population": population_mean,
            "fixed_robust": fixed_mean,
            "student_minus_population": student_mean - population_mean,
            "student_minus_fixed": student_mean - fixed_mean,
        }

    if promotion:
        recommendation = (
            "The frozen student confirms both population-level benefit and "
            "superiority to the preregistered strongest robust controller on "
            "this benchmark. Next run compute-matched SAC, ESCP, and RE-SAC "
            "comparators under the identical frozen protocol.")
    elif (gates["population_event_wins"]
          and gates["population_cluster_ci_positive"]
          and gates["teacher_headroom_recovery"]
          and gates["zero_termination"]):
        recommendation = (
            "The frozen student confirms a robust-population improvement but "
            "not superiority to the preregistered strongest robust controller. "
            "Retain this as an ensemble-distillation result, not a stronger "
            "BAPR control claim.")
    else:
        recommendation = (
            "The frozen student does not confirm its retrospective promotion. "
            "Do not tune on these event seeds or launch cross-algorithm claims.")

    payload = {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "confirmatory": True,
        "env": protocol.ENV,
        "family": protocol.FAMILY,
        "frozen_teacher_group": protocol.TEACHER_GROUP,
        "frozen_student_seed": protocol.STUDENT_SEED,
        "frozen_student_manifest": protocol.FROZEN_STUDENT_MANIFEST_RECORD,
        "frozen_student_parameters": protocol.FROZEN_STUDENT_PARAMETER_RECORD,
        "frozen_selection_analysis": protocol.FROZEN_SELECTION_ANALYSIS_RECORD,
        "event_seeds": list(protocol.AUDIT_EVENT_SEEDS),
        "fixed_strongest_robust_arm": protocol.FIXED_STRONGEST_ROBUST_ARM,
        "thresholds": {
            "min_event_wins_vs_population": (
                protocol.MIN_EVENT_WINS_VS_POPULATION),
            "min_event_wins_vs_fixed_robust": (
                protocol.MIN_EVENT_WINS_VS_FIXED_ROBUST),
            "min_teacher_headroom_recovery": protocol.MIN_HEADROOM_RECOVERY,
            "max_terminated_rate": protocol.MAX_TERMINATED_RATE,
            "cluster_95pct_t_critical": protocol.CLUSTER_T_CRITICAL_95,
        },
        "event_means": event_means,
        "student_vs_robust_population": student_vs_population,
        "student_vs_fixed_strongest_robust": student_vs_fixed,
        "teacher_vs_robust_population": teacher_vs_population,
        "student_vs_learned_teacher": _comparison(student, teacher),
        "oracle_vs_robust_population": _comparison(oracle, population),
        "teacher_headroom_recovery": recovery,
        "student_switching_terminated_rate": terminated_rate,
        "student_stationary": _stationary_summary(
            events, "student_learned"),
        "fixed_robust_stationary": _stationary_summary(
            events, protocol.FIXED_STRONGEST_ROBUST_ARM),
        "student_posterior": _posterior_summary(
            events, "student_learned"),
        "gates": gates,
        "promotion": promotion,
        "recommendation": recommendation,
    }
    protocol.write_json_atomic(protocol.analysis_json(), payload)
    protocol.write_text_atomic(protocol.analysis_markdown(), _markdown(payload))
    print(
        "POLICY DISTILLATION CONFIRMATION ANALYSIS COMPLETE: "
        f"{protocol.ANALYSIS_ROOT}",
        flush=True,
    )
    return payload


def main() -> None:
    run()


if __name__ == "__main__":
    main()
