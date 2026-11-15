"""Aggregate the frozen final-stack BAPR mechanism audit."""
from __future__ import annotations

import math
import statistics
from typing import Any

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_final_mechanism_audit_v1 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_final_mechanism_audit_v1 as auditor,
)


T_CRITICAL_95 = 2.7764451051977987


def _mean(values) -> float:
    values = [float(value) for value in values]
    if not values:
        raise ValueError("cannot average an empty sequence")
    return float(statistics.fmean(values))


def _paired(candidate: dict[int, float], reference: dict[int, float]) \
        -> dict[str, Any]:
    if set(candidate) != set(protocol.EVENT_SEEDS) or set(reference) != set(
            protocol.EVENT_SEEDS):
        raise ValueError("mechanism comparison has incomplete event clusters")
    deltas = {
        int(seed): float(candidate[seed]) - float(reference[seed])
        for seed in protocol.EVENT_SEEDS
    }
    values = list(deltas.values())
    mean = _mean(values)
    sd = statistics.stdev(values)
    half = T_CRITICAL_95 * sd / math.sqrt(len(values))
    return {
        "candidate_mean": _mean(candidate.values()),
        "reference_mean": _mean(reference.values()),
        "mean_delta": mean,
        "event_deltas": {str(seed): value for seed, value in deltas.items()},
        "event_wins": sum(value > 0.0 for value in values),
        "clustered_95pct_interval": [mean - half, mean + half],
    }


def _positive(row: dict[str, Any], wins: int = 4) -> bool:
    return bool(
        row["mean_delta"] > 0.0
        and row["event_wins"] >= int(wins)
        and row["clustered_95pct_interval"][0] > 0.0
    )


def _load() -> dict[int, dict[int, dict[str, Any]]]:
    output = {}
    for student_seed in protocol.STUDENT_SEEDS:
        output[int(student_seed)] = {}
        for event_seed in protocol.EVENT_SEEDS:
            auditor.validate_manifest(student_seed, event_seed)
            payload = protocol.read_json(
                protocol.audit_dir(student_seed, event_seed) / "results.json")
            output[int(student_seed)][int(event_seed)] = {
                "switching": {
                    row["arm"]: row for row in payload["switching"]},
                "stationary": {
                    (row["arm"], int(row["mode"])): row
                    for row in payload["stationary"]
                },
            }
    return output


def _arm_values(events, arm: str) -> dict[int, float]:
    return {
        int(event_seed): float(event["switching"][arm]["return_mean"])
        for event_seed, event in events.items()
    }


def _summary_metric(events, arm: str, key: str) -> float:
    return _mean(event["switching"][arm][key] for event in events.values())


def _posterior(events, arm: str) -> dict[str, float]:
    rows = [
        event["switching"][arm]["posterior_metrics"]
        for event in events.values()
    ]
    return {
        key: _mean(row[key] for row in rows)
        for key in (
            "mode_accuracy", "brier_score",
            "median_switch_delay", "p90_switch_delay",
        )
    }


def _stationary_matrix(events) -> tuple[dict[str, Any], int]:
    matrix = {}
    diagonal = 0
    for mode in protocol.MODES:
        row = {}
        for arm in (protocol.TRUE_ARM, protocol.ZERO_ARM,
                    protocol.UNIFORM_ARM, *protocol.FIXED_ARMS):
            row[arm] = _mean(
                value
                for event in events.values()
                for value in event["stationary"][(arm, mode)]["returns"]
            )
        matching = row[f"student_fixed_{mode}"]
        if not np.isclose(
                row[protocol.TRUE_ARM], matching, atol=1e-5, rtol=1e-7):
            raise ValueError(
                f"true/fixed stationary mismatch for mode {mode}")
        fixed = {arm: row[arm] for arm in protocol.FIXED_ARMS}
        if np.isclose(matching, max(fixed.values()), atol=1e-5, rtol=1e-7):
            diagonal += 1
        matrix[str(mode)] = row
    return matrix, diagonal


def _transient(events, arm: str) -> dict[str, float]:
    return {
        f"{start}:{end}": _mean(
            event["switching"][arm]["post_switch_reward_mean"][
                f"{start}:{end}"]
            for event in events.values()
        )
        for start, end in protocol.TRANSIENT_BINS
    }


def _analyze_student(events) -> dict[str, Any]:
    values = {
        arm: _arm_values(events, arm) for arm in protocol.SWITCHING_ARMS
    }
    best_fixed = {
        event_seed: max(values[arm][event_seed]
                        for arm in protocol.FIXED_ARMS)
        for event_seed in protocol.EVENT_SEEDS
    }
    comparisons = {
        "true_minus_robust": _paired(
            values[protocol.TRUE_ARM], values[protocol.ROBUST_ARM]),
        "true_minus_best_fixed_envelope": _paired(
            values[protocol.TRUE_ARM], best_fixed),
        "true_minus_zero": _paired(
            values[protocol.TRUE_ARM], values[protocol.ZERO_ARM]),
        "true_minus_uniform": _paired(
            values[protocol.TRUE_ARM], values[protocol.UNIFORM_ARM]),
        "true_minus_cyclic_wrong": _paired(
            values[protocol.TRUE_ARM], values[protocol.CYCLIC_ARM]),
        "true_minus_shuffled_wrong": _paired(
            values[protocol.TRUE_ARM], values[protocol.SHUFFLED_ARM]),
        "learned_minus_robust": _paired(
            values[protocol.LEARNED_ARM], values[protocol.ROBUST_ARM]),
        "fallback_minus_robust": _paired(
            values[protocol.FALLBACK_ARM], values[protocol.ROBUST_ARM]),
        "fallback_minus_learned": _paired(
            values[protocol.FALLBACK_ARM], values[protocol.LEARNED_ARM]),
        "true_minus_fallback": _paired(
            values[protocol.TRUE_ARM], values[protocol.FALLBACK_ARM]),
    }
    for delay in protocol.DELAY_STEPS:
        arm = f"student_true_delay_{delay}"
        comparisons[f"delay_{delay}_minus_robust"] = _paired(
            values[arm], values[protocol.ROBUST_ARM])
        comparisons[f"delay_{delay}_minus_fallback"] = _paired(
            values[arm], values[protocol.FALLBACK_ARM])
    stationary, diagonal = _stationary_matrix(events)
    context_specialized = bool(
        diagonal >= 3
        and _positive(comparisons["true_minus_best_fixed_envelope"])
        and _positive(comparisons["true_minus_zero"])
    )
    causal_value = _positive(comparisons["fallback_minus_robust"])
    fallback_value = _positive(
        comparisons["fallback_minus_learned"], wins=3)
    return {
        "arm_means": {
            arm: _mean(rows.values()) for arm, rows in values.items()
        },
        "comparisons": comparisons,
        "stationary_matrix": stationary,
        "stationary_diagonal_optima": diagonal,
        "posterior": _posterior(events, protocol.LEARNED_ARM),
        "fallback_action_fraction": _summary_metric(
            events, protocol.FALLBACK_ARM, "fallback_action_fraction"),
        "termination_rates": {
            arm: _summary_metric(events, arm, "terminated_rate")
            for arm in protocol.SWITCHING_ARMS
        },
        "post_switch_reward": {
            arm: _transient(events, arm)
            for arm in (
                protocol.ROBUST_ARM,
                protocol.LEARNED_ARM,
                protocol.FALLBACK_ARM,
                protocol.TRUE_ARM,
                *protocol.DELAY_ARMS,
            )
        },
        "gates": {
            "context_specialized": context_specialized,
            "causal_adaptation_value": causal_value,
            "fallback_incremental_value": fallback_value,
            "delay_5_beats_robust": _positive(
                comparisons["delay_5_minus_robust"]),
            "delay_10_beats_robust": _positive(
                comparisons["delay_10_minus_robust"]),
        },
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Final BAPR mechanism audit",
        "",
        "This is a frozen, checkpoint-only post-confirmation diagnostic. "
        "It cannot select another model, estimator, fallback threshold, or "
        "environment configuration.",
        "",
        "| student | robust | learned | fallback | true | context | causal "
        "| fallback value | delay 5 | delay 10 |",
        "|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for seed in protocol.STUDENT_SEEDS:
        row = payload["students"][str(seed)]
        means = row["arm_means"]
        gates = row["gates"]
        lines.append(
            f"| {seed} | {means[protocol.ROBUST_ARM]:.1f} | "
            f"{means[protocol.LEARNED_ARM]:.1f} | "
            f"{means[protocol.FALLBACK_ARM]:.1f} | "
            f"{means[protocol.TRUE_ARM]:.1f} | "
            f"{gates['context_specialized']} | "
            f"{gates['causal_adaptation_value']} | "
            f"{gates['fallback_incremental_value']} | "
            f"{gates['delay_5_beats_robust']} | "
            f"{gates['delay_10_beats_robust']} |"
        )
    lines.extend([
        "",
        "## Frozen decision",
        "",
        f"- Context-specialized students: "
        f"`{payload['gate_counts']['context_specialized']}/5`.",
        f"- Students with causal fallback value over robust 719: "
        f"`{payload['gate_counts']['causal_adaptation_value']}/5`.",
        f"- Students with incremental fallback value over learned-only: "
        f"`{payload['gate_counts']['fallback_incremental_value']}/5`.",
        f"- Overall mechanism pass: `{payload['mechanism_pass']}`.",
        "",
        payload["interpretation"],
        "",
    ])
    return "\n".join(lines)


def run() -> dict[str, Any]:
    protocol.validate_registration()
    data = _load()
    students = {
        str(seed): _analyze_student(data[int(seed)])
        for seed in protocol.STUDENT_SEEDS
    }
    counts = {
        "context_specialized": sum(
            row["gates"]["context_specialized"]
            for row in students.values()),
        "causal_adaptation_value": sum(
            row["gates"]["causal_adaptation_value"]
            for row in students.values()),
        "fallback_incremental_value": sum(
            row["gates"]["fallback_incremental_value"]
            for row in students.values()),
        "delay_5_beats_robust": sum(
            row["gates"]["delay_5_beats_robust"]
            for row in students.values()),
        "delay_10_beats_robust": sum(
            row["gates"]["delay_10_beats_robust"]
            for row in students.values()),
    }
    no_termination = all(
        rate == 0.0
        for row in students.values()
        for rate in row["termination_rates"].values()
    )
    mechanism_pass = bool(
        counts["context_specialized"] >= 4
        and counts["causal_adaptation_value"] >= 4
        and no_termination
    )
    if mechanism_pass:
        interpretation = (
            "The final result is supported by repeatable dynamic context "
            "specialization and a causal deployable path, rather than by a "
            "single unconditional compressed policy. The fallback ablation "
            "is reported separately and is not required to be beneficial "
            "for every initialization."
        )
    else:
        interpretation = (
            "The final return does not establish a repeatable causal "
            "adaptation mechanism across frozen student initializations. "
            "Do not launch a larger confirmatory claim experiment from this "
            "checkpoint family."
        )
    payload = {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "registration": protocol.registration_record(),
        "confirmatory": False,
        "selection_forbidden": True,
        "student_seeds": list(protocol.STUDENT_SEEDS),
        "event_seeds": list(protocol.EVENT_SEEDS),
        "students": students,
        "gate_counts": counts,
        "all_termination_rates_zero": no_termination,
        "mechanism_pass": mechanism_pass,
        "interpretation": interpretation,
    }
    protocol.write_json_atomic(protocol.analysis_json(), payload)
    protocol.write_text_atomic(protocol.analysis_markdown(), _markdown(payload))
    marker = protocol.pass_marker()
    if mechanism_pass:
        protocol.write_text_atomic(
            marker,
            f"{protocol.PROTOCOL_VERSION}\nmechanism_pass=true\n",
        )
    elif marker.exists():
        marker.unlink()
    print(
        f"FINAL MECHANISM ANALYSIS COMPLETE: pass={mechanism_pass}",
        flush=True,
    )
    return payload


def main() -> None:
    run()


if __name__ == "__main__":
    main()
