#!/usr/bin/env python3
"""Aggregate the paired shared-regime headroom screen and enforce its gate."""
from __future__ import annotations

import csv
import math
import os
import statistics
from pathlib import Path

from jax_experiments.analysis import bapr_regime_screen as protocol
from jax_experiments.analysis.run_bapr_regime_screen_audit import (
    validate_audit,
)


CONDITIONS = (
    "sac",
    "escp",
    "resac",
    "regime_robust",
    "oracle_dynamic",
    "fixed_context_0",
    "fixed_context_1",
    "fixed_context_2",
    "fixed_context_3",
)
COMPARATORS = (
    "regime_robust",
    "fixed_context_0",
    "fixed_context_1",
    "fixed_context_2",
    "fixed_context_3",
)
T_CRITICAL_95_DF4 = 2.7764451051977987


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _condition(role: str, mode_label: str) -> str:
    if role != "regime_oracle":
        return role
    if mode_label == "dynamic":
        return "oracle_dynamic"
    return f"fixed_context_{int(mode_label)}"


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return float(statistics.fmean(values))


def _indicator(value: str | bool | float | int) -> float:
    if isinstance(value, bool):
        return float(value)
    normalized = str(value).strip().lower()
    if normalized == "true":
        return 1.0
    if normalized == "false":
        return 0.0
    return float(value)


def _mean_sd(values: list[float]) -> tuple[float, float]:
    return _mean(values), (
        float(statistics.stdev(values)) if len(values) > 1 else 0.0)


def _paired_ci(values: list[float]) -> dict[str, float | int]:
    mean, sd = _mean_sd(values)
    half = T_CRITICAL_95_DF4 * sd / math.sqrt(len(values))
    return {
        "n": len(values),
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "wins": sum(value > 0.0 for value in values),
    }


def _physical_task_identity(row: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(
        (key, value) for key, value in row.items()
        if key.endswith(("_mean", "_min", "_max"))
        and not key.startswith(("return_", "steps_"))
    ))


def _load_event(event_seed: int) -> dict[str, object]:
    conditions: dict[str, dict[str, object]] = {}
    canonical_tasks = None
    canonical_sequences = None
    for role in protocol.AUDIT_ROLES:
        validate_audit(role, event_seed)
        directory = protocol.audit_dir(role, event_seed)
        task_rows = _rows(directory / "task_returns.csv")
        switching_rows = _rows(directory / "switching_returns.csv")
        mode_labels = sorted({
            str(row["eval_oracle_mode_id"]) for row in task_rows},
            key=lambda value: -1 if value == "dynamic" else int(value),
        )
        for mode_label in mode_labels:
            condition = _condition(role, mode_label)
            selected_tasks = [
                row for row in task_rows
                if str(row["eval_oracle_mode_id"]) == mode_label]
            selected_switching = [
                row for row in switching_rows
                if str(row["eval_oracle_mode_id"]) == mode_label]
            selected_tasks.sort(key=lambda row: int(float(row["mode_id_mean"])))
            selected_switching.sort(key=lambda row: int(row["episode"]))
            task_identity = tuple(
                _physical_task_identity(row) for row in selected_tasks)
            sequence_identity = tuple(
                row["switch_sequence_source_indices"]
                for row in selected_switching)
            if canonical_tasks is None:
                canonical_tasks = task_identity
                canonical_sequences = sequence_identity
            elif (task_identity != canonical_tasks
                    or sequence_identity != canonical_sequences):
                raise ValueError(
                    f"event stream is not paired for seed {event_seed}, "
                    f"condition {condition}")
            stationary_by_mode = {
                str(int(float(row["mode_id_mean"]))): float(row["return_mean"])
                for row in selected_tasks
            }
            conditions[condition] = {
                "stationary": _mean(list(stationary_by_mode.values())),
                "stationary_by_mode": stationary_by_mode,
                "stationary_termination_rate": _mean([
                    float(row["terminated_rate"]) for row in selected_tasks]),
                "switching": _mean([
                    float(row["return"]) for row in selected_switching]),
                "switching_termination_rate": _mean([
                    _indicator(row["terminated"]) for row in selected_switching]),
            }
    if set(conditions) != set(CONDITIONS):
        raise ValueError(
            f"event seed {event_seed} has conditions {sorted(conditions)}")
    return {"event_seed": event_seed, "conditions": conditions}


def _summary(events: list[dict[str, object]], condition: str) -> dict[str, object]:
    rows = [event["conditions"][condition] for event in events]
    stationary = [float(row["stationary"]) for row in rows]
    switching = [float(row["switching"]) for row in rows]
    s_mean, s_sd = _mean_sd(stationary)
    w_mean, w_sd = _mean_sd(switching)
    return {
        "stationary_mean": s_mean,
        "stationary_sd": s_sd,
        "switching_mean": w_mean,
        "switching_sd": w_sd,
        "stationary_termination_rate": _mean([
            float(row["stationary_termination_rate"]) for row in rows]),
        "switching_termination_rate": _mean([
            float(row["switching_termination_rate"]) for row in rows]),
    }


def _comparison(events: list[dict[str, object]], comparator: str) -> dict[str, object]:
    stationary = []
    switching = []
    for event in events:
        rows = event["conditions"]
        stationary.append(
            float(rows["oracle_dynamic"]["stationary"])
            - float(rows[comparator]["stationary"]))
        switching.append(
            float(rows["oracle_dynamic"]["switching"])
            - float(rows[comparator]["switching"]))
    return {
        "stationary": _paired_ci(stationary),
        "switching": _paired_ci(switching),
    }


def _fixed_context_matrix(events: list[dict[str, object]]) -> dict[str, object]:
    rows = {}
    diagonal_wins = 0
    for task_mode in range(4):
        values = {}
        for context_mode in range(4):
            values[str(context_mode)] = _mean([
                float(event["conditions"][
                    f"fixed_context_{context_mode}"][
                        "stationary_by_mode"][str(task_mode)])
                for event in events
            ])
        winner = max(values, key=values.get)
        diagonal = int(winner) == task_mode
        diagonal_wins += int(diagonal)
        rows[str(task_mode)] = {
            "returns": values,
            "winner": int(winner),
            "diagonal": diagonal,
        }
    return {"rows": rows, "diagonal_wins": diagonal_wins}


def _fmt(mean: float, sd: float) -> str:
    return f"{mean:.1f} +/- {sd:.1f}"


def _fmt_ci(row: dict[str, object]) -> str:
    return (
        f"{float(row['mean']):+.1f} "
        f"[{float(row['ci95_low']):+.1f}, "
        f"{float(row['ci95_high']):+.1f}] "
        f"({int(row['wins'])}/{int(row['n'])})")


def _render_markdown(payload: dict[str, object]) -> str:
    summaries = payload["summaries"]
    comparisons = payload["oracle_comparisons"]
    matrix = payload["fixed_context_matrix"]
    gate = payload["promotion_gate"]
    labels = {
        "sac": "SAC",
        "escp": "ESCP",
        "resac": "RE-SAC",
        "regime_robust": "shared robust",
        "oracle_dynamic": "dynamic true-mode oracle",
        **{f"fixed_context_{mode}": f"fixed context {mode}"
           for mode in range(4)},
    }
    lines = [
        "# Shared-regime BAPR headroom screen",
        "",
        "HalfCheetah `mean_variance`, training seed 0, five paired sealed "
        "event streams. Every controller received 5.6M environment steps "
        "and 350k gradient updates. Returns are event-stream means and "
        "standard deviations; paired differences include a t-based 95% CI.",
        "",
        "| Controller | Stationary | Switching | Switch termination |",
        "|---|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        row = summaries[condition]
        lines.append(
            f"| {labels[condition]} | "
            f"{_fmt(row['stationary_mean'], row['stationary_sd'])} | "
            f"{_fmt(row['switching_mean'], row['switching_sd'])} | "
            f"{row['switching_termination_rate']:.3f} |")
    lines += [
        "",
        "| Oracle comparison | Stationary delta | Switching delta |",
        "|---|---:|---:|",
    ]
    for comparator in COMPARATORS:
        row = comparisons[comparator]
        lines.append(
            f"| dynamic oracle - {labels[comparator]} | "
            f"{_fmt_ci(row['stationary'])} | "
            f"{_fmt_ci(row['switching'])} |")
    lines += [
        "",
        "| Physics mode | context 0 | context 1 | context 2 | context 3 | "
        "winner | diagonal |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for mode in range(4):
        row = matrix["rows"][str(mode)]
        values = row["returns"]
        lines.append(
            f"| {mode} | {values['0']:.1f} | {values['1']:.1f} | "
            f"{values['2']:.1f} | {values['3']:.1f} | {row['winner']} | "
            f"{'yes' if row['diagonal'] else 'no'} |")
    lines += [
        "",
        "## Promotion gate",
        "",
        f"- Oracle beats robust and every fixed context in stationary mean: "
        f"**{'PASS' if gate['stationary_beats_all'] else 'FAIL'}**",
        f"- Oracle beats robust and every fixed context in switching mean: "
        f"**{'PASS' if gate['switching_beats_all'] else 'FAIL'}**",
        f"- Matching fixed context is best in at least 3/4 stationary rows: "
        f"**{matrix['diagonal_wins']}/4 "
        f"({'PASS' if gate['diagonal_at_least_3_of_4'] else 'FAIL'})**",
        f"- Overall learned-estimator promotion: "
        f"**{'PASS' if gate['passed'] else 'FAIL'}**",
        "",
        ("The mechanism screen supports training the causal learned posterior "
         "with robust fallback."
         if gate["passed"] else
         "Do not train the learned posterior yet. The privileged controller "
         "itself has not established uniform adaptation headroom over robust "
         "and fixed-context alternatives."),
        "",
    ]
    return "\n".join(lines)


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def analyze() -> dict[str, object]:
    events = [_load_event(seed) for seed in protocol.AUDIT_EVENT_SEEDS]
    summaries = {
        condition: _summary(events, condition) for condition in CONDITIONS}
    comparisons = {
        comparator: _comparison(events, comparator)
        for comparator in COMPARATORS
    }
    matrix = _fixed_context_matrix(events)
    stationary_pass = all(
        comparisons[name]["stationary"]["mean"] > 0.0
        for name in COMPARATORS)
    switching_pass = all(
        comparisons[name]["switching"]["mean"] > 0.0
        for name in COMPARATORS)
    diagonal_pass = matrix["diagonal_wins"] >= 3
    payload = {
        "schema": "bapr.shared-regime-screen-analysis.v1",
        "status": "complete",
        "protocol": {
            "env": protocol.ENV,
            "family": protocol.FAMILY,
            "training_seed": protocol.TRAINING_SEED,
            "event_seeds": list(protocol.AUDIT_EVENT_SEEDS),
            "final_iteration": protocol.FINAL_ITERATION,
            "total_steps": protocol.FINAL_TOTAL_STEPS,
            "update_count": protocol.FINAL_UPDATE_COUNT,
        },
        "events": events,
        "summaries": summaries,
        "oracle_comparisons": comparisons,
        "fixed_context_matrix": matrix,
        "promotion_gate": {
            "stationary_beats_all": stationary_pass,
            "switching_beats_all": switching_pass,
            "diagonal_at_least_3_of_4": diagonal_pass,
            "passed": stationary_pass and switching_pass and diagonal_pass,
        },
    }
    protocol.write_json_atomic(protocol.analysis_json(), payload)
    _write_text_atomic(protocol.analysis_markdown(), _render_markdown(payload))
    print(_render_markdown(payload), flush=True)
    return payload


def main() -> None:
    analyze()


if __name__ == "__main__":
    main()
