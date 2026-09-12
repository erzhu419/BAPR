"""Analyze context causality and delayed-oracle reachability."""
from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

from jax_experiments.analysis import regime_polarity_context_delay as protocol
from jax_experiments.analysis import regime_polarity_headroom as base
from jax_experiments.analysis.run_regime_polarity_context_delay_audit import (
    validate_audit,
)


T_CRITICAL_95_DF2 = 4.302652729911275
IDENTITY_RTOL = 1e-7
IDENTITY_ATOL = 1e-5
MIN_ORACLE_ROBUST_GAIN = 0.10
MIN_DELAY_10_RETENTION = 0.70
MIN_DELAY_25_RETENTION = 0.50


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(values) -> float:
    values = list(values)
    if not values:
        raise ValueError("cannot average an empty sequence")
    return float(statistics.fmean(values))


def _sd(values) -> float:
    values = list(values)
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _indicator(value: object) -> float:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return 1.0
    if normalized == "false":
        return 0.0
    return float(value)


def _close(left: float, right: float) -> bool:
    return math.isclose(
        float(left), float(right),
        rel_tol=IDENTITY_RTOL, abs_tol=IDENTITY_ATOL)


def _paired_ci(values: list[float]) -> dict[str, float | int | list[float]]:
    if len(values) != len(base.TRAINING_SEEDS):
        raise ValueError(
            f"paired inference requires {len(base.TRAINING_SEEDS)} "
            f"training seeds, got {len(values)}")
    mean = _mean(values)
    sd = _sd(values)
    half = T_CRITICAL_95_DF2 * sd / math.sqrt(len(values))
    return {
        "n_training_seeds": len(values),
        "values": values,
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "wins": sum(value > 0.0 for value in values),
    }


def _load_case_event(
        env: str, seed: int, event_seed: int, case: str,
) -> dict[str, object]:
    case_spec = protocol.evaluation_case(case)
    directory = protocol.case_dir(env, seed, event_seed, case)
    switching_rows = _rows(directory / "switching_returns.csv")
    switching_rows.sort(key=lambda row: int(row["episode"]))
    trace_rows = _rows(directory / "switching_trace.csv")
    trace_rows.sort(
        key=lambda row: (int(row["episode"]), int(row["step"])))
    payload: dict[str, object] = {
        "switch_identity": tuple(
            row["switch_sequence_source_indices"]
            for row in switching_rows),
        "trace_physics_identity": tuple(
            (
                int(row["episode"]),
                int(row["step"]),
                int(row["physics_action_task_id"]),
            )
            for row in trace_rows
        ),
        "switching_returns": [
            float(row["return"]) for row in switching_rows],
        "switching_termination": [
            _indicator(row["terminated"]) for row in switching_rows],
    }
    if case_spec.stationary:
        task_rows = _rows(directory / "task_returns.csv")
        task_rows.sort(key=lambda row: int(float(row["mode_id_mean"])))
        payload.update({
            "task_identity": tuple(
                tuple(sorted(
                    (key, value) for key, value in row.items()
                    if key.endswith(("_mean", "_min", "_max"))
                    and not key.startswith(("return_", "steps_"))))
                for row in task_rows),
            "mode_returns": {
                int(float(row["mode_id_mean"])): float(row["return_mean"])
                for row in task_rows
            },
            "stationary_termination": _mean(
                float(row["terminated_rate"]) for row in task_rows),
        })
    return payload


def _load_all() -> dict[str, object]:
    data: dict[str, object] = {}
    for env in protocol.ENVS:
        data[env] = {}
        for seed in base.TRAINING_SEEDS:
            seed_data = {
                case: {} for case in protocol.CASE_LABELS
            }
            for event_seed in base.AUDIT_EVENT_SEEDS:
                validate_audit(env, seed, event_seed)
                event_cases = {
                    case: _load_case_event(env, seed, event_seed, case)
                    for case in protocol.CASE_LABELS
                }
                reference = event_cases["true"]
                for case, candidate in event_cases.items():
                    if (candidate["switch_identity"]
                            != reference["switch_identity"]
                            or candidate["trace_physics_identity"]
                            != reference["trace_physics_identity"]):
                        raise ValueError(
                            "event streams are not paired: "
                            f"{env}/seed{seed}/event{event_seed}/{case}")
                    if (protocol.evaluation_case(case).stationary
                            and candidate["task_identity"]
                            != reference["task_identity"]):
                        raise ValueError(
                            "stationary streams are not paired: "
                            f"{env}/seed{seed}/event{event_seed}/{case}")
                    seed_data[case][event_seed] = candidate
                for mode in base.MODES:
                    dynamic = reference["mode_returns"][mode]
                    matching = event_cases[
                        f"fixed_{mode}"]["mode_returns"][mode]
                    cyclic = event_cases[
                        "cyclic"]["mode_returns"][mode]
                    shifted = event_cases[
                        f"fixed_{(mode + 1) % len(base.MODES)}"
                    ]["mode_returns"][mode]
                    if not _close(dynamic, matching):
                        raise ValueError(
                            "true and matching fixed contexts diverged: "
                            f"{env}/seed{seed}/event{event_seed}/mode{mode}")
                    if not _close(cyclic, shifted):
                        raise ValueError(
                            "cyclic and shifted fixed contexts diverged: "
                            f"{env}/seed{seed}/event{event_seed}/mode{mode}")
            data[env][seed] = seed_data
    return data


def _seed_metrics(events: dict[int, dict[str, object]]) -> dict[str, object]:
    switching = [
        value
        for event in events.values()
        for value in event["switching_returns"]
    ]
    switching_termination = [
        value
        for event in events.values()
        for value in event["switching_termination"]
    ]
    metrics: dict[str, object] = {
        "switching_mean": _mean(switching),
        "switching_termination": _mean(switching_termination),
    }
    first = next(iter(events.values()))
    if "mode_returns" in first:
        mode_returns = {
            mode: _mean(
                event["mode_returns"][mode] for event in events.values())
            for mode in base.MODES
        }
        metrics.update({
            "stationary_by_mode": mode_returns,
            "stationary_mean": _mean(mode_returns.values()),
            "stationary_worst": min(mode_returns.values()),
            "stationary_termination": _mean(
                event["stationary_termination"]
                for event in events.values()),
        })
    return metrics


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    keys = sorted({
        key
        for row in rows
        for key, value in row.items()
        if key != "stationary_by_mode" and isinstance(value, (int, float))
    })
    payload: dict[str, object] = {}
    for key in keys:
        values = [float(row[key]) for row in rows]
        payload[f"{key}_mean"] = _mean(values)
        payload[f"{key}_sd"] = _sd(values)
    if all("stationary_by_mode" in row for row in rows):
        payload["stationary_by_mode"] = {
            str(mode): _mean(
                float(row["stationary_by_mode"][mode]) for row in rows)
            for mode in base.MODES
        }
    return payload


def _relative(delta: float, reference: float) -> float:
    return float(delta) / max(abs(float(reference)), 100.0)


def _analyze_env(env_data: dict[int, object]) -> dict[str, object]:
    by_seed = {
        seed: {
            case: _seed_metrics(env_data[seed][case])
            for case in protocol.CASE_LABELS
        }
        for seed in base.TRAINING_SEEDS
    }
    summaries = {
        case: _summary([
            by_seed[seed][case] for seed in base.TRAINING_SEEDS])
        for case in protocol.CASE_LABELS
    }
    best_fixed_by_seed = {
        seed: max(
            protocol.FIXED_LABELS,
            key=lambda case: float(
                by_seed[seed][case]["switching_mean"]))
        for seed in base.TRAINING_SEEDS
    }
    best_fixed_values = {
        seed: float(
            by_seed[seed][best_fixed_by_seed[seed]]["switching_mean"])
        for seed in base.TRAINING_SEEDS
    }

    comparison_specs = {
        "true_minus_robust": (
            [
                float(by_seed[seed]["true"]["switching_mean"])
                - float(by_seed[seed]["robust_model"]["switching_mean"])
                for seed in base.TRAINING_SEEDS
            ],
            _mean(
                by_seed[seed]["robust_model"]["switching_mean"]
                for seed in base.TRAINING_SEEDS),
        ),
        "true_minus_zero": (
            [
                float(by_seed[seed]["true"]["switching_mean"])
                - float(by_seed[seed]["zero"]["switching_mean"])
                for seed in base.TRAINING_SEEDS
            ],
            _mean(
                by_seed[seed]["zero"]["switching_mean"]
                for seed in base.TRAINING_SEEDS),
        ),
        "true_minus_best_fixed_envelope": (
            [
                float(by_seed[seed]["true"]["switching_mean"])
                - best_fixed_values[seed]
                for seed in base.TRAINING_SEEDS
            ],
            _mean(best_fixed_values.values()),
        ),
        "true_minus_shuffled": (
            [
                float(by_seed[seed]["true"]["switching_mean"])
                - float(by_seed[seed]["shuffled"]["switching_mean"])
                for seed in base.TRAINING_SEEDS
            ],
            _mean(
                by_seed[seed]["shuffled"]["switching_mean"]
                for seed in base.TRAINING_SEEDS),
        ),
    }
    comparisons = {}
    for name, (values, reference) in comparison_specs.items():
        ci = _paired_ci(values)
        comparisons[name] = {
            **ci,
            "relative_gain": _relative(float(ci["mean"]), reference),
        }

    stationary_matrix = {}
    diagonal_optimal = 0
    for mode in base.MODES:
        row = {
            case: _mean(
                float(by_seed[seed][case][
                    "stationary_by_mode"][mode])
                for seed in base.TRAINING_SEEDS)
            for case in protocol.FULL_CASE_LABELS
        }
        stationary_matrix[str(mode)] = row
        fixed_values = {
            fixed_mode: row[f"fixed_{fixed_mode}"]
            for fixed_mode in base.MODES
        }
        if _close(fixed_values[mode], max(fixed_values.values())):
            diagonal_optimal += 1

    true_robust_headroom = {
        seed: (
            float(by_seed[seed]["true"]["switching_mean"])
            - float(by_seed[seed]["robust_model"]["switching_mean"]))
        for seed in base.TRAINING_SEEDS
    }
    delay_analysis = {}
    for delay in protocol.DELAY_STEPS:
        label = f"delayed_{delay}"
        delay_minus_robust = [
            float(by_seed[seed][label]["switching_mean"])
            - float(by_seed[seed]["robust_model"]["switching_mean"])
            for seed in base.TRAINING_SEEDS
        ]
        retention = [
            (
                delay_minus_robust[index]
                / true_robust_headroom[seed]
                if true_robust_headroom[seed] > 0.0
                else math.nan
            )
            for index, seed in enumerate(base.TRAINING_SEEDS)
        ]
        finite_retention = [
            value for value in retention if math.isfinite(value)]
        delay_analysis[str(delay)] = {
            "summary": summaries[label],
            "minus_robust": _paired_ci(delay_minus_robust),
            "headroom_retention_by_seed": retention,
            "headroom_retention_mean": (
                _mean(finite_retention)
                if finite_retention else math.nan),
            "beats_robust_all_seeds": all(
                value > 0.0 for value in delay_minus_robust),
        }

    true_robust = comparisons["true_minus_robust"]
    true_fixed = comparisons["true_minus_best_fixed_envelope"]
    delay_10 = delay_analysis["10"]
    delay_25 = delay_analysis["25"]
    survival_ok = (
        float(summaries["true"]["switching_termination_mean"]) < 0.95
        and float(delay_25["summary"][
            "switching_termination_mean"]) < 0.95)
    context_causality_ok = (
        int(true_fixed["wins"]) == len(base.TRAINING_SEEDS)
        and diagonal_optimal >= 3)
    robust_headroom_ok = (
        int(true_robust["wins"]) == len(base.TRAINING_SEEDS)
        and float(true_robust["relative_gain"])
        >= MIN_ORACLE_ROBUST_GAIN)
    delay_reachability_ok = (
        bool(delay_10["beats_robust_all_seeds"])
        and float(delay_10["headroom_retention_mean"])
        >= MIN_DELAY_10_RETENTION
        and bool(delay_25["beats_robust_all_seeds"])
        and float(delay_25["headroom_retention_mean"])
        >= MIN_DELAY_25_RETENTION)
    continuation_gate = (
        survival_ok
        and context_causality_ok
        and robust_headroom_ok
        and delay_reachability_ok)

    return {
        "training_seed_metrics": {
            str(seed): by_seed[seed] for seed in base.TRAINING_SEEDS},
        "summaries": summaries,
        "best_fixed_by_training_seed": {
            str(seed): best_fixed_by_seed[seed]
            for seed in base.TRAINING_SEEDS},
        "stationary_context_matrix": stationary_matrix,
        "diagonal_optimal_modes": diagonal_optimal,
        "comparisons": comparisons,
        "delayed_oracle": delay_analysis,
        "gate_checks": {
            "survival_ok": survival_ok,
            "context_causality_ok": context_causality_ok,
            "robust_headroom_ok": robust_headroom_ok,
            "delay_reachability_ok": delay_reachability_ok,
        },
        "fresh_five_seed_confirmation_candidate": continuation_gate,
    }


def _fmt_ci(row: dict[str, object]) -> str:
    return (
        f"{float(row['mean']):+.1f} "
        f"[{float(row['ci95_low']):+.1f}, "
        f"{float(row['ci95_high']):+.1f}]")


def _render(payload: dict[str, object]) -> str:
    lines = [
        "# Polarity context-causality and delay audit",
        "",
        "This is a checkpoint-only paired audit on the existing "
        "actuator-polarity controllers. Each training seed is evaluated on "
        "three identical event streams under true, zero, fixed, cyclic, "
        "shuffled, and delayed contexts. The three existing training seeds "
        "are exploratory units; they are not extended post hoc.",
        "",
        "| Env | Robust | True | Per-seed fixed envelope | "
        "True-robust | True-fixed envelope | Diagonal | "
        "Delay 10 retained | Delay 25 retained | Candidate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for env in protocol.ENVS:
        result = payload["environments"][env]
        summaries = result["summaries"]
        fixed_mean = _mean(
            float(result["training_seed_metrics"][str(seed)][
                result["best_fixed_by_training_seed"][str(seed)]
            ]["switching_mean"])
            for seed in base.TRAINING_SEEDS
        )
        true_robust = result["comparisons"]["true_minus_robust"]
        true_fixed = result["comparisons"][
            "true_minus_best_fixed_envelope"]
        delay_10 = result["delayed_oracle"]["10"]
        delay_25 = result["delayed_oracle"]["25"]
        lines.append(
            f"| {base.env_slug(env)} | "
            f"{summaries['robust_model']['switching_mean_mean']:.1f} | "
            f"{summaries['true']['switching_mean_mean']:.1f} | "
            f"{fixed_mean:.1f} | "
            f"{_fmt_ci(true_robust)} "
            f"({100.0 * true_robust['relative_gain']:+.1f}%) | "
            f"{_fmt_ci(true_fixed)} | "
            f"{result['diagonal_optimal_modes']}/4 | "
            f"{100.0 * delay_10['headroom_retention_mean']:.1f}% | "
            f"{100.0 * delay_25['headroom_retention_mean']:.1f}% | "
            f"{'PASS' if result['fresh_five_seed_confirmation_candidate'] else 'FAIL'} |"
        )

    lines += [
        "",
        "## Delayed-oracle ladder",
        "",
        "| Env | Delay | Return | Delta vs robust | "
        "Headroom retained | Beats robust all 3 seeds |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for env in protocol.ENVS:
        result = payload["environments"][env]
        for delay in protocol.DELAY_STEPS:
            row = result["delayed_oracle"][str(delay)]
            lines.append(
                f"| {base.env_slug(env)} | {delay} | "
                f"{row['summary']['switching_mean_mean']:.1f} | "
                f"{_fmt_ci(row['minus_robust'])} | "
                f"{100.0 * row['headroom_retention_mean']:.1f}% | "
                f"{row['beats_robust_all_seeds']} |"
            )

    candidates = [
        base.env_slug(env) for env in protocol.ENVS
        if payload["environments"][env][
            "fresh_five_seed_confirmation_candidate"]
    ]
    lines += [
        "",
        "## Decision",
        "",
        "The continuation gate requires all of the following within an "
        "environment: true context beats the robust controller on every "
        "exploratory seed with at least 10% mean gain; true context beats "
        "the per-seed best fixed-context envelope on every seed; at least "
        "3/4 stationary rows are diagonal-optimal; delay 10 and delay 25 "
        "retain at least 70% and 50% of instantaneous-oracle headroom while "
        "beating robust on every seed; and termination is not pathological.",
        "",
        "Fresh five-seed confirmation candidates: "
        + (", ".join(candidates) if candidates else "none")
        + ".",
        "",
        "A learned estimator remains blocked at this stage. Passing this "
        "audit authorizes a new preregistered robust-versus-oracle five-seed "
        "confirmation, not estimator training.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    data = _load_all()
    environments = {
        env: _analyze_env(data[env]) for env in protocol.ENVS
    }
    payload = {
        "schema": protocol.ANALYSIS_SCHEMA,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "source_protocol": base.PROTOCOL_VERSION,
        "training_seeds": list(base.TRAINING_SEEDS),
        "event_seeds": list(base.AUDIT_EVENT_SEEDS),
        "evaluation_cases": list(protocol.CASE_LABELS),
        "thresholds": {
            "minimum_oracle_robust_relative_gain": (
                MIN_ORACLE_ROBUST_GAIN),
            "minimum_delay_10_headroom_retention": (
                MIN_DELAY_10_RETENTION),
            "minimum_delay_25_headroom_retention": (
                MIN_DELAY_25_RETENTION),
        },
        "environments": environments,
    }
    markdown = _render(payload)
    base.write_json_atomic(protocol.analysis_json(), payload)
    base.write_text_atomic(protocol.analysis_markdown(), markdown)
    base.write_text_atomic(protocol.REPORT, markdown)
    print(markdown)


if __name__ == "__main__":
    main()
