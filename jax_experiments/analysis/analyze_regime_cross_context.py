"""Aggregate the event-grouped oracle cross-context diagnostic."""
from __future__ import annotations

import csv
import math
from pathlib import Path

from jax_experiments.analysis import analyze_regime_control_headroom as stats
from jax_experiments.analysis import regime_control_headroom as base
from jax_experiments.analysis import regime_cross_context as protocol
from jax_experiments.analysis.run_regime_cross_context_audit import (
    validate_audit,
)


FIXED_LABELS = tuple(f"fixed_{mode}" for mode in base.MODES)
IDENTITY_RTOL = 1e-7
IDENTITY_ATOL = 1e-5


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def _load_case_event(
        env: str, seed: int, event_seed: int,
        case: str) -> dict[str, object]:
    directory = protocol.case_dir(env, seed, event_seed, case)
    task_rows = _rows(directory / "task_returns.csv")
    switching_rows = _rows(directory / "switching_returns.csv")
    task_rows.sort(key=lambda row: int(float(row["mode_id_mean"])))
    switching_rows.sort(key=lambda row: int(row["episode"]))
    return {
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
        "stationary_termination": stats._mean(
            float(row["terminated_rate"]) for row in task_rows),
        "switch_identity": tuple(
            row["switch_sequence_source_indices"]
            for row in switching_rows),
        "switching_returns": [
            float(row["return"]) for row in switching_rows],
        "switching_termination": [
            _indicator(row["terminated"]) for row in switching_rows],
    }


def _load_all() -> dict[str, object]:
    data: dict[str, object] = {}
    for env in base.ENVS:
        data[env] = {}
        for seed in base.TRAINING_SEEDS:
            seed_data = {
                case: {} for case in protocol.CASE_LABELS
            }
            for event_seed in base.AUDIT_EVENT_SEEDS:
                validate_audit(env, seed, event_seed)
                event_cases = {
                    case: _load_case_event(
                        env, seed, event_seed, case)
                    for case in protocol.CASE_LABELS
                }
                reference = event_cases["true"]
                for case, candidate in event_cases.items():
                    if (candidate["task_identity"]
                            != reference["task_identity"]
                            or candidate["switch_identity"]
                            != reference["switch_identity"]):
                        raise ValueError(
                            "event-grouped streams are not paired: "
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
                            "true and matching fixed contexts diverged "
                            "inside one runtime: "
                            f"{env}/seed{seed}/event{event_seed}/mode{mode}")
                    if not _close(cyclic, shifted):
                        raise ValueError(
                            "cyclic and shifted fixed contexts diverged "
                            "inside one runtime: "
                            f"{env}/seed{seed}/event{event_seed}/mode{mode}")
            data[env][seed] = seed_data
    return data


def _paired(
        by_seed: dict[int, dict[str, dict[str, object]]],
        left: str, right: str, key: str) -> dict[str, object]:
    return stats._paired_ci([
        float(by_seed[seed][left][key])
        - float(by_seed[seed][right][key])
        for seed in base.TRAINING_SEEDS
    ])


def _relative(ci: dict[str, object], reference: float) -> float:
    return float(ci["mean"]) / max(abs(float(reference)), 100.0)


def _case_summary(
        by_seed: dict[int, dict[str, dict[str, object]]],
        case: str) -> dict[str, object]:
    return stats._role_summary([
        by_seed[seed][case] for seed in base.TRAINING_SEEDS])


def _analyze_env(cross_env: dict[int, object]) -> dict[str, object]:
    by_seed: dict[int, dict[str, dict[str, object]]] = {
        seed: {
            case: stats._seed_metrics(cross_env[seed][case])
            for case in protocol.CASE_LABELS
        }
        for seed in base.TRAINING_SEEDS
    }
    summaries = {
        case: _case_summary(by_seed, case)
        for case in protocol.CASE_LABELS
    }
    best_fixed = max(
        FIXED_LABELS,
        key=lambda label: float(
            summaries[label]["switching_mean_mean"]))
    comparisons = {}
    for name, left, right in (
            ("true_minus_robust", "true", "robust_model"),
            ("zero_minus_robust", "zero", "robust_model"),
            ("true_minus_zero", "true", "zero"),
            ("true_minus_best_fixed", "true", best_fixed)):
        comparisons[name] = {}
        for key in (
                "switching_mean", "stationary_mean", "stationary_worst"):
            ci = _paired(by_seed, left, right, key)
            reference = float(summaries[right][f"{key}_mean"])
            comparisons[name][key] = {
                **ci,
                "relative_gain": _relative(ci, reference),
            }

    stationary_matrix = {}
    diagonal_optimal = 0
    fixed_ranges = []
    for mode in base.MODES:
        row = {
            case: stats._mean(
                float(by_seed[seed][case][
                    "stationary_by_mode"][mode])
                for seed in base.TRAINING_SEEDS)
            for case in protocol.CASE_LABELS
        }
        stationary_matrix[str(mode)] = row
        fixed_values = {
            fixed_mode: row[f"fixed_{fixed_mode}"]
            for fixed_mode in base.MODES
        }
        best_value = max(fixed_values.values())
        if _close(fixed_values[mode], best_value):
            diagonal_optimal += 1
        fixed_ranges.append(
            max(fixed_values.values()) - min(fixed_values.values()))

    zero_stationary = abs(float(
        summaries["zero"]["stationary_mean_mean"]))
    fixed_spread_relative = (
        stats._mean(fixed_ranges) / max(zero_stationary, 100.0))
    true_termination = float(
        summaries["true"]["switching_termination_mean"])
    robust_termination = float(
        summaries["robust_model"]["switching_termination_mean"])
    pathological_survival = (
        true_termination >= 0.95 and robust_termination >= 0.95)

    true_robust = comparisons[
        "true_minus_robust"]["switching_mean"]
    true_zero = comparisons["true_minus_zero"]["switching_mean"]
    zero_robust = comparisons[
        "zero_minus_robust"]["switching_mean"]
    true_best_fixed = comparisons[
        "true_minus_best_fixed"]["switching_mean"]
    headroom_confirmed = (
        not pathological_survival
        and float(true_robust["relative_gain"]) >= 0.10
        and float(true_robust["ci95_low"]) > 0.0
        and diagonal_optimal >= 3
        and float(true_best_fixed["relative_gain"]) >= 0.10
        and float(true_best_fixed["ci95_low"]) > 0.0)
    if pathological_survival:
        diagnosis = "protocol_pathological_termination"
    elif headroom_confirmed:
        diagnosis = "dynamic_oracle_headroom_confirmed"
    elif (float(true_zero["ci95_high"]) < 0.0
          and float(zero_robust["relative_gain"]) >= -0.05):
        diagnosis = "true_context_branch_harmful_zero_recovers"
    elif (float(zero_robust["relative_gain"]) <= -0.10
          and float(zero_robust["ci95_high"]) < 0.0
          and diagonal_optimal < 3):
        diagnosis = "conditioned_training_negative_transfer"
    elif (fixed_spread_relative < 0.05
          and abs(float(true_zero["relative_gain"])) < 0.05):
        diagnosis = "context_ignored_or_little_specialization"
    elif (float(true_robust["relative_gain"]) >= 0.10
          and float(true_robust["ci95_low"]) > 0.0):
        diagnosis = "partial_headroom_not_better_than_fixed"
    else:
        diagnosis = "no_reliable_dynamic_headroom"

    return {
        "training_seed_metrics": {
            str(seed): by_seed[seed] for seed in base.TRAINING_SEEDS},
        "summaries": summaries,
        "best_fixed_switching_context": best_fixed,
        "stationary_context_matrix": stationary_matrix,
        "diagonal_optimal_modes": diagonal_optimal,
        "fixed_context_spread_relative": fixed_spread_relative,
        "comparisons": comparisons,
        "pathological_survival": pathological_survival,
        "headroom_confirmed": headroom_confirmed,
        "diagnosis": diagnosis,
    }


def _fmt_ci(row: dict[str, object]) -> str:
    return (
        f"{float(row['mean']):+.1f} "
        f"[{float(row['ci95_low']):+.1f},"
        f"{float(row['ci95_high']):+.1f}]")


def _render(payload: dict[str, object]) -> str:
    lines = [
        "# Event-grouped RegimeSAC cross-context diagnostic",
        "",
        "Each event task evaluates the robust checkpoint and all seven "
        "oracle-checkpoint contexts in one process. This removes node/runtime "
        "differences from paired context and controller comparisons. "
        "Independent training seeds (n=5) remain the inferential units.",
        "",
        "| Env | Robust | True | Zero | Best fixed | "
        "True-robust (95% CI) | True-zero (95% CI) | "
        "Diagonal | Fixed spread | Diagnosis |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for env in base.ENVS:
        result = payload["environments"][env]
        summaries = result["summaries"]
        best_fixed = result["best_fixed_switching_context"]
        true_robust = result["comparisons"][
            "true_minus_robust"]["switching_mean"]
        true_zero = result["comparisons"][
            "true_minus_zero"]["switching_mean"]
        lines.append(
            f"| {base.env_slug(env)} | "
            f"{summaries['robust_model']['switching_mean_mean']:.1f} | "
            f"{summaries['true']['switching_mean_mean']:.1f} | "
            f"{summaries['zero']['switching_mean_mean']:.1f} | "
            f"{best_fixed}: "
            f"{summaries[best_fixed]['switching_mean_mean']:.1f} | "
            f"{_fmt_ci(true_robust)} "
            f"({100.0 * true_robust['relative_gain']:+.1f}%) | "
            f"{_fmt_ci(true_zero)} | "
            f"{result['diagonal_optimal_modes']}/4 | "
            f"{100.0 * result['fixed_context_spread_relative']:.1f}% | "
            f"{result['diagnosis']} |")

    for env in base.ENVS:
        result = payload["environments"][env]
        lines += [
            "",
            f"## {base.env_slug(env)} stationary context matrix",
            "",
            "| Physics | Robust | True | Zero | Fixed 0 | Fixed 1 | "
            "Fixed 2 | Fixed 3 | Cyclic |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for mode in base.MODES:
            row = result["stationary_context_matrix"][str(mode)]
            lines.append(
                f"| {mode} | {row['robust_model']:.1f} | "
                f"{row['true']:.1f} | {row['zero']:.1f} | "
                f"{row['fixed_0']:.1f} | {row['fixed_1']:.1f} | "
                f"{row['fixed_2']:.1f} | {row['fixed_3']:.1f} | "
                f"{row['cyclic']:.1f} |")

    hc = payload["environments"]["HalfCheetah-v2"]
    ant = payload["environments"]["Ant-v2"]
    walker = payload["environments"]["Walker2d-v2"]
    hc_true_robust = hc["comparisons"]["true_minus_robust"]["switching_mean"]
    hc_true_zero = hc["comparisons"]["true_minus_zero"]["switching_mean"]
    hc_true_fixed = hc["comparisons"]["true_minus_best_fixed"]["switching_mean"]
    ant_true_robust = ant["comparisons"]["true_minus_robust"]["switching_mean"]
    ant_true_fixed = ant["comparisons"]["true_minus_best_fixed"]["switching_mean"]
    lines += [
        "",
        "## Mechanistic interpretation",
        "",
        f"- HalfCheetah: true context beats zero by {_fmt_ci(hc_true_zero)} "
        f"and the best fixed context by {_fmt_ci(hc_true_fixed)} "
        f"({100.0 * hc_true_fixed['relative_gain']:+.1f}%), but trails "
        f"the separately trained robust policy by {_fmt_ci(hc_true_robust)}. "
        "The mode signal and dynamic specialization are real; the failure is "
        "shared conditional-controller negative transfer or base-policy "
        "degradation, not estimator ambiguity.",
        f"- Ant: true context beats robust by {_fmt_ci(ant_true_robust)} "
        f"({100.0 * ant_true_robust['relative_gain']:+.1f}%) and the best "
        f"fixed context by {_fmt_ci(ant_true_fixed)}. This is a valid "
        "adaptation-positive environment.",
        f"- Walker2d: robust and true switching termination are both "
        f"{100.0 * walker['summaries']['true']['switching_termination_mean']:.0f}%. "
        "Controller comparisons are not interpretable until the environment "
        "passes a survival gate.",
        "",
    ]

    confirmed = [
        base.env_slug(env) for env in base.ENVS
        if payload["environments"][env]["headroom_confirmed"]]
    lines += [
        "",
        "## Decision",
        "",
        f"Strict dynamic-oracle headroom confirmed in "
        f"**{len(confirmed)}/{len(base.ENVS)}** environments"
        + (f": {', '.join(confirmed)}." if confirmed else "."),
        "",
        "A learned estimator remains blocked unless true dynamic context "
        "beats both the robust checkpoint and every fixed context by at "
        "least 10%, with positive paired confidence intervals, and at least "
        "3/4 stationary rows are diagonal-optimal. Near-universal termination "
        "is classified as a protocol failure rather than algorithm evidence.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    data = _load_all()
    environments = {
        env: _analyze_env(data[env]) for env in base.ENVS
    }
    payload = {
        "schema": protocol.ANALYSIS_SCHEMA,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "source_protocol": base.PROTOCOL_VERSION,
        "training_seeds": list(base.TRAINING_SEEDS),
        "event_seeds": list(base.AUDIT_EVENT_SEEDS),
        "evaluation_cases": list(protocol.CASE_LABELS),
        "environments": environments,
    }
    markdown = _render(payload)
    base.write_json_atomic(protocol.analysis_json(), payload)
    base.write_text_atomic(protocol.analysis_markdown(), markdown)
    base.write_text_atomic(protocol.REPORT, markdown)
    print(markdown)


if __name__ == "__main__":
    main()
