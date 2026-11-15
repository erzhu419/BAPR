"""Aggregate the fixed-delta multi-seed regime-adapter confirmation."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from jax_experiments.analysis import regime_adapter_confirmation as protocol
from jax_experiments.analysis.run_regime_adapter_confirmation_audit import (
    CASES,
    validate_audit,
)


T_CRITICAL_95 = {
    3: 3.182446305284263,
    4: 2.7764451051977987,
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean_ci(values) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if len(values) < 2:
        low = high = mean
    else:
        critical = T_CRITICAL_95[len(values) - 1]
        half = critical * float(np.std(values, ddof=1)) / math.sqrt(len(values))
        low, high = mean - half, mean + half
    return {
        "mean": mean,
        "ci95_low": low,
        "ci95_high": high,
        "n_training_seeds": int(len(values)),
    }


def _event_value(directory: Path, case: str, metric: str) -> float:
    if metric == "stationary_return":
        rows = _read_rows(directory / case / "task_returns.csv")
        return float(np.mean([float(row["return_mean"]) for row in rows]))
    rows = _read_rows(directory / case / "switching_returns.csv")
    if metric == "switching_return":
        return float(np.mean([float(row["return"]) for row in rows]))
    if metric == "switching_termination":
        return float(np.mean([
            row["terminated"].lower() in ("true", "1") for row in rows]))
    raise ValueError(f"unknown metric {metric!r}")


def _seed_value(seed: int, case: str, metric: str) -> float:
    return float(np.mean([
        _event_value(protocol.audit_dir(seed, event_seed), case, metric)
        for event_seed in protocol.EVENT_SEEDS
    ]))


def _seed_mode_values(seed: int, case: str) -> np.ndarray:
    by_mode = {mode: [] for mode in protocol.MODES}
    for event_seed in protocol.EVENT_SEEDS:
        rows = _read_rows(
            protocol.audit_dir(seed, event_seed)
            / case / "task_returns.csv")
        for row in rows:
            by_mode[int(row["task_index"])].append(float(row["return_mean"]))
    return np.asarray([
        np.mean(by_mode[mode]) for mode in protocol.MODES
    ], dtype=np.float64)


def _relative_gain(candidate, baseline) -> float:
    return float(
        np.mean(np.asarray(candidate) - np.asarray(baseline))
        / max(abs(float(np.mean(baseline))), 1e-8))


def _paired(candidate, baseline) -> dict[str, float | int]:
    candidate = np.asarray(candidate, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    delta = candidate - baseline
    return {
        **_mean_ci(delta),
        "relative_gain": _relative_gain(candidate, baseline),
        "wins": int(np.sum(delta > 0)),
    }


def analyze() -> dict:
    for seed in protocol.TRAINING_SEEDS:
        for event_seed in protocol.EVENT_SEEDS:
            validate_audit(seed, event_seed)

    case_names = tuple(case.label for case in CASES)
    metrics = (
        "stationary_return", "switching_return", "switching_termination")
    seed_values = {
        case: {
            metric: {
                seed: _seed_value(seed, case, metric)
                for seed in protocol.TRAINING_SEEDS
            }
            for metric in metrics
        }
        for case in case_names
    }

    def values(case: str, metric: str, seeds) -> np.ndarray:
        return np.asarray([
            seed_values[case][metric][seed] for seed in seeds
        ], dtype=np.float64)

    fixed_cases = tuple(f"fixed_adapter_{mode}" for mode in protocol.MODES)
    seed_rows = []
    for seed in protocol.TRAINING_SEEDS:
        fixed_stationary = {
            case: seed_values[case]["stationary_return"][seed]
            for case in fixed_cases
        }
        fixed_switching = {
            case: seed_values[case]["switching_return"][seed]
            for case in fixed_cases
        }
        best_stationary = max(fixed_stationary, key=fixed_stationary.get)
        best_switching = max(fixed_switching, key=fixed_switching.get)
        robust_stationary = seed_values[
            "robust_continue"]["stationary_return"][seed]
        identity_stationary = seed_values[
            "identity_adapter"]["stationary_return"][seed]
        robust_switching = seed_values[
            "robust_continue"]["switching_return"][seed]
        identity_switching = seed_values[
            "identity_adapter"]["switching_return"][seed]
        seed_rows.append({
            "training_seed": seed,
            "selection_role": (
                "development" if seed == protocol.DEVELOPMENT_SEED
                else "holdout"),
            "robust_stationary": robust_stationary,
            "identity_stationary": identity_stationary,
            "stationary_relative_gain": (
                (identity_stationary - robust_stationary)
                / max(abs(robust_stationary), 1e-8)),
            "best_fixed_stationary_case": best_stationary,
            "best_fixed_stationary": fixed_stationary[best_stationary],
            "identity_minus_best_fixed_stationary": (
                identity_stationary - fixed_stationary[best_stationary]),
            "robust_switching": robust_switching,
            "identity_switching": identity_switching,
            "switching_relative_gain": (
                (identity_switching - robust_switching)
                / max(abs(robust_switching), 1e-8)),
            "best_fixed_switching_case": best_switching,
            "best_fixed_switching": fixed_switching[best_switching],
            "identity_minus_best_fixed_switching": (
                identity_switching - fixed_switching[best_switching]),
            "robust_termination": seed_values[
                "robust_continue"]["switching_termination"][seed],
            "identity_termination": seed_values[
                "identity_adapter"]["switching_termination"][seed],
        })

    holdout = protocol.HOLDOUT_SEEDS
    holdout_robust_stationary = values(
        "robust_continue", "stationary_return", holdout)
    holdout_identity_stationary = values(
        "identity_adapter", "stationary_return", holdout)
    holdout_robust_switching = values(
        "robust_continue", "switching_return", holdout)
    holdout_identity_switching = values(
        "identity_adapter", "switching_return", holdout)
    holdout_robust_termination = values(
        "robust_continue", "switching_termination", holdout)
    holdout_identity_termination = values(
        "identity_adapter", "switching_termination", holdout)
    holdout_base_stationary = values(
        "frozen_base", "stationary_return", holdout)
    holdout_base_switching = values(
        "frozen_base", "switching_return", holdout)
    holdout_best_fixed_stationary = np.asarray([
        max(
            seed_values[case]["stationary_return"][seed]
            for case in fixed_cases)
        for seed in holdout
    ])
    holdout_best_fixed_switching = np.asarray([
        max(
            seed_values[case]["switching_return"][seed]
            for case in fixed_cases)
        for seed in holdout
    ])

    stationary_vs_robust = _paired(
        holdout_identity_stationary, holdout_robust_stationary)
    switching_vs_robust = _paired(
        holdout_identity_switching, holdout_robust_switching)
    stationary_vs_best_fixed = _paired(
        holdout_identity_stationary, holdout_best_fixed_stationary)
    switching_vs_best_fixed = _paired(
        holdout_identity_switching, holdout_best_fixed_switching)
    stationary_vs_frozen_base = _paired(
        holdout_identity_stationary, holdout_base_stationary)
    switching_vs_frozen_base = _paired(
        holdout_identity_switching, holdout_base_switching)
    robust_stationary_vs_frozen_base = _paired(
        holdout_robust_stationary, holdout_base_stationary)
    robust_switching_vs_frozen_base = _paired(
        holdout_robust_switching, holdout_base_switching)
    termination_delta = (
        holdout_identity_termination - holdout_robust_termination)

    specialization_rows = []
    for seed in protocol.TRAINING_SEEDS:
        matrix = np.stack([
            _seed_mode_values(seed, case) for case in fixed_cases
        ], axis=1)
        diagonal = np.diag(matrix)
        best = np.max(matrix, axis=1)
        diagonal_optimal = np.isclose(
            diagonal, best, rtol=0.0, atol=1e-6)
        specialization_rows.append({
            "training_seed": seed,
            "selection_role": (
                "development" if seed == protocol.DEVELOPMENT_SEED
                else "holdout"),
            "diagonal_optimal_modes": int(np.sum(diagonal_optimal)),
            "diagonal_minus_best_by_mode": (
                diagonal - best).astype(float).tolist(),
            "fixed_controller_matrix": matrix.astype(float).tolist(),
        })
    holdout_specialization = [
        row for row in specialization_rows
        if row["selection_role"] == "holdout"
    ]
    holdout_diagonal_wins = sum(
        row["diagonal_optimal_modes"] for row in holdout_specialization)

    gate = {
        "stationary_gain_at_least_5pct": (
            stationary_vs_robust["relative_gain"] >= 0.05),
        "switching_gain_at_least_10pct": (
            switching_vs_robust["relative_gain"] >= 0.10),
        "stationary_gain_ci_above_zero": (
            stationary_vs_robust["ci95_low"] > 0),
        "switching_gain_ci_above_zero": (
            switching_vs_robust["ci95_low"] > 0),
        "identity_beats_robust_all_four_holdout_seeds": (
            stationary_vs_robust["wins"] == len(holdout)
            and switching_vs_robust["wins"] == len(holdout)),
        "beats_best_fixed_stationary_with_positive_ci": (
            stationary_vs_best_fixed["wins"] == len(holdout)
            and stationary_vs_best_fixed["ci95_low"] > 0),
        "beats_best_fixed_switching_with_positive_ci": (
            switching_vs_best_fixed["wins"] == len(holdout)
            and switching_vs_best_fixed["ci95_low"] > 0),
        "termination_noninferior": bool(np.max(termination_delta) <= 0.05),
    }
    gate["pass"] = all(gate.values())

    pooled = {}
    for case in case_names:
        pooled[case] = {
            metric: _mean_ci(values(case, metric, protocol.TRAINING_SEEDS))
            for metric in metrics
        }

    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "residual_delta": protocol.DELTA,
        "controller_map": list(protocol.CONTROLLER_MAP),
        "event_seeds": list(protocol.EVENT_SEEDS),
        "inferential_scope": (
            "Five event streams are averaged within each policy-training "
            "seed. The primary gate uses only untouched training seeds "
            "16/24/32/40; seed 8 is development-only."),
        "seed_rows": seed_rows,
        "holdout": {
            "training_seeds": list(holdout),
            "identity_minus_robust_stationary": stationary_vs_robust,
            "identity_minus_robust_switching": switching_vs_robust,
            "identity_minus_best_fixed_stationary": stationary_vs_best_fixed,
            "identity_minus_best_fixed_switching": switching_vs_best_fixed,
            "termination_delta": _mean_ci(termination_delta),
            "promotion_gate": gate,
        },
        "mechanism_diagnostic": {
            "identity_minus_frozen_base_stationary": (
                stationary_vs_frozen_base),
            "identity_minus_frozen_base_switching": (
                switching_vs_frozen_base),
            "robust_minus_frozen_base_stationary": (
                robust_stationary_vs_frozen_base),
            "robust_minus_frozen_base_switching": (
                robust_switching_vs_frozen_base),
            "identity_share_of_robust_continuation_gain": {
                "stationary": (
                    stationary_vs_frozen_base["mean"]
                    / robust_stationary_vs_frozen_base["mean"]),
                "switching": (
                    switching_vs_frozen_base["mean"]
                    / robust_switching_vs_frozen_base["mean"]),
            },
            "specialization_rows": specialization_rows,
            "holdout_diagonal_optimal_modes": holdout_diagonal_wins,
            "holdout_total_modes": (
                len(holdout) * len(protocol.MODES)),
            "interpretation": (
                "Adapters improve the frozen pre-fork controller and usually "
                "specialize correctly, but four separately optimized heads "
                "recover less value than one shared robust continuation. "
                "The bottleneck is controller sample/optimization efficiency, "
                "not absent mode-conditioned control value."),
        },
        "pooled_five_seed_descriptive": pooled,
        "decision": (
            "train_causal_router_against_confirmed_adapter_bank"
            if gate["pass"] else
            "stop_before_learned_router_and_reassess_controller_variance"),
    }


def _markdown(payload: dict) -> str:
    lines = [
        "# Regime-adapter multi-seed confirmation",
        "",
        "The primary inference excludes development seed 8. Each row first "
        "averages five paired event streams within one independently trained "
        "policy seed.",
        "",
        "| Seed | Role | Robust stat | Identity stat | Gain | Robust switch | "
        "Identity switch | Gain | Identity-best fixed switch |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["seed_rows"]:
        lines.append(
            f"| {row['training_seed']} | {row['selection_role']} | "
            f"{row['robust_stationary']:.1f} | "
            f"{row['identity_stationary']:.1f} | "
            f"{100 * row['stationary_relative_gain']:+.1f}% | "
            f"{row['robust_switching']:.1f} | "
            f"{row['identity_switching']:.1f} | "
            f"{100 * row['switching_relative_gain']:+.1f}% | "
            f"{row['identity_minus_best_fixed_switching']:+.1f} |")
    holdout = payload["holdout"]
    stat = holdout["identity_minus_robust_stationary"]
    switch = holdout["identity_minus_robust_switching"]
    fixed = holdout["identity_minus_best_fixed_switching"]
    gate = holdout["promotion_gate"]
    diagnostic = payload["mechanism_diagnostic"]
    base_stat = diagnostic["identity_minus_frozen_base_stationary"]
    base_switch = diagnostic["identity_minus_frozen_base_switching"]
    robust_base_stat = diagnostic["robust_minus_frozen_base_stationary"]
    robust_base_switch = diagnostic["robust_minus_frozen_base_switching"]
    lines.extend([
        "",
        "## Untouched-seed inference",
        "",
        f"- Stationary: {stat['mean']:+.1f}, 95% CI "
        f"[{stat['ci95_low']:+.1f}, {stat['ci95_high']:+.1f}], "
        f"relative {100 * stat['relative_gain']:+.1f}%, "
        f"wins {stat['wins']}/4.",
        f"- Switching: {switch['mean']:+.1f}, 95% CI "
        f"[{switch['ci95_low']:+.1f}, {switch['ci95_high']:+.1f}], "
        f"relative {100 * switch['relative_gain']:+.1f}%, "
        f"wins {switch['wins']}/4.",
        f"- Identity minus per-seed best fixed switching: {fixed['mean']:+.1f}, "
        f"95% CI [{fixed['ci95_low']:+.1f}, {fixed['ci95_high']:+.1f}], "
        f"wins {fixed['wins']}/4.",
        f"- Promotion gate: {'PASS' if gate['pass'] else 'FAIL'}.",
        "",
        "## Mechanism diagnosis",
        "",
        f"- Identity versus frozen base, stationary: {base_stat['mean']:+.1f}, "
        f"95% CI [{base_stat['ci95_low']:+.1f}, "
        f"{base_stat['ci95_high']:+.1f}], wins {base_stat['wins']}/4.",
        f"- Identity versus frozen base, switching: {base_switch['mean']:+.1f}, "
        f"95% CI [{base_switch['ci95_low']:+.1f}, "
        f"{base_switch['ci95_high']:+.1f}], wins {base_switch['wins']}/4.",
        f"- Robust continuation versus frozen base, stationary: "
        f"{robust_base_stat['mean']:+.1f}; switching: "
        f"{robust_base_switch['mean']:+.1f}.",
        f"- Correct-mode adapter is stationary-optimal in "
        f"{diagnostic['holdout_diagonal_optimal_modes']}/"
        f"{diagnostic['holdout_total_modes']} holdout mode rows.",
        f"- Adapter bank recovers "
        f"{100 * diagnostic['identity_share_of_robust_continuation_gain']['stationary']:.1f}% "
        f"of the shared robust stationary gain and "
        f"{100 * diagnostic['identity_share_of_robust_continuation_gain']['switching']:.1f}% "
        f"of its switching gain.",
        "",
        diagnostic["interpretation"],
        "",
        f"Decision: `{payload['decision']}`.",
        "",
        "A pass authorizes training a causal router on separate training and "
        "calibration streams. It is not yet a learned-adaptation result.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    payload = analyze()
    protocol.ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    protocol.write_json_atomic(protocol.analysis_json(), payload)
    protocol.analysis_markdown().write_text(
        _markdown(payload), encoding="utf-8")
    print(_markdown(payload))


if __name__ == "__main__":
    main()
