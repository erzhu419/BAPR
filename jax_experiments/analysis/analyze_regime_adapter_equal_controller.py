"""Aggregate the full-budget independent-adapter upper-bound diagnostic."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from jax_experiments.analysis import regime_adapter_equal_controller as protocol
from jax_experiments.analysis.run_regime_adapter_equal_controller_audit import (
    CASES,
    validate_audit,
)


T_CRITICAL_95 = {4: 2.7764451051977987}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean_ci(values) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if len(values) < 2:
        low = high = mean
    else:
        critical = T_CRITICAL_95[len(values)]
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
            mode = int(float(row["mode_id_mean"]))
            by_mode[mode].append(float(row["return_mean"]))
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

    def values(case: str, metric: str) -> np.ndarray:
        return np.asarray([
            seed_values[case][metric][seed]
            for seed in protocol.TRAINING_SEEDS
        ], dtype=np.float64)

    fixed_cases = tuple(f"fixed_adapter_{mode}" for mode in protocol.MODES)
    robust_stationary = values("robust_continue", "stationary_return")
    identity_stationary = values("identity_adapter", "stationary_return")
    base_stationary = values("frozen_base", "stationary_return")
    robust_switching = values("robust_continue", "switching_return")
    identity_switching = values("identity_adapter", "switching_return")
    base_switching = values("frozen_base", "switching_return")
    robust_termination = values(
        "robust_continue", "switching_termination")
    identity_termination = values(
        "identity_adapter", "switching_termination")
    best_fixed_stationary = np.asarray([
        max(
            seed_values[case]["stationary_return"][seed]
            for case in fixed_cases)
        for seed in protocol.TRAINING_SEEDS
    ], dtype=np.float64)
    best_fixed_switching = np.asarray([
        max(
            seed_values[case]["switching_return"][seed]
            for case in fixed_cases)
        for seed in protocol.TRAINING_SEEDS
    ], dtype=np.float64)

    stationary_vs_robust = _paired(
        identity_stationary, robust_stationary)
    switching_vs_robust = _paired(
        identity_switching, robust_switching)
    stationary_vs_best_fixed = _paired(
        identity_stationary, best_fixed_stationary)
    switching_vs_best_fixed = _paired(
        identity_switching, best_fixed_switching)
    stationary_vs_base = _paired(identity_stationary, base_stationary)
    switching_vs_base = _paired(identity_switching, base_switching)
    robust_stationary_vs_base = _paired(
        robust_stationary, base_stationary)
    robust_switching_vs_base = _paired(
        robust_switching, base_switching)
    termination_delta = identity_termination - robust_termination

    seed_rows = []
    specialization_rows = []
    for seed_index, seed in enumerate(protocol.TRAINING_SEEDS):
        matrix = np.stack([
            _seed_mode_values(seed, case) for case in fixed_cases
        ], axis=1)
        diagonal = np.diag(matrix)
        best = np.max(matrix, axis=1)
        diagonal_optimal = np.isclose(
            diagonal, best, rtol=0.0, atol=1e-6)
        specialization_rows.append({
            "training_seed": seed,
            "diagonal_optimal_modes": int(np.sum(diagonal_optimal)),
            "diagonal_minus_best_by_mode": (
                diagonal - best).astype(float).tolist(),
            "fixed_controller_matrix": matrix.astype(float).tolist(),
        })
        seed_rows.append({
            "training_seed": seed,
            "robust_stationary": float(robust_stationary[seed_index]),
            "identity_stationary": float(identity_stationary[seed_index]),
            "stationary_relative_gain": float(
                (identity_stationary[seed_index]
                 - robust_stationary[seed_index])
                / max(abs(robust_stationary[seed_index]), 1e-8)),
            "best_fixed_stationary": float(
                best_fixed_stationary[seed_index]),
            "identity_minus_best_fixed_stationary": float(
                identity_stationary[seed_index]
                - best_fixed_stationary[seed_index]),
            "robust_switching": float(robust_switching[seed_index]),
            "identity_switching": float(identity_switching[seed_index]),
            "switching_relative_gain": float(
                (identity_switching[seed_index]
                 - robust_switching[seed_index])
                / max(abs(robust_switching[seed_index]), 1e-8)),
            "best_fixed_switching": float(
                best_fixed_switching[seed_index]),
            "identity_minus_best_fixed_switching": float(
                identity_switching[seed_index]
                - best_fixed_switching[seed_index]),
            "robust_termination": float(robust_termination[seed_index]),
            "identity_termination": float(identity_termination[seed_index]),
        })

    gate = {
        "stationary_gain_at_least_5pct": (
            stationary_vs_robust["relative_gain"] >= 0.05),
        "switching_gain_at_least_10pct": (
            switching_vs_robust["relative_gain"] >= 0.10),
        "stationary_gain_ci_above_zero": (
            stationary_vs_robust["ci95_low"] > 0),
        "switching_gain_ci_above_zero": (
            switching_vs_robust["ci95_low"] > 0),
        "identity_beats_robust_all_four_seeds": (
            stationary_vs_robust["wins"] == len(protocol.TRAINING_SEEDS)
            and switching_vs_robust["wins"]
            == len(protocol.TRAINING_SEEDS)),
        "beats_best_fixed_stationary_with_positive_ci": (
            stationary_vs_best_fixed["wins"]
            == len(protocol.TRAINING_SEEDS)
            and stationary_vs_best_fixed["ci95_low"] > 0),
        "beats_best_fixed_switching_with_positive_ci": (
            switching_vs_best_fixed["wins"]
            == len(protocol.TRAINING_SEEDS)
            and switching_vs_best_fixed["ci95_low"] > 0),
        "termination_noninferior": bool(np.max(termination_delta) <= 0.05),
    }
    gate["pass"] = all(gate.values())

    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "diagnostic_only": True,
        "compute_matched": False,
        "residual_delta": protocol.DELTA,
        "training_seeds": list(protocol.TRAINING_SEEDS),
        "event_seeds": list(protocol.EVENT_SEEDS),
        "budget": {
            "shared_pretrain_steps": protocol.SOURCE_TOTAL_STEPS,
            "per_controller_post_fork_steps": (
                protocol.PER_CONTROLLER_POST_FORK_STEPS),
            "robust_total_steps": protocol.FINAL_TOTAL_STEPS,
            "adapter_bank_aggregate_total_steps": (
                protocol.BANK_AGGREGATE_TOTAL_STEPS),
        },
        "inferential_scope": (
            "Five event streams are averaged within each independently "
            "trained policy seed; n=4 training seeds is the inferential unit."
        ),
        "seed_rows": seed_rows,
        "paired_results": {
            "identity_minus_robust_stationary": stationary_vs_robust,
            "identity_minus_robust_switching": switching_vs_robust,
            "identity_minus_best_fixed_stationary": (
                stationary_vs_best_fixed),
            "identity_minus_best_fixed_switching": switching_vs_best_fixed,
            "identity_minus_frozen_base_stationary": stationary_vs_base,
            "identity_minus_frozen_base_switching": switching_vs_base,
            "robust_minus_frozen_base_stationary": (
                robust_stationary_vs_base),
            "robust_minus_frozen_base_switching": (
                robust_switching_vs_base),
            "termination_delta": _mean_ci(termination_delta),
        },
        "specialization_rows": specialization_rows,
        "diagonal_optimal_modes": int(sum(
            row["diagonal_optimal_modes"]
            for row in specialization_rows)),
        "total_mode_rows": (
            len(protocol.TRAINING_SEEDS) * len(protocol.MODES)),
        "decision_gate": gate,
        "decision": (
            "replace_independent_critics_with_shared_all_mode_backbone"
            if gate["pass"] else
            "reject_independent_adapter_optimization"),
        "interpretation": (
            "This deliberately spends four times the robust post-fork data. "
            "A pass only establishes controller specialization headroom and "
            "authorizes a compute-efficient shared-backbone redesign; it is "
            "not a publishable BAPR comparison."
        ),
    }


def _markdown(payload: dict) -> str:
    paired = payload["paired_results"]
    stat = paired["identity_minus_robust_stationary"]
    switch = paired["identity_minus_robust_switching"]
    fixed_stat = paired["identity_minus_best_fixed_stationary"]
    fixed_switch = paired["identity_minus_best_fixed_switching"]
    lines = [
        "# Equal-per-controller adapter upper bound",
        "",
        "**Diagnostic only.** Each of four fixed-mode adapters receives the "
        "same 2.8M post-fork transitions as the single robust continuation. "
        "The bank therefore uses four times the post-fork data and is not a "
        "paper comparison.",
        "",
        "| Seed | Robust stat | Identity stat | Gain | Robust switch | "
        "Identity switch | Gain | Identity-best fixed switch |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["seed_rows"]:
        lines.append(
            f"| {row['training_seed']} | {row['robust_stationary']:.1f} | "
            f"{row['identity_stationary']:.1f} | "
            f"{100 * row['stationary_relative_gain']:+.1f}% | "
            f"{row['robust_switching']:.1f} | "
            f"{row['identity_switching']:.1f} | "
            f"{100 * row['switching_relative_gain']:+.1f}% | "
            f"{row['identity_minus_best_fixed_switching']:+.1f} |")
    lines.extend([
        "",
        "## Paired inference",
        "",
        f"- Identity minus robust stationary: {stat['mean']:+.1f}, "
        f"95% CI [{stat['ci95_low']:+.1f}, {stat['ci95_high']:+.1f}], "
        f"relative {100 * stat['relative_gain']:+.1f}%, "
        f"wins {stat['wins']}/4.",
        f"- Identity minus robust switching: {switch['mean']:+.1f}, "
        f"95% CI [{switch['ci95_low']:+.1f}, "
        f"{switch['ci95_high']:+.1f}], "
        f"relative {100 * switch['relative_gain']:+.1f}%, "
        f"wins {switch['wins']}/4.",
        f"- Identity minus best fixed stationary: {fixed_stat['mean']:+.1f}, "
        f"95% CI [{fixed_stat['ci95_low']:+.1f}, "
        f"{fixed_stat['ci95_high']:+.1f}].",
        f"- Identity minus best fixed switching: "
        f"{fixed_switch['mean']:+.1f}, 95% CI "
        f"[{fixed_switch['ci95_low']:+.1f}, "
        f"{fixed_switch['ci95_high']:+.1f}].",
        f"- Correct-mode controller is stationary-optimal in "
        f"{payload['diagonal_optimal_modes']}/"
        f"{payload['total_mode_rows']} rows.",
        f"- Decision gate: "
        f"{'PASS' if payload['decision_gate']['pass'] else 'FAIL'}.",
        "",
        payload["interpretation"],
        "",
        f"Decision: `{payload['decision']}`.",
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
