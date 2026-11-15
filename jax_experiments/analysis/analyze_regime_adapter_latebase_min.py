"""Aggregate the late-base min-target adapter diagnostic."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from jax_experiments.analysis import regime_adapter_latebase_min as protocol
from jax_experiments.analysis.run_regime_adapter_latebase_min_audit import (
    CASES,
    validate_audit,
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def _relative_gain(candidate: float, baseline: float) -> float:
    return float(
        (float(candidate) - float(baseline))
        / max(abs(float(baseline)), 1e-8))


def analyze() -> dict:
    for seed in protocol.TRAINING_SEEDS:
        for event_seed in protocol.EVENT_SEEDS:
            validate_audit(seed, event_seed)

    case_names = tuple(case.label for case in CASES)
    metrics = (
        "stationary_return", "switching_return", "switching_termination")
    values = {
        seed: {
            case: {
                metric: _seed_value(seed, case, metric)
                for metric in metrics
            }
            for case in case_names
        }
        for seed in protocol.TRAINING_SEEDS
    }

    fixed_cases = tuple(f"fixed_adapter_{mode}" for mode in protocol.MODES)
    seed_rows = []
    specialization_rows = []
    for seed in protocol.TRAINING_SEEDS:
        robust = values[seed]["late_robust"]
        frozen = values[seed]["frozen_base"]
        identity = values[seed]["identity_adapter"]
        fixed_stationary = [
            values[seed][case]["stationary_return"]
            for case in fixed_cases
        ]
        fixed_switching = [
            values[seed][case]["switching_return"]
            for case in fixed_cases
        ]
        matrix = np.stack([
            _seed_mode_values(seed, case) for case in fixed_cases
        ], axis=1)
        diagonal = np.diag(matrix)
        row_best = np.max(matrix, axis=1)
        diagonal_optimal = np.isclose(
            diagonal, row_best, rtol=0.0, atol=1e-6)
        base_scale = max(
            abs(robust["stationary_return"]),
            abs(robust["switching_return"]), 1.0)
        copy_error = max(
            abs(frozen["stationary_return"]
                - robust["stationary_return"]),
            abs(frozen["switching_return"]
                - robust["switching_return"])) / base_scale
        seed_rows.append({
            "training_seed": seed,
            "late_robust_stationary": robust["stationary_return"],
            "identity_stationary": identity["stationary_return"],
            "stationary_relative_gain": _relative_gain(
                identity["stationary_return"],
                robust["stationary_return"]),
            "best_fixed_stationary": max(fixed_stationary),
            "identity_minus_best_fixed_stationary": (
                identity["stationary_return"]
                - max(fixed_stationary)),
            "late_robust_switching": robust["switching_return"],
            "identity_switching": identity["switching_return"],
            "switching_relative_gain": _relative_gain(
                identity["switching_return"],
                robust["switching_return"]),
            "best_fixed_switching": max(fixed_switching),
            "identity_minus_best_fixed_switching": (
                identity["switching_return"]
                - max(fixed_switching)),
            "late_robust_termination": robust["switching_termination"],
            "identity_termination": identity["switching_termination"],
            "frozen_base_copy_relative_error": copy_error,
            "diagonal_optimal_modes": int(np.sum(diagonal_optimal)),
        })
        specialization_rows.append({
            "training_seed": seed,
            "diagonal_optimal_modes": int(np.sum(diagonal_optimal)),
            "diagonal_minus_best_by_mode": (
                diagonal - row_best).astype(float).tolist(),
            "fixed_controller_matrix": matrix.astype(float).tolist(),
        })

    stationary_gains = np.asarray([
        row["stationary_relative_gain"] for row in seed_rows])
    switching_gains = np.asarray([
        row["switching_relative_gain"] for row in seed_rows])
    copy_errors = np.asarray([
        row["frozen_base_copy_relative_error"] for row in seed_rows])
    diagonal_counts = np.asarray([
        row["diagonal_optimal_modes"] for row in seed_rows])
    termination_gaps = np.asarray([
        row["identity_termination"] - row["late_robust_termination"]
        for row in seed_rows])

    gate = {
        "canonical_base_copy_within_1pct_all_seeds": bool(
            np.all(copy_errors <= 0.01)),
        "stationary_gain_at_least_5pct_each_seed": bool(
            np.all(stationary_gains >= 0.05)),
        "diagonal_optimal_at_least_3_of_4_each_seed": bool(
            np.all(diagonal_counts >= 3)),
    }
    gate["pass"] = all(gate.values())
    if not gate["canonical_base_copy_within_1pct_all_seeds"]:
        decision = "invalidate_protocol_base_copy"
    elif gate["pass"]:
        decision = "authorize_switch_matched_distribution_training"
    else:
        decision = "reject_latebase_fixed_mode_residual_headroom"

    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "diagnostic_only": True,
        "compute_matched": False,
        "critic_target_mode": "min",
        "alpha_frozen": True,
        "training_seeds": list(protocol.TRAINING_SEEDS),
        "event_seeds": list(protocol.EVENT_SEEDS),
        "budget": {
            "late_robust_base_steps": protocol.SOURCE_TOTAL_STEPS,
            "per_controller_post_fork_steps": (
                protocol.PER_CONTROLLER_POST_FORK_STEPS),
            "per_controller_total_steps": protocol.FINAL_TOTAL_STEPS,
            "adapter_bank_aggregate_total_steps": (
                protocol.BANK_AGGREGATE_TOTAL_STEPS),
        },
        "inferential_scope": (
            "Five event streams are averaged within each policy seed. "
            "Only two deliberately difficult training seeds are used, so "
            "this is a fail-fast mechanism diagnostic, not final inference."),
        "seed_rows": seed_rows,
        "specialization_rows": specialization_rows,
        "mean_stationary_relative_gain": float(np.mean(stationary_gains)),
        "mean_switching_relative_gain": float(np.mean(switching_gains)),
        "secondary_checks": {
            "switching_termination_noninferior": bool(
                np.all(termination_gaps <= 0.05)),
        },
        "decision_gate": gate,
        "decision": decision,
        "interpretation": (
            "A pass only shows that a mature robust controller has usable "
            "fixed-mode residual headroom under matched critic targets and "
            "a fixed entropy temperature. It authorizes switch-distribution "
            "training; it is not a publishable BAPR result."),
    }


def _markdown(payload: dict) -> str:
    lines = [
        "# Late-base min-target adapter diagnostic",
        "",
        "**Diagnostic only.** The two previously failing seeds start from the "
        "completed 8.4M-step robust controller. Four canonically initialized "
        "fixed-mode residuals each receive 0.7M additional transitions.",
        "",
        "| Seed | Robust stat | Identity stat | Gain | Robust switch | "
        "Identity switch | Gain | Diagonal optimal | Base-copy error |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["seed_rows"]:
        lines.append(
            f"| {row['training_seed']} | "
            f"{row['late_robust_stationary']:.1f} | "
            f"{row['identity_stationary']:.1f} | "
            f"{100 * row['stationary_relative_gain']:+.1f}% | "
            f"{row['late_robust_switching']:.1f} | "
            f"{row['identity_switching']:.1f} | "
            f"{100 * row['switching_relative_gain']:+.1f}% | "
            f"{row['diagonal_optimal_modes']}/4 | "
            f"{100 * row['frozen_base_copy_relative_error']:.2f}% |")
    lines.extend([
        "",
        f"- Mean stationary relative gain: "
        f"{100 * payload['mean_stationary_relative_gain']:+.1f}%.",
        f"- Mean switching relative gain: "
        f"{100 * payload['mean_switching_relative_gain']:+.1f}%.",
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
