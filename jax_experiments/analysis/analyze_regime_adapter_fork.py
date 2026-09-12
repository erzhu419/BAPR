"""Aggregate the seed-8 frozen-base independent-adapter development screen."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from jax_experiments.analysis import regime_adapter_fork as protocol
from jax_experiments.analysis.run_regime_adapter_audit import (
    validate_audit,
)
from jax_experiments.analysis.run_regime_adapter_calibration import (
    validate_calibration,
)


T_CRITICAL_95_DF4 = 2.7764451051977987


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean_ci(values) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if len(values) < 2:
        low = high = mean
    else:
        half = T_CRITICAL_95_DF4 * float(
            np.std(values, ddof=1)) / math.sqrt(len(values))
        low, high = mean - half, mean + half
    return {"mean": mean, "ci95_low": low, "ci95_high": high,
            "n_event_streams": int(len(values))}


def _event_value(directory: Path, case: str, metric: str) -> float:
    if metric == "stationary_return":
        rows = _read_rows(directory / case / "task_returns.csv")
        return float(np.mean([float(row["return_mean"]) for row in rows]))
    rows = _read_rows(directory / case / "switching_returns.csv")
    if metric == "switching_return":
        return float(np.mean([float(row["return"]) for row in rows]))
    if metric == "switching_termination":
        return float(np.mean([
            float(row["terminated"] in ("True", "true", "1"))
            for row in rows]))
    raise ValueError(f"unknown metric {metric!r}")


def _case_names(seed: int, delta: float) -> list[str]:
    first = protocol.audit_dir(
        seed, delta, protocol.AUDIT_EVENT_SEEDS[0])
    return sorted(
        path.name for path in first.iterdir()
        if path.is_dir() and (path / "summary.csv").is_file())


def analyze() -> dict:
    if protocol.DEVELOPMENT_SEEDS != (8,):
        raise RuntimeError("v1 analyzer is preregistered for seed 8 only")
    seed = protocol.DEVELOPMENT_SEEDS[0]
    variants = {}
    for delta in protocol.RESIDUAL_DELTAS:
        validate_calibration(seed, delta)
        for event_seed in protocol.AUDIT_EVENT_SEEDS:
            validate_audit(seed, delta, event_seed)
        case_names = _case_names(seed, delta)
        series = {
            case: {
                metric: [
                    _event_value(
                        protocol.audit_dir(seed, delta, event_seed),
                        case, metric)
                    for event_seed in protocol.AUDIT_EVENT_SEEDS
                ]
                for metric in (
                    "stationary_return", "switching_return",
                    "switching_termination")
            }
            for case in case_names
        }
        summaries = {
            case: {metric: _mean_ci(values)
                   for metric, values in metrics.items()}
            for case, metrics in series.items()
        }
        robust_switch = np.asarray(
            series["robust_continue"]["switching_return"])
        utility_switch = np.asarray(
            series["utility_adapter"]["switching_return"])
        robust_stationary = np.asarray(
            series["robust_continue"]["stationary_return"])
        utility_stationary = np.asarray(
            series["utility_adapter"]["stationary_return"])
        robust_termination = float(np.mean(
            series["robust_continue"]["switching_termination"]))
        utility_termination = float(np.mean(
            series["utility_adapter"]["switching_termination"]))
        switch_delta = utility_switch - robust_switch
        stationary_delta = utility_stationary - robust_stationary
        relative_switch = float(
            np.mean(switch_delta) / max(abs(np.mean(robust_switch)), 1e-8))
        relative_stationary = float(
            np.mean(stationary_delta)
            / max(abs(np.mean(robust_stationary)), 1e-8))
        fixed_switch = {
            mode: float(np.mean(
                series[f"fixed_adapter_{mode}"]["switching_return"]))
            for mode in protocol.MODES
        }
        best_fixed = max(fixed_switch, key=fixed_switch.get)
        development_gate = {
            "switching_relative_gain_at_least_10pct": relative_switch >= 0.10,
            "stationary_relative_gain_at_least_5pct": (
                relative_stationary >= 0.05),
            "switching_wins_at_least_4_of_5": int(np.sum(
                switch_delta > 0)) >= 4,
            "utility_beats_every_fixed_switching": (
                float(np.mean(utility_switch)) > max(fixed_switch.values())),
            "termination_noninferior": (
                utility_termination <= robust_termination + 0.05),
        }
        development_gate["pass"] = all(development_gate.values())
        utility = protocol.read_json(
            protocol.calibration_dir(seed, delta) / "utility_map.json")
        variants[protocol.delta_slug(delta)] = {
            "residual_delta": delta,
            "training_seed": seed,
            "controller_map": utility["controller_map"],
            "cases": summaries,
            "utility_minus_robust_switching": {
                **_mean_ci(switch_delta),
                "relative_gain": relative_switch,
                "wins": int(np.sum(switch_delta > 0)),
            },
            "utility_minus_robust_stationary": {
                **_mean_ci(stationary_delta),
                "relative_gain": relative_stationary,
                "wins": int(np.sum(stationary_delta > 0)),
            },
            "best_fixed_switching": {
                "mode": int(best_fixed),
                "mean": fixed_switch[best_fixed],
            },
            "development_gate": development_gate,
        }
    passing = [
        name for name, row in variants.items()
        if row["development_gate"]["pass"]
    ]
    selected = None
    if passing:
        selected = max(
            passing,
            key=lambda name: variants[name][
                "utility_minus_robust_switching"]["mean"])
    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "inferential_scope": (
            "Development screen only: five event streams share one policy "
            "training seed and are not independent training replicates."),
        "budget": {
            "shared_pretrain_steps": protocol.SOURCE_TOTAL_STEPS,
            "robust_additional_steps": (
                protocol.ROBUST_EXTRA_ITERS * protocol.SAMPLES_PER_ITER),
            "adapter_bank_aggregate_additional_steps": (
                len(protocol.MODES)
                * protocol.ADAPTER_EXTRA_ITERS_PER_MODE
                * protocol.SAMPLES_PER_ITER),
            "per_adapter_additional_steps": (
                protocol.ADAPTER_EXTRA_ITERS_PER_MODE
                * protocol.SAMPLES_PER_ITER),
        },
        "variants": variants,
        "passing_variants": passing,
        "selected_for_multiseed": selected,
        "decision": (
            "expand_selected_delta_to_five_training_seeds"
            if selected is not None else
            "do_not_train_estimator_or_expand_adapters"),
    }


def _markdown(payload: dict) -> str:
    lines = [
        "# Frozen-base independent-adapter development screen",
        "",
        "This is a one-training-seed development screen. The five sealed "
        "event streams are paired disturbance replicates, not independent "
        "policy-training seeds; their intervals are descriptive only.",
        "",
        "| Delta | Utility map | Robust switch | Utility switch | Delta | "
        "Relative | Wins | Gate |",
        "|---:|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for name, row in payload["variants"].items():
        robust = row["cases"]["robust_continue"]["switching_return"]["mean"]
        utility = row["cases"]["utility_adapter"]["switching_return"]["mean"]
        delta = row["utility_minus_robust_switching"]
        lines.append(
            f"| {row['residual_delta']:.2f} | "
            f"`{row['controller_map']}` | {robust:.1f} | {utility:.1f} | "
            f"{delta['mean']:+.1f} | {100 * delta['relative_gain']:+.1f}% | "
            f"{delta['wins']}/5 | "
            f"{'pass' if row['development_gate']['pass'] else 'fail'} |")
    lines.extend([
        "",
        f"Decision: `{payload['decision']}`.",
        "",
        "Promotion requires switching gain >=10%, stationary gain >=5%, "
        "at least 4/5 paired event wins, utility routing above every fixed "
        "adapter, and termination no worse than robust by more than 0.05. "
        "A passing delta must still be rerun over five independent training "
        "seeds before any paper claim or learned-estimator training.",
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
