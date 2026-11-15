"""Aggregate v16 switch-weighted estimator development audits."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_switch_weighted_estimator_v16 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_switch_weighted_estimator_audit_v16 as audit,
)


def _arm_metrics(seed: int, arm: str) -> dict[str, Any]:
    rows = [
        protocol.read_json(protocol.event_result(seed, event_seed))["switching"][arm]
        for event_seed in protocol.AUDIT_EVENT_SEEDS
    ]
    values = [float(value) for row in rows for value in row["returns"]]
    event_returns = {
        str(event_seed): float(np.mean(row["returns"]))
        for event_seed, row in zip(protocol.AUDIT_EVENT_SEEDS, rows)
    }
    return {
        "mean": float(np.mean(values)),
        "event_returns": event_returns,
        "terminated_rate": float(np.mean([
            row["terminated_rate"] for row in rows
        ])),
        "routing_mode_accuracy": _optional_mean(
            rows, "routing_mode_accuracy"),
        "switch_window_routing_accuracy": _optional_mean(
            rows, "switch_window_routing_accuracy"),
        "stable_routing_accuracy": _optional_mean(
            rows, "stable_routing_accuracy"),
        "wrong_specialist_action_fraction": float(np.mean([
            row["wrong_specialist_action_fraction"] for row in rows
        ])),
        "controller_mismatch_action_fraction": float(np.mean([
            row["controller_mismatch_action_fraction"] for row in rows
        ])),
        "mapped_robust_action_fraction": float(np.mean([
            row["mapped_robust_action_fraction"] for row in rows
        ])),
    }


def _optional_mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _relative_gain(value: float, robust: float) -> float:
    return (
        (value - robust) / abs(robust)
        if robust != 0.0 else float("-inf")
    )


def _recovery(value: float, robust: float, oracle: float) -> float | None:
    headroom = oracle - robust
    return (value - robust) / headroom if headroom > 0.0 else None


def _event_wins(arm: dict[str, Any], baseline: dict[str, Any]) -> int:
    return sum(
        arm["event_returns"][str(event_seed)]
        > baseline["event_returns"][str(event_seed)]
        for event_seed in protocol.AUDIT_EVENT_SEEDS
    )


def _seed_metrics(seed: int) -> dict[str, Any]:
    audit.validate_audit(seed)
    arms = {arm: _arm_metrics(seed, arm) for arm in protocol.ARMS}
    robust = arms["robust_sac"]
    oracle = arms["true_mode_safe_utility"]
    delayed = arms["delayed_oracle_4_safe_utility"]
    frozen = arms["frozen_v5_posterior_map"]
    candidate = arms["switch_weighted_v16_posterior_map"]

    oracle_gain = _relative_gain(oracle["mean"], robust["mean"])
    oracle_wins = _event_wins(oracle, robust)
    headroom = bool(
        oracle_gain >= protocol.MIN_HEADROOM_GAIN
        and oracle_wins == len(protocol.AUDIT_EVENT_SEEDS)
        and oracle["terminated_rate"] == 0.0
    )
    delayed_recovery = _recovery(
        delayed["mean"], robust["mean"], oracle["mean"])
    delay_pass = bool(
        headroom
        and delayed_recovery is not None
        and delayed_recovery >= protocol.MIN_CAUSAL_RETENTION
        and _event_wins(delayed, robust) == len(protocol.AUDIT_EVENT_SEEDS)
        and delayed["terminated_rate"] == 0.0
    )
    candidate_gain = _relative_gain(candidate["mean"], robust["mean"])
    candidate_recovery = _recovery(
        candidate["mean"], robust["mean"], oracle["mean"])
    candidate_event_wins = _event_wins(candidate, robust)
    candidate_strict_pass = bool(
        headroom
        and candidate_gain >= protocol.MIN_ESTIMATOR_GAIN
        and candidate_recovery is not None
        and candidate_recovery >= protocol.MIN_CAUSAL_RETENTION
        and candidate_event_wins == len(protocol.AUDIT_EVENT_SEEDS)
        and candidate["terminated_rate"] == 0.0
    )
    return {
        "arms": arms,
        "relative_safe_oracle_headroom": oracle_gain,
        "safe_oracle_event_wins": oracle_wins,
        "headroom_available": headroom,
        "delay_4_recovery": delayed_recovery,
        "delay_4_pass": delay_pass,
        "candidate_relative_gain": candidate_gain,
        "candidate_oracle_recovery": candidate_recovery,
        "candidate_event_wins": candidate_event_wins,
        "candidate_strict_pass": candidate_strict_pass,
        "candidate_beats_frozen_v5": bool(
            candidate["mean"] > frozen["mean"]),
        "candidate_minus_frozen_v5": float(
            candidate["mean"] - frozen["mean"]),
        "switch_accuracy_change": float(
            candidate["switch_window_routing_accuracy"]
            - frozen["switch_window_routing_accuracy"]),
        "stable_accuracy_change": float(
            candidate["stable_routing_accuracy"]
            - frozen["stable_routing_accuracy"]),
    }


def _mean_ci95(values: list[float]) -> list[float]:
    values = [float(value) for value in values]
    mean = float(np.mean(values))
    if len(values) < 2:
        return [mean, mean]
    t_critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(
        len(values), 1.96)
    half = t_critical * float(np.std(values, ddof=1)) / math.sqrt(len(values))
    return [mean - half, mean + half]


def analyze() -> dict[str, Any]:
    protocol.validate_registration()
    seeds = {
        str(seed): _seed_metrics(seed) for seed in protocol.AUDIT_POLICY_SEEDS
    }
    headroom_count = sum(row["headroom_available"] for row in seeds.values())
    delay_count = sum(row["delay_4_pass"] for row in seeds.values())
    strict_count = sum(row["candidate_strict_pass"] for row in seeds.values())
    v5_seed_wins = sum(
        row["candidate_beats_frozen_v5"] for row in seeds.values())
    validation_wins = sum(
        seeds[str(seed)]["candidate_beats_frozen_v5"]
        for seed in protocol.VALIDATION_POLICY_SEEDS
    )
    differences = [
        row["candidate_minus_frozen_v5"] for row in seeds.values()
    ]
    difference_mean = float(np.mean(differences))
    causal_margin = bool(
        headroom_count >= protocol.REQUIRED_SEED_PASSES
        and delay_count >= protocol.REQUIRED_SEED_PASSES
    )
    estimator_comparison_pass = bool(
        v5_seed_wins >= protocol.REQUIRED_V5_SEED_WINS
        and validation_wins >= protocol.REQUIRED_VALIDATION_POLICY_WINS
        and difference_mean > 0.0
    )
    development_pass = bool(
        causal_margin
        and strict_count >= protocol.REQUIRED_SEED_PASSES
        and estimator_comparison_pass
    )

    if headroom_count < protocol.REQUIRED_SEED_PASSES:
        diagnosis = "v16_audit_headroom_not_reproducible"
    elif delay_count < protocol.REQUIRED_SEED_PASSES:
        diagnosis = "v16_four_step_causal_margin_not_reproducible"
    elif strict_count < protocol.REQUIRED_SEED_PASSES:
        diagnosis = "switch_weighted_estimator_fails_strict_control_gate"
    elif validation_wins < protocol.REQUIRED_VALIDATION_POLICY_WINS:
        diagnosis = "switch_weighted_estimator_fails_heldout_policy_banks"
    elif not estimator_comparison_pass:
        diagnosis = "switch_weighted_estimator_has_no_incremental_v5_value"
    else:
        diagnosis = "switch_weighted_estimator_supported_in_development"

    robust_values = [
        row["arms"]["robust_sac"]["mean"] for row in seeds.values()
    ]
    comparisons = {}
    for arm in protocol.ARMS[1:]:
        values = [row["arms"][arm]["mean"] for row in seeds.values()]
        paired = [a - b for a, b in zip(values, robust_values)]
        comparisons[arm] = {
            "mean": float(np.mean(values)),
            "robust_mean": float(np.mean(robust_values)),
            "paired_difference_mean": float(np.mean(paired)),
            "paired_difference_ci95": _mean_ci95(paired),
            "seed_wins": int(sum(a > b for a, b in zip(values, robust_values))),
        }
    candidate_values = [
        row["arms"]["switch_weighted_v16_posterior_map"]["mean"]
        for row in seeds.values()
    ]
    frozen_values = [
        row["arms"]["frozen_v5_posterior_map"]["mean"]
        for row in seeds.values()
    ]
    candidate_vs_frozen = {
        "candidate_mean": float(np.mean(candidate_values)),
        "frozen_v5_mean": float(np.mean(frozen_values)),
        "paired_difference_mean": difference_mean,
        "paired_difference_ci95": _mean_ci95(differences),
        "seed_wins": int(v5_seed_wins),
        "validation_policy_wins": int(validation_wins),
        "switch_accuracy_improvement_seed_count": int(sum(
            row["switch_accuracy_change"] > 0.0 for row in seeds.values())),
        "stable_accuracy_non_degradation_seed_count": int(sum(
            row["stable_accuracy_change"] >= 0.0 for row in seeds.values())),
        "pass": estimator_comparison_pass,
    }
    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "policy_seeds": list(protocol.AUDIT_POLICY_SEEDS),
        "validation_policy_seeds": list(protocol.VALIDATION_POLICY_SEEDS),
        "event_seeds": list(protocol.AUDIT_EVENT_SEEDS),
        "seed_metrics": seeds,
        "comparisons": comparisons,
        "candidate_vs_frozen_v5": candidate_vs_frozen,
        "headroom_seed_count": int(headroom_count),
        "delay_4_pass_seed_count": int(delay_count),
        "candidate_strict_pass_seed_count": int(strict_count),
        "causal_margin_reproducible": causal_margin,
        "estimator_comparison_pass": estimator_comparison_pass,
        "estimator_development_pass": development_pass,
        "fresh_policy_bank_confirmation_authorized": development_pass,
        "close_inverse_estimator_family": not development_pass,
        "diagnosis": diagnosis,
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Switch-weighted expected-action estimator development result",
        "",
        "Policies, utility maps, environment, posterior filter, and MAP routing are frozen; only inverse-evidence training changed.",
        "",
        "| Seed | Robust | Oracle | V5 MAP | V16 MAP | Gain | Recovery | d(V5) | Switch acc d | Strict |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for seed in protocol.AUDIT_POLICY_SEEDS:
        row = payload["seed_metrics"][str(seed)]
        arms = row["arms"]
        lines.append(
            f"| {seed} | {arms['robust_sac']['mean']:.1f} | "
            f"{arms['true_mode_safe_utility']['mean']:.1f} | "
            f"{arms['frozen_v5_posterior_map']['mean']:.1f} | "
            f"{arms['switch_weighted_v16_posterior_map']['mean']:.1f} | "
            f"{_pct(row['candidate_relative_gain'])} | "
            f"{_pct(row['candidate_oracle_recovery'])} | "
            f"{row['candidate_minus_frozen_v5']:+.1f} | "
            f"{_pct(row['switch_accuracy_change'])} | "
            f"{'yes' if row['candidate_strict_pass'] else 'no'} |"
        )
    comparison = payload["candidate_vs_frozen_v5"]
    lines.extend([
        "",
        f"Headroom: **{payload['headroom_seed_count']}/5**; delay-4: "
        f"**{payload['delay_4_pass_seed_count']}/5**; strict v16: "
        f"**{payload['candidate_strict_pass_seed_count']}/5**.",
        f"V16 versus frozen v5: **{comparison['seed_wins']}/5** seed wins, "
        f"held-out policy wins **{comparison['validation_policy_wins']}/2**, "
        f"paired mean **{comparison['paired_difference_mean']:+.1f}**, "
        f"95% CI [{comparison['paired_difference_ci95'][0]:+.1f}, "
        f"{comparison['paired_difference_ci95'][1]:+.1f}].",
        f"Diagnosis: **{payload['diagnosis']}**.",
        f"Estimator development pass: **{payload['estimator_development_pass']}**.",
        f"Fresh-policy-bank confirmation authorized: "
        f"**{payload['fresh_policy_bank_confirmation_authorized']}**.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    payload = analyze()
    markdown = render(payload)
    protocol.write_json_atomic(protocol.analysis_json(), payload)
    protocol.write_text_atomic(protocol.analysis_markdown(), markdown)
    protocol.write_text_atomic(protocol.REPORT, markdown)
    print(
        "V16 SWITCH-WEIGHTED ESTIMATOR ANALYSIS COMPLETE: "
        f"diagnosis={payload['diagnosis']} "
        f"pass={payload['estimator_development_pass']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
