"""Aggregate the fresh-policy-bank v17 estimator confirmation."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from jax_experiments.analysis import (
    analyze_regime_polarity_robust_warmstart_specialist_v11 as policy_analysis,
)
from jax_experiments.analysis import (
    regime_polarity_fresh_bank_estimator_confirmation_v17 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_fresh_bank_estimator_confirmation_audit_v17 as audit,
)


def _mean(values) -> float:
    return float(np.mean([float(value) for value in values]))


def _optional_mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _arm_metrics(payload: dict[str, Any], arm: str) -> dict[str, Any]:
    rows = [
        payload["switching_holdout"][str(event_seed)][arm]
        for event_seed in protocol.SWITCHING_EVENT_SEEDS
    ]
    return {
        "mean": _mean(row["return_mean"] for row in rows),
        "event_returns": {
            str(event_seed): float(row["return_mean"])
            for event_seed, row in zip(protocol.SWITCHING_EVENT_SEEDS, rows)
        },
        "terminated_rate": _mean(row["terminated_rate"] for row in rows),
        "routing_mode_accuracy": _optional_mean(
            rows, "routing_mode_accuracy"),
        "switch_window_routing_accuracy": _optional_mean(
            rows, "switch_window_routing_accuracy"),
        "stable_routing_accuracy": _optional_mean(
            rows, "stable_routing_accuracy"),
        "wrong_specialist_action_fraction": _mean(
            row["wrong_specialist_action_fraction"] for row in rows),
        "controller_mismatch_action_fraction": _mean(
            row["controller_mismatch_action_fraction"] for row in rows),
        "mapped_robust_action_fraction": _mean(
            row["mapped_robust_action_fraction"] for row in rows),
    }


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
        for event_seed in protocol.SWITCHING_EVENT_SEEDS
    )


def _seed_metrics(seed: int) -> dict[str, Any]:
    audit.validate_audit(seed)
    payload = protocol.read_json(protocol.audit_result(seed))
    policy_analysis.protocol = protocol
    stationary = policy_analysis._stationary_metrics(payload)
    arms = {arm: _arm_metrics(payload, arm) for arm in protocol.ARMS}
    robust = arms["robust_sac"]
    oracle = arms["true_mode_safe_utility"]
    delayed = arms["delayed_oracle_4_safe_utility"]
    v5 = arms["frozen_v5_posterior_map"]
    v16 = arms["switch_weighted_v16_posterior_map"]

    oracle_gain = _relative_gain(oracle["mean"], robust["mean"])
    oracle_event_wins = _event_wins(oracle, robust)
    headroom = bool(
        oracle_gain >= protocol.MIN_HEADROOM_GAIN
        and oracle_event_wins == len(protocol.SWITCHING_EVENT_SEEDS)
        and oracle["terminated_rate"] == 0.0
    )
    policy_bank_pass = bool(stationary["pass"] and headroom)
    delayed_recovery = _recovery(
        delayed["mean"], robust["mean"], oracle["mean"])
    delay_pass = bool(
        headroom
        and delayed_recovery is not None
        and delayed_recovery >= protocol.MIN_CAUSAL_RETENTION
        and _event_wins(delayed, robust)
        == len(protocol.SWITCHING_EVENT_SEEDS)
        and delayed["terminated_rate"] == 0.0
    )
    v16_gain = _relative_gain(v16["mean"], robust["mean"])
    v16_recovery = _recovery(
        v16["mean"], robust["mean"], oracle["mean"])
    v16_event_wins = _event_wins(v16, robust)
    v16_strict_pass = bool(
        headroom
        and v16_gain >= protocol.MIN_ESTIMATOR_GAIN
        and v16_recovery is not None
        and v16_recovery >= protocol.MIN_CAUSAL_RETENTION
        and v16_event_wins == len(protocol.SWITCHING_EVENT_SEEDS)
        and v16["terminated_rate"] == 0.0
    )
    switch_accuracy_change = float(
        v16["switch_window_routing_accuracy"]
        - v5["switch_window_routing_accuracy"])
    wrong_action_change = float(
        v16["wrong_specialist_action_fraction"]
        - v5["wrong_specialist_action_fraction"])
    mechanism_pass = bool(
        switch_accuracy_change >= 0.0
        and wrong_action_change <= 0.0
    )
    return {
        "utility_map": payload["utility_map"],
        "stationary": stationary,
        "arms": arms,
        "relative_safe_oracle_headroom": oracle_gain,
        "safe_oracle_event_wins": oracle_event_wins,
        "headroom_available": headroom,
        "policy_bank_pass": policy_bank_pass,
        "delay_4_recovery": delayed_recovery,
        "delay_4_pass": delay_pass,
        "v16_relative_gain": v16_gain,
        "v16_oracle_recovery": v16_recovery,
        "v16_event_wins": v16_event_wins,
        "v16_strict_pass": v16_strict_pass,
        "v16_beats_v5": bool(v16["mean"] > v5["mean"]),
        "v16_minus_v5": float(v16["mean"] - v5["mean"]),
        "switch_accuracy_change": switch_accuracy_change,
        "wrong_specialist_action_fraction_change": wrong_action_change,
        "mechanism_pass": mechanism_pass,
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
        str(seed): _seed_metrics(seed) for seed in protocol.TRAINING_SEEDS
    }
    policy_bank_count = sum(
        row["policy_bank_pass"] for row in seeds.values())
    headroom_count = sum(
        row["headroom_available"] for row in seeds.values())
    delay_count = sum(row["delay_4_pass"] for row in seeds.values())
    strict_count = sum(row["v16_strict_pass"] for row in seeds.values())
    v16_seed_wins = sum(row["v16_beats_v5"] for row in seeds.values())
    mechanism_count = sum(row["mechanism_pass"] for row in seeds.values())
    differences = [row["v16_minus_v5"] for row in seeds.values()]
    difference_mean = float(np.mean(differences))
    difference_ci95 = _mean_ci95(differences)

    policy_bank_pass = bool(
        policy_bank_count >= protocol.REQUIRED_SEED_PASSES)
    causal_margin_pass = bool(
        headroom_count >= protocol.REQUIRED_SEED_PASSES
        and delay_count >= protocol.REQUIRED_SEED_PASSES
    )
    control_pass = bool(
        strict_count >= protocol.REQUIRED_SEED_PASSES
        and v16_seed_wins >= protocol.REQUIRED_V16_SEED_WINS
        and difference_mean > 0.0
        and difference_ci95[0] > 0.0
    )
    mechanism_pass = bool(
        mechanism_count >= protocol.REQUIRED_MECHANISM_SEED_PASSES)
    confirmation_pass = bool(
        policy_bank_pass
        and causal_margin_pass
        and control_pass
        and mechanism_pass
    )

    if not policy_bank_pass:
        diagnosis = "fresh_actor_only_policy_banks_not_reproducible"
    elif headroom_count < protocol.REQUIRED_SEED_PASSES:
        diagnosis = "fresh_safe_oracle_headroom_not_reproducible"
    elif delay_count < protocol.REQUIRED_SEED_PASSES:
        diagnosis = "fresh_four_step_causal_margin_not_reproducible"
    elif strict_count < protocol.REQUIRED_SEED_PASSES:
        diagnosis = "v16_fails_fresh_bank_strict_control_gate"
    elif v16_seed_wins < protocol.REQUIRED_V16_SEED_WINS:
        diagnosis = "v16_fails_fresh_bank_seed_win_gate"
    elif difference_ci95[0] <= 0.0:
        diagnosis = "v16_fresh_bank_return_advantage_not_precise"
    elif not mechanism_pass:
        diagnosis = "v16_fresh_bank_mechanism_not_supported"
    else:
        diagnosis = "v16_fresh_bank_confirmation_supported"

    robust_values = [
        row["arms"]["robust_sac"]["mean"] for row in seeds.values()
    ]
    comparisons = {}
    for arm in protocol.ARMS[1:]:
        values = [row["arms"][arm]["mean"] for row in seeds.values()]
        paired = [value - robust for value, robust in zip(
            values, robust_values)]
        comparisons[arm] = {
            "mean": float(np.mean(values)),
            "robust_mean": float(np.mean(robust_values)),
            "paired_difference_mean": float(np.mean(paired)),
            "paired_difference_ci95": _mean_ci95(paired),
            "seed_wins": int(sum(value > robust for value, robust in zip(
                values, robust_values))),
        }
    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "training_seeds": list(protocol.TRAINING_SEEDS),
        "event_seeds": list(protocol.SWITCHING_EVENT_SEEDS),
        "seed_metrics": seeds,
        "comparisons": comparisons,
        "policy_bank_pass_count": int(policy_bank_count),
        "headroom_seed_count": int(headroom_count),
        "delay_4_pass_seed_count": int(delay_count),
        "v16_strict_pass_seed_count": int(strict_count),
        "v16_seed_wins_over_v5": int(v16_seed_wins),
        "v16_minus_v5_mean": difference_mean,
        "v16_minus_v5_ci95": difference_ci95,
        "mechanism_pass_seed_count": int(mechanism_count),
        "policy_bank_confirmation_pass": policy_bank_pass,
        "causal_margin_confirmation_pass": causal_margin_pass,
        "control_confirmation_pass": control_pass,
        "mechanism_confirmation_pass": mechanism_pass,
        "estimator_confirmation_pass": confirmation_pass,
        "promote_v16_to_final_baseline_comparison": confirmation_pass,
        "close_executed_action_inverse_estimator_family": (
            not confirmation_pass),
        "diagnosis": diagnosis,
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Fresh-policy-bank estimator confirmation result",
        "",
        "All robust sources and actor-only specialists use new policy seeds; "
        "v5 and v16 estimator parameters remain frozen.",
        "",
        "| Seed | Modes | Robust | Oracle | V5 MAP | V16 MAP | d(V5) | "
        "Switch acc d | Wrong-action d | Strict | Mechanism |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for seed in protocol.TRAINING_SEEDS:
        row = payload["seed_metrics"][str(seed)]
        arms = row["arms"]
        lines.append(
            f"| {seed} | {row['stationary']['mode_wins']}/4 | "
            f"{arms['robust_sac']['mean']:.1f} | "
            f"{arms['true_mode_safe_utility']['mean']:.1f} | "
            f"{arms['frozen_v5_posterior_map']['mean']:.1f} | "
            f"{arms['switch_weighted_v16_posterior_map']['mean']:.1f} | "
            f"{row['v16_minus_v5']:+.1f} | "
            f"{_pct(row['switch_accuracy_change'])} | "
            f"{_pct(row['wrong_specialist_action_fraction_change'])} | "
            f"{'yes' if row['v16_strict_pass'] else 'no'} | "
            f"{'yes' if row['mechanism_pass'] else 'no'} |"
        )
    lines.extend([
        "",
        f"Policy banks: **{payload['policy_bank_pass_count']}/5**; headroom: "
        f"**{payload['headroom_seed_count']}/5**; delay-4: "
        f"**{payload['delay_4_pass_seed_count']}/5**; strict v16: "
        f"**{payload['v16_strict_pass_seed_count']}/5**.",
        f"V16 versus v5: **{payload['v16_seed_wins_over_v5']}/5** seed "
        f"wins, paired mean **{payload['v16_minus_v5_mean']:+.1f}**, 95% CI "
        f"[{payload['v16_minus_v5_ci95'][0]:+.1f}, "
        f"{payload['v16_minus_v5_ci95'][1]:+.1f}].",
        f"Mechanism gate: **{payload['mechanism_pass_seed_count']}/5** seeds.",
        f"Diagnosis: **{payload['diagnosis']}**.",
        f"Estimator confirmation pass: "
        f"**{payload['estimator_confirmation_pass']}**.",
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
        "V17 FRESH-BANK ESTIMATOR CONFIRMATION COMPLETE: "
        f"diagnosis={payload['diagnosis']} "
        f"pass={payload['estimator_confirmation_pass']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
