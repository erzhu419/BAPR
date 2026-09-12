"""Aggregate the frozen v14 router confirmation on v15 holdout events."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_conflict_fallback_confirmation_v15 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_conflict_fallback_confirmation_audit_v15 as audit,
)


def _arm_metrics(seed: int, arm: str) -> dict[str, Any]:
    rows = [
        protocol.read_json(protocol.event_result(seed, event_seed))["switching"][arm]
        for event_seed in protocol.EVENT_SEEDS
    ]
    values = [float(value) for row in rows for value in row["returns"]]
    event_returns = {
        str(event_seed): float(np.mean(row["returns"]))
        for event_seed, row in zip(protocol.EVENT_SEEDS, rows)
    }
    posterior_rows = [
        row["posterior_metrics"] for row in rows
        if row.get("posterior_metrics") is not None
    ]
    return {
        "mean": float(np.mean(values)),
        "event_returns": event_returns,
        "terminated_rate": float(np.mean([
            row["terminated_rate"] for row in rows
        ])),
        "routing_mode_accuracy": (
            float(np.mean([
                row["routing_mode_accuracy"] for row in rows
                if row.get("routing_mode_accuracy") is not None
            ]))
            if any(row.get("routing_mode_accuracy") is not None for row in rows)
            else None
        ),
        "posterior_mode_accuracy": (
            float(np.mean([row["mode_accuracy"] for row in posterior_rows]))
            if posterior_rows else None
        ),
        "posterior_brier_score": (
            float(np.mean([row["brier_score"] for row in posterior_rows]))
            if posterior_rows else None
        ),
        "wrong_specialist_action_fraction": float(np.mean([
            row["wrong_specialist_action_fraction"] for row in rows
        ])),
        "mapped_robust_action_fraction": float(np.mean([
            row["mapped_robust_action_fraction"] for row in rows
        ])),
        "candidate_fallback_action_fraction": float(np.mean([
            row["candidate_fallback_action_fraction"] for row in rows
        ])),
        "evidence_conflict_count_mean": float(np.mean([
            row["evidence_conflict_count"] for row in rows
        ])),
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
        for event_seed in protocol.EVENT_SEEDS
    )


def _seed_metrics(seed: int) -> dict[str, Any]:
    audit.validate_audit(seed)
    arms = {arm: _arm_metrics(seed, arm) for arm in protocol.ARMS}
    robust = arms["robust_sac"]
    oracle = arms["true_mode_safe_utility"]
    delayed = arms["delayed_oracle_4_safe_utility"]
    plain_map = arms["posterior_map_safe_utility"]
    selected = arms[protocol.SELECTED_ARM]

    oracle_gain = _relative_gain(oracle["mean"], robust["mean"])
    oracle_wins = _event_wins(oracle, robust)
    headroom = bool(
        oracle_gain >= protocol.MIN_HEADROOM_GAIN
        and oracle_wins == len(protocol.EVENT_SEEDS)
        and oracle["terminated_rate"] == 0.0
    )
    delayed_recovery = _recovery(
        delayed["mean"], robust["mean"], oracle["mean"])
    delay_pass = bool(
        headroom
        and delayed_recovery is not None
        and delayed_recovery >= protocol.MIN_CAUSAL_RETENTION
        and _event_wins(delayed, robust) == len(protocol.EVENT_SEEDS)
        and delayed["terminated_rate"] == 0.0
    )

    selected_gain = _relative_gain(selected["mean"], robust["mean"])
    selected_recovery = _recovery(
        selected["mean"], robust["mean"], oracle["mean"])
    selected_event_wins = _event_wins(selected, robust)
    selected_strict_pass = bool(
        headroom
        and selected_gain >= protocol.MIN_ROUTER_GAIN
        and selected_recovery is not None
        and selected_recovery >= protocol.MIN_CAUSAL_RETENTION
        and selected_event_wins == len(protocol.EVENT_SEEDS)
        and selected["terminated_rate"] == 0.0
    )
    return {
        "arms": arms,
        "relative_safe_oracle_headroom": oracle_gain,
        "safe_oracle_event_wins": oracle_wins,
        "headroom_available": headroom,
        "delay_4_recovery": delayed_recovery,
        "delay_4_pass": delay_pass,
        "selected_relative_gain": selected_gain,
        "selected_oracle_recovery": selected_recovery,
        "selected_event_wins": selected_event_wins,
        "selected_strict_pass": selected_strict_pass,
        "selected_beats_plain_map": bool(selected["mean"] > plain_map["mean"]),
        "selected_minus_plain_map": float(
            selected["mean"] - plain_map["mean"]),
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
    headroom_count = sum(row["headroom_available"] for row in seeds.values())
    delay_count = sum(row["delay_4_pass"] for row in seeds.values())
    selected_pass_count = sum(
        row["selected_strict_pass"] for row in seeds.values())
    map_seed_wins = sum(
        row["selected_beats_plain_map"] for row in seeds.values())
    map_differences = [
        row["selected_minus_plain_map"] for row in seeds.values()
    ]
    map_difference_mean = float(np.mean(map_differences))

    causal_margin = bool(
        headroom_count >= protocol.REQUIRED_SEED_PASSES
        and delay_count >= protocol.REQUIRED_SEED_PASSES
    )
    incremental_map_pass = bool(
        map_seed_wins >= protocol.REQUIRED_MAP_SEED_WINS
        and map_difference_mean > 0.0
    )
    confirmation_pass = bool(
        causal_margin
        and selected_pass_count >= protocol.REQUIRED_SEED_PASSES
        and incremental_map_pass
    )

    if headroom_count < protocol.REQUIRED_SEED_PASSES:
        diagnosis = "holdout_policy_bank_headroom_not_reproducible"
    elif delay_count < protocol.REQUIRED_SEED_PASSES:
        diagnosis = "holdout_four_step_causal_margin_not_reproducible"
    elif selected_pass_count < protocol.REQUIRED_SEED_PASSES:
        diagnosis = "frozen_conflict_fallback_not_strictly_reproducible"
    elif not incremental_map_pass:
        diagnosis = "frozen_conflict_fallback_has_no_incremental_map_value"
    else:
        diagnosis = "frozen_conflict_fallback_router_confirmed_on_holdout_events"

    robust_values = [
        row["arms"]["robust_sac"]["mean"] for row in seeds.values()
    ]
    comparisons = {}
    for arm in protocol.ARMS[1:]:
        values = [row["arms"][arm]["mean"] for row in seeds.values()]
        differences = [a - b for a, b in zip(values, robust_values)]
        comparisons[arm] = {
            "mean": float(np.mean(values)),
            "robust_mean": float(np.mean(robust_values)),
            "paired_difference_mean": float(np.mean(differences)),
            "paired_difference_ci95": _mean_ci95(differences),
            "seed_wins": int(sum(a > b for a, b in zip(values, robust_values))),
        }

    selected_values = [
        row["arms"][protocol.SELECTED_ARM]["mean"] for row in seeds.values()
    ]
    plain_map_values = [
        row["arms"]["posterior_map_safe_utility"]["mean"]
        for row in seeds.values()
    ]
    selected_vs_plain_map = {
        "selected_mean": float(np.mean(selected_values)),
        "plain_map_mean": float(np.mean(plain_map_values)),
        "paired_difference_mean": map_difference_mean,
        "paired_difference_ci95": _mean_ci95(map_differences),
        "seed_wins": int(map_seed_wins),
        "pass": incremental_map_pass,
    }

    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "training_seeds": list(protocol.TRAINING_SEEDS),
        "event_seeds": list(protocol.EVENT_SEEDS),
        "selected_arm": protocol.SELECTED_ARM,
        "seed_metrics": seeds,
        "comparisons": comparisons,
        "selected_vs_plain_map": selected_vs_plain_map,
        "headroom_seed_count": int(headroom_count),
        "delay_4_pass_seed_count": int(delay_count),
        "selected_strict_pass_seed_count": int(selected_pass_count),
        "causal_margin_reproducible": causal_margin,
        "incremental_plain_map_pass": incremental_map_pass,
        "router_confirmation_pass": confirmation_pass,
        "fresh_policy_bank_confirmation_authorized": confirmation_pass,
        "switch_focused_estimator_retraining_required": bool(
            causal_margin and not confirmation_pass),
        "diagnosis": diagnosis,
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Frozen conflict-fallback router holdout confirmation",
        "",
        "The v14 confirm-3 router and all controller, estimator, and utility-map parameters were frozen before these events.",
        "",
        "| Seed | Robust | Oracle | Delay-4 | Plain MAP | Confirm-3 | Gain | Recovery | Strict | Beats MAP |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for seed in protocol.TRAINING_SEEDS:
        row = payload["seed_metrics"][str(seed)]
        arms = row["arms"]
        lines.append(
            f"| {seed} | {arms['robust_sac']['mean']:.1f} | "
            f"{arms['true_mode_safe_utility']['mean']:.1f} | "
            f"{arms['delayed_oracle_4_safe_utility']['mean']:.1f} | "
            f"{arms['posterior_map_safe_utility']['mean']:.1f} | "
            f"{arms[protocol.SELECTED_ARM]['mean']:.1f} | "
            f"{_pct(row['selected_relative_gain'])} | "
            f"{_pct(row['selected_oracle_recovery'])} | "
            f"{'yes' if row['selected_strict_pass'] else 'no'} | "
            f"{'yes' if row['selected_beats_plain_map'] else 'no'} |"
        )
    comparison = payload["selected_vs_plain_map"]
    lines.extend([
        "",
        f"Headroom seeds: **{payload['headroom_seed_count']}/5**; "
        f"delay-4 passes: **{payload['delay_4_pass_seed_count']}/5**; "
        f"strict router passes: **{payload['selected_strict_pass_seed_count']}/5**.",
        f"Confirm-3 versus plain MAP: **{comparison['seed_wins']}/5** seed wins, "
        f"paired mean **{comparison['paired_difference_mean']:+.1f}**, "
        f"95% CI [{comparison['paired_difference_ci95'][0]:+.1f}, "
        f"{comparison['paired_difference_ci95'][1]:+.1f}].",
        f"Diagnosis: **{payload['diagnosis']}**.",
        f"Holdout confirmation pass: **{payload['router_confirmation_pass']}**.",
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
        "V15 CONFLICT FALLBACK CONFIRMATION COMPLETE: "
        f"diagnosis={payload['diagnosis']} "
        f"pass={payload['router_confirmation_pass']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
