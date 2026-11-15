"""Aggregate the v14 evidence-conflict fallback router development audit."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_conflict_fallback_router_v14 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_conflict_fallback_router_audit_v14 as audit,
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


def _event_wins(arm: dict[str, Any], robust: dict[str, Any]) -> int:
    return sum(
        arm["event_returns"][str(event_seed)]
        > robust["event_returns"][str(event_seed)]
        for event_seed in protocol.EVENT_SEEDS
    )


def _seed_metrics(seed: int) -> dict[str, Any]:
    audit.validate_audit(seed)
    arms = {arm: _arm_metrics(seed, arm) for arm in protocol.ARMS}
    robust = arms["robust_sac"]
    oracle = arms["true_mode_safe_utility"]
    delayed = arms["delayed_oracle_4_safe_utility"]
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

    candidates = {}
    for arm in protocol.CANDIDATE_ARMS:
        row = arms[arm]
        gain = _relative_gain(row["mean"], robust["mean"])
        recovery = _recovery(row["mean"], robust["mean"], oracle["mean"])
        event_wins = _event_wins(row, robust)
        candidates[arm] = {
            "relative_gain": gain,
            "oracle_recovery": recovery,
            "event_wins": event_wins,
            "pass": bool(
                headroom
                and gain >= protocol.MIN_ROUTER_GAIN
                and recovery is not None
                and recovery >= protocol.MIN_CAUSAL_RETENTION
                and event_wins == len(protocol.EVENT_SEEDS)
                and row["terminated_rate"] == 0.0
            ),
        }
    return {
        "arms": arms,
        "relative_safe_oracle_headroom": oracle_gain,
        "safe_oracle_event_wins": oracle_wins,
        "headroom_available": headroom,
        "delay_4_recovery": delayed_recovery,
        "delay_4_pass": delay_pass,
        "candidates": candidates,
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
    causal_margin = bool(
        headroom_count >= protocol.REQUIRED_SEED_PASSES
        and delay_count >= protocol.REQUIRED_SEED_PASSES
    )

    candidate_summary = {}
    for arm in protocol.CANDIDATE_ARMS:
        pass_count = sum(
            row["candidates"][arm]["pass"] for row in seeds.values())
        means = [row["arms"][arm]["mean"] for row in seeds.values()]
        fallback = float(np.mean([
            row["arms"][arm]["candidate_fallback_action_fraction"]
            for row in seeds.values()
        ]))
        recoveries = [
            row["candidates"][arm]["oracle_recovery"]
            for row in seeds.values()
            if row["headroom_available"]
            and row["candidates"][arm]["oracle_recovery"] is not None
        ]
        candidate_summary[arm] = {
            "seed_pass_count": int(pass_count),
            "mean_return": float(np.mean(means)),
            "mean_oracle_recovery": float(np.mean(recoveries)),
            "mean_fallback_action_fraction": fallback,
            "confirmation_steps": protocol.confirm_steps_for_arm(arm),
        }

    selected_arm = max(
        protocol.CANDIDATE_ARMS,
        key=lambda arm: (
            candidate_summary[arm]["seed_pass_count"],
            candidate_summary[arm]["mean_return"],
            -candidate_summary[arm]["mean_fallback_action_fraction"],
            -candidate_summary[arm]["confirmation_steps"],
        ),
    )
    selected_pass_count = candidate_summary[selected_arm]["seed_pass_count"]
    router_development_pass = bool(
        causal_margin and selected_pass_count >= protocol.REQUIRED_SEED_PASSES)
    estimator_retraining_needed = bool(
        causal_margin and not router_development_pass)

    if headroom_count < protocol.REQUIRED_SEED_PASSES:
        diagnosis = "fresh_event_policy_bank_headroom_not_reproducible"
    elif delay_count < protocol.REQUIRED_SEED_PASSES:
        diagnosis = "four_step_causal_margin_not_reproducible"
    elif router_development_pass:
        diagnosis = "causal_conflict_fallback_router_supported_in_development"
    else:
        diagnosis = "switch_focused_estimator_retraining_required"

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

    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "training_seeds": list(protocol.TRAINING_SEEDS),
        "event_seeds": list(protocol.EVENT_SEEDS),
        "seed_metrics": seeds,
        "candidate_summary": candidate_summary,
        "selected_arm": selected_arm,
        "comparisons": comparisons,
        "headroom_seed_count": int(headroom_count),
        "delay_4_pass_seed_count": int(delay_count),
        "causal_margin_reproducible": causal_margin,
        "router_development_pass": router_development_pass,
        "estimator_retraining_needed": estimator_retraining_needed,
        "diagnosis": diagnosis,
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Evidence-conflict fallback router development result",
        "",
        "All controllers, utility maps, and estimator parameters are frozen. Candidate routers use no mode ID or switch clock.",
        "",
        "| Candidate | Seed passes | Mean return | Mean recovery | Fallback actions |",
        "|:---|---:|---:|---:|---:|",
    ]
    for arm in protocol.CANDIDATE_ARMS:
        row = payload["candidate_summary"][arm]
        lines.append(
            f"| {arm} | {row['seed_pass_count']}/5 | "
            f"{row['mean_return']:.1f} | "
            f"{_pct(row['mean_oracle_recovery'])} | "
            f"{_pct(row['mean_fallback_action_fraction'])} |"
        )
    lines.extend([
        "",
        f"Selected arm: **{payload['selected_arm']}**.",
        f"Headroom seeds: **{payload['headroom_seed_count']}/5**; "
        f"delay-4 passes: **{payload['delay_4_pass_seed_count']}/5**.",
        f"Diagnosis: **{payload['diagnosis']}**.",
        f"Router development pass: **{payload['router_development_pass']}**.",
        f"Estimator retraining needed: **{payload['estimator_retraining_needed']}**.",
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
        "V14 CONFLICT FALLBACK ANALYSIS COMPLETE: "
        f"diagnosis={payload['diagnosis']} "
        f"selected={payload['selected_arm']} "
        f"pass={payload['router_development_pass']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
