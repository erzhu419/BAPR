"""Aggregate the v13 frozen-estimator transfer audit."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_frozen_estimator_transfer_v13 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_frozen_estimator_transfer_audit_v13 as audit,
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
    posterior_metrics = [
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
            float(np.mean([
                row["mode_accuracy"] for row in posterior_metrics
            ])) if posterior_metrics else None
        ),
        "posterior_brier_score": (
            float(np.mean([
                row["brier_score"] for row in posterior_metrics
            ])) if posterior_metrics else None
        ),
        "mapped_robust_action_fraction": float(np.mean([
            row["mapped_robust_action_fraction"] for row in rows
        ])),
        "delayed_action_fraction": float(np.mean([
            row["delayed_action_fraction"] for row in rows
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
    posterior = arms[protocol.PRIMARY_ARM]

    oracle_gain = _relative_gain(oracle["mean"], robust["mean"])
    delayed_recovery = _recovery(
        delayed["mean"], robust["mean"], oracle["mean"])
    posterior_gain = _relative_gain(posterior["mean"], robust["mean"])
    posterior_recovery = _recovery(
        posterior["mean"], robust["mean"], oracle["mean"])
    oracle_wins = _event_wins(oracle, robust)
    delayed_wins = _event_wins(delayed, robust)
    posterior_wins = _event_wins(posterior, robust)

    headroom_available = bool(
        oracle_gain >= protocol.MIN_HEADROOM_GAIN
        and oracle_wins == len(protocol.EVENT_SEEDS)
        and oracle["terminated_rate"] == 0.0
    )
    delay_pass = bool(
        headroom_available
        and delayed_recovery is not None
        and delayed_recovery >= protocol.MIN_CAUSAL_RETENTION
        and delayed_wins == len(protocol.EVENT_SEEDS)
        and delayed["terminated_rate"] == 0.0
    )
    posterior_pass = bool(
        headroom_available
        and posterior_gain >= protocol.MIN_POSTERIOR_GAIN
        and posterior_recovery is not None
        and posterior_recovery >= protocol.MIN_CAUSAL_RETENTION
        and posterior_wins == len(protocol.EVENT_SEEDS)
        and posterior["terminated_rate"] == 0.0
        and posterior["posterior_mode_accuracy"] is not None
        and posterior["posterior_mode_accuracy"] >= protocol.MIN_MODE_ACCURACY
    )
    return {
        "arms": arms,
        "relative_safe_oracle_headroom": oracle_gain,
        "safe_oracle_event_wins": oracle_wins,
        "headroom_available": headroom_available,
        "delay_4_recovery": delayed_recovery,
        "delay_4_event_wins": delayed_wins,
        "delay_4_pass": delay_pass,
        "posterior_relative_gain": posterior_gain,
        "posterior_recovery": posterior_recovery,
        "posterior_event_wins": posterior_wins,
        "posterior_pass": posterior_pass,
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
    posterior_count = sum(row["posterior_pass"] for row in seeds.values())
    causal_margin_reproducible = bool(
        headroom_count >= protocol.REQUIRED_SEED_PASSES
        and delay_count >= protocol.REQUIRED_SEED_PASSES
    )
    frozen_estimator_confirmed = bool(
        causal_margin_reproducible
        and posterior_count >= protocol.REQUIRED_SEED_PASSES
    )
    estimator_retraining_authorized = bool(
        causal_margin_reproducible
        and posterior_count < protocol.REQUIRED_SEED_PASSES
    )

    if headroom_count < protocol.REQUIRED_SEED_PASSES:
        diagnosis = "fresh_schedule_policy_bank_headroom_not_reproducible"
    elif delay_count < protocol.REQUIRED_SEED_PASSES:
        diagnosis = "four_step_causal_delay_consumes_policy_bank_margin"
    elif posterior_count < protocol.REQUIRED_SEED_PASSES:
        diagnosis = "frozen_v5_estimator_transfer_is_limiting"
    else:
        diagnosis = "frozen_v5_causal_safe_utility_confirmed"

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
        "comparisons": comparisons,
        "headroom_seed_count": int(headroom_count),
        "delay_4_pass_seed_count": int(delay_count),
        "posterior_pass_seed_count": int(posterior_count),
        "causal_margin_reproducible": causal_margin_reproducible,
        "frozen_estimator_confirmed": frozen_estimator_confirmed,
        "estimator_retraining_authorized": estimator_retraining_authorized,
        "diagnosis": diagnosis,
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Frozen-estimator transfer result",
        "",
        "The v12 policy banks, robust-inclusive utility maps, and v5 estimator are frozen. This audit performs no training.",
        "",
        "| Seed | Robust | Safe oracle | Delay-4 oracle | Frozen-v5 MAP | Oracle gain | Delay recovery | MAP gain | MAP recovery | MAP accuracy | Pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for seed in protocol.TRAINING_SEEDS:
        row = payload["seed_metrics"][str(seed)]
        arms = row["arms"]
        lines.append(
            f"| {seed} | {arms['robust_sac']['mean']:.1f} | "
            f"{arms['true_mode_safe_utility']['mean']:.1f} | "
            f"{arms['delayed_oracle_4_safe_utility']['mean']:.1f} | "
            f"{arms[protocol.PRIMARY_ARM]['mean']:.1f} | "
            f"{_pct(row['relative_safe_oracle_headroom'])} | "
            f"{_pct(row['delay_4_recovery'])} | "
            f"{_pct(row['posterior_relative_gain'])} | "
            f"{_pct(row['posterior_recovery'])} | "
            f"{_pct(arms[protocol.PRIMARY_ARM]['posterior_mode_accuracy'])} | "
            f"{'yes' if row['posterior_pass'] else 'no'} |"
        )
    lines.extend([
        "",
        f"Headroom seeds: **{payload['headroom_seed_count']}/5**. "
        f"Delay-4 passes: **{payload['delay_4_pass_seed_count']}/5**. "
        f"Frozen-estimator passes: **{payload['posterior_pass_seed_count']}/5**.",
        "",
        f"Diagnosis: **{payload['diagnosis']}**.",
        f"Frozen estimator confirmed: **{payload['frozen_estimator_confirmed']}**.",
        f"Estimator retraining authorized: **{payload['estimator_retraining_authorized']}**.",
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
        "V13 FROZEN ESTIMATOR ANALYSIS COMPLETE: "
        f"diagnosis={payload['diagnosis']} "
        f"confirmed={payload['frozen_estimator_confirmed']} "
        f"retraining={payload['estimator_retraining_authorized']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
