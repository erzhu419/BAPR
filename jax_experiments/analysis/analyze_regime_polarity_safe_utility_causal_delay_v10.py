"""Aggregate the frozen v10 safe-utility causal-delay diagnostic."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_safe_utility_causal_delay_v10 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_safe_utility_causal_delay_audit_v10 as audit,
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
        "delayed_action_fraction": float(np.mean([
            row["delayed_action_fraction"] for row in rows
        ])),
    }


def _retention(value: float, robust: float, oracle: float) -> float | None:
    headroom = oracle - robust
    return (value - robust) / headroom if headroom > 0.0 else None


def _seed_metrics(seed: int) -> dict[str, Any]:
    audit.validate_audit(seed)
    arms = {
        arm: _arm_metrics(seed, arm) for arm in protocol.ARMS
    }
    robust = arms["robust_sac"]["mean"]
    oracle = arms["true_mode_safe_utility"]["mean"]
    posterior = arms["posterior_map_safe_utility"]["mean"]
    relative_headroom = (
        (oracle - robust) / abs(robust) if robust != 0.0 else float("-inf")
    )
    headroom_available = bool(relative_headroom >= protocol.MIN_HEADROOM_GAIN)
    delay_retention = {
        str(delay): {
            "stale": _retention(
                arms[f"stale_delay_{delay}"]["mean"], robust, oracle),
            "robust_handoff": _retention(
                arms[f"robust_handoff_{delay}"]["mean"], robust, oracle),
        }
        for delay in protocol.DELAYS
    }
    posterior_recovery = _retention(posterior, robust, oracle)
    best_delay_4 = max(
        value for value in delay_retention["4"].values()
        if value is not None
    ) if oracle > robust else None
    return {
        "arms": arms,
        "relative_safe_oracle_headroom": relative_headroom,
        "headroom_available": headroom_available,
        "posterior_recovery": posterior_recovery,
        "delay_retention": delay_retention,
        "best_delay_4_retention": best_delay_4,
        "delay_4_retains_headroom": bool(
            headroom_available
            and best_delay_4 is not None
            and best_delay_4 >= protocol.MIN_CAUSAL_RETENTION
        ),
        "posterior_recovers_headroom": bool(
            headroom_available
            and posterior_recovery is not None
            and posterior_recovery >= protocol.MIN_CAUSAL_RETENTION
        ),
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
    headroom_seeds = [
        row for row in seeds.values() if row["headroom_available"]
    ]
    headroom_count = len(headroom_seeds)
    delay_4_count = sum(
        row["delay_4_retains_headroom"] for row in headroom_seeds)
    posterior_count = sum(
        row["posterior_recovers_headroom"] for row in headroom_seeds)
    causal_margin_reproducible = bool(
        headroom_count >= protocol.MIN_REPRODUCIBLE_SEEDS
        and delay_4_count >= protocol.MIN_REPRODUCIBLE_SEEDS
    )
    estimator_retraining_authorized = bool(
        causal_margin_reproducible
        and posterior_count < protocol.MIN_REPRODUCIBLE_SEEDS
    )

    if headroom_count < protocol.MIN_REPRODUCIBLE_SEEDS:
        diagnosis = "policy_bank_headroom_not_reproducible"
    elif delay_4_count < protocol.MIN_REPRODUCIBLE_SEEDS:
        diagnosis = "causal_delay_consumes_safe_oracle_margin"
    elif posterior_count < protocol.MIN_REPRODUCIBLE_SEEDS:
        diagnosis = "posterior_or_switch_local_control_is_limiting"
    else:
        diagnosis = "causal_safe_utility_supported"

    comparisons = {}
    robust_values = [
        row["arms"]["robust_sac"]["mean"] for row in seeds.values()
    ]
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
        "headroom_seed_count": headroom_count,
        "delay_4_retention_seed_count": delay_4_count,
        "posterior_recovery_seed_count": posterior_count,
        "causal_margin_reproducible": causal_margin_reproducible,
        "estimator_retraining_authorized": estimator_retraining_authorized,
        "diagnosis": diagnosis,
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Safe-utility causal-delay diagnostic",
        "",
        "All controllers, utility maps, and estimator parameters are frozen from v9. No model is trained in this diagnostic.",
        "",
        "| Seed | Robust | Zero-delay safe oracle | Posterior MAP | Oracle gain | MAP recovery | Stale d1/d2/d4/d8 | Robust handoff d1/d2/d4/d8 |",
        "|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for seed in protocol.TRAINING_SEEDS:
        row = payload["seed_metrics"][str(seed)]
        arms = row["arms"]
        stale = "/".join(
            _pct(row["delay_retention"][str(delay)]["stale"])
            for delay in protocol.DELAYS
        )
        handoff = "/".join(
            _pct(row["delay_retention"][str(delay)]["robust_handoff"])
            for delay in protocol.DELAYS
        )
        lines.append(
            f"| {seed} | {arms['robust_sac']['mean']:.1f} | "
            f"{arms['true_mode_safe_utility']['mean']:.1f} | "
            f"{arms['posterior_map_safe_utility']['mean']:.1f} | "
            f"{100.0 * row['relative_safe_oracle_headroom']:.1f}% | "
            f"{_pct(row['posterior_recovery'])} | {stale} | {handoff} |"
        )
    lines.extend([
        "",
        f"Headroom seeds: **{payload['headroom_seed_count']}/5**. "
        f"Delay-4 retention seeds: **{payload['delay_4_retention_seed_count']}/5**. "
        f"Posterior recovery seeds: **{payload['posterior_recovery_seed_count']}/5**.",
        "",
        f"Diagnosis: **{payload['diagnosis']}**.",
        f"Estimator retraining authorized: **{payload['estimator_retraining_authorized']}**.",
        "",
        "A stale-delay arm keeps the previous mapped controller after each regime onset. A robust-handoff arm uses robust SAC for the same privileged delay, then switches to the correct mapped controller.",
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
        "V10 CAUSAL-DELAY ANALYSIS COMPLETE: "
        f"diagnosis={payload['diagnosis']} "
        f"estimator_retraining={payload['estimator_retraining_authorized']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
