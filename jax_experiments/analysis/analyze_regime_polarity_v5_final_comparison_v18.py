"""Aggregate the preregistered v18 final BAPR comparison."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_v5_final_comparison_v18 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_v5_final_comparison_audit_v18 as audit,
)


def _mean(values) -> float:
    return float(np.mean([float(value) for value in values]))


def _mean_ci95(values: list[float]) -> list[float]:
    values = [float(value) for value in values]
    mean = float(np.mean(values))
    if len(values) < 2:
        return [mean, mean]
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(
        len(values), 1.96)
    half = critical * float(np.std(values, ddof=1)) / math.sqrt(len(values))
    return [mean - half, mean + half]


def _switching_metrics(payload: dict[str, Any], arm: str) -> dict[str, Any]:
    rows = [
        payload["switching_holdout"][str(event_seed)][arm]
        for event_seed in protocol.SWITCHING_EVENT_SEEDS
    ]
    return {
        "mean": _mean(row["return_mean"] for row in rows),
        "terminated_rate": _mean(row["terminated_rate"] for row in rows),
        "event_returns": {
            str(event_seed): float(row["return_mean"])
            for event_seed, row in zip(protocol.SWITCHING_EVENT_SEEDS, rows)
        },
        "routing_mode_accuracy": (
            _mean(row["routing_mode_accuracy"] for row in rows)
            if all(row.get("routing_mode_accuracy") is not None for row in rows)
            else None
        ),
    }


def _stationary_metrics(payload: dict[str, Any], arm: str) -> dict[str, Any]:
    rows = [
        payload["stationary_holdout"][str(event_seed)][arm][str(mode)]
        for event_seed in protocol.STATIONARY_HOLDOUT_EVENT_SEEDS
        for mode in protocol.MODES
    ]
    return {
        "mean": _mean(row["return_mean"] for row in rows),
        "terminated_rate": _mean(row["terminated_rate"] for row in rows),
    }


def _relative_gain(value: float, baseline: float) -> float:
    if baseline == 0.0:
        return float("-inf")
    return float((value - baseline) / abs(baseline))


def _oracle_recovery(primary: float, robust: float, oracle: float) -> float | None:
    headroom = oracle - robust
    return float((primary - robust) / headroom) if headroom > 0.0 else None


def _seed_metrics(seed: int) -> dict[str, Any]:
    audit.validate_audit(seed)
    payload = protocol.read_json(protocol.audit_result(seed))
    switching = {
        arm: _switching_metrics(payload, arm)
        for arm in protocol.SWITCHING_ARMS
    }
    stationary = {
        arm: _stationary_metrics(payload, arm)
        for arm in protocol.STATIONARY_ARMS
    }
    robust = switching["robust_sac"]
    oracle = switching["bapr_true_mode_oracle"]
    primary = switching[protocol.PRIMARY_ARM]
    oracle_headroom = _relative_gain(oracle["mean"], robust["mean"])
    recovery = _oracle_recovery(
        primary["mean"], robust["mean"], oracle["mean"])
    recovery_pass = bool(
        oracle_headroom >= protocol.MIN_ORACLE_HEADROOM
        and recovery is not None
        and recovery >= protocol.MIN_ORACLE_RECOVERY
        and primary["terminated_rate"] <= robust["terminated_rate"])

    stationary_comparators = (
        "robust_sac", "escp_recurrent", "resac_b0", "sac5_best_static")
    strongest_name = max(
        stationary_comparators,
        key=lambda arm: stationary[arm]["mean"])
    strongest = stationary[strongest_name]
    primary_stationary = stationary[protocol.PRIMARY_ARM]
    stationary_retention = (
        primary_stationary["mean"] / strongest["mean"]
        if strongest["mean"] > 0.0 else float("-inf"))
    stationary_pass = bool(
        stationary_retention >= protocol.MIN_STATIONARY_RETENTION
        and primary_stationary["terminated_rate"]
        <= strongest["terminated_rate"])
    return {
        "calibration": {
            "bapr_utility_map": payload["calibration"]["bapr_utility_map"],
            "sac5_best_static": payload["calibration"]["sac5_best_static"],
            "sac5_mode_map": payload["calibration"]["sac5_mode_map"],
        },
        "switching": switching,
        "stationary": stationary,
        "oracle_relative_headroom": oracle_headroom,
        "oracle_recovery": recovery,
        "oracle_recovery_pass": recovery_pass,
        "stationary_strongest_single": strongest_name,
        "stationary_retention": float(stationary_retention),
        "stationary_retention_pass": stationary_pass,
    }


def _comparison(
    seeds: dict[str, Any], comparator: str,
) -> dict[str, Any]:
    differences = []
    seed_wins = 0
    event_wins = 0
    primary_termination = []
    comparator_termination = []
    for seed in protocol.TRAINING_SEEDS:
        row = seeds[str(seed)]["switching"]
        primary = row[protocol.PRIMARY_ARM]
        baseline = row[comparator]
        difference = primary["mean"] - baseline["mean"]
        differences.append(float(difference))
        seed_wins += int(difference > 0.0)
        event_wins += sum(
            primary["event_returns"][str(event_seed)]
            > baseline["event_returns"][str(event_seed)]
            for event_seed in protocol.SWITCHING_EVENT_SEEDS)
        primary_termination.append(primary["terminated_rate"])
        comparator_termination.append(baseline["terminated_rate"])
    mean_difference = _mean(differences)
    ci95 = _mean_ci95(differences)
    no_termination_increase = bool(
        _mean(primary_termination) <= _mean(comparator_termination))
    passed = bool(
        seed_wins >= protocol.REQUIRED_SEED_WINS
        and event_wins >= protocol.REQUIRED_EVENT_WINS
        and mean_difference > 0.0
        and ci95[0] > 0.0
        and no_termination_increase)
    return {
        "comparator": comparator,
        "paired_differences": differences,
        "paired_difference_mean": mean_difference,
        "paired_difference_ci95": ci95,
        "seed_wins": int(seed_wins),
        "event_wins": int(event_wins),
        "primary_terminated_rate": _mean(primary_termination),
        "comparator_terminated_rate": _mean(comparator_termination),
        "no_termination_increase": no_termination_increase,
        "pass": passed,
    }


def analyze() -> dict[str, Any]:
    protocol.validate_registration()
    seeds = {
        str(seed): _seed_metrics(seed) for seed in protocol.TRAINING_SEEDS
    }
    comparators = (
        *protocol.STANDARD_BASELINES,
        protocol.EQUAL_POLICY_BUDGET_BASELINE,
    )
    comparisons = {
        comparator: _comparison(seeds, comparator)
        for comparator in comparators
    }
    recovery_count = sum(
        row["oracle_recovery_pass"] for row in seeds.values())
    stationary_count = sum(
        row["stationary_retention_pass"] for row in seeds.values())
    recovery_pass = bool(
        recovery_count >= protocol.REQUIRED_RECOVERY_SEEDS)
    stationary_pass = bool(
        stationary_count >= protocol.REQUIRED_STATIONARY_SEEDS)
    standard_pass = bool(all(
        comparisons[name]["pass"] for name in protocol.STANDARD_BASELINES))
    equal_budget_pass = bool(
        comparisons[protocol.EQUAL_POLICY_BUDGET_BASELINE]["pass"])
    final_pass = bool(
        standard_pass and equal_budget_pass
        and recovery_pass and stationary_pass)
    limited_claim = bool(
        standard_pass and recovery_pass and stationary_pass
        and not equal_budget_pass)
    if final_pass:
        diagnosis = "frozen_v5_bapr_supported_at_equal_policy_budget"
    elif limited_claim:
        diagnosis = "adaptation_supported_but_not_equal_policy_budget_advantage"
    elif not standard_pass:
        diagnosis = "frozen_v5_bapr_fails_standard_baseline_gate"
    elif not recovery_pass:
        diagnosis = "frozen_v5_bapr_fails_oracle_recovery_gate"
    else:
        diagnosis = "frozen_v5_bapr_fails_stationary_retention_gate"
    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "training_seeds": list(protocol.TRAINING_SEEDS),
        "switching_event_seeds": list(protocol.SWITCHING_EVENT_SEEDS),
        "primary_arm": protocol.PRIMARY_ARM,
        "seed_metrics": seeds,
        "comparisons": comparisons,
        "oracle_recovery_pass_count": int(recovery_count),
        "stationary_retention_pass_count": int(stationary_count),
        "standard_baseline_gate_pass": standard_pass,
        "equal_policy_budget_gate_pass": equal_budget_pass,
        "oracle_recovery_gate_pass": recovery_pass,
        "stationary_retention_gate_pass": stationary_pass,
        "strong_final_algorithm_claim": final_pass,
        "limited_adaptation_claim": limited_claim,
        "diagnosis": diagnosis,
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# V18 frozen-v5 final equal-policy-budget comparison",
        "",
        "The v17 policy bank and v5 posterior were frozen before these event "
        "streams and baseline runs.",
        "",
        "| Seed | Robust | BAPR oracle | BAPR v5 | ESCP | RE-SAC | "
        "SAC5 causal | Recovery | Stationary retention |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in protocol.TRAINING_SEEDS:
        row = payload["seed_metrics"][str(seed)]
        arms = row["switching"]
        recovery = row["oracle_recovery"]
        lines.append(
            f"| {seed} | {arms['robust_sac']['mean']:.1f} | "
            f"{arms['bapr_true_mode_oracle']['mean']:.1f} | "
            f"{arms[protocol.PRIMARY_ARM]['mean']:.1f} | "
            f"{arms['escp_recurrent']['mean']:.1f} | "
            f"{arms['resac_b0']['mean']:.1f} | "
            f"{arms[protocol.EQUAL_POLICY_BUDGET_BASELINE]['mean']:.1f} | "
            f"{'n/a' if recovery is None else f'{100.0 * recovery:.1f}%'} | "
            f"{100.0 * row['stationary_retention']:.1f}% |"
        )
    lines.extend([
        "",
        "| Comparator | Paired difference | 95% CI | Seed wins | Event wins | "
        "Pass |",
        "|---|---:|---:|---:|---:|:---:|",
    ])
    for comparator, row in payload["comparisons"].items():
        lines.append(
            f"| {comparator} | {row['paired_difference_mean']:+.1f} | "
            f"[{row['paired_difference_ci95'][0]:+.1f}, "
            f"{row['paired_difference_ci95'][1]:+.1f}] | "
            f"{row['seed_wins']}/5 | {row['event_wins']}/15 | "
            f"{'yes' if row['pass'] else 'no'} |"
        )
    lines.extend([
        "",
        f"Oracle recovery: **{payload['oracle_recovery_pass_count']}/5**; "
        f"stationary retention: **{payload['stationary_retention_pass_count']}/5**.",
        f"Strong final algorithm claim: "
        f"**{payload['strong_final_algorithm_claim']}**.",
        f"Limited adaptation claim: **{payload['limited_adaptation_claim']}**.",
        f"Diagnosis: **{payload['diagnosis']}**.",
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
        "V18 FINAL COMPARISON COMPLETE: "
        f"diagnosis={payload['diagnosis']} "
        f"strong={payload['strong_final_algorithm_claim']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
