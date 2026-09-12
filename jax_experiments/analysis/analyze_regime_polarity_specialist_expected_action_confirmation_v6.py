"""Analyze fresh policy-bank confirmation for frozen expected-action v5."""
from __future__ import annotations

import math
import statistics

from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_confirmation_v6 as protocol,
)
from jax_experiments.analysis.run_regime_polarity_specialist_expected_action_confirmation_audit_v6 import (
    validate_audit,
)


def _mean(values) -> float:
    return float(statistics.fmean(values))


def _seed_metrics(seed: int) -> dict:
    validate_audit(seed)
    events = [
        protocol.read_json(protocol.event_result(seed, event_seed))
        for event_seed in protocol.EVENT_SEEDS
    ]
    arms = {}
    for arm in protocol.ARMS:
        rows = [event["switching"][arm] for event in events]
        arms[arm] = {
            "mean": _mean(row["return_mean"] for row in rows),
            "event_returns": {
                str(event_seed): float(row["return_mean"])
                for event_seed, row in zip(protocol.EVENT_SEEDS, rows)
            },
            "terminated_rate": _mean(row["terminated_rate"] for row in rows),
            "fallback_action_fraction": _mean(
                row["fallback_action_fraction"] for row in rows),
            "adaptive_mode_accuracy": _mean(
                row["adaptive_mode_accuracy"]
                for row in rows
                if row["adaptive_mode_accuracy"] is not None
            ) if any(
                row["adaptive_mode_accuracy"] is not None for row in rows
            ) else None,
        }
        option_rows = [
            row for row in rows if "option_switch_count_mean" in row]
        if option_rows:
            arms[arm]["option_switch_count_mean"] = _mean(
                row["option_switch_count_mean"] for row in option_rows)

    robust = arms["robust_sac"]
    oracle = arms["dynamic_oracle"]
    for row in arms.values():
        row["relative_gain"] = (
            (row["mean"] - robust["mean"]) / abs(robust["mean"])
            if robust["mean"] != 0.0 else float("-inf")
        )
        row["headroom_available"] = bool(oracle["mean"] > robust["mean"])
        row["oracle_recovery"] = (
            (row["mean"] - robust["mean"])
            / (oracle["mean"] - robust["mean"])
            if row["headroom_available"] else None
        )
        row["event_wins_vs_robust"] = sum(
            row["event_returns"][str(event_seed)]
            > robust["event_returns"][str(event_seed)]
            for event_seed in protocol.EVENT_SEEDS
        )
    return {"arms": arms}


def _passes(row: dict) -> bool:
    return bool(
        row["relative_gain"] >= protocol.MIN_GAIN
        and row["oracle_recovery"] is not None
        and row["oracle_recovery"] >= protocol.MIN_ORACLE_RECOVERY
        and row["mean"] >= protocol.MIN_SWITCHING_RETURN
        and row["terminated_rate"] == 0.0
        and row["event_wins_vs_robust"] == len(protocol.EVENT_SEEDS)
    )


def _paired_interval(differences: list[float]) -> list[float]:
    mean = _mean(differences)
    if len(differences) < 2:
        return [mean, mean]
    standard_error = statistics.stdev(differences) / math.sqrt(len(differences))
    # Five frozen seeds imply four degrees of freedom.
    radius = 2.7764451051977987 * standard_error
    return [float(mean - radius), float(mean + radius)]


def analyze() -> dict:
    seeds = {
        str(seed): _seed_metrics(seed)
        for seed in protocol.TRAINING_SEEDS
    }
    seed_pass = {
        str(seed): _passes(
            seeds[str(seed)]["arms"][protocol.PRIMARY_ARM])
        for seed in protocol.TRAINING_SEEDS
    }
    primary_rows = [
        seeds[str(seed)]["arms"][protocol.PRIMARY_ARM]
        for seed in protocol.TRAINING_SEEDS
    ]
    robust_rows = [
        seeds[str(seed)]["arms"]["robust_sac"]
        for seed in protocol.TRAINING_SEEDS
    ]
    oracle_rows = [
        seeds[str(seed)]["arms"]["dynamic_oracle"]
        for seed in protocol.TRAINING_SEEDS
    ]
    differences = [
        primary["mean"] - robust["mean"]
        for primary, robust in zip(primary_rows, robust_rows)
    ]
    available_recovery = [
        row["oracle_recovery"]
        for row in primary_rows
        if row["oracle_recovery"] is not None
    ]
    aggregate = {
        "primary_mean": _mean(row["mean"] for row in primary_rows),
        "robust_mean": _mean(row["mean"] for row in robust_rows),
        "oracle_mean": _mean(row["mean"] for row in oracle_rows),
        "mean_relative_gain": _mean(
            row["relative_gain"] for row in primary_rows),
        "headroom_available_seeds": len(available_recovery),
        "mean_oracle_recovery_available_seeds": _mean(available_recovery),
        "seed_wins": sum(value > 0.0 for value in differences),
        "event_wins": sum(
            row["event_wins_vs_robust"] for row in primary_rows),
        "paired_difference_mean": _mean(differences),
        "paired_difference_ci95": _paired_interval(differences),
        "mean_mode_accuracy": _mean(
            row["adaptive_mode_accuracy"] for row in primary_rows),
        "mean_robust_action_fraction": _mean(
            row["fallback_action_fraction"] for row in primary_rows),
        "max_terminated_rate": max(
            row["terminated_rate"] for row in primary_rows),
    }
    primary_pass = all(seed_pass.values())
    if primary_pass:
        diagnosis = (
            "frozen v5 transfers across all five fresh policy banks"
        )
        next_step = (
            "freeze the complete router and compare it with corrected "
            "ESCP and RE-SAC under the same fresh seeds and compute accounting"
        )
    elif aggregate["seed_wins"] >= 4:
        diagnosis = (
            "v5 is directionally positive but does not meet the strict "
            "per-bank confirmation gate"
        )
        next_step = (
            "report heterogeneous controller-bank transfer and inspect only "
            "the failed banks before authorizing new training"
        )
    else:
        diagnosis = (
            "v5 does not reproduce on fresh independently trained policy banks"
        )
        next_step = (
            "stop estimator-only development and move to switch-matched control"
        )
    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "training_seeds": list(protocol.TRAINING_SEEDS),
        "event_seeds": list(protocol.EVENT_SEEDS),
        "frozen_estimator": protocol.estimator_records(),
        "primary_arm": protocol.PRIMARY_ARM,
        "source_seed_metrics": seeds,
        "source_seed_pass": seed_pass,
        "aggregate": aggregate,
        "primary_pass": primary_pass,
        "diagnosis": diagnosis,
        "next_step": next_step,
    }


def report(payload: dict) -> str:
    lines = [
        "# Fresh policy-bank confirmation for frozen v5",
        "",
        "This graph freezes the v5 expected-action estimator, its selected "
        "filter, and the confirm-3 router. Only five new robust/specialist "
        "policy banks are trained. The controller seeds are "
        f"`{', '.join(map(str, protocol.TRAINING_SEEDS))}` and the event seeds "
        f"are `{', '.join(map(str, protocol.EVENT_SEEDS))}`.",
        "",
        "A source seed passes only when confirm-3 beats robust by at least 10%, "
        "recovers at least 70% of dynamic-oracle headroom, reaches return 2200, "
        "wins all three event streams, and has zero termination. The primary "
        "confirmation requires all five source seeds to pass.",
        "",
        "| Seed | Robust | Oracle | Frozen v5 | Gain | Recovery | Accuracy | "
        "Event wins | Pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for seed in protocol.TRAINING_SEEDS:
        rows = payload["source_seed_metrics"][str(seed)]["arms"]
        primary = rows[protocol.PRIMARY_ARM]
        recovery = primary["oracle_recovery"]
        recovery_text = (
            "n/a" if recovery is None else f"{100*recovery:.1f}%")
        lines.append(
            f"| {seed} | {rows['robust_sac']['mean']:.1f} | "
            f"{rows['dynamic_oracle']['mean']:.1f} | {primary['mean']:.1f} | "
            f"{100*primary['relative_gain']:.1f}% | "
            f"{recovery_text} | "
            f"{100*primary['adaptive_mode_accuracy']:.2f}% | "
            f"{primary['event_wins_vs_robust']}/{len(protocol.EVENT_SEEDS)} | "
            f"{payload['source_seed_pass'][str(seed)]} |"
        )
    aggregate = payload["aggregate"]
    lines += [
        "",
        f"Mean frozen-v5 return: **{aggregate['primary_mean']:.1f}**; robust: "
        f"**{aggregate['robust_mean']:.1f}**; oracle: "
        f"**{aggregate['oracle_mean']:.1f}**.",
        "",
        f"Mean relative gain: **{100*aggregate['mean_relative_gain']:.1f}%**; "
        "mean oracle recovery on banks with positive oracle headroom: "
        f"**{100*aggregate['mean_oracle_recovery_available_seeds']:.1f}%** "
        f"({aggregate['headroom_available_seeds']}/"
        f"{len(protocol.TRAINING_SEEDS)} banks); "
        f"seed wins: **{aggregate['seed_wins']}/{len(protocol.TRAINING_SEEDS)}**; "
        f"event wins: **{aggregate['event_wins']}/"
        f"{len(protocol.TRAINING_SEEDS)*len(protocol.EVENT_SEEDS)}**.",
        "",
        f"Primary pass: **{payload['primary_pass']}**",
        "",
        f"Diagnosis: {payload['diagnosis']}.",
        "",
        f"Decision: {payload['next_step']}.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    payload = analyze()
    text = report(payload)
    protocol.write_json_atomic(protocol.analysis_json(), payload)
    protocol.write_text_atomic(protocol.analysis_markdown(), text)
    protocol.write_text_atomic(protocol.REPORT, text)
    print(
        "FRESH POLICY-BANK CONFIRMATION COMPLETE: "
        f"primary={payload['primary_pass']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
