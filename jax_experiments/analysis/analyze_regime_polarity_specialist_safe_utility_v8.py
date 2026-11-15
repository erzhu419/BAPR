"""Analyze robust-inclusive utility routing on fresh policy banks."""
from __future__ import annotations

import math
import statistics

from jax_experiments.analysis import (
    regime_polarity_specialist_safe_utility_v8 as protocol,
)
from jax_experiments.analysis.run_regime_polarity_specialist_safe_utility_audit_v8 import (
    validate_audit,
)


def _mean(values) -> float:
    return float(statistics.fmean(values))


def _paired_interval(differences: list[float]) -> list[float]:
    mean = _mean(differences)
    if len(differences) < 2:
        return [mean, mean]
    standard_error = statistics.stdev(differences) / math.sqrt(len(differences))
    radius = 2.7764451051977987 * standard_error
    return [float(mean - radius), float(mean + radius)]


def _seed_metrics(seed: int) -> dict:
    validate_audit(seed)
    calibration = protocol.read_json(protocol.calibration_result(seed))
    events = [
        protocol.read_json(protocol.event_result(seed, event_seed))
        for event_seed in protocol.HOLDOUT_EVENT_SEEDS
    ]
    arms = {}
    for arm in protocol.ARMS:
        rows = [event["switching"][arm] for event in events]
        arms[arm] = {
            "mean": _mean(row["return_mean"] for row in rows),
            "terminated_rate": _mean(
                row["terminated_rate"] for row in rows),
            "event_returns": {
                str(event_seed): float(row["return_mean"])
                for event_seed, row in zip(
                    protocol.HOLDOUT_EVENT_SEEDS, rows)
            },
            "adaptive_mode_accuracy": _mean(
                row["adaptive_mode_accuracy"]
                for row in rows
                if row["adaptive_mode_accuracy"] is not None
            ) if any(
                row["adaptive_mode_accuracy"] is not None for row in rows
            ) else None,
            "fallback_action_fraction": _mean(
                row["fallback_action_fraction"] for row in rows),
        }

    robust = arms["robust_sac"]
    safe_oracle = arms["true_mode_safe_utility"]
    headroom_gain = (
        (safe_oracle["mean"] - robust["mean"]) / abs(robust["mean"])
        if robust["mean"] != 0.0 else float("-inf")
    )
    headroom_available = headroom_gain >= protocol.MIN_HEADROOM_GAIN
    for row in arms.values():
        row["relative_gain"] = (
            (row["mean"] - robust["mean"]) / abs(robust["mean"])
            if robust["mean"] != 0.0 else float("-inf")
        )
        row["oracle_recovery"] = (
            (row["mean"] - robust["mean"])
            / (safe_oracle["mean"] - robust["mean"])
            if headroom_available else None
        )
        row["event_wins"] = sum(
            row["event_returns"][str(event_seed)]
            > robust["event_returns"][str(event_seed)]
            for event_seed in protocol.HOLDOUT_EVENT_SEEDS
        )
        row["event_no_regression"] = sum(
            row["event_returns"][str(event_seed)]
            >= (1.0 - protocol.MAX_NO_HEADROOM_REGRESSION)
            * robust["event_returns"][str(event_seed)]
            for event_seed in protocol.HOLDOUT_EVENT_SEEDS
        )

    def passes(row: dict) -> bool:
        if row["terminated_rate"] != 0.0:
            return False
        if headroom_available:
            return bool(
                row["relative_gain"] >= protocol.MIN_PRIMARY_GAIN
                and row["oracle_recovery"] is not None
                and row["oracle_recovery"] >= protocol.MIN_ORACLE_RECOVERY
                and row["event_wins"] == len(protocol.HOLDOUT_EVENT_SEEDS)
            )
        return bool(
            row["relative_gain"] >= -protocol.MAX_NO_HEADROOM_REGRESSION
            and row["event_no_regression"]
            == len(protocol.HOLDOUT_EVENT_SEEDS)
        )

    return {
        "utility_map": {
            mode: row["controller"]
            for mode, row in calibration["utility_map"].items()
        },
        "calibration_origin": calibration["origin"],
        "safe_oracle_headroom_gain": headroom_gain,
        "headroom_available": headroom_available,
        "arms": arms,
        "arm_pass": {
            arm: passes(row) for arm, row in arms.items()
        },
    }


def analyze() -> dict:
    seeds = {
        str(seed): _seed_metrics(seed)
        for seed in protocol.TRAINING_SEEDS
    }
    primary_pass = {
        str(seed): seeds[str(seed)]["arm_pass"][protocol.PRIMARY_ARM]
        for seed in protocol.TRAINING_SEEDS
    }
    map_pass = {
        str(seed): seeds[str(seed)]["arm_pass"][
            "posterior_map_safe_utility"]
        for seed in protocol.TRAINING_SEEDS
    }
    primary_rows = [
        seeds[str(seed)]["arms"][protocol.PRIMARY_ARM]
        for seed in protocol.TRAINING_SEEDS
    ]
    map_rows = [
        seeds[str(seed)]["arms"]["posterior_map_safe_utility"]
        for seed in protocol.TRAINING_SEEDS
    ]
    robust_rows = [
        seeds[str(seed)]["arms"]["robust_sac"]
        for seed in protocol.TRAINING_SEEDS
    ]
    safe_oracle_rows = [
        seeds[str(seed)]["arms"]["true_mode_safe_utility"]
        for seed in protocol.TRAINING_SEEDS
    ]
    differences = [
        primary["mean"] - robust["mean"]
        for primary, robust in zip(primary_rows, robust_rows)
    ]
    map_differences = [
        adaptive["mean"] - robust["mean"]
        for adaptive, robust in zip(map_rows, robust_rows)
    ]
    available_recovery = [
        row["oracle_recovery"]
        for row in primary_rows
        if row["oracle_recovery"] is not None
    ]
    aggregate = {
        "primary_mean": _mean(row["mean"] for row in primary_rows),
        "robust_mean": _mean(row["mean"] for row in robust_rows),
        "safe_oracle_mean": _mean(row["mean"] for row in safe_oracle_rows),
        "mean_relative_gain": _mean(
            row["relative_gain"] for row in primary_rows),
        "headroom_banks": len(available_recovery),
        "mean_oracle_recovery_headroom_banks": (
            _mean(available_recovery) if available_recovery else None
        ),
        "seed_wins": sum(value > 0.0 for value in differences),
        "event_wins": sum(row["event_wins"] for row in primary_rows),
        "paired_difference_mean": _mean(differences),
        "paired_difference_ci95": _paired_interval(differences),
        "max_terminated_rate": max(
            row["terminated_rate"] for row in primary_rows),
        "map_mean": _mean(row["mean"] for row in map_rows),
        "map_mean_relative_gain": _mean(
            row["relative_gain"] for row in map_rows),
        "map_mean_oracle_recovery_headroom_banks": _mean(
            row["oracle_recovery"]
            for row in map_rows
            if row["oracle_recovery"] is not None
        ),
        "map_seed_wins": sum(value > 0.0 for value in map_differences),
        "map_event_wins": sum(row["event_wins"] for row in map_rows),
        "map_paired_difference_mean": _mean(map_differences),
        "map_paired_difference_ci95": _paired_interval(map_differences),
        "map_max_terminated_rate": max(
            row["terminated_rate"] for row in map_rows),
    }
    all_primary_pass = all(primary_pass.values())
    all_map_pass = all(map_pass.values())
    if all_primary_pass:
        conclusion = (
            "robust-inclusive utility routing transfers across heterogeneous "
            "fresh policy banks under the headroom-aware no-regression gate"
        )
        next_step = (
            "freeze the utility rule and run a final independent confirmation "
            "against corrected ESCP and RE-SAC"
        )
    elif all_map_pass:
        conclusion = (
            "the simpler posterior-MAP safe-utility router passes every fresh "
            "bank; confirm-3 is an unnecessary source of transient loss"
        )
        next_step = (
            "freeze posterior-MAP safe utility as the candidate and run a fully "
            "new policy-bank confirmation with explicit training-cost accounting"
        )
    else:
        conclusion = (
            "existing robust and specialist checkpoints cannot provide a stable "
            "safe utility composition on fresh banks"
        )
        next_step = (
            "train bounded mode options from immutable robust actors with an "
            "explicit stationary no-regression loss"
        )
    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "training_seeds": list(protocol.TRAINING_SEEDS),
        "calibration_event_seeds": list(protocol.CALIBRATION_EVENT_SEEDS),
        "holdout_event_seeds": list(protocol.HOLDOUT_EVENT_SEEDS),
        "primary_arm": protocol.PRIMARY_ARM,
        "source_seed_metrics": seeds,
        "source_seed_primary_pass": primary_pass,
        "source_seed_map_pass": map_pass,
        "aggregate": aggregate,
        "primary_pass": all_primary_pass,
        "map_pass": all_map_pass,
        "conclusion": conclusion,
        "next_step": next_step,
    }


def _map_text(mapping: dict[str, str]) -> str:
    return "[" + ",".join(
        "R" if mapping[str(mode)] == "robust_sac" else f"S{mode}"
        for mode in protocol.MODES
    ) + "]"


def report(payload: dict) -> str:
    lines = [
        "# Robust-inclusive specialist utility on fresh policy banks",
        "",
        "Each per-mode map is frozen on three stationary calibration streams. "
        "A matching specialist is enabled only when it beats robust by at least "
        "5%, wins all three calibration streams, and has zero termination. All "
        "reported switching results use three untouched event streams.",
        "",
        "Banks with at least 10% safe-oracle headroom must gain at least 10%, "
        "recover at least 70% of that headroom, and win all holdout streams. "
        "Banks without such headroom must remain within 5% of robust on every "
        "holdout stream.",
        "",
        "| Seed | Map | Robust | Safe oracle | MAP utility | Confirm-3 utility | "
        "Confirm gain | Confirm recovery | Headroom | MAP pass | Confirm pass |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|",
    ]
    for seed in protocol.TRAINING_SEEDS:
        row = payload["source_seed_metrics"][str(seed)]
        arms = row["arms"]
        primary = arms[protocol.PRIMARY_ARM]
        recovery = primary["oracle_recovery"]
        lines.append(
            f"| {seed} | `{_map_text(row['utility_map'])}` | "
            f"{arms['robust_sac']['mean']:.1f} | "
            f"{arms['true_mode_safe_utility']['mean']:.1f} | "
            f"{arms['posterior_map_safe_utility']['mean']:.1f} | "
            f"{primary['mean']:.1f} | {100*primary['relative_gain']:.1f}% | "
            f"{'n/a' if recovery is None else f'{100*recovery:.1f}%'} | "
            f"{row['headroom_available']} | "
            f"{payload['source_seed_map_pass'][str(seed)]} | "
            f"{payload['source_seed_primary_pass'][str(seed)]} |"
        )
    aggregate = payload["aggregate"]
    recovery = aggregate["mean_oracle_recovery_headroom_banks"]
    lines += [
        "",
        f"Mean confirm-3 return: **{aggregate['primary_mean']:.1f}**; robust: "
        f"**{aggregate['robust_mean']:.1f}**; safe oracle: "
        f"**{aggregate['safe_oracle_mean']:.1f}**.",
        "",
        f"Mean gain: **{100*aggregate['mean_relative_gain']:.1f}%**; seed wins: "
        f"**{aggregate['seed_wins']}/{len(protocol.TRAINING_SEEDS)}**; event "
        f"wins: **{aggregate['event_wins']}/"
        f"{len(protocol.TRAINING_SEEDS)*len(protocol.HOLDOUT_EVENT_SEEDS)}**; "
        "mean headroom recovery: "
        f"**{'n/a' if recovery is None else f'{100*recovery:.1f}%'}**.",
        "",
        f"Mean MAP return: **{aggregate['map_mean']:.1f}**; mean gain: "
        f"**{100*aggregate['map_mean_relative_gain']:.1f}%**; mean recovery on "
        "headroom banks: "
        f"**{100*aggregate['map_mean_oracle_recovery_headroom_banks']:.1f}%**; "
        f"seed wins: **{aggregate['map_seed_wins']}/"
        f"{len(protocol.TRAINING_SEEDS)}**; event wins: "
        f"**{aggregate['map_event_wins']}/"
        f"{len(protocol.TRAINING_SEEDS)*len(protocol.HOLDOUT_EVENT_SEEDS)}**.",
        "",
        f"MAP composition pass: **{payload['map_pass']}**; confirm-3 primary "
        f"pass: **{payload['primary_pass']}**.",
        "",
        f"Conclusion: {payload['conclusion']}.",
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
        "SAFE UTILITY ANALYSIS COMPLETE: "
        f"map={payload['map_pass']} primary={payload['primary_pass']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
