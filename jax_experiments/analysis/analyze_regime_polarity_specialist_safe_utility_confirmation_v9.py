"""Analyze the independent v9 MAP safe-utility confirmation."""
from __future__ import annotations

import math
import statistics

from jax_experiments.analysis import (
    regime_polarity_specialist_safe_utility_confirmation_v9 as protocol,
)
from jax_experiments.analysis.run_regime_polarity_specialist_safe_utility_confirmation_audit_v9 import (
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
            "adaptive_mode_accuracy": (
                _mean(
                    row["adaptive_mode_accuracy"] for row in rows
                    if row.get("adaptive_mode_accuracy") is not None
                )
                if any(
                    row.get("adaptive_mode_accuracy") is not None
                    for row in rows
                ) else None
            ),
        }

    baseline_stationary = {}
    for method in protocol.BASELINE_METHODS:
        values = [
            mode_row["return_mean"]
            for event in events
            for mode_row in event["baseline_stationary"][method]
        ]
        baseline_stationary[method] = _mean(values)

    robust = arms["robust_sac"]
    safe_oracle = arms["true_mode_safe_utility"]
    primary = arms[protocol.PRIMARY_ARM]
    headroom = safe_oracle["mean"] - robust["mean"]
    headroom_gain = (
        headroom / abs(robust["mean"])
        if robust["mean"] != 0.0 else float("-inf")
    )
    headroom_available = headroom_gain >= protocol.MIN_HEADROOM_GAIN
    primary_gain = (
        (primary["mean"] - robust["mean"]) / abs(robust["mean"])
        if robust["mean"] != 0.0 else float("-inf")
    )
    oracle_recovery = (
        (primary["mean"] - robust["mean"]) / headroom
        if headroom_available and headroom != 0.0 else None
    )
    event_wins = sum(
        primary["event_returns"][str(event_seed)]
        > robust["event_returns"][str(event_seed)]
        for event_seed in protocol.HOLDOUT_EVENT_SEEDS
    )
    event_no_regression = sum(
        primary["event_returns"][str(event_seed)]
        >= robust["event_returns"][str(event_seed)]
        - protocol.MAX_NO_HEADROOM_REGRESSION
        * abs(robust["event_returns"][str(event_seed)])
        for event_seed in protocol.HOLDOUT_EVENT_SEEDS
    )
    if primary["terminated_rate"] != 0.0:
        composition_pass = False
    elif headroom_available:
        composition_pass = bool(
            primary_gain >= protocol.MIN_PRIMARY_GAIN
            and oracle_recovery is not None
            and oracle_recovery >= protocol.MIN_ORACLE_RECOVERY
            and event_wins == len(protocol.HOLDOUT_EVENT_SEEDS)
        )
    else:
        composition_pass = bool(
            primary_gain >= -protocol.MAX_NO_HEADROOM_REGRESSION
            and event_no_regression == len(protocol.HOLDOUT_EVENT_SEEDS)
        )

    return {
        "utility_map": {
            mode: row["controller"]
            for mode, row in calibration["utility_map"].items()
        },
        "safe_oracle_headroom_gain": headroom_gain,
        "headroom_available": headroom_available,
        "primary_relative_gain": primary_gain,
        "primary_oracle_recovery": oracle_recovery,
        "primary_event_wins": event_wins,
        "primary_event_no_regression": event_no_regression,
        "composition_pass": composition_pass,
        "arms": arms,
        "baseline_stationary_mean": baseline_stationary,
    }


def _comparison(seeds: dict, comparator: str) -> dict:
    differences = []
    relative_gains = []
    event_wins = 0
    termination_gaps = []
    for seed in protocol.TRAINING_SEEDS:
        row = seeds[str(seed)]["arms"]
        primary = row[protocol.PRIMARY_ARM]
        baseline = row[comparator]
        difference = primary["mean"] - baseline["mean"]
        differences.append(difference)
        relative_gains.append(
            difference / abs(baseline["mean"])
            if baseline["mean"] != 0.0 else float("inf")
        )
        event_wins += sum(
            primary["event_returns"][str(event_seed)]
            > baseline["event_returns"][str(event_seed)]
            for event_seed in protocol.HOLDOUT_EVENT_SEEDS
        )
        termination_gaps.append(
            primary["terminated_rate"] - baseline["terminated_rate"])
    interval = _paired_interval(differences)
    seed_wins = sum(value > 0.0 for value in differences)
    confirmed = bool(
        interval[0] > 0.0
        and seed_wins >= protocol.MIN_BASELINE_SEED_WINS
        and max(termination_gaps) <= 0.0
    )
    return {
        "primary_mean": _mean(
            seeds[str(seed)]["arms"][protocol.PRIMARY_ARM]["mean"]
            for seed in protocol.TRAINING_SEEDS),
        "comparator_mean": _mean(
            seeds[str(seed)]["arms"][comparator]["mean"]
            for seed in protocol.TRAINING_SEEDS),
        "paired_difference_mean": _mean(differences),
        "paired_difference_ci95": interval,
        "mean_relative_gain": _mean(relative_gains),
        "seed_wins": seed_wins,
        "event_wins": event_wins,
        "max_termination_gap": max(termination_gaps),
        "confirmed": confirmed,
    }


def analyze() -> dict:
    protocol.validate_registration()
    seeds = {
        str(seed): _seed_metrics(seed) for seed in protocol.TRAINING_SEEDS
    }
    composition_pass = all(
        row["composition_pass"] for row in seeds.values())
    comparisons = {
        comparator: _comparison(seeds, comparator)
        for comparator in (
            "robust_sac", "escp_recurrent", "resac_b0")
    }
    comparative_pass = bool(
        composition_pass
        and all(row["confirmed"] for row in comparisons.values())
    )
    if comparative_pass:
        conclusion = (
            "the frozen posterior-MAP safe-utility bank independently passes "
            "its composition gate and outperforms all paired baselines"
        )
        next_step = (
            "freeze this implementation for the paper and expand only the "
            "environment and seed coverage required by the final benchmark"
        )
    elif composition_pass:
        conclusion = (
            "the frozen safe-utility composition transfers, but five fresh "
            "seeds do not establish superiority over every paired baseline"
        )
        next_step = (
            "report the policy-bank result with its five-controller cost and "
            "treat unsupported baseline-superiority claims as unresolved"
        )
    else:
        conclusion = (
            "the v8 safe-utility result does not transfer to every independent "
            "policy bank under the frozen gate"
        )
        next_step = (
            "stop expanding this policy-bank candidate and retain it only as a "
            "diagnostic upper-bound construction"
        )
    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "registration": protocol.file_record(protocol.REGISTRATION_PATH),
        "training_seeds": list(protocol.TRAINING_SEEDS),
        "calibration_event_seeds": list(protocol.CALIBRATION_EVENT_SEEDS),
        "holdout_event_seeds": list(protocol.HOLDOUT_EVENT_SEEDS),
        "primary_arm": protocol.PRIMARY_ARM,
        "seed_metrics": seeds,
        "composition_pass": composition_pass,
        "comparisons": comparisons,
        "comparative_pass": comparative_pass,
        "cost": {
            "per_controller_steps": protocol.FINAL_TOTAL_STEPS,
            "per_controller_updates": protocol.FINAL_UPDATE_COUNT,
            "bapr_policy_count": len(protocol.ROLES),
            "bapr_policy_bank_steps_per_seed": (
                len(protocol.ROLES) * protocol.FINAL_TOTAL_STEPS),
            "bapr_policy_bank_updates_per_seed": (
                len(protocol.ROLES) * protocol.FINAL_UPDATE_COUNT),
            "single_baseline_steps_per_seed": protocol.FINAL_TOTAL_STEPS,
            "single_baseline_updates_per_seed": protocol.FINAL_UPDATE_COUNT,
            "policy_training_cost_ratio": len(protocol.ROLES),
            "frozen_estimator_pretraining_excluded": True,
        },
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
        "# Independent posterior-MAP safe-utility confirmation",
        "",
        "The per-mode robust/specialist map is selected on three stationary "
        "calibration streams. All switching returns use three disjoint holdout "
        "streams. ESCP and RE-SAC are trained from scratch with the same seed, "
        "environment, horizon, and per-controller interaction budget.",
        "",
        "| Seed | Map | Robust SAC | Safe oracle | MAP utility | ESCP | RE-SAC | "
        "MAP gain | Recovery | Gate |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for seed in protocol.TRAINING_SEEDS:
        row = payload["seed_metrics"][str(seed)]
        arms = row["arms"]
        recovery = row["primary_oracle_recovery"]
        lines.append(
            f"| {seed} | `{_map_text(row['utility_map'])}` | "
            f"{arms['robust_sac']['mean']:.1f} | "
            f"{arms['true_mode_safe_utility']['mean']:.1f} | "
            f"{arms[protocol.PRIMARY_ARM]['mean']:.1f} | "
            f"{arms['escp_recurrent']['mean']:.1f} | "
            f"{arms['resac_b0']['mean']:.1f} | "
            f"{100 * row['primary_relative_gain']:.1f}% | "
            f"{'n/a' if recovery is None else f'{100 * recovery:.1f}%'} | "
            f"{row['composition_pass']} |"
        )
    lines += [
        "",
        f"Composition gate: **{payload['composition_pass']}**. Paired baseline "
        f"confirmation: **{payload['comparative_pass']}**.",
        "",
        "| Comparator | MAP mean | Comparator mean | Difference | 95% paired CI | "
        "Seed wins | Event wins | Confirmed |",
        "|:---|---:|---:|---:|:---:|:---:|:---:|:---:|",
    ]
    for comparator, row in payload["comparisons"].items():
        interval = row["paired_difference_ci95"]
        lines.append(
            f"| {comparator} | {row['primary_mean']:.1f} | "
            f"{row['comparator_mean']:.1f} | "
            f"{row['paired_difference_mean']:.1f} | "
            f"[{interval[0]:.1f}, {interval[1]:.1f}] | "
            f"{row['seed_wins']}/5 | {row['event_wins']}/15 | "
            f"{row['confirmed']} |"
        )
    cost = payload["cost"]
    lines += [
        "",
        "## Cost accounting",
        "",
        f"Each BAPR bank trains five policies: {cost['bapr_policy_bank_steps_per_seed']:,} "
        f"environment steps and {cost['bapr_policy_bank_updates_per_seed']:,} "
        "updates per seed. Each SAC/ESCP/RE-SAC comparator trains one policy: "
        f"{cost['single_baseline_steps_per_seed']:,} steps and "
        f"{cost['single_baseline_updates_per_seed']:,} updates. The policy-training "
        f"ratio is therefore {cost['policy_training_cost_ratio']}x; frozen estimator "
        "pretraining is additional and excluded from that ratio.",
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
        "V9 SAFE UTILITY ANALYSIS COMPLETE: "
        f"composition={payload['composition_pass']} "
        f"comparative={payload['comparative_pass']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
