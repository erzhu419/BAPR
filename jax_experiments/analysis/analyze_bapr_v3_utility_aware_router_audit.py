"""Aggregate validation or sealed holdouts for utility-aware routing."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from jax_experiments.analysis import (
    analyze_bapr_v3_budget_matched_audit as paired,
)
from jax_experiments.analysis import bapr_v3_utility_aware_router as protocol
from jax_experiments.analysis import (
    run_bapr_v3_utility_aware_router_audit as audit,
)
from jax_experiments.analysis import (
    bapr_v3_learned_control_router as estimator,
)


METRICS = ("stationary", "switching_slow", "switching_full_cycle")


def event_value(group, controller: str, metric: str) -> float:
    if metric == "stationary":
        return statistics.mean(
            float(group["stationary"][controller][str(mode)]["mean"])
            for mode in range(4))
    kind = "slow_pair" if metric == "switching_slow" else "full_cycle"
    return float(group["switching"][kind][controller]["mean"])


def paired_comparison(left, right, seeds):
    differences = [a - b for a, b in zip(left, right)]
    if len(differences) == 5:
        center, lower, upper = paired.paired_interval(differences)
    elif len(differences) == 2:
        center = statistics.mean(differences)
        half_width = (
            12.7062047364 * statistics.stdev(differences)
            / np.sqrt(len(differences)))
        lower, upper = center - half_width, center + half_width
    else:
        raise ValueError(
            "utility-router comparison requires two validation or five "
            "holdout streams")
    return {
        "differences": dict(zip(seeds, differences)),
        "mean": center,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "wins": sum(value > 0.0 for value in differences),
    }


def _routing_rows(groups, metric: str):
    rows = []
    for group in groups.values():
        if metric == "stationary":
            for mode in range(4):
                rows.extend(group["stationary"][
                    "learned_utility_router"][str(mode)]["routing"])
        else:
            kind = (
                "slow_pair" if metric == "switching_slow"
                else "full_cycle")
            rows.extend(group["switching"][kind][
                "learned_utility_router"]["routing"])
    return rows


def summarize(
    role: str,
    decision_variant: str = protocol.BASELINE_DECISION_VARIANT,
) -> dict:
    protocol.configure()
    table = protocol.load_utility_table()
    seeds = protocol.event_seeds(role)
    groups = {}
    table_record = estimator.file_record(protocol.TABLE_PATH)
    manifest_record = estimator.file_record(estimator.MANIFEST_PATH)
    parameter_record = estimator.file_record(estimator.MODEL_PATH)
    for seed in seeds:
        path = protocol.audit_group_path(role, seed, decision_variant)
        payload = json.loads(path.read_text(encoding="utf-8"))
        audit.validate_group(payload)
        if (payload["utility_table_file"] != table_record
                or payload["estimator_manifest_file"] != manifest_record
                or payload["estimator_parameter_file"] != parameter_record
                or payload.get(
                    "decision_variant",
                    protocol.BASELINE_DECISION_VARIANT) != decision_variant
                or tuple(payload["oracle_controller_map"])
                != tuple(table["oracle_controller_map"])):
            raise ValueError(f"utility-router provenance mismatch: {path}")
        groups[seed] = payload

    series = {
        controller: {
            metric: [event_value(groups[seed], controller, metric)
                     for seed in seeds]
            for metric in METRICS
        }
        for controller in audit.CONTROLLERS
    }
    controllers = {
        controller: {
            metric: {
                "per_event_seed": dict(zip(
                    seeds, series[controller][metric])),
                "mean": statistics.mean(series[controller][metric]),
                "sd": statistics.stdev(series[controller][metric]),
            }
            for metric in METRICS
        }
        for controller in audit.CONTROLLERS
    }
    comparisons = {
        baseline: {
            metric: paired_comparison(
                series["learned_utility_router"][metric],
                series[baseline][metric], seeds)
            for metric in METRICS
        }
        for baseline in ("robust", "dynamic_utility_oracle")
    }

    recovery = {}
    for metric in METRICS:
        robust = controllers["robust"][metric]["mean"]
        oracle = controllers["dynamic_utility_oracle"][metric]["mean"]
        learned = controllers["learned_utility_router"][metric]["mean"]
        headroom = oracle - robust
        recovery[metric] = {
            "oracle_headroom": headroom,
            "learned_gain": learned - robust,
            "fraction": (
                (learned - robust) / headroom if headroom > 0.0 else None),
        }

    routing = {}
    for metric in METRICS:
        rows = _routing_rows(groups, metric)
        routing[metric] = {
            name: float(np.mean([float(row[name]) for row in rows]))
            for name in (
                "coverage", "fallback_rate", "conditional_accuracy",
                "action_accuracy", "wrong_route_rate",
                "median_switch_delay")
        }

    robust_comparison = comparisons["robust"]
    if role == "validation":
        gate = {
            **{
                f"{metric}_positive_mean": (
                    robust_comparison[metric]["mean"] > 0.0)
                for metric in METRICS
            },
            **{
                f"{metric}_all_wins": (
                    robust_comparison[metric]["wins"] == len(seeds))
                for metric in METRICS
            },
            **{
                f"{metric}_oracle_headroom_positive": (
                    recovery[metric]["oracle_headroom"] > 0.0)
                for metric in METRICS
            },
            **{
                f"{metric}_recovery_at_least_65pct": (
                    recovery[metric]["fraction"] is not None
                    and recovery[metric]["fraction"] >= 0.65)
                for metric in METRICS
            },
            "full_cycle_action_accuracy_at_least_90pct": (
                routing["switching_full_cycle"]["action_accuracy"] >= 0.90),
            "full_cycle_wrong_route_at_most_10pct": (
                routing["switching_full_cycle"]["wrong_route_rate"] <= 0.10),
            "full_cycle_median_delay_below_75": (
                routing["switching_full_cycle"][
                    "median_switch_delay"] < 75.0),
        }
    else:
        gate = {
            **{
                f"{metric}_positive_ci": (
                    robust_comparison[metric]["ci95_lower"] > 0.0)
                for metric in METRICS
            },
            **{
                f"{metric}_five_of_five": (
                    robust_comparison[metric]["wins"] == 5)
                for metric in METRICS
            },
            **{
                f"{metric}_oracle_recovery_at_least_70pct": (
                    recovery[metric]["fraction"] is not None
                    and recovery[metric]["fraction"] >= 0.70)
                for metric in METRICS
            },
            "full_cycle_action_accuracy_at_least_90pct": (
                routing["switching_full_cycle"]["action_accuracy"] >= 0.90),
            "full_cycle_wrong_route_at_most_8pct": (
                routing["switching_full_cycle"]["wrong_route_rate"] <= 0.08),
            "full_cycle_median_delay_below_75": (
                routing["switching_full_cycle"][
                    "median_switch_delay"] < 75.0),
        }
    gate["passed"] = all(gate.values())
    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "role": role,
        "decision_variant": decision_variant,
        "family": protocol.FAMILY,
        "env": protocol.ENV,
        "training_seed": 0,
        "oracle_controller_map": table["oracle_controller_map"],
        "nondominated_controllers": table["nondominated_controllers"],
        "event_seeds": list(seeds),
        "controllers": controllers,
        "learned_comparisons": comparisons,
        "oracle_recovery": recovery,
        "routing": routing,
        "gate": gate,
    }


def _percentage(value):
    return "n/a" if value is None else f"{value:.1%}"


def render(summary) -> str:
    metrics = {
        "stationary": "Stationary",
        "switching_slow": "Slow pair",
        "switching_full_cycle": "Full cycle",
    }
    lines = [
        f"# Utility-aware router {summary['role']}",
        "",
        f"Decision variant: `{summary['decision_variant']}`.",
        f"Frozen oracle map: `{summary['oracle_controller_map']}`; "
        f"robust controller code is `{protocol.ROBUST_CONTROLLER}`.",
        "",
        "| Metric | Robust | Utility oracle | Learned router | "
        "Learned - robust (95% CI) | Wins | Recovery |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metric, label in metrics.items():
        robust = summary["controllers"]["robust"][metric]
        oracle = summary["controllers"]["dynamic_utility_oracle"][metric]
        learned = summary["controllers"]["learned_utility_router"][metric]
        comparison = summary["learned_comparisons"]["robust"][metric]
        recovery = summary["oracle_recovery"][metric]["fraction"]
        lines.append(
            f"| {label} | {robust['mean']:.1f} +/- {robust['sd']:.1f} | "
            f"{oracle['mean']:.1f} +/- {oracle['sd']:.1f} | "
            f"{learned['mean']:.1f} +/- {learned['sd']:.1f} | "
            f"{comparison['mean']:+.1f} "
            f"[{comparison['ci95_lower']:+.1f},"
            f"{comparison['ci95_upper']:+.1f}] | "
            f"{comparison['wins']}/{len(summary['event_seeds'])} | "
            f"{_percentage(recovery)} |")
    lines += [
        "",
        "| Routing metric | Stationary | Slow pair | Full cycle |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (
        ("coverage", "Committed coverage"),
        ("action_accuracy", "Action-controller accuracy"),
        ("wrong_route_rate", "Wrong action-controller rate"),
        ("median_switch_delay", "Median switch delay"),
    ):
        values = [summary["routing"][metric][key] for metric in METRICS]
        if key == "median_switch_delay":
            rendered = [f"{value:.1f}" for value in values]
        else:
            rendered = [f"{value:.1%}" for value in values]
        lines.append(f"| {label} | " + " | ".join(rendered) + " |")
    lines += [
        "",
        f"- Utility-router {summary['role']} gate: "
        f"**{'PASS' if summary['gate']['passed'] else 'FAIL'}**",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("validation", "holdout"),
                        required=True)
    parser.add_argument(
        "--decision-variant", choices=tuple(protocol.DECISION_VARIANTS),
        default=protocol.BASELINE_DECISION_VARIANT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.role, args.decision_variant)
    estimator.write_json_atomic(args.json_output.resolve(), summary)
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(render(summary), encoding="utf-8")
    print(
        "UTILITY ROUTER ANALYSIS COMPLETE: "
        f"role={args.role} "
        f"gate={'PASS' if summary['gate']['passed'] else 'FAIL'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
