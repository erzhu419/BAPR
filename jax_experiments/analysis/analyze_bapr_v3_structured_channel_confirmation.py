"""Aggregate the frozen five-seed CUSUM return confirmation."""
from __future__ import annotations

import json
import statistics

import numpy as np

from jax_experiments.analysis import (
    analyze_bapr_v3_budget_matched_audit as paired,
)
from jax_experiments.analysis import (
    bapr_v3_structured_channel_confirmation as protocol,
)
from jax_experiments.analysis import (
    run_bapr_v3_structured_channel_confirmation as runner,
)
from jax_experiments.analysis import (
    run_bapr_v3_utility_aware_router_audit as base_audit,
)


METRICS = ("stationary", "switching_slow", "switching_full_cycle")


def event_return(group, controller: str, metric: str) -> float:
    if metric == "stationary":
        return statistics.mean(
            float(group["stationary"][controller][str(mode)]["mean"])
            for mode in range(4))
    kind = "slow_pair" if metric == "switching_slow" else "full_cycle"
    return float(group["switching"][kind][controller]["mean"])


def event_termination(group, controller: str, metric: str) -> float:
    if metric == "stationary":
        episodes = [
            episode
            for mode in range(4)
            for episode in group["stationary"][controller][str(mode)][
                "episodes"]
        ]
    else:
        kind = "slow_pair" if metric == "switching_slow" else "full_cycle"
        episodes = group["switching"][kind][controller]["episodes"]
        return float(np.mean([
            int(row["termination_count"]) > 0 for row in episodes]))
    return float(np.mean([bool(row["terminated"]) for row in episodes]))


def paired_comparison(left, right):
    differences = [float(a - b) for a, b in zip(left, right)]
    center, lower, upper = paired.paired_interval(differences)
    return {
        "differences": dict(zip(protocol.EVENT_SEEDS, differences)),
        "mean": center,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "wins": sum(value > 0.0 for value in differences),
    }


def routing_rows(groups, metric: str):
    rows = []
    for group in groups.values():
        if metric == "stationary":
            for mode in range(4):
                rows.extend(group["stationary"]["learned_utility_router"][
                    str(mode)]["routing"])
        else:
            kind = "slow_pair" if metric == "switching_slow" else "full_cycle"
            rows.extend(group["switching"][kind][
                "learned_utility_router"]["routing"])
    return rows


def summarize():
    protocol.configure()
    groups = {}
    for seed in protocol.EVENT_SEEDS:
        payload = json.loads(protocol.group_path(seed).read_text(
            encoding="utf-8"))
        runner.validate_group(payload)
        groups[seed] = payload

    series = {
        controller: {
            metric: [event_return(groups[seed], controller, metric)
                     for seed in protocol.EVENT_SEEDS]
            for metric in METRICS
        }
        for controller in base_audit.CONTROLLERS
    }
    controllers = {
        controller: {
            metric: {
                "per_event_seed": dict(zip(
                    protocol.EVENT_SEEDS, series[controller][metric])),
                "mean": statistics.mean(series[controller][metric]),
                "sd": statistics.stdev(series[controller][metric]),
                "termination_rate": statistics.mean(
                    event_termination(groups[seed], controller, metric)
                    for seed in protocol.EVENT_SEEDS),
            }
            for metric in METRICS
        }
        for controller in base_audit.CONTROLLERS
    }
    comparisons = {
        baseline: {
            metric: paired_comparison(
                series["learned_utility_router"][metric],
                series[baseline][metric])
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
            "fraction": ((learned - robust) / headroom
                         if headroom > 0.0 else None),
        }
    routing = {}
    for metric in METRICS:
        rows = routing_rows(groups, metric)
        routing[metric] = {
            name: float(np.mean([float(row[name]) for row in rows]))
            for name in (
                "coverage", "fallback_rate", "conditional_accuracy",
                "action_accuracy", "wrong_route_rate",
                "median_switch_delay")
        }

    robust_comparison = comparisons["robust"]
    full_recovery = recovery["switching_full_cycle"]["fraction"]
    stationary_term_delta = (
        controllers["learned_utility_router"]["stationary"][
            "termination_rate"]
        - controllers["robust"]["stationary"]["termination_rate"])
    full_term_delta = (
        controllers["learned_utility_router"]["switching_full_cycle"][
            "termination_rate"]
        - controllers["robust"]["switching_full_cycle"][
            "termination_rate"])
    gate = {
        "full_cycle_positive_ci": (
            robust_comparison["switching_full_cycle"]["ci95_lower"] > 0.0),
        "full_cycle_minimum_wins": (
            robust_comparison["switching_full_cycle"]["wins"]
            >= protocol.MIN_FULL_CYCLE_WINS),
        "full_cycle_oracle_headroom_positive": (
            recovery["switching_full_cycle"]["oracle_headroom"] > 0.0),
        "full_cycle_oracle_recovery": (
            full_recovery is not None
            and full_recovery >= protocol.MIN_ORACLE_RECOVERY),
        "stationary_return_noninferior": (
            robust_comparison["stationary"]["ci95_lower"]
            > -protocol.STATIONARY_NONINFERIORITY_MARGIN),
        "stationary_termination_noninferior": (
            stationary_term_delta <= protocol.TERMINATION_RATE_MARGIN),
        "full_cycle_termination_noninferior": (
            full_term_delta <= protocol.TERMINATION_RATE_MARGIN),
        "committed_route_accuracy": (
            routing["switching_full_cycle"]["conditional_accuracy"]
            >= protocol.MIN_COMMITTED_ROUTE_ACCURACY),
    }
    gate["passed"] = all(gate.values())
    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "family": protocol.FAMILY,
        "env": protocol.ENV,
        "decision_variant": protocol.DECISION_VARIANT,
        "event_seeds": list(protocol.EVENT_SEEDS),
        "primary_endpoints": runner.primary_endpoints(),
        "controllers": controllers,
        "learned_comparisons": comparisons,
        "oracle_recovery": recovery,
        "routing": routing,
        "termination_rate_deltas": {
            "stationary": stationary_term_delta,
            "switching_full_cycle": full_term_delta,
        },
        "gate": gate,
    }


def render(summary) -> str:
    labels = {
        "stationary": "Stationary",
        "switching_slow": "Slow pair",
        "switching_full_cycle": "Full cycle",
    }
    lines = [
        "# Structured-channel CUSUM confirmation",
        "",
        f"Frozen decision variant: `{protocol.DECISION_VARIANT}`; fresh event "
        f"seeds: `{list(protocol.EVENT_SEEDS)}`.",
        "",
        "| Metric | Robust | Utility oracle | CUSUM router | "
        "CUSUM - robust (95% CI) | Wins | Recovery |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in METRICS:
        robust = summary["controllers"]["robust"][metric]
        oracle = summary["controllers"]["dynamic_utility_oracle"][metric]
        learned = summary["controllers"]["learned_utility_router"][metric]
        comparison = summary["learned_comparisons"]["robust"][metric]
        recovery = summary["oracle_recovery"][metric]["fraction"]
        recovery_text = "n/a" if recovery is None else f"{recovery:.1%}"
        lines.append(
            f"| {labels[metric]} | {robust['mean']:.1f} +/- {robust['sd']:.1f} "
            f"| {oracle['mean']:.1f} +/- {oracle['sd']:.1f} | "
            f"{learned['mean']:.1f} +/- {learned['sd']:.1f} | "
            f"{comparison['mean']:+.1f} "
            f"[{comparison['ci95_lower']:+.1f},"
            f"{comparison['ci95_upper']:+.1f}] | "
            f"{comparison['wins']}/5 | {recovery_text} |")
    full_route = summary["routing"]["switching_full_cycle"]
    lines += [
        "",
        f"- Full-cycle committed route accuracy: "
        f"`{full_route['conditional_accuracy']:.1%}`",
        f"- Full-cycle robust fallback rate: "
        f"`{full_route['fallback_rate']:.1%}`",
        f"- Confirmation gate: "
        f"**{'PASS' if summary['gate']['passed'] else 'FAIL'}**",
        "",
        "Exact-controller accuracy is diagnostic only. Deliberate robust "
        "fallback is evaluated through return and termination, not counted as "
        "a primary failure.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    summary = summarize()
    protocol.write_json_atomic(protocol.ANALYSIS_JSON, summary)
    protocol.ANALYSIS_REPORT.parent.mkdir(parents=True, exist_ok=True)
    protocol.ANALYSIS_REPORT.write_text(render(summary), encoding="utf-8")
    print(
        "CUSUM CONFIRMATION ANALYSIS COMPLETE: "
        f"gate={'PASS' if summary['gate']['passed'] else 'FAIL'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
