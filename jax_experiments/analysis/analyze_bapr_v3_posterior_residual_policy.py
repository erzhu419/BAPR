"""Aggregate the nonlinear posterior residual development audit."""
from __future__ import annotations

import json
import statistics
from typing import Any

import numpy as np

from jax_experiments.analysis import (
    analyze_bapr_v3_posterior_residual_screen as screen_analysis,
)
from jax_experiments.analysis import (
    bapr_v3_posterior_residual_policy as protocol,
)
from jax_experiments.analysis import (
    run_bapr_v3_posterior_residual_policy_audit as runner,
)
from jax_experiments.analysis import (
    run_bapr_v3_utility_aware_router_audit as base_audit,
)


METRICS = ("stationary", "switching_slow", "switching_full_cycle")
BASELINES = ("learned_utility_router", "robust", "dynamic_utility_oracle")


def candidate_return(group, metric: str) -> float:
    if metric == "stationary":
        return statistics.mean(
            float(group["stationary"][str(mode)]["mean"])
            for mode in range(4))
    kind = "slow_pair" if metric == "switching_slow" else "full_cycle"
    return float(group["switching"][kind]["mean"])


def baseline_return(group, controller: str, metric: str) -> float:
    return screen_analysis.event_return(group, controller, metric)


def candidate_termination(group, metric: str) -> float:
    if metric == "stationary":
        episodes = [
            episode
            for mode in range(4)
            for episode in group["stationary"][str(mode)]["episodes"]
        ]
        return float(np.mean([bool(row["terminated"]) for row in episodes]))
    kind = "slow_pair" if metric == "switching_slow" else "full_cycle"
    return float(np.mean([
        int(row["termination_count"]) > 0
        for row in group["switching"][kind]["episodes"]]))


def routing_means(groups, metric: str):
    rows = []
    for group in groups.values():
        if metric == "stationary":
            for mode in range(4):
                rows.extend(group["stationary"][str(mode)]["routing"])
        else:
            kind = "slow_pair" if metric == "switching_slow" else "full_cycle"
            rows.extend(group["switching"][kind]["routing"])
    return {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in (
            "adaptation_strength_mean", "adaptation_positive_rate",
            "action_delta_l2_mean", "wrong_route_rate",
            "median_switch_delay",
        )
    }


def summarize() -> dict[str, Any]:
    protocol.configure()
    model_manifest = protocol.load_manifest()
    groups = {}
    references = {}
    for seed in protocol.DEVELOPMENT_RETURN_EVENT_SEEDS:
        payload = json.loads(protocol.audit_group_path(seed).read_text(
            encoding="utf-8"))
        runner.validate_group(payload)
        groups[seed] = payload
        reference_path = protocol.screen.utility.audit_group_path(
            "validation", seed, protocol.DECISION_VARIANT)
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        base_audit.validate_group(reference)
        references[seed] = reference

    candidate = {
        metric: [candidate_return(groups[seed], metric)
                 for seed in protocol.DEVELOPMENT_RETURN_EVENT_SEEDS]
        for metric in METRICS
    }
    baselines = {
        controller: {
            metric: [baseline_return(references[seed], controller, metric)
                     for seed in protocol.DEVELOPMENT_RETURN_EVENT_SEEDS]
            for metric in METRICS
        }
        for controller in BASELINES
    }
    comparisons = {}
    for controller in BASELINES:
        comparisons[controller] = {}
        for metric in METRICS:
            differences = [
                value - baseline
                for value, baseline in zip(
                    candidate[metric], baselines[controller][metric])
            ]
            comparisons[controller][metric] = {
                "per_event_seed": dict(zip(
                    protocol.DEVELOPMENT_RETURN_EVENT_SEEDS, differences)),
                "mean": statistics.mean(differences),
                "wins": sum(value > 0.0 for value in differences),
            }
    termination_deltas = {}
    for metric in ("stationary", "switching_full_cycle"):
        termination_deltas[metric] = statistics.mean(
            candidate_termination(groups[seed], metric)
            - screen_analysis.event_termination(
                references[seed], "learned_utility_router", metric)
            for seed in protocol.DEVELOPMENT_RETURN_EVENT_SEEDS)
    routing = {metric: routing_means(groups, metric) for metric in METRICS}
    gate = screen_analysis.variant_gate(
        list(comparisons["learned_utility_router"][
            "switching_full_cycle"]["per_event_seed"].values()),
        list(comparisons["robust"][
            "switching_full_cycle"]["per_event_seed"].values()),
        list(comparisons["learned_utility_router"][
            "stationary"]["per_event_seed"].values()),
        termination_deltas["stationary"],
        termination_deltas["switching_full_cycle"],
        routing["switching_full_cycle"]["adaptation_positive_rate"],
        True,
    )
    controllers = {
        "posterior_residual_policy": {
            metric: {
                "per_event_seed": dict(zip(
                    protocol.DEVELOPMENT_RETURN_EVENT_SEEDS,
                    candidate[metric])),
                "mean": statistics.mean(candidate[metric]),
            }
            for metric in METRICS
        }
    }
    for controller in BASELINES:
        controllers[controller] = {
            metric: {
                "per_event_seed": dict(zip(
                    protocol.DEVELOPMENT_RETURN_EVENT_SEEDS,
                    baselines[controller][metric])),
                "mean": statistics.mean(baselines[controller][metric]),
            }
            for metric in METRICS
        }
    robust = controllers["robust"]["switching_full_cycle"]["mean"]
    oracle = controllers["dynamic_utility_oracle"][
        "switching_full_cycle"]["mean"]
    learned = controllers["posterior_residual_policy"][
        "switching_full_cycle"]["mean"]
    recovery = ((learned - robust) / (oracle - robust)
                if oracle > robust else None)
    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "family": protocol.FAMILY,
        "env": protocol.ENV,
        "decision_variant": protocol.DECISION_VARIANT,
        "development_event_seeds": list(
            protocol.DEVELOPMENT_RETURN_EVENT_SEEDS),
        "sealed_confirmation_event_seeds": list(
            protocol.SEALED_CONFIRMATION_EVENT_SEEDS),
        "model_manifest_file": protocol.file_record(protocol.MANIFEST_PATH),
        "selected_model_stage": model_manifest["selected_stage"],
        "selected_model_validation_metrics":
            model_manifest["selected_validation_metrics"],
        "controllers": controllers,
        "candidate_comparisons": comparisons,
        "termination_rate_deltas_vs_hard_router": termination_deltas,
        "routing": routing,
        "full_cycle_oracle_recovery": recovery,
        "gate": gate,
    }


def render(summary: dict[str, Any]) -> str:
    labels = {
        "stationary": "Stationary",
        "switching_slow": "Slow pair",
        "switching_full_cycle": "Full cycle",
    }
    lines = [
        "# Nonlinear posterior residual policy audit",
        "",
        f"Selected training stage: `{summary['selected_model_stage']}`.",
        "",
        "| Metric | Robust | Hard CUSUM | Utility oracle | Nonlinear residual | "
        "Delta hard | Delta robust |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in METRICS:
        controllers = summary["controllers"]
        comparisons = summary["candidate_comparisons"]
        lines.append(
            f"| {labels[metric]} | "
            f"{controllers['robust'][metric]['mean']:.1f} | "
            f"{controllers['learned_utility_router'][metric]['mean']:.1f} | "
            f"{controllers['dynamic_utility_oracle'][metric]['mean']:.1f} | "
            f"{controllers['posterior_residual_policy'][metric]['mean']:.1f} | "
            f"{comparisons['learned_utility_router'][metric]['mean']:+.1f} | "
            f"{comparisons['robust'][metric]['mean']:+.1f} |")
    lines += [
        "",
        f"- Full-cycle oracle recovery: "
        f"`{summary['full_cycle_oracle_recovery']:.1%}`",
        f"- Development gate: "
        f"**{'PASS' if summary['gate']['passed'] else 'FAIL'}**",
        "",
        "The sealed confirmation streams remain unopened unless this gate "
        "passes exactly as predeclared.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    summary = summarize()
    protocol.write_json_atomic(protocol.ANALYSIS_JSON, summary)
    protocol.ANALYSIS_REPORT.parent.mkdir(parents=True, exist_ok=True)
    protocol.ANALYSIS_REPORT.write_text(render(summary), encoding="utf-8")
    print(
        "NONLINEAR RESIDUAL ANALYSIS COMPLETE: "
        f"gate={'PASS' if summary['gate']['passed'] else 'FAIL'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
