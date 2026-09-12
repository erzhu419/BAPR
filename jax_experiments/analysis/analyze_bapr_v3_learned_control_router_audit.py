"""Aggregate sealed holdouts for the learned control-equivalence router."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from jax_experiments.analysis import (
    analyze_bapr_v3_budget_matched_audit as paired,
)
from jax_experiments.analysis import bapr_v3_learned_control_router as protocol
from jax_experiments.analysis import (
    run_bapr_v3_learned_control_router_audit as audit,
)


def event_value(group, controller: str, metric: str) -> float:
    if metric == "switching":
        return float(group["switching"][controller]["mean"])
    return statistics.mean(
        float(group["stationary"][controller][str(mode)]["mean"])
        for mode in range(4)
    )


def paired_comparison(left, right):
    differences = [a - b for a, b in zip(left, right)]
    center, lower, upper = paired.paired_interval(differences)
    return {
        "differences": dict(zip(protocol.HOLDOUT_EVENT_SEEDS, differences)),
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
                rows.extend(group["stationary"]["learned_router"][
                    str(mode)]["routing"])
        else:
            rows.extend(group["switching"]["learned_router"]["routing"])
    return rows


def summarize() -> dict:
    protocol.configure()
    manifest = protocol.load_manifest()
    controller_map, _ = protocol.control.load_controller_map()
    groups = {}
    manifest_record = protocol.file_record(protocol.MANIFEST_PATH)
    parameter_record = protocol.file_record(protocol.MODEL_PATH)
    for seed in protocol.HOLDOUT_EVENT_SEEDS:
        path = protocol.audit_group_path(seed)
        payload = json.loads(path.read_text(encoding="utf-8"))
        audit.validate_group(payload)
        if (tuple(payload["controller_map"]) != controller_map
                or payload["router_manifest_file"] != manifest_record
                or payload["router_parameter_file"] != parameter_record):
            raise ValueError(f"router holdout provenance mismatch: {path}")
        groups[seed] = payload

    series = {
        controller: {
            metric: [event_value(groups[seed], controller, metric)
                     for seed in protocol.HOLDOUT_EVENT_SEEDS]
            for metric in ("stationary", "switching")
        }
        for controller in audit.CONTROLLERS
    }
    controllers = {
        controller: {
            metric: {
                "per_event_seed": dict(zip(
                    protocol.HOLDOUT_EVENT_SEEDS,
                    series[controller][metric])),
                "mean": statistics.mean(series[controller][metric]),
                "sd": statistics.stdev(series[controller][metric]),
            }
            for metric in ("stationary", "switching")
        }
        for controller in audit.CONTROLLERS
    }
    comparisons = {
        baseline: {
            metric: paired_comparison(
                series["learned_router"][metric],
                series[baseline][metric])
            for metric in ("stationary", "switching")
        }
        for baseline in ("robust", "dynamic_control_oracle")
    }

    recovery = {}
    for metric in ("stationary", "switching"):
        robust = controllers["robust"][metric]["mean"]
        oracle = controllers["dynamic_control_oracle"][metric]["mean"]
        learned = controllers["learned_router"][metric]["mean"]
        denominator = oracle - robust
        recovery[metric] = {
            "oracle_headroom": denominator,
            "learned_gain": learned - robust,
            "fraction": (
                (learned - robust) / denominator
                if denominator > 0.0 else None),
        }

    routing = {}
    for metric in ("stationary", "switching"):
        rows = _routing_rows(groups, metric)
        routing[metric] = {
            name: float(np.mean([float(row[name]) for row in rows]))
            for name in (
                "coverage", "conditional_accuracy", "wrong_route_rate",
                "effective_accuracy", "median_switch_delay")
        }

    robust_comparison = comparisons["robust"]
    gate = {
        "stationary_positive_ci": (
            robust_comparison["stationary"]["ci95_lower"] > 0.0),
        "switching_positive_ci": (
            robust_comparison["switching"]["ci95_lower"] > 0.0),
        "stationary_five_of_five": (
            robust_comparison["stationary"]["wins"] == 5),
        "switching_five_of_five": (
            robust_comparison["switching"]["wins"] == 5),
        "stationary_oracle_recovery_at_least_70pct": (
            recovery["stationary"]["fraction"] is not None
            and recovery["stationary"]["fraction"] >= 0.70),
        "switching_oracle_recovery_at_least_70pct": (
            recovery["switching"]["fraction"] is not None
            and recovery["switching"]["fraction"] >= 0.70),
        "switching_conditional_accuracy_at_least_90pct": (
            routing["switching"]["conditional_accuracy"] >= 0.90),
        "switching_wrong_route_at_most_8pct": (
            routing["switching"]["wrong_route_rate"] <= 0.08),
        "switching_median_delay_below_50": (
            routing["switching"]["median_switch_delay"] < 50.0),
    }
    gate["passed"] = all(gate.values())
    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "family": protocol.FAMILY,
        "env": protocol.ENV,
        "training_seed": 0,
        "controller_map": list(controller_map),
        "train_event_seeds": manifest["train_event_seeds"],
        "validation_event_seeds": manifest["validation_event_seeds"],
        "holdout_event_seeds": list(protocol.HOLDOUT_EVENT_SEEDS),
        "router_config": manifest["router_config"],
        "controllers": controllers,
        "learned_comparisons": comparisons,
        "oracle_recovery": recovery,
        "routing": routing,
        "gate": gate,
    }


def render(summary) -> str:
    lines = [
        "# Learned control-equivalence router holdout",
        "",
        f"Frozen controller map: `{summary['controller_map']}`.",
        "Training, validation, and five final holdout streams are disjoint.",
        "",
        "| Controller | Stationary mean +/- SD | Switching mean +/- SD |",
        "|---|---:|---:|",
    ]
    for controller in audit.CONTROLLERS:
        values = summary["controllers"][controller]
        lines.append(
            f"| `{controller}` | {values['stationary']['mean']:.1f} +/- "
            f"{values['stationary']['sd']:.1f} | "
            f"{values['switching']['mean']:.1f} +/- "
            f"{values['switching']['sd']:.1f} |")
    lines += [
        "",
        "| Learned comparison | Stationary difference (95% CI) | Wins | "
        "Switching difference (95% CI) | Wins |",
        "|---|---:|---:|---:|---:|",
    ]
    for baseline, metrics in summary["learned_comparisons"].items():
        stationary = metrics["stationary"]
        switching = metrics["switching"]
        lines.append(
            f"| learned - `{baseline}` | {stationary['mean']:+.1f} "
            f"[{stationary['ci95_lower']:+.1f},"
            f"{stationary['ci95_upper']:+.1f}] | "
            f"{stationary['wins']}/5 | {switching['mean']:+.1f} "
            f"[{switching['ci95_lower']:+.1f},"
            f"{switching['ci95_upper']:+.1f}] | "
            f"{switching['wins']}/5 |")
    lines += [
        "",
        "| Metric | Stationary | Switching |",
        "|---|---:|---:|",
        f"| Oracle gain recovered | "
        f"{summary['oracle_recovery']['stationary']['fraction']:.1%} | "
        f"{summary['oracle_recovery']['switching']['fraction']:.1%} |",
        f"| Router coverage | {summary['routing']['stationary']['coverage']:.1%} | "
        f"{summary['routing']['switching']['coverage']:.1%} |",
        f"| Conditional accuracy | "
        f"{summary['routing']['stationary']['conditional_accuracy']:.1%} | "
        f"{summary['routing']['switching']['conditional_accuracy']:.1%} |",
        f"| Wrong-route rate | "
        f"{summary['routing']['stationary']['wrong_route_rate']:.1%} | "
        f"{summary['routing']['switching']['wrong_route_rate']:.1%} |",
        f"| Median switch delay | "
        f"{summary['routing']['stationary']['median_switch_delay']:.1f} | "
        f"{summary['routing']['switching']['median_switch_delay']:.1f} |",
        "",
        f"- Learned-router promotion gate: "
        f"**{'PASS' if summary['gate']['passed'] else 'FAIL'}**",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize()
    protocol.write_json_atomic(args.json_output.resolve(), summary)
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(render(summary), encoding="utf-8")
    print(
        "LEARNED CONTROL ROUTER ANALYSIS COMPLETE: "
        f"gate={'PASS' if summary['gate']['passed'] else 'FAIL'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
