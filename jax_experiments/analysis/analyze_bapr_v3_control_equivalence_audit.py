"""Aggregate untouched holdout audits for a frozen controller map."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from jax_experiments.analysis import (
    analyze_bapr_v3_budget_matched_audit as paired,
)
from jax_experiments.analysis import bapr_v3_control_equivalence as protocol
from jax_experiments.analysis import (
    run_bapr_v3_control_equivalence_audit as audit,
)


COMPARISONS = (
    "robust", "fixed_mode_0", "fixed_mode_1",
    "fixed_mode_2", "fixed_mode_3",
)


def event_value(group: dict, controller: str, metric: str) -> float:
    if metric == "switching":
        return float(group["switching"][controller]["mean"])
    return statistics.mean(
        group["stationary"][controller][str(mode)]["mean"]
        for mode in protocol.specialist_protocol.MODES
    )


def summarize() -> dict:
    protocol.configure()
    controller_map, mapping_payload = protocol.load_controller_map()
    groups = {}
    for seed in protocol.HOLDOUT_EVENT_SEEDS:
        path = protocol.holdout_group_path(seed)
        payload = json.loads(path.read_text(encoding="utf-8"))
        audit.validate_group(payload)
        if tuple(payload["controller_map"]) != controller_map:
            raise ValueError(f"holdout map mismatch: {path}")
        groups[seed] = payload

    controllers = (
        "robust", "dynamic_oracle", audit.CONTROLLER,
        "fixed_mode_0", "fixed_mode_1", "fixed_mode_2", "fixed_mode_3",
    )
    series = {
        controller: {
            metric: [event_value(groups[seed], controller, metric)
                     for seed in protocol.HOLDOUT_EVENT_SEEDS]
            for metric in ("stationary", "switching")
        }
        for controller in controllers
    }
    controller_summary = {
        controller: {
            metric: {
                "per_event_seed": dict(zip(
                    protocol.HOLDOUT_EVENT_SEEDS,
                    series[controller][metric],
                )),
                "mean": statistics.mean(series[controller][metric]),
                "sd": statistics.stdev(series[controller][metric]),
            }
            for metric in ("stationary", "switching")
        }
        for controller in controllers
    }

    comparisons = {}
    for controller in COMPARISONS:
        comparisons[controller] = {}
        for metric in ("stationary", "switching"):
            differences = [
                mapped - baseline
                for mapped, baseline in zip(
                    series[audit.CONTROLLER][metric],
                    series[controller][metric],
                )
            ]
            center, lower, upper = paired.paired_interval(differences)
            comparisons[controller][metric] = {
                "differences": dict(zip(
                    protocol.HOLDOUT_EVENT_SEEDS, differences)),
                "mean": center,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "wins": sum(value > 0.0 for value in differences),
                "passes_positive_mean": center > 0.0,
            }

    matrix = {}
    mapping_agreement = 0
    for physics_mode in protocol.specialist_protocol.MODES:
        values = {
            specialist_mode: statistics.mean(
                groups[seed]["stationary"][
                    f"fixed_mode_{specialist_mode}"][str(physics_mode)][
                        "mean"]
                for seed in protocol.HOLDOUT_EVENT_SEEDS
            )
            for specialist_mode in protocol.specialist_protocol.MODES
        }
        winner = max(values, key=values.get)
        agrees = winner == controller_map[physics_mode]
        mapping_agreement += int(agrees)
        matrix[str(physics_mode)] = {
            "mean_returns": {str(key): value for key, value in values.items()},
            "holdout_best_specialist": winner,
            "calibrated_specialist": controller_map[physics_mode],
            "agrees": agrees,
        }

    stationary_pass = all(
        comparisons[name]["stationary"]["passes_positive_mean"]
        for name in COMPARISONS
    )
    switching_pass = all(
        comparisons[name]["switching"]["passes_positive_mean"]
        for name in COMPARISONS
    )
    distinct = len(set(controller_map))
    gate = {
        "mapped_beats_all_stationary": stationary_pass,
        "mapped_beats_all_switching": switching_pass,
        "holdout_mapping_agreement": mapping_agreement,
        "holdout_mapping_agreement_at_least_3_of_4": mapping_agreement >= 3,
        "distinct_specialists": distinct,
        "at_least_3_distinct_specialists": distinct >= 3,
        "passed": (
            stationary_pass and switching_pass
            and mapping_agreement >= 3 and distinct >= 3
        ),
    }
    return {
        "schema": "bapr.v3-control-equivalence-analysis.v1",
        "family": protocol.FAMILY,
        "env": protocol.ENV,
        "training_seed": protocol.specialist_protocol.SEED,
        "calibration_event_seeds": mapping_payload["calibration_event_seeds"],
        "holdout_event_seeds": list(protocol.HOLDOUT_EVENT_SEEDS),
        "controller_map": list(controller_map),
        "controllers": controller_summary,
        "mapped_comparisons": comparisons,
        "holdout_specialist_matrix": matrix,
        "gate": gate,
    }


def render(summary: dict) -> str:
    lines = [
        "# Structured-channel control-equivalence holdout",
        "",
        f"Calibration-frozen controller map: `{summary['controller_map']}`.",
        "Calibration streams are disjoint from the five holdout streams.",
        "",
        "| Controller | Stationary mean +/- SD | Switching mean +/- SD |",
        "|---|---:|---:|",
    ]
    order = (
        "robust", "dynamic_oracle", audit.CONTROLLER,
        "fixed_mode_0", "fixed_mode_1", "fixed_mode_2", "fixed_mode_3",
    )
    for controller in order:
        values = summary["controllers"][controller]
        lines.append(
            f"| `{controller}` | {values['stationary']['mean']:.1f} +/- "
            f"{values['stationary']['sd']:.1f} | "
            f"{values['switching']['mean']:.1f} +/- "
            f"{values['switching']['sd']:.1f} |"
        )
    lines += [
        "",
        "| Mapped comparison | Stationary difference (95% CI) | Wins | "
        "Switching difference (95% CI) | Wins |",
        "|---|---:|---:|---:|---:|",
    ]
    for controller, metrics in summary["mapped_comparisons"].items():
        stationary = metrics["stationary"]
        switching = metrics["switching"]
        lines.append(
            f"| mapped - `{controller}` | {stationary['mean']:+.1f} "
            f"[{stationary['ci95_lower']:+.1f},"
            f"{stationary['ci95_upper']:+.1f}] | "
            f"{stationary['wins']}/5 | {switching['mean']:+.1f} "
            f"[{switching['ci95_lower']:+.1f},"
            f"{switching['ci95_upper']:+.1f}] | "
            f"{switching['wins']}/5 |"
        )
    lines += [
        "",
        "| Physics mode | Calibration specialist | Holdout best | Agrees |",
        "|---:|---:|---:|---:|",
    ]
    for mode, row in summary["holdout_specialist_matrix"].items():
        lines.append(
            f"| {mode} | {row['calibrated_specialist']} | "
            f"{row['holdout_best_specialist']} | "
            f"{'yes' if row['agrees'] else 'no'} |"
        )
    gate = summary["gate"]
    lines += [
        "",
        f"- Beats robust and every fixed specialist in stationary mean: "
        f"**{'PASS' if gate['mapped_beats_all_stationary'] else 'FAIL'}**",
        f"- Beats robust and every fixed specialist in switching mean: "
        f"**{'PASS' if gate['mapped_beats_all_switching'] else 'FAIL'}**",
        f"- Calibration/holdout map agreement: "
        f"**{gate['holdout_mapping_agreement']}/4**",
        f"- Distinct selected specialists: **{gate['distinct_specialists']}**",
        f"- Control-equivalence promotion gate: "
        f"**{'PASS' if gate['passed'] else 'FAIL'}**",
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
        "CONTROL-EQUIVALENCE ANALYSIS COMPLETE: "
        f"gate={'PASS' if summary['gate']['passed'] else 'FAIL'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
