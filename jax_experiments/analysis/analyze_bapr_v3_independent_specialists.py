"""Aggregate strict independent-specialist audit groups."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from jax_experiments.analysis import (
    analyze_bapr_v3_budget_matched_audit as paired,
)
from jax_experiments.analysis import (
    bapr_v3_independent_specialists as protocol,
)
from jax_experiments.analysis import (
    run_bapr_v3_independent_specialist_audit as audit,
)


EVENT_SEEDS = (1100, 1200, 1300, 1400, 1500)


def group_path(results_root: Path, family: str, event_seed: int) -> Path:
    return (
        results_root / family / protocol.env_short()
        / f"event_seed_{event_seed}" / "group.json"
    )


def load_groups(results_root: Path, family: str) -> dict[int, dict]:
    bundles = protocol.validate_family_bundles(family)
    expected_bundle_hashes = {
        name: protocol.fork_protocol.sha256_file(
            protocol.family_bundle_root(family) / name
            / protocol.BUNDLE_MANIFEST)
        for name in bundles
    }
    groups = {}
    for event_seed in EVENT_SEEDS:
        path = group_path(results_root, family, event_seed)
        payload = json.loads(path.read_text(encoding="utf-8"))
        audit.validate_group(payload)
        if (payload.get("family") != family
                or payload.get("env") != protocol.ENV
                or payload.get("training_seed") != protocol.SEED
                or payload.get("event_seed") != event_seed
                or payload.get("bundle_manifest_sha256")
                != expected_bundle_hashes):
            raise ValueError(f"audit group provenance mismatch: {path}")
        groups[event_seed] = payload
    return groups


def _stationary_event_value(group: dict, controller: str) -> float:
    return statistics.mean(
        group["stationary"][controller][str(mode)]["mean"]
        for mode in protocol.MODES)


def _switching_event_value(group: dict, controller: str) -> float:
    return float(group["switching"][controller]["mean"])


def summarize(results_root: Path, family: str) -> dict:
    groups = load_groups(results_root, family)
    controllers = {}
    series = {}
    for controller in audit.CONTROLLERS:
        stationary = [
            _stationary_event_value(groups[seed], controller)
            for seed in EVENT_SEEDS]
        switching = [
            _switching_event_value(groups[seed], controller)
            for seed in EVENT_SEEDS]
        series[controller] = {
            "stationary": stationary, "switching": switching}
        controllers[controller] = {
            "stationary": {
                "per_event_seed": dict(zip(EVENT_SEEDS, stationary)),
                "mean": statistics.mean(stationary),
                "sd": statistics.stdev(stationary),
            },
            "switching": {
                "per_event_seed": dict(zip(EVENT_SEEDS, switching)),
                "mean": statistics.mean(switching),
                "sd": statistics.stdev(switching),
            },
        }

    comparison_controllers = (
        "robust", *(f"fixed_mode_{mode}" for mode in protocol.MODES))
    comparisons = {}
    for controller in comparison_controllers:
        metrics = {}
        for metric in ("stationary", "switching"):
            differences = [
                dynamic - comparison
                for dynamic, comparison in zip(
                    series["dynamic_oracle"][metric],
                    series[controller][metric])
            ]
            mean, lower, upper = paired.paired_interval(differences)
            metrics[metric] = {
                "differences": dict(zip(EVENT_SEEDS, differences)),
                "mean": mean,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "wins": sum(value > 0.0 for value in differences),
                "passes_positive_mean": mean > 0.0,
            }
        comparisons[controller] = metrics

    matrix = {}
    diagonal_wins = 0
    for physics_mode in protocol.MODES:
        row = {}
        for specialist_mode in protocol.MODES:
            row[str(specialist_mode)] = statistics.mean(
                groups[seed]["stationary"][
                    f"fixed_mode_{specialist_mode}"][str(physics_mode)][
                        "mean"]
                for seed in EVENT_SEEDS)
        best = max(protocol.MODES, key=lambda mode: row[str(mode)])
        diagonal = best == physics_mode
        diagonal_wins += int(diagonal)
        matrix[str(physics_mode)] = {
            "specialist_mean_returns": row,
            "best_specialist": best,
            "diagonal": diagonal,
        }

    stationary_pass = all(
        comparisons[name]["stationary"]["passes_positive_mean"]
        for name in comparison_controllers)
    switching_pass = all(
        comparisons[name]["switching"]["passes_positive_mean"]
        for name in comparison_controllers)
    gate = {
        "dynamic_beats_all_stationary": stationary_pass,
        "dynamic_beats_all_switching": switching_pass,
        "diagonal_wins": diagonal_wins,
        "diagonal_at_least_3_of_4": diagonal_wins >= 3,
        "passed": stationary_pass and switching_pass and diagonal_wins >= 3,
    }
    return {
        "schema": "bapr.v3-independent-specialist-analysis.v1",
        "family": family,
        "env": protocol.env_short(),
        "training_seed": protocol.SEED,
        "event_seeds": list(EVENT_SEEDS),
        "event_seed_role": "paired evaluation streams; not policy seeds",
        "controllers": controllers,
        "dynamic_comparisons": comparisons,
        "stationary_specialist_matrix": matrix,
        "gate": gate,
    }


def report(summary: dict) -> str:
    controllers = summary["controllers"]
    comparisons = summary["dynamic_comparisons"]
    gate = summary["gate"]
    lines = [
        f"# Independent specialist ladder: {summary['family']} / "
        f"{summary['env']}",
        "",
        "Five paired event streams evaluate one policy seed. Each mode "
        "specialist has an independent policy, critic, target critic, entropy "
        "temperature, optimizer state, and replay continuation.",
        "",
        "| Controller | Stationary mean ± SD | Switching mean ± SD |",
        "|---|---:|---:|",
    ]
    for name in audit.CONTROLLERS:
        stationary = controllers[name]["stationary"]
        switching = controllers[name]["switching"]
        lines.append(
            f"| `{name}` | {stationary['mean']:.1f} ± "
            f"{stationary['sd']:.1f} | {switching['mean']:.1f} ± "
            f"{switching['sd']:.1f} |")
    lines += [
        "",
        "| Dynamic comparison | Stationary difference (95% CI) | Wins | "
        "Switching difference (95% CI) | Wins |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in comparisons.items():
        stationary = metrics["stationary"]
        switching = metrics["switching"]
        lines.append(
            f"| dynamic - `{name}` | {stationary['mean']:+.1f} "
            f"[{stationary['ci95_lower']:+.1f},"
            f"{stationary['ci95_upper']:+.1f}] | "
            f"{stationary['wins']}/5 | {switching['mean']:+.1f} "
            f"[{switching['ci95_lower']:+.1f},"
            f"{switching['ci95_upper']:+.1f}] | "
            f"{switching['wins']}/5 |")
    lines += [
        "",
        "| Physics mode | Specialist 0 | Specialist 1 | Specialist 2 | "
        "Specialist 3 | Best | Diagonal? |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in protocol.MODES:
        row = summary["stationary_specialist_matrix"][str(mode)]
        values = row["specialist_mean_returns"]
        lines.append(
            f"| {mode} | {values['0']:.1f} | {values['1']:.1f} | "
            f"{values['2']:.1f} | {values['3']:.1f} | "
            f"{row['best_specialist']} | "
            f"{'yes' if row['diagonal'] else 'no'} |")
    lines += [
        "",
        f"- Dynamic beats robust and every fixed specialist in stationary "
        f"mean: **{'PASS' if gate['dynamic_beats_all_stationary'] else 'FAIL'}**",
        f"- Dynamic beats robust and every fixed specialist in switching "
        f"mean: **{'PASS' if gate['dynamic_beats_all_switching'] else 'FAIL'}**",
        f"- Diagonal stationary optima: **{gate['diagonal_wins']}/4**",
        f"- Preregistered specialist-headroom gate: "
        f"**{'PASS' if gate['passed'] else 'FAIL'}**",
        "",
        "A failed gate blocks learned-estimator training. Event seeds are "
        "evaluation streams, not independently trained policy seeds.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("legacy", "stochastic_headroom", "structured_channel"),
        default="legacy",
    )
    parser.add_argument("--env")
    parser.add_argument("--family", required=True)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    if args.profile == "stochastic_headroom":
        if not args.env:
            parser.error("--env is required for stochastic_headroom")
        protocol.configure_stochastic_headroom(args.env)
    elif args.profile == "structured_channel":
        if not args.env:
            parser.error("--env is required for structured_channel")
        protocol.configure_structured_channel_headroom(args.env)
    elif args.env and args.env != protocol.ENV:
        parser.error(f"legacy profile only supports {protocol.ENV}")
    protocol._require_family(args.family)
    if args.results_root is None:
        args.results_root = protocol.AUDIT_BASE
    return args


def main() -> None:
    args = parse_args()
    summary = summarize(args.results_root.resolve(), args.family)
    protocol.write_json_atomic(args.json_output.resolve(), summary)
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(report(summary), encoding="utf-8")
    print(
        f"INDEPENDENT SPECIALIST ANALYSIS COMPLETE: family={args.family} "
        f"gate={'PASS' if summary['gate']['passed'] else 'FAIL'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
