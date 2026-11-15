#!/usr/bin/env python3
"""Strictly validate and summarize one BAPR-v3 shared-fork audit pair."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if str(_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORT_ROOT))

from jax_experiments.analysis import (
    analyze_bapr_v3_budget_matched_fork_audit as audit,
)


def load_pair_outputs(
    pair_root: Path,
    results_root: Path,
    family: str,
    env_short: str,
    event_seeds: tuple[int, ...],
) -> dict[tuple[str, str, str, int], audit.v1.OutputMetrics]:
    env = f"{env_short}-v2"
    pair_dir = audit.pair_directory(
        pair_root, family, env, audit.EXPECTED_TRAINING_SEED)
    audit.validate_pair_provenance(
        pair_dir, family, env, audit.EXPECTED_TRAINING_SEED)

    outputs = {}
    for event_seed in event_seeds:
        group = audit.validate_group_bundle(
            audit.group_directory(
                results_root, family, env, event_seed),
            pair_dir,
            family,
            env,
            audit.EXPECTED_TRAINING_SEED,
            event_seed,
        )
        for source, metrics in group.items():
            outputs[(family, env_short, source, event_seed)] = metrics
    expected_outputs = len(event_seeds) * len(audit.SOURCES)
    if len(outputs) != expected_outputs:
        raise ValueError(
            f"expected {expected_outputs} controller outputs, found "
            f"{len(outputs)}")
    return outputs


def render_pair_report(
    pair_root: Path,
    results_root: Path,
    family: str,
    env_short: str,
    event_seeds: tuple[int, ...],
    *,
    outputs: dict[
        tuple[str, str, str, int], audit.v1.OutputMetrics
    ] | None = None,
) -> str:
    if outputs is None:
        outputs = load_pair_outputs(
            pair_root, results_root, family, env_short, event_seeds)
    expected_outputs = len(event_seeds) * len(audit.SOURCES)

    pair_lines, passed = audit.v1.render_pair(
        outputs, family, env_short, event_seeds)
    lines = [
        f"# BAPR-v3 shared-fork audit: {family} / {env_short}",
        "",
        f"Validated one exact producer pair and **{expected_outputs}/"
        f"{expected_outputs}** controller outputs across event seeds "
        f"`{','.join(map(str, event_seeds))}`.",
        "",
        f"Pair numerical gate: **{'PASS' if passed else 'FAIL'}**.",
        "",
        *pair_lines,
        "",
        "The event seeds are paired evaluation streams for one training seed "
        "(`seed=0`), not independent policy seeds.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def summarize_pair(
    pair_root: Path,
    results_root: Path,
    family: str,
    env_short: str,
    event_seeds: tuple[int, ...],
    outputs: dict[tuple[str, str, str, int], audit.v1.OutputMetrics],
) -> dict:
    controllers = {}
    series = {}
    for source in audit.SOURCES:
        stationary = audit.v1.values_for(
            outputs, family, env_short, source.directory,
            event_seeds, "stationary")
        switching = audit.v1.values_for(
            outputs, family, env_short, source.directory,
            event_seeds, "switching")
        series[source.directory] = {
            "stationary": stationary,
            "switching": switching,
        }
        controllers[source.directory] = {
            "label": source.label,
            "stationary": {
                "per_event_seed": dict(zip(event_seeds, stationary)),
                "mean": statistics.mean(stationary),
                "sd": statistics.stdev(stationary),
            },
            "switching": {
                "per_event_seed": dict(zip(event_seeds, switching)),
                "mean": statistics.mean(switching),
                "sd": statistics.stdev(switching),
            },
        }

    comparison_sources = (
        "robust", *(f"fixed_mode_{mode}" for mode in audit.FIXED_MODES))
    comparisons = {}
    for comparison_source in comparison_sources:
        metrics = {}
        for metric in ("stationary", "switching"):
            differences = [
                oracle - comparison
                for oracle, comparison in zip(
                    series["oracle"][metric],
                    series[comparison_source][metric],
                )
            ]
            mean, lower, upper = audit.v1.paired_interval(differences)
            metrics[metric] = {
                "differences": dict(zip(event_seeds, differences)),
                "mean": mean,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "wins": sum(value > 0.0 for value in differences),
                "passes_positive_mean": mean > 0.0,
            }
        comparisons[comparison_source] = metrics

    fixed_context_matrix = {}
    diagonal_wins = 0
    for task_mode in audit.FIXED_MODES:
        row = {}
        for context_mode in audit.FIXED_MODES:
            values = [
                outputs[(
                    family, env_short, f"fixed_mode_{context_mode}",
                    event_seed,
                )].task_returns[task_mode]
                for event_seed in event_seeds
            ]
            row[str(context_mode)] = statistics.mean(values)
        best_mode = max(audit.FIXED_MODES, key=lambda mode: row[str(mode)])
        diagonal = best_mode == task_mode
        diagonal_wins += int(diagonal)
        fixed_context_matrix[str(task_mode)] = {
            "context_mean_returns": row,
            "best_context": best_mode,
            "diagonal": diagonal,
        }

    stationary_pass = all(
        comparisons[source]["stationary"]["passes_positive_mean"]
        for source in comparison_sources)
    switching_pass = all(
        comparisons[source]["switching"]["passes_positive_mean"]
        for source in comparison_sources)
    diagonal_pass = diagonal_wins >= 3
    return {
        "schema": "bapr.v3-ant-specialization-pair-summary.v1",
        "variant": pair_root.name,
        "family": family,
        "env": env_short,
        "training_seed": audit.EXPECTED_TRAINING_SEED,
        "event_seeds": list(event_seeds),
        "event_seed_role": "paired evaluation streams; not policy seeds",
        "pair_root": str(pair_root),
        "results_root": str(results_root),
        "validated_controller_outputs": len(outputs),
        "controllers": controllers,
        "oracle_comparisons": comparisons,
        "fixed_context_matrix": fixed_context_matrix,
        "gate": {
            "oracle_beats_all_stationary": stationary_pass,
            "oracle_beats_all_switching": switching_pass,
            "diagonal_wins": diagonal_wins,
            "diagonal_at_least_3_of_4": diagonal_pass,
            "passed": stationary_pass and switching_pass and diagonal_pass,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", type=Path, default=audit.DEFAULT_PAIR_ROOT)
    parser.add_argument(
        "--results-root", type=Path, default=audit.DEFAULT_RESULTS_ROOT)
    parser.add_argument("--family", required=True, choices=audit.FAMILIES)
    parser.add_argument("--env", required=True, choices=audit.ENVS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        pair_root = args.pair_root.resolve()
        results_root = args.results_root.resolve()
        outputs = load_pair_outputs(
            pair_root,
            results_root,
            args.family,
            args.env,
            audit.DEFAULT_EVENT_SEEDS,
        )
        report = render_pair_report(
            pair_root,
            results_root,
            args.family,
            args.env,
            audit.DEFAULT_EVENT_SEEDS,
            outputs=outputs,
        )
        summary = summarize_pair(
            pair_root,
            results_root,
            args.family,
            args.env,
            audit.DEFAULT_EVENT_SEEDS,
            outputs,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"PAIR AUDIT INCOMPLETE OR INVALID: {exc}") from exc
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
        print(f"Wrote {output}", flush=True)
    if args.json_output:
        json_output = args.json_output.resolve()
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {json_output}", flush=True)
    sys.stdout.write(report)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
