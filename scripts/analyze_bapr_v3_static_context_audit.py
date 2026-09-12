#!/usr/bin/env python3
"""Validate and summarize the BAPR-v3 fixed-context counterfactual audit."""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXED_ROOT = (
    ROOT / "jax_experiments" / "results_bapr_v3_static_context_audit_v1")
DEFAULT_BASELINE_ROOT = (
    ROOT / "jax_experiments" / "results_bapr_v3_inverse_audit_v2")
DEFAULT_EVENT_SEEDS = (1100, 1200, 1300, 1400, 1500)
FIXED_MODES = (0, 1, 2, 3)
EXPECTED_ROWS = {
    "summary.csv": 3,
    "task_returns.csv": 4,
    "switching_returns.csv": 5,
    "switching_trace.csv": 5000,
}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"missing audit output: {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def finite_float(row: dict[str, str], key: str, path: Path) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid {key} in {path}") from exc
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key} in {path}: {value}")
    return value


def validate_fixed_output(
    directory: Path,
    fixed_mode: int,
    expected_next_iter: int,
    expected_total_steps: int,
) -> dict[str, list[dict[str, str]]]:
    outputs = {}
    for filename, expected_rows in EXPECTED_ROWS.items():
        path = directory / filename
        rows = read_rows(path)
        if len(rows) != expected_rows:
            raise ValueError(
                f"{path} has {len(rows)} rows, expected {expected_rows}")
        for row in rows:
            if int(finite_float(row, "checkpoint_next_iter", path)) != expected_next_iter:
                raise ValueError(f"wrong checkpoint generation in {path}")
            if int(finite_float(row, "checkpoint_total_steps", path)) != expected_total_steps:
                raise ValueError(f"wrong checkpoint step count in {path}")
            if row.get("eval_context_source") != "oracle":
                raise ValueError(f"wrong context source in {path}")
            if int(finite_float(row, "eval_oracle_mode_id", path)) != fixed_mode:
                raise ValueError(f"wrong fixed context id in {path}")
        outputs[filename] = rows
    return outputs


def summary_metrics(rows: list[dict[str, str]], path: Path) -> tuple[float, float]:
    stationary = [
        row for row in rows
        if row.get("metric_group") == "stationary" and row.get("split") == "test"
    ]
    switching = [row for row in rows if row.get("metric_group") == "switching"]
    if len(stationary) != 1 or len(switching) != 1:
        raise ValueError(f"unexpected summary groups in {path}")
    return (
        finite_float(stationary[0], "return_mean", path),
        finite_float(switching[0], "switch_return_mean", path),
    )


def baseline_metrics(directory: Path) -> tuple[float, float]:
    path = directory / "summary.csv"
    rows = read_rows(path)
    return summary_metrics(rows, path)


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def paired_interval(values: list[float]) -> tuple[float, float, float]:
    if len(values) != 5:
        raise ValueError("the preregistered paired interval requires five streams")
    mean = statistics.mean(values)
    half_width = 2.776445105 * statistics.stdev(values) / math.sqrt(len(values))
    return mean, mean - half_width, mean + half_width


def fmt_mean_sd(values: list[float]) -> str:
    mean, sd = mean_sd(values)
    return f"{mean:.1f}+/-{sd:.1f}"


def fmt_interval(values: list[float]) -> str:
    mean, lower, upper = paired_interval(values)
    return f"{mean:+.1f} [{lower:+.1f},{upper:+.1f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True)
    parser.add_argument("--fixed-root", type=Path, default=DEFAULT_FIXED_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--event-seed", action="append", type=int)
    parser.add_argument("--checkpoint-next-iter", type=int, default=2600)
    parser.add_argument("--checkpoint-total-steps", type=int, default=10_400_000)
    args = parser.parse_args()

    env = args.env.removesuffix("-v2")
    seeds = tuple(args.event_seed or DEFAULT_EVENT_SEEDS)
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise SystemExit("exactly five distinct paired event seeds are required")

    fixed_stationary: dict[int, list[float]] = {mode: [] for mode in FIXED_MODES}
    fixed_switching: dict[int, list[float]] = {mode: [] for mode in FIXED_MODES}
    task_matrix_samples: dict[tuple[int, int], list[float]] = {
        (task_mode, context_mode): []
        for task_mode in FIXED_MODES for context_mode in FIXED_MODES
    }

    for context_mode in FIXED_MODES:
        for seed in seeds:
            directory = (
                args.fixed_root / env / f"fixed_mode_{context_mode}"
                / f"event_seed_{seed}")
            outputs = validate_fixed_output(
                directory, context_mode, args.checkpoint_next_iter,
                args.checkpoint_total_steps)
            stationary, switching = summary_metrics(
                outputs["summary.csv"], directory / "summary.csv")
            fixed_stationary[context_mode].append(stationary)
            fixed_switching[context_mode].append(switching)
            for row in outputs["task_returns.csv"]:
                task_mode = int(finite_float(
                    row, "task_index", directory / "task_returns.csv"))
                if task_mode not in FIXED_MODES:
                    raise ValueError(f"unexpected task mode {task_mode} in {directory}")
                task_matrix_samples[(task_mode, context_mode)].append(
                    finite_float(row, "return_mean", directory / "task_returns.csv"))

    baselines: dict[str, dict[str, list[float]]] = {}
    for source in ("robust", "oracle", "learned"):
        stationary_values = []
        switching_values = []
        for seed in seeds:
            stationary, switching = baseline_metrics(
                args.baseline_root / env / source / f"event_seed_{seed}")
            stationary_values.append(stationary)
            switching_values.append(switching)
        baselines[source] = {
            "stationary": stationary_values,
            "switching": switching_values,
        }

    print(f"# {env} fixed-context audit")
    print()
    print(f"Validated {len(FIXED_MODES) * len(seeds)}/20 outputs at "
          f"next_iter={args.checkpoint_next_iter}, "
          f"steps={args.checkpoint_total_steps}.")
    print()
    print("| Context | Stationary mean+/-SD | Switching mean+/-SD |")
    print("|---|---:|---:|")
    for source, label in (
        ("robust", "robust base"),
        ("oracle", "dynamic oracle"),
        ("learned", "learned online"),
    ):
        print(
            f"| {label} | {fmt_mean_sd(baselines[source]['stationary'])} | "
            f"{fmt_mean_sd(baselines[source]['switching'])} |")
    for mode in FIXED_MODES:
        print(
            f"| fixed mode {mode} | {fmt_mean_sd(fixed_stationary[mode])} | "
            f"{fmt_mean_sd(fixed_switching[mode])} |")

    print()
    print("| Comparison | Stationary paired difference (95% CI) | "
          "Positive streams | Switching paired difference (95% CI) | "
          "Positive streams |")
    print("|---|---:|---:|---:|---:|")
    best_stationary_mode = max(
        FIXED_MODES, key=lambda mode: statistics.mean(fixed_stationary[mode]))
    best_switching_mode = max(
        FIXED_MODES, key=lambda mode: statistics.mean(fixed_switching[mode]))
    per_seed_best_stationary = [
        max(fixed_stationary[mode][index] for mode in FIXED_MODES)
        for index in range(len(seeds))
    ]
    per_seed_best_switching = [
        max(fixed_switching[mode][index] for mode in FIXED_MODES)
        for index in range(len(seeds))
    ]

    comparisons = [
        (
            f"learned - fixed mode {best_stationary_mode}/{best_switching_mode}",
            [
                learned - fixed
                for learned, fixed in zip(
                    baselines["learned"]["stationary"],
                    fixed_stationary[best_stationary_mode])
            ],
            [
                learned - fixed
                for learned, fixed in zip(
                    baselines["learned"]["switching"],
                    fixed_switching[best_switching_mode])
            ],
        ),
        (
            "learned - per-stream best fixed",
            [
                learned - fixed
                for learned, fixed in zip(
                    baselines["learned"]["stationary"],
                    per_seed_best_stationary)
            ],
            [
                learned - fixed
                for learned, fixed in zip(
                    baselines["learned"]["switching"],
                    per_seed_best_switching)
            ],
        ),
        (
            "dynamic oracle - per-stream best fixed",
            [
                oracle - fixed
                for oracle, fixed in zip(
                    baselines["oracle"]["stationary"],
                    per_seed_best_stationary)
            ],
            [
                oracle - fixed
                for oracle, fixed in zip(
                    baselines["oracle"]["switching"],
                    per_seed_best_switching)
            ],
        ),
    ]
    for label, stationary, switching in comparisons:
        print(
            f"| {label} | {fmt_interval(stationary)} | "
            f"{sum(value > 0 for value in stationary)}/5 | "
            f"{fmt_interval(switching)} | "
            f"{sum(value > 0 for value in switching)}/5 |")

    print()
    print("| Physics mode | Context 0 | Context 1 | Context 2 | Context 3 | Best |")
    print("|---:|---:|---:|---:|---:|---:|")
    diagonal_wins = 0
    for task_mode in FIXED_MODES:
        values = [
            statistics.mean(task_matrix_samples[(task_mode, context_mode)])
            for context_mode in FIXED_MODES
        ]
        best_mode = max(FIXED_MODES, key=lambda mode: values[mode])
        diagonal_wins += int(best_mode == task_mode)
        cells = " | ".join(f"{value:.1f}" for value in values)
        print(f"| {task_mode} | {cells} | {best_mode} |")
    print()
    print(f"Diagonal-optimal rows: {diagonal_wins}/4")


if __name__ == "__main__":
    main()
