#!/usr/bin/env python3
"""Validate and summarize the exploratory v1 BAPR-v3 budget audit.

Expected directory layout beneath ``--results-root``::

    FAMILY/ENV/SOURCE/event_seed_SEED/{summary,task_returns,
                                      switching_returns,switching_trace}.csv

``FAMILY`` is ``deterministic_mean`` or ``mean_variance``; ``ENV`` is
``Ant`` or ``HalfCheetah``; and ``SOURCE`` is ``robust``, ``oracle``, or
``fixed_mode_0`` through ``fixed_mode_3``.  With the five preregistered event
seeds this is exactly 4 family/environment pairs x 6 sources x 5 streams =
120 evaluation outputs.

The script is intentionally all-or-nothing: it validates every expected file,
row count, checkpoint identity, run identity, and evaluation context before it
prints any scientific summary.  It therefore cannot silently analyze a
partially synchronized audit.

The underlying v1 controller arms were independent heterogeneous-runtime
jobs, not continuations of a shared checkpoint.  Consequently the return
screen below is descriptive and cannot pass the causal mechanism gate even if
all numerical return criteria are positive.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = (
    ROOT / "jax_experiments" / "results_bapr_v3_budget_matched_audit_v1")

FAMILIES = ("deterministic_mean", "mean_variance")
ENVS = ("Ant", "HalfCheetah")
FIXED_MODES = (0, 1, 2, 3)
DEFAULT_EVENT_SEEDS = (1100, 1200, 1300, 1400, 1500)
EXPECTED_CHECKPOINT_NEXT_ITER = 1400
EXPECTED_CHECKPOINT_TOTAL_STEPS = 5_600_000
EXPECTED_TRAINING_SEED = 0
EXPECTED_ROWS = {
    "summary.csv": 3,
    "task_returns.csv": 4,
    "switching_returns.csv": 5,
    "switching_trace.csv": 5000,
}
T_CRITICAL_DF4_95 = 2.776445105


@dataclass(frozen=True)
class SourceSpec:
    directory: str
    label: str
    training_variant: str
    context_source: str
    fixed_mode: int | None = None


SOURCES = (
    SourceSpec("robust", "equal-budget robust", "robust_long", "robust"),
    SourceSpec("oracle", "dynamic oracle", "oracle_direct", "oracle"),
    *(SourceSpec(
        f"fixed_mode_{mode}", f"fixed context {mode}", "oracle_direct",
        "oracle", mode) for mode in FIXED_MODES),
)
SOURCE_BY_DIRECTORY = {source.directory: source for source in SOURCES}


@dataclass(frozen=True)
class OutputMetrics:
    stationary: float
    switching: float
    task_returns: dict[int, float]


def read_rows(path: Path, expected_count: int) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"missing output file: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"missing CSV header: {path}")
        rows = list(reader)
    if len(rows) != expected_count:
        raise ValueError(
            f"wrong row count in {path}: found {len(rows)}, "
            f"expected {expected_count}")
    return rows


def require_text(
    row: dict[str, str], key: str, expected: str, path: Path, row_number: int,
) -> None:
    actual = row.get(key)
    if actual != expected:
        raise ValueError(
            f"wrong {key} in {path} row {row_number}: "
            f"found {actual!r}, expected {expected!r}")


def finite_float(
    row: dict[str, str], key: str, path: Path, row_number: int,
) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid {key} in {path} row {row_number}") from exc
    if not math.isfinite(value):
        raise ValueError(
            f"non-finite {key} in {path} row {row_number}: {value}")
    return value


def exact_int(
    row: dict[str, str], key: str, path: Path, row_number: int,
) -> int:
    value = finite_float(row, key, path, row_number)
    if not value.is_integer():
        raise ValueError(
            f"non-integral {key} in {path} row {row_number}: {value}")
    return int(value)


def expected_run_name(
    family: str, env: str, source: SourceSpec, training_seed: int,
) -> str:
    return (
        f"budget_v1_{family}_{source.training_variant}_{env}_s{training_seed}")


def validate_metadata(
    rows: list[dict[str, str]],
    path: Path,
    family: str,
    env: str,
    source: SourceSpec,
    checkpoint_next_iter: int,
    checkpoint_total_steps: int,
    training_seed: int,
) -> None:
    run_name = expected_run_name(family, env, source, training_seed)
    expected_oracle_mode = (
        "dynamic" if source.fixed_mode is None else str(source.fixed_mode))
    for row_number, row in enumerate(rows, start=2):
        require_text(row, "algo", "bapr_v3", path, row_number)
        require_text(row, "env", env, path, row_number)
        require_text(row, "run_name", run_name, path, row_number)
        require_text(
            row, "eval_context_source", source.context_source, path,
            row_number)
        require_text(
            row, "eval_oracle_mode_id", expected_oracle_mode, path,
            row_number)
        require_text(row, "eval_advantage", "off", path, row_number)
        require_text(
            row, "heldout_task_stream", "validation", path, row_number)
        if exact_int(
                row, "checkpoint_next_iter", path,
                row_number) != checkpoint_next_iter:
            raise ValueError(
                f"wrong checkpoint_next_iter in {path} row {row_number}")
        if exact_int(
                row, "checkpoint_total_steps", path,
                row_number) != checkpoint_total_steps:
            raise ValueError(
                f"wrong checkpoint_total_steps in {path} row {row_number}")
        if exact_int(row, "seed", path, row_number) != training_seed:
            raise ValueError(f"wrong training seed in {path} row {row_number}")


def select_summary_metrics(
    rows: list[dict[str, str]], path: Path,
) -> tuple[float, float]:
    stationary_train = [
        (index, row) for index, row in enumerate(rows, start=2)
        if row.get("metric_group") == "stationary"
        and row.get("split") == "train"
    ]
    stationary_test = [
        (index, row) for index, row in enumerate(rows, start=2)
        if row.get("metric_group") == "stationary"
        and row.get("split") == "test"
    ]
    switching = [
        (index, row) for index, row in enumerate(rows, start=2)
        if row.get("metric_group") == "switching"
        and row.get("split") == "test_sequence"
    ]
    if not (
        len(stationary_train) == len(stationary_test) == len(switching) == 1
    ):
        raise ValueError(
            f"summary groups in {path} must be exactly stationary/train, "
            "stationary/test, and switching/test_sequence")
    stationary_index, stationary_row = stationary_test[0]
    switching_index, switching_row = switching[0]
    if exact_int(
            stationary_row, "n_tasks", path,
            stationary_index) != len(FIXED_MODES):
        raise ValueError(
            f"stationary test summary in {path} must cover four tasks")
    if exact_int(
            switching_row, "switching_episodes", path,
            switching_index) != len(DEFAULT_EVENT_SEEDS):
        raise ValueError(
            f"switching summary in {path} must cover five episodes")
    return (
        finite_float(stationary_row, "return_mean", path, stationary_index),
        finite_float(
            switching_row, "switch_return_mean", path, switching_index),
    )


def validate_task_returns(
    rows: list[dict[str, str]], path: Path,
) -> dict[int, float]:
    task_returns: dict[int, float] = {}
    for row_number, row in enumerate(rows, start=2):
        require_text(row, "split", "test", path, row_number)
        task_index = exact_int(row, "task_index", path, row_number)
        if task_index not in FIXED_MODES:
            raise ValueError(
                f"unexpected task_index={task_index} in {path} row {row_number}")
        if task_index in task_returns:
            raise ValueError(f"duplicate task_index={task_index} in {path}")
        task_returns[task_index] = finite_float(
            row, "return_mean", path, row_number)
    if set(task_returns) != set(FIXED_MODES):
        raise ValueError(f"task rows in {path} do not cover modes 0-3 exactly")
    return task_returns


def validate_switching_returns(
    rows: list[dict[str, str]], path: Path,
) -> None:
    episodes = set()
    for row_number, row in enumerate(rows, start=2):
        episode = exact_int(row, "episode", path, row_number)
        if episode in episodes:
            raise ValueError(f"duplicate switching episode {episode} in {path}")
        episodes.add(episode)
        finite_float(row, "return", path, row_number)
    if episodes != set(range(5)):
        raise ValueError(
            f"switching episodes in {path} must be exactly 0-4")


def validate_switching_trace(
    rows: list[dict[str, str]], path: Path,
) -> None:
    steps_by_episode: dict[int, set[int]] = {episode: set() for episode in range(5)}
    for row_number, row in enumerate(rows, start=2):
        episode = exact_int(row, "episode", path, row_number)
        step = exact_int(row, "step", path, row_number)
        if episode not in steps_by_episode:
            raise ValueError(
                f"unexpected trace episode {episode} in {path} row {row_number}")
        if step in steps_by_episode[episode]:
            raise ValueError(
                f"duplicate trace episode/step ({episode}, {step}) in {path}")
        steps_by_episode[episode].add(step)
    expected_steps = set(range(1, 1001))
    for episode, steps in steps_by_episode.items():
        if steps != expected_steps:
            raise ValueError(
                f"trace episode {episode} in {path} must contain steps 1-1000")


def output_directory(
    results_root: Path, family: str, env: str, source: SourceSpec,
    event_seed: int,
) -> Path:
    return (
        results_root / family / env / source.directory
        / f"event_seed_{event_seed}")


def expected_output_directories(
    results_root: Path, event_seeds: tuple[int, ...],
) -> set[Path]:
    return {
        output_directory(results_root, family, env, source, event_seed)
        for family in FAMILIES
        for env in ENVS
        for source in SOURCES
        for event_seed in event_seeds
    }


def validate_tree_exactness(
    results_root: Path, expected_directories: set[Path],
) -> None:
    if not results_root.is_dir():
        raise ValueError(f"results root does not exist: {results_root}")
    discovered = {
        path for path in results_root.rglob("event_seed_*") if path.is_dir()
    }
    missing = sorted(expected_directories - discovered)
    unexpected = sorted(discovered - expected_directories)
    if missing or unexpected:
        details = []
        if missing:
            details.append(
                f"missing {len(missing)} output directories (first: {missing[0]})")
        if unexpected:
            details.append(
                f"unexpected {len(unexpected)} output directories "
                f"(first: {unexpected[0]})")
        raise ValueError("; ".join(details))


def validate_output(
    directory: Path,
    family: str,
    env: str,
    source: SourceSpec,
    checkpoint_next_iter: int,
    checkpoint_total_steps: int,
    training_seed: int,
) -> OutputMetrics:
    outputs = {
        filename: read_rows(directory / filename, expected_count)
        for filename, expected_count in EXPECTED_ROWS.items()
    }
    for filename, rows in outputs.items():
        validate_metadata(
            rows, directory / filename, family, env, source,
            checkpoint_next_iter, checkpoint_total_steps, training_seed)

    stationary, switching = select_summary_metrics(
        outputs["summary.csv"], directory / "summary.csv")
    task_returns = validate_task_returns(
        outputs["task_returns.csv"], directory / "task_returns.csv")
    validate_switching_returns(
        outputs["switching_returns.csv"],
        directory / "switching_returns.csv")
    validate_switching_trace(
        outputs["switching_trace.csv"], directory / "switching_trace.csv")
    return OutputMetrics(stationary, switching, task_returns)


def load_complete_audit(
    results_root: Path,
    event_seeds: tuple[int, ...],
    checkpoint_next_iter: int,
    checkpoint_total_steps: int,
    training_seed: int,
) -> dict[tuple[str, str, str, int], OutputMetrics]:
    expected_directories = expected_output_directories(results_root, event_seeds)
    validate_tree_exactness(results_root, expected_directories)
    outputs = {}
    for family in FAMILIES:
        for env in ENVS:
            for source in SOURCES:
                for event_seed in event_seeds:
                    directory = output_directory(
                        results_root, family, env, source, event_seed)
                    outputs[(family, env, source.directory, event_seed)] = (
                        validate_output(
                            directory, family, env, source,
                            checkpoint_next_iter, checkpoint_total_steps,
                            training_seed))
    if len(outputs) != 120:
        raise ValueError(
            f"internal audit cardinality error: validated {len(outputs)}, "
            "expected 120")
    return outputs


def mean_sd(values: list[float]) -> tuple[float, float]:
    if len(values) != 5:
        raise ValueError("mean/SD requires five preregistered event streams")
    return statistics.mean(values), statistics.stdev(values)


def paired_interval(values: list[float]) -> tuple[float, float, float]:
    mean, sd = mean_sd(values)
    half_width = T_CRITICAL_DF4_95 * sd / math.sqrt(len(values))
    return mean, mean - half_width, mean + half_width


def fmt_mean_sd(values: list[float]) -> str:
    mean, sd = mean_sd(values)
    return f"{mean:.1f} ± {sd:.1f}"


def fmt_interval(values: list[float]) -> str:
    mean, lower, upper = paired_interval(values)
    return f"{mean:+.1f} [{lower:+.1f}, {upper:+.1f}]"


def values_for(
    outputs: dict[tuple[str, str, str, int], OutputMetrics],
    family: str,
    env: str,
    source: str,
    event_seeds: tuple[int, ...],
    metric: str,
) -> list[float]:
    return [
        float(getattr(outputs[(family, env, source, seed)], metric))
        for seed in event_seeds
    ]


def render_pair(
    outputs: dict[tuple[str, str, str, int], OutputMetrics],
    family: str,
    env: str,
    event_seeds: tuple[int, ...],
) -> tuple[list[str], bool]:
    lines = [f"## {family} / {env}", ""]
    lines.extend([
        "| Controller | Stationary mean ± SD | Switching mean ± SD |",
        "|---|---:|---:|",
    ])
    series: dict[str, dict[str, list[float]]] = {}
    for source in SOURCES:
        stationary = values_for(
            outputs, family, env, source.directory, event_seeds, "stationary")
        switching = values_for(
            outputs, family, env, source.directory, event_seeds, "switching")
        series[source.directory] = {
            "stationary": stationary,
            "switching": switching,
        }
        lines.append(
            f"| {source.label} | {fmt_mean_sd(stationary)} | "
            f"{fmt_mean_sd(switching)} |")

    lines.extend([
        "",
        "| Paired comparison | Stationary difference (95% CI) | Wins | "
        "Switching difference (95% CI) | Wins | Mean gate |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    comparison_sources = (
        "robust", *(f"fixed_mode_{mode}" for mode in FIXED_MODES))
    comparison_passes = []
    for comparison_source in comparison_sources:
        comparison_label = SOURCE_BY_DIRECTORY[comparison_source].label
        stationary_diff = [
            oracle - comparison
            for oracle, comparison in zip(
                series["oracle"]["stationary"],
                series[comparison_source]["stationary"])
        ]
        switching_diff = [
            oracle - comparison
            for oracle, comparison in zip(
                series["oracle"]["switching"],
                series[comparison_source]["switching"])
        ]
        stationary_pass = statistics.mean(stationary_diff) > 0.0
        switching_pass = statistics.mean(switching_diff) > 0.0
        comparison_passes.extend([stationary_pass, switching_pass])
        mean_gate = stationary_pass and switching_pass
        lines.append(
            f"| dynamic oracle - {comparison_label} | "
            f"{fmt_interval(stationary_diff)} | "
            f"{sum(value > 0 for value in stationary_diff)}/5 | "
            f"{fmt_interval(switching_diff)} | "
            f"{sum(value > 0 for value in switching_diff)}/5 | "
            f"{'PASS' if mean_gate else 'FAIL'} |")

    lines.extend([
        "",
        "| Physics mode | Fixed context 0 | Fixed context 1 | "
        "Fixed context 2 | Fixed context 3 | Best | Diagonal? |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    diagonal_wins = 0
    for task_mode in FIXED_MODES:
        cells = []
        for context_mode in FIXED_MODES:
            values = [
                outputs[(
                    family, env, f"fixed_mode_{context_mode}", event_seed
                )].task_returns[task_mode]
                for event_seed in event_seeds
            ]
            cells.append(statistics.mean(values))
        best_mode = max(FIXED_MODES, key=lambda mode: cells[mode])
        diagonal = best_mode == task_mode
        diagonal_wins += int(diagonal)
        lines.append(
            f"| {task_mode} | "
            + " | ".join(f"{value:.1f}" for value in cells)
            + f" | {best_mode} | {'yes' if diagonal else 'no'} |")

    stationary_pass = all(
        statistics.mean(series["oracle"]["stationary"])
        > statistics.mean(series[source]["stationary"])
        for source in comparison_sources)
    switching_pass = all(
        statistics.mean(series["oracle"]["switching"])
        > statistics.mean(series[source]["switching"])
        for source in comparison_sources)
    diagonal_pass = diagonal_wins >= 3
    pair_pass = (
        stationary_pass and switching_pass and diagonal_pass
        and all(comparison_passes))
    lines.extend([
        "",
        f"- Dynamic oracle beats robust and every fixed context in stationary "
        f"paired mean: **{'PASS' if stationary_pass else 'FAIL'}**",
        f"- Dynamic oracle beats robust and every fixed context in switching "
        f"paired mean: **{'PASS' if switching_pass else 'FAIL'}**",
        f"- Stationary fixed-context diagonal optima: **{diagonal_wins}/4 "
        f"({'PASS' if diagonal_pass else 'FAIL'})**",
        f"- Preregistered family/environment gate: "
        f"**{'PASS' if pair_pass else 'FAIL'}**",
        "",
    ])
    return lines, pair_pass


def render_report(
    outputs: dict[tuple[str, str, str, int], OutputMetrics],
    results_root: Path,
    event_seeds: tuple[int, ...],
    checkpoint_next_iter: int,
    checkpoint_total_steps: int,
    training_seed: int,
) -> str:
    lines = [
        "# BAPR-v3 v1 exploratory budget audit",
        "",
        f"Validated **120/120** evaluation outputs under `{results_root}` at "
        f"checkpoint next iteration `{checkpoint_next_iter}`, total steps "
        f"`{checkpoint_total_steps}`, training seed `{training_seed}`, and "
        f"paired event seeds `{', '.join(map(str, event_seeds))}`.",
        "",
        "Gate semantics: paired mean differences must be strictly positive; "
        "95% paired t intervals and stream-wise wins are descriptive and are "
        "reported separately.",
        "",
        "**Causal eligibility: INVALID.** The v1 `robust_long` and "
        "`oracle_direct` arms were independent jobs on heterogeneous "
        "JAX/Flax/Brax runtimes and did not fork from one shared iteration-700 "
        "checkpoint. Numerical PASS labels below are exploratory only and "
        "must not authorize learned-latent training.",
        "",
    ]
    pair_results = []
    family_results: dict[str, list[bool]] = {family: [] for family in FAMILIES}
    for family in FAMILIES:
        for env in ENVS:
            pair_lines, pair_pass = render_pair(
                outputs, family, env, event_seeds)
            lines.extend(pair_lines)
            pair_results.append(pair_pass)
            family_results[family].append(pair_pass)

    lines.extend([
        "## Exploratory numerical gate",
        "",
    ])
    for family in FAMILIES:
        passed = all(family_results[family])
        lines.append(
            f"- `{family}` passes both environments: "
            f"**{'PASS' if passed else 'FAIL'}**")
    overall_pass = all(pair_results)
    lines.extend([
        f"- All four family/environment pairs pass: "
        f"**{'PASS' if overall_pass else 'FAIL'}**",
        "- Causal shared-checkpoint gate: **INVALID / NOT EVALUATED**",
        "",
        "## Scope caveats",
        "",
        "- The five event seeds are paired evaluation streams for one trained "
        "policy seed (`s0`); they are not five independent training seeds.",
        "- These CSV identities cannot establish that `robust_long` and "
        "`oracle_direct` had bit-exact pre-iteration-700 parameters. Exact "
        "pretraining pairing requires shared-checkpoint provenance or parameter "
        "hash evidence outside these evaluation outputs.",
    ])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--event-seed", action="append", type=int)
    parser.add_argument(
        "--checkpoint-next-iter", type=int,
        default=EXPECTED_CHECKPOINT_NEXT_ITER)
    parser.add_argument(
        "--checkpoint-total-steps", type=int,
        default=EXPECTED_CHECKPOINT_TOTAL_STEPS)
    parser.add_argument(
        "--training-seed", type=int, default=EXPECTED_TRAINING_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    event_seeds = tuple(args.event_seed or DEFAULT_EVENT_SEEDS)
    if len(event_seeds) != 5 or len(set(event_seeds)) != 5:
        raise SystemExit(
            "exactly five distinct paired event seeds are required")
    if args.checkpoint_next_iter != EXPECTED_CHECKPOINT_NEXT_ITER:
        raise SystemExit(
            "the preregistered audit requires checkpoint_next_iter=1400")
    if args.checkpoint_total_steps != EXPECTED_CHECKPOINT_TOTAL_STEPS:
        raise SystemExit(
            "the preregistered audit requires checkpoint_total_steps=5600000")
    if args.training_seed != EXPECTED_TRAINING_SEED:
        raise SystemExit("the preregistered audit requires training seed 0")

    try:
        outputs = load_complete_audit(
            args.results_root, event_seeds, args.checkpoint_next_iter,
            args.checkpoint_total_steps, args.training_seed)
    except ValueError as exc:
        raise SystemExit(f"AUDIT INCOMPLETE OR INVALID: {exc}") from exc

    report = render_report(
        outputs, args.results_root, event_seeds,
        args.checkpoint_next_iter, args.checkpoint_total_steps,
        args.training_seed)
    sys.stdout.write(report)


if __name__ == "__main__":
    main()
