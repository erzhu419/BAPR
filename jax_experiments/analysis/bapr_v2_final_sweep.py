"""Summarize corrected BAPR-v2 full-task and switching-stream sweeps."""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PHASE1_ROOT = ROOT / "jax_experiments" / "results_bapr_v2_phase1"
CAPACITY_ROOT = ROOT / "jax_experiments" / "results_bapr_v2_oracle_capacity"
SWEEP_ROOT = (
    ROOT / "jax_experiments" / "results_bapr_v2_final_sweeps_streamfix2"
)
SWITCH_ROOT = (
    ROOT / "jax_experiments" / "results_bapr_v2_final_sweeps_streamfix3"
)
REPORT = ROOT / "reports" / "bapr_v2_final_sweep_2026-07-10.md"
CSV_PATH = ROOT / "reports" / "bapr_v2_final_sweep_2026-07-10.csv"
ENVS = ("Ant", "HalfCheetah", "Hopper", "Walker2d")


@dataclass
class RunMetrics:
    run_name: str
    variant: str
    env: str
    family: str
    train_mean: float
    test_mean: float
    test_worst: float
    test_p10: float
    switching_mean: float
    switch_latent_span: float
    switch_auc: float
    detection_rate: float
    detection_delay: float
    latent_mae: float
    latent_correlation: float
    termination_count: float


def finite(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def fmt(value, digits=1):
    value = finite(value)
    return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}"


def gain(value, base):
    value, base = finite(value), finite(base)
    if not math.isfinite(value) or not math.isfinite(base) or abs(base) < 1e-9:
        return math.nan
    return 100.0 * (value - base) / abs(base)


def last_iteration(run_dir: Path) -> int:
    path = run_dir / "logs" / "iteration.npy"
    try:
        values = np.load(path)
        return int(values[-1]) if len(values) else -1
    except Exception:
        return -1


def expected_runs(expected_iteration: int) -> list[str]:
    names = []
    for root, pattern in ((PHASE1_ROOT, "v82*_s0"),
                          (CAPACITY_ROOT, "v83*_s0")):
        for run_dir in sorted(root.glob(pattern)):
            if last_iteration(run_dir) >= expected_iteration:
                names.append(run_dir.name)
    return names


def completed_result_runs(*roots: Path) -> list[str]:
    """Recover evaluable run names after training checkpoints are archived."""
    names = set()
    for root in roots:
        if not root.is_dir():
            continue
        for summary in root.glob("v8*_s0/summary.csv"):
            if summary.is_file() and summary.stat().st_size > 0:
                names.add(summary.parent.name)
    return sorted(names)


def eligible_runs(
        expected_iteration: int,
        stationary_results: Path,
        switching_results: Path) -> list[str]:
    return sorted(set(expected_runs(expected_iteration)) | set(
        completed_result_runs(stationary_results, switching_results)))


def split_run_name(run_name: str) -> tuple[str, str]:
    for env in ENVS:
        suffix = f"_{env}_s0"
        if run_name.endswith(suffix):
            return run_name[:-len(suffix)], env
    raise ValueError(f"unrecognized BAPR-v2 run name: {run_name}")


def family_for(variant: str) -> str:
    if variant.startswith(("v82a_", "v82b_")):
        return "robust"
    if variant.startswith(("v82c_", "v82d_", "v83")):
        return "oracle"
    if variant.startswith(("v82e_", "v82f_", "v82g_", "v82h_", "v82i_")):
        return "learned"
    return "other"


def load_run(stationary_dir: Path, switching_dir: Path) -> RunMetrics | None:
    stationary_summary_path = stationary_dir / "summary.csv"
    task_path = stationary_dir / "task_returns.csv"
    switching_summary_path = switching_dir / "summary.csv"
    trace_path = switching_dir / "switching_trace.csv"
    if not all(path.is_file() and path.stat().st_size > 0 for path in (
            stationary_summary_path, task_path,
            switching_summary_path, trace_path)):
        return None
    stationary_summary = list(csv.DictReader(stationary_summary_path.open()))
    switching_summary = list(csv.DictReader(switching_summary_path.open()))
    tasks = list(csv.DictReader(task_path.open()))
    stationary = {
        row.get("split"): row for row in stationary_summary
        if row.get("metric_group") == "stationary"
    }
    switching = next((
        row for row in switching_summary
        if row.get("metric_group") == "switching"
    ), {})
    test_returns = [
        finite(row.get("return_mean")) for row in tasks
        if row.get("split") == "test"
        and math.isfinite(finite(row.get("return_mean")))
    ]
    if not test_returns or "train" not in stationary or "test" not in stationary:
        return None
    variant, env = split_run_name(stationary_dir.name)
    return RunMetrics(
        run_name=stationary_dir.name,
        variant=variant,
        env=env,
        family=family_for(variant),
        train_mean=finite(stationary["train"].get("return_mean")),
        test_mean=finite(stationary["test"].get("return_mean")),
        test_worst=min(test_returns),
        test_p10=float(np.percentile(test_returns, 10)),
        switching_mean=finite(switching.get("switch_return_mean")),
        switch_latent_span=finite(
            switching.get("switch_latent_span_mean")),
        switch_auc=finite(switching.get("switch_auc")),
        detection_rate=finite(switching.get("detection_rate")),
        detection_delay=finite(switching.get("median_detection_delay")),
        latent_mae=finite(switching.get("latent_mae_after")),
        latent_correlation=finite(
            switching.get("latent_correlation_after")),
        termination_count=finite(switching.get("termination_count_mean")),
    )


def joint_ratio(candidate: RunMetrics, base: RunMetrics) -> float:
    ratios = []
    for value, denominator in (
            (candidate.test_mean, base.test_mean),
            (candidate.switching_mean, base.switching_mean)):
        if (math.isfinite(value) and math.isfinite(denominator)
                and denominator > 1e-9):
            ratios.append(value / denominator)
    return min(ratios) if len(ratios) == 2 else -math.inf


def choose(rows: list[RunMetrics], family: str, base: RunMetrics):
    candidates = [row for row in rows if row.family == family]
    return max(candidates, key=lambda row: joint_ratio(row, base), default=None)


def csv_row(row: RunMetrics) -> dict:
    return {
        key: getattr(row, key) for key in RunMetrics.__dataclass_fields__
    }


def write_csv(path: Path, rows: list[RunMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(RunMetrics.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(csv_row(row) for row in rows)


def build_report(rows: list[RunMetrics], expected: list[str]) -> str:
    complete_names = {row.run_name for row in rows}
    missing = sorted(set(expected) - complete_names)
    lines = [
        "# BAPR-v2 Corrected Final Sweep",
        "",
        "## Audit",
        "",
        f"- Complete sweeps: **{len(rows)}/{len(expected)}** currently eligible final checkpoints.",
        "- Stationary metrics cover all 40 saved train tasks and all 40 saved test tasks, with 3 deterministic episodes per task.",
        "- Switching is a fixed 1000-step nonstationary stream over the maximally separated saved task pair; direction alternates across episodes.",
        "- Physics termination resets simulator state but does not reset the mode clock or causal adaptation state. A normalized latent span below 0.5 invalidates the switching protocol.",
        "- `joint` selection maximizes the worse of OOD-mean and switching-stream ratios versus the declared `v82a_robust_mean` base. Returns from the two protocols are never directly subtracted from each other.",
    ]
    if missing:
        lines.extend(["", "### Missing", ""])
        lines.extend(f"- `{name}`" for name in missing)

    lines.extend([
        "", "## Per-run metrics", "",
        "| variant | env | family | ID mean | OOD mean | OOD worst | OOD p10 | switch stream | span | AUC | delay | latent corr | terminations |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(rows, key=lambda item: (item.variant, item.env)):
        lines.append(
            f"| {row.variant} | {row.env} | {row.family} | "
            f"{fmt(row.train_mean)} | {fmt(row.test_mean)} | "
            f"{fmt(row.test_worst)} | {fmt(row.test_p10)} | "
            f"{fmt(row.switching_mean)} | {fmt(row.switch_latent_span, 3)} | "
            f"{fmt(row.switch_auc, 3)} | "
            f"{fmt(row.detection_delay)} | {fmt(row.latent_correlation, 3)} | "
            f"{fmt(row.termination_count)} |"
        )

    lines.extend([
        "", "## Mechanism ladder", "",
        "| env | robust base | best oracle: OOD / switch gain | best learned: OOD / switch gain | learned AUC / delay |",
        "|---|---|---|---|---|",
    ])
    oracle_passes = 0
    learned_recovery_passes = 0
    learned_detector_passes = 0
    measurable_envs = 0
    for env in ENVS:
        env_rows = [row for row in rows if row.env == env]
        base = next((row for row in env_rows
                     if row.variant == "v82a_robust_mean"), None)
        if base is None:
            lines.append(f"| {env} | missing | n/a | n/a | n/a |")
            continue
        oracle = choose(env_rows, "oracle", base)
        learned = choose(env_rows, "learned", base)
        protocol_valid = False
        if oracle is not None:
            protocol_valid = (
                math.isfinite(base.switch_latent_span)
                and base.switch_latent_span >= 0.5
                and math.isfinite(oracle.switch_latent_span)
                and oracle.switch_latent_span >= 0.5
            )
            measurable_envs += int(protocol_valid)
            oracle_ood = gain(oracle.test_mean, base.test_mean)
            oracle_switch = gain(oracle.switching_mean, base.switching_mean)
            oracle_passes += int(
                protocol_valid
                and oracle_ood >= 10.0 and oracle_switch >= 10.0)
            oracle_text = (
                f"{oracle.variant}: {fmt(oracle_ood)}% / "
                f"{fmt(oracle_switch)}%")
        else:
            oracle_ood = oracle_switch = math.nan
            oracle_text = "n/a"
        if learned is not None:
            learned_ood = gain(learned.test_mean, base.test_mean)
            learned_switch = gain(
                learned.switching_mean, base.switching_mean)
            recoveries = []
            if math.isfinite(oracle_ood) and oracle_ood > 0:
                recoveries.append(learned_ood / oracle_ood)
            if math.isfinite(oracle_switch) and oracle_switch > 0:
                recoveries.append(learned_switch / oracle_switch)
            if recoveries:
                learned_recovery_passes += int(
                    protocol_valid and min(recoveries) >= 0.70)
            detector_ok = (
                learned.switch_auc >= 0.8
                and math.isfinite(learned.detection_delay)
                and learned.detection_delay < 50)
            learned_detector_passes += int(protocol_valid and detector_ok)
            learned_text = (
                f"{learned.variant}: {fmt(learned_ood)}% / "
                f"{fmt(learned_switch)}%")
            detector_text = (
                f"{fmt(learned.switch_auc, 3)} / "
                f"{fmt(learned.detection_delay)}")
        else:
            learned_text = detector_text = "n/a"
        lines.append(
            f"| {env} | {fmt(base.test_mean)} / "
            f"{fmt(base.switching_mean)} | {oracle_text} | "
            f"{learned_text} | {detector_text} |")

    lines.extend([
        "", "## Predeclared gates", "",
        f"- Oracle improves both OOD mean and switching stream by >=10%: **{oracle_passes}/{measurable_envs} measurable environments**; required 3/4.",
        f"- Learned policy recovers >=70% of each positive oracle gain: **{learned_recovery_passes}/{measurable_envs} measurable environments**.",
        f"- Selected learned detector has AUC >=0.8 and median delay <50 steps: **{learned_detector_passes}/{measurable_envs} measurable environments**.",
        "",
    ])
    if missing:
        lines.append(
            "**Verdict withheld:** corrected sweeps are incomplete; partial winners are diagnostic only.")
    elif oracle_passes < 3:
        lines.append(
            "**Oracle gate failed:** this gravity protocol does not provide broad adaptation headroom for the current policy family. Do not expand learned variants to five seeds.")
    else:
        lines.append(
            "**Oracle gate passed:** learned teacher-student variants may proceed, subject to recovery and detector gates.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path,
        help="Backward-compatible root used for both stationary and switching.")
    parser.add_argument(
        "--stationary-results", type=Path, default=SWEEP_ROOT)
    parser.add_argument(
        "--switching-results", type=Path, default=SWITCH_ROOT)
    parser.add_argument("--expected-iteration", type=int, default=599)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    args = parser.parse_args()
    if args.results is not None:
        args.stationary_results = args.results
        args.switching_results = args.results

    expected = eligible_runs(
        args.expected_iteration,
        args.stationary_results,
        args.switching_results,
    )
    rows = []
    for name in expected:
        row = load_run(
            args.stationary_results / name,
            args.switching_results / name,
        )
        if row is not None:
            rows.append(row)
    write_csv(args.csv, rows)
    report = build_report(rows, expected)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
