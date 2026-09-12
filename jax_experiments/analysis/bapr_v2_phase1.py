"""Audit and compare the BAPR-v2 phase-1 mechanism screen.

The scheduler terminal state is intentionally not used here. A run is complete
only when its local ``iteration.npy`` reaches the expected final iteration.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "jax_experiments" / "results_bapr_v2_phase1"
DEFAULT_REPORT = ROOT / "reports" / "bapr_v2_phase1_results_2026-07-10.md"
DEFAULT_CSV = ROOT / "reports" / "bapr_v2_phase1_results_2026-07-10.csv"

ENVS = ["Ant", "HalfCheetah", "Hopper", "Walker2d"]
VARIANTS = [
    "v82a_robust_mean",
    "v82b_robust_lcb",
    "v82c_oracle_r025",
    "v82d_oracle_r100",
    "v82e_supervised_fb",
    "v82f_supervised_nofb",
    "v82g_hybrid_fb",
    "v82h_hybrid_nofb",
    "v82i_supervised_h128",
]
ORACLE_VARIANTS = {"v82c_oracle_r025", "v82d_oracle_r100"}
LEARNED_VARIANTS = {
    "v82e_supervised_fb",
    "v82f_supervised_nofb",
    "v82g_hybrid_fb",
    "v82h_hybrid_nofb",
    "v82i_supervised_h128",
}

METRICS = {
    "stationary_id": "eval_stationary_id.npy",
    "stationary_ood": "eval_stationary_ood.npy",
    "switching": "eval_switching_online.npy",
    "context_gate": "v2_eval_context_gate.npy",
    "context_error": "v2_eval_context_error.npy",
    "live_gate": "v2_live_gate.npy",
    "latent_std": "v2_latent_std.npy",
    "policy_context_effect": "v2_policy_context_effect.npy",
    "policy_latent_sensitivity": "v2_policy_latent_sensitivity.npy",
}


@dataclass
class RunSummary:
    variant: str
    env: str
    complete: bool
    last_iteration: int
    eval_points: int
    stationary_id: float = math.nan
    stationary_id_std: float = math.nan
    stationary_ood: float = math.nan
    stationary_ood_std: float = math.nan
    switching: float = math.nan
    switching_std: float = math.nan
    adapt_score: float = math.nan
    worst_protocol: float = math.nan
    context_gate: float = math.nan
    context_error: float = math.nan
    live_gate: float = math.nan
    latent_std: float = math.nan
    policy_context_effect: float = math.nan
    policy_latent_sensitivity: float = math.nan


def finite_values(path: Path) -> list[float]:
    if not path.exists():
        return []
    try:
        values = np.asarray(np.load(path), dtype=float).reshape(-1)
    except Exception:
        return []
    return [float(value) for value in values if np.isfinite(value)]


def tail_stats(path: Path, tail: int) -> tuple[float, float, int]:
    values = finite_values(path)
    if not values:
        return math.nan, math.nan, 0
    selected = values[-tail:]
    return mean(selected), pstdev(selected), len(values)


def summarize_run(
    results: Path,
    variant: str,
    env: str,
    expected_iteration: int,
    tail: int,
) -> RunSummary:
    logs = results / f"{variant}_{env}_s0" / "logs"
    iterations = finite_values(logs / "iteration.npy")
    last_iteration = int(iterations[-1]) if iterations else -1
    summary = RunSummary(
        variant=variant,
        env=env,
        complete=last_iteration >= expected_iteration,
        last_iteration=last_iteration,
        eval_points=0,
    )
    for name, filename in METRICS.items():
        value, std, count = tail_stats(logs / filename, tail)
        setattr(summary, name, value)
        if name in {"stationary_id", "stationary_ood", "switching"}:
            setattr(summary, f"{name}_std", std)
            summary.eval_points = max(summary.eval_points, count)
    protocols = [
        summary.stationary_id,
        summary.stationary_ood,
        summary.switching,
    ]
    if all(math.isfinite(value) for value in protocols):
        summary.adapt_score = min(summary.stationary_ood, summary.switching)
        summary.worst_protocol = min(protocols)
    return summary


def percent_gain(value: float, baseline: float) -> float:
    if not (math.isfinite(value) and math.isfinite(baseline)):
        return math.nan
    return 100.0 * (value - baseline) / max(abs(baseline), 1e-9)


def ratio(value: float, baseline: float) -> float:
    if not (math.isfinite(value) and math.isfinite(baseline)):
        return math.nan
    return value / max(abs(baseline), 1e-9)


def best_run(rows: list[RunSummary], variants: set[str]) -> RunSummary | None:
    candidates = [
        row for row in rows
        if row.complete and row.variant in variants and math.isfinite(row.adapt_score)
    ]
    return max(candidates, key=lambda row: row.adapt_score, default=None)


def comparison_rows(rows: list[RunSummary]) -> list[dict]:
    out = []
    for env in ENVS:
        env_rows = [row for row in rows if row.env == env]
        robust = next(
            (row for row in env_rows
             if row.variant == "v82a_robust_mean" and row.complete),
            None,
        )
        oracle = best_run(env_rows, ORACLE_VARIANTS)
        learned = best_run(env_rows, LEARNED_VARIANTS)
        oracle_gain = (
            percent_gain(oracle.adapt_score, robust.adapt_score)
            if oracle and robust else math.nan
        )
        learned_gain = (
            percent_gain(learned.adapt_score, robust.adapt_score)
            if learned and robust else math.nan
        )
        recovered = math.nan
        if oracle and learned and robust and oracle.adapt_score > robust.adapt_score:
            recovered = (
                (learned.adapt_score - robust.adapt_score)
                / (oracle.adapt_score - robust.adapt_score)
            )
        out.append({
            "env": env,
            "robust_variant": robust.variant if robust else "missing",
            "robust_adapt_score": robust.adapt_score if robust else math.nan,
            "oracle_variant": oracle.variant if oracle else "missing",
            "oracle_adapt_score": oracle.adapt_score if oracle else math.nan,
            "oracle_gain_pct": oracle_gain,
            "learned_variant": learned.variant if learned else "missing",
            "learned_adapt_score": learned.adapt_score if learned else math.nan,
            "learned_gain_pct": learned_gain,
            "learned_oracle_recovery": recovered,
            "stationary_retention": (
                ratio(learned.stationary_id, robust.stationary_id)
                if learned and robust else math.nan
            ),
            "learned_gate": learned.context_gate if learned else math.nan,
            "learned_live_gate": learned.live_gate if learned else math.nan,
        })
    return out


def fmt(value: float, digits: int = 1) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def markdown_report(
    rows: list[RunSummary], comparisons: list[dict], expected_iteration: int,
    tail: int,
) -> str:
    complete = sum(row.complete for row in rows)
    lines = [
        "# BAPR-v2 Phase-1 Results",
        "",
        "## Audit",
        "",
        f"- Complete runs: **{complete}/{len(rows)}**. Completion requires "
        f"`iteration >= {expected_iteration}`; scheduler terminal state is ignored.",
        f"- Scores are means over the last **{tail}** available eval checkpoints.",
        "- `adapt score = min(stationary OOD, switching online)`; the oracle and "
        "learned winners are selected independently per environment by this score.",
        "- Continuous training-loop stationary eval uses only the first task. "
        "These are provisional online proxies, not full-task or worst-task scores; "
        "the corrected final sweep is required for a final mechanism verdict.",
        "- Switch AUC and detection delay are not measurable from these aggregate "
        "arrays. Per-step switch labels and detector traces are required.",
        "",
    ]
    incomplete = [row for row in rows if not row.complete]
    if incomplete:
        lines.extend(["### Incomplete", ""])
        for row in incomplete:
            lines.append(
                f"- `{row.variant}/{row.env}`: iteration {row.last_iteration}"
            )
        lines.append("")

    lines.extend([
        "## Per-run tail metrics",
        "",
        "| variant | env | iter | ID | OOD | switching | adapt score | gate | live gate | action effect | latent sensitivity |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        suffix = "" if row.complete else " (incomplete)"
        lines.append(
            f"| {row.variant}{suffix} | {row.env} | {row.last_iteration} | "
            f"{fmt(row.stationary_id)} | {fmt(row.stationary_ood)} | "
            f"{fmt(row.switching)} | {fmt(row.adapt_score)} | "
            f"{fmt(row.context_gate, 3)} | {fmt(row.live_gate, 3)} | "
            f"{fmt(row.policy_context_effect, 3)} | "
            f"{fmt(row.policy_latent_sensitivity, 3)} |"
        )

    lines.extend([
        "",
        "## Mechanism ladder",
        "",
        "| env | robust score | best oracle (gain) | best learned (gain) | oracle recovery | ID retention | learned gates |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in comparisons:
        recovery = (
            f"{100.0 * row['learned_oracle_recovery']:.1f}%"
            if math.isfinite(row["learned_oracle_recovery"]) else "n/a"
        )
        retention = (
            f"{100.0 * row['stationary_retention']:.1f}%"
            if math.isfinite(row["stationary_retention"]) else "n/a"
        )
        lines.append(
            f"| {row['env']} | {fmt(row['robust_adapt_score'])} | "
            f"{row['oracle_variant']} {fmt(row['oracle_adapt_score'])} "
            f"({fmt(row['oracle_gain_pct'])}%) | "
            f"{row['learned_variant']} {fmt(row['learned_adapt_score'])} "
            f"({fmt(row['learned_gain_pct'])}%) | {recovery} | {retention} | "
            f"{fmt(row['learned_gate'], 3)} / {fmt(row['learned_live_gate'], 3)} |"
        )

    oracle_passes = sum(
        math.isfinite(row["oracle_gain_pct"]) and row["oracle_gain_pct"] >= 10.0
        for row in comparisons
    )
    recovery_values = [
        row["learned_oracle_recovery"] for row in comparisons
        if math.isfinite(row["learned_oracle_recovery"])
    ]
    retention_values = [
        row["stationary_retention"] for row in comparisons
        if math.isfinite(row["stationary_retention"])
    ]
    lines.extend([
        "",
        "## Predeclared gates",
        "",
        f"- Oracle adaptation gain >=10%: **{oracle_passes}/4 environments** "
        "(required: at least 3/4).",
        "- Learned recovery >=70% of positive oracle gain: "
        f"**{sum(value >= 0.70 for value in recovery_values)}/{len(recovery_values)} "
        "measurable environments**.",
        "- Selected learned stationary retention >=95%: "
        f"**{sum(value >= 0.95 for value in retention_values)}/{len(retention_values)} "
        "measurable environments**.",
        "- Switch AUC >=0.8 and median delay <50 steps: **not measurable** from "
        "the current aggregate logging.",
        "",
    ])
    if complete != len(rows):
        lines.append(
            "**Verdict withheld:** the matrix is incomplete; partial values must not "
            "be used as the phase-1 decision."
        )
    elif oracle_passes < 3:
        lines.append(
            "**Provisional verdict:** the online-proxy oracle gate fails. Do not "
            "expand learned latent variants to multiple seeds; first validate the "
            "stronger oracle-conditioned policy and corrected full-task sweep."
        )
    else:
        lines.append(
            "**Provisional verdict:** the online proxy suggests oracle adaptation "
            "headroom. The corrected full-task sweep must confirm it before the "
            "learned estimator is expanded."
        )
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[RunSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [asdict(row) for row in rows]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--expected-iteration", type=int, default=599)
    parser.add_argument("--tail", type=int, default=5)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.tail <= 0:
        raise SystemExit("--tail must be positive")

    rows = [
        summarize_run(
            args.results, variant, env, args.expected_iteration, args.tail)
        for variant in VARIANTS
        for env in ENVS
    ]
    comparisons = comparison_rows(rows)
    report = markdown_report(
        rows, comparisons, args.expected_iteration, args.tail)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report)
    write_csv(args.csv, rows)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "runs": [asdict(row) for row in rows],
            "comparisons": comparisons,
        }, indent=2, allow_nan=True))
    print(report)

    incomplete = [row for row in rows if not row.complete]
    if incomplete and not args.allow_incomplete:
        raise SystemExit(f"{len(incomplete)} phase-1 runs are incomplete")


if __name__ == "__main__":
    main()
