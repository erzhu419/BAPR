"""Analyze the BAPR-v2 v83 oracle policy-capacity ladder."""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict
from pathlib import Path

from jax_experiments.analysis.bapr_v2_phase1 import (
    ENVS,
    ROOT,
    RunSummary,
    fmt,
    percent_gain,
    summarize_run,
)


VARIANTS = [
    "v83a_oracle_scaled_r100",
    "v83b_oracle_direct_b0",
    "v83c_oracle_direct_b1",
    "v83d_oracle_expert5_b0",
    "v83e_oracle_residual_r200_b0",
]
DEFAULT_RESULTS = (
    ROOT / "jax_experiments" / "results_bapr_v2_oracle_capacity"
)
DEFAULT_PHASE1 = ROOT / "jax_experiments" / "results_bapr_v2_phase1"
DEFAULT_REPORT = ROOT / "reports" / "bapr_v2_oracle_capacity_2026-07-10.md"
DEFAULT_CSV = ROOT / "reports" / "bapr_v2_oracle_capacity_2026-07-10.csv"


def complete_row(rows: list[RunSummary], variant: str,
                 env: str) -> RunSummary | None:
    return next(
        (row for row in rows
         if row.variant == variant and row.env == env and row.complete),
        None,
    )


def best_row(rows: list[RunSummary], env: str) -> RunSummary | None:
    candidates = [
        row for row in rows
        if row.env == env and row.complete and math.isfinite(row.adapt_score)
    ]
    return max(candidates, key=lambda row: row.adapt_score, default=None)


def write_csv(path: Path, rows: list[RunSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [asdict(row) for row in rows]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def report_text(
    rows: list[RunSummary], robust_rows: list[RunSummary],
    old_oracle_rows: list[RunSummary], expected_iteration: int, tail: int,
) -> str:
    complete = sum(row.complete for row in rows)
    lines = [
        "# BAPR-v2 v83 Oracle Capacity",
        "",
        "## Audit",
        "",
        f"- Complete v83 runs: **{complete}/{len(rows)}**; completion requires "
        f"`iteration >= {expected_iteration}`.",
        f"- Metrics are means over the last **{tail}** eval checkpoints.",
        "- Adaptation score is `min(stationary OOD, switching online)`.",
        "- Training-loop stationary eval is a first-task proxy for continuous "
        "protocols. The corrected 40-task final sweep controls the final verdict.",
        "- `v83a` isolates the corrected pow1p5 latent normalization; `v83b/c` "
        "test full conditioned actors; `v83d` tests oracle-routed experts; "
        "`v83e` tests whether the bounded residual radius was the bottleneck.",
        "",
    ]
    incomplete = [row for row in rows if not row.complete]
    if incomplete:
        lines.extend(["### Incomplete", ""])
        lines.extend(
            f"- `{row.variant}/{row.env}`: iteration {row.last_iteration}"
            for row in incomplete
        )
        lines.append("")

    lines.extend([
        "## Capacity runs",
        "",
        "| variant | env | iter | ID | OOD | switching | adapt | action effect | latent sensitivity |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        suffix = "" if row.complete else " (incomplete)"
        lines.append(
            f"| {row.variant}{suffix} | {row.env} | {row.last_iteration} | "
            f"{fmt(row.stationary_id)} | {fmt(row.stationary_ood)} | "
            f"{fmt(row.switching)} | {fmt(row.adapt_score)} | "
            f"{fmt(row.policy_context_effect, 3)} | "
            f"{fmt(row.policy_latent_sensitivity, 3)} |"
        )

    lines.extend([
        "",
        "## Environment decisions",
        "",
        "| env | v82 robust | old oracle r1 | scaled r1 | best v83 (gain vs robust) | sensitivity |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    gains = []
    scale_gains = []
    for env in ENVS:
        robust = complete_row(robust_rows, "v82a_robust_mean", env)
        old = complete_row(old_oracle_rows, "v82d_oracle_r100", env)
        scaled = complete_row(rows, "v83a_oracle_scaled_r100", env)
        best = best_row(rows, env)
        gain = (
            percent_gain(best.adapt_score, robust.adapt_score)
            if best and robust else math.nan
        )
        scale_gain = (
            percent_gain(scaled.adapt_score, old.adapt_score)
            if scaled and old else math.nan
        )
        gains.append(gain)
        scale_gains.append(scale_gain)
        lines.append(
            f"| {env} | {fmt(robust.adapt_score) if robust else 'n/a'} | "
            f"{fmt(old.adapt_score) if old else 'n/a'} | "
            f"{fmt(scaled.adapt_score) if scaled else 'n/a'} | "
            f"{best.variant + ' ' + fmt(best.adapt_score) if best else 'n/a'} "
            f"({fmt(gain)}%) | "
            f"{fmt(best.policy_latent_sensitivity, 3) if best else 'n/a'} |"
        )

    headroom_passes = sum(
        math.isfinite(gain) and gain >= 10.0 for gain in gains)
    sensitive = [
        best_row(rows, env).policy_latent_sensitivity
        for env in ENVS if best_row(rows, env) is not None
    ]
    lines.extend([
        "",
        "## Gate",
        "",
        f"- Best oracle capacity beats v82 robust by >=10%: "
        f"**{headroom_passes}/4 environments** (required: at least 3/4).",
        "- Best-policy latent sensitivity >0.01: "
        f"**{sum(math.isfinite(x) and x > 0.01 for x in sensitive)}/"
        f"{len(sensitive)} measurable environments**.",
        "- Latent normalization effect (v83a versus old r1 oracle): "
        + ", ".join(
            f"{env} {fmt(gain)}%" for env, gain in zip(ENVS, scale_gains)
        ) + ".",
        "",
    ])
    if complete != len(rows):
        lines.append(
            "**Verdict withheld:** the capacity ladder is incomplete."
        )
    elif headroom_passes >= 3:
        lines.append(
            "**Provisional verdict:** the online proxy shows oracle adaptation "
            "headroom. Confirm it with the corrected full-task sweep before "
            "returning to learned latent and fallback-gate work."
        )
    else:
        lines.append(
            "**Provisional verdict:** even the stronger oracle policy fails the "
            "online-proxy headroom gate. Confirm with the corrected full-task "
            "sweep before changing the task construction or paper direction."
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--phase1", type=Path, default=DEFAULT_PHASE1)
    parser.add_argument("--expected-iteration", type=int, default=599)
    parser.add_argument("--tail", type=int, default=5)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.tail <= 0:
        raise SystemExit("--tail must be positive")

    rows = [
        summarize_run(
            args.results, variant, env, args.expected_iteration, args.tail)
        for variant in VARIANTS for env in ENVS
    ]
    robust_rows = [
        summarize_run(
            args.phase1, "v82a_robust_mean", env,
            args.expected_iteration, args.tail)
        for env in ENVS
    ]
    old_oracle_rows = [
        summarize_run(
            args.phase1, "v82d_oracle_r100", env,
            args.expected_iteration, args.tail)
        for env in ENVS
    ]
    text = report_text(
        rows, robust_rows, old_oracle_rows,
        args.expected_iteration, args.tail)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(text)
    write_csv(args.csv, rows)
    print(text)
    incomplete = [row for row in rows if not row.complete]
    if incomplete and not args.allow_incomplete:
        raise SystemExit(f"{len(incomplete)} v83 runs are incomplete")


if __name__ == "__main__":
    main()
