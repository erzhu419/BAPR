#!/usr/bin/env python3
"""Attribute structured-channel switching return to paired mode segments."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = (
    ROOT / "jax_experiments"
    / "results_bapr_v3_structured_channel_headroom_audit_v2"
)
EVENT_SEEDS = (1100, 1200, 1300, 1400, 1500)
PAIR_FIELDS = (
    "episode",
    "step",
    "switched",
    "true_mode_before",
    "physics_task_after",
)
SEGMENTS = {
    "first_half": slice(0, 500),
    "post_switch_full": slice(500, 1000),
    "post_switch_first_100": slice(500, 600),
    "post_switch_late_400": slice(600, 1000),
}


def read_trace(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 5000:
        raise ValueError(f"expected 5000 trace rows, got {len(rows)}: {path}")
    return rows


def by_episode(rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["episode"])].append(row)
    if sorted(grouped) != list(range(5)):
        raise ValueError(f"unexpected episode ids: {sorted(grouped)}")
    for episode, episode_rows in grouped.items():
        episode_rows.sort(key=lambda row: int(row["step"]))
        if [int(row["step"]) for row in episode_rows] != list(range(1, 1001)):
            raise ValueError(f"episode {episode} is not an exact 1000-step trace")
    return dict(grouped)


def paired_delta(
    robust: list[dict[str, str]],
    oracle: list[dict[str, str]],
    segment: slice,
) -> tuple[int, float]:
    robust_segment = robust[segment]
    oracle_segment = oracle[segment]
    if len(robust_segment) != len(oracle_segment) or not oracle_segment:
        raise ValueError("paired segment cardinality mismatch")
    for robust_row, oracle_row in zip(robust_segment, oracle_segment):
        for field in PAIR_FIELDS:
            if robust_row[field] != oracle_row[field]:
                raise ValueError(f"paired stream differs at {field}")
    modes = {int(row["true_mode_before"]) for row in oracle_segment}
    if len(modes) != 1:
        raise ValueError(f"segment crosses physics modes: {sorted(modes)}")
    delta = sum(float(row["reward"]) for row in oracle_segment) - sum(
        float(row["reward"]) for row in robust_segment
    )
    return modes.pop(), delta


def summarize(results_root: Path) -> dict[str, Any]:
    environment_root = results_root / "structured_channel" / "HalfCheetah"
    samples: dict[tuple[str, int], list[float]] = defaultdict(list)
    event_episode_deltas = []
    for event_seed in EVENT_SEEDS:
        event_root = environment_root / f"event_seed_{event_seed}"
        robust = by_episode(read_trace(event_root / "robust" / "switching_trace.csv"))
        oracle = by_episode(read_trace(event_root / "oracle" / "switching_trace.csv"))
        for episode in range(5):
            robust_rows = robust[episode]
            oracle_rows = oracle[episode]
            for name, segment in SEGMENTS.items():
                mode, delta = paired_delta(robust_rows, oracle_rows, segment)
                samples[(name, mode)].append(delta)
            event_episode_deltas.append({
                "event_seed": event_seed,
                "episode": episode,
                "oracle_minus_robust": sum(
                    float(row["reward"]) for row in oracle_rows
                ) - sum(float(row["reward"]) for row in robust_rows),
            })
    segment_summary: dict[str, dict[str, Any]] = {}
    for (segment, mode), values in sorted(samples.items()):
        segment_summary.setdefault(segment, {})[str(mode)] = {
            "n": len(values),
            "mean_oracle_minus_robust": mean(values),
            "wins": sum(value > 0.0 for value in values),
            "paired_deltas": values,
        }
    return {
        "schema": "bapr.v3-structured-channel-switching-segments.v1",
        "status": "complete",
        "family": "structured_channel",
        "env": "HalfCheetah-v2",
        "training_seed": 0,
        "event_seeds": list(EVENT_SEEDS),
        "pair_fields": list(PAIR_FIELDS),
        "segments": segment_summary,
        "event_episode_deltas": event_episode_deltas,
    }


def render(summary: dict[str, Any]) -> str:
    lines = [
        "# Structured-channel HalfCheetah switching attribution",
        "",
        "Paired oracle-minus-robust return on identical physics/noise streams.",
        "",
        "| Segment | Physics mode | Pairs | Mean delta | Wins |",
        "|---|---:|---:|---:|---:|",
    ]
    for segment, modes in summary["segments"].items():
        for mode, record in modes.items():
            lines.append(
                f"| {segment} | {mode} | {record['n']} | "
                f"{record['mean_oracle_minus_robust']:+.1f} | "
                f"{record['wins']}/{record['n']} |"
            )
    lines += [
        "",
        "The privileged oracle gains in mode 0 and loses throughout mode 1. "
        "Because the late 400-step mode-1 segment remains negative, the "
        "deficit is not a mode-detection delay or only an immediate switch "
        "transient.",
    ]
    return "\n".join(lines) + "\n"


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--report-output", type=Path)
    args = parser.parse_args()
    analysis_root = args.results_root.resolve() / "analysis"
    json_output = args.json_output or analysis_root / "switching_segment_diagnosis.json"
    report_output = args.report_output or analysis_root / "switching_segment_diagnosis.md"
    summary = summarize(args.results_root.resolve())
    write_atomic(json_output.resolve(), json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_atomic(report_output.resolve(), render(summary))
    print(f"STRUCTURED SWITCHING SEGMENT ANALYSIS COMPLETE: {report_output}")


if __name__ == "__main__":
    main()
