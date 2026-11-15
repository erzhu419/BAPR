#!/usr/bin/env python3
"""Validate one stochastic-headroom family where its checkpoints reside.

Packet-loss and burst-torque pairs were trained on different nodes.  This
entry point preserves the strict pair/group validators while allowing each
node to validate its own two pairs and 60 evaluation outputs.  Only the small
report and machine-readable summary need to be synchronized afterwards.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    analyze_bapr_v3_budget_matched_fork_audit as audit,
)
from jax_experiments.analysis import run_bapr_v3_stochastic_headroom as protocol


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = (
    ROOT / "jax_experiments"
    / "results_bapr_v3_stochastic_headroom_audit_v1"
)
EXPECTED_PAIR_COUNT = 2
EXPECTED_OUTPUT_COUNT = 60


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _expected_groups(
    results_root: Path,
    family: str,
    event_seeds: tuple[int, ...],
) -> set[Path]:
    return {
        audit.group_directory(
            results_root, family, f"{env}-v2", event_seed
        )
        for env in audit.ENVS
        for event_seed in event_seeds
    }


def load_complete_family(
    pair_root: Path,
    results_root: Path,
    family: str,
    event_seeds: tuple[int, ...],
) -> tuple[
    dict[tuple[str, str, str, int], audit.v1.OutputMetrics],
    dict[tuple[str, str], dict[str, Any]],
]:
    if family not in protocol.FAMILIES:
        raise ValueError(f"unsupported stochastic family: {family!r}")
    expected_groups = _expected_groups(results_root, family, event_seeds)
    family_root = results_root / family
    if not family_root.is_dir():
        raise ValueError(f"family results root does not exist: {family_root}")
    discovered = {
        path for path in family_root.rglob("event_seed_*") if path.is_dir()
    }
    missing = sorted(expected_groups - discovered)
    unexpected = sorted(discovered - expected_groups)
    if missing or unexpected:
        details = []
        if missing:
            details.append(
                f"missing {len(missing)} groups (first: {missing[0]})"
            )
        if unexpected:
            details.append(
                f"unexpected {len(unexpected)} groups (first: {unexpected[0]})"
            )
        raise ValueError("; ".join(details))

    # The shared validator uses this allow-list for pair identity checks.
    audit.FAMILIES = (family,)
    pair_manifests: dict[tuple[str, str], dict[str, Any]] = {}
    outputs: dict[tuple[str, str, str, int], audit.v1.OutputMetrics] = {}
    for env_short in audit.ENVS:
        env = f"{env_short}-v2"
        pair_dir = audit.pair_directory(
            pair_root, family, env, audit.EXPECTED_TRAINING_SEED
        )
        pair_manifests[(family, env_short)] = audit.validate_pair_provenance(
            pair_dir, family, env, audit.EXPECTED_TRAINING_SEED
        )
        for event_seed in event_seeds:
            group = audit.validate_group_bundle(
                audit.group_directory(
                    results_root, family, env, event_seed
                ),
                pair_dir,
                family,
                env,
                audit.EXPECTED_TRAINING_SEED,
                event_seed,
            )
            for source, metrics in group.items():
                outputs[(family, env_short, source, event_seed)] = metrics

    if (
        len(pair_manifests) != EXPECTED_PAIR_COUNT
        or len(outputs) != EXPECTED_OUTPUT_COUNT
    ):
        raise ValueError(
            "internal cardinality error: expected two pairs and 60 outputs, "
            f"found {len(pair_manifests)} and {len(outputs)}"
        )
    return outputs, pair_manifests


def _metric_values(
    outputs: dict[tuple[str, str, str, int], audit.v1.OutputMetrics],
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


def build_summary(
    outputs: dict[tuple[str, str, str, int], audit.v1.OutputMetrics],
    pair_manifests: dict[tuple[str, str], dict[str, Any]],
    family: str,
    event_seeds: tuple[int, ...],
) -> tuple[dict[str, Any], str]:
    report_lines = [
        f"# BAPR-v3 strict stochastic audit: {family}",
        "",
        "Validated **2/2** exact producer pairs and **60/60** evaluation "
        "outputs on the checkpoint-owning node.",
        "",
    ]
    environments: dict[str, Any] = {}
    pair_passes = []
    for env in audit.ENVS:
        pair_lines, pair_pass = audit.v1.render_pair(
            outputs, family, env, event_seeds
        )
        report_lines.extend(pair_lines)
        pair_passes.append(pair_pass)

        source_metrics = {}
        for source in audit.SOURCES:
            stationary = _metric_values(
                outputs,
                family,
                env,
                source.directory,
                event_seeds,
                "stationary",
            )
            switching = _metric_values(
                outputs,
                family,
                env,
                source.directory,
                event_seeds,
                "switching",
            )
            source_metrics[source.directory] = {
                "stationary": stationary,
                "stationary_mean": statistics.mean(stationary),
                "switching": switching,
                "switching_mean": statistics.mean(switching),
                "task_returns": {
                    str(seed): outputs[
                        (family, env, source.directory, seed)
                    ].task_returns
                    for seed in event_seeds
                },
            }
        environments[env] = {
            "gate_pass": pair_pass,
            "sources": source_metrics,
        }

    family_pass = all(pair_passes)
    report_lines.extend(
        [
            "## Family decision",
            "",
            f"- `{family}` passes Ant and HalfCheetah: "
            f"**{'PASS' if family_pass else 'FAIL'}**",
            "- Five event seeds are paired streams from training seed 0; "
            "they are not independent training seeds.",
            "",
        ]
    )
    summary = {
        "schema": "bapr.v3-stochastic-headroom-family-audit.v1",
        "family": family,
        "training_seed": audit.EXPECTED_TRAINING_SEED,
        "event_seeds": list(event_seeds),
        "validated_pair_count": len(pair_manifests),
        "validated_output_count": len(outputs),
        "checkpoint_next_iter": audit.EXPECTED_CHECKPOINT_NEXT_ITER,
        "checkpoint_total_steps": audit.EXPECTED_CHECKPOINT_TOTAL_STEPS,
        "environments": environments,
        "family_gate_pass": family_pass,
    }
    return summary, "\n".join(report_lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=protocol.FAMILIES)
    parser.add_argument("--pair-root", type=Path, default=protocol.SAVE_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--event-seed", action="append", type=int)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--report-out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    event_seeds = tuple(args.event_seed or audit.DEFAULT_EVENT_SEEDS)
    if event_seeds != audit.DEFAULT_EVENT_SEEDS:
        raise SystemExit(
            "the preregistered ordered event seeds are "
            f"{audit.DEFAULT_EVENT_SEEDS}"
        )
    try:
        outputs, pair_manifests = load_complete_family(
            args.pair_root.resolve(),
            args.results_root.resolve(),
            args.family,
            event_seeds,
        )
        summary, report = build_summary(
            outputs, pair_manifests, args.family, event_seeds
        )
    except ValueError as exc:
        raise SystemExit(f"AUDIT INCOMPLETE OR INVALID: {exc}") from exc

    if args.json_out:
        _atomic_write(
            args.json_out.resolve(),
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
        )
    if args.report_out:
        _atomic_write(args.report_out.resolve(), report)
    sys.stdout.write(report)


if __name__ == "__main__":
    main()
