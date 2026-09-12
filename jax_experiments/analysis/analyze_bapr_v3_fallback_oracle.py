"""Aggregate the privileged fallback-oracle capacity audit."""
from __future__ import annotations

import json
import statistics
from typing import Any

import numpy as np

from jax_experiments.analysis import bapr_v3_fallback_oracle as protocol
from jax_experiments.analysis import (
    run_bapr_v3_fallback_oracle as runner,
)
from jax_experiments.analysis import (
    run_bapr_v3_utility_aware_router_audit as base_audit,
)


def _termination_rate(record) -> float:
    return float(np.mean([
        int(row["termination_count"]) > 0 for row in record["episodes"]
    ]))


def summarize() -> dict[str, Any]:
    protocol.configure()
    candidate = {}
    hard = {}
    robust = {}
    dynamic_oracle = {}
    candidate_termination = {}
    hard_termination = {}
    routing_rows = []
    for seed in protocol.DEVELOPMENT_EVENT_SEEDS:
        group = json.loads(protocol.group_path(seed).read_text(encoding="utf-8"))
        runner.validate_group(group)
        reference_path = protocol.discrete.prior.utility.audit_group_path(
            "validation", seed, protocol.DECISION_VARIANT)
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        base_audit.validate_group(reference)
        candidate[seed] = float(group["switching_full_cycle"]["mean"])
        candidate_termination[seed] = _termination_rate(
            group["switching_full_cycle"])
        for destination, controller in (
                (hard, "learned_utility_router"),
                (robust, "robust"),
                (dynamic_oracle, "dynamic_utility_oracle")):
            destination[seed] = float(
                reference["switching"]["full_cycle"][controller]["mean"])
        hard_termination[seed] = _termination_rate(
            reference["switching"]["full_cycle"][
                "learned_utility_router"])
        routing_rows.extend(group["switching_full_cycle"]["routing"])

    delta_hard = {
        seed: candidate[seed] - hard[seed]
        for seed in protocol.DEVELOPMENT_EVENT_SEEDS
    }
    termination_delta = statistics.mean(
        candidate_termination[seed] - hard_termination[seed]
        for seed in protocol.DEVELOPMENT_EVENT_SEEDS)
    gate = {
        "beats_hard_each_seed": all(value > 0.0 for value in delta_hard.values()),
        "minimum_mean_gain": (
            statistics.mean(delta_hard.values())
            >= protocol.MIN_FULL_CYCLE_MEAN_GAIN),
        "termination_noninferior": (
            termination_delta <= protocol.TERMINATION_RATE_MARGIN),
    }
    gate["passed"] = all(gate.values())
    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "family": protocol.FAMILY,
        "env": protocol.ENV,
        "decision_variant": protocol.DECISION_VARIANT,
        "development_event_seeds": list(protocol.DEVELOPMENT_EVENT_SEEDS),
        "sealed_confirmation_event_seeds": list(
            protocol.SEALED_CONFIRMATION_EVENT_SEEDS),
        "full_cycle": {
            "robust": {"per_event_seed": robust,
                       "mean": statistics.mean(robust.values())},
            "hard_cusum": {"per_event_seed": hard,
                           "mean": statistics.mean(hard.values())},
            "dynamic_utility_oracle": {
                "per_event_seed": dynamic_oracle,
                "mean": statistics.mean(dynamic_oracle.values()),
            },
            "fallback_oracle": {
                "per_event_seed": candidate,
                "mean": statistics.mean(candidate.values()),
            },
            "fallback_oracle_minus_hard": {
                "per_event_seed": delta_hard,
                "mean": statistics.mean(delta_hard.values()),
            },
        },
        "routing": {
            "hard_fallback_rate": float(np.mean([
                row["hard_fallback_rate"] for row in routing_rows])),
            "fallback_specialist_replacement_rate": float(np.mean([
                row["fallback_specialist_replacement_rate"]
                for row in routing_rows])),
            "wrong_route_rate": float(np.mean([
                row["wrong_route_rate"] for row in routing_rows])),
        },
        "termination_rate_delta_vs_hard": termination_delta,
        "gate": gate,
    }


def render(summary: dict[str, Any]) -> str:
    full = summary["full_cycle"]
    lines = [
        "# Privileged fallback-oracle capacity audit",
        "",
        "| Metric | Robust | Hard CUSUM | Dynamic oracle | Fallback oracle | "
        "Delta hard |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Full cycle | {full['robust']['mean']:.1f} | "
        f"{full['hard_cusum']['mean']:.1f} | "
        f"{full['dynamic_utility_oracle']['mean']:.1f} | "
        f"{full['fallback_oracle']['mean']:.1f} | "
        f"{full['fallback_oracle_minus_hard']['mean']:+.1f} |",
        "",
        "Per-seed fallback-oracle minus hard CUSUM: "
        + ", ".join(
            f"`{seed}: {value:+.1f}`"
            for seed, value in full[
                "fallback_oracle_minus_hard"]["per_event_seed"].items()),
        "",
        f"- Hard fallback rate: "
        f"`{summary['routing']['hard_fallback_rate']:.1%}`",
        f"- Privileged specialist replacement rate: "
        f"`{summary['routing']['fallback_specialist_replacement_rate']:.1%}`",
        f"- Capacity gate: "
        f"**{'PASS' if summary['gate']['passed'] else 'FAIL'}**",
        "",
        "This is a privileged capacity diagnostic, not a deployable result. "
        "The sealed confirmation streams remain unopened.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    summary = summarize()
    protocol.write_json_atomic(protocol.ANALYSIS_JSON, summary)
    protocol.ANALYSIS_REPORT.parent.mkdir(parents=True, exist_ok=True)
    protocol.ANALYSIS_REPORT.write_text(render(summary), encoding="utf-8")
    print(
        "FALLBACK ORACLE ANALYSIS COMPLETE: "
        f"gate={'PASS' if summary['gate']['passed'] else 'FAIL'}",
        flush=True,
    )


if __name__ == "__main__":
    main()

