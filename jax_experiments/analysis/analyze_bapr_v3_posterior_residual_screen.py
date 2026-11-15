"""Aggregate the posterior-conditioned residual development screen."""
from __future__ import annotations

import json
import statistics
from typing import Any

import numpy as np

from jax_experiments.analysis import bapr_v3_posterior_residual as protocol
from jax_experiments.analysis import (
    run_bapr_v3_posterior_residual_screen as runner,
)
from jax_experiments.analysis import (
    run_bapr_v3_utility_aware_router_audit as base_audit,
)


METRICS = ("stationary", "switching_slow", "switching_full_cycle")


def event_return(group: dict[str, Any], controller: str, metric: str) -> float:
    if metric == "stationary":
        return statistics.mean(
            float(group["stationary"][controller][str(mode)]["mean"])
            for mode in range(4))
    kind = "slow_pair" if metric == "switching_slow" else "full_cycle"
    return float(group["switching"][kind][controller]["mean"])


def event_termination(
    group: dict[str, Any], controller: str, metric: str,
) -> float:
    if metric == "stationary":
        episodes = [
            episode
            for mode in range(4)
            for episode in group["stationary"][controller][str(mode)][
                "episodes"]
        ]
        return float(np.mean([bool(row["terminated"]) for row in episodes]))
    kind = "slow_pair" if metric == "switching_slow" else "full_cycle"
    episodes = group["switching"][kind][controller]["episodes"]
    return float(np.mean([
        int(row["termination_count"]) > 0 for row in episodes]))


def _return_vector(group, controller: str, metric: str) -> np.ndarray:
    if metric == "stationary":
        return np.asarray([
            float(row["return"])
            for mode in range(4)
            for row in group["stationary"][controller][str(mode)]["episodes"]
        ], dtype=np.float64)
    kind = "slow_pair" if metric == "switching_slow" else "full_cycle"
    return np.asarray([
        float(row["return"])
        for row in group["switching"][kind][controller]["episodes"]
    ], dtype=np.float64)


def baseline_replay(group: dict[str, Any]) -> dict[str, Any]:
    seed = int(group["event_seed"])
    reference_path = protocol.utility.audit_group_path(
        "validation", seed, protocol.DECISION_VARIANT)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    base_audit.validate_group(reference)
    maximum = 0.0
    matched = True
    for controller in (
            "robust", "dynamic_utility_oracle", "learned_utility_router"):
        for metric in METRICS:
            left = _return_vector(group, controller, metric)
            right = _return_vector(reference, controller, metric)
            if left.shape != right.shape:
                matched = False
                continue
            maximum = max(maximum, float(np.max(np.abs(left - right))))
            matched = matched and bool(np.allclose(
                left, right, rtol=1e-7, atol=1e-5))
    return {
        "reference_file": protocol.file_record(reference_path),
        "matched": matched,
        "maximum_absolute_return_difference": maximum,
    }


def _routing_means(groups: dict[int, dict[str, Any]], metric: str):
    rows = []
    for group in groups.values():
        if metric == "stationary":
            for mode in range(4):
                rows.extend(group["stationary"]["posterior_residual"][
                    str(mode)]["routing"])
        else:
            kind = "slow_pair" if metric == "switching_slow" else "full_cycle"
            rows.extend(group["switching"][kind][
                "posterior_residual"]["routing"])
    return {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in (
            "adaptation_strength_mean",
            "adaptation_positive_rate",
            "action_delta_l2_mean",
            "wrong_route_rate",
        )
    }


def variant_gate(
    full_delta_hard: list[float],
    full_delta_robust: list[float],
    stationary_delta_hard: list[float],
    stationary_termination_delta: float,
    full_termination_delta: float,
    adaptation_rate: float,
    replay_matches: bool,
) -> dict[str, bool]:
    gate = {
        "baseline_replay_matches": bool(replay_matches),
        "full_cycle_improves_each_seed": min(full_delta_hard) > 0.0,
        "full_cycle_minimum_mean_gain": (
            statistics.mean(full_delta_hard)
            >= protocol.MIN_FULL_CYCLE_MEAN_GAIN),
        "full_cycle_beats_robust_each_seed": min(full_delta_robust) > 0.0,
        "stationary_mean_noninferior": (
            statistics.mean(stationary_delta_hard)
            >= -protocol.STATIONARY_MEAN_MARGIN),
        "stationary_each_seed_noninferior": (
            min(stationary_delta_hard)
            >= -protocol.STATIONARY_PER_SEED_MARGIN),
        "stationary_termination_noninferior": (
            stationary_termination_delta
            <= protocol.TERMINATION_RATE_MARGIN),
        "full_cycle_termination_noninferior": (
            full_termination_delta <= protocol.TERMINATION_RATE_MARGIN),
        "residual_is_used": adaptation_rate >= protocol.MIN_ADAPTATION_RATE,
    }
    gate["passed"] = all(gate.values())
    return gate


def summarize() -> dict[str, Any]:
    protocol.configure()
    variants = {}
    for variant, cap in protocol.RESIDUAL_VARIANTS.items():
        groups = {}
        replay = {}
        for seed in protocol.DEVELOPMENT_EVENT_SEEDS:
            payload = json.loads(protocol.group_path(
                variant, seed).read_text(encoding="utf-8"))
            runner.validate_group(payload)
            groups[seed] = payload
            replay[seed] = baseline_replay(payload)
        series = {
            controller: {
                metric: [event_return(groups[seed], controller, metric)
                         for seed in protocol.DEVELOPMENT_EVENT_SEEDS]
                for metric in METRICS
            }
            for controller in runner.CONTROLLERS
        }
        controllers = {
            controller: {
                metric: {
                    "per_event_seed": dict(zip(
                        protocol.DEVELOPMENT_EVENT_SEEDS,
                        series[controller][metric])),
                    "mean": statistics.mean(series[controller][metric]),
                }
                for metric in METRICS
            }
            for controller in runner.CONTROLLERS
        }
        comparisons = {}
        for baseline in (
                "learned_utility_router", "robust", "dynamic_utility_oracle"):
            comparisons[baseline] = {}
            for metric in METRICS:
                differences = [
                    residual - base
                    for residual, base in zip(
                        series["posterior_residual"][metric],
                        series[baseline][metric])
                ]
                comparisons[baseline][metric] = {
                    "per_event_seed": dict(zip(
                        protocol.DEVELOPMENT_EVENT_SEEDS, differences)),
                    "mean": statistics.mean(differences),
                    "wins": sum(value > 0.0 for value in differences),
                }
        termination_deltas = {}
        for metric in ("stationary", "switching_full_cycle"):
            termination_deltas[metric] = statistics.mean(
                event_termination(groups[seed], "posterior_residual", metric)
                - event_termination(
                    groups[seed], "learned_utility_router", metric)
                for seed in protocol.DEVELOPMENT_EVENT_SEEDS)
        routing = {
            metric: _routing_means(groups, metric) for metric in METRICS
        }
        gate = variant_gate(
            list(comparisons["learned_utility_router"][
                "switching_full_cycle"]["per_event_seed"].values()),
            list(comparisons["robust"][
                "switching_full_cycle"]["per_event_seed"].values()),
            list(comparisons["learned_utility_router"][
                "stationary"]["per_event_seed"].values()),
            termination_deltas["stationary"],
            termination_deltas["switching_full_cycle"],
            routing["switching_full_cycle"]["adaptation_positive_rate"],
            all(row["matched"] for row in replay.values()),
        )
        variants[variant] = {
            "cap": cap,
            "advantage_scale": float(next(iter(groups.values()))[
                "advantage_scale"]),
            "baseline_replay": replay,
            "controllers": controllers,
            "residual_comparisons": comparisons,
            "termination_rate_deltas_vs_hard_router": termination_deltas,
            "routing": routing,
            "gate": gate,
        }

    passing = [
        name for name, row in variants.items() if row["gate"]["passed"]
    ]
    selected = (
        max(passing, key=lambda name: (
            variants[name]["controllers"]["posterior_residual"][
                "switching_full_cycle"]["mean"],
            -variants[name]["cap"],
        )) if passing else None
    )
    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "family": protocol.FAMILY,
        "env": protocol.ENV,
        "decision_variant": protocol.DECISION_VARIANT,
        "development_event_seeds": list(protocol.DEVELOPMENT_EVENT_SEEDS),
        "sealed_confirmation_event_seeds": list(
            protocol.SEALED_CONFIRMATION_EVENT_SEEDS),
        "primary_endpoints": runner.primary_endpoints(),
        "variants": variants,
        "passing_variants": passing,
        "selected_variant": selected,
        "screen_passed": selected is not None,
    }


def render(summary: dict[str, Any]) -> str:
    lines = [
        "# Posterior-conditioned residual capacity screen",
        "",
        "Development streams only; the five confirmation streams remain ",
        "sealed. The CUSUM posterior and every controller are frozen.",
        "",
        "| Variant | Cap | Stationary residual | Delta hard | Full residual | "
        "Delta hard | Delta robust | Adapt rate | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, row in summary["variants"].items():
        controllers = row["controllers"]
        comparisons = row["residual_comparisons"]
        lines.append(
            f"| {variant} | {row['cap']:.2f} | "
            f"{controllers['posterior_residual']['stationary']['mean']:.1f} | "
            f"{comparisons['learned_utility_router']['stationary']['mean']:+.1f} | "
            f"{controllers['posterior_residual']['switching_full_cycle']['mean']:.1f} | "
            f"{comparisons['learned_utility_router']['switching_full_cycle']['mean']:+.1f} | "
            f"{comparisons['robust']['switching_full_cycle']['mean']:+.1f} | "
            f"{row['routing']['switching_full_cycle']['adaptation_positive_rate']:.1%} | "
            f"{'PASS' if row['gate']['passed'] else 'FAIL'} |")
    lines += [
        "",
        f"Screen: **{'PASS' if summary['screen_passed'] else 'FAIL'}**",
        f"Selected variant: `{summary['selected_variant']}`",
        "",
        "Passing this screen permits a new confirmation or residual-policy "
        "distillation. Failure closes action interpolation; it does not permit "
        "another CUSUM threshold sweep.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    summary = summarize()
    protocol.write_json_atomic(protocol.ANALYSIS_JSON, summary)
    protocol.ANALYSIS_REPORT.parent.mkdir(parents=True, exist_ok=True)
    protocol.ANALYSIS_REPORT.write_text(render(summary), encoding="utf-8")
    print(
        "POSTERIOR RESIDUAL ANALYSIS COMPLETE: "
        f"gate={'PASS' if summary['screen_passed'] else 'FAIL'}, "
        f"selected={summary['selected_variant']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
