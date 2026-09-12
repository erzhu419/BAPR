"""Aggregate the preregistered v19 specialist stability screen."""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_specialist_stability_v19 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_stability_audit_v19 as audit,
)


def _mean(values) -> float:
    return float(np.mean([float(value) for value in values]))


def _stationary_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    split = payload["stationary_holdout"]
    modes = {}
    for mode in protocol.MODES:
        robust_rows = [
            split[str(event_seed)]["robust_sac"][str(mode)]
            for event_seed in protocol.STATIONARY_HOLDOUT_EVENT_SEEDS
        ]
        specialist_rows = [
            split[str(event_seed)]["matching_specialist"][str(mode)]
            for event_seed in protocol.STATIONARY_HOLDOUT_EVENT_SEEDS
        ]
        robust_mean = _mean(row["return_mean"] for row in robust_rows)
        specialist_mean = _mean(
            row["return_mean"] for row in specialist_rows)
        relative_gain = (
            (specialist_mean - robust_mean) / abs(robust_mean)
            if robust_mean != 0.0 else float("-inf")
        )
        event_wins = sum(
            specialist["return_mean"] > robust["return_mean"]
            for robust, specialist in zip(robust_rows, specialist_rows)
        )
        terminated_rate = _mean(
            row["terminated_rate"] for row in specialist_rows)
        passed = bool(
            relative_gain >= protocol.MIN_HOLDOUT_MODE_GAIN
            and event_wins == len(protocol.STATIONARY_HOLDOUT_EVENT_SEEDS)
            and terminated_rate == 0.0
        )
        modes[str(mode)] = {
            "robust_mean": robust_mean,
            "specialist_mean": specialist_mean,
            "relative_gain": float(relative_gain),
            "event_wins": int(event_wins),
            "terminated_rate": terminated_rate,
            "pass": passed,
        }
    mode_wins = sum(int(row["pass"]) for row in modes.values())
    return {
        "modes": modes,
        "mode_wins": mode_wins,
        "pass": bool(mode_wins >= protocol.MIN_HOLDOUT_MODE_WINS),
    }


def _switching_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    split = payload["switching_holdout"]
    arms = {}
    for arm in (
        "robust_sac",
        "dynamic_specialist_oracle",
        "true_mode_safe_utility",
    ):
        rows = [
            split[str(event_seed)][arm]
            for event_seed in protocol.SWITCHING_EVENT_SEEDS
        ]
        arms[arm] = {
            "mean": _mean(row["return_mean"] for row in rows),
            "event_returns": {
                str(event_seed): float(row["return_mean"])
                for event_seed, row in zip(
                    protocol.SWITCHING_EVENT_SEEDS, rows)
            },
            "terminated_rate": _mean(
                row["terminated_rate"] for row in rows),
        }
    robust = arms["robust_sac"]
    safe = arms["true_mode_safe_utility"]
    relative_gain = (
        (safe["mean"] - robust["mean"]) / abs(robust["mean"])
        if robust["mean"] != 0.0 else float("-inf")
    )
    event_wins = sum(
        safe["event_returns"][str(event_seed)]
        > robust["event_returns"][str(event_seed)]
        for event_seed in protocol.SWITCHING_EVENT_SEEDS
    )
    passed = bool(
        relative_gain >= protocol.MIN_SWITCHING_GAIN
        and event_wins == len(protocol.SWITCHING_EVENT_SEEDS)
        and safe["terminated_rate"] == 0.0
    )
    return {
        "arms": arms,
        "safe_relative_gain": float(relative_gain),
        "safe_event_wins": int(event_wins),
        "pass": passed,
    }


def _cell_metrics(variant: str, seed: int) -> dict[str, Any]:
    audit.validate_audit(variant, seed)
    payload = protocol.read_json(protocol.audit_result(variant, seed))
    stationary = _stationary_metrics(payload)
    switching = _switching_metrics(payload)
    return {
        "utility_map": payload["utility_map"],
        "stationary": stationary,
        "switching": switching,
        "pass": bool(stationary["pass"] and switching["pass"]),
    }


def _comparison(
    candidate: str,
    cells: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    control_cells = cells[protocol.CONTROL_VARIANT]
    candidate_cells = cells[candidate]
    seed_rows = {}
    event_deltas = []
    stationary_retentions = []
    for seed in protocol.TRAINING_SEEDS:
        key = str(seed)
        control = control_cells[key]
        current = candidate_cells[key]
        control_safe = control["switching"]["arms"][
            "true_mode_safe_utility"]
        candidate_safe = current["switching"]["arms"][
            "true_mode_safe_utility"]
        per_event = {
            str(event_seed): (
                candidate_safe["event_returns"][str(event_seed)]
                - control_safe["event_returns"][str(event_seed)]
            )
            for event_seed in protocol.SWITCHING_EVENT_SEEDS
        }
        event_deltas.extend(per_event.values())
        mode_retentions = {}
        for mode in protocol.MODES:
            control_return = control["stationary"]["modes"][str(mode)][
                "specialist_mean"]
            candidate_return = current["stationary"]["modes"][str(mode)][
                "specialist_mean"]
            retention = (
                1.0 + (candidate_return - control_return) / abs(control_return)
                if control_return != 0.0 else float("-inf")
            )
            mode_retentions[str(mode)] = float(retention)
            stationary_retentions.append(retention)
        seed_rows[key] = {
            "safe_switching_delta": float(
                candidate_safe["mean"] - control_safe["mean"]),
            "event_deltas": {
                event: float(value) for event, value in per_event.items()
            },
            "stationary_mode_retentions": mode_retentions,
            "candidate_safe_gain": float(
                current["switching"]["safe_relative_gain"]),
            "control_safe_gain": float(
                control["switching"]["safe_relative_gain"]),
            "candidate_cell_pass": bool(current["pass"]),
        }
    seed_deltas = [
        row["safe_switching_delta"] for row in seed_rows.values()
    ]
    candidate_mode_passes = sum(
        cells[candidate][str(seed)]["stationary"]["mode_wins"]
        for seed in protocol.TRAINING_SEEDS
    )
    control_mode_passes = sum(
        control_cells[str(seed)]["stationary"]["mode_wins"]
        for seed in protocol.TRAINING_SEEDS
    )
    candidate_worst_gain = min(
        row["candidate_safe_gain"] for row in seed_rows.values())
    control_worst_gain = min(
        row["control_safe_gain"] for row in seed_rows.values())
    checks = {
        "all_cells_pass": all(
            row["candidate_cell_pass"] for row in seed_rows.values()),
        "positive_mean_switching_delta": _mean(seed_deltas) > 0.0,
        "paired_seed_wins": sum(value > 0.0 for value in seed_deltas)
        >= protocol.REQUIRED_PAIRED_SEED_WINS,
        "paired_event_wins": sum(value > 0.0 for value in event_deltas)
        >= protocol.REQUIRED_PAIRED_EVENT_WINS,
        "better_worst_seed_gain": candidate_worst_gain > control_worst_gain,
        "stationary_retention": min(stationary_retentions)
        >= protocol.MIN_STATIONARY_RETENTION,
        "mode_passes_not_reduced": (
            candidate_mode_passes >= control_mode_passes),
    }
    return {
        "candidate": candidate,
        "control": protocol.CONTROL_VARIANT,
        "seed_rows": seed_rows,
        "mean_safe_switching_delta": _mean(seed_deltas),
        "paired_seed_wins": int(sum(value > 0.0 for value in seed_deltas)),
        "paired_event_wins": int(sum(value > 0.0 for value in event_deltas)),
        "minimum_stationary_retention": float(min(stationary_retentions)),
        "candidate_worst_seed_safe_gain": float(candidate_worst_gain),
        "control_worst_seed_safe_gain": float(control_worst_gain),
        "candidate_mode_passes": int(candidate_mode_passes),
        "control_mode_passes": int(control_mode_passes),
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def analyze() -> dict[str, Any]:
    protocol.validate_registration()
    cells = {
        variant: {
            str(seed): _cell_metrics(variant, seed)
            for seed in protocol.TRAINING_SEEDS
        }
        for variant in protocol.VARIANTS
    }
    comparisons = {
        variant: _comparison(variant, cells)
        for variant in protocol.VARIANTS
        if variant != protocol.CONTROL_VARIANT
    }
    passing = [
        variant for variant, row in comparisons.items() if row["pass"]
    ]
    mean_safe_return = {
        variant: _mean(
            cells[variant][str(seed)]["switching"]["arms"]
            ["true_mode_safe_utility"]["mean"]
            for seed in protocol.TRAINING_SEEDS
        )
        for variant in protocol.VARIANTS
    }
    selected = None
    if passing:
        best = max(passing, key=lambda variant: mean_safe_return[variant])
        if (
            "full_state" in passing
            and mean_safe_return["full_state"]
            >= 0.98 * mean_safe_return[best]
        ):
            selected = "full_state"
        else:
            selected = best
    if selected is None:
        diagnosis = "initialization_does_not_resolve_policy_bank_variance"
        next_step = (
            "do not alter the frozen posterior or v18 holdout; inspect "
            "per-mode optimization traces and close this initialization family"
        )
    else:
        diagnosis = "specialist_stability_candidate_selected"
        next_step = (
            f"freeze {selected} and run a new five-seed equal-policy-budget "
            "confirmation against causal SAC5"
        )
    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "training_seeds": list(protocol.TRAINING_SEEDS),
        "variants": list(protocol.VARIANTS),
        "cells": cells,
        "comparisons": comparisons,
        "mean_safe_switching_return": mean_safe_return,
        "passing_candidates": passing,
        "selected_candidate": selected,
        "diagnosis": diagnosis,
        "next_step": next_step,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# V19 specialist training stability result",
        "",
        "This fresh-seed screen changes only specialist initialization. The "
        "environment, posterior, gate, routing policy, and v18 holdout remain "
        "frozen.",
        "",
        "| Variant | Seed | Holdout mode wins | Robust / safe switching | "
        "Safe gain | Event wins | Cell gate |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for variant in protocol.VARIANTS:
        for seed in protocol.TRAINING_SEEDS:
            cell = payload["cells"][variant][str(seed)]
            switching_row = cell["switching"]
            arms = switching_row["arms"]
            lines.append(
                f"| {variant} | {seed} | "
                f"{cell['stationary']['mode_wins']}/4 | "
                f"{arms['robust_sac']['mean']:.1f} / "
                f"{arms['true_mode_safe_utility']['mean']:.1f} | "
                f"{100.0 * switching_row['safe_relative_gain']:+.1f}% | "
                f"{switching_row['safe_event_wins']}/3 | "
                f"{'PASS' if cell['pass'] else 'FAIL'} |"
            )
    lines += [
        "",
        "| Candidate | Mean delta vs control | Seed wins | Event wins | "
        "Min stationary retention | Worst gain candidate / control | Decision |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for variant, row in payload["comparisons"].items():
        lines.append(
            f"| {variant} | {row['mean_safe_switching_delta']:+.1f} | "
            f"{row['paired_seed_wins']}/3 | {row['paired_event_wins']}/9 | "
            f"{100.0 * row['minimum_stationary_retention']:.1f}% | "
            f"{100.0 * row['candidate_worst_seed_safe_gain']:+.1f}% / "
            f"{100.0 * row['control_worst_seed_safe_gain']:+.1f}% | "
            f"{'PASS' if row['pass'] else 'FAIL'} |"
        )
    lines += [
        "",
        "Passing candidates: `" + json.dumps(
            payload["passing_candidates"]) + "`.",
        "",
        "Selected candidate: `" + str(payload["selected_candidate"]) + "`.",
        "",
        f"Diagnosis: **{payload['diagnosis']}**.",
        "",
        "Next step: " + payload["next_step"] + ".",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    payload = analyze()
    markdown = _markdown(payload)
    protocol.write_json_atomic(protocol.analysis_json(), payload)
    protocol.write_text_atomic(protocol.analysis_markdown(), markdown)
    protocol.write_text_atomic(protocol.REPORT, markdown)
    print(markdown, end="")


if __name__ == "__main__":
    main()
