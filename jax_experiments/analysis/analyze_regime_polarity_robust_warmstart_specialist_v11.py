"""Aggregate the robust-warm-started specialist development screen."""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_robust_warmstart_specialist_v11 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_robust_warmstart_specialist_audit_v11 as audit,
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
        robust = _mean(row["return_mean"] for row in robust_rows)
        specialist = _mean(row["return_mean"] for row in specialist_rows)
        relative_gain = (
            (specialist - robust) / abs(robust)
            if robust != 0.0 else float("-inf")
        )
        event_wins = sum(
            specialist_row["return_mean"] > robust_row["return_mean"]
            for robust_row, specialist_row in zip(
                robust_rows, specialist_rows)
        )
        terminated_rate = _mean(
            row["terminated_rate"] for row in specialist_rows)
        passed = bool(
            relative_gain >= protocol.MIN_HOLDOUT_MODE_GAIN
            and event_wins == len(protocol.STATIONARY_HOLDOUT_EVENT_SEEDS)
            and terminated_rate == 0.0
        )
        modes[str(mode)] = {
            "robust_mean": robust,
            "specialist_mean": specialist,
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


def analyze() -> dict[str, Any]:
    protocol.validate_registration()
    cells = {
        variant: {
            str(seed): _cell_metrics(variant, seed)
            for seed in protocol.TRAINING_SEEDS
        }
        for variant in protocol.VARIANTS
    }
    variant_pass = {
        variant: all(
            cells[variant][str(seed)]["pass"]
            for seed in protocol.TRAINING_SEEDS
        )
        for variant in protocol.VARIANTS
    }
    passing = [variant for variant, passed in variant_pass.items() if passed]
    if passing:
        diagnosis = "robust_warmstart_candidate_supported"
        next_step = (
            "freeze the simpler passing initialization and run a completely "
            "new five-seed policy-bank confirmation"
        )
    else:
        diagnosis = "robust_warmstart_not_sufficient"
        next_step = (
            "close independent specialist-bank optimization as the main BAPR "
            "path; retain the mixture as a diagnostic upper bound"
        )
    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "training_seeds": list(protocol.TRAINING_SEEDS),
        "variants": list(protocol.VARIANTS),
        "cells": cells,
        "variant_pass": variant_pass,
        "passing_variants": passing,
        "diagnosis": diagnosis,
        "next_step": next_step,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Robust-warm-started specialist development result",
        "",
        "The matched robust actor is the only shared initialization. No "
        "estimator, gate, residual, or environment parameter is changed.",
        "",
        "| Variant | Seed | Holdout mode wins | Robust / safe switching | "
        "Safe gain | Switching event wins | Cell gate |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for variant in protocol.VARIANTS:
        for seed in protocol.TRAINING_SEEDS:
            cell = payload["cells"][variant][str(seed)]
            switching = cell["switching"]
            arms = switching["arms"]
            lines.append(
                f"| {variant} | {seed} | "
                f"{cell['stationary']['mode_wins']}/4 | "
                f"{arms['robust_sac']['mean']:.1f} / "
                f"{arms['true_mode_safe_utility']['mean']:.1f} | "
                f"{100.0 * switching['safe_relative_gain']:+.1f}% | "
                f"{switching['safe_event_wins']}/3 | "
                f"{'PASS' if cell['pass'] else 'FAIL'} |"
            )
    lines += [
        "",
        "Variant decisions: `" + json.dumps(
            payload["variant_pass"], sort_keys=True) + "`.",
        "",
        f"Diagnosis: **{payload['diagnosis']}**.",
        "",
        "Next step: " + payload["next_step"] + ".",
        "",
        "The gate requires at least 3/4 independently held-out stationary "
        "mode wins and at least 10% safe-oracle switching gain on all three "
        "switching streams for both development seeds.",
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
