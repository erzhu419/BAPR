"""Aggregate the registered Ant V24 switch-recovery screen."""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from jax_experiments.analysis import (
    analyze_regime_polarity_specialist_stability_v19 as stationary_metrics,
)
from jax_experiments.analysis import (
    regime_polarity_ant_switch_recovery_v24 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_ant_switch_recovery_audit_v24 as audit,
)


def _mean(values) -> float:
    return float(np.mean([float(value) for value in values]))


def _switching_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    split = payload["switching_holdout"]
    arms = {}
    for arm in audit.SWITCHING_ARMS:
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
            "fallback_action_fraction": _mean(
                row["fallback_action_fraction"] for row in rows),
        }
    robust = arms["robust_sac"]
    candidate = arms["true_mode_safe_transient_fallback"]
    relative_gain = (
        (candidate["mean"] - robust["mean"]) / abs(robust["mean"])
        if robust["mean"] != 0.0 else float("-inf")
    )
    event_wins = sum(
        candidate["event_returns"][str(event_seed)]
        > robust["event_returns"][str(event_seed)]
        for event_seed in protocol.SWITCHING_EVENT_SEEDS
    )
    passed = bool(
        relative_gain >= protocol.MIN_SWITCHING_GAIN
        and event_wins == len(protocol.SWITCHING_EVENT_SEEDS)
        and candidate["terminated_rate"] == 0.0
    )
    return {
        "arms": arms,
        "transient_fallback_relative_gain": float(relative_gain),
        "transient_fallback_event_wins": int(event_wins),
        "pass": passed,
    }


def _cell_metrics(variant: str, seed: int) -> dict[str, Any]:
    audit.validate_audit(variant, seed)
    payload = protocol.read_json(protocol.audit_result(variant, seed))
    stationary_metrics.protocol = protocol
    stationary = stationary_metrics._stationary_metrics(payload)
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
    summaries = {}
    for variant in protocol.VARIANTS:
        rows = cells[variant]
        summaries[variant] = {
            "seed_passes": sum(row["pass"] for row in rows.values()),
            "stationary_mode_passes": sum(
                row["stationary"]["mode_wins"] for row in rows.values()),
            "mean_robust_switching": _mean(
                row["switching"]["arms"]["robust_sac"]["mean"]
                for row in rows.values()),
            "mean_raw_oracle_switching": _mean(
                row["switching"]["arms"]
                ["dynamic_specialist_oracle"]["mean"]
                for row in rows.values()),
            "mean_transient_fallback_switching": _mean(
                row["switching"]["arms"]
                ["true_mode_safe_transient_fallback"]["mean"]
                for row in rows.values()),
            "mean_transient_fallback_gain": _mean(
                row["switching"]["transient_fallback_relative_gain"]
                for row in rows.values()),
            "transient_fallback_termination_rate": _mean(
                row["switching"]["arms"]
                ["true_mode_safe_transient_fallback"]["terminated_rate"]
                for row in rows.values()),
            "pass": all(row["pass"] for row in rows.values()),
        }
    passing = [
        variant for variant in protocol.VARIANTS
        if summaries[variant]["pass"]
    ]
    selected = max(
        passing,
        key=lambda variant: (
            summaries[variant]["mean_transient_fallback_switching"],
            -protocol.VARIANTS.index(variant),
        ),
        default=None,
    )
    raw = summaries["switch_state"]
    risk = summaries["switch_state_risk"]
    risk_reduces_termination = bool(
        risk["transient_fallback_termination_rate"]
        < raw["transient_fallback_termination_rate"]
    )
    if selected is not None:
        diagnosis = "switch_recovery_stabilizes_ant_policy_bank"
        next_step = (
            f"freeze {selected} and run a new five-seed Ant confirmation "
            "before training or transferring the causal estimator"
        )
    elif risk_reduces_termination:
        diagnosis = "termination_risk_helps_but_ant_bank_remains_unstable"
        next_step = (
            "retain the immutable robust fallback and replace scalar terminal "
            "penalty with a learned constrained termination-risk critic"
        )
    else:
        diagnosis = "switch_state_rollouts_do_not_stabilize_ant_policy_bank"
        next_step = (
            "close independent Ant specialists; test a jointly trained "
            "robust-plus-mode controller without changing the benchmark"
        )
    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "environment": protocol.ENV,
        "training_seeds": list(protocol.TRAINING_SEEDS),
        "variants": list(protocol.VARIANTS),
        "cells": cells,
        "summaries": summaries,
        "risk_reduces_termination": risk_reduces_termination,
        "passing_variants": passing,
        "selected_candidate": selected,
        "gate_pass": selected is not None,
        "diagnosis": diagnosis,
        "next_step": next_step,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Ant switch-recovery specialist result",
        "",
        "The robust prefix generates real switch states. Only post-switch "
        "target-mode transitions train each specialist; the risk arm adds a "
        "fixed terminal penalty. Evaluation uses an immutable eight-step "
        "robust fallback after every switch.",
        "",
        "| Variant | Seed | Mode wins | Robust / fallback switching | Gain | "
        "Termination | Gate |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for variant in protocol.VARIANTS:
        for seed in protocol.TRAINING_SEEDS:
            cell = payload["cells"][variant][str(seed)]
            switching = cell["switching"]
            robust = switching["arms"]["robust_sac"]
            candidate = switching["arms"][
                "true_mode_safe_transient_fallback"]
            lines.append(
                f"| {variant} | {seed} | "
                f"{cell['stationary']['mode_wins']}/4 | "
                f"{robust['mean']:.1f} / {candidate['mean']:.1f} | "
                f"{100.0 * switching['transient_fallback_relative_gain']:+.1f}% | "
                f"{100.0 * candidate['terminated_rate']:.1f}% | "
                f"{'PASS' if cell['pass'] else 'FAIL'} |"
            )
    lines += [
        "",
        "Passing variants: `" + json.dumps(payload["passing_variants"]) + "`.",
        "",
        "Selected candidate: `" + str(payload["selected_candidate"]) + "`.",
        "",
        f"Registered gate: **{'PASS' if payload['gate_pass'] else 'FAIL'}**.",
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
