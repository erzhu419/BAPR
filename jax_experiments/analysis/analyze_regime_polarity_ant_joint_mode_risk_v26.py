"""Aggregate the registered Ant V26 joint-controller screen."""
from __future__ import annotations

import json

from jax_experiments.analysis import (
    analyze_regime_polarity_ant_switch_recovery_v24 as base,
)
from jax_experiments.analysis import (
    regime_polarity_ant_joint_mode_risk_v26 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_ant_joint_mode_risk_audit_v26 as audit,
)


def _bind() -> None:
    base.protocol = protocol
    base.audit = audit


def analyze():
    _bind()
    protocol.validate_registration()
    cells = {
        variant: {
            str(seed): base._cell_metrics(variant, seed)
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
            "mean_robust_switching": base._mean(
                row["switching"]["arms"]["robust_sac"]["mean"]
                for row in rows.values()),
            "mean_raw_joint_switching": base._mean(
                row["switching"]["arms"]
                ["dynamic_specialist_oracle"]["mean"]
                for row in rows.values()),
            "mean_safe_joint_switching": base._mean(
                row["switching"]["arms"]
                ["true_mode_safe_transient_fallback"]["mean"]
                for row in rows.values()),
            "mean_safe_joint_gain": base._mean(
                row["switching"]["transient_fallback_relative_gain"]
                for row in rows.values()),
            "safe_joint_termination_rate": base._mean(
                row["switching"]["arms"]
                ["true_mode_safe_transient_fallback"]["terminated_rate"]
                for row in rows.values()),
            "pass": all(row["pass"] for row in rows.values()),
        }
    passing = [
        variant for variant in protocol.VARIANTS
        if summaries[variant]["pass"]
    ]
    selected = next(
        (variant for variant in protocol.VARIANTS if variant in passing),
        None,
    )
    if selected == "joint_equal_budget":
        diagnosis = "joint_mode_controller_stabilizes_ant_at_equal_budget"
        next_step = (
            "freeze the equal-budget joint controller and run a new five-seed "
            "Ant confirmation before attaching a causal mode estimator"
        )
    elif selected == "joint_data_matched":
        diagnosis = "joint_mode_controller_requires_bank_matched_data"
        next_step = (
            "freeze the data-matched capacity result, then test return-aware "
            "compression into the equal-budget controller on new seeds"
        )
    else:
        diagnosis = "joint_mode_controller_does_not_stabilize_ant"
        next_step = (
            "close Ant controller optimization under actuator polarity and "
            "retain it as a negative transfer case; do not train an estimator"
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
        "passing_variants": passing,
        "selected_candidate": selected,
        "gate_pass": selected is not None,
        "diagnosis": diagnosis,
        "next_step": next_step,
    }


def _markdown(payload) -> str:
    lines = [
        "# Ant joint mode-conditioned risk-controller result",
        "",
        "Both budgets use one shared true-mode-conditioned actor, critic, and "
        "relative termination-risk critic with an immutable robust fallback. "
        "The data-matched arm supplies four times the continuation data so "
        "each mode receives the same unique-data scale as a V25 specialist.",
        "",
        "| Variant | Seed | Mode wins | Robust / safe joint switching | Gain | "
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
