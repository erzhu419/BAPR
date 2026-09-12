"""Aggregate the registered Ant V25 constrained-risk screen."""
from __future__ import annotations

import json

from jax_experiments.analysis import (
    analyze_regime_polarity_ant_switch_recovery_v24 as base,
)
from jax_experiments.analysis import (
    regime_polarity_ant_constrained_risk_v25 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_ant_constrained_risk_audit_v25 as audit,
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
            "mean_raw_oracle_switching": base._mean(
                row["switching"]["arms"]
                ["dynamic_specialist_oracle"]["mean"]
                for row in rows.values()),
            "mean_transient_fallback_switching": base._mean(
                row["switching"]["arms"]
                ["true_mode_safe_transient_fallback"]["mean"]
                for row in rows.values()),
            "mean_transient_fallback_gain": base._mean(
                row["switching"]["transient_fallback_relative_gain"]
                for row in rows.values()),
            "transient_fallback_termination_rate": base._mean(
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
    relative_safer = bool(
        summaries["risk_q_relative"]
        ["transient_fallback_termination_rate"]
        < summaries["risk_q_absolute"]
        ["transient_fallback_termination_rate"]
    )
    if selected is not None:
        diagnosis = "learned_constrained_risk_stabilizes_ant_policy_bank"
        next_step = (
            f"freeze {selected} and run a new five-seed Ant confirmation "
            "before any estimator transfer"
        )
    else:
        diagnosis = "learned_risk_does_not_stabilize_independent_ant_bank"
        next_step = (
            "close independent Ant specialist optimization and train one "
            "joint robust-plus-mode conditioned controller with the same "
            "relative termination-risk constraint"
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
        "relative_objective_is_safer": relative_safer,
        "passing_variants": passing,
        "selected_candidate": selected,
        "gate_pass": selected is not None,
        "diagnosis": diagnosis,
        "next_step": next_step,
    }


def _markdown(payload) -> str:
    lines = [
        "# Ant constrained termination-risk result",
        "",
        "Both variants use a learned discounted termination-risk critic. "
        "The absolute arm penalizes candidate risk directly; the relative "
        "arm penalizes only risk above the immutable robust action.",
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
