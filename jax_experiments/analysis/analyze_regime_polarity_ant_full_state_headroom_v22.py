"""Aggregate the registered Ant frozen-recipe headroom screen."""
from __future__ import annotations

from jax_experiments.analysis import (
    analyze_regime_polarity_specialist_stability_v19 as metrics,
)
from jax_experiments.analysis import (
    regime_polarity_ant_full_state_headroom_v22 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_ant_full_state_audit_v22 as audit,
)


def _bind() -> None:
    metrics.protocol = protocol
    metrics.audit = audit


def analyze() -> dict:
    _bind()
    protocol.validate_registration()
    cells = {
        str(seed): metrics._cell_metrics(protocol.CONTROL_VARIANT, seed)
        for seed in protocol.TRAINING_SEEDS
    }
    passed = all(row["pass"] for row in cells.values())
    switching = [
        row["switching"]["arms"]["true_mode_safe_utility"]["mean"]
        for row in cells.values()
    ]
    robust = [
        row["switching"]["arms"]["robust_sac"]["mean"]
        for row in cells.values()
    ]
    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "environment": protocol.ENV,
        "training_seeds": list(protocol.TRAINING_SEEDS),
        "cells": cells,
        "mean_robust_switching": metrics._mean(robust),
        "mean_safe_oracle_switching": metrics._mean(switching),
        "paired_seed_wins": sum(
            current > baseline for current, baseline in zip(switching, robust)),
        "gate_pass": bool(passed),
        "diagnosis": (
            "ant_has_stable_frozen_recipe_policy_bank_headroom"
            if passed else
            "ant_fails_frozen_recipe_policy_bank_headroom"
        ),
        "next_step": (
            "train an Ant-specific causal expected-action estimator without "
            "changing the policy bank"
            if passed else
            "retain Ant as a negative transfer result and do not train an "
            "estimator on this policy bank"
        ),
    }


def _markdown(payload: dict) -> str:
    lines = [
        "# Ant frozen-recipe policy-bank headroom result",
        "",
        "This development screen transfers the frozen V21 robust/full-state "
        "specialist recipe to Ant. It does not train or tune a router.",
        "",
        "| Seed | Stationary mode wins | Robust / safe-oracle switching | "
        "Gain | Event wins | Gate |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for seed in protocol.TRAINING_SEEDS:
        cell = payload["cells"][str(seed)]
        switch = cell["switching"]
        arms = switch["arms"]
        lines.append(
            f"| {seed} | {cell['stationary']['mode_wins']}/4 | "
            f"{arms['robust_sac']['mean']:.1f} / "
            f"{arms['true_mode_safe_utility']['mean']:.1f} | "
            f"{100.0 * switch['safe_relative_gain']:+.1f}% | "
            f"{switch['safe_event_wins']}/3 | "
            f"{'PASS' if cell['pass'] else 'FAIL'} |"
        )
    lines += [
        "",
        f"Mean robust / safe oracle: {payload['mean_robust_switching']:.1f} / "
        f"{payload['mean_safe_oracle_switching']:.1f}.",
        "",
        f"Registered gate: **{'PASS' if payload['gate_pass'] else 'FAIL'}**.",
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
