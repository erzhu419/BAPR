"""Aggregate the Ant matching-mode checkpoint-selection development screen."""
from __future__ import annotations

from jax_experiments.analysis import (
    analyze_regime_polarity_specialist_stability_v19 as metrics,
)
from jax_experiments.analysis import (
    regime_polarity_ant_matching_checkpoint_v23 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_ant_matching_checkpoint_audit_v23 as audit,
)


def _bind() -> None:
    metrics.protocol = protocol
    metrics.audit = audit


def analyze() -> dict:
    _bind()
    protocol.validate_registration()
    cells = {
        variant: {
            str(seed): metrics._cell_metrics(variant, seed)
            for seed in protocol.TRAINING_SEEDS
        }
        for variant in protocol.VARIANTS
    }
    summaries = {}
    for variant in protocol.VARIANTS:
        variant_cells = cells[variant]
        safe = [
            row["switching"]["arms"]["true_mode_safe_utility"]["mean"]
            for row in variant_cells.values()
        ]
        robust = [
            row["switching"]["arms"]["robust_sac"]["mean"]
            for row in variant_cells.values()
        ]
        summaries[variant] = {
            "seed_passes": sum(row["pass"] for row in variant_cells.values()),
            "stationary_mode_passes": sum(
                row["stationary"]["mode_wins"]
                for row in variant_cells.values()),
            "mean_robust_switching": metrics._mean(robust),
            "mean_safe_switching": metrics._mean(safe),
            "mean_safe_gain": metrics._mean([
                row["switching"]["safe_relative_gain"]
                for row in variant_cells.values()
            ]),
            "switching_termination_rate": metrics._mean([
                row["switching"]["arms"]["true_mode_safe_utility"]
                ["terminated_rate"]
                for row in variant_cells.values()
            ]),
            "pass": all(row["pass"] for row in variant_cells.values()),
        }
    passing = [
        variant for variant in protocol.VARIANTS
        if summaries[variant]["pass"]
    ]
    selected = max(
        passing,
        key=lambda variant: (
            summaries[variant]["mean_safe_switching"],
            -protocol.VARIANTS.index(variant),
        ),
        default=None,
    )
    if selected is not None:
        diagnosis = "matching_mode_selection_stabilizes_ant_policy_bank"
        next_step = (
            f"freeze {selected} and test it once on entirely new Ant policy "
            "seeds before training an estimator"
        )
    else:
        diagnosis = "checkpoint_selection_does_not_stabilize_ant_policy_bank"
        next_step = (
            "close checkpoint selection and actor-period tuning; train "
            "specialists on switch-state rollouts with explicit termination "
            "risk while keeping the robust actor as fallback"
        )
    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "environment": protocol.ENV,
        "training_seeds": list(protocol.TRAINING_SEEDS),
        "trained_modes": list(protocol.TRAINED_MODES),
        "frozen_modes": list(protocol.FROZEN_MODES),
        "cells": cells,
        "summaries": summaries,
        "passing_variants": passing,
        "selected_candidate": selected,
        "gate_pass": selected is not None,
        "diagnosis": diagnosis,
        "next_step": next_step,
    }


def _markdown(payload: dict) -> str:
    lines = [
        "# Ant matching-mode checkpoint result",
        "",
        "Modes 0/1 are newly selected on matching-mode validation. Modes 2/3 "
        "reuse the frozen V22 final policies.",
        "",
        "| Variant | Seed | Mode wins | Robust / safe switching | Gain | "
        "Termination | Gate |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for variant in protocol.VARIANTS:
        for seed in protocol.TRAINING_SEEDS:
            cell = payload["cells"][variant][str(seed)]
            switching = cell["switching"]
            robust = switching["arms"]["robust_sac"]
            safe = switching["arms"]["true_mode_safe_utility"]
            lines.append(
                f"| {variant} | {seed} | "
                f"{cell['stationary']['mode_wins']}/4 | "
                f"{robust['mean']:.1f} / {safe['mean']:.1f} | "
                f"{100.0 * switching['safe_relative_gain']:+.1f}% | "
                f"{100.0 * safe['terminated_rate']:.1f}% | "
                f"{'PASS' if cell['pass'] else 'FAIL'} |"
            )
    lines += [
        "",
        f"Selected candidate: `{payload['selected_candidate']}`.",
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
