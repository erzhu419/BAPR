"""Aggregate the preregistered v20 specialist policy-stability screen."""
from __future__ import annotations

import json

from jax_experiments.analysis import (
    analyze_regime_polarity_specialist_stability_v19 as base,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_policy_stability_v20 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_policy_stability_audit_v20 as audit,
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
    comparisons = {
        variant: base._comparison(variant, cells)
        for variant in protocol.VARIANTS
        if variant != protocol.CONTROL_VARIANT
    }
    passing = [
        variant for variant, row in comparisons.items() if row["pass"]
    ]
    mean_safe_return = {
        variant: base._mean(
            cells[variant][str(seed)]["switching"]["arms"]
            ["true_mode_safe_utility"]["mean"]
            for seed in protocol.TRAINING_SEEDS
        )
        for variant in protocol.VARIANTS
    }
    selected = None
    if passing:
        best = max(passing, key=lambda name: mean_safe_return[name])
        if (
            "full_state_best" in passing
            and mean_safe_return["full_state_best"]
            >= 0.98 * mean_safe_return[best]
        ):
            selected = "full_state_best"
        else:
            selected = best
    if selected is None:
        diagnosis = "policy_selection_and_period_do_not_resolve_bank_variance"
        next_step = (
            "close validation checkpoint selection and actor-period thinning; "
            "do not alter the frozen posterior or v18/v19 holdouts"
        )
    else:
        diagnosis = "specialist_policy_stability_candidate_selected"
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


def _markdown(payload) -> str:
    lines = [
        "# V20 full-state specialist policy-stability result",
        "",
        "This fresh-seed screen changes only policy selection and actor update "
        "frequency after a full-controller warm start.",
        "",
        "| Variant | Seed | Holdout mode wins | Robust / safe switching | "
        "Safe gain | Event wins | Cell gate |",
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
