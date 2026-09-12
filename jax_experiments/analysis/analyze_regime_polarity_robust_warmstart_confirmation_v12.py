"""Aggregate the frozen five-seed actor-only warm-start confirmation."""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from jax_experiments.analysis import (
    analyze_regime_polarity_robust_warmstart_specialist_v11 as base,
)
from jax_experiments.analysis import (
    regime_polarity_robust_warmstart_confirmation_v12 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_robust_warmstart_confirmation_audit_v12 as audit,
)


def _bind() -> None:
    base.protocol = protocol


def _cell(seed: int) -> dict[str, Any]:
    _bind()
    audit.validate_audit(seed)
    payload = protocol.read_json(protocol.audit_result(seed))
    stationary = base._stationary_metrics(payload)
    switching = base._switching_metrics(payload)
    return {
        "utility_map": payload["utility_map"],
        "stationary": stationary,
        "switching": switching,
        "pass": bool(stationary["pass"] and switching["pass"]),
    }


def analyze() -> dict[str, Any]:
    protocol.validate_registration()
    cells = {
        str(seed): _cell(seed) for seed in protocol.TRAINING_SEEDS
    }
    seed_passes = sum(int(cell["pass"]) for cell in cells.values())
    confirmed = bool(seed_passes >= protocol.REQUIRED_SEED_PASSES)
    switching_gains = [
        float(cell["switching"]["safe_relative_gain"])
        for cell in cells.values()
    ]
    mode_passes = {
        str(mode): sum(
            int(cells[str(seed)]["stationary"]["modes"][str(mode)]["pass"])
            for seed in protocol.TRAINING_SEEDS
        )
        for mode in protocol.MODES
    }
    if confirmed:
        diagnosis = "actor_only_warmstart_policy_bank_confirmed"
        next_step = (
            "freeze these policy banks and train a causal mode estimator; "
            "evaluate it with the previously established delayed-oracle budget"
        )
    else:
        diagnosis = "actor_only_warmstart_policy_bank_not_confirmed"
        next_step = (
            "close actor-only warm-start specialists as confirmed BAPR evidence; "
            "do not select full-state post hoc"
        )
    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "training_seeds": list(protocol.TRAINING_SEEDS),
        "variant": "actor_only",
        "cells": cells,
        "seed_passes": int(seed_passes),
        "required_seed_passes": protocol.REQUIRED_SEED_PASSES,
        "mode_passes": mode_passes,
        "switching_gain_mean": float(np.mean(switching_gains)),
        "switching_gain_median": float(np.median(switching_gains)),
        "confirmed": confirmed,
        "diagnosis": diagnosis,
        "next_step": next_step,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Actor-only robust-warm-start policy-bank confirmation",
        "",
        "All controllers use fresh confirmation seeds. The robust source and "
        "specialists are compared under matched budgets, and each switching "
        "event uses a distinct frozen four-mode cycle.",
        "",
        "| Seed | Holdout mode wins | Robust / safe switching | Safe gain | "
        "Switch schedules won | Seed gate |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for seed in protocol.TRAINING_SEEDS:
        cell = payload["cells"][str(seed)]
        switching = cell["switching"]
        arms = switching["arms"]
        lines.append(
            f"| {seed} | {cell['stationary']['mode_wins']}/4 | "
            f"{arms['robust_sac']['mean']:.1f} / "
            f"{arms['true_mode_safe_utility']['mean']:.1f} | "
            f"{100.0 * switching['safe_relative_gain']:+.1f}% | "
            f"{switching['safe_event_wins']}/3 | "
            f"{'PASS' if cell['pass'] else 'FAIL'} |"
        )
    lines += [
        "",
        f"Seed decisions: **{payload['seed_passes']}/5 pass** "
        f"(required {payload['required_seed_passes']}/5).",
        "",
        "Per-mode replication: `" + json.dumps(
            payload["mode_passes"], sort_keys=True) + "`.",
        "",
        f"Mean/median switching gain: "
        f"{100.0 * payload['switching_gain_mean']:+.1f}% / "
        f"{100.0 * payload['switching_gain_median']:+.1f}%.",
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
