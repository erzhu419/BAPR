"""Aggregate the frozen V33 canonical-controller training ablation."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_canonical_training_ablation_v33 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_canonical_training_audit_v33 as audit,
)


T_CRITICAL_95_DF4 = 2.7764451051977987


def _mean(values) -> float:
    return float(np.mean([float(value) for value in values]))


def _arm(recipe: str, route: str) -> str:
    return audit.arm(recipe, route)


def _switching_seed_mean(payload: dict[str, Any], arm: str) -> float:
    return _mean(
        payload["switching_holdout"][str(event_seed)][arm]["return_mean"]
        for event_seed in protocol.SWITCHING_EVENT_SEEDS
    )


def _stationary_seed_mean(payload: dict[str, Any], arm: str) -> float:
    return _mean(
        payload["stationary_holdout"][str(event_seed)][arm][str(mode)][
            "return_mean"
        ]
        for event_seed in protocol.STATIONARY_HOLDOUT_EVENT_SEEDS
        for mode in protocol.MODES
    )


def _paired(
    payloads: dict[int, dict[str, Any]], left: str, right: str,
) -> dict[str, Any]:
    values = {
        str(seed): (
            _switching_seed_mean(payload, left)
            - _switching_seed_mean(payload, right)
        )
        for seed, payload in payloads.items()
    }
    differences = np.asarray(list(values.values()), dtype=np.float64)
    mean = float(np.mean(differences))
    radius = T_CRITICAL_95_DF4 * float(
        np.std(differences, ddof=1) / math.sqrt(len(differences)))
    return {
        "left": left,
        "right": right,
        "paired_seed_differences": values,
        "paired_mean": mean,
        "ci95_low": float(mean - radius),
        "ci95_high": float(mean + radius),
        "seed_wins": int(np.sum(differences > 0.0)),
    }


def analyze() -> dict[str, Any]:
    protocol.validate_registration()
    payloads: dict[int, dict[str, Any]] = {}
    for seed in protocol.TRAINING_SEEDS:
        audit.validate_audit(seed)
        payloads[seed] = protocol.read_json(protocol.audit_result(seed))

    switching_means = {
        arm: _mean(_switching_seed_mean(payload, arm)
                   for payload in payloads.values())
        for arm in protocol.ARMS
    }
    stationary_means = {
        arm: _mean(_stationary_seed_mean(payload, arm)
                   for payload in payloads.values())
        for arm in protocol.ARMS
    }

    recipe_metrics: dict[str, Any] = {}
    robust = protocol.ROBUST_ARM
    warmstart_causal = _arm(protocol.WARMSTART_RECIPE, "causal")
    for recipe in protocol.CANONICAL_RECIPES:
        oracle_arm = _arm(recipe, "oracle")
        causal_arm = _arm(recipe, "causal")
        no_comp_arm = _arm(recipe, "no_compensation")
        retention = {}
        equivalence_pass = True
        for seed, payload in payloads.items():
            oracle_stationary = _stationary_seed_mean(payload, oracle_arm)
            causal_stationary = _stationary_seed_mean(payload, causal_arm)
            ratio = (
                causal_stationary / oracle_stationary
                if oracle_stationary > 0.0 else None
            )
            retention[str(seed)] = {
                "oracle_return": oracle_stationary,
                "causal_return": causal_stationary,
                "ratio": float(ratio) if ratio is not None else None,
                "pass": bool(
                    ratio is not None
                    and ratio >= protocol.MIN_STATIONARY_RETENTION),
            }
            equivalence_pass = equivalence_pass and all(
                row["pass"] is True
                for row in payload["oracle_equivalence"][recipe].values()
            )
        causal_vs_robust = _paired(payloads, causal_arm, robust)
        causal_vs_warmstart = (
            None if recipe == protocol.WARMSTART_RECIPE
            else _paired(payloads, causal_arm, warmstart_causal)
        )
        oracle_vs_robust = _paired(payloads, oracle_arm, robust)
        inference_loss = {
            str(seed): (
                _switching_seed_mean(payload, oracle_arm)
                - _switching_seed_mean(payload, causal_arm)
            )
            for seed, payload in payloads.items()
        }
        retention_passes = sum(row["pass"] for row in retention.values())
        eligible = bool(
            recipe != protocol.WARMSTART_RECIPE
            and causal_vs_warmstart is not None
            and causal_vs_warmstart["paired_mean"] > 0.0
            and causal_vs_warmstart["seed_wins"]
            >= protocol.REQUIRED_RECIPE_WINS
            and retention_passes >= protocol.REQUIRED_RETENTION_SEEDS
            and equivalence_pass
        )
        recipe_metrics[recipe] = {
            "switching": {
                "no_compensation_mean": switching_means[no_comp_arm],
                "oracle_mean": switching_means[oracle_arm],
                "causal_mean": switching_means[causal_arm],
            },
            "stationary": {
                "oracle_mean": stationary_means[oracle_arm],
                "causal_mean": stationary_means[causal_arm],
            },
            "causal_vs_robust": causal_vs_robust,
            "causal_vs_warmstart": causal_vs_warmstart,
            "oracle_vs_robust": oracle_vs_robust,
            "inference_loss_by_seed": inference_loss,
            "inference_loss_mean": _mean(inference_loss.values()),
            "stationary_retention": retention,
            "stationary_retention_seed_passes": int(retention_passes),
            "oracle_equivalence_pass": bool(equivalence_pass),
            "promotion_eligible": eligible,
        }

    eligible = [
        recipe for recipe in (protocol.DIRECT_RECIPE, protocol.CANONICAL_RECIPE)
        if recipe_metrics[recipe]["promotion_eligible"]
    ]
    selected = (
        max(
            eligible,
            key=lambda recipe: recipe_metrics[recipe]["switching"][
                "causal_mean"],
        )
        if eligible else protocol.WARMSTART_RECIPE
    )
    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "registration": protocol.file_record(protocol.REGISTRATION_PATH),
        "training_seeds": list(protocol.TRAINING_SEEDS),
        "robust_switching_mean": switching_means[robust],
        "robust_stationary_mean": stationary_means[robust],
        "recipe_metrics": recipe_metrics,
        "selected_recipe": selected,
        "new_recipe_promoted": selected != protocol.WARMSTART_RECIPE,
        "selection_interpretation": (
            "exploratory development selection; not a new efficacy confirmation"
        ),
        "accounting": {
            "interactions_per_final_recipe_seed": protocol.FINAL_TOTAL_STEPS,
            "warmstart_source_plus_finetune": True,
            "frozen_estimator_retrained": False,
            "new_training_seeds": len(protocol.TRAINING_SEEDS),
        },
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# V33 canonical-controller training ablation",
        "",
        "Five new development seeds compare equal-interaction controller "
        "training recipes. This is recipe selection, not confirmatory evidence.",
        "",
        "| Recipe | Oracle switch | Causal switch | Causal stationary | "
        "Causal - robust | Causal - warm-start | Eligible |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for recipe in protocol.CANONICAL_RECIPES:
        row = payload["recipe_metrics"][recipe]
        vs_robust = row["causal_vs_robust"]
        vs_warmstart = row["causal_vs_warmstart"]
        lines.append(
            f"| {recipe} | {row['switching']['oracle_mean']:.1f} | "
            f"{row['switching']['causal_mean']:.1f} | "
            f"{row['stationary']['causal_mean']:.1f} | "
            f"{vs_robust['paired_mean']:+.1f} "
            f"({vs_robust['seed_wins']}/5) | "
            + (
                "reference | " if vs_warmstart is None else
                f"{vs_warmstart['paired_mean']:+.1f} "
                f"({vs_warmstart['seed_wins']}/5) | "
            )
            + ("yes" if row["promotion_eligible"] else "no")
            + " |"
        )
    lines.extend([
        "",
        f"Equal-budget robust SAC switching mean: "
        f"{payload['robust_switching_mean']:.1f}.",
        f"Selected development recipe: **{payload['selected_recipe']}**.",
        "",
        "A new recipe is promoted only if its causal mean exceeds the frozen "
        "warm-start recipe, it wins at least 4/5 paired seeds, preserves at "
        "least 95% stationary oracle performance in at least 4/5 seeds, and "
        "passes exact oracle trajectory equivalence. The frozen V32 7/10 "
        "average-effect result and failed composite gate are unchanged.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    payload = analyze()
    text = render(payload)
    protocol.ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    protocol.write_json_atomic(protocol.analysis_json(), payload)
    protocol.write_text_atomic(protocol.analysis_markdown(), text)
    protocol.write_text_atomic(protocol.REPORT, text)
    print(text, flush=True)


if __name__ == "__main__":
    main()
