#!/usr/bin/env python3
"""Export compact, path-free per-seed records for forward-dynamics vs. inverse-dynamics
and direct-classifier vs. inverse-dynamics identification baselines into reproducibility/."""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
FWD_SRC = ROOT / "jax_experiments/results_regime_polarity_posterior_analysis_v1/analysis.json"
INV_SRC = ROOT / "jax_experiments/results_regime_polarity_inverse_system_id_analysis_v3/analysis.json"
CLS_SRC = ROOT / "jax_experiments/results_mechanism_gap_v34/analysis/identification/analysis.json"
OUT_PATH = ROOT / "reproducibility/identification_baselines/summary.json"


def paired_ci(xs: list[float], ys: list[float]) -> dict:
    diffs = [float(a - b) for a, b in zip(xs, ys, strict=True)]
    n = len(diffs)
    mean = statistics.fmean(diffs)
    sd = statistics.stdev(diffs)
    half = float(student_t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    return {
        "n_seeds": n,
        "mean": mean,
        "ci95": [mean - half, mean + half],
        "wins": sum(d > 0 for d in diffs),
        "differences": diffs,
    }


def extract_screen(raw: dict, model_family: str) -> dict:
    seeds = [int(s) for s in raw["test_controller_seeds"]]
    per_seed = {}
    for s in seeds:
        row = raw["by_training_seed"][str(s)]
        per_seed[str(s)] = {
            "switching": {k: float(row["switching"][k]["mean"]) for k in ("robust", "oracle", "learned_map", "learned_soft")},
            "stationary": {k: float(row["stationary"][k]["mean"]) for k in ("robust", "oracle", "learned_map", "learned_soft")},
            "posterior_map": {
                "mode_accuracy": float(row["posterior"]["learned_map"]["mode_accuracy"]),
                "brier_score": float(row["posterior"]["learned_map"]["brier_score"]),
                "median_switch_delay": float(row["posterior"]["learned_map"]["median_switch_delay"]),
                "p90_switch_delay": float(row["posterior"]["learned_map"]["p90_switch_delay"]),
            },
            "posterior_soft": {
                "mode_accuracy": float(row["posterior"]["learned_soft"]["mode_accuracy"]),
                "brier_score": float(row["posterior"]["learned_soft"]["brier_score"]),
                "median_switch_delay": float(row["posterior"]["learned_soft"]["median_switch_delay"]),
                "p90_switch_delay": float(row["posterior"]["learned_soft"]["p90_switch_delay"]),
            },
        }
    sw = {
        arm: [per_seed[str(s)]["switching"][arm] for s in seeds]
        for arm in ("robust", "oracle", "learned_map", "learned_soft")
    }
    return {
        "model_family": model_family,
        "training_controller_seeds": [int(x) for x in raw["training_controller_seeds"]],
        "validation_controller_seeds": [int(x) for x in raw["validation_controller_seeds"]],
        "test_controller_seeds": seeds,
        "test_event_seeds": [int(x) for x in raw["test_event_seeds"]],
        "per_seed": per_seed,
        "switching_means": {k: statistics.fmean(v) for k, v in sw.items()},
        "posterior_summary": {
            "learned_map": {
                "mode_accuracy": float(raw["posterior_summary"]["learned_map"]["mode_accuracy"]),
                "brier_score": float(raw["posterior_summary"]["learned_map"]["brier_score"]),
                "median_switch_delay": float(raw["posterior_summary"]["learned_map"]["median_switch_delay"]),
                "p90_switch_delay": float(raw["posterior_summary"]["learned_map"]["p90_switch_delay"]),
            },
            "learned_soft": {
                "mode_accuracy": float(raw["posterior_summary"]["learned_soft"]["mode_accuracy"]),
                "brier_score": float(raw["posterior_summary"]["learned_soft"]["brier_score"]),
                "median_switch_delay": float(raw["posterior_summary"]["learned_soft"]["median_switch_delay"]),
                "p90_switch_delay": float(raw["posterior_summary"]["learned_soft"]["p90_switch_delay"]),
            },
        },
        "comparisons_vs_robust": {
            "learned_map_minus_robust": paired_ci(sw["learned_map"], sw["robust"]),
            "learned_soft_minus_robust": paired_ci(sw["learned_soft"], sw["robust"]),
        },
    }


def main() -> None:
    fwd = json.loads(FWD_SRC.read_text())
    inv = json.loads(INV_SRC.read_text())
    cls_raw = json.loads(CLS_SRC.read_text())

    fwd_clean = extract_screen(fwd, "forward_dynamics_reward_ensemble")
    inv_clean = extract_screen(inv, "transition_only_inverse_dynamics_ensemble")

    seeds_5 = fwd_clean["test_controller_seeds"]
    inv_map = [inv_clean["per_seed"][str(s)]["switching"]["learned_map"] for s in seeds_5]
    fwd_map = [fwd_clean["per_seed"][str(s)]["switching"]["learned_map"] for s in seeds_5]
    inv_soft = [inv_clean["per_seed"][str(s)]["switching"]["learned_soft"] for s in seeds_5]
    fwd_soft = [fwd_clean["per_seed"][str(s)]["switching"]["learned_soft"] for s in seeds_5]

    payload = {
        "protocol": "bapr-identification-baselines-summary",
        "historical_cohorts_pooled": False,
        "forward_vs_inverse_same_split": {
            "forward_dynamics": fwd_clean,
            "inverse_dynamics": inv_clean,
            "paired_inverse_minus_forward": {
                "map_routing": paired_ci(inv_map, fwd_map),
                "soft_routing": paired_ci(inv_soft, fwd_soft),
            },
        },
        "direct_classifier_vs_inverse_same_data": {
            "policy_seeds": [int(s) for s in cls_raw["seeds"]],
            "estimator_seeds": [94103, 94121, 94139],
            "seed_means": {
                k: [float(x) for x in cls_raw["seed_means"][k]]
                for k in (
                    "oracle", "uncompensated", "frozen", "no_filter", "single_head",
                    "inverse_94103", "classifier_94103",
                    "inverse_94121", "classifier_94121",
                    "inverse_94139", "classifier_94139",
                )
            },
            "means": {
                k: float(cls_raw["means"][k])
                for k in (
                    "oracle", "uncompensated", "frozen", "no_filter", "single_head",
                    "inverse_94103", "classifier_94103",
                    "inverse_94121", "classifier_94121",
                    "inverse_94139", "classifier_94139",
                )
            },
            "paired_comparisons": {
                "frozen_minus_no_filter": cls_raw["comparisons"]["frozen-minus-no_filter"],
                "frozen_minus_single_head": cls_raw["comparisons"]["frozen-minus-single_head"],
                "inverse_minus_classifier_94103": cls_raw["comparisons"]["paired-inverse-minus-classifier-94103"],
                "inverse_minus_classifier_94121": cls_raw["comparisons"]["paired-inverse-minus-classifier-94121"],
                "inverse_minus_classifier_94139": cls_raw["comparisons"]["paired-inverse-minus-classifier-94139"],
            },
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
