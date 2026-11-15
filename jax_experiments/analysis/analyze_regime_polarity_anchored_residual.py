"""Aggregate the fresh-seed anchored residual development decision."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from jax_experiments.analysis import (
    audit_regime_polarity_anchored_residual as audit,
)
from jax_experiments.analysis import (
    regime_polarity_anchored_eval as common,
)
from jax_experiments.analysis import (
    regime_polarity_anchored_residual as protocol,
)


def _finite(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _bootstrap_ci(values, seed: int = 20260729) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(values), size=(20_000, len(values)))
    means = np.mean(values[indices], axis=1)
    return [
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    ]


def _comparison(seed_rows, candidate: str) -> dict[str, Any]:
    deltas = []
    relative = []
    termination_gaps = []
    wins = 0
    per_seed = []
    for row in seed_rows:
        robust = row["arms"]["robust_continue"]
        selected = row["arms"][candidate]
        delta = selected["return_mean"] - robust["return_mean"]
        gain = delta / max(abs(robust["return_mean"]), 1.0)
        termination_gap = (
            selected["terminated_rate"] - robust["terminated_rate"])
        deltas.append(delta)
        relative.append(gain)
        termination_gaps.append(termination_gap)
        wins += int(delta > 0.0)
        per_seed.append({
            "training_seed": row["training_seed"],
            "robust_return": robust["return_mean"],
            "candidate_return": selected["return_mean"],
            "delta": delta,
            "relative_gain": gain,
            "termination_gap": termination_gap,
        })
    return {
        "candidate": candidate,
        "mean_delta": float(np.mean(deltas)),
        "mean_relative_gain": float(np.mean(relative)),
        "paired_delta_ci95": _bootstrap_ci(deltas),
        "seed_wins": int(wins),
        "max_termination_gap": float(np.max(termination_gaps)),
        "per_seed": per_seed,
    }


def _load_seed(seed: int) -> dict[str, Any]:
    manifest = audit.validate(seed)
    arm_returns = {arm: [] for arm in common.AUDIT_ARMS}
    arm_termination = {arm: [] for arm in common.AUDIT_ARMS}
    stationary: dict[tuple[str, int], list[float]] = {
        (arm, mode): []
        for arm in common.AUDIT_ARMS
        for mode in protocol.MODES
    }
    posterior = []
    fallback = []
    for event_seed in protocol.AUDIT_EVENT_SEEDS:
        result = protocol.read_json(
            protocol.audit_dir(seed)
            / f"event_seed_{event_seed}/results.json")
        for row in result["switching"]:
            arm_returns[row["arm"]].extend(row["returns"])
            arm_termination[row["arm"]].append(row["terminated_rate"])
            if row["arm"] == "learned_safe":
                posterior.append(row["posterior_metrics"])
                fallback.append(row["fallback"])
        for row in result["stationary"]:
            stationary[(row["arm"], int(row["mode"]))].extend(
                row["returns"])
    arms = {
        arm: {
            "return_mean": float(np.mean(arm_returns[arm])),
            "return_std": float(np.std(arm_returns[arm])),
            "terminated_rate": float(np.mean(arm_termination[arm])),
            "episodes": len(arm_returns[arm]),
        }
        for arm in common.AUDIT_ARMS
    }
    stationary_rows = [
        {
            "arm": arm,
            "mode": int(mode),
            "return_mean": float(np.mean(stationary[(arm, mode)])),
            "return_std": float(np.std(stationary[(arm, mode)])),
            "episodes": len(stationary[(arm, mode)]),
        }
        for arm in common.AUDIT_ARMS
        for mode in protocol.MODES
    ]
    posterior_summary = {
        key: float(np.mean([row[key] for row in posterior]))
        for key in (
            "mode_accuracy",
            "brier_score",
            "median_switch_delay",
            "p90_switch_delay",
        )
    }
    posterior_summary.update({
        "adaptation_enabled_fraction": float(np.mean([
            row["adaptation_enabled_fraction"] for row in fallback
        ])),
        "fallback_fraction": float(np.mean([
            row["fallback_fraction"] for row in fallback
        ])),
    })
    return {
        "training_seed": int(seed),
        "mode_mask": manifest["mode_mask"],
        "arms": arms,
        "stationary": stationary_rows,
        "learned_safe_posterior": posterior_summary,
    }


def analyze() -> dict[str, Any]:
    seeds = [_load_seed(seed) for seed in protocol.TRAINING_SEEDS]
    base = _comparison(seeds, "anchored_base")
    oracle = _comparison(seeds, "oracle_residual")
    oracle_safe = _comparison(seeds, "oracle_safe")
    learned_raw = _comparison(seeds, "learned_raw")
    learned_safe = _comparison(seeds, "learned_safe")

    base_pass = (
        all(
            row["relative_gain"] >= -(1.0 - protocol.MIN_BASE_PRESERVATION)
            for row in base["per_seed"])
        and base["max_termination_gap"] <= protocol.MAX_TERMINATION_GAP
    )
    oracle_pass = (
        oracle["mean_relative_gain"] >= protocol.MIN_ORACLE_RELATIVE_GAIN
        and oracle["seed_wins"] >= protocol.MIN_SEED_WINS
        and oracle["max_termination_gap"] <= protocol.MAX_TERMINATION_GAP
    )
    learned_pass = (
        learned_safe["mean_relative_gain"]
        >= protocol.MIN_LEARNED_RELATIVE_GAIN
        and learned_safe["seed_wins"] >= protocol.MIN_SEED_WINS
        and learned_safe["max_termination_gap"]
        <= protocol.MAX_TERMINATION_GAP
    )
    estimator_checks = []
    for row in seeds:
        metrics = row["learned_safe_posterior"]
        estimator_checks.append({
            "training_seed": row["training_seed"],
            "pass": bool(
                metrics["mode_accuracy"] >= protocol.MIN_MODE_ACCURACY
                and metrics["brier_score"] <= protocol.MAX_BRIER_SCORE
                and metrics["median_switch_delay"]
                <= protocol.MAX_MEDIAN_SWITCH_DELAY
                and metrics["p90_switch_delay"]
                <= protocol.MAX_P90_SWITCH_DELAY
            ),
            **metrics,
        })
    estimator_pass = all(row["pass"] for row in estimator_checks)
    any_enabled = all(any(row["mode_mask"]) for row in seeds)
    promotion = bool(
        base_pass and oracle_pass and learned_pass
        and estimator_pass and any_enabled)
    failures = []
    if not base_pass:
        failures.append("anchored base did not preserve paired robust control")
    if not oracle_pass:
        failures.append("zero-initialized oracle residual lacked stable headroom")
    if not learned_pass:
        failures.append("calibrated learned fallback lacked stable gain")
    if not estimator_pass:
        failures.append("frozen estimator failed under the new controller")
    if not any_enabled:
        failures.append(
            "calibration did not enable a residual mode for every seed")
    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "env": protocol.ENV,
        "family": protocol.FAMILY,
        "development_only": True,
        "training_seeds": list(protocol.TRAINING_SEEDS),
        "audit_event_seeds": list(protocol.AUDIT_EVENT_SEEDS),
        "thresholds": {
            "min_base_preservation": protocol.MIN_BASE_PRESERVATION,
            "min_oracle_relative_gain": protocol.MIN_ORACLE_RELATIVE_GAIN,
            "min_learned_relative_gain": protocol.MIN_LEARNED_RELATIVE_GAIN,
            "min_seed_wins": protocol.MIN_SEED_WINS,
            "max_termination_gap": protocol.MAX_TERMINATION_GAP,
            "confidence_threshold": protocol.CONFIDENCE_THRESHOLD,
        },
        "seed_results": seeds,
        "comparisons": {
            "anchored_base": base,
            "oracle_residual": oracle,
            "oracle_safe": oracle_safe,
            "learned_raw": learned_raw,
            "learned_safe": learned_safe,
        },
        "estimator_checks": estimator_checks,
        "gates": {
            "base_preservation": bool(base_pass),
            "oracle_headroom": bool(oracle_pass),
            "learned_safe_gain": bool(learned_pass),
            "estimator": bool(estimator_pass),
            "at_least_one_calibrated_mode_per_seed": bool(any_enabled),
        },
        "promotion_to_untouched_five_seed_confirmation": promotion,
        "failures": failures,
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Anchored residual development result",
        "",
        (
            f"Decision: **{'PASS' if result['promotion_to_untouched_five_seed_confirmation'] else 'FAIL'}** "
            "for promotion to an untouched five-seed confirmation."
        ),
        "",
        "| arm | mean delta | relative gain | wins | max term gap |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in (
            "anchored_base", "oracle_residual", "oracle_safe",
            "learned_raw", "learned_safe"):
        row = result["comparisons"][arm]
        lines.append(
            f"| {arm} | {row['mean_delta']:.1f} | "
            f"{100.0 * row['mean_relative_gain']:.1f}% | "
            f"{row['seed_wins']}/{len(protocol.TRAINING_SEEDS)} | "
            f"{row['max_termination_gap']:.3f} |")
    lines.extend([
        "",
        "| seed | mode mask | robust | base | oracle | learned safe |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    for row in result["seed_results"]:
        arms = row["arms"]
        lines.append(
            f"| {row['training_seed']} | "
            f"{''.join('1' if value else '0' for value in row['mode_mask'])} | "
            f"{arms['robust_continue']['return_mean']:.1f} | "
            f"{arms['anchored_base']['return_mean']:.1f} | "
            f"{arms['oracle_residual']['return_mean']:.1f} | "
            f"{arms['learned_safe']['return_mean']:.1f} |")
    if result["failures"]:
        lines.extend(["", "Failed gates:"])
        lines.extend(f"- {failure}" for failure in result["failures"])
    lines.extend([
        "",
        (
            "This is a three-seed development screen. A pass permits a new "
            "untouched five-seed run; it is not itself a paper claim."
        ),
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    result = analyze()
    protocol.write_json_atomic(protocol.analysis_json(), result)
    protocol.write_text_atomic(
        protocol.analysis_markdown(), _markdown(result))
    print(
        "ANCHORED ANALYSIS COMPLETE: "
        f"promotion={result['promotion_to_untouched_five_seed_confirmation']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
