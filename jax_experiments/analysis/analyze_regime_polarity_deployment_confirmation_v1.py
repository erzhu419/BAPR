"""Aggregate the preregistered fresh ten-seed deployment confirmation."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from jax_experiments.analysis import (
    regime_polarity_deployment_confirmation_v1 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_deployment_confirmation_audit_v1 as auditor,
)


BASELINES = ("sac", "escp_recurrent", "resac_b0")


def _seed_summary(payload: dict[str, Any]) -> dict[str, Any]:
    events = payload["events"]
    switching = [float(row["switching"]["return_mean"]) for row in events]
    stationary = [
        float(mode["return_mean"])
        for row in events for mode in row["stationary"]
    ]
    return {
        "switching_mean": float(np.mean(switching)),
        "stationary_mean": float(np.mean(stationary)),
        "stationary_by_mode": {
            str(mode): float(np.mean([
                event["stationary"][mode]["return_mean"]
                for event in events]))
            for mode in protocol.MODES
        },
        "switching_terminated_rate": float(np.mean([
            row["switching"]["terminated_rate"] for row in events])),
        "stationary_terminated_rate": float(np.mean([
            mode["terminated_rate"]
            for row in events for mode in row["stationary"]])),
    }


def _paired(candidate, reference) -> dict[str, Any]:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    expected = (len(protocol.TRAINING_SEEDS),)
    if candidate.shape != expected or reference.shape != expected:
        raise ValueError(f"confirmation requires paired vectors {expected}")
    deltas = candidate - reference
    mean = float(np.mean(deltas))
    sd = float(np.std(deltas, ddof=1))
    se = sd / math.sqrt(len(deltas))
    if se == 0.0:
        statistic = math.inf if mean > 0 else (-math.inf if mean < 0 else 0.0)
        p_value = 0.0 if mean > 0 else 1.0
        lower = mean
        simultaneous_lower = mean
    else:
        statistic = mean / se
        p_value = float(student_t.sf(statistic, len(deltas) - 1))
        lower = mean - float(student_t.ppf(
            1.0 - protocol.FAMILY_ALPHA, len(deltas) - 1)) * se
        simultaneous_lower = mean - float(student_t.ppf(
            1.0 - protocol.FAMILY_ALPHA / len(BASELINES),
            len(deltas) - 1)) * se
    return {
        "candidate_mean": float(np.mean(candidate)),
        "reference_mean": float(np.mean(reference)),
        "mean_delta": mean,
        "relative_delta": mean / max(abs(float(np.mean(reference))), 1e-8),
        "paired_standard_deviation": sd,
        "paired_t_statistic": statistic,
        "one_sided_p_value": p_value,
        "one_sided_95pct_lower_bound": lower,
        "bonferroni_simultaneous_95pct_lower_bound": simultaneous_lower,
        "seed_slot_deltas": {
            str(seed): float(value)
            for seed, value in zip(protocol.TRAINING_SEEDS, deltas)
        },
        "seed_slot_wins": int(np.sum(deltas > 0.0)),
    }


def _apply_holm(rows: dict[str, dict[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda key: rows[key]["one_sided_p_value"])
    adjusted_so_far = 0.0
    for rank, key in enumerate(ordered):
        scale = len(ordered) - rank
        adjusted = min(1.0, scale * rows[key]["one_sided_p_value"])
        adjusted_so_far = max(adjusted_so_far, adjusted)
        rows[key]["holm_adjusted_p_value"] = adjusted_so_far
        rows[key]["holm_reject_at_0p05"] = bool(
            adjusted_so_far <= protocol.FAMILY_ALPHA)


def _load(method: str, seed: int) -> dict[str, Any]:
    auditor.validate_manifest(method, seed)
    return protocol.read_json(protocol.audit_dir(method, seed) / "results.json")


def analyze() -> dict[str, Any]:
    registration = protocol.validate_registration()
    release = protocol.validate_mechanism_release()
    payloads = {
        method: {seed: _load(method, seed) for seed in protocol.TRAINING_SEEDS}
        for method in protocol.METHODS
    }
    summaries = {
        method: {
            seed: _seed_summary(payloads[method][seed])
            for seed in protocol.TRAINING_SEEDS
        }
        for method in protocol.METHODS
    }

    for event_index, event_seed in enumerate(protocol.EVENT_SEEDS):
        hashes = {
            payloads[method][seed]["events"][event_index]["switching"]
            ["mode_trace_sha256"]
            for method in protocol.METHODS for seed in protocol.TRAINING_SEEDS
        }
        if len(hashes) != 1:
            raise ValueError(
                f"confirmation mode streams differ for event {event_seed}: {hashes}")

    method_rows: dict[str, Any] = {}
    vectors: dict[str, dict[str, np.ndarray]] = {}
    for method in protocol.METHODS:
        switching = np.asarray([
            summaries[method][seed]["switching_mean"]
            for seed in protocol.TRAINING_SEEDS], dtype=np.float64)
        stationary = np.asarray([
            summaries[method][seed]["stationary_mean"]
            for seed in protocol.TRAINING_SEEDS], dtype=np.float64)
        vectors[method] = {"switching": switching, "stationary": stationary}
        method_rows[method] = {
            "seed_summaries": {
                str(seed): summaries[method][seed]
                for seed in protocol.TRAINING_SEEDS
            },
            "switching_mean": float(np.mean(switching)),
            "switching_std_over_seeds": float(np.std(switching, ddof=1)),
            "stationary_mean": float(np.mean(stationary)),
            "stationary_std_over_seeds": float(np.std(stationary, ddof=1)),
            "stationary_by_mode": {
                str(mode): float(np.mean([
                    summaries[method][seed]["stationary_by_mode"][str(mode)]
                    for seed in protocol.TRAINING_SEEDS]))
                for mode in protocol.MODES
            },
            "switching_terminated_rate": float(np.mean([
                summaries[method][seed]["switching_terminated_rate"]
                for seed in protocol.TRAINING_SEEDS])),
            "stationary_terminated_rate": float(np.mean([
                summaries[method][seed]["stationary_terminated_rate"]
                for seed in protocol.TRAINING_SEEDS])),
        }

    comparisons = {
        baseline: _paired(
            vectors["bapr"]["switching"], vectors[baseline]["switching"])
        for baseline in BASELINES
    }
    _apply_holm(comparisons)
    for baseline, row in comparisons.items():
        row["stationary_retention"] = float(
            np.mean(vectors["bapr"]["stationary"])
            / max(np.mean(vectors[baseline]["stationary"]), 1e-8))
        row["switching_termination_gap"] = float(
            method_rows["bapr"]["switching_terminated_rate"]
            - method_rows[baseline]["switching_terminated_rate"])
        row["registered_pass"] = bool(
            row["mean_delta"] > 0.0
            and row["holm_reject_at_0p05"]
            and row["bonferroni_simultaneous_95pct_lower_bound"] > 0.0
            and row["seed_slot_wins"] >= protocol.MIN_SEED_WINS
            and row["stationary_retention"]
            >= protocol.MIN_STATIONARY_RETENTION
            and row["switching_termination_gap"]
            <= protocol.MAX_TERMINATION_GAP)

    strongest = np.asarray([
        max(vectors[method]["switching"][index] for method in BASELINES)
        for index in range(len(protocol.TRAINING_SEEDS))
    ])
    strongest_diagnostic = _paired(vectors["bapr"]["switching"], strongest)
    strongest_diagnostic["inferential_role"] = (
        "diagnostic_only; comparator selected separately in each seed slot")

    baseline_reproduction = {
        "escp_recurrent_minus_sac": _paired(
            vectors["escp_recurrent"]["switching"],
            vectors["sac"]["switching"]),
        "resac_b0_minus_sac": _paired(
            vectors["resac_b0"]["switching"],
            vectors["sac"]["switching"]),
    }
    primary_pass = all(row["registered_pass"] for row in comparisons.values())
    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "registration": protocol.registration_record(),
        "mechanism_release_evidence": release,
        "training_seeds": list(protocol.TRAINING_SEEDS),
        "event_seeds": list(protocol.EVENT_SEEDS),
        "methods": method_rows,
        "registered_comparisons": comparisons,
        "strongest_seedwise_envelope_diagnostic": strongest_diagnostic,
        "baseline_reproduction": baseline_reproduction,
        "primary_pass": primary_pass,
        "claim_scope": (
            "deployment performance of the frozen BAPR teacher-estimator-"
            "student pipeline on the HalfCheetah actuator-polarity protocol; "
            "not sample efficiency, end-to-end training-budget parity, or "
            "universal nonstationary-RL superiority"
        ),
        "registered_source_count": len(registration["source_records"]),
    }


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Fresh ten-seed BAPR deployment confirmation",
        "",
        "The old five model seeds are excluded. All four methods use the same "
        "five untouched event streams and strict deterministic evaluation.",
        "",
        "| method | switching mean +/- sd | stationary mean +/- sd | "
        "switch termination |",
        "|---|---:|---:|---:|",
    ]
    for method in protocol.METHODS:
        row = payload["methods"][method]
        lines.append(
            f"| {method} | {row['switching_mean']:.1f} +/- "
            f"{row['switching_std_over_seeds']:.1f} | "
            f"{row['stationary_mean']:.1f} +/- "
            f"{row['stationary_std_over_seeds']:.1f} | "
            f"{row['switching_terminated_rate']:.3f} |")
    lines += [
        "",
        "## Registered comparisons",
        "",
        "| baseline | delta | relative | wins | Holm p | simultaneous lower | "
        "stationary retention | pass |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for baseline in BASELINES:
        row = payload["registered_comparisons"][baseline]
        lines.append(
            f"| {baseline} | {row['mean_delta']:+.1f} | "
            f"{row['relative_delta']:+.1%} | {row['seed_slot_wins']}/10 | "
            f"{row['holm_adjusted_p_value']:.4g} | "
            f"{row['bonferroni_simultaneous_95pct_lower_bound']:+.1f} | "
            f"{row['stationary_retention']:.1%} | "
            f"{row['registered_pass']} |")
    diagnostic = payload["strongest_seedwise_envelope_diagnostic"]
    lines += [
        "",
        "## Decision",
        "",
        f"- Overall registered pass: `{payload['primary_pass']}`.",
        f"- BAPR minus per-seed strongest envelope (diagnostic only): "
        f"{diagnostic['mean_delta']:+.1f}, "
        f"{diagnostic['seed_slot_wins']}/10 wins.",
        f"- Scope: {payload['claim_scope']}.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    payload = analyze()
    text = markdown(payload)
    protocol.ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    protocol.write_json_atomic(protocol.analysis_json(), payload)
    protocol.write_text_atomic(protocol.analysis_markdown(), text)
    protocol.write_text_atomic(protocol.RESULT_REPORT, text)
    print(
        f"DEPLOYMENT CONFIRMATION COMPLETE: pass={payload['primary_pass']}",
        flush=True)


if __name__ == "__main__":
    main()

