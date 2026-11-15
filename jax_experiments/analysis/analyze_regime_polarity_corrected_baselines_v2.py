"""Aggregate frozen BAPR/SAC with corrected ESCP and RE-SAC baselines."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_corrected_baselines_v2 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_corrected_audit_v2 as auditor,
)


def _two_sample_interval(candidate, reference) -> list[float]:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if candidate.shape != reference.shape or candidate.shape != (5,):
        raise ValueError("corrected comparison requires two five-seed vectors")
    delta = float(np.mean(candidate) - np.mean(reference))
    standard_error = math.sqrt(
        float(np.var(candidate, ddof=1) / candidate.size)
        + float(np.var(reference, ddof=1) / reference.size))
    margin = protocol.frozen.CLUSTER_T_CRITICAL_95 * standard_error
    return [delta - margin, delta + margin]


def _seed_summary(payload: dict[str, Any]) -> dict[str, Any]:
    events = payload["events"]
    switching = [float(row["switching"]["return_mean"]) for row in events]
    stationary = [
        float(mode["return_mean"])
        for row in events for mode in row["stationary"]]
    by_mode = {
        str(mode): float(np.mean([
            event["stationary"][mode]["return_mean"]
            for event in events]))
        for mode in protocol.MODES
    }
    return {
        "switching_mean": float(np.mean(switching)),
        "stationary_mean": float(np.mean(stationary)),
        "stationary_by_mode": by_mode,
        "switching_terminated_rate": float(np.mean([
            row["switching"]["terminated_rate"] for row in events])),
        "stationary_terminated_rate": float(np.mean([
            mode["terminated_rate"]
            for row in events for mode in row["stationary"]])),
    }


def _comparison(candidate, reference) -> dict[str, Any]:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    deltas = candidate - reference
    return {
        "candidate_mean": float(np.mean(candidate)),
        "reference_mean": float(np.mean(reference)),
        "mean_delta": float(np.mean(deltas)),
        "relative_delta": float(
            np.mean(deltas) / max(abs(np.mean(reference)), 1e-8)),
        "conservative_two_sample_95pct_interval": _two_sample_interval(
            candidate, reference),
        "registered_seed_slot_deltas": {
            str(seed): float(value)
            for seed, value in zip(protocol.TRAINING_SEEDS, deltas)
        },
        "registered_seed_slot_wins": int(np.sum(deltas > 0.0)),
    }


def _payload(method: str, seed: int) -> dict[str, Any]:
    if method in protocol.REUSED_METHODS:
        path = protocol.validate_reused_result(method, seed)
    else:
        auditor.validate_manifest(method, seed)
        path = protocol.audit_dir(method, seed) / "results.json"
    return protocol.read_json(path)


def analyze() -> dict[str, Any]:
    registration = protocol.validate_registration()
    payloads = {
        method: {
            seed: _payload(method, seed)
            for seed in protocol.TRAINING_SEEDS
        }
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
            for method in protocol.METHODS
            for seed in protocol.TRAINING_SEEDS
        }
        if len(hashes) != 1:
            raise ValueError(
                f"corrected mode streams differ for event {event_seed}: "
                f"{hashes}")

    method_rows = {}
    vectors = {}
    for method in protocol.METHODS:
        switching = np.asarray([
            summaries[method][seed]["switching_mean"]
            for seed in protocol.TRAINING_SEEDS])
        stationary = np.asarray([
            summaries[method][seed]["stationary_mean"]
            for seed in protocol.TRAINING_SEEDS])
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
        "escp_recurrent_minus_sac": _comparison(
            vectors["escp_recurrent"]["switching"],
            vectors["sac"]["switching"]),
        "resac_b0_minus_sac": _comparison(
            vectors["resac_b0"]["switching"],
            vectors["sac"]["switching"]),
    }
    for baseline in ("sac", "escp_recurrent", "resac_b0"):
        comparisons[f"bapr_minus_{baseline}"] = _comparison(
            vectors["bapr"]["switching"],
            vectors[baseline]["switching"])

    strongest_switching = np.asarray([
        max(vectors[method]["switching"][index]
            for method in ("sac", "escp_recurrent", "resac_b0"))
        for index in range(len(protocol.TRAINING_SEEDS))])
    strongest_stationary = np.asarray([
        max(vectors[method]["stationary"][index]
            for method in ("sac", "escp_recurrent", "resac_b0"))
        for index in range(len(protocol.TRAINING_SEEDS))])
    primary = _comparison(
        vectors["bapr"]["switching"], strongest_switching)
    primary["stationary_retention"] = float(
        np.mean(vectors["bapr"]["stationary"])
        / max(np.mean(strongest_stationary), 1e-8))
    primary["pass"] = bool(
        primary["mean_delta"] > 0.0
        and primary["conservative_two_sample_95pct_interval"][0] > 0.0
        and primary["registered_seed_slot_wins"]
        >= protocol.frozen.MIN_SEED_WINS
        and primary["stationary_retention"]
        >= protocol.frozen.MIN_STATIONARY_RETENTION)
    comparisons["bapr_minus_strongest_corrected_baseline"] = primary

    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "registration": protocol.file_record(protocol.REGISTRATION_PATH),
        "training_seeds": list(protocol.TRAINING_SEEDS),
        "event_seeds": list(protocol.EVENT_SEEDS),
        "methods": method_rows,
        "comparisons": comparisons,
        "primary_pass": primary["pass"],
        "baseline_reproduction": {
            "escp_recurrent_beats_sac_switching_mean": bool(
                comparisons["escp_recurrent_minus_sac"]["mean_delta"] > 0.0),
            "resac_b0_beats_sac_switching_mean": bool(
                comparisons["resac_b0_minus_sac"]["mean_delta"] > 0.0),
        },
        "claim_scope": (
            "deployment-performance comparison on the frozen HalfCheetah "
            "actuator-polarity protocol; BAPR retains its frozen multi-"
            "controller training history, while all single-controller "
            "baselines use the same 5.6M-step/350k-update budget"
        ),
        "registered_source_count": len(registration["source_files"]),
    }


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Polarity BAPR corrected-baseline comparison",
        "",
        "This replaces the state-only ESCP approximation and the invalid "
        "legacy RE-SAC arm. BAPR and SAC are immutable SHA-256-locked audit "
        "results; recurrent ESCP and released-B0 RE-SAC are newly trained.",
        "",
        "| method | switching | stationary | switching termination |",
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
        "## Mode diagnostics",
        "",
        "| method | mode 0 | mode 1 | mode 2 | mode 3 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in protocol.METHODS:
        row = payload["methods"][method]["stationary_by_mode"]
        lines.append(
            f"| {method} | {row['0']:.1f} | {row['1']:.1f} | "
            f"{row['2']:.1f} | {row['3']:.1f} |")
    primary = payload["comparisons"][
        "bapr_minus_strongest_corrected_baseline"]
    interval = primary["conservative_two_sample_95pct_interval"]
    escp = payload["comparisons"]["escp_recurrent_minus_sac"]
    resac = payload["comparisons"]["resac_b0_minus_sac"]
    lines += [
        "",
        "## Registered decision",
        "",
        f"- Recurrent ESCP minus SAC switching: {escp['mean_delta']:+.1f} "
        f"({escp['registered_seed_slot_wins']}/5 seed-slot wins)",
        f"- Released-B0 RE-SAC minus SAC switching: "
        f"{resac['mean_delta']:+.1f} "
        f"({resac['registered_seed_slot_wins']}/5 seed-slot wins)",
        f"- BAPR minus strongest corrected baseline: "
        f"{primary['mean_delta']:+.1f}",
        f"- Conservative 95% interval: "
        f"[{interval[0]:+.1f}, {interval[1]:+.1f}]",
        f"- Registered seed-slot wins: "
        f"{primary['registered_seed_slot_wins']}/5",
        f"- Stationary retention: {primary['stationary_retention']:.1%}",
        f"- Primary pass: `{payload['primary_pass']}`",
        "",
        payload["claim_scope"],
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    payload = analyze()
    text = markdown(payload)
    protocol.ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    protocol.write_json_atomic(protocol.analysis_json(), payload)
    protocol.write_text_atomic(protocol.analysis_markdown(), text)
    protocol.write_text_atomic(protocol.REPORT, text)
    print(
        "CORRECTED COMPARISON COMPLETE: "
        f"primary_pass={payload['primary_pass']}", flush=True)


if __name__ == "__main__":
    main()
