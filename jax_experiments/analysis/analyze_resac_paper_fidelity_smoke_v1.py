"""Aggregate the preregistered paper-fidelity JAX smoke."""
from __future__ import annotations

import csv
import math
import statistics

from jax_experiments.analysis import resac_paper_fidelity_smoke_v1 as protocol
from jax_experiments.analysis.run_resac_paper_fidelity_audit_v1 import (
    validate_audit,
)


def _rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(values) -> float:
    values = [float(value) for value in values]
    if not values or not all(math.isfinite(value) for value in values):
        return math.nan
    return float(statistics.fmean(values))


def _sd(values) -> float:
    values = [float(value) for value in values]
    if not values or not all(math.isfinite(value) for value in values):
        return math.nan
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _event_metrics(env: str, role: str, seed: int, event_seed: int) -> dict:
    rows = _rows(
        protocol.audit_event_dir(env, role, seed, event_seed) / "summary.csv")
    stationary_rows = [
        row for row in rows
        if row["metric_group"] == "stationary" and row["split"] == "test"
    ]
    switching_rows = [
        row for row in rows if row["metric_group"] == "switching"
    ]
    if len(stationary_rows) != 1 or len(switching_rows) != 1:
        raise ValueError("fidelity audit summary has ambiguous rows")
    stationary = stationary_rows[0]
    switching = switching_rows[0]
    return {
        "stationary_ood": float(stationary["return_mean"]),
        "stationary_termination": float(
            stationary["terminated_rate_mean"]),
        "switching": float(switching["switch_return_mean"]),
        "switching_termination_count": float(
            switching["termination_count_mean"]),
    }


def _seed_metrics(env: str, role: str, seed: int) -> dict:
    events = {
        event_seed: _event_metrics(env, role, seed, event_seed)
        for event_seed in protocol.EVENT_SEEDS
    }
    return {
        "event_metrics": {str(key): value for key, value in events.items()},
        "stationary_ood": _mean(
            value["stationary_ood"] for value in events.values()),
        "stationary_termination": _mean(
            value["stationary_termination"] for value in events.values()),
        "switching": _mean(
            value["switching"] for value in events.values()),
        "switching_termination_count": _mean(
            value["switching_termination_count"]
            for value in events.values()),
    }


def _role_summary(seed_rows: dict[int, dict]) -> dict:
    output = {}
    for key in (
        "stationary_ood",
        "stationary_termination",
        "switching",
        "switching_termination_count",
    ):
        values = [float(seed_rows[seed][key]) for seed in protocol.TRAINING_SEEDS]
        output[key] = {"mean": _mean(values), "sd": _sd(values)}
    return output


def _analyze_env(env: str) -> dict:
    by_seed = {}
    for seed in protocol.TRAINING_SEEDS:
        by_seed[seed] = {}
        for role in protocol.ROLES:
            validate_audit(env, role, seed)
            by_seed[seed][role] = _seed_metrics(env, role, seed)
    summaries = {
        role: _role_summary({
            seed: by_seed[seed][role] for seed in protocol.TRAINING_SEEDS
        })
        for role in protocol.ROLES
    }
    deltas = {}
    for adaptive in ("escp", "resac"):
        role_deltas = {}
        for metric in ("stationary_ood", "switching"):
            values = [
                float(by_seed[seed][adaptive][metric])
                - float(by_seed[seed]["sac"][metric])
                for seed in protocol.TRAINING_SEEDS
            ]
            role_deltas[metric] = {
                "per_seed": dict(zip(protocol.TRAINING_SEEDS, values)),
                "mean": _mean(values),
                "wins": sum(value > 0.0 for value in values),
            }
        deltas[f"{adaptive}_minus_sac"] = role_deltas
    resac_wins = sum(
        deltas["resac_minus_sac"][metric]["wins"]
        for metric in ("stationary_ood", "switching"))
    escp_wins = sum(
        deltas["escp_minus_sac"][metric]["wins"]
        for metric in ("stationary_ood", "switching"))
    checks = {
        "resac_mean_beats_sac_stationary_ood": (
            deltas["resac_minus_sac"]["stationary_ood"]["mean"] > 0.0),
        "resac_mean_beats_sac_switching": (
            deltas["resac_minus_sac"]["switching"]["mean"] > 0.0),
        "resac_wins_at_least_3_of_4_seed_metrics": resac_wins >= 3,
        "escp_mean_beats_sac_stationary_ood": (
            deltas["escp_minus_sac"]["stationary_ood"]["mean"] > 0.0),
        "escp_mean_beats_sac_switching": (
            deltas["escp_minus_sac"]["switching"]["mean"] > 0.0),
        "escp_wins_at_least_3_of_4_seed_metrics": escp_wins >= 3,
    }
    return {
        "training_seed_metrics": {
            str(seed): by_seed[seed] for seed in protocol.TRAINING_SEEDS},
        "summaries": summaries,
        "paired_deltas": deltas,
        "gate_checks": checks,
        "paper_relation_recovered": all(checks.values()),
    }


def analyze() -> dict:
    environments = {env: _analyze_env(env) for env in protocol.ENVS}
    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "training_seeds": list(protocol.TRAINING_SEEDS),
        "event_seeds": list(protocol.EVENT_SEEDS),
        "inference_scope": (
            "two-seed diagnostic only; not a confirmatory paper result"),
        "environments": environments,
        "global_smoke_pass": all(
            row["paper_relation_recovered"]
            for row in environments.values()),
    }


def report(payload: dict) -> str:
    lines = [
        "# RE-SAC paper-fidelity JAX smoke",
        "",
        "This is a two-policy-seed diagnostic. It aligns the continuous "
        "gravity generator and the 10k random plus 1000 x (1k rollout, 1k "
        "update) budget. It does not claim an exact reproduction of the "
        "original PyTorch ESCP architecture.",
        "",
    ]
    for env in protocol.ENVS:
        row = payload["environments"][env]
        lines += [
            f"## {env}",
            "",
            "| Method | Stationary OOD | Switching |",
            "|---|---:|---:|",
        ]
        for role in protocol.ROLES:
            summary = row["summaries"][role]
            lines.append(
                f"| {role.upper()} | "
                f"{summary['stationary_ood']['mean']:.1f} +/- "
                f"{summary['stationary_ood']['sd']:.1f} | "
                f"{summary['switching']['mean']:.1f} +/- "
                f"{summary['switching']['sd']:.1f} |")
        lines += [
            "",
            f"Paper ordering recovered: **{row['paper_relation_recovered']}**",
            "",
        ]
        for name, metrics in row["paired_deltas"].items():
            lines.append(
                f"- `{name}`: stationary "
                f"{metrics['stationary_ood']['mean']:+.1f}, switching "
                f"{metrics['switching']['mean']:+.1f}.")
        lines.append("")
    lines += [
        f"Global smoke pass: **{payload['global_smoke_pass']}**",
        "",
        "A failed smoke blocks using the current JAX RE-SAC result as a "
        "scientific baseline. A pass authorizes a fresh five-seed run; these "
        "two seeds are not pooled into that confirmation.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    payload = analyze()
    text = report(payload)
    protocol.write_json_atomic(protocol.analysis_json(), payload)
    protocol.write_text_atomic(protocol.analysis_markdown(), text)
    protocol.write_text_atomic(protocol.REPORT, text)
    print(
        "RE-SAC FIDELITY SMOKE COMPLETE: "
        f"pass={payload['global_smoke_pass']}",
        flush=True)


if __name__ == "__main__":
    main()
