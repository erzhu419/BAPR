"""Aggregate BAPR-v8 over independent policy-training seeds."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from jax_experiments.analysis import bapr_v8_seed_validation as protocol


T_CRITICAL_95_DF4 = 2.7764451051977987
BASELINES = ("sac", "escp", "resac")
EXPECTED_AUDIT_EPISODES = 5


def validate_audit_payload(
    payload: dict[str, Any], seed: int, event_seed: int,
) -> None:
    """Validate immutable audit contents without reopening model bundles."""
    if (payload.get("schema") != protocol.AUDIT_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("training_seed") != seed
            or payload.get("event_seed") != event_seed
            or payload.get("family") != protocol.FAMILY
            or payload.get("env") != protocol.ENV
            or payload.get("decision_variant") != protocol.DECISION_VARIANT
            or set(payload.get("stationary") or {})
            != set(protocol.CONTROLLERS)
            or set(payload.get("switching") or {})
            != {"slow_pair", "full_cycle"}):
        raise ValueError("invalid BAPR-v8 audit identity")
    for controller in protocol.CONTROLLERS:
        stationary = payload["stationary"][controller]
        if set(stationary) != set(map(str, protocol.MODES)):
            raise ValueError("incomplete BAPR-v8 stationary audit")
        for record in stationary.values():
            returns = record.get("returns") or []
            if (len(returns) != EXPECTED_AUDIT_EPISODES
                    or not all(math.isfinite(float(value))
                               for value in returns)):
                raise ValueError("invalid BAPR-v8 stationary returns")
        for kind in ("slow_pair", "full_cycle"):
            record = payload["switching"][kind][controller]
            if (len(record.get("episodes") or [])
                    != EXPECTED_AUDIT_EPISODES
                    or not math.isfinite(float(record.get("mean")))):
                raise ValueError("invalid BAPR-v8 switching returns")


def _ci95(values) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (len(protocol.TRAINING_SEEDS),):
        raise ValueError("primary CI requires five independent training seeds")
    mean = float(np.mean(array))
    sem = float(np.std(array, ddof=1) / math.sqrt(len(array)))
    half = T_CRITICAL_95_DF4 * sem
    return mean - half, mean + half


def _controller_metrics(group: dict[str, Any], controller: str):
    stationary_rows = list(group["stationary"][controller].values())
    return {
        "stationary": float(np.mean([
            row["mean"] for row in stationary_rows])),
        "stationary_termination": float(np.mean([
            row["terminated_rate"] for row in stationary_rows])),
        "slow_pair": float(
            group["switching"]["slow_pair"][controller]["mean"]),
        "slow_pair_termination": float(
            group["switching"]["slow_pair"][controller][
                "termination_rate"]),
        "full_cycle": float(
            group["switching"]["full_cycle"][controller]["mean"]),
        "full_cycle_termination": float(
            group["switching"]["full_cycle"][controller][
                "termination_rate"]),
    }


def _seed_summary(seed: int) -> dict[str, Any]:
    groups = []
    for event_seed in protocol.EVALUATION_EVENT_SEEDS:
        path = protocol.audit_path(seed, event_seed)
        payload = protocol.read_json(path)
        validate_audit_payload(payload, seed, event_seed)
        groups.append(payload)
    controllers = {}
    for controller in protocol.CONTROLLERS:
        rows = [_controller_metrics(group, controller) for group in groups]
        controllers[controller] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
        }
    strongest = max(
        BASELINES,
        key=lambda name: controllers[name]["full_cycle"])
    bapr = controllers["bapr"]
    oracle = controllers["oracle"]
    sac = controllers["sac"]
    denominator = oracle["full_cycle"] - sac["full_cycle"]
    recovery = (
        (bapr["full_cycle"] - sac["full_cycle"]) / denominator
        if denominator > 0.0 else None)
    return {
        "training_seed": seed,
        "controllers": controllers,
        "strongest_baseline": strongest,
        "bapr_minus_strongest_full_cycle": (
            bapr["full_cycle"] - controllers[strongest]["full_cycle"]),
        "bapr_minus_strongest_stationary": (
            bapr["stationary"] - controllers[strongest]["stationary"]),
        "bapr_minus_strongest_termination": (
            bapr["full_cycle_termination"]
            - controllers[strongest]["full_cycle_termination"]),
        "oracle_recovery_from_sac": recovery,
    }


def analyze() -> dict[str, Any]:
    seeds = [_seed_summary(seed) for seed in protocol.TRAINING_SEEDS]
    aggregate = {}
    for controller in protocol.CONTROLLERS:
        aggregate[controller] = {}
        for metric in (
                "stationary", "slow_pair", "full_cycle",
                "stationary_termination", "slow_pair_termination",
                "full_cycle_termination"):
            values = [row["controllers"][controller][metric] for row in seeds]
            aggregate[controller][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "values": values,
            }

    differences = {
        baseline: [
            row["controllers"]["bapr"]["full_cycle"]
            - row["controllers"][baseline]["full_cycle"]
            for row in seeds
        ]
        for baseline in BASELINES
    }
    paired = {
        baseline: {
            "mean": float(np.mean(values)),
            "ci95": list(_ci95(values)),
            "wins": int(np.sum(np.asarray(values) > 0.0)),
            "values": values,
        }
        for baseline, values in differences.items()
    }
    strongest_differences = [
        row["bapr_minus_strongest_full_cycle"] for row in seeds]
    stationary_differences = [
        row["bapr_minus_strongest_stationary"] for row in seeds]
    termination_differences = [
        row["bapr_minus_strongest_termination"] for row in seeds]
    strongest_ci = _ci95(strongest_differences)
    gate = {
        "positive_full_cycle_ci_vs_strongest": strongest_ci[0] > 0.0,
        "at_least_four_of_five_full_cycle_wins": int(np.sum(
            np.asarray(strongest_differences) > 0.0)) >= 4,
        "stationary_noninferiority_100": (
            float(np.mean(stationary_differences)) >= -100.0),
        "termination_noninferiority_0p05": (
            float(np.mean(termination_differences)) <= 0.05),
    }
    gate["pass"] = all(gate.values())
    payload = {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "family": protocol.FAMILY,
        "env": protocol.ENV,
        "training_seeds": list(protocol.TRAINING_SEEDS),
        "evaluation_event_seeds": list(protocol.EVALUATION_EVENT_SEEDS),
        "independent_unit": "policy training seed",
        "seed_summaries": seeds,
        "aggregate": aggregate,
        "paired_bapr_minus_baseline_full_cycle": paired,
        "bapr_minus_per_seed_strongest": {
            "mean": float(np.mean(strongest_differences)),
            "ci95": list(strongest_ci),
            "wins": int(np.sum(np.asarray(strongest_differences) > 0.0)),
            "values": strongest_differences,
        },
        "promotion_gate": gate,
    }
    protocol.write_json_atomic(protocol.analysis_path(), payload)
    write_report(payload)
    return payload


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.1f} +/- {std:.1f}"


def write_report(payload: dict[str, Any]) -> None:
    aggregate = payload["aggregate"]
    lines = [
        "# BAPR-v8 independent-training-seed validation",
        "",
        "Promotion gate: **{}**".format(
            "PASS" if payload["promotion_gate"]["pass"] else "FAIL"),
        "",
        "The independent statistical unit is the policy training seed. Each "
        "seed is evaluated on the same five paired, previously sealed event "
        "streams.",
        "",
        "| Controller | Stationary | Slow switching | Full cycle | Full-cycle termination |",
        "|---|---:|---:|---:|---:|",
    ]
    for controller in protocol.CONTROLLERS:
        row = aggregate[controller]
        lines.append(
            "| {} | {} | {} | {} | {:.3f} +/- {:.3f} |".format(
                controller,
                _fmt(row["stationary"]["mean"], row["stationary"]["std"]),
                _fmt(row["slow_pair"]["mean"], row["slow_pair"]["std"]),
                _fmt(row["full_cycle"]["mean"], row["full_cycle"]["std"]),
                row["full_cycle_termination"]["mean"],
                row["full_cycle_termination"]["std"],
            ))
    lines += [
        "",
        "| Paired full-cycle contrast | Mean | 95% CI | Wins |",
        "|---|---:|---:|---:|",
    ]
    for baseline in BASELINES:
        row = payload["paired_bapr_minus_baseline_full_cycle"][baseline]
        lines.append(
            f"| BAPR - {baseline} | {row['mean']:+.1f} | "
            f"[{row['ci95'][0]:+.1f}, {row['ci95'][1]:+.1f}] | "
            f"{row['wins']}/5 |")
    strongest = payload["bapr_minus_per_seed_strongest"]
    lines += [
        "",
        "Against the strongest baseline selected independently within each "
        f"training seed, BAPR is {strongest['mean']:+.1f} with 95% CI "
        f"[{strongest['ci95'][0]:+.1f}, {strongest['ci95'][1]:+.1f}] and "
        f"wins {strongest['wins']}/5 seeds.",
        "",
        "Gate details: `" + json_compact(payload["promotion_gate"]) + "`",
        "",
    ]
    path = protocol.report_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def json_compact(value: dict[str, Any]) -> str:
    return ", ".join(f"{key}={str(item).lower()}"
                     for key, item in value.items())


def main() -> None:
    payload = analyze()
    print(
        "BAPR V8 SEED VALIDATION COMPLETE: "
        f"gate={'PASS' if payload['promotion_gate']['pass'] else 'FAIL'} "
        f"output={protocol.analysis_path()}", flush=True)


if __name__ == "__main__":
    main()
