"""Analyze the independent source-controller oracle ladder."""
from __future__ import annotations

import statistics

from jax_experiments.analysis import regime_polarity_source_headroom_v1 as protocol
from jax_experiments.analysis.run_regime_polarity_source_headroom_audit_v1 import (
    ARMS,
    validate_audit,
)


def _mean(values) -> float:
    values = list(values)
    return float(statistics.fmean(values))


def _event(seed: int, event_seed: int) -> dict:
    return protocol.read_json(protocol.audit_event_result(seed, event_seed))


def _seed_metrics(seed: int) -> dict:
    validate_audit(seed)
    events = [_event(seed, event_seed) for event_seed in protocol.EVENT_SEEDS]
    stationary = {}
    switching = {}
    for arm in ARMS:
        by_mode = {
            mode: _mean(
                event["stationary"][arm][str(mode)]["return_mean"]
                for event in events)
            for mode in protocol.MODES
        }
        stationary[arm] = {
            "by_mode": {str(mode): value for mode, value in by_mode.items()},
            "mean": _mean(by_mode.values()),
            "worst": min(by_mode.values()),
            "termination_rate": _mean(
                event["stationary"][arm][str(mode)]["terminated_rate"]
                for event in events for mode in protocol.MODES),
        }
        switching[arm] = {
            "mean": _mean(
                event["switching"][arm]["return_mean"]
                for event in events),
            "termination_count": _mean(
                event["switching"][arm]["termination_count_mean"]
                for event in events),
        }

    matrix = {}
    diagonal_wins = 0
    for physics_mode in protocol.MODES:
        values = {
            specialist_mode: stationary[
                f"specialist_{specialist_mode}"]["by_mode"][str(physics_mode)]
            for specialist_mode in protocol.MODES
        }
        best = max(values, key=values.get)
        diagonal_wins += int(best == physics_mode)
        matrix[str(physics_mode)] = {
            "specialist_returns": {
                str(key): value for key, value in values.items()},
            "best_specialist": int(best),
            "diagonal": best == physics_mode,
        }

    comparators = protocol.ROLES
    strongest_stationary = max(
        comparators, key=lambda arm: stationary[arm]["mean"])
    strongest_worst = max(
        comparators, key=lambda arm: stationary[arm]["worst"])
    strongest_switching = max(
        comparators, key=lambda arm: switching[arm]["mean"])
    dynamic_stationary = stationary["dynamic_oracle"]["mean"]
    dynamic_worst = stationary["dynamic_oracle"]["worst"]
    dynamic_switching = switching["dynamic_oracle"]["mean"]
    stationary_gain = (
        dynamic_stationary - stationary[strongest_stationary]["mean"]
    ) / max(abs(stationary[strongest_stationary]["mean"]), 100.0)
    worst_gain = (
        dynamic_worst - stationary[strongest_worst]["worst"]
    ) / max(abs(stationary[strongest_worst]["worst"]), 100.0)
    switching_gain = (
        dynamic_switching - switching[strongest_switching]["mean"]
    ) / max(abs(switching[strongest_switching]["mean"]), 100.0)
    checks = {
        "dynamic_stationary_gain_at_least_10pct": (
            stationary_gain >= protocol.MIN_DYNAMIC_GAIN),
        "dynamic_worst_mode_gain_at_least_10pct": (
            worst_gain >= protocol.MIN_DYNAMIC_GAIN),
        "dynamic_switching_gain_at_least_10pct": (
            switching_gain >= protocol.MIN_DYNAMIC_GAIN),
        "dynamic_beats_every_fixed_specialist_switching": all(
            dynamic_switching > switching[f"specialist_{mode}"]["mean"]
            for mode in protocol.MODES),
        "diagonal_optimal_at_least_3_of_4": (
            diagonal_wins >= protocol.MIN_DIAGONAL_MODES),
        "dynamic_has_no_more_switching_terminations": (
            switching["dynamic_oracle"]["termination_count"]
            <= switching[strongest_switching]["termination_count"]),
    }
    return {
        "stationary": stationary,
        "switching": switching,
        "policy_by_mode_matrix": matrix,
        "diagonal_wins": diagonal_wins,
        "strongest_comparators": {
            "stationary_mean": strongest_stationary,
            "stationary_worst": strongest_worst,
            "switching": strongest_switching,
        },
        "relative_gains": {
            "stationary_mean": stationary_gain,
            "stationary_worst": worst_gain,
            "switching": switching_gain,
        },
        "gate_checks": checks,
        "seed_gate_pass": all(checks.values()),
    }


def analyze() -> dict:
    seeds = {
        str(seed): _seed_metrics(seed) for seed in protocol.TRAINING_SEEDS
    }
    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "training_seeds": list(protocol.TRAINING_SEEDS),
        "event_seeds": list(protocol.EVENT_SEEDS),
        "training_seed_metrics": seeds,
        "headroom_gate_pass": all(
            row["seed_gate_pass"] for row in seeds.values()),
        "next_step": (
            "train posterior-conditioned heads on frozen strong sources"
            if all(row["seed_gate_pass"] for row in seeds.values())
            else "stop estimator training; source-controller upper bound failed"),
    }


def report(payload: dict) -> str:
    lines = [
        "# Independent source-controller headroom",
        "",
        "This two-seed development screen trains robust SAC, ESCP, and four "
        "fully independent fixed-mode SAC specialists. The privileged dynamic "
        "oracle selects the matching specialist from the true physical mode. "
        "No learned estimator is trained in this stage.",
        "",
    ]
    for seed in protocol.TRAINING_SEEDS:
        row = payload["training_seed_metrics"][str(seed)]
        lines += [
            f"## Seed {seed}",
            "",
            "| Arm | Stationary mean | Worst mode | Switching |",
            "|---|---:|---:|---:|",
        ]
        for arm in ARMS:
            lines.append(
                f"| `{arm}` | {row['stationary'][arm]['mean']:.1f} | "
                f"{row['stationary'][arm]['worst']:.1f} | "
                f"{row['switching'][arm]['mean']:.1f} |")
        gains = row["relative_gains"]
        lines += [
            "",
            f"- Dynamic gains over strongest comparator: stationary "
            f"{100 * gains['stationary_mean']:+.1f}%, worst mode "
            f"{100 * gains['stationary_worst']:+.1f}%, switching "
            f"{100 * gains['switching']:+.1f}%.",
            f"- Diagonal specialist optima: {row['diagonal_wins']}/4.",
            f"- Seed gate: **{row['seed_gate_pass']}**.",
            "",
        ]
    lines += [
        f"Headroom gate pass: **{payload['headroom_gate_pass']}**",
        "",
        f"Decision: `{payload['next_step']}`.",
        "",
        "Only a pass on both fresh development seeds authorizes another BAPR "
        "controller/estimator training stage.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    payload = analyze()
    text = report(payload)
    protocol.write_json_atomic(protocol.analysis_json(), payload)
    protocol.write_text_atomic(protocol.analysis_markdown(), text)
    protocol.write_text_atomic(protocol.REPORT, text)
    print(
        "SOURCE HEADROOM ANALYSIS COMPLETE: "
        f"pass={payload['headroom_gate_pass']}",
        flush=True)


if __name__ == "__main__":
    main()
