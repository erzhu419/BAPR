"""Analyze the stationary controller-capacity diagnostic."""
from __future__ import annotations

import statistics

from jax_experiments.analysis import (
    regime_polarity_specialist_capacity_diagnostic_v7 as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_confirmation_v6 as source,
)
from jax_experiments.analysis.run_regime_polarity_specialist_capacity_diagnostic_v7 import (
    validate_audit,
)


def _mean(values) -> float:
    return float(statistics.fmean(values))


def _seed_metrics(seed: int) -> dict:
    validate_audit(seed)
    events = [
        protocol.read_json(protocol.event_result(seed, event_seed))
        for event_seed in protocol.EVENT_SEEDS
    ]
    matrix = {}
    for role in protocol.ROLES:
        matrix[role] = {}
        for mode in protocol.MODES:
            rows = [event["stationary"][role][str(mode)] for event in events]
            matrix[role][str(mode)] = {
                "mean": _mean(row["return_mean"] for row in rows),
                "terminated_rate": _mean(
                    row["terminated_rate"] for row in rows),
                "event_returns": {
                    str(event_seed): float(row["return_mean"])
                    for event_seed, row in zip(protocol.EVENT_SEEDS, rows)
                },
            }

    mode_rows = {}
    for mode in protocol.MODES:
        robust = matrix["robust_sac"][str(mode)]["mean"]
        specialist = matrix[f"specialist_{mode}"][str(mode)]["mean"]
        mode_rows[str(mode)] = {
            "robust": robust,
            "diagonal_specialist": specialist,
            "advantage": specialist - robust,
            "relative_gain": (
                (specialist - robust) / abs(robust)
                if robust != 0.0 else float("-inf")
            ),
            "specialist_wins": specialist > robust,
            "specialist_terminated_rate": matrix[
                f"specialist_{mode}"][str(mode)]["terminated_rate"],
        }
    robust_mean = _mean(row["robust"] for row in mode_rows.values())
    oracle_mean = _mean(
        row["diagonal_specialist"] for row in mode_rows.values())
    oracle_gain = (
        (oracle_mean - robust_mean) / abs(robust_mean)
        if robust_mean != 0.0 else float("-inf")
    )
    diagonal_wins = sum(row["specialist_wins"] for row in mode_rows.values())

    source_analysis = protocol.read_json(source.analysis_json())
    source_rows = source_analysis["source_seed_metrics"][str(seed)]["arms"]
    primary = source_rows[source.PRIMARY_ARM]
    switching = {
        "robust": float(source_rows["robust_sac"]["mean"]),
        "dynamic_oracle": float(source_rows["dynamic_oracle"]["mean"]),
        "frozen_v5": float(primary["mean"]),
        "frozen_v5_relative_gain": float(primary["relative_gain"]),
        "frozen_v5_oracle_recovery": primary["oracle_recovery"],
    }
    capacity_sufficient = bool(
        diagonal_wins >= protocol.MIN_DIAGONAL_WINS
        and oracle_gain >= protocol.MIN_STATIONARY_ORACLE_GAIN
        and max(
            matrix[f"specialist_{mode}"][str(mode)]["terminated_rate"]
            for mode in protocol.MODES
        ) == 0.0
    )
    if not capacity_sufficient:
        classification = "controller_capacity_failure"
    elif primary["relative_gain"] < source.MIN_GAIN or (
        primary["oracle_recovery"] is None
        or primary["oracle_recovery"] < source.MIN_ORACLE_RECOVERY
    ):
        classification = "switch_transient_failure"
    else:
        classification = "stationary_and_switching_capacity_sufficient"
    return {
        "matrix": matrix,
        "modes": mode_rows,
        "stationary_robust_mean": robust_mean,
        "stationary_oracle_mean": oracle_mean,
        "stationary_oracle_gain": oracle_gain,
        "diagonal_wins": diagonal_wins,
        "capacity_sufficient": capacity_sufficient,
        "switching": switching,
        "classification": classification,
    }


def analyze() -> dict:
    seeds = {
        str(seed): _seed_metrics(seed)
        for seed in protocol.TRAINING_SEEDS
    }
    capacity_failures = [
        seed for seed in protocol.TRAINING_SEEDS
        if seeds[str(seed)]["classification"] == "controller_capacity_failure"
    ]
    transient_failures = [
        seed for seed in protocol.TRAINING_SEEDS
        if seeds[str(seed)]["classification"] == "switch_transient_failure"
    ]
    if capacity_failures:
        conclusion = (
            "failed v6 banks include intrinsic specialist-controller capacity "
            "failures; estimator or router tuning cannot recover that headroom"
        )
        next_step = (
            "freeze each robust actor and train bounded mode options with an "
            "explicit per-mode no-regression objective before revisiting routing"
        )
    elif transient_failures:
        conclusion = (
            "stationary specialists are adequate and the remaining v6 loss is "
            "localized to mode-switch transients"
        )
        next_step = (
            "train only a switch-transient fallback policy using stale and soft "
            "belief rollouts; keep the frozen stationary controllers"
        )
    else:
        conclusion = "the failed v6 labels are not reproduced by stationary audit"
        next_step = "inspect the v6 absolute-return and event-win gates"
    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "training_seeds": list(protocol.TRAINING_SEEDS),
        "event_seeds": list(protocol.EVENT_SEEDS),
        "thresholds": {
            "minimum_diagonal_wins": protocol.MIN_DIAGONAL_WINS,
            "minimum_stationary_oracle_gain": (
                protocol.MIN_STATIONARY_ORACLE_GAIN
            ),
        },
        "source_seed_metrics": seeds,
        "controller_capacity_failure_seeds": capacity_failures,
        "switch_transient_failure_seeds": transient_failures,
        "conclusion": conclusion,
        "next_step": next_step,
    }


def report(payload: dict) -> str:
    lines = [
        "# Stationary controller-capacity diagnostic for failed v6 banks",
        "",
        "This checkpoint-only diagnostic cross-evaluates the robust controller "
        "and all four fixed-mode specialists from failed v6 policy banks on "
        "three fresh stationary event streams. It does not refit the estimator, "
        "router, or controller.",
        "",
        "A bank has sufficient stationary adaptation capacity only when the "
        "diagonal specialist beats robust in at least 3/4 modes, the mean "
        "diagonal oracle gain is at least 10%, and diagonal termination is zero.",
        "",
        "| Seed | Stationary robust | Stationary oracle | Oracle gain | "
        "Diagonal wins | v6 switching gain | Classification |",
        "|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for seed in protocol.TRAINING_SEEDS:
        row = payload["source_seed_metrics"][str(seed)]
        lines.append(
            f"| {seed} | {row['stationary_robust_mean']:.1f} | "
            f"{row['stationary_oracle_mean']:.1f} | "
            f"{100*row['stationary_oracle_gain']:.1f}% | "
            f"{row['diagonal_wins']}/{len(protocol.MODES)} | "
            f"{100*row['switching']['frozen_v5_relative_gain']:.1f}% | "
            f"{row['classification']} |"
        )
    lines += [
        "",
        "## Per-mode diagonal comparison",
        "",
        "| Seed | Mode | Robust | Specialist | Gain | Terminated |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in protocol.TRAINING_SEEDS:
        for mode in protocol.MODES:
            row = payload["source_seed_metrics"][str(seed)]["modes"][str(mode)]
            lines.append(
                f"| {seed} | {mode} | {row['robust']:.1f} | "
                f"{row['diagonal_specialist']:.1f} | "
                f"{100*row['relative_gain']:.1f}% | "
                f"{100*row['specialist_terminated_rate']:.1f}% |"
            )
    lines += [
        "",
        f"Controller-capacity failures: `{payload['controller_capacity_failure_seeds']}`.",
        f"Switch-transient failures: `{payload['switch_transient_failure_seeds']}`.",
        "",
        f"Conclusion: {payload['conclusion']}.",
        "",
        f"Decision: {payload['next_step']}.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    payload = analyze()
    text = report(payload)
    protocol.write_json_atomic(protocol.analysis_json(), payload)
    protocol.write_text_atomic(protocol.analysis_markdown(), text)
    protocol.write_text_atomic(protocol.REPORT, text)
    print(
        "STATIONARY CONTROLLER-CAPACITY ANALYSIS COMPLETE: "
        f"capacity_failures={payload['controller_capacity_failure_seeds']} "
        f"transient_failures={payload['switch_transient_failure_seeds']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
