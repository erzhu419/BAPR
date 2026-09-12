"""Aggregate the unseen five-seed causal-posterior screen."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from jax_experiments.analysis import regime_polarity_posterior as protocol
from jax_experiments.analysis import (
    run_regime_polarity_posterior_audit as audit,
)


T_CRITICAL_95_DF4 = 2.7764451051977987


def _paired(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1))
    half = T_CRITICAL_95_DF4 * sd / math.sqrt(len(array))
    return {
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "wins": int(np.sum(array > 0.0)),
        "n_training_seeds": len(array),
    }


def _mean(rows, key):
    return float(np.mean([float(row[key]) for row in rows]))


def _seed_summary(seed: int) -> dict[str, Any]:
    audit.validate_audit(seed)
    events = [
        protocol.read_json(
            protocol.audit_dir(seed)
            / f"event_seed_{event_seed}/results.json")
        for event_seed in protocol.TEST_EVENT_SEEDS
    ]
    output: dict[str, Any] = {
        "stationary": {},
        "switching": {},
        "posterior": {},
    }
    for arm in audit.ARMS:
        mode_values = {}
        mode_termination = {}
        for mode in protocol.MODES:
            rows = [
                row
                for event in events
                for row in event["stationary"]
                if row["arm"] == arm and int(row["mode"]) == mode
            ]
            mode_values[str(mode)] = _mean(rows, "return_mean")
            mode_termination[str(mode)] = _mean(
                rows, "terminated_rate")
        output["stationary"][arm] = {
            "by_mode": mode_values,
            "termination_by_mode": mode_termination,
            "mean": float(np.mean(list(mode_values.values()))),
            "worst": float(np.min(list(mode_values.values()))),
            "termination_mean": float(np.mean(
                list(mode_termination.values()))),
        }
        switching_rows = [
            row
            for event in events
            for row in event["switching"]
            if row["arm"] == arm
        ]
        output["switching"][arm] = {
            "mean": _mean(switching_rows, "return_mean"),
            "termination_mean": _mean(
                switching_rows, "terminated_rate"),
        }
        if arm in audit.LEARNED_ARMS:
            metric_rows = [
                row["posterior_metrics"] for row in switching_rows
            ]
            delays = [
                int(delay)
                for row in metric_rows
                for delay in row["switch_delays"]
            ]
            output["posterior"][arm] = {
                name: _mean(metric_rows, name)
                for name in (
                    "mode_accuracy",
                    "mean_true_probability",
                    "negative_log_likelihood",
                    "brier_score",
                    "expected_calibration_error",
                    "mean_normalized_entropy",
                )
            }
            output["posterior"][arm].update({
                "switch_count": len(delays),
                "median_switch_delay": float(np.median(delays)),
                "p90_switch_delay": float(np.percentile(delays, 90)),
                "aleatoric_mean": _mean([
                    row["uncertainty"] for row in switching_rows
                ], "aleatoric_mean"),
                "epistemic_mean": _mean([
                    row["uncertainty"] for row in switching_rows
                ], "epistemic_mean"),
            })
    return output


def _arm_summary(
    by_seed: dict[str, dict[str, Any]],
    section: str,
    arm: str,
    metric: str,
) -> dict[str, float]:
    values = [
        float(by_seed[str(seed)][section][arm][metric])
        for seed in protocol.TEST_CONTROLLER_SEEDS
    ]
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)),
    }


def _main() -> None:
    by_seed = {
        str(seed): _seed_summary(seed)
        for seed in protocol.TEST_CONTROLLER_SEEDS
    }
    switching_summary = {
        arm: _arm_summary(by_seed, "switching", arm, "mean")
        for arm in audit.ARMS
    }
    stationary_summary = {
        arm: {
            "mean": _arm_summary(
                by_seed, "stationary", arm, "mean"),
            "worst": _arm_summary(
                by_seed, "stationary", arm, "worst"),
        }
        for arm in audit.ARMS
    }

    paired_switching = {}
    paired_stationary = {}
    recovery = {}
    for arm in audit.LEARNED_ARMS:
        switching_delta = []
        stationary_delta = []
        seed_recovery: list[float | None] = []
        nonpositive_headroom_seeds = []
        for seed in protocol.TEST_CONTROLLER_SEEDS:
            row = by_seed[str(seed)]
            robust_switch = row["switching"]["robust"]["mean"]
            oracle_switch = row["switching"]["oracle"]["mean"]
            arm_switch = row["switching"][arm]["mean"]
            switching_delta.append(arm_switch - robust_switch)
            stationary_delta.append(
                row["stationary"][arm]["mean"]
                - row["stationary"]["robust"]["mean"])
            headroom = oracle_switch - robust_switch
            if headroom > 1e-8:
                seed_recovery.append(
                    (arm_switch - robust_switch) / headroom)
            else:
                seed_recovery.append(None)
                nonpositive_headroom_seeds.append(int(seed))
        paired_switching[arm] = _paired(switching_delta)
        paired_stationary[arm] = _paired(stationary_delta)
        eligible_recovery = [
            value for value in seed_recovery if value is not None]
        all_recovery_defined = not nonpositive_headroom_seeds
        recovery[arm] = {
            # A seed-level ratio is undefined when the oracle itself does not
            # beat robust. Keep the preregistered gate conservative instead
            # of silently dropping that seed from the primary mean.
            "mean": (
                float(np.mean(eligible_recovery))
                if all_recovery_defined else None
            ),
            "minimum": (
                float(np.min(eligible_recovery))
                if all_recovery_defined else None
            ),
            "eligible_mean": (
                float(np.mean(eligible_recovery))
                if eligible_recovery else None
            ),
            "eligible_minimum": (
                float(np.min(eligible_recovery))
                if eligible_recovery else None
            ),
            "eligible_seed_count": len(eligible_recovery),
            "nonpositive_oracle_headroom_seeds": (
                nonpositive_headroom_seeds),
            "by_seed": {
                str(seed): (
                    float(value) if value is not None else None)
                for seed, value in zip(
                    protocol.TEST_CONTROLLER_SEEDS,
                    seed_recovery,
                )
            },
        }

    posterior_summary = {}
    for arm in audit.LEARNED_ARMS:
        rows = [
            by_seed[str(seed)]["posterior"][arm]
            for seed in protocol.TEST_CONTROLLER_SEEDS
        ]
        delays = []
        for seed in protocol.TEST_CONTROLLER_SEEDS:
            for event_seed in protocol.TEST_EVENT_SEEDS:
                event = protocol.read_json(
                    protocol.audit_dir(seed)
                    / f"event_seed_{event_seed}/results.json")
                switching = next(
                    row for row in event["switching"]
                    if row["arm"] == arm)
                delays.extend(
                    switching["posterior_metrics"]["switch_delays"])
        posterior_summary[arm] = {
            name: _mean(rows, name)
            for name in (
                "mode_accuracy",
                "mean_true_probability",
                "negative_log_likelihood",
                "brier_score",
                "expected_calibration_error",
                "mean_normalized_entropy",
                "aleatoric_mean",
                "epistemic_mean",
            )
        }
        posterior_summary[arm].update({
            "switch_count": len(delays),
            "median_switch_delay": float(np.median(delays)),
            "p90_switch_delay": float(np.percentile(delays, 90)),
        })

    soft = posterior_summary["learned_soft"]
    inference_gate = {
        "mode_accuracy_at_least_threshold": (
            soft["mode_accuracy"] >= protocol.MIN_MODE_ACCURACY),
        "median_switch_delay_at_most_threshold": (
            soft["median_switch_delay"]
            <= protocol.MAX_MEDIAN_SWITCH_DELAY),
        "p90_switch_delay_at_most_threshold": (
            soft["p90_switch_delay"]
            <= protocol.MAX_P90_SWITCH_DELAY),
        "brier_score_at_most_threshold": (
            soft["brier_score"] <= protocol.MAX_BRIER_SCORE),
    }
    inference_gate_pass = all(inference_gate.values())

    deployment_gates = {}
    for arm in audit.LEARNED_ARMS:
        termination_gap = float(np.mean([
            by_seed[str(seed)]["switching"][arm]["termination_mean"]
            - by_seed[str(seed)]["switching"]["robust"]["termination_mean"]
            for seed in protocol.TEST_CONTROLLER_SEEDS
        ]))
        mean_recovery = recovery[arm]["mean"]
        checks = {
            "headroom_recovery_at_least_threshold": (
                mean_recovery is not None
                and mean_recovery >= protocol.MIN_HEADROOM_RECOVERY),
            "policy_seed_wins_at_least_threshold": (
                paired_switching[arm]["wins"]
                >= protocol.MIN_POLICY_SEED_WINS),
            "termination_gap_at_most_threshold": (
                termination_gap <= protocol.MAX_TERMINATION_GAP),
        }
        deployment_gates[arm] = {
            "checks": checks,
            "pass": all(checks.values()),
            "termination_gap": termination_gap,
        }

    control_feasibility_pass = any(
        row["pass"] for row in deployment_gates.values())
    training_authorized = inference_gate_pass and control_feasibility_pass
    if not inference_gate_pass:
        decision = "estimator_failed"
    elif not control_feasibility_pass:
        decision = "estimator_passed_but_frozen_control_failed"
    elif deployment_gates["learned_soft"]["pass"]:
        decision = "direct_posterior_policy_training_authorized"
    else:
        decision = "soft_belief_policy_training_authorized_by_map_control"

    payload = {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "env": protocol.ENV,
        "family": protocol.FAMILY,
        "training_controller_seeds": list(
            protocol.TRAIN_CONTROLLER_SEEDS),
        "validation_controller_seeds": list(
            protocol.VALIDATION_CONTROLLER_SEEDS),
        "test_controller_seeds": list(
            protocol.TEST_CONTROLLER_SEEDS),
        "test_event_seeds": list(protocol.TEST_EVENT_SEEDS),
        "by_training_seed": by_seed,
        "switching_summary": switching_summary,
        "stationary_summary": stationary_summary,
        "paired_switching_vs_robust": paired_switching,
        "paired_stationary_vs_robust": paired_stationary,
        "headroom_recovery": recovery,
        "posterior_summary": posterior_summary,
        "inference_gate": inference_gate,
        "inference_gate_pass": inference_gate_pass,
        "deployment_gates": deployment_gates,
        "control_feasibility_pass": control_feasibility_pass,
        "posterior_conditioned_training_authorized": training_authorized,
        "decision": decision,
        "model_manifest": protocol.file_record(protocol.MODEL_MANIFEST),
        "audit_manifests": {
            str(seed): protocol.file_record(
                protocol.audit_manifest(seed))
            for seed in protocol.TEST_CONTROLLER_SEEDS
        },
    }
    protocol.write_json_atomic(protocol.analysis_json(), payload)

    lines = [
        "# Polarity posterior screen",
        "",
        f"Decision: **{decision}**.",
        "",
        "| Arm | Switching mean | Delta vs robust | Seed wins | "
        "Oracle headroom recovered |",
        "|---|---:|---:|---:|---:|",
    ]
    robust_mean = switching_summary["robust"]["mean"]
    for arm in audit.ARMS:
        mean = switching_summary[arm]["mean"]
        if arm in audit.LEARNED_ARMS:
            delta = paired_switching[arm]["mean"]
            wins = paired_switching[arm]["wins"]
            recovered = recovery[arm]["mean"]
            recovered_text = (
                f"{100.0 * recovered:.1f}%"
                if recovered is not None
                else "undefined"
            )
            lines.append(
                f"| {arm} | {mean:.1f} | {delta:+.1f} | {wins}/5 | "
                f"{recovered_text} |")
        else:
            lines.append(
                f"| {arm} | {mean:.1f} | {mean - robust_mean:+.1f} | "
                "- | - |")
    undefined_recovery_seeds = sorted({
        seed
        for arm in audit.LEARNED_ARMS
        for seed in recovery[arm]["nonpositive_oracle_headroom_seeds"]
    })
    if undefined_recovery_seeds:
        lines.extend([
            "",
            "Oracle-headroom recovery is undefined for the full sealed split "
            "because oracle does not beat robust on seed(s) "
            + ", ".join(str(seed) for seed in undefined_recovery_seeds)
            + ". These seeds remain in every paired control gate.",
        ])
    lines.extend([
        "",
        "## Causal inference",
        "",
        "| Metric | Learned soft | Gate |",
        "|---|---:|---:|",
        f"| Mode accuracy | {soft['mode_accuracy']:.3f} | "
        f">= {protocol.MIN_MODE_ACCURACY:.2f} |",
        f"| Median switch delay | {soft['median_switch_delay']:.1f} | "
        f"<= {protocol.MAX_MEDIAN_SWITCH_DELAY:.0f} |",
        f"| P90 switch delay | {soft['p90_switch_delay']:.1f} | "
        f"<= {protocol.MAX_P90_SWITCH_DELAY:.0f} |",
        f"| Brier score | {soft['brier_score']:.3f} | "
        f"<= {protocol.MAX_BRIER_SCORE:.2f} |",
        "",
        "Inference gate: "
        f"**{'PASS' if inference_gate_pass else 'FAIL'}**. "
        "Posterior-conditioned SAC training: "
        f"**{'AUTHORIZED' if training_authorized else 'BLOCKED'}**.",
        "",
    ])
    protocol.write_text_atomic(
        protocol.analysis_markdown(), "\n".join(lines))
    print(
        "POLARITY POSTERIOR ANALYSIS COMPLETE: "
        f"decision={decision} output={protocol.ANALYSIS_ROOT}",
        flush=True,
    )


def run_analysis(protocol_module, audit_module) -> None:
    """Run the shared aggregate logic against one sealed protocol revision."""
    global protocol, audit
    previous_protocol = protocol
    previous_audit = audit
    protocol = protocol_module
    audit = audit_module
    try:
        _main()
    finally:
        protocol = previous_protocol
        audit = previous_audit


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
