"""Aggregate the preregistered V31 fresh-policy confirmation."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_action_compensation_confirmation_v31 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_action_compensation_confirmation_audit_v31 as audit,
)


T_CRITICAL_95_DF4 = 2.7764451051977987


def _mean(values) -> float:
    return float(np.mean([float(value) for value in values]))


def _switching_seed_mean(payload: dict[str, Any], arm: str) -> float:
    return _mean(
        payload["switching_holdout"][str(event_seed)][arm]["return_mean"]
        for event_seed in protocol.SWITCHING_EVENT_SEEDS
    )


def _stationary_seed_mean(payload: dict[str, Any], arm: str) -> float:
    return _mean(
        payload["stationary_holdout"][str(event_seed)][arm][str(mode)][
            "return_mean"]
        for event_seed in protocol.STATIONARY_HOLDOUT_EVENT_SEEDS
        for mode in protocol.MODES
    )


def _comparison(
    payloads: dict[int, dict[str, Any]], left: str, right: str,
) -> dict[str, Any]:
    seed_values = {
        str(seed): (
            _switching_seed_mean(payload, left)
            - _switching_seed_mean(payload, right)
        )
        for seed, payload in payloads.items()
    }
    differences = np.asarray(list(seed_values.values()), dtype=np.float64)
    mean = float(np.mean(differences))
    standard_error = float(
        np.std(differences, ddof=1) / math.sqrt(len(differences)))
    radius = T_CRITICAL_95_DF4 * standard_error
    event_wins = sum(
        payload["switching_holdout"][str(event_seed)][left]["return_mean"]
        > payload["switching_holdout"][str(event_seed)][right]["return_mean"]
        for payload in payloads.values()
        for event_seed in protocol.SWITCHING_EVENT_SEEDS
    )
    row = {
        "left": left,
        "right": right,
        "paired_seed_differences": seed_values,
        "paired_mean": mean,
        "ci95_low": float(mean - radius),
        "ci95_high": float(mean + radius),
        "seed_wins": int(np.sum(differences > 0.0)),
        "event_wins": int(event_wins),
    }
    row["gate_pass"] = bool(
        row["paired_mean"] > 0.0
        and row["ci95_low"] > 0.0
        and row["seed_wins"] >= protocol.REQUIRED_SEED_WINS
        and row["event_wins"] >= protocol.REQUIRED_EVENT_WINS
    )
    return row


def analyze() -> dict[str, Any]:
    protocol.validate_registration()
    payloads = {}
    for seed in protocol.TRAINING_SEEDS:
        audit.validate_audit(seed)
        payloads[int(seed)] = protocol.read_json(protocol.audit_result(seed))

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
    comparisons = {
        "causal_vs_no_compensation": _comparison(
            payloads, protocol.CAUSAL_COMPENSATION_ARM,
            protocol.NO_COMPENSATION_ARM),
        "causal_vs_robust_source": _comparison(
            payloads, protocol.CAUSAL_COMPENSATION_ARM,
            protocol.ROBUST_SOURCE_ARM),
        "causal_vs_equal_budget_sac": _comparison(
            payloads, protocol.CAUSAL_COMPENSATION_ARM,
            protocol.ROBUST_LONG_ARM),
        "causal_vs_equal_budget_escp": _comparison(
            payloads, protocol.CAUSAL_COMPENSATION_ARM,
            protocol.ESCP_ARM),
        "causal_vs_equal_budget_resac": _comparison(
            payloads, protocol.CAUSAL_COMPENSATION_ARM,
            protocol.RESAC_ARM),
    }

    recovery = {}
    stationary_retention = {}
    headroom = {}
    for seed, payload in payloads.items():
        no_comp = _switching_seed_mean(
            payload, protocol.NO_COMPENSATION_ARM)
        oracle = _switching_seed_mean(
            payload, protocol.ORACLE_COMPENSATION_ARM)
        causal = _switching_seed_mean(
            payload, protocol.CAUSAL_COMPENSATION_ARM)
        robust_long = _switching_seed_mean(payload, protocol.ROBUST_LONG_ARM)
        available = oracle - no_comp
        recovered = (
            (causal - no_comp) / available if available > 0.0 else None)
        recovery[str(seed)] = {
            "oracle_headroom": float(available),
            "causal_recovery": (
                float(recovered) if recovered is not None else None),
            "pass": bool(
                recovered is not None
                and recovered >= protocol.MIN_ORACLE_RECOVERY),
        }
        oracle_stationary = _stationary_seed_mean(
            payload, protocol.ORACLE_COMPENSATION_ARM)
        causal_stationary = _stationary_seed_mean(
            payload, protocol.CAUSAL_COMPENSATION_ARM)
        ratio = (
            causal_stationary / oracle_stationary
            if oracle_stationary > 0.0 else None)
        stationary_retention[str(seed)] = {
            "oracle_return": oracle_stationary,
            "causal_return": causal_stationary,
            "ratio": float(ratio) if ratio is not None else None,
            "pass": bool(
                ratio is not None
                and ratio >= protocol.MIN_STATIONARY_ORACLE_RETENTION),
        }
        relative = (
            (oracle - robust_long) / abs(robust_long)
            if robust_long != 0.0 else None)
        headroom[str(seed)] = {
            "oracle_return": oracle,
            "equal_budget_sac_return": robust_long,
            "relative_headroom": (
                float(relative) if relative is not None else None),
            "pass": bool(
                relative is not None
                and relative >= protocol.MIN_ORACLE_HEADROOM),
        }

    recovery_passes = sum(row["pass"] for row in recovery.values())
    retention_passes = sum(
        row["pass"] for row in stationary_retention.values())
    headroom_passes = sum(row["pass"] for row in headroom.values())
    equivalence_error = max(
        float(row["max_abs_return_error"])
        for payload in payloads.values()
        for row in payload["oracle_equivalence"].values()
    )
    equivalence_pass = bool(equivalence_error <= protocol.EXACT_RETURN_ATOL)
    causal_rows = [
        payload["switching_holdout"][str(event_seed)][
            protocol.CAUSAL_COMPENSATION_ARM]
        for payload in payloads.values()
        for event_seed in protocol.SWITCHING_EVENT_SEEDS
    ]
    diagnostics = {
        "mode_accuracy": _mean(
            row["posterior_metrics"]["mode_accuracy"] for row in causal_rows),
        "brier_score": _mean(
            row["posterior_metrics"]["brier_score"] for row in causal_rows),
        "median_detection_delay": float(np.median([
            delay for row in causal_rows
            for delay in row["switch_detection_delays"]
        ])),
        "mean_abs_execution_signal_error": _mean(
            row["mean_abs_execution_signal_error"] for row in causal_rows),
    }
    primary_comparisons = (
        "causal_vs_equal_budget_sac",
        "causal_vs_equal_budget_escp",
        "causal_vs_equal_budget_resac",
    )
    primary_pass = bool(
        equivalence_pass
        and comparisons["causal_vs_no_compensation"]["gate_pass"]
        and all(comparisons[key]["gate_pass"] for key in primary_comparisons)
        and recovery_passes >= protocol.REQUIRED_RECOVERY_SEEDS
        and retention_passes >= protocol.REQUIRED_STATIONARY_RETENTION_SEEDS
        and headroom_passes >= protocol.REQUIRED_RECOVERY_SEEDS
    )
    if primary_pass:
        diagnosis = "fresh_policy_canonical_compensation_confirmed"
    elif headroom_passes < protocol.REQUIRED_RECOVERY_SEEDS:
        diagnosis = "canonical_controller_lacks_equal_budget_oracle_headroom"
    elif not equivalence_pass:
        diagnosis = "compensation_equivalence_protocol_failure"
    elif recovery_passes < protocol.REQUIRED_RECOVERY_SEEDS:
        diagnosis = "frozen_estimator_does_not_recover_oracle_headroom"
    elif retention_passes < protocol.REQUIRED_STATIONARY_RETENTION_SEEDS:
        diagnosis = "causal_compensation_loses_stationary_oracle_performance"
    else:
        failed = [key for key in primary_comparisons
                  if not comparisons[key]["gate_pass"]]
        diagnosis = "canonical_compensation_fails_comparators:" + ",".join(failed)

    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "registration": protocol.file_record(protocol.REGISTRATION_PATH),
        "training_seeds": list(protocol.TRAINING_SEEDS),
        "reference_mode": protocol.REFERENCE_MODE,
        "switching_return_means": switching_means,
        "stationary_return_means": stationary_means,
        "comparisons": comparisons,
        "oracle_recovery": recovery,
        "oracle_recovery_seed_passes": int(recovery_passes),
        "stationary_oracle_retention": stationary_retention,
        "stationary_retention_seed_passes": int(retention_passes),
        "equal_budget_oracle_headroom": headroom,
        "oracle_headroom_seed_passes": int(headroom_passes),
        "oracle_equivalence_max_abs_return_error": equivalence_error,
        "oracle_equivalence_pass": equivalence_pass,
        "causal_switch_diagnostics": diagnostics,
        "fresh_policy_confirmation_pass": primary_pass,
        "diagnosis": diagnosis,
        "accounting": {
            "canonical_training_interactions_per_seed": (
                protocol.FINAL_TOTAL_STEPS),
            "equal_budget_interactions_per_comparator_seed": (
                protocol.FINAL_TOTAL_STEPS),
            "new_policy_seeds": len(protocol.TRAINING_SEEDS),
            "reference_policies_deployed_per_seed": 1,
            "estimator_retrained": False,
        },
    }


def render(payload: dict[str, Any]) -> str:
    names = {
        protocol.ROBUST_SOURCE_ARM: "Robust SAC, 5.6M",
        protocol.ROBUST_LONG_ARM: "Robust SAC, 8.4M",
        protocol.NO_COMPENSATION_ARM: "Canonical mode 0, no compensation",
        protocol.ORACLE_COMPENSATION_ARM: "Canonical + true-mode compensation",
        protocol.CAUSAL_COMPENSATION_ARM: "Canonical + frozen v5 compensation",
        protocol.ESCP_ARM: "Recurrent ESCP, 8.4M",
        protocol.RESAC_ARM: "RE-SAC b0, 8.4M",
    }
    lines = [
        "# V31 fresh-policy canonical action-compensation confirmation",
        "",
        "Mode 0, five policy seeds, training budgets, event streams, and "
        "decision gates were frozen before training.",
        "",
        "| Arm | Switching return | Stationary return |",
        "|---|---:|---:|",
    ]
    for arm in protocol.ARMS:
        lines.append(
            f"| {names[arm]} | {payload['switching_return_means'][arm]:.1f} "
            f"| {payload['stationary_return_means'][arm]:.1f} |")
    lines.extend([
        "",
        "## Paired switching comparisons",
        "",
        "| Comparison | Difference | 95% CI | Seeds | Events | Pass |",
        "|---|---:|---:|---:|---:|:---:|",
    ])
    labels = {
        "causal_vs_no_compensation": "Causal - no compensation",
        "causal_vs_robust_source": "Causal - robust SAC 5.6M",
        "causal_vs_equal_budget_sac": "Causal - robust SAC 8.4M",
        "causal_vs_equal_budget_escp": "Causal - ESCP 8.4M",
        "causal_vs_equal_budget_resac": "Causal - RE-SAC 8.4M",
    }
    for key, label in labels.items():
        row = payload["comparisons"][key]
        lines.append(
            f"| {label} | {row['paired_mean']:+.1f} | "
            f"[{row['ci95_low']:+.1f}, {row['ci95_high']:+.1f}] | "
            f"{row['seed_wins']}/5 | {row['event_wins']}/15 | "
            f"{'yes' if row['gate_pass'] else 'no'} |")
    diagnostics = payload["causal_switch_diagnostics"]
    lines.extend([
        "",
        "## Decision",
        "",
        f"Fresh-policy confirmation: **{payload['fresh_policy_confirmation_pass']}**.",
        f"Diagnosis: **{payload['diagnosis']}**.",
        f"Oracle headroom: {payload['oracle_headroom_seed_passes']}/5 seeds; "
        f"causal recovery: {payload['oracle_recovery_seed_passes']}/5; "
        f"stationary retention: {payload['stationary_retention_seed_passes']}/5.",
        f"Frozen-v5 mode accuracy: {diagnostics['mode_accuracy']:.4f}; "
        f"median switch delay: {diagnostics['median_detection_delay']:.1f} steps.",
        "",
        "The claim remains limited to HalfCheetah actuator-polarity, whose "
        "sign transform is exactly invertible. HalfCheetah has no health "
        "termination here, so this does not establish safety.",
        "",
        "## Accounting",
        "",
        "The compensation path uses one 5.6M robust source plus 2.8M fixed-mode "
        "fine-tuning per seed. SAC, ESCP, and RE-SAC comparators each receive "
        "8.4M interactions. The frozen v5 estimator is not retrained.",
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
