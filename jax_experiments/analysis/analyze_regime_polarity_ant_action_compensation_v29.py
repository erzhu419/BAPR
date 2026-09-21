"""Aggregate the registered V29 Ant oracle compensation audit."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_ant_action_compensation_v29 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_ant_action_compensation_audit_v29 as audit,
)


T_CRITICAL_95_DF2 = 4.302652729696142


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


def _switching_termination(payload: dict[str, Any], arm: str) -> float:
    return _mean(
        payload["switching_holdout"][str(event_seed)][arm]["terminated_rate"]
        for event_seed in protocol.SWITCHING_EVENT_SEEDS
    )


def _stationary_termination(payload: dict[str, Any], arm: str) -> float:
    return _mean(
        payload["stationary_holdout"][str(event_seed)][arm][str(mode)][
            "terminated_rate"]
        for event_seed in protocol.STATIONARY_HOLDOUT_EVENT_SEEDS
        for mode in protocol.MODES
    )


def _comparison(
    payloads: dict[int, dict[str, Any]], left: str, right: str,
) -> dict[str, Any]:
    seed_differences = {
        str(seed): (
            _switching_seed_mean(payload, left)
            - _switching_seed_mean(payload, right)
        )
        for seed, payload in payloads.items()
    }
    differences = np.asarray(
        list(seed_differences.values()), dtype=np.float64)
    mean = float(np.mean(differences))
    standard_error = float(
        np.std(differences, ddof=1) / math.sqrt(len(differences)))
    radius = T_CRITICAL_95_DF2 * standard_error
    event_wins = sum(
        payload["switching_holdout"][str(event_seed)][left]["return_mean"]
        > payload["switching_holdout"][str(event_seed)][right]["return_mean"]
        for payload in payloads.values()
        for event_seed in protocol.SWITCHING_EVENT_SEEDS
    )
    return {
        "left": left,
        "right": right,
        "paired_seed_differences": seed_differences,
        "paired_mean": mean,
        "ci95_low": float(mean - radius),
        "ci95_high": float(mean + radius),
        "seed_wins": int(np.sum(differences > 0.0)),
        "event_wins": int(event_wins),
    }


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
    switching_termination = {
        arm: _mean(_switching_termination(payload, arm)
                   for payload in payloads.values())
        for arm in protocol.ARMS
    }
    stationary_termination = {
        arm: _mean(_stationary_termination(payload, arm)
                   for payload in payloads.values())
        for arm in protocol.ARMS
    }
    comparisons = {
        "oracle_vs_robust": _comparison(
            payloads, protocol.ORACLE_COMPENSATION_ARM, protocol.ROBUST_ARM),
        "oracle_vs_no_compensation": _comparison(
            payloads, protocol.ORACLE_COMPENSATION_ARM,
            protocol.NO_COMPENSATION_ARM),
        "oracle_vs_dynamic_bank": _comparison(
            payloads, protocol.ORACLE_COMPENSATION_ARM,
            protocol.DYNAMIC_BANK_ARM),
    }

    seed_rows = {}
    for seed, payload in payloads.items():
        robust_switch = _switching_seed_mean(payload, protocol.ROBUST_ARM)
        oracle_switch = _switching_seed_mean(
            payload, protocol.ORACLE_COMPENSATION_ARM)
        seed_rows[str(seed)] = {
            "reference_mode": int(
                payload["calibration"]["selected_reference_mode"]),
            "robust_switching_return": robust_switch,
            "oracle_switching_return": oracle_switch,
            "switching_relative_gain_over_robust": float(
                (oracle_switch - robust_switch) / abs(robust_switch)),
            "robust_stationary_return": _stationary_seed_mean(
                payload, protocol.ROBUST_ARM),
            "oracle_stationary_return": _stationary_seed_mean(
                payload, protocol.ORACLE_COMPENSATION_ARM),
            "oracle_switching_termination": _switching_termination(
                payload, protocol.ORACLE_COMPENSATION_ARM),
            "oracle_stationary_termination": _stationary_termination(
                payload, protocol.ORACLE_COMPENSATION_ARM),
        }
        row = seed_rows[str(seed)]
        row["gate_pass"] = bool(
            row["switching_relative_gain_over_robust"]
            >= protocol.MIN_SWITCHING_GAIN_OVER_ROBUST
            and row["oracle_stationary_return"]
            > row["robust_stationary_return"]
            and row["oracle_switching_termination"] == 0.0
            and row["oracle_stationary_termination"] == 0.0
        )

    equivalence_errors = [
        float(row["max_abs_return_error"])
        for payload in payloads.values()
        for row in payload["oracle_equivalence"].values()
    ]
    equivalence_pass = all(
        row.get("pass") is True
        for payload in payloads.values()
        for row in payload["oracle_equivalence"].values()
    )
    max_signal_error = max(
        float(payload["switching_holdout"][str(event_seed)][
            protocol.ORACLE_COMPENSATION_ARM][
                "max_abs_execution_signal_error"])
        for payload in payloads.values()
        for event_seed in protocol.SWITCHING_EVENT_SEEDS
    )
    primary_pass = bool(
        equivalence_pass
        and max(equivalence_errors) <= protocol.EXACT_RETURN_ATOL
        and max_signal_error <= protocol.EXACT_ACTION_ATOL
        and all(row["gate_pass"] for row in seed_rows.values())
        and comparisons["oracle_vs_robust"]["seed_wins"]
        == protocol.REQUIRED_SEED_WINS
        and comparisons["oracle_vs_robust"]["event_wins"]
        == protocol.REQUIRED_EVENT_WINS
        and comparisons["oracle_vs_no_compensation"]["seed_wins"]
        == protocol.REQUIRED_SEED_WINS
        and comparisons["oracle_vs_no_compensation"]["event_wins"]
        == protocol.REQUIRED_EVENT_WINS
    )
    if primary_pass:
        diagnosis = "ant_oracle_action_compensation_supported"
        next_step = (
            "train an Ant causal mode estimator against the frozen compensated "
            "reference policy on new development data")
    elif not equivalence_pass or max_signal_error > protocol.EXACT_ACTION_ATOL:
        diagnosis = "ant_action_compensation_protocol_failure"
        next_step = "debug the environment transform; do not train an estimator"
    elif any(
        row["oracle_switching_termination"] > 0.0
        or row["oracle_stationary_termination"] > 0.0
        for row in seed_rows.values()
    ):
        diagnosis = "ant_reference_policy_is_not_survival_safe"
        next_step = "stop estimator work and retain Ant as a safety counterexample"
    else:
        diagnosis = "ant_oracle_compensation_lacks_consistent_headroom"
        next_step = "stop estimator work; action symmetry alone is insufficient"

    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "registration": protocol.file_record(protocol.REGISTRATION_PATH),
        "training_seeds": list(protocol.TRAINING_SEEDS),
        "seed_results": seed_rows,
        "switching_return_means": switching_means,
        "stationary_return_means": stationary_means,
        "switching_termination_means": switching_termination,
        "stationary_termination_means": stationary_termination,
        "comparisons": comparisons,
        "oracle_equivalence_max_abs_return_error": max(equivalence_errors),
        "oracle_equivalence_pass": equivalence_pass,
        "oracle_max_abs_execution_signal_error": max_signal_error,
        "estimator_training_authorized": primary_pass,
        "diagnosis": diagnosis,
        "next_step": next_step,
        "accounting": {
            "new_training_interactions": 0,
            "reused_policy_seeds": len(protocol.TRAINING_SEEDS),
            "reused_policies_per_seed": 5,
        },
    }


def render(payload: dict[str, Any]) -> str:
    names = {
        protocol.ROBUST_ARM: "Robust SAC",
        protocol.NO_COMPENSATION_ARM: "Reference, no compensation",
        protocol.ORACLE_COMPENSATION_ARM: "Reference + true-mode compensation",
        protocol.DYNAMIC_BANK_ARM: "V22 dynamic specialist oracle",
    }
    lines = [
        "# V29 Ant oracle action-compensation audit",
        "",
        "This no-training development audit reuses the three frozen V22 Ant "
        "policy seeds and new calibration/holdout streams.",
        "",
        "| Arm | Switching return | Stationary return | Switch term. | Static term. |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in protocol.ARMS:
        lines.append(
            f"| {names[arm]} | {payload['switching_return_means'][arm]:.1f} "
            f"| {payload['stationary_return_means'][arm]:.1f} "
            f"| {payload['switching_termination_means'][arm]:.1%} "
            f"| {payload['stationary_termination_means'][arm]:.1%} |")
    lines.extend([
        "",
        "| Seed | Ref. mode | Robust switch | Oracle switch | Gain | Oracle term. | Gate |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ])
    for seed, row in payload["seed_results"].items():
        lines.append(
            f"| {seed} | {row['reference_mode']} "
            f"| {row['robust_switching_return']:.1f} "
            f"| {row['oracle_switching_return']:.1f} "
            f"| {row['switching_relative_gain_over_robust']:+.1%} "
            f"| {row['oracle_switching_termination']:.1%} "
            f"| {'PASS' if row['gate_pass'] else 'FAIL'} |")
    lines.extend(["", "## Paired switching comparisons", "",
                  "| Comparison | Difference | 95% CI | Seeds | Events |",
                  "|---|---:|---:|---:|---:|"])
    labels = {
        "oracle_vs_robust": "Oracle compensation - robust SAC",
        "oracle_vs_no_compensation": "Oracle compensation - no compensation",
        "oracle_vs_dynamic_bank": "Oracle compensation - dynamic bank",
    }
    for key, label in labels.items():
        row = payload["comparisons"][key]
        lines.append(
            f"| {label} | {row['paired_mean']:+.1f} "
            f"| [{row['ci95_low']:+.1f}, {row['ci95_high']:+.1f}] "
            f"| {row['seed_wins']}/3 | {row['event_wins']}/9 |")
    lines.extend([
        "",
        "## Decision",
        "",
        f"Oracle compensation gate: **{'PASS' if payload['estimator_training_authorized'] else 'FAIL'}**.",
        f"Diagnosis: **{payload['diagnosis']}**.",
        f"Maximum native-equivalence return error: "
        f"`{payload['oracle_equivalence_max_abs_return_error']:.6g}`.",
        f"Next step: {payload['next_step']}.",
        "",
        "This is an Ant development upper-bound audit, not an independent "
        "cross-environment confirmation. It uses true mode and performs no "
        "estimator or policy training.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    payload = analyze()
    markdown = render(payload)
    protocol.ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    protocol.write_json_atomic(protocol.analysis_json(), payload)
    protocol.write_text_atomic(protocol.analysis_markdown(), markdown)
    protocol.write_text_atomic(protocol.REPORT, markdown)
    print(
        "V29 ANT ACTION COMPENSATION ANALYSIS COMPLETE: "
        f"diagnosis={payload['diagnosis']} "
        f"estimator_authorized={payload['estimator_training_authorized']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
