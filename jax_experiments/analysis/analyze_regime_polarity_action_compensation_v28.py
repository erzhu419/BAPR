"""Aggregate the preregistered V28 action-compensation mechanism audit."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_action_compensation_v28 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_action_compensation_audit_v28 as audit,
)
from jax_experiments.analysis import (
    run_regime_polarity_action_compensation_structural_v28 as structural,
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
    structural_manifest = structural.validate_structural_audit()
    payloads = {}
    for seed in protocol.TRAINING_SEEDS:
        audit.validate_audit(seed)
        payloads[int(seed)] = protocol.read_json(protocol.audit_result(seed))

    switching_means = {
        arm: _mean(
            _switching_seed_mean(payload, arm)
            for payload in payloads.values())
        for arm in protocol.ARMS
    }
    stationary_means = {
        arm: _mean(
            _stationary_seed_mean(payload, arm)
            for payload in payloads.values())
        for arm in protocol.ARMS
    }
    comparisons = {
        "causal_vs_no_compensation": _comparison(
            payloads, protocol.CAUSAL_COMPENSATION_ARM,
            protocol.NO_COMPENSATION_ARM),
        "causal_vs_robust": _comparison(
            payloads, protocol.CAUSAL_COMPENSATION_ARM,
            protocol.ROBUST_ARM),
        "causal_vs_v21_bank": _comparison(
            payloads, protocol.CAUSAL_COMPENSATION_ARM,
            protocol.V21_BANK_ARM),
        "v21_bank_vs_causal": _comparison(
            payloads, protocol.V21_BANK_ARM,
            protocol.CAUSAL_COMPENSATION_ARM),
        "causal_vs_sac5": _comparison(
            payloads, protocol.CAUSAL_COMPENSATION_ARM,
            protocol.SAC5_ARM),
    }

    recovery = {}
    for seed, payload in payloads.items():
        no_comp = _switching_seed_mean(
            payload, protocol.NO_COMPENSATION_ARM)
        oracle = _switching_seed_mean(
            payload, protocol.ORACLE_COMPENSATION_ARM)
        causal = _switching_seed_mean(
            payload, protocol.CAUSAL_COMPENSATION_ARM)
        headroom = oracle - no_comp
        recovery[str(seed)] = {
            "oracle_headroom": float(headroom),
            "causal_recovery": (
                float((causal - no_comp) / headroom)
                if headroom > 0.0 else None),
            "pass": bool(
                headroom > 0.0
                and (causal - no_comp) / headroom
                >= protocol.MIN_ORACLE_RECOVERY),
        }
    recovery_seed_passes = sum(row["pass"] for row in recovery.values())

    oracle_equivalence_max = max(
        float(event["max_abs_return_error"])
        for payload in payloads.values()
        for event in payload["oracle_equivalence"].values()
    )
    oracle_equivalence_pass = bool(
        oracle_equivalence_max <= protocol.EXACT_RETURN_ATOL)

    causal_rows = [
        payload["switching_holdout"][str(event_seed)][
            protocol.CAUSAL_COMPENSATION_ARM]
        for payload in payloads.values()
        for event_seed in protocol.SWITCHING_EVENT_SEEDS
    ]
    switch_diagnostics = {
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
        "wrong_route_fraction_after_switch": {
            str(size): _mean(
                row["wrong_route_fraction_after_switch"][str(size)]
                for row in causal_rows)
            for size in (1, 2, 4, 8)
        },
    }

    primary_pass = bool(
        structural_manifest["status"] == "complete"
        and oracle_equivalence_pass
        and comparisons["causal_vs_no_compensation"]["gate_pass"]
        and comparisons["causal_vs_robust"]["gate_pass"]
        and recovery_seed_passes >= protocol.REQUIRED_RECOVERY_SEEDS
    )
    if comparisons["v21_bank_vs_causal"]["gate_pass"]:
        bank_conclusion = "v21_specialist_bank_adds_confirmed_value"
    elif comparisons["causal_vs_v21_bank"]["gate_pass"]:
        bank_conclusion = "single_policy_compensation_is_superior"
    else:
        bank_conclusion = "bank_necessity_unresolved"
    if primary_pass:
        diagnosis = "causal_action_compensation_supported"
    elif not oracle_equivalence_pass:
        diagnosis = "action_compensation_protocol_failure"
    elif not comparisons["causal_vs_no_compensation"]["gate_pass"]:
        diagnosis = "causal_compensation_does_not_beat_uncompensated_reference"
    elif not comparisons["causal_vs_robust"]["gate_pass"]:
        diagnosis = "causal_compensation_does_not_beat_robust_sac"
    else:
        diagnosis = "causal_compensation_fails_oracle_recovery"

    return {
        "schema": protocol.ANALYSIS_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "registration": protocol.file_record(protocol.REGISTRATION_PATH),
        "structural_manifest": protocol.file_record(
            protocol.STRUCTURAL_MANIFEST),
        "training_seeds": list(protocol.TRAINING_SEEDS),
        "selected_reference_modes": {
            str(seed): int(payload["calibration"]["selected_reference_mode"])
            for seed, payload in payloads.items()
        },
        "switching_return_means": switching_means,
        "stationary_return_means": stationary_means,
        "comparisons": comparisons,
        "oracle_recovery": recovery,
        "oracle_recovery_seed_passes": int(recovery_seed_passes),
        "oracle_equivalence_max_abs_return_error": oracle_equivalence_max,
        "oracle_equivalence_pass": oracle_equivalence_pass,
        "causal_switch_diagnostics": switch_diagnostics,
        "causal_compensation_claim_pass": primary_pass,
        "bank_conclusion": bank_conclusion,
        "diagnosis": diagnosis,
        "accounting": {
            "new_training_interactions": 0,
            "bapr_policy_count": 5,
            "bapr_original_training_interactions": 16_800_000,
            "sac5_policy_count": 5,
            "sac5_original_training_interactions": 28_000_000,
        },
    }


def render(payload: dict[str, Any]) -> str:
    names = {
        protocol.ROBUST_ARM: "Robust SAC",
        protocol.NO_COMPENSATION_ARM: "Reference, no compensation",
        protocol.ORACLE_COMPENSATION_ARM: "Reference + true-mode compensation",
        protocol.CAUSAL_COMPENSATION_ARM: "Reference + causal v5 compensation",
        protocol.V21_BANK_ARM: "V21 bank + causal v5",
        protocol.SAC5_ARM: "SAC5 + causal v5",
    }
    lines = [
        "# V28 actuator-polarity action-compensation audit",
        "",
        "This is a no-training mechanism audit over the five frozen V21 "
        "HalfCheetah policy seeds and new calibration/holdout streams.",
        "",
        "| Arm | Switching return | Stationary return |",
        "|---|---:|---:|",
    ]
    for arm in protocol.ARMS:
        lines.append(
            f"| {names[arm]} | {payload['switching_return_means'][arm]:.1f} "
            f"| {payload['stationary_return_means'][arm]:.1f} |")
    lines.extend(["", "## Paired switching comparisons", "",
                  "| Comparison | Difference | 95% CI | Seeds | Events | Pass |",
                  "|---|---:|---:|---:|---:|---:|"])
    labels = {
        "causal_vs_no_compensation": "Causal compensation - no compensation",
        "causal_vs_robust": "Causal compensation - robust SAC",
        "causal_vs_v21_bank": "Causal compensation - V21 bank",
        "v21_bank_vs_causal": "V21 bank - causal compensation",
        "causal_vs_sac5": "Causal compensation - SAC5",
    }
    for key, label in labels.items():
        row = payload["comparisons"][key]
        lines.append(
            f"| {label} | {row['paired_mean']:+.1f} | "
            f"[{row['ci95_low']:+.1f}, {row['ci95_high']:+.1f}] | "
            f"{row['seed_wins']}/5 | {row['event_wins']}/15 | "
            f"{'PASS' if row['gate_pass'] else 'FAIL'} |")
    diagnostics = payload["causal_switch_diagnostics"]
    lines.extend([
        "",
        "## Mechanism checks",
        "",
        f"- End-to-end oracle equivalence: "
        f"{'PASS' if payload['oracle_equivalence_pass'] else 'FAIL'}; maximum "
        f"paired return error "
        f"`{payload['oracle_equivalence_max_abs_return_error']:.6g}`.",
        f"- Oracle-headroom recovery: "
        f"`{payload['oracle_recovery_seed_passes']}/5` seeds pass the frozen "
        f"{protocol.MIN_ORACLE_RECOVERY:.0%} threshold.",
        f"- Causal mode accuracy: `{diagnostics['mode_accuracy']:.4f}`; median "
        f"switch detection delay: `{diagnostics['median_detection_delay']:.1f}` "
        "steps.",
        f"- Mean executed-signal error from causal mode mistakes: "
        f"`{diagnostics['mean_abs_execution_signal_error']:.6f}`.",
        "",
        "## Decision",
        "",
        f"Causal compensation claim: "
        f"**{'PASS' if payload['causal_compensation_claim_pass'] else 'FAIL'}**.",
        f"Bank comparison: **{payload['bank_conclusion']}**.",
        f"Diagnosis: **{payload['diagnosis']}**.",
        "",
        "HalfCheetah has no health termination in this implementation. These "
        "results measure return and switching control loss, not safety. The "
        "sign-transform result is specific to actuator polarity and is not "
        "claimed for bus uncertainty or non-invertible gain loss.",
        "",
        "## Accounting",
        "",
        "No new policy or estimator training was performed. V21 BAPR reuses "
        "five policies trained with 16.8M total interactions; SAC5 reuses five "
        "policies trained with 28.0M. Policy count, interactions, and estimator "
        "cost are separate quantities.",
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
        "V28 ACTION COMPENSATION ANALYSIS COMPLETE: "
        f"diagnosis={payload['diagnosis']} "
        f"pass={payload['causal_compensation_claim_pass']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
