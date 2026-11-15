"""Aggregate the preregistered ten-seed V32 power confirmation."""
from __future__ import annotations

from typing import Any

from jax_experiments.analysis import (
    analyze_regime_polarity_action_compensation_confirmation_v31 as base,
)
from jax_experiments.analysis import (
    regime_polarity_action_compensation_power_confirmation_v32 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_action_compensation_power_audit_v32 as audit,
)


T_CRITICAL_95_DF9 = 2.2621571627409915


def _bind() -> None:
    base.protocol = protocol
    base.audit = audit
    base.T_CRITICAL_95_DF4 = T_CRITICAL_95_DF9


def analyze() -> dict[str, Any]:
    _bind()
    payload = base.analyze()
    passed = bool(payload["fresh_policy_confirmation_pass"])
    payload["prospective_power_confirmation_pass"] = passed
    payload["pilot_accounting"] = {
        "v31_used_for_sample_size_only": True,
        "v31_results_pooled": False,
        "v32_new_policy_seeds": len(protocol.TRAINING_SEEDS),
        "planned_two_sided_power": dict(protocol.PILOT_POWER_ESTIMATES),
    }
    if passed:
        payload["diagnosis"] = "prospective_ten_seed_power_confirmation_passed"
    return payload


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
        "# V32 prospective power confirmation",
        "",
        "The V31 five-seed result was used only to choose n=10. This table "
        "contains ten new policy seeds and new event streams; V31 outcomes "
        "are not pooled.",
        "",
        "| Arm | Switching return | Stationary return |",
        "|---|---:|---:|",
    ]
    for arm in protocol.ARMS:
        lines.append(
            f"| {names[arm]} | {payload['switching_return_means'][arm]:.1f} "
            f"| {payload['stationary_return_means'][arm]:.1f} |"
        )
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
            f"{row['seed_wins']}/10 | {row['event_wins']}/30 | "
            f"{'yes' if row['gate_pass'] else 'no'} |"
        )
    diagnostics = payload["causal_switch_diagnostics"]
    lines.extend([
        "",
        "## Decision",
        "",
        "Prospective power confirmation: "
        f"**{payload['prospective_power_confirmation_pass']}**.",
        f"Diagnosis: **{payload['diagnosis']}**.",
        f"Oracle headroom: {payload['oracle_headroom_seed_passes']}/10 seeds; "
        f"causal recovery: {payload['oracle_recovery_seed_passes']}/10; "
        f"stationary retention: {payload['stationary_retention_seed_passes']}/10.",
        f"Frozen-v5 mode accuracy: {diagnostics['mode_accuracy']:.4f}; "
        f"median switch delay: {diagnostics['median_detection_delay']:.1f} steps.",
        "",
        "The decision requires positive paired means with two-sided 95% CI "
        "lower bounds above zero against equal-budget SAC, ESCP, and RE-SAC, "
        "plus at least 8/10 seed wins and 24/30 event wins.",
        "",
        "The claim remains limited to HalfCheetah actuator polarity. The "
        "compensation path and all comparators receive 8.4M interactions per "
        "policy seed; the frozen v5 estimator is not retrained.",
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
