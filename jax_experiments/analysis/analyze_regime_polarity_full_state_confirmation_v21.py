"""Aggregate the preregistered V21 full-state confirmation."""
from __future__ import annotations

from jax_experiments.analysis import (
    analyze_regime_polarity_v5_final_comparison_v18 as base,
)
from jax_experiments.analysis import (
    regime_polarity_full_state_final_confirmation_v21 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_full_state_confirmation_audit_v21 as audit,
)


def _bind() -> None:
    base.protocol = protocol
    base.audit = audit


def analyze():
    _bind()
    payload = base.analyze()
    if payload["strong_final_algorithm_claim"]:
        payload["diagnosis"] = (
            "full_state_final_bapr_supported_at_equal_policy_budget")
    elif payload["limited_adaptation_claim"]:
        payload["diagnosis"] = (
            "full_state_final_adaptation_supported_but_not_equal_budget")
    elif not payload["standard_baseline_gate_pass"]:
        payload["diagnosis"] = (
            "full_state_final_bapr_fails_standard_baseline_gate")
    elif not payload["oracle_recovery_gate_pass"]:
        payload["diagnosis"] = (
            "full_state_final_bapr_fails_oracle_recovery_gate")
    else:
        payload["diagnosis"] = (
            "full_state_final_bapr_fails_stationary_retention_gate")
    return payload


def render(payload) -> str:
    rendered = base.render(payload)
    rendered = rendered.replace(
        "# V18 frozen-v5 final equal-policy-budget comparison",
        "# V21 full-state-final equal-policy-budget confirmation",
        1,
    )
    rendered = rendered.replace(
        "The v17 policy bank and v5 posterior were frozen before these event "
        "streams and baseline runs.",
        "The full-state-final specialist recipe and v5 posterior were frozen "
        "before these new policy seeds, event streams, and baseline runs.",
        1,
    )
    return rendered


def main() -> None:
    payload = analyze()
    markdown = render(payload)
    protocol.write_json_atomic(protocol.analysis_json(), payload)
    protocol.write_text_atomic(protocol.analysis_markdown(), markdown)
    protocol.write_text_atomic(protocol.REPORT, markdown)
    print(
        "V21 FULL-STATE CONFIRMATION COMPLETE: "
        f"diagnosis={payload['diagnosis']} "
        f"strong={payload['strong_final_algorithm_claim']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
