"""Aggregate Ant v22 with the protocol-specific audit API adapter."""
from __future__ import annotations

from jax_experiments.analysis import (
    analyze_regime_polarity_ant_full_state_headroom_v22 as aggregate,
)


def analyze() -> dict:
    original_validate = aggregate.audit.validate_audit

    def validate_audit(_variant: str, seed: int):
        return original_validate(seed)

    aggregate.audit.validate_audit = validate_audit
    try:
        payload = aggregate.analyze()
    finally:
        aggregate.audit.validate_audit = original_validate
    payload["aggregation_amendment"] = {
        "id": "ant-v22-validate-audit-arity",
        "scope": "aggregation-only",
        "audit_artifacts_unchanged": True,
    }
    return payload


def main() -> None:
    payload = analyze()
    markdown = aggregate._markdown(payload)
    markdown += (
        "\nAggregation amendment: adapted the protocol-specific "
        "`validate_audit(seed)` API; audit artifacts and gates are unchanged.\n"
    )
    protocol = aggregate.protocol
    protocol.write_json_atomic(protocol.analysis_json(), payload)
    protocol.write_text_atomic(protocol.analysis_markdown(), markdown)
    protocol.write_text_atomic(protocol.REPORT, markdown)
    print(markdown, end="")


if __name__ == "__main__":
    main()
