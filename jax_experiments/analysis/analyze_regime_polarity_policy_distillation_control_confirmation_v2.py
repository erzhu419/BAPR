"""Aggregate the frozen mode-head policy-compression confirmation."""
from __future__ import annotations

from jax_experiments.analysis import (
    analyze_regime_polarity_policy_distillation_confirmation as base,
)
from jax_experiments.analysis import (
    regime_polarity_policy_distillation_control_confirmation_v2 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_policy_distillation_control_confirmation_audit_v2 as audit,
)


def _bind_confirmation() -> None:
    base.protocol = protocol
    base.audit = audit


def run():
    protocol.validate_frozen_candidate()
    _bind_confirmation()
    return base.run()


def main() -> None:
    run()


if __name__ == "__main__":
    main()

