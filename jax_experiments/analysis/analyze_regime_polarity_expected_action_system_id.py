"""Aggregate the expected-action supervision ablation."""
from jax_experiments.analysis import analyze_regime_polarity_posterior as base
from jax_experiments.analysis import (
    regime_polarity_expected_action_system_id as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_expected_action_system_id_audit as audit,
)


def main():
    base.run_analysis(protocol, audit)


if __name__ == "__main__":
    main()
