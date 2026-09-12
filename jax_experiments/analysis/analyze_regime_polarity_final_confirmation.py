"""Aggregate the independent frozen-estimator final confirmation."""
from jax_experiments.analysis import analyze_regime_polarity_posterior as base
from jax_experiments.analysis import (
    regime_polarity_final_confirmation as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_final_confirmation_audit as audit,
)


def main():
    protocol.validate_frozen_estimator()
    base.run_analysis(protocol, audit)


if __name__ == "__main__":
    main()
