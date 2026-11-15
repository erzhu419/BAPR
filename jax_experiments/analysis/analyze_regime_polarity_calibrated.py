"""Aggregate the unseen five-seed calibrated-posterior screen."""
from jax_experiments.analysis import analyze_regime_polarity_posterior as base
from jax_experiments.analysis import regime_polarity_evidence_calibration as protocol
from jax_experiments.analysis import run_regime_polarity_calibrated_audit as audit


def main() -> None:
    base.run_analysis(protocol, audit)


if __name__ == "__main__":
    main()
