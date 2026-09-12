"""Aggregate one stable-LCB conservative residual v4 variant."""
from jax_experiments.analysis.regime_polarity_conservative_residual_stable import (
    bind,
)

protocol = bind()

from jax_experiments.analysis import (  # noqa: E402
    analyze_regime_polarity_conservative_residual as implementation,
)

implementation.protocol = protocol


if __name__ == "__main__":
    implementation.main()
