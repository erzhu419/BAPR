"""Compare BAPR-v6 balanced options with frozen v3 baselines."""
from jax_experiments.analysis import analyze_bapr_v4_persistent_option as base
from jax_experiments.analysis import bapr_v6_balanced_option as protocol


if __name__ == "__main__":
    base.analyze(
        candidate_protocol=protocol,
        version="v6",
        title="BAPR-v6 optimizer-equivalent option development screen",
    )
