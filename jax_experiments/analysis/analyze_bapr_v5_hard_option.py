"""Compare BAPR-v5 hard persistent options with frozen v3 baselines."""
from jax_experiments.analysis import analyze_bapr_v4_persistent_option as base
from jax_experiments.analysis import bapr_v5_hard_option as protocol


if __name__ == "__main__":
    base.analyze(
        candidate_protocol=protocol,
        version="v5",
        title="BAPR-v5 isolated hard-option development screen",
    )
