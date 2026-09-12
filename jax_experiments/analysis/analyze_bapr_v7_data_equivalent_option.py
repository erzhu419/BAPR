"""Compare BAPR-v7 data-equivalent options with frozen v3 baselines."""
from jax_experiments.analysis import analyze_bapr_v4_persistent_option as base
from jax_experiments.analysis import bapr_v7_data_equivalent_option as protocol


if __name__ == "__main__":
    base.analyze(
        candidate_protocol=protocol,
        version="v7",
        title="BAPR-v7 unique-data-equivalent option development screen",
    )
