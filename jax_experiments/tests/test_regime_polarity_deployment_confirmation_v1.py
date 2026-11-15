from __future__ import annotations

import numpy as np

from jax_experiments.analysis import (
    analyze_regime_polarity_deployment_confirmation_v1 as analyzer,
)
from jax_experiments.analysis import (
    regime_polarity_deployment_confirmation_v1 as protocol,
)


def test_fresh_cohort_and_event_split_are_exact():
    assert len(protocol.TRAINING_SEEDS) == 10
    assert len(set(protocol.TRAINING_SEEDS)) == 10
    assert not set(protocol.TRAINING_SEEDS) & set(protocol.frozen.TRAINING_SEEDS)
    assert len(protocol.EVENT_SEEDS) == 5
    assert not set(protocol.EVENT_SEEDS) & set(protocol.mechanism.EVENT_SEEDS)
    protocol.assert_split_integrity()


def test_registered_method_and_budget_contract():
    assert protocol.METHODS == (
        "bapr", "sac", "escp_recurrent", "resac_b0")
    assert protocol.FINAL_TOTAL_STEPS == 5_600_000
    assert protocol.FINAL_UPDATE_COUNT == 350_000
    assert protocol.MIN_SEED_WINS == 8
    assert protocol.MIN_STATIONARY_RETENTION == 0.95


def test_registration_source_closure_exists():
    paths = protocol.registration_source_paths()
    assert len(paths) == len(set(paths))
    assert all(path.is_file() for path in paths)


def test_paired_statistics_and_holm_are_directional():
    baseline = np.arange(10, dtype=np.float64)
    rows = {
        "sac": analyzer._paired(baseline + 3.0, baseline),
        "escp_recurrent": analyzer._paired(baseline + 2.0, baseline),
        "resac_b0": analyzer._paired(baseline + 1.0, baseline),
    }
    analyzer._apply_holm(rows)
    assert all(row["holm_reject_at_0p05"] for row in rows.values())
    assert all(
        row["bonferroni_simultaneous_95pct_lower_bound"] > 0.0
        for row in rows.values())
    assert all(row["seed_slot_wins"] == 10 for row in rows.values())


def test_expected_checkpoint_is_controller_budget_matched():
    for method in protocol.TRAINED_METHODS:
        record = protocol.expected_checkpoint(method)
        assert record["next_iteration"] == protocol.MAX_ITERS
        assert record["total_steps"] == protocol.FINAL_TOTAL_STEPS
        assert record["update_count"] == protocol.FINAL_UPDATE_COUNT

