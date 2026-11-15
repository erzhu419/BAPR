from __future__ import annotations

import jax

from jax_experiments.analysis import (
    analyze_regime_polarity_ant_branch_risk_v30 as analysis,
)
from jax_experiments.analysis import (
    regime_polarity_ant_branch_risk_v30 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_ant_branch_risk_audit_v30 as audit,
)


def test_v30_source_events_are_new() -> None:
    previous = {
        *protocol.parent.CALIBRATION_EVENT_SEEDS,
        *protocol.parent.STATIONARY_HOLDOUT_EVENT_SEEDS,
        *protocol.parent.SWITCHING_EVENT_SEEDS,
        *protocol.parent.policy_parent.CALIBRATION_EVENT_SEEDS,
        *protocol.parent.policy_parent.STATIONARY_HOLDOUT_EVENT_SEEDS,
        *protocol.parent.policy_parent.SWITCHING_EVENT_SEEDS,
    }
    assert not previous.intersection(protocol.SOURCE_EVENT_SEEDS)


def test_branch_keys_are_reproducible_and_replicate_specific() -> None:
    first = audit._branch_key(85003, 203001, 25, 0)
    repeated = audit._branch_key(85003, 203001, 25, 0)
    second = audit._branch_key(85003, 203001, 25, 1)
    assert bool(jax.numpy.array_equal(first, repeated))
    assert not bool(jax.numpy.array_equal(first, second))


def test_pair_counter_distinguishes_rescue_from_harm() -> None:
    counter = audit._empty_counter()
    audit._add_pair(
        counter,
        {"terminated": True, "return": 1.0},
        {"terminated": False, "return": 2.0},
    )
    audit._add_pair(
        counter,
        {"terminated": False, "return": 3.0},
        {"terminated": True, "return": 1.0},
    )
    row = audit._finalize(counter)
    assert row["candidate_termination_risk"] == 0.5
    assert row["fallback_termination_risk"] == 0.5
    assert row["rescue_fraction_given_candidate_failure"] == 1.0
    assert row["harm_fraction_given_candidate_survival"] == 1.0
    assert row["fallback_minus_candidate_return"] == -0.5


def test_seed_gate_requires_real_candidate_failures() -> None:
    horizon = str(protocol.MAX_RISK_HORIZON)
    safe = {
        "n": 100,
        "candidate_terminated": 0,
        "fallback_terminated": 0,
        "candidate_termination_risk": 0.0,
        "fallback_termination_risk": 0.0,
        "absolute_risk_reduction": 0.0,
        "rescue_fraction_given_candidate_failure": 0.0,
        "harm_fraction_given_candidate_survival": 0.0,
    }
    payload = {
        "identity": {"reference_mode": 2},
        "snapshot_count": 10,
        "overall_horizons": {horizon: safe},
        "unique_candidate_horizons": {
            horizon: {"n": 25, "terminated": 0}},
        "mode_horizons": {
            str(mode): {horizon: safe} for mode in protocol.MODES},
    }
    row = analysis._seed_decision(payload)
    assert row["informative"] is False
    assert row["gate_pass"] is False
