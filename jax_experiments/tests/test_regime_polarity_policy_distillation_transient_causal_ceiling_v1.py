"""Tests for the delayed-oracle causal-ceiling protocol."""
from __future__ import annotations

from jax_experiments.analysis import (
    regime_polarity_policy_distillation_transient_causal_ceiling_v1 as protocol,
)


def test_causal_ceiling_grid_is_complete_and_disjoint():
    assert protocol.STUDENT_SEEDS == (1709, 1811, 1901)
    assert len(protocol.EVENT_SEEDS) == len(set(protocol.EVENT_SEEDS)) == 5
    assert len(protocol.all_audit_manifests()) == 15
    protocol.assert_split_integrity()


def test_delayed_oracle_ladder_is_registered_once():
    assert protocol.DELAY_STEPS == (1, 2, 5, 10)
    assert tuple(protocol.arm_delay(arm) for arm in protocol.DELAY_ARMS) \
        == protocol.DELAY_STEPS
    assert len(protocol.ARMS) == len(set(protocol.ARMS)) == 8


def test_cross_student_failure_and_models_are_frozen():
    for student_seed in protocol.STUDENT_SEEDS:
        protocol.validate_upstream(student_seed)


def test_identity_separates_diagnostic_mode_from_deployable_inputs():
    identity = protocol.identity(1709, protocol.EVENT_SEEDS[0])
    assert identity["selection_forbidden"] is True
    assert identity["diagnostic_only_inputs"] == [
        "mode_id_after_registered_delay"]
    assert "mode_id" in identity["forbidden_online_inputs_for_deployable_arms"]
