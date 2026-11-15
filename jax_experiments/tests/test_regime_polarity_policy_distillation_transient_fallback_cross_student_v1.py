"""Tests for the frozen cross-student transient-fallback protocol."""
from __future__ import annotations

from jax_experiments.analysis import (
    regime_polarity_policy_distillation_transient_fallback_cross_student_v1 as protocol,
)


def test_cross_student_grid_is_complete_and_disjoint():
    assert protocol.STUDENT_SEEDS == (1709, 1811, 1901)
    assert len(protocol.EVENT_SEEDS) == len(set(protocol.EVENT_SEEDS)) == 5
    assert len(protocol.all_audit_manifests()) == 15
    protocol.assert_split_integrity()


def test_selected_fallback_is_frozen_without_reselection():
    assert protocol.SELECTED_CONFIG_NAME == "evidence_1p0_k1"
    assert protocol.parent.require_config(
        protocol.SELECTED_CONFIG_NAME).to_dict() \
        == protocol.SELECTED_CONFIG_VALUES


def test_every_student_checkpoint_is_hash_frozen():
    for student_seed in protocol.STUDENT_SEEDS:
        protocol.validate_upstream(student_seed)


def test_identity_declares_causal_and_forbidden_inputs():
    identity = protocol.identity(1709, protocol.EVENT_SEEDS[0])
    assert "causal_mode_posterior" in identity["online_inputs"]
    assert "one_step_mode_log_likelihood" in identity["online_inputs"]
    assert "mode_id" in identity["forbidden_online_inputs"]
    assert "switch_clock" in identity["forbidden_online_inputs"]
