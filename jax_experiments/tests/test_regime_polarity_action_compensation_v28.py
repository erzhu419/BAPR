"""Focused tests for the frozen V28 action-compensation audit."""
from __future__ import annotations

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_action_compensation_v28 as protocol,
)
from scripts import submit_regime_polarity_action_compensation_v28 as submit


def test_v28_event_splits_and_balanced_schedules_are_frozen():
    protocol.assert_protocol_integrity()
    splits = (
        set(protocol.CALIBRATION_EVENT_SEEDS),
        set(protocol.STATIONARY_HOLDOUT_EVENT_SEEDS),
        set(protocol.SWITCHING_EVENT_SEEDS),
    )
    assert not any(
        left & right
        for index, left in enumerate(splits)
        for right in splits[index + 1:]
    )
    assert all(set(sequence) == set(protocol.MODES)
               for sequence in protocol.SWITCHING_SCHEDULES.values())


def test_v28_sign_compensation_preserves_executed_signal():
    rng = np.random.RandomState(28)
    for act_dim in (3, 6, 8):
        gains = protocol.mode_gain_vectors(act_dim)
        action = rng.uniform(-1.0, 1.0, size=(13, act_dim)).astype(np.float32)
        for reference_mode in protocol.MODES:
            expected = action * gains[reference_mode]
            for target_mode in protocol.MODES:
                command = protocol.compensate_action(
                    action, reference_mode, target_mode)
                actual = command * gains[target_mode]
                np.testing.assert_allclose(actual, expected, atol=0.0, rtol=0.0)


def test_v28_is_seven_cpu_only_file_gated_tasks():
    rows = submit.candidates("high")
    specs = [spec for _, spec, _ in rows]
    assert len(rows) == 7
    assert all(spec["vram"] == 0 for spec in specs)
    assert all(spec["allowed_nodes"] == submit.CPU_NODES for spec in specs)
    assert all("local" not in spec["allowed_nodes"] for spec in specs)
    assert sum("/audit/" in spec["signature"] for spec in specs) == 5
    assert sum(spec["signature"].endswith("/analysis") for spec in specs) == 1


def test_v28_audits_wait_for_structure_and_compact_v21_inputs_only():
    for seed in protocol.TRAINING_SEEDS:
        spec = submit.audit_spec(seed, "high")
        assert str(protocol.STRUCTURAL_MANIFEST) in spec["wait_for_files"]
        assert str(protocol.v5_model.MODEL_PATH) in spec["wait_for_files"]
        assert all("checkpoints" not in path for path in spec["wait_for_files"])
        assert all("train_state.pkl" not in path for path in spec["wait_for_files"])
        for mode in protocol.MODES:
            assert all(
                str(path) in spec["wait_for_files"]
                for path in protocol.specialist_required_paths(seed, mode)
            )
        for slot in protocol.SAC_REPLICA_SLOTS:
            assert all(
                str(path) in spec["wait_for_files"]
                for path in protocol.bundle_required_paths(
                    "sac_replica", seed, slot)
            )


def test_v28_registration_separates_policy_count_and_interactions():
    payload = protocol.registration_payload()
    assert payload["accounting"]["new_training_interactions"] == 0
    assert payload["accounting"]["reused_policy_count_bapr"] == 5
    assert payload["accounting"]["reused_policy_count_sac5"] == 5
    assert payload["accounting"]["v21_bapr_training_interactions"] == 16_800_000
    assert payload["accounting"]["v21_sac5_training_interactions"] == 28_000_000
    assert payload["scope"]["ant_causal_estimator_available"] is False
