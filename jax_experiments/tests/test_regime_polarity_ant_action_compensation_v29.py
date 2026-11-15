from __future__ import annotations

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_ant_action_compensation_v29 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_ant_action_compensation_audit_v29 as audit,
)


def test_v29_event_splits_are_disjoint() -> None:
    splits = [
        set(protocol.CALIBRATION_EVENT_SEEDS),
        set(protocol.STATIONARY_HOLDOUT_EVENT_SEEDS),
        set(protocol.SWITCHING_EVENT_SEEDS),
    ]
    assert all(
        not left.intersection(right)
        for index, left in enumerate(splits)
        for right in splits[index + 1:]
    )


def test_ant_compensation_preserves_executed_signal() -> None:
    action = np.linspace(-0.95, 0.95, 8, dtype=np.float32)
    gains = np.asarray(protocol.mode_gain_vectors(8), dtype=np.float32)
    for reference_mode in protocol.MODES:
        for actual_mode in protocol.MODES:
            command = protocol.compensate_action(
                action, reference_mode, actual_mode)
            np.testing.assert_allclose(
                gains[actual_mode] * command,
                gains[reference_mode] * action,
                atol=protocol.EXACT_ACTION_ATOL,
                rtol=0.0,
            )


def test_rollout_summary_counts_terminations() -> None:
    row = audit._rollout_summary(
        [1.0, 2.0], [0, 2], [1000, 125], 2000, [0.0, 0.0])
    assert row["termination_count"] == 2
    assert row["terminated_rate"] == 0.5
    assert row["mean_first_termination_step"] == 562.5
    assert row["max_abs_execution_signal_error"] == 0.0
