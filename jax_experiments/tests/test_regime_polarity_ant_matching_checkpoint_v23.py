"""Focused tests for the Ant V23 checkpoint-selection diagnostic."""
from __future__ import annotations

import pytest

from jax_experiments.analysis import (
    regime_polarity_ant_matching_checkpoint_v23 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_ant_matching_checkpoint_v23 as producer,
)
from jax_experiments.analysis import (
    train_regime_polarity_ant_matching_checkpoint_v23 as trainer,
)
from scripts import submit_regime_polarity_ant_matching_checkpoint_v23 as submit


def test_matching_validation_selects_exact_mode():
    tasks = [{"mode_id": mode, "value": mode + 10} for mode in protocol.MODES]
    assert trainer.select_matching_task(tasks, 1) == [
        {"mode_id": 1, "value": 11}]
    with pytest.raises(ValueError):
        trainer.select_matching_task(tasks, 9)


def test_v23_partitions_modes_and_uses_fresh_event_splits():
    protocol.assert_protocol_integrity()
    assert set(protocol.TRAINED_MODES) == {0, 1}
    assert set(protocol.FROZEN_MODES) == {2, 3}
    assert not set(protocol.CALIBRATION_EVENT_SEEDS) & set(
        protocol.STATIONARY_HOLDOUT_EVENT_SEEDS)
    assert not set(protocol.CALIBRATION_EVENT_SEEDS) & set(
        protocol.SWITCHING_EVENT_SEEDS)


def test_v23_reuses_only_v22_modes_two_and_three():
    seed = protocol.TRAINING_SEEDS[0]
    for variant in protocol.VARIANTS:
        for mode in protocol.FROZEN_MODES:
            assert protocol.bundle_dir(variant, seed, mode) == (
                protocol.parent.bundle_dir(
                    protocol.parent.CONTROL_VARIANT, seed, mode))
        for mode in protocol.TRAINED_MODES:
            assert "matching_checkpoint_v23" in str(
                protocol.bundle_dir(variant, seed, mode))


def test_v23_corrects_selection_and_preserves_budget():
    seed = protocol.TRAINING_SEEDS[0]
    for variant, period in (
        ("matching_best", 1),
        ("period2_matching_best", 2),
    ):
        expected = producer.expected_config(variant, seed, 0)
        assert expected["sac_select_best_eval"] is True
        assert expected["sac_actor_update_period"] == period
        assert expected["max_iters"] == protocol.FINAL_NEXT_ITERATION
        assert expected["stochastic_mode_fixed_id"] == 0


def test_v23_run_binds_the_matching_mode_training_entrypoint(monkeypatch):
    observed = {}

    def fake_run(variant, seed, mode):
        command = producer.base.training_command(variant, seed, mode)
        observed["module"] = command[command.index("-m") + 1]

    monkeypatch.setattr(producer.base, "run", fake_run)
    producer.run("matching_best", protocol.TRAINING_SEEDS[0], 0)
    assert observed["module"] == (
        "jax_experiments.analysis."
        "train_regime_polarity_ant_matching_checkpoint_v23"
    )


def test_v23_dag_has_twelve_gpu_tasks_and_nineteen_total():
    rows = submit.candidates("high")
    specs = [spec for _, spec, _ in rows]
    assert len(rows) == 19
    assert sum(spec["vram"] > 0 for spec in specs) == 12
    assert sum("/audit/" in spec["signature"] for spec in specs) == 6
    assert sum(spec["signature"].endswith("/analysis") for spec in specs) == 1


def test_v23_gpu_tasks_exclude_local_and_sync_only_bundles():
    specs = [
        spec for _, spec, _ in submit.candidates("high")
        if spec["vram"] > 0
    ]
    assert specs
    assert all("local" not in spec["allowed_nodes"] for spec in specs)
    assert all(spec["vram"] == 2700 for spec in specs)
    assert all("checkpoints" not in spec["result_dir"] for spec in specs)
