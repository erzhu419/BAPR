from __future__ import annotations

import sys
from pathlib import Path

from jax_experiments.analysis import (
    regime_polarity_switch_weighted_estimator_v16 as protocol,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import submit_regime_polarity_switch_weighted_estimator_v16 as submitter


def test_v16_splits_and_training_objective_are_frozen():
    protocol.assert_protocol_integrity()
    assert protocol.TRAIN_POLICY_SEEDS == (71003, 71021, 71039)
    assert protocol.VALIDATION_POLICY_SEEDS == (71057, 71079)
    assert set(protocol.TRAIN_EVENT_SEEDS).isdisjoint(
        protocol.VALIDATION_EVENT_SEEDS)
    assert set(protocol.AUDIT_EVENT_SEEDS).isdisjoint(
        {*protocol.TRAIN_EVENT_SEEDS, *protocol.VALIDATION_EVENT_SEEDS})
    assert protocol.SWITCH_WINDOW_STEPS == 8
    assert protocol.SWITCH_SAMPLE_WEIGHT == 8.0
    assert protocol.CLASSIFICATION_LOSS_WEIGHT == 0.10
    assert protocol.FILTER_CONFIG == {
        "hazard_rate": 0.002,
        "evidence_scale": 1.0,
        "posterior_decay": 0.98,
    }


def test_v16_scheduler_dependency_graph_and_resources():
    rows = submitter.candidates("high")
    assert len(rows) == 7
    assert len({signature for signature, _, _ in rows}) == 7
    training = rows[0][1]
    assert training["vram"] == 1024
    assert training["ram_mb"] == 4096
    assert "ckpt_dir" in training
    audits = [spec for _, spec, _ in rows[1:6]]
    assert all(spec["vram"] == 0 for spec in audits)
    assert all("ckpt_dir" not in spec for spec in audits)
    assert rows[-1][1]["vram"] == 0


def test_v16_registration_data_is_waited_for_and_staged():
    data_files = {
        Path(path) for path in submitter.registration_data_files()
    }
    for _, spec, _ in submitter.candidates("high"):
        wait_files = {Path(path) for path in spec["wait_for_files"]}
        stage_roots = [Path(path) for path in spec["stage_input_paths"]]
        assert data_files <= wait_files
        assert all(
            any(path == root or path.is_relative_to(root) for root in stage_roots)
            for path in data_files
        )
