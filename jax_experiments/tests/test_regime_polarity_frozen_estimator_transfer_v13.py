from __future__ import annotations

import sys
from pathlib import Path

from jax_experiments.analysis import (
    regime_polarity_frozen_estimator_transfer_v13 as protocol,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import submit_regime_polarity_frozen_estimator_transfer_v13 as submitter


def test_v13_splits_schedules_and_gate_are_frozen():
    protocol.assert_protocol_integrity()
    assert len(protocol.TRAINING_SEEDS) == 5
    assert len(protocol.EVENT_SEEDS) == 3
    assert protocol.DELAY_STEPS == 4
    assert protocol.REQUIRED_SEED_PASSES == 4
    assert protocol.MIN_CAUSAL_RETENTION == 0.70
    traces = {
        tuple(protocol.switching_sequence(seed, episode)
              for episode in range(protocol.SWITCHING_EPISODES))
        for seed in protocol.EVENT_SEEDS
    }
    assert len(traces) == len(protocol.EVENT_SEEDS)


def test_v13_is_checkpoint_only_and_stages_minimal_frozen_inputs():
    rows = submitter.candidates("high")
    assert len(rows) == 6
    assert len({signature for signature, _, _ in rows}) == 6
    assert all(int(spec["vram"]) == 0 for _, spec, _ in rows)
    assert all("ckpt_dir" not in spec for _, spec, _ in rows)
    for _, spec, _ in rows:
        assert "checkpoints" not in spec["result_dir"]
        assert "paper/" in spec["stage_excludes"]


def test_v13_registration_data_is_waited_for_and_staged():
    data_files = {
        path for path in protocol.registration_source_paths()
        if path.suffix != ".py"
    }
    for _, spec, _ in submitter.candidates("high"):
        wait_files = {Path(path) for path in spec["wait_for_files"]}
        stage_roots = [Path(path) for path in spec["stage_input_paths"]]
        assert data_files <= wait_files
        assert all(
            any(path == root or path.is_relative_to(root) for root in stage_roots)
            for path in data_files
        )
