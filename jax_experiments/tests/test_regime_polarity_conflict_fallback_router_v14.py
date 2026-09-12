from __future__ import annotations

import sys
from pathlib import Path

from jax_experiments.analysis import (
    regime_polarity_conflict_fallback_router_v14 as protocol,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import submit_regime_polarity_conflict_fallback_router_v14 as submitter


def test_v14_splits_and_causal_router_variants_are_frozen():
    protocol.assert_protocol_integrity()
    assert len(protocol.TRAINING_SEEDS) == 5
    assert len(protocol.EVENT_SEEDS) == 3
    assert protocol.POSTERIOR_EXIT_CONFIDENCE == 0.80
    assert protocol.CONFLICT_LOG_LIKELIHOOD_MARGIN == 3.0
    assert protocol.MIN_FALLBACK_ACTIONS_AFTER_CONFLICT == 1
    assert [
        protocol.confirm_steps_for_arm(arm)
        for arm in protocol.CANDIDATE_ARMS
    ] == [1, 2, 3]
    assert all(
        protocol.confirm_steps_for_arm(arm) == 0
        for arm in protocol.BASE_ARMS
    )


def test_v14_scheduler_is_checkpoint_only():
    rows = submitter.candidates("high")
    assert len(rows) == 6
    assert len({signature for signature, _, _ in rows}) == 6
    assert all(int(spec["vram"]) == 0 for _, spec, _ in rows)
    assert all("ckpt_dir" not in spec for _, spec, _ in rows)
    assert all("checkpoints" not in spec["result_dir"] for _, spec, _ in rows)


def test_v14_registration_data_is_waited_for_and_staged():
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
