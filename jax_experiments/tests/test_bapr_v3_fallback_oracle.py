"""Tests for the privileged fallback-oracle capacity protocol."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from jax_experiments.analysis import bapr_v3_fallback_oracle as protocol


def test_fallback_oracle_preserves_committed_decision():
    assert protocol.select_controller(2, False, 0, (0, 4, 2, 3)) == 2


def test_fallback_oracle_uses_true_mode_only_on_fallback():
    mapping = (0, 4, 2, 3)
    assert protocol.select_controller(-1, True, 0, mapping) == 0
    assert protocol.select_controller(-1, True, 1, mapping) == 4


def test_submission_declares_scheduler_dependencies():
    path = Path(__file__).resolve().parents[2] / "scripts" \
        / "submit_bapr_v3_fallback_oracle.py"
    spec = importlib.util.spec_from_file_location("_fallback_oracle_submit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    _, evaluation, _ = module.evaluation_spec(6100, "high")
    assert evaluation["require_node"] == "jtl311linux"
    assert "slurm" not in evaluation["cmd"].lower()
    assert "auto-adopt" not in evaluation["cmd"].lower()
    _, analysis, _ = module.analysis_spec("high")
    assert analysis["wait_for_files"] == [
        str(protocol.group_path(seed))
        for seed in protocol.DEVELOPMENT_EVENT_SEEDS
    ]

