"""Tests for the posterior-discrete fallback-fill protocol."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from jax_experiments.analysis import (
    bapr_v3_posterior_discrete_selector as protocol,
)


def _table():
    return {
        "controllers": [4, 0, 2, 3],
        "utility_matrix": [
            [10.0, 20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0, 0.0],
            [10.0, 0.0, 20.0, 0.0],
            [10.0, 0.0, 0.0, 20.0],
        ],
    }


def _patch_utility_matrix(table):
    return np.asarray(table["utility_matrix"]), tuple(table["controllers"])


def test_expected_utility_selector_uses_real_controller(monkeypatch=None):
    original = protocol.prior.utility.utility_matrix
    protocol.prior.utility.utility_matrix = _patch_utility_matrix
    try:
        selected, diagnostics = protocol.select_expected_utility_controller(
            np.asarray([1.0, 0.0, 0.0, 0.0]), _table())
        assert selected == 0
        assert diagnostics["best_expected_advantage"] == 10.0
    finally:
        protocol.prior.utility.utility_matrix = original


def test_expected_utility_selector_defaults_to_robust_on_tie():
    original = protocol.prior.utility.utility_matrix
    protocol.prior.utility.utility_matrix = _patch_utility_matrix
    try:
        selected, _ = protocol.select_expected_utility_controller(
            np.full((4,), 0.25), _table())
        assert selected == 4
    finally:
        protocol.prior.utility.utility_matrix = original


def test_development_and_confirmation_seeds_are_disjoint():
    assert not set(protocol.DEVELOPMENT_EVENT_SEEDS).intersection(
        protocol.SEALED_CONFIRMATION_EVENT_SEEDS)


def test_submission_is_scheduler_only_and_pinned_to_artifacts():
    path = Path(__file__).resolve().parents[2] / "scripts" \
        / "submit_bapr_v3_posterior_discrete_selector.py"
    spec = importlib.util.spec_from_file_location("_posterior_discrete_submit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    _, task, _ = module.evaluation_spec(6100, "high")
    assert task["require_node"] == "jtl311linux"
    assert "slurm" not in task["cmd"].lower()
    assert "auto-adopt" not in task["cmd"].lower()
    _, analysis, _ = module.analysis_spec("high")
    assert analysis["wait_for_files"] == [
        str(protocol.group_path(seed))
        for seed in protocol.DEVELOPMENT_EVENT_SEEDS
    ]
