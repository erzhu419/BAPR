"""Protocol tests for the checkpoint-only policy-ensemble diagnostic."""
from __future__ import annotations

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_policy_ensemble as protocol,
)
from jax_experiments.analysis.run_regime_polarity_policy_ensemble_audit import (
    _reduce_actions,
)
from jax_experiments.analysis.analyze_regime_polarity_policy_ensemble import (
    _recommendation,
)
from scripts import submit_regime_polarity_policy_ensemble as submitter


def test_controller_groups_and_event_seeds_are_separate():
    assert protocol.GROUPS == ("development", "final")
    assert len(protocol.controller_seeds("development")) == 5
    assert len(protocol.controller_seeds("final")) == 5
    assert set(protocol.controller_seeds("development")).isdisjoint(
        protocol.controller_seeds("final"))
    assert len(protocol.EVENT_SEEDS) == 3
    assert set(protocol.EVENT_SEEDS).isdisjoint({
        91_001, 91_002, 91_003,
        95_001, 95_002, 95_003,
        96_501, 96_502, 96_503,
    })
    assert len(protocol.arms("development")) == 16
    assert len(protocol.arms("final")) == 16


def test_action_reductions_are_deterministic_and_bounded():
    actions = np.asarray([
        [-1.0, 0.5, 0.2],
        [0.0, -0.5, 0.7],
        [1.0, 0.0, -0.2],
    ], dtype=np.float32)
    np.testing.assert_allclose(
        _reduce_actions(actions, "individual", 1), actions[1])
    np.testing.assert_allclose(
        _reduce_actions(actions, "mean"), [0.0, 0.0, 0.23333333])
    np.testing.assert_allclose(
        _reduce_actions(actions, "median"), [0.0, 0.0, 0.2])
    for reduction in ("mean", "median"):
        selected = _reduce_actions(actions, reduction)
        assert np.all(selected >= -1.0)
        assert np.all(selected <= 1.0)


def test_scheduler_graph_is_cpu_only_batched_and_file_gated():
    audits = submitter.candidates("audit", "high")
    analyses = submitter.candidates("analysis", "high")
    assert len(audits) == 6
    assert len(analyses) == 1
    signatures = [row[0] for row in audits + analyses]
    assert len(signatures) == len(set(signatures))
    for _, spec, _ in audits:
        assert spec["vram"] == 0
        assert spec["cpu"] == 32
        assert spec["allowed_nodes"] == submitter.CPU_NODES
        assert len(spec["wait_for_files"]) == 42
        assert len(spec["stage_input_paths"]) == 11
        assert "JAX_PLATFORMS=cpu" in spec["cmd"]
        assert "auto-adopt" not in spec["cmd"]
        assert "jtl311linux" not in spec["allowed_nodes"]
    assert len(analyses[0][1]["wait_for_files"]) == 6
    assert analyses[0][1]["allowed_nodes"] == submitter.CPU_NODES


def test_learned_arms_forbid_privileged_online_inputs():
    identity = protocol.identity("final", protocol.EVENT_SEEDS[0])
    assert identity["development_only"] is True
    assert identity["online_inputs"] == [
        "observation",
        "commanded_action",
        "next_observation",
    ]
    assert identity["online_forbidden_for_learned_arms"] == [
        "mode_id",
        "action_gain",
        "executed_action",
        "switch_clock",
    ]
    learned = [
        arm for arm in protocol.arms("final")
        if arm.context_kind == "learned"
    ]
    assert [arm.label for arm in learned] == [
        "learned_mean", "learned_median"]


def test_recommendation_selects_the_stronger_passing_reduction():
    groups = {
        group: {
            "ensemble_checks": {
                "mean": {"learned_pass": True, "oracle_pass": True},
                "median": {"learned_pass": True, "oracle_pass": True},
            },
            "arm_summaries": {
                "learned_mean": {"switching_mean": 1600.0},
                "learned_median": {
                    "switching_mean": 2000.0 if group == "development"
                    else 1900.0,
                },
            },
        }
        for group in protocol.GROUPS
    }
    recommendation = _recommendation(groups)
    assert "(median)" in recommendation
    assert "individual robust-controller distribution" in recommendation
