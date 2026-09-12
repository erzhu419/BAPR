"""Tests for the equal-per-controller adapter upper-bound protocol."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from jax_experiments.analysis import analyze_regime_adapter_equal_controller
from jax_experiments.analysis import regime_adapter_confirmation
from jax_experiments.analysis import regime_adapter_equal_controller as protocol
from jax_experiments.analysis import regime_adapter_fork
from jax_experiments.analysis import (
    run_regime_adapter_equal_controller_branch as runner,
)
from scripts import submit_regime_adapter_equal_controller as submitter


def test_protocol_restarts_once_at_common_source_boundary() -> None:
    assert protocol.TRAINING_SEEDS == (16, 24, 32, 40)
    assert protocol.DELTA == 0.5
    assert protocol.SOURCE_NEXT_ITERATION == 1400
    assert protocol.FINAL_NEXT_ITERATION == 2100
    assert protocol.ADDITIONAL_ITERS == 700
    assert protocol.PER_CONTROLLER_POST_FORK_STEPS == 2_800_000
    assert protocol.BANK_AGGREGATE_POST_FORK_STEPS == 11_200_000
    assert protocol.BANK_AGGREGATE_TOTAL_STEPS == 16_800_000
    assert set(protocol.EVENT_SEEDS).isdisjoint(
        regime_adapter_confirmation.EVENT_SEEDS)

    command = runner._training_command(
        16, 0, protocol.run_dir(16, 0))
    assert command[command.index("--max_iters") + 1] == "2100"
    assert command[command.index("--min_resume_iteration") + 1] == "1400"
    assert "1575" not in command


def test_retry_boundaries_and_replay_probe_are_strict() -> None:
    assert runner._expected_steps_at_resume(1400) == 5_600_000
    assert runner._expected_steps_at_resume(1749) == 6_996_000
    assert runner._expected_steps_at_resume(2100) == 8_400_000
    try:
        runner._expected_steps_at_resume(1399)
    except ValueError:
        pass
    else:
        raise AssertionError("pre-source resume boundary was accepted")

    with tempfile.TemporaryDirectory() as directory:
        valid = Path(directory) / "valid.npz"
        corrupt = Path(directory) / "corrupt.npz"
        np.savez(valid, obs=np.zeros((0, 1), dtype=np.float32))
        corrupt.write_bytes(b"")
        assert runner._replay_is_readable(valid) is True
        assert runner._replay_is_readable(corrupt) is False


def test_scheduler_graph_is_file_gated_and_allows_node007() -> None:
    training = submitter.candidates("train", "high")
    audits = submitter.candidates("audit", "high")
    analysis = submitter.candidates("analysis", "high")
    assert len(training) == 16
    assert len(audits) == 20
    assert len(analysis) == 1
    signatures = [row[0] for row in training + audits + analysis]
    assert len(signatures) == len(set(signatures))

    for _, spec, manifest in training:
        assert spec["vram"] == submitter.MEASURED_VRAM_MB == 2300
        assert spec["allowed_nodes"] == submitter.GPU_NODES
        assert "node007" in spec["allowed_nodes"]
        assert "preferred_node" not in spec
        assert "require_node" not in spec
        assert spec["resume_managed_by_cmd"] is True
        assert spec["reroute_on_node_down"] is True
        assert len(spec["wait_for_files"]) == 4
        expected_run_dir = protocol.run_dir(
            int(manifest.parent.parent.name.removeprefix("seed_")),
            int(manifest.parent.name.removeprefix("mode_")),
        )
        assert spec["ckpt_dir"] == str(expected_run_dir)
        assert spec["ckpt_glob"] == "checkpoints/train_state.pkl"
    for _, spec, _ in audits:
        assert spec["vram"] == 0
        assert spec["cpu"] == 32
        assert spec["allowed_nodes"] == submitter.CPU_NODES
        assert len(spec["wait_for_files"]) == (
            len(regime_adapter_fork.required_bundle_paths(
                regime_adapter_fork.robust_bundle_dir(16)))
            + len(protocol.required_bundle_paths(16)))
        assert len(spec["stage_input_paths"]) == 5
        assert spec["allow_cpu_training"] is True
    assert len(analysis[0][1]["wait_for_files"]) == 20


def test_gate_uses_four_training_seeds_and_rejects_fixed_shortcuts() -> None:
    original_validate = analyze_regime_adapter_equal_controller.validate_audit
    original_seed_value = (
        analyze_regime_adapter_equal_controller._seed_value)
    original_seed_mode_values = (
        analyze_regime_adapter_equal_controller._seed_mode_values)

    def fake_seed_value(seed: int, case: str, metric: str) -> float:
        del seed
        if metric == "switching_termination":
            return 0.0
        if case == "identity_adapter":
            return 120.0
        if case.startswith("fixed_adapter_"):
            return 105.0
        if case == "frozen_base":
            return 80.0
        return 100.0

    def fake_mode_values(seed: int, case: str) -> np.ndarray:
        del seed
        controller = int(case.rsplit("_", 1)[1])
        values = np.full(4, 100.0)
        values[controller] = 120.0
        return values

    analyze_regime_adapter_equal_controller.validate_audit = lambda *_: {}
    analyze_regime_adapter_equal_controller._seed_value = fake_seed_value
    analyze_regime_adapter_equal_controller._seed_mode_values = (
        fake_mode_values)
    try:
        payload = analyze_regime_adapter_equal_controller.analyze()
    finally:
        analyze_regime_adapter_equal_controller.validate_audit = (
            original_validate)
        analyze_regime_adapter_equal_controller._seed_value = (
            original_seed_value)
        analyze_regime_adapter_equal_controller._seed_mode_values = (
            original_seed_mode_values)

    assert payload["paired_results"]["identity_minus_robust_switching"][
        "n_training_seeds"] == 4
    assert payload["decision_gate"]["pass"] is True
    assert payload["diagonal_optimal_modes"] == 16
    assert payload["decision"] == (
        "replace_independent_critics_with_shared_all_mode_backbone")
