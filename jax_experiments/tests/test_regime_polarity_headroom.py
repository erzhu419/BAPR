"""Protocol tests for the actuator-polarity oracle-headroom screen."""
from __future__ import annotations

from jax_experiments.analysis import regime_polarity_headroom as protocol
from jax_experiments.analysis import (
    run_regime_control_headroom_controller as common_controller,
)
from scripts import submit_regime_polarity_headroom as submitter


def test_protocol_is_multi_environment_multi_seed_and_equal_budget():
    assert protocol.FAMILY == "actuator_polarity"
    assert len(protocol.ENVS) == 4
    assert protocol.TRAINING_SEEDS == (8, 16, 24)
    assert len(protocol.AUDIT_EVENT_SEEDS) == 3
    assert protocol.ROLES == ("robust", "oracle")
    assert protocol.FINAL_TOTAL_STEPS == 5_600_000
    assert protocol.FINAL_UPDATE_COUNT == 350_000
    assert protocol.MIN_RELATIVE_GAIN == 0.15
    assert protocol.MIN_PASSING_ENVS == 3


def test_controller_command_uses_true_or_zero_context_only():
    original = common_controller.protocol
    common_controller.protocol = protocol
    try:
        robust = common_controller.training_command(
            "HalfCheetah-v2", "robust", 8)
        oracle = common_controller.training_command(
            "HalfCheetah-v2", "oracle", 8)
    finally:
        common_controller.protocol = original
    assert robust[robust.index("--regime_context_source") + 1] == "robust"
    assert oracle[oracle.index("--regime_context_source") + 1] == "oracle"
    assert robust[robust.index("--stochastic_mode_family") + 1] == (
        "actuator_polarity")
    assert robust[robust.index("--max_iters") + 1] == "1400"


def test_scheduler_graph_is_batched_gated_and_excludes_jtl311():
    training = submitter.candidates("training", "high")
    audits = submitter.candidates("audit", "high")
    analysis = submitter.candidates("analysis", "high")
    assert len(training) == 24
    assert len(audits) == 24
    assert len(analysis) == 1
    signatures = [row[0] for row in training + audits + analysis]
    assert len(signatures) == len(set(signatures))

    for _, spec, _ in training:
        assert spec["vram"] == submitter.MEASURED_VRAM_MB
        assert spec["allowed_nodes"] == submitter.GPU_NODES
        assert "jtl311linux" not in spec["allowed_nodes"]
        assert "JAX_PLATFORMS=cuda" in spec["cmd"]
        assert spec["resume_managed_by_cmd"] is True
    for _, spec, _ in audits:
        assert spec["vram"] == 0
        assert spec["cpu"] == 32
        assert spec["allowed_nodes"] == submitter.CPU_NODES
        assert len(spec["wait_for_files"]) == 4
    assert len(analysis[0][1]["wait_for_files"]) == 24
