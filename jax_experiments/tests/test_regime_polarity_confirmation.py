"""Protocol tests for the fresh polarity-headroom confirmation."""
from __future__ import annotations

from jax_experiments.analysis import regime_polarity_confirmation as protocol
from jax_experiments.analysis import regime_polarity_headroom as exploratory
from scripts import submit_regime_polarity_confirmation as submitter


def test_confirmation_is_fresh_halfcheetah_only():
    assert protocol.ENVS == ("HalfCheetah-v2",)
    assert len(protocol.TRAINING_SEEDS) == 5
    assert set(protocol.TRAINING_SEEDS).isdisjoint(
        exploratory.TRAINING_SEEDS)
    assert len(set(protocol.AUDIT_EVENT_SEEDS)) == 3
    assert protocol.MAX_ITERS == exploratory.MAX_ITERS
    assert protocol.FINAL_TOTAL_STEPS == 5_600_000
    assert protocol.FINAL_UPDATE_COUNT == 350_000


def test_confirmation_scheduler_graph_is_file_gated():
    training = submitter.candidates("training", "high")
    audits = submitter.candidates("audit", "high")
    analysis = submitter.candidates("analysis", "high")
    assert len(training) == 10
    assert len(audits) == 10
    assert len(analysis) == 1
    signatures = [row[0] for row in training + audits + analysis]
    assert len(signatures) == len(set(signatures))
    for _, spec, _ in training:
        assert spec["vram"] == submitter.MEASURED_VRAM_MB == 2300
        assert "jtl311linux" not in spec["allowed_nodes"]
        assert spec["resume_managed_by_cmd"] is True
        assert spec["ckpt_glob"] == "train_state.pkl"
        assert "JAX_PLATFORMS=cuda" in spec["cmd"]
        assert "auto-adopt" not in spec["cmd"]
    for _, spec, _ in audits:
        assert spec["vram"] == 0
        assert spec["allowed_nodes"] == submitter.CPU_NODES
        assert len(spec["wait_for_files"]) == 4
        assert "JAX_PLATFORMS=cpu" in spec["cmd"]
    assert len(analysis[0][1]["wait_for_files"]) == 10


def test_confirmation_gate_is_preregistered():
    assert protocol.MIN_RELATIVE_GAIN == 0.10
    assert protocol.MAX_TERMINATION_GAP == 0.05
    assert protocol.MIN_MODE_WINS == 3
    assert protocol.MIN_PASSING_ENVS == 1
    assert protocol.ROBUST_TRACE_CONTEXT_MODE_ID == -1
