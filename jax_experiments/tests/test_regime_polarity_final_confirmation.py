"""Protocol tests for the frozen-estimator final confirmation."""
from __future__ import annotations

from jax_experiments.analysis import (
    regime_polarity_expected_action_system_id as development,
)
from jax_experiments.analysis import (
    regime_polarity_final_confirmation as protocol,
)
from jax_experiments.analysis import regime_polarity_confirmation as prior
from jax_experiments.analysis import (
    run_regime_polarity_final_confirmation_audit as audit,
)
from scripts import submit_regime_polarity_final_confirmation as submitter


def test_estimator_is_byte_frozen():
    protocol.validate_frozen_estimator()
    assert (
        protocol.file_record(protocol.MODEL_MANIFEST)
        == protocol.FROZEN_MODEL_MANIFEST_RECORD
    )
    assert (
        protocol.file_record(protocol.MODEL_PATH)
        == protocol.FROZEN_MODEL_PARAMETER_RECORD
    )


def test_policy_and_event_splits_are_fresh():
    development_policy_seeds = (
        set(development.TRAIN_CONTROLLER_SEEDS)
        | set(development.VALIDATION_CONTROLLER_SEEDS)
        | set(development.TEST_CONTROLLER_SEEDS)
    )
    prior_event_seeds = {
        91_001, 91_002, 91_003,
        92_001, 92_002,
        93_001, 93_002,
        94_001, 94_002, 94_003,
    }
    assert len(protocol.TRAINING_SEEDS) == 5
    assert set(protocol.TRAINING_SEEDS).isdisjoint(
        development_policy_seeds)
    assert set(protocol.TRAINING_SEEDS).isdisjoint(prior.TRAINING_SEEDS)
    assert len(protocol.AUDIT_EVENT_SEEDS) == 3
    assert set(protocol.AUDIT_EVENT_SEEDS).isdisjoint(prior_event_seeds)
    assert protocol.FINAL_TOTAL_STEPS == 5_600_000
    assert protocol.FINAL_UPDATE_COUNT == 350_000


def test_final_scheduler_graph_is_file_gated():
    training = submitter.candidates("training", "high")
    audits = submitter.candidates("audit", "high")
    analysis = submitter.candidates("analysis", "high")
    assert len(training) == 10
    assert len(audits) == 5
    assert len(analysis) == 1
    signatures = [row[0] for row in training + audits + analysis]
    assert len(signatures) == len(set(signatures))

    for _, spec, _ in training:
        assert spec["vram"] == submitter.MEASURED_VRAM_MB == 2300
        assert "jtl311linux" not in spec["allowed_nodes"]
        assert spec["resume_managed_by_cmd"] is True
        assert spec["ckpt_glob"] == "train_state.pkl"
        assert spec["wait_for_files"] == [
            str(protocol.MODEL_MANIFEST),
            str(protocol.MODEL_PATH),
        ]
        assert spec["stage_input_paths"] == [
            str(protocol.MODEL_MANIFEST.parent),
        ]
        assert str(protocol.RUN_ROOT) in spec["ckpt_dir"]
        assert "JAX_PLATFORMS=cuda" in spec["cmd"]
        assert "auto-adopt" not in spec["cmd"]

    for _, spec, _ in audits:
        assert spec["vram"] == 0
        assert spec["allowed_nodes"] == submitter.CPU_NODES
        assert len(spec["wait_for_files"]) == 10
        assert spec["stage_input_paths"][0] == str(
            protocol.MODEL_MANIFEST.parent)
        assert "JAX_PLATFORMS=cpu" in spec["cmd"]
        assert "auto-adopt" not in spec["cmd"]

    assert len(analysis[0][1]["wait_for_files"]) == 5
    assert "jtl311linux" not in analysis[0][1]["allowed_nodes"]


def test_final_audit_forbids_privileged_online_inputs():
    identity = audit._identity(protocol.TRAINING_SEEDS[0])
    assert identity["online_inputs"] == [
        "observation",
        "commanded_action",
        "next_observation",
    ]
    assert identity["online_forbidden"] == [
        "mode_id",
        "action_gain",
        "executed_action",
        "switch_clock",
    ]
    assert set(audit.ARMS) == {
        "robust",
        "oracle",
        "learned_soft",
        "learned_map",
    }
