"""Registration tests for the causal-fallback final comparison."""
from __future__ import annotations

from pathlib import Path

from jax_experiments.analysis import (
    regime_polarity_fallback_final_comparison_v1 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_fallback_final_baseline_v1 as baseline,
)
from scripts import submit_regime_polarity_fallback_final_comparison_v1 as submit


def test_final_splits_and_model_seeds_are_unique():
    assert len(protocol.TRAINING_SEEDS) == len(set(protocol.TRAINING_SEEDS)) == 5
    assert len(protocol.FINAL_EVENT_SEEDS) \
        == len(set(protocol.FINAL_EVENT_SEEDS)) == 5
    protocol.assert_split_integrity()


def test_resac_preserves_positive_regularization_and_original_lr():
    command = baseline.training_command("resac", protocol.TRAINING_SEEDS[0])
    assert command[command.index("--weight_reg") + 1] == "0.01"
    assert command[command.index("--beta_ood") + 1] == "0.01"
    assert command[command.index("--lr") + 1] == "1e-05"
    assert command[command.index("--beta") + 1] == "-2.0"


def test_registration_covers_every_existing_source():
    paths = protocol.registration_source_paths()
    assert paths
    assert all(path.is_file() for path in paths)
    payload = protocol.registration_payload()
    assert payload["identity"]["fallback_config"] == {
        "name": "evidence_1p0_k1",
        "contradiction_threshold": 1.0,
        "stable_steps": 1,
        "enter_confidence": 0.6,
        "exit_confidence": 0.9,
    }
    assert len(payload["source_records"]) == len(set(paths))
    protocol.validate_registration()


def test_scheduler_graph_is_batched_and_node_restricted():
    rows = submit.candidates("all", "high")
    assert len(rows) == 41
    assert len({signature for signature, _, _ in rows}) == 41
    training = [spec for signature, spec, _ in rows if "/train/" in signature]
    audits = [spec for signature, spec, _ in rows if "/audit/" in signature]
    assert len(training) == 20
    assert len(audits) == 20
    assert all(spec["allowed_nodes"] == ["jtl311linux"] for spec in training)
    assert {spec["vram"] for spec in training} == {2300, 2400}
    assert all(spec["allow_gpu_over_one_third"] for spec in training)
    artifact_sources = {
        str(path) for path in submit._registration_artifact_sources()
    }
    registration_dirs = set(submit._registration_stage_inputs())
    assert all(
        any(
            source == directory or source.startswith(directory + "/")
            for directory in registration_dirs
        )
        for source in artifact_sources
    )
    assert all(Path(path).is_dir() for path in registration_dirs)
    assert all(
        registration_dirs <= set(spec["stage_input_paths"])
        for _, spec, _ in rows
    )
    assert all(spec["vram"] == 0 for spec in audits)
    assert all(spec["allowed_nodes"] == submit.CPU_NODES for spec in audits)


def test_bapr_audit_identity_excludes_privileged_online_inputs():
    identity = protocol.audit_identity("bapr", protocol.TRAINING_SEEDS[0])
    forbidden = set(identity["online_forbidden_for_bapr"])
    assert forbidden == {
        "mode_id", "action_gain", "executed_action", "switch_clock"}


if __name__ == "__main__":
    test_final_splits_and_model_seeds_are_unique()
    test_resac_preserves_positive_regularization_and_original_lr()
    test_registration_covers_every_existing_source()
    test_scheduler_graph_is_batched_and_node_restricted()
    test_bapr_audit_identity_excludes_privileged_online_inputs()
