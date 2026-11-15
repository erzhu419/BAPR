"""Focused tests for the preregistered V31 confirmation DAG."""
from __future__ import annotations

import pickle

import numpy as np
import pytest
from flax import nnx

from jax_experiments.analysis import (
    regime_polarity_action_compensation_confirmation_v31 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_action_compensation_confirmation_audit_v31_compat
    as audit_compat,
)
from scripts import (
    submit_regime_polarity_action_compensation_confirmation_v31 as submit,
)


def test_v31_freezes_new_seeds_mode0_and_equal_interaction_budgets():
    protocol.assert_protocol_integrity()
    assert protocol.REFERENCE_MODE == 0
    assert protocol.TRAINING_SEEDS == (87003, 87021, 87039, 87057, 87079)
    assert protocol.SOURCE_TOTAL_STEPS == 5_600_000
    assert protocol.FINAL_TOTAL_STEPS == 8_400_000
    assert protocol.MAX_ITERS == 2100
    assert protocol.replica_training_seed(87003, 0) == 87003


def test_v31_old_evidence_supports_preselected_canonical_reference():
    basis = protocol._old_mode0_basis()
    assert basis["selected_reference_mode"] == 0
    assert basis["all_five_old_policy_seeds_positive"] is True
    assert len(basis["rows"]) == 5
    assert all(row["mode0_minus_robust"] > 0.0
               for row in basis["rows"].values())


def test_v31_dag_has_25_gpu_trainers_and_file_gated_cpu_tasks():
    rows = submit.candidates("high")
    specs = [spec for _, spec, _ in rows]
    gpu = [spec for spec in specs if spec["vram"] > 0]
    cpu = [spec for spec in specs if spec["vram"] == 0]
    assert len(rows) == 31
    assert len(gpu) == 25
    assert len(cpu) == 6
    assert all(spec["allowed_nodes"] == submit.GPU_NODES for spec in gpu)
    assert all(spec["allowed_nodes"] == submit.CPU_NODES for spec in cpu)
    assert all("local" not in spec["allowed_nodes"] for spec in specs)
    assert max(spec["vram"] for spec in gpu) == 2600


def test_v31_reference_waits_for_source_and_audit_waits_for_compact_outputs():
    for seed in protocol.TRAINING_SEEDS:
        reference = submit.reference_spec(seed, "high")
        assert all(str(path) in reference["wait_for_files"]
                   for path in protocol.source_required_paths(seed))
        audit = submit.audit_spec(seed, "high")
        assert all(str(path) in audit["wait_for_files"]
                   for path in protocol.audit_required_paths(seed))
        assert all("checkpoints" not in path
                   for path in audit["wait_for_files"])
        assert all("train_state.pkl" not in path
                   for path in audit["wait_for_files"])


def test_v31_registration_has_no_result_driven_reference_selection():
    payload = protocol.registration_payload()
    boundary = payload["frozen_boundary"]
    assert boundary["reference_mode_selected_before_new_training"] is True
    assert boundary["no_new_reference_calibration"] is True
    assert boundary["v5_estimator_unchanged"] is True
    assert payload["budgets"]["canonical_compensation_training_path"][
        "total_unique_interactions"] == 8_400_000
    assert payload["decision_gate"]["primary_comparators"] == [
        protocol.ROBUST_LONG_ARM, protocol.ESCP_ARM, protocol.RESAC_ARM]


def test_v31_audit_compat_accepts_publisher_state_and_rejects_other_payloads(
    tmp_path,
):
    state_path = tmp_path / "state.pkl"
    with state_path.open("wb") as handle:
        pickle.dump(nnx.State({"x": nnx.Param(np.array([1.0]))}), handle)
    loaded = audit_compat.load_policy_tree(state_path)
    assert isinstance(loaded, nnx.State)

    invalid_path = tmp_path / "invalid.pkl"
    with invalid_path.open("wb") as handle:
        pickle.dump([1.0], handle)
    with pytest.raises(ValueError, match="expected policy parameter tree"):
        audit_compat.load_policy_tree(invalid_path)


def test_v31_audit_compat_aliases_canonical_reference_for_v28_helpers(
    monkeypatch,
):
    action = object()
    stack = {"actions": {"canonical_reference": action}}
    monkeypatch.setattr(
        audit_compat,
        "_FROZEN_LOAD_POLICY_STACKS",
        lambda seed: (stack, {"actions": {}}, {"seed": seed}),
    )
    loaded, _, baselines = audit_compat.load_policy_stacks(87003)
    assert loaded["actions"]["specialist_0"] is action
    assert baselines == {"seed": 87003}


def test_v31_audit_amendment_changes_execution_wiring_only():
    payload = audit_compat.amendment_payload()
    assert payload["scientific_protocol_changes"] == []
    assert len(payload["corrections"]) == 3
    assert payload["parent_registration"] == protocol.file_record(
        protocol.REGISTRATION_PATH)
