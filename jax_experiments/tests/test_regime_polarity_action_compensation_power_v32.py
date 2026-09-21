"""Focused tests for the prospective V32 ten-seed confirmation."""
from __future__ import annotations

import pickle

import numpy as np
import pytest
from flax import nnx

from jax_experiments.analysis import (
    regime_polarity_action_compensation_power_confirmation_v32 as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_model_v5 as estimator_model,
)
from jax_experiments.analysis import (
    run_regime_polarity_action_compensation_power_audit_v32 as audit,
)
from scripts import (
    submit_regime_polarity_action_compensation_power_v32 as submit,
)


def test_v32_freezes_ten_new_seeds_and_equal_interaction_budgets():
    protocol.assert_protocol_integrity()
    assert len(protocol.TRAINING_SEEDS) == 10
    assert len(set(protocol.TRAINING_SEEDS)) == 10
    assert protocol.REFERENCE_MODE == 0
    assert protocol.SOURCE_TOTAL_STEPS == 5_600_000
    assert protocol.FINAL_TOTAL_STEPS == 8_400_000
    assert protocol.MAX_ITERS == 2100
    assert protocol.REQUIRED_SEED_WINS == 8
    assert protocol.REQUIRED_EVENT_WINS == 24


def test_v32_uses_failed_v31_only_for_prospective_power_planning():
    basis = protocol._pilot_power_basis()
    assert basis["v31_confirmation_pass"] is False
    assert basis["v31_not_pooled_with_v32"] is True
    assert all(
        row["planned_n"] == 10
        and row["estimated_two_sided_power"] >= 0.90
        for row in basis["comparisons"].values()
    )


def test_v32_dag_has_50_gpu_trainers_and_file_gated_cpu_tasks():
    rows = submit.candidates("high")
    specs = [spec for _, spec, _ in rows]
    gpu = [spec for spec in specs if spec["vram"] > 0]
    cpu = [spec for spec in specs if spec["vram"] == 0]
    assert len(rows) == 61
    assert len(gpu) == 50
    assert len(cpu) == 11
    assert all(spec["allowed_nodes"] == submit.GPU_NODES for spec in gpu)
    assert all(spec["allowed_nodes"] == submit.CPU_NODES for spec in cpu)
    assert all("local" not in spec["allowed_nodes"] for spec in specs)
    assert max(spec["vram"] for spec in gpu) == 2600


def test_v32_commands_use_only_v32_entry_points():
    commands = [spec["cmd"] for _, spec, _ in submit.candidates("high")]
    assert all("_v31" not in command for command in commands)
    assert sum("power_audit_v32" in command for command in commands) == 10
    assert sum("power_v32" in command for command in commands) == 1


def test_v32_audits_wait_for_compact_outputs_not_checkpoints():
    rows = submit.candidates("high")
    audits = [
        spec for signature, spec, _ in rows if "/audit/seed-" in signature
    ]
    assert len(audits) == 10
    for spec in audits:
        assert all("checkpoints" not in path for path in spec["wait_for_files"])
        assert all("train_state.pkl" not in path for path in spec["wait_for_files"])


def test_v32_registration_freezes_non_pooled_claim_and_gates():
    payload = protocol.registration_payload()
    boundary = payload["frozen_boundary"]
    assert boundary["v31_failed_result_preserved"] is True
    assert boundary["v31_results_not_pooled"] is True
    assert boundary["ten_new_policy_seeds"] is True
    gate = payload["decision_gate"]
    assert gate["required_seed_wins_of_10"] == 8
    assert gate["required_event_wins_of_30"] == 24


def test_v32_audit_accepts_publisher_state_and_aliases_reference(
    tmp_path, monkeypatch,
):
    state_path = tmp_path / "state.pkl"
    with state_path.open("wb") as handle:
        pickle.dump(nnx.State({"x": nnx.Param(np.array([1.0]))}), handle)
    assert isinstance(audit.load_policy_tree(state_path), nnx.State)

    invalid_path = tmp_path / "invalid.pkl"
    with invalid_path.open("wb") as handle:
        pickle.dump([1.0], handle)
    with pytest.raises(ValueError, match="expected policy parameter tree"):
        audit.load_policy_tree(invalid_path)

    action = object()
    stack = {"actions": {"canonical_reference": action}}
    monkeypatch.setattr(
        audit,
        "_FROZEN_LOAD_POLICY_STACKS",
        lambda seed: (stack, {"actions": {}}, {"seed": seed}),
    )
    loaded, _, baselines = audit.load_policy_stacks(protocol.TRAINING_SEEDS[0])
    assert loaded["actions"]["specialist_0"] is action
    assert baselines == {"seed": protocol.TRAINING_SEEDS[0]}


def test_v32_audit_binds_registered_v5_estimator():
    audit.install_runtime_bindings()
    assert audit.frozen.compensation.v5_model is estimator_model
