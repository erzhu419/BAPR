"""Focused tests for the V21 full-state confirmation protocol."""
from __future__ import annotations

from jax_experiments.analysis import (
    regime_polarity_full_state_final_confirmation_v21 as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_full_state_specialist_v21 as specialist_protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_full_state_confirmation_baseline_v21 as baseline,
)
from jax_experiments.analysis import (
    run_regime_polarity_full_state_specialist_v21 as specialist,
)
from scripts import submit_regime_polarity_full_state_confirmation_v21 as submit


def test_v21_uses_five_new_policy_seeds_and_disjoint_event_splits():
    protocol.assert_protocol_integrity()
    assert len(protocol.TRAINING_SEEDS) == 5
    splits = (
        set(protocol.CALIBRATION_EVENT_SEEDS),
        set(protocol.STATIONARY_HOLDOUT_EVENT_SEEDS),
        set(protocol.SWITCHING_EVENT_SEEDS),
    )
    assert not any(
        left & right
        for index, left in enumerate(splits)
        for right in splits[index + 1:]
    )
    assert all(set(sequence) == set(protocol.MODES)
               for sequence in protocol.SWITCHING_SCHEDULES.values())


def test_v21_dag_has_55_gpu_producers_and_61_total_tasks():
    rows = submit.candidates("high")
    specs = [spec for _, spec, _ in rows]
    assert len(rows) == 61
    assert sum(spec["vram"] > 0 for spec in specs) == 55
    assert sum("/audit/" in spec["signature"] for spec in specs) == 5
    assert sum(spec["signature"].endswith("/analysis") for spec in specs) == 1


def test_v21_gpu_jobs_exclude_local_and_have_measured_vram_requests():
    gpu_specs = [
        spec for _, spec, _ in submit.candidates("high") if spec["vram"] > 0
    ]
    assert gpu_specs
    assert all("local" not in spec["allowed_nodes"] for spec in gpu_specs)
    assert all(spec["allowed_nodes"] == submit.GPU_NODES for spec in gpu_specs)
    assert all(0 < spec["vram"] <= 2600 for spec in gpu_specs)


def test_v21_specialists_wait_for_source_and_sync_no_checkpoint():
    seed = protocol.TRAINING_SEEDS[0]
    spec = submit.specialist_spec(seed, 0, "high")
    assert all(str(path) in spec["wait_for_files"]
               for path in protocol.source_required_paths(seed))
    assert spec["result_dir"] == str(protocol.specialist_bundle_dir(seed, 0))
    assert "checkpoints" not in spec["result_dir"]
    assert all("train_state.pkl" not in str(path)
               for path in protocol.specialist_required_paths(seed, 0))


def test_v21_audits_are_cpu_only_and_wait_for_all_policy_outputs():
    seed = protocol.TRAINING_SEEDS[0]
    spec = submit.audit_spec(seed, "high")
    assert spec["vram"] == 0
    assert spec["allowed_nodes"] == submit.CPU_NODES
    for mode in protocol.MODES:
        assert all(str(path) in spec["wait_for_files"]
                   for path in protocol.specialist_required_paths(seed, mode))
    for method in protocol.TRAINED_METHODS:
        assert all(str(path) in spec["wait_for_files"]
                   for path in protocol.bundle_required_paths(method, seed))
    for slot in protocol.SAC_REPLICA_SLOTS:
        assert all(str(path) in spec["wait_for_files"]
                   for path in protocol.bundle_required_paths(
                       "sac_replica", seed, slot))


def test_v21_freezes_final_period1_specialists_without_selection():
    seed = protocol.TRAINING_SEEDS[0]
    expected = specialist.expected_config(
        specialist_protocol.SPECIALIST_VARIANT, seed, 0)
    assert specialist_protocol.actor_update_period(
        specialist_protocol.SPECIALIST_VARIANT) == 1
    assert specialist_protocol.select_best_validation(
        specialist_protocol.SPECIALIST_VARIANT) is False
    assert expected["sac_actor_update_period"] == 1
    assert expected["sac_select_best_eval"] is False
    assert expected["max_iters"] == protocol.SPECIALIST_FINAL_NEXT_ITERATION
    assert expected["stochastic_mode_fixed_id"] == 0


def test_v21_baseline_budget_matches_registered_v18_convention():
    seed = protocol.TRAINING_SEEDS[0]
    for kind, slot in (
        ("sac_replica", 1),
        ("escp_recurrent", None),
        ("resac_b0", None),
    ):
        expected = baseline.expected_config(kind, seed, slot)
        assert expected["max_iters"] == 1400
        assert expected["samples_per_iter"] == 4000
        assert expected["updates_per_iter"] == 250
        assert expected["stochastic_mode_family"] == protocol.FAMILY
        assert expected["stochastic_mode_dwell_steps"] == 250


def test_v21_development_gate_uses_only_valid_full_state_final_arm():
    evidence = protocol._development_evidence()
    assert evidence["v20_seed_passes"] == 3
    assert evidence["v20_stationary_mode_passes"] == 12
    assert evidence["v20_switching_event_wins"] == 9
    assert evidence["v20_registered_candidate_selected"] is None
    assert evidence["selection_arm_execution_deviation_recorded"] is True
