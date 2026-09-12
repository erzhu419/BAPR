"""Focused protocol tests for the preregistered v18 comparison."""
from __future__ import annotations

from jax_experiments.analysis import (
    analyze_regime_polarity_v5_final_comparison_v18 as analyzer,
)
from jax_experiments.analysis import (
    regime_polarity_v5_final_comparison_v18 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_v5_final_baseline_v18 as producer,
)
from scripts import submit_regime_polarity_v5_final_comparison_v18 as submit


def test_v18_dag_has_30_gpu_producers_and_36_total_tasks():
    rows = submit.candidates("high")
    specs = [spec for _, spec, _ in rows]
    assert len(rows) == 36
    assert sum(spec["vram"] > 0 for spec in specs) == 30
    assert sum("/audit/" in spec["signature"] for spec in specs) == 5
    assert sum(spec["signature"].endswith("/analysis") for spec in specs) == 1


def test_v18_eval_tasks_are_cpu_only_and_wait_for_producers():
    spec = submit.audit_spec(protocol.TRAINING_SEEDS[0], "high")
    assert spec["vram"] == 0
    assert spec["allowed_nodes"] == submit.CPU_NODES
    assert all("node00" in node for node in spec["allowed_nodes"])
    for method in protocol.TRAINED_METHODS:
        assert str(protocol.bundle_manifest(method, protocol.TRAINING_SEEDS[0])) \
            in spec["wait_for_files"]
    for slot in protocol.SAC_REPLICA_SLOTS:
        assert str(protocol.bundle_manifest(
            "sac_replica", protocol.TRAINING_SEEDS[0], slot)) \
            in spec["wait_for_files"]


def test_v18_producers_sync_only_evaluation_bundles():
    seed = protocol.TRAINING_SEEDS[0]
    spec = submit.producer_spec("sac_replica", seed, 1, "high")
    assert spec["result_dir"] == str(protocol.sac_bundle_dir(seed, 1))
    assert "checkpoints" not in spec["result_dir"]
    required = protocol.bundle_required_paths("sac_replica", seed, 1)
    assert all("replay_buffer" not in str(path) for path in required)
    assert all("train_state.pkl" not in str(path) for path in required)


def test_v18_training_seed_and_budget_are_fixed():
    seed = protocol.TRAINING_SEEDS[0]
    assert protocol.replica_training_seed(seed, 1) == seed + 1_000_000
    assert protocol.replica_training_seed(seed, 4) == seed + 4_000_000
    for kind, slot in (
        ("sac_replica", 1),
        ("escp_recurrent", None),
        ("resac_b0", None),
    ):
        expected = producer.expected_config(kind, seed, slot)
        assert expected["max_iters"] == 1400
        assert expected["samples_per_iter"] == 4000
        assert expected["updates_per_iter"] == 250
        assert expected["stochastic_mode_family"] == protocol.FAMILY
        assert expected["stochastic_mode_dwell_steps"] == 250


def test_v18_primary_comparison_gate_requires_ci_seed_and_event_wins():
    seeds = {}
    for seed in protocol.TRAINING_SEEDS:
        primary_events = {
            str(event): 110.0 for event in protocol.SWITCHING_EVENT_SEEDS}
        baseline_events = {
            str(event): 100.0 for event in protocol.SWITCHING_EVENT_SEEDS}
        seeds[str(seed)] = {
            "switching": {
                protocol.PRIMARY_ARM: {
                    "mean": 110.0,
                    "terminated_rate": 0.0,
                    "event_returns": primary_events,
                },
                "robust_sac": {
                    "mean": 100.0,
                    "terminated_rate": 0.0,
                    "event_returns": baseline_events,
                },
            }
        }
    result = analyzer._comparison(seeds, "robust_sac")
    assert result["pass"] is True
    assert result["seed_wins"] == 5
    assert result["event_wins"] == 15
    assert result["paired_difference_ci95"][0] > 0.0


def test_v18_splits_and_switching_schedules_are_disjoint_and_balanced():
    protocol.assert_protocol_integrity()
    splits = [
        set(protocol.CALIBRATION_EVENT_SEEDS),
        set(protocol.STATIONARY_HOLDOUT_EVENT_SEEDS),
        set(protocol.SWITCHING_EVENT_SEEDS),
    ]
    assert not any(
        left & right
        for index, left in enumerate(splits)
        for right in splits[index + 1:]
    )
    assert all(set(row) == set(protocol.MODES)
               for row in protocol.SWITCHING_SCHEDULES.values())
