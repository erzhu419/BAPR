"""Tests for the frozen mode-head context-ablation diagnostic."""
from __future__ import annotations

import numpy as np

from jax_experiments.analysis import (
    analyze_regime_polarity_policy_distillation_context_ablation_v1 as analyzer,
)
from jax_experiments.analysis import (
    regime_polarity_policy_distillation_context_ablation_v1 as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_policy_distillation_control_confirmation_v2 as confirmation,
)
from jax_experiments.analysis import (
    regime_polarity_policy_distillation_control_v2 as development,
)
from jax_experiments.analysis import (
    run_regime_polarity_policy_distillation_context_ablation_audit_v1 as audit,
)
from scripts import (
    submit_regime_polarity_policy_distillation_context_ablation_v1 as submitter,
)


def test_split_is_new_and_candidate_is_frozen():
    protocol.validate_frozen_candidate()
    prior = {
        *development.TRAIN_EVENT_SEEDS,
        *development.DAGGER_EVENT_SEEDS,
        *development.SUPERVISED_VALIDATION_EVENT_SEEDS,
        *development.CONTROL_VALIDATION_EVENT_SEEDS,
        *development.AUDIT_EVENT_SEEDS,
        *development.SEALED_CONFIRMATION_EVENT_SEEDS,
        *confirmation.AUDIT_EVENT_SEEDS,
        *development.ensemble.EVENT_SEEDS,
    }
    assert len(protocol.AUDIT_EVENT_SEEDS) == 5
    assert set(protocol.AUDIT_EVENT_SEEDS).isdisjoint(prior)
    assert protocol.TEACHER_GROUP == "mode_heads"
    assert protocol.STUDENT_SEED == 1811


def test_arm_ladder_and_shuffled_maps_are_well_formed():
    assert len(protocol.ARM_LABELS) == 12
    assert len(set(protocol.ARM_LABELS)) == len(protocol.ARM_LABELS)
    for seed in protocol.AUDIT_EVENT_SEEDS:
        mapping = protocol.shuffled_mode_map(seed)
        assert sorted(mapping) == list(protocol.MODES)
        assert all(mode != mapping[mode] for mode in protocol.MODES)


def test_student_context_overrides_are_exact():
    posterior = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    seed = protocol.AUDIT_EVENT_SEEDS[0]
    assert np.allclose(
        audit._student_context("student_learned", 2, posterior, seed),
        posterior,
    )
    assert np.allclose(
        audit._student_context("student_uniform", 2, posterior, seed),
        np.full((4,), 0.25),
    )
    assert int(np.argmax(audit._student_context(
        "student_oracle", 2, posterior, seed))) == 2
    assert int(np.argmax(audit._student_context(
        "student_fixed_1", 2, posterior, seed))) == 1
    assert int(np.argmax(audit._student_context(
        "student_cyclic", 2, posterior, seed))) == 3
    assert int(np.argmax(audit._student_context(
        "student_shuffled", 2, posterior, seed))) == (
            protocol.shuffled_mode_map(seed)[2])


def test_event_cluster_comparison_is_paired():
    candidate = {
        seed: [2.0, 2.0] for seed in protocol.AUDIT_EVENT_SEEDS}
    reference = {
        seed: [1.0, 1.0] for seed in protocol.AUDIT_EVENT_SEEDS}
    result = analyzer._comparison(candidate, reference)
    assert result["event_seed_wins"] == 5
    assert result["episode_wins"] == 10
    assert result["cluster_count"] == 5
    assert result["cluster_95pct_t_interval"] == [1.0, 1.0]


def test_scheduler_graph_is_cpu_only_and_file_gated():
    audits = submitter.candidates("audit", "high")
    analyses = submitter.candidates("analysis", "high")
    assert len(audits) == 5
    assert len(analyses) == 1
    for _, spec, _ in audits:
        assert spec["vram"] == 0
        assert spec["cpu"] == 32
        assert spec["allowed_nodes"] == submitter.CPU_NODES
        assert "JAX_PLATFORMS=cpu" in spec["cmd"]
    assert len(analyses[0][1]["wait_for_files"]) == 5

