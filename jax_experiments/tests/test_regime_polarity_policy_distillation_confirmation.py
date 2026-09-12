"""Tests for the frozen policy-distillation confirmation protocol."""
from __future__ import annotations

from jax_experiments.analysis import (
    analyze_regime_polarity_policy_distillation_confirmation as analyzer,
)
from jax_experiments.analysis import (
    regime_polarity_policy_distillation as retrospective,
)
from jax_experiments.analysis import (
    regime_polarity_policy_distillation_confirmation as protocol,
)
from jax_experiments.analysis.run_regime_polarity_policy_distillation_audit import (
    arm_labels,
)
from scripts import (
    submit_regime_polarity_policy_distillation_confirmation as submitter,
)


def test_confirmation_candidate_and_splits_are_frozen():
    assert protocol.TEACHER_GROUP == "combined"
    assert protocol.STUDENT_SEED == 1511
    assert len(protocol.AUDIT_EVENT_SEEDS) == 5
    assert len(set(protocol.AUDIT_EVENT_SEEDS)) == 5
    prior = {
        *retrospective.TRAIN_EVENT_SEEDS,
        *retrospective.DAGGER_EVENT_SEEDS,
        *retrospective.VALIDATION_EVENT_SEEDS,
        *retrospective.AUDIT_EVENT_SEEDS,
        *retrospective.ensemble.EVENT_SEEDS,
    }
    assert set(protocol.AUDIT_EVENT_SEEDS).isdisjoint(prior)
    assert protocol.FIXED_STRONGEST_ROBUST_ARM in arm_labels("combined")


def test_confirmation_identity_is_independent_and_causal():
    identity = protocol.audit_identity(
        protocol.TEACHER_GROUP,
        protocol.STUDENT_SEED,
        protocol.AUDIT_EVENT_SEEDS[0],
    )
    assert identity["confirmatory"] is True
    assert identity["benchmark_role"] == (
        "independent_frozen_student_confirmation")
    assert identity["frozen_student_manifest"] == (
        protocol.FROZEN_STUDENT_MANIFEST_RECORD)
    assert identity["online_inputs"] == [
        "observation", "commanded_action", "next_observation"]
    assert "mode_id" in identity["forbidden_online_inputs"]


def test_confirmation_frozen_artifacts_match_registration():
    assert protocol.file_record(protocol.model_manifest(
        protocol.TEACHER_GROUP, protocol.STUDENT_SEED)) == (
            protocol.FROZEN_STUDENT_MANIFEST_RECORD)
    assert protocol.file_record(protocol.model_path(
        protocol.TEACHER_GROUP, protocol.STUDENT_SEED)) == (
            protocol.FROZEN_STUDENT_PARAMETER_RECORD)
    assert protocol.file_record(protocol.selection_analysis_path()) == (
        protocol.FROZEN_SELECTION_ANALYSIS_RECORD)


def test_cluster_comparison_uses_five_event_means():
    candidate = {
        seed: [2.0, 2.0] for seed in protocol.AUDIT_EVENT_SEEDS}
    reference = {
        seed: [1.0, 1.0] for seed in protocol.AUDIT_EVENT_SEEDS}
    result = analyzer._comparison(candidate, reference)
    assert result["event_seed_wins"] == 5
    assert result["episode_wins"] == 10
    assert result["cluster_count"] == 5
    assert result["mean_delta"] == 1.0
    assert result["cluster_95pct_t_interval"] == [1.0, 1.0]


def test_scheduler_graph_is_five_cpu_audits_then_aggregate():
    audits = submitter.candidates("audit", "high")
    analyses = submitter.candidates("analysis", "high")
    assert len(audits) == 5
    assert len(analyses) == 1
    signatures = [row[0] for row in audits + analyses]
    assert len(signatures) == len(set(signatures))
    for _, spec, _ in audits:
        assert spec["allowed_nodes"] == submitter.CPU_NODES
        assert spec["vram"] == 0
        assert spec["cpu"] == 32
        assert "JAX_PLATFORMS=cpu" in spec["cmd"]
        assert not any(node.startswith("jtl110gpu")
                       for node in spec["allowed_nodes"])
        assert str(protocol.selection_analysis_path()) in spec["wait_for_files"]
    assert analyses[0][1]["allowed_nodes"] == submitter.CPU_NODES
    assert len(analyses[0][1]["wait_for_files"]) == 5
