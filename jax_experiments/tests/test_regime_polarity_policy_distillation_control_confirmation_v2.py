"""Tests for the frozen mode-head policy-compression confirmation."""
from __future__ import annotations

from jax_experiments.analysis import (
    analyze_regime_polarity_policy_distillation_control_confirmation_v2 as analyzer,
)
from jax_experiments.analysis import (
    regime_polarity_policy_distillation_confirmation as prior_confirmation,
)
from jax_experiments.analysis import (
    regime_polarity_policy_distillation_control_confirmation_v2 as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_policy_distillation_control_v2 as development,
)
from jax_experiments.analysis import (
    run_regime_polarity_policy_distillation_control_confirmation_audit_v2 as audit,
)
from scripts import (
    submit_regime_polarity_policy_distillation_control_confirmation_v2 as submitter,
)


def test_candidate_and_confirmation_split_are_frozen():
    assert protocol.TEACHER_GROUP == "mode_heads"
    assert protocol.STUDENT_SEED == 1811
    assert len(protocol.AUDIT_EVENT_SEEDS) == 5
    assert len(set(protocol.AUDIT_EVENT_SEEDS)) == 5
    prior = {
        *development.TRAIN_EVENT_SEEDS,
        *development.DAGGER_EVENT_SEEDS,
        *development.SUPERVISED_VALIDATION_EVENT_SEEDS,
        *development.CONTROL_VALIDATION_EVENT_SEEDS,
        *development.AUDIT_EVENT_SEEDS,
        *development.SEALED_CONFIRMATION_EVENT_SEEDS,
        *prior_confirmation.AUDIT_EVENT_SEEDS,
        *development.ensemble.EVENT_SEEDS,
    }
    assert set(protocol.AUDIT_EVENT_SEEDS).isdisjoint(prior)
    assert protocol.FIXED_STRONGEST_ROBUST_ARM in audit.arm_labels(
        protocol.TEACHER_GROUP)


def test_identity_is_confirmatory_and_causal():
    identity = protocol.audit_identity(
        protocol.TEACHER_GROUP,
        protocol.STUDENT_SEED,
        protocol.AUDIT_EVENT_SEEDS[0],
    )
    assert identity["confirmatory"] is True
    assert identity["benchmark_role"] == (
        "independent_mode_head_student_confirmation")
    assert identity["frozen_student_manifest"] == (
        protocol.FROZEN_STUDENT_MANIFEST_RECORD)
    assert identity["online_inputs"] == [
        "observation", "commanded_action", "next_observation"]
    assert "mode_id" in identity["forbidden_online_inputs"]


def test_frozen_artifacts_match_registration():
    protocol.validate_frozen_candidate()
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
    analyzer._bind_confirmation()
    result = analyzer.base._comparison(candidate, reference)
    assert result["event_seed_wins"] == 5
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
        assert str(protocol.selection_analysis_path()) in spec["wait_for_files"]
    assert len(analyses[0][1]["wait_for_files"]) == 5

