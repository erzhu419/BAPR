"""Tests for the sealed regime-adapter multi-seed confirmation."""
from __future__ import annotations

import numpy as np

from jax_experiments.analysis import analyze_regime_adapter_confirmation
from jax_experiments.analysis import regime_adapter_confirmation as protocol
from jax_experiments.analysis import regime_adapter_fork as development
from jax_experiments.analysis import run_regime_adapter_branch
from jax_experiments.analysis import (
    run_regime_adapter_confirmation_audit as confirmation_audit,
)
from scripts import submit_regime_adapter_confirmation as submitter


def test_protocol_freezes_development_choice_and_untouched_seeds() -> None:
    assert protocol.DELTA == 0.5
    assert protocol.CONTROLLER_MAP == (0, 1, 2, 3)
    assert protocol.DEVELOPMENT_SEED == 8
    assert protocol.HOLDOUT_SEEDS == (16, 24, 32, 40)
    assert protocol.TRAINING_SEEDS == (8, 16, 24, 32, 40)
    assert len(protocol.EVENT_SEEDS) == 5
    assert set(protocol.EVENT_SEEDS).isdisjoint(development.AUDIT_EVENT_SEEDS)
    assert run_regime_adapter_branch.ACTION_EQUIVALENCE_ATOL == 1e-5
    assert (
        run_regime_adapter_branch.CRITIC_EQUIVALENCE_GLOBAL_RTOL == 2e-3)


def test_scheduler_graph_is_unpinned_file_gated_and_has_no_calibration() -> None:
    training = submitter.candidates("train", "high")
    audits = submitter.candidates("audit", "high")
    analysis = submitter.candidates("analysis", "high")
    assert len(training) == 25
    assert len(audits) == 25
    assert len(analysis) == 1
    signatures = [row[0] for row in training + audits + analysis]
    assert len(signatures) == len(set(signatures))
    assert not any("calibration" in signature for signature in signatures)

    for _, spec, _ in training:
        assert spec["vram"] == submitter.MEASURED_VRAM_MB == 2300
        assert "allowed_nodes" not in spec
        assert "preferred_node" not in spec
        assert "require_node" not in spec
        assert spec["resume_managed_by_cmd"] is True
        assert spec["reroute_on_node_down"] is True
    for _, spec, _ in audits:
        assert spec["vram"] == 0
        assert spec["cpu"] == 32
        assert spec["allowed_nodes"] == submitter.CPU_NODES
        assert len(spec["wait_for_files"]) == 25
        assert len(spec["stage_input_paths"]) == 5
        assert spec["allow_cpu_training"] is True
    assert len(analysis[0][1]["wait_for_files"]) == 25


def test_primary_gate_uses_only_four_holdout_training_seeds() -> None:
    original_validate = analyze_regime_adapter_confirmation.validate_audit
    original_seed_value = analyze_regime_adapter_confirmation._seed_value
    original_seed_mode_values = (
        analyze_regime_adapter_confirmation._seed_mode_values)

    def fake_seed_value(seed: int, case: str, metric: str) -> float:
        del seed
        if metric == "switching_termination":
            return 0.0
        if case == "identity_adapter":
            return 120.0
        if case.startswith("fixed_adapter_"):
            return 110.0
        if case == "frozen_base":
            return 80.0
        return 100.0

    analyze_regime_adapter_confirmation.validate_audit = lambda *_: {}
    analyze_regime_adapter_confirmation._seed_value = fake_seed_value
    analyze_regime_adapter_confirmation._seed_mode_values = (
        lambda _seed, case: (
            np.full(4, 120.0) if case == "identity_adapter"
            else np.full(4, 110.0)))
    try:
        payload = analyze_regime_adapter_confirmation.analyze()
    finally:
        analyze_regime_adapter_confirmation.validate_audit = original_validate
        analyze_regime_adapter_confirmation._seed_value = original_seed_value
        analyze_regime_adapter_confirmation._seed_mode_values = (
            original_seed_mode_values)

    holdout = payload["holdout"]
    assert holdout["training_seeds"] == [16, 24, 32, 40]
    assert holdout["identity_minus_robust_switching"][
        "n_training_seeds"] == 4
    assert holdout["identity_minus_robust_switching"]["ci95_low"] == 20.0
    assert holdout["promotion_gate"][
        "identity_beats_robust_all_four_holdout_seeds"] is True
    assert holdout["promotion_gate"]["pass"] is True
    assert payload["decision"] == (
        "train_causal_router_against_confirmed_adapter_bank")


def test_training_provenance_is_workspace_relocation_safe() -> None:
    local = (
        "/home/erzhu419/mine_code/BAPR/jax_experiments/"
        "eval_bundles_regime_adapter_fork_v1/seed_16/"
        "robust_continue/bundle_manifest.json")
    remote = (
        "/home/zhengliang01/scheduleurm_work/BAPR/jax_experiments/"
        "eval_bundles_regime_adapter_fork_v1/seed_16/"
        "robust_continue/bundle_manifest.json")
    assert confirmation_audit._portable_manifest_key(local) == (
        confirmation_audit._portable_manifest_key(remote))
