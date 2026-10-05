"""Untouched cohort, frozen runner reuse, dependency staging and primary-only decision."""
import json

import pytest

from jax_experiments.analysis import reacher_information_confirmation_protocol_v38 as p
from jax_experiments.analysis import run_reacher_information_confirmation_v38 as run
from scripts import submit_reacher_information_confirmation_v38 as submit


def test_confirmation_excludes_all_previous_cohorts_and_freezes_settings():
    assert len(set(p.POLICY_SEEDS)) == 5
    prior = {p.source.PROVIDER_SEED, *p.source.POLICY_SEEDS,
             *p.development.confirmation.POLICY_SEEDS, *p.development.POLICY_SEEDS}
    assert not set(p.POLICY_SEEDS) & prior
    assert not set(p.EVENT_SEEDS) & {*p.source.EVENT_SEEDS, *p.development.EVENT_SEEDS,
                                   *p.development.confirmation.EVENT_SEEDS}
    assert p.STATIONARY_EVENT_SEED != p.development.STATIONARY_EVENT_SEED
    assert p.ARMS == p.development.ARMS
    assert p.CONTROLS == p.development.CONTROLS and p.RUNTIME == p.development.RUNTIME
    assert p.PRIMARY_MODEL == 95103 and p.FROZEN_METADATA == p.development.FROZEN_METADATA
    assert p.RUN_ROOT != p.development.RUN_ROOT and p.SMOKE != p.development.SMOKE


def test_shared_trainer_uses_confirmation_paths_without_mutating_development(monkeypatch):
    captured = []
    evidence = lambda *unused: None
    monkeypatch.setattr(run.shared.source, "load_evidence", lambda *unused: evidence)
    monkeypatch.setattr(run.shared.common, "train_policy", lambda *args, **kwargs: captured.append((args, kwargs)))
    before = (p.development.RUN_ROOT, p.development.POLICY_SEEDS, p.development.PROTOCOL)
    run.shared.train("conditioned_no_filter", p.POLICY_SEEDS[0], protocol=p)
    args, kwargs = captured[0]
    assert args == ("reacher", "conditioned_no_filter", p.POLICY_SEEDS[0])
    assert kwargs["run_root"] == p.RUN_ROOT and kwargs["bundle_root"] == p.BUNDLE_ROOT
    assert kwargs["protocol_name"] == p.PROTOCOL and kwargs["sampling_device"] == "cpu"
    assert kwargs["conditioning"] == (evidence, p.CONTROLS, False)
    assert (p.development.RUN_ROOT, p.development.POLICY_SEEDS, p.development.PROTOCOL) == before


def test_confirmation_dag_has_twenty_single_seed_trainings_and_minimal_sync():
    rows = submit.candidates()
    assert len(rows) == 27 and sum(row["vram"] > 0 for row, output in rows) == 21
    assert len({row["signature"] for row, output in rows}) == 27
    training = [row for row, output in rows if "/train-" in row["signature"]]
    assert len(training) == 20
    for row, output in rows:
        assert "local" not in row["allowed_nodes"]
        assert str(p.PREREG) in row["wait_for_files"]
        assert str(p.FROZEN_METADATA) in row["wait_for_files"]
        assert "run_reacher_information_confirmation_v38" in row["cmd"]
        assert not any("checkpoints" in path or "/dataset" in path for path in row["stage_input_paths"])
        if "/train-" in row["signature"]:
            assert row["vram"] == 1500 and row.get("require_node") is None
            assert str(p.SMOKE) in row["wait_for_files"]
            assert row["resume_managed_by_cmd"] and row["ckpt_glob"] == "train_state.pkl"
            assert row["ckpt_dir"].startswith(str(p.RUN_ROOT))
        elif "/audit-" in row["signature"]:
            seed = int(row["signature"].rsplit("-", 1)[1])
            assert row["vram"] == 0
            assert all(str(p.screen_bundle(arm, seed) / "bundle.json") in row["wait_for_files"] for arm in p.ARMS)
        elif row["description"] == "aggregate":
            assert row["cpu"] == 1 and row["ram_mb"] == 2048
            assert all(str(p.audit_path(seed)) in row["wait_for_files"] for seed in p.POLICY_SEEDS)
            assert str(p.source.model_dir(p.PRIMARY_MODEL)) not in row["stage_input_paths"]
            assert all("eval_bundles" not in path for path in row["stage_input_paths"]
                       if path != str(p.FROZEN_METADATA.parent))


def write_audits(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "audit_path", lambda seed: tmp_path / f"{seed}.json")
    monkeypatch.setattr(p, "ANALYSIS", tmp_path / "analysis.json")
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({"provider": {"total_steps": 1600000, "update_count": 100000}}))
    monkeypatch.setattr(p, "FROZEN_METADATA", frozen)
    monkeypatch.setattr(run.shared.confirmation, "frozen_model_metadata",
                        lambda: {"calibration_interactions": 260000, "updates": 4500})
    for difference, seed in zip((4., 4., 4., 4., -.1), p.POLICY_SEEDS, strict=True):
        means = {f"causal_{p.PRIMARY_MODEL}": -8. + difference, "conditioned": -8.,
                 "no_filter": -9., "conditioned_no_filter": -7., "robust": -23., "oracle": -3.}
        budgets = {arm: {"protocol": p.PROTOCOL, "seed": seed, "arm": arm, "runtime_versions": p.RUNTIME,
                         "sampling_device": "cpu", "next_iteration": 400, "total_steps": 1600000,
                         "update_count": 100000, "policy_input_dim": 15 if arm in p.CONDITIONED_ARMS else 11,
                         "policy_parameters": 70916 if arm in p.CONDITIONED_ARMS else 69892,
                         "belief_controls": list(p.CONTROLS), "temporal_filter": arm == "conditioned"}
                   for arm in p.ARMS}
        event = {"steps": 200, "switches_observed": 3, "termination": False}
        p.audit_path(seed).write_text(json.dumps({
            "seed": seed, "runtime_versions": p.RUNTIME, "budgets": budgets, "means": means,
            "events": {arm: [event] * 15 for arm in means},
            "stationary": {arm: [{"return": value}] for arm, value in means.items()}}))


def test_positive_primary_does_not_require_all_five_wins_or_positive_secondary(tmp_path, monkeypatch):
    write_audits(tmp_path, monkeypatch)
    run.analyze()
    result = json.loads(p.ANALYSIS.read_text())
    assert "development_evidence" not in result
    assert result["confirmation_evidence"]["positive_primary_paired_ci"]
    assert result["confirmation_evidence"]["primary_wins"] == 4
    assert result["comparisons"][f"causal_{p.PRIMARY_MODEL}-minus-conditioned"]["n_policy_seeds"] == 5
    assert result["comparisons"]["no_filter-minus-conditioned_no_filter"]["ci95"][1] < 0
    assert result["shared_estimator_cost"]["charged_to_both_same_information_routes"]


def test_positive_secondary_cannot_replace_negative_primary(tmp_path, monkeypatch):
    write_audits(tmp_path, monkeypatch)
    for seed in p.POLICY_SEEDS:
        path = p.audit_path(seed)
        row = json.loads(path.read_text())
        row["means"][f"causal_{p.PRIMARY_MODEL}"] = -12.
        row["means"]["no_filter"] = -5.
        path.write_text(json.dumps(row))
    run.analyze()
    result = json.loads(p.ANALYSIS.read_text())
    assert not result["confirmation_evidence"]["positive_primary_paired_ci"]
    assert result["comparisons"]["no_filter-minus-conditioned_no_filter"]["ci95"][0] > 0
    assert "retain negative" in result["next"]


def test_misrouted_development_budget_cannot_publish_confirmation(tmp_path, monkeypatch):
    write_audits(tmp_path, monkeypatch)
    path = p.audit_path(p.POLICY_SEEDS[-1])
    row = json.loads(path.read_text())
    row["budgets"]["conditioned"]["protocol"] = p.development.PROTOCOL
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="budget"):
        run.analyze()
    assert not p.ANALYSIS.exists()


def test_cpu_audit_loads_four_confirmation_actors(monkeypatch, tmp_path):
    write_audits(tmp_path, monkeypatch)
    metadata = json.loads(p.audit_path(p.POLICY_SEEDS[0]).read_text())["budgets"]
    monkeypatch.setattr(p, "screen_bundle", lambda arm, seed: tmp_path / arm)
    for arm in p.ARMS:
        directory = p.screen_bundle(arm, p.POLICY_SEEDS[0])
        directory.mkdir()
        (directory / "bundle.json").write_text(json.dumps(metadata[arm]))
    captured = []
    monkeypatch.setattr(run.shared.source, "audit", lambda *args, **kwargs: captured.append((args, kwargs)))
    run.shared.audit(p.POLICY_SEEDS[0], protocol=p)
    assert captured == [((p.POLICY_SEEDS[0],), {"protocol": p, "conditioned_arms": p.CONDITIONED_ARMS})]
