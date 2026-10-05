"""Fresh seeds, immutable estimator inputs, budget gates and scheduler placement."""
import json

import pytest

from jax_experiments.analysis import reacher_confirmation_protocol_v36 as p
from jax_experiments.analysis import run_reacher_confirmation_v36 as run
from scripts import submit_reacher_confirmation_v36 as submit


def test_fresh_cohort_keeps_estimator_and_evaluation_protocol_frozen():
    assert len(set(p.POLICY_SEEDS)) == 5
    assert not set(p.POLICY_SEEDS) & {p.source.PROVIDER_SEED, *p.source.POLICY_SEEDS}
    assert p.MODEL_SEEDS == (p.source.PRIMARY_MODEL,)
    assert p.CONTROLS == p.source.CONTROLS and p.RUNTIME == p.source.RUNTIME
    assert not set(p.EVENT_SEEDS) & set(p.source.EVENT_SEEDS)
    assert p.STATIONARY_EVENT_SEED != p.source.STATIONARY_EVENT_SEED
    assert p.RUN_ROOT != p.source.RUN_ROOT


def test_new_trainer_routes_outputs_without_mutating_old_protocol(monkeypatch):
    captured = []
    old_roots = (run.common.p.RUN_ROOT, run.common.p.BUNDLE_ROOT)
    monkeypatch.setattr(run.common, "train_policy", lambda *args, **kwargs: captured.append((args, kwargs)))
    run.train("robust", p.POLICY_SEEDS[0])
    args, kwargs = captured[0]
    assert args == ("reacher", "robust", p.POLICY_SEEDS[0])
    assert kwargs["sampling_device"] == "cpu" and kwargs["protocol_name"] == p.PROTOCOL
    assert kwargs["run_root"] == p.RUN_ROOT and kwargs["bundle_root"] == p.BUNDLE_ROOT
    assert (run.common.p.RUN_ROOT, run.common.p.BUNDLE_ROOT) == old_roots


def test_dag_has_individual_seed_jobs_and_minimal_inputs():
    rows = submit.candidates()
    assert len(rows) == 17 and sum(row["vram"] > 0 for row, output in rows) == 11
    training = [row for row, output in rows if "/train-" in row["signature"]]
    assert len(training) == 10
    for row, output in rows:
        assert "local" not in row["allowed_nodes"]
        assert str(p.PREREG) in row["wait_for_files"]
        assert str(p.FROZEN_METADATA) in row["wait_for_files"]
        assert str(p.source.DATA_DIR) not in row["stage_input_paths"]
        assert str(p.source.model_dir(p.PRIMARY_MODEL)) not in row["stage_input_paths"] or row["description"].startswith(("smoke", "audit"))
        if "/train-" in row["signature"]:
            assert row["wait_for_files"].count(str(p.SMOKE)) == 1
            assert row["resume_managed_by_cmd"] and row["ckpt_glob"] == "train_state.pkl"
            assert row["vram"] == 1500 and "JAX_PLATFORMS=cuda,cpu" in row["cmd"]
            assert row.get("require_node") is None
        if "/audit-" in row["signature"]:
            seed = int(row["signature"].rsplit("-", 1)[1])
            assert row["vram"] == 0 and submit.legacy.legacy.CPU_EVAL_PYTHON in row["cmd"]
            assert all(str(p.screen_bundle(arm, seed) / "bundle.json") in row["wait_for_files"]
                       for arm in ("reference", "robust"))
        if row["description"] == "aggregate":
            assert row["cpu"] == 1 and row["ram_mb"] == 2048
            assert all(str(p.audit_path(seed)) in row["wait_for_files"] for seed in p.POLICY_SEEDS)


def test_prepare_metadata_does_not_replace_frozen_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "FROZEN_METADATA", tmp_path / "frozen/metadata.json")
    monkeypatch.setattr(p.source, "model_dir", lambda seed: tmp_path / "model")
    monkeypatch.setattr(p.source, "PROVIDER_BUNDLE", tmp_path / "provider")
    for name, filename in (("model", "complete.json"), ("provider", "bundle.json")):
        directory = tmp_path / name
        directory.mkdir()
        (directory / filename).write_text(json.dumps({"seed": 7}))
    submit.prepare_metadata()
    before = p.FROZEN_METADATA.read_text()
    (tmp_path / "model/complete.json").write_text(json.dumps({"seed": 8}))
    submit.prepare_metadata()
    assert p.FROZEN_METADATA.read_text() == before


def write_audits(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "ANALYSIS", tmp_path / "analysis.json")
    monkeypatch.setattr(p, "audit_path", lambda seed: tmp_path / f"{seed}.json")
    monkeypatch.setattr(p, "FROZEN_METADATA", tmp_path / "frozen.json")
    p.FROZEN_METADATA.write_text(json.dumps({
        "model": {"model_seed": p.PRIMARY_MODEL, "provider_seed": p.source.PROVIDER_SEED,
                  "runtime_versions": p.RUNTIME, "filter_controls": list(p.CONTROLS),
                  "updates": 4500, "calibration_interactions": 260000, "parameters": 704010},
        "provider": {"seed": p.source.PROVIDER_SEED, "total_steps": 1600000, "update_count": 100000}}))
    for index, seed in enumerate(p.POLICY_SEEDS):
        means = {"oracle": 2.0, "robust": 0.0, f"causal_{p.PRIMARY_MODEL}": 1 + 0.05 * index,
                 "no_filter": 0.8, "single_head": 0.9, **{f"fixed_{mode}": -1.0 for mode in range(4)}}
        event = {"steps": 200, "switches_observed": 3, "mode_accuracy": 0.8,
                 "brier": 0.4, "action_error_energy": 1.0,
                 "delays": [{"delay": value, "window_steps": 50} for value in (9, None, 10)],
                 "switch_windows": {str(w): {"reward_sum": -0.1} for w in (1, 2, 4, 8, 16, 32)}}
        budgets = {arm: {"protocol": p.PROTOCOL, "seed": seed, "arm": arm,
                         "runtime_versions": p.RUNTIME, "sampling_device": "cpu",
                         "next_iteration": 400, "total_steps": 1600000, "update_count": 100000}
                   for arm in ("reference", "robust")}
        p.audit_path(seed).write_text(json.dumps({
            "seed": seed, "runtime_versions": p.RUNTIME, "budgets": budgets, "means": means,
            "events": {arm: [event] * 15 for arm in means},
            "stationary": {arm: [{"return": value}] for arm, value in means.items()}}))


def test_analysis_uses_five_new_seeds_and_includes_missed_switches(tmp_path, monkeypatch):
    write_audits(tmp_path, monkeypatch)
    run.analyze()
    result = json.loads(p.ANALYSIS.read_text())
    assert result["causal_screen"]["passes"]
    assert result["comparisons"][f"causal_{p.PRIMARY_MODEL}-minus-robust"]["n_policy_seeds"] == 5
    metrics = result["identification"][f"causal_{p.PRIMARY_MODEL}"]
    assert metrics["switches"] == 225 and metrics["detected"] == 150
    assert metrics["median_delay_including_undetected"] == 10
    assert result["shared_estimator_cost"]["calibration_interactions"] == 260000
    assert result["shared_estimator_cost"]["equal_total_training_cost"] is False
    assert result["shared_estimator_cost"]["retrained_in_confirmation"] is False


def test_invalid_budget_cannot_publish_confirmation(tmp_path, monkeypatch):
    write_audits(tmp_path, monkeypatch)
    path = p.audit_path(p.POLICY_SEEDS[-1])
    row = json.loads(path.read_text())
    row["budgets"]["robust"]["total_steps"] = 1500000
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="budget"):
        run.analyze()
    assert not p.ANALYSIS.exists()
