"""Actual CLI controls, update budgets, factorial analysis and frozen actions."""
import sys
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
from flax import nnx
import pytest

from jax_experiments import train
from jax_experiments.algos.resac_torch import TorchPolicy
from jax_experiments.analysis import resac_update_density_protocol_v42 as p
from jax_experiments.analysis import run_resac_update_density_v42 as run
from jax_experiments.configs.default import Config
from scripts import submit_resac_update_density_v42 as submit


@pytest.mark.parametrize("arm", p.ARMS)
def test_actual_cli_and_registered_budget(arm, monkeypatch):
    command = run.training_command(arm, p.SEEDS[0])
    parsed = []
    monkeypatch.setattr(sys, "argv", ["train.py", *command[command.index("jax_experiments.train") + 1:]])
    monkeypatch.setattr(train, "train", parsed.append)
    train.main()
    config = parsed[0]
    assert all(getattr(config, key) == value for key, value in p.settings(arm).items())
    assert config.samples_per_iter * config.max_iters + config.initial_random_steps == 1010000
    count = 1000000 if arm.startswith("high_") else 62500
    assert p.budget(arm)["update_count"] == count
    assert p.actor_updates(arm) == count // 2
    metadata = {"identity": p.identity(arm, config.seed), "runtime_versions": p.RUNTIME,
                "norm_definition": p.NORM_DEFINITION, "budget": p.budget(arm),
                "actor_update_count": p.actor_updates(arm), "config": vars(config)}
    p.validate(metadata, arm, config.seed)
    metadata["budget"] = {**p.budget(arm), "total_steps": 1000000}
    with pytest.raises(ValueError, match="budget"):
        p.validate(metadata, arm, config.seed)


def test_factorial_controls_only_change_density_and_norm():
    profiles = [{key: value for key, value in p.settings(arm).items()
                 if key not in {"weight_reg", "updates_per_iter"}} for arm in p.ARMS]
    assert all(profile == profiles[0] for profile in profiles)
    assert p.settings("high_full")["weight_reg"] == p.settings("low_full")["weight_reg"] == 0.01
    assert p.settings("high_no_shift")["weight_reg"] == p.settings("low_no_shift")["weight_reg"] == 0


def test_compiled_final_actor_preserves_deterministic_actions():
    policy = TorchPolicy(3, 2, 8, rngs=nnx.Rngs(41))
    observations = np.random.default_rng(5).normal(size=(12, 3)).astype(np.float32)
    actor = run.FrozenActor(policy)
    for obs in observations:
        np.testing.assert_allclose(actor.select_action(obs), policy.deterministic(jnp.asarray(obs)[None])[0],
                                   rtol=1e-5, atol=1e-7)


def audits():
    values = {"low_full": -10., "low_no_shift": 100., "high_full": 50., "high_no_shift": 120.}
    return [{"protocol": p.PROTOCOL, "seed": seed, "runtime_versions": p.RUNTIME,
             "budgets": {arm: p.budget(arm) for arm in p.ARMS},
             "actor_updates": {arm: p.actor_updates(arm) for arm in p.ARMS},
             "switching": {"event": {arm: {"returns": [value + i, value + i]} for arm, value in values.items()}},
             "stationary": {"event": {arm: {"0": {"return_mean": value + i}} for arm, value in values.items()}},
             "diagnostics": {}} for i, seed in enumerate(p.SEEDS)]


def test_planned_contrasts_and_interaction_use_seed_level_pairs():
    result = run.summarize(audits())
    assert result["comparisons"]["high_full-minus-low_full"]["mean"] == 60
    assert result["comparisons"]["high_no_shift-minus-low_no_shift"]["mean"] == 20
    assert result["density_norm_interaction"]["mean"] == 40
    assert result["density_norm_interaction"]["ci95"] == [40., 40.]
    assert result["stationary"]["low_full"] == [-10., -9., -8.]
    bad = audits()
    bad[1]["actor_updates"]["high_full"] -= 1
    with pytest.raises(ValueError, match="mixed or incomplete"):
        run.summarize(bad)


def test_batch_dependencies_and_minimal_sync():
    rows = submit.candidates()
    assert len(rows) == 17 and sum(row["vram"] > 0 for row, _ in rows) == 13
    for row, output in rows:
        assert "local" not in row["allowed_nodes"]
        assert "checkpoints" not in row["result_dir"]
        if "train-" in row["signature"]:
            assert row["wait_for_files"] == [str(p.PREREG), str(p.SMOKE)]
            assert row["resume_managed_by_cmd"] and not row["allow_initial_resume_scan_error"]
        if "audit-" in row["signature"]:
            assert row["vram"] == 0 and len(row["wait_for_files"]) == 6
    assert len(rows[-1][0]["wait_for_files"]) == 4


def test_audit_entry_runs_all_registered_arms(monkeypatch):
    evaluator = SimpleNamespace(
        _stationary_runtime=lambda config, runtime, event, mode: {"return_mean": float(mode)},
        _switching_runtime=lambda config, runtime, event: {
            "returns": [1., 2.], "episode_sequences": [[0, 1, 2, 3]]})
    monkeypatch.setattr(run.common, "evaluate", evaluator)
    monkeypatch.setattr(run, "load_actor", lambda arm, seed, **kwargs: (SimpleNamespace(**p.settings(arm)), None))
    monkeypatch.setattr(run, "read", lambda path: {})
    monkeypatch.setattr(p, "check_runtime", lambda: p.RUNTIME)
    written = []
    monkeypatch.setattr(run, "write", lambda path, row: written.append((path, row)))
    run.audit(p.SEEDS[0])
    path, row = written[0]
    assert path == p.audit_path(p.SEEDS[0]) and evaluator.protocol is p
    assert set(row["stationary"]) == set(map(str, p.STATIONARY_HOLDOUT_EVENT_SEEDS))
    assert set(row["switching"]) == set(map(str, p.SWITCHING_EVENT_SEEDS))
    assert all(set(event) == set(p.ARMS) for event in row["switching"].values())
    assert row["budgets"] == {arm: p.budget(arm) for arm in p.ARMS}
