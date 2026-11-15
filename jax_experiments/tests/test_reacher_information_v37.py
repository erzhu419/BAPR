"""Causal belief parity, direct-command semantics, compact DAG and seed-unit analysis."""
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jax_experiments.analysis import reacher_information_protocol_v37 as p
from jax_experiments.analysis import run_reacher_information_v37 as run
from jax_experiments.networks.policy import GaussianPolicy
from jax_experiments.tests.test_mechanism_gap_v34 import ToyWorld
from scripts import submit_reacher_information_v37 as submit


def test_new_seeds_leave_confirmation_and_estimator_untouched():
    assert len(set(p.POLICY_SEEDS)) == 3
    assert not set(p.POLICY_SEEDS) & {p.source.PROVIDER_SEED, *p.source.POLICY_SEEDS,
                                     *p.confirmation.POLICY_SEEDS}
    assert p.PRIMARY_MODEL == 95103 and p.FROZEN_METADATA == p.confirmation.FROZEN_METADATA
    assert p.CONTROLS == p.source.CONTROLS and p.RUNTIME == p.source.RUNTIME
    assert not set(p.EVENT_SEEDS) & set(p.confirmation.EVENT_SEEDS)


@pytest.mark.parametrize("temporal", [True, False])
def test_training_uses_supplied_reacher_belief_causally(monkeypatch, temporal):
    def unexpected():
        raise AssertionError("Reacher must not load the HalfCheetah inverse")

    monkeypatch.setattr(run.common, "frozen_inverse", unexpected)
    world = ToyWorld()
    policy = GaussianPolicy(5, 1, 8, rngs=nnx.Rngs(17))
    likelihood = jnp.asarray((-1., 2., 0., -2.))
    evidence = lambda obs, command, next_obs: likelihood
    scan = run.common.build_training_scan(world, nnx.graphdef(policy), evidence, True,
                                          controls=p.CONTROLS, temporal=temporal)
    _, transitions = scan(nnx.state(policy, nnx.Param), world.reset(0, None), jnp.full(4, .25),
                          jnp.asarray(0), jax.random.split(jax.random.PRNGKey(31), 3), jnp.asarray((0, 3, 2)))
    obs, command, reward, next_obs, done = transitions
    expected = run.common.update_belief(jnp.full(4, .25), likelihood, p.CONTROLS, temporal)
    np.testing.assert_allclose(obs[0, -4:], np.full(4, .25))
    np.testing.assert_allclose(next_obs[0, -4:], expected, atol=1e-7)
    np.testing.assert_allclose(obs[1:, -4:], next_obs[:-1, -4:])
    assert int(jnp.argmax(obs[1, -4:])) == 1  # physical mode is 3, not inferred mode 1
    expected_second = run.common.update_belief(expected, likelihood, p.CONTROLS, temporal)
    np.testing.assert_allclose(next_obs[1, -4:], expected_second, atol=1e-7)


@pytest.mark.parametrize("route", p.CONDITIONED_ARMS)
def test_conditioned_evaluation_has_no_external_action_compensation(route):
    world = ToyWorld()
    world.horizon, world.dwell = 4, 1
    policy = GaussianPolicy(5, 1, 8, rngs=nnx.Rngs(17))
    graph, params = nnx.graphdef(policy), nnx.state(policy, nnx.Param)
    evidence = lambda obs, command, next_obs: jnp.asarray((-5., 5., -5., -5.))
    episode = run.common.build_episode(world, graph, route, evidence, route == "conditioned")
    values = episode(params, jax.random.PRNGKey(7), jnp.asarray((1, 1, 1, 1)))
    beliefs, commands, next_observations = map(np.asarray, (values[2], values[5], values[6]))
    observations = np.concatenate((np.zeros((1, 1)), next_observations[:-1]))
    expected = policy.deterministic(jnp.asarray(np.concatenate((observations, beliefs), axis=1)))
    np.testing.assert_allclose(commands, expected, atol=1e-6)
    metrics = run.common.trajectory_metrics(values, world.dwell, route)
    assert metrics["action_error_energy"] is None
    assert metrics["mode_accuracy"] is not None


def test_trainer_loads_frozen_inverse_on_cpu_and_preserves_output_roots(monkeypatch):
    captured = []
    evidence = lambda *unused: jnp.zeros(4)

    def load(seed, gains):
        assert seed == p.PRIMARY_MODEL
        assert gains.device.platform == "cpu"
        return evidence

    monkeypatch.setattr(run.source, "load_evidence", load)
    monkeypatch.setattr(run.common, "train_policy", lambda *args, **kwargs: captured.append((args, kwargs)))
    run.train("conditioned_no_filter", p.POLICY_SEEDS[0])
    args, kwargs = captured[0]
    assert args == ("reacher", "conditioned_no_filter", p.POLICY_SEEDS[0])
    assert kwargs["conditioning"] == (evidence, p.CONTROLS, False)
    assert kwargs["sampling_device"] == "cpu" and kwargs["protocol_name"] == p.PROTOCOL
    assert kwargs["run_root"] == p.RUN_ROOT and kwargs["bundle_root"] == p.BUNDLE_ROOT
    assert run.conditioning("robust") is None


def test_dag_gates_all_four_actors_and_never_stages_full_checkpoints():
    rows = submit.candidates()
    assert len(rows) == 17 and sum(row["vram"] > 0 for row, output in rows) == 13
    for row, output in rows:
        assert "local" not in row["allowed_nodes"]
        assert str(p.PREREG) in row["wait_for_files"]
        assert str(p.FROZEN_METADATA) in row["wait_for_files"]
        assert not any("checkpoints" in path or "/dataset" in path for path in row["stage_input_paths"])
        if "/train-" in row["signature"]:
            assert str(p.SMOKE) in row["wait_for_files"]
            assert row["vram"] == 1500 and row.get("require_node") is None
            assert row["resume_managed_by_cmd"] and row["ckpt_glob"] == "train_state.pkl"
            assert row["allowed_nodes"] == ["jtl110gpu", "jtl110gpu2", "jtl311linux"]
            uses_inverse = any(arm in row["description"] for arm in p.CONDITIONED_ARMS)
            assert (str(p.source.model_dir(p.PRIMARY_MODEL)) in row["stage_input_paths"]) == uses_inverse
        elif "/audit-" in row["signature"]:
            seed = int(row["signature"].rsplit("-", 1)[1])
            assert row["vram"] == 0
            assert all(str(p.screen_bundle(arm, seed) / "bundle.json") in row["wait_for_files"] for arm in p.ARMS)
        elif row["description"] == "aggregate":
            assert row["cpu"] == 1
            assert all(str(p.audit_path(seed)) in row["wait_for_files"] for seed in p.POLICY_SEEDS)
            assert str(p.source.model_dir(p.PRIMARY_MODEL)) not in row["stage_input_paths"]


def write_audits(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "audit_path", lambda seed: tmp_path / f"{seed}.json")
    monkeypatch.setattr(p, "ANALYSIS", tmp_path / "analysis.json")
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({"provider": {"total_steps": 1600000, "update_count": 100000}}))
    monkeypatch.setattr(p, "FROZEN_METADATA", frozen)
    monkeypatch.setattr(run.confirmation, "frozen_model_metadata",
                        lambda: {"calibration_interactions": 260000, "updates": 4500})
    for index, seed in enumerate(p.POLICY_SEEDS):
        means = {f"causal_{p.PRIMARY_MODEL}": -6. + .1 * index, "no_filter": -5.5 + .1 * index,
                 "conditioned": -8., "conditioned_no_filter": -7., "robust": -23., "oracle": -4.}
        budgets = {arm: {"protocol": p.PROTOCOL, "seed": seed, "arm": arm, "runtime_versions": p.RUNTIME,
                         "sampling_device": "cpu", "next_iteration": 400, "total_steps": 1600000,
                         "update_count": 100000, "policy_input_dim": 15 if arm in p.CONDITIONED_ARMS else 11,
                         "policy_parameters": 1000, "belief_controls": list(p.CONTROLS),
                         "temporal_filter": arm == "conditioned"} for arm in p.ARMS}
        event = {"steps": 200, "switches_observed": 3, "termination": False}
        p.audit_path(seed).write_text(json.dumps({
            "seed": seed, "runtime_versions": p.RUNTIME, "budgets": budgets, "means": means,
            "events": {arm: [event] * 15 for arm in means},
            "stationary": {arm: [{"return": value}] for arm, value in means.items()}}))


def test_matched_analysis_keeps_three_seed_unit_and_shared_cost(tmp_path, monkeypatch):
    write_audits(tmp_path, monkeypatch)
    run.analyze()
    result = json.loads(p.ANALYSIS.read_text())
    assert result["comparisons"][f"causal_{p.PRIMARY_MODEL}-minus-conditioned"]["n_policy_seeds"] == 3
    assert result["comparisons"]["no_filter-minus-conditioned_no_filter"]["wins"] == 3
    assert result["shared_estimator_cost"]["charged_to_both_same_information_routes"]
    assert not result["shared_estimator_cost"]["retrained"]


@pytest.mark.parametrize("field,value", [("total_steps", 1500000), ("temporal_filter", False)])
def test_incomplete_budget_or_belief_mismatch_cannot_publish(tmp_path, monkeypatch, field, value):
    write_audits(tmp_path, monkeypatch)
    path = p.audit_path(p.POLICY_SEEDS[-1])
    row = json.loads(path.read_text())
    row["budgets"]["conditioned"][field] = value
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        run.analyze()
    assert not p.ANALYSIS.exists()
