"""Matrix compensation, causal timing and compact-only experiment dependencies."""
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx, struct

from jax_experiments.analysis import submission_extensions_protocol_v47 as p
from jax_experiments.analysis import run_submission_extensions_v47 as run
from jax_experiments.envs.brax_env import apply_action_disturbance
from jax_experiments.networks.executed_action_inverse import candidate_action_evidence
from scripts import submit_submission_extensions_v47 as submit


@struct.dataclass
class State:
    obs: jax.Array
    reward: jax.Array
    done: jax.Array


class World:
    sys = jnp.asarray(0)
    horizon, dwell = 4, 1
    gains = jnp.asarray(p.TRANSFORMS, dtype=jnp.float32)

    @staticmethod
    def reset(unused, key):
        return State(jnp.zeros(2), jnp.asarray(0.), jnp.asarray(0.))

    @staticmethod
    def step(state, action):
        return state.replace(obs=state.obs + action, reward=-jnp.sum(action**2))


class Actor(nnx.Module):
    def deterministic(self, observations):
        return jnp.broadcast_to(jnp.asarray((.4, .2)), (observations.shape[0], 2))


def test_permutation_compensation_preserves_box_and_same_noisy_signal():
    action, key = jnp.asarray((.7, -.3)), jax.random.PRNGKey(5)
    for mode in range(4):
        command = run.common.compensated_command(action, World.gains, mode)
        assert np.max(np.abs(command)) <= 1
        np.testing.assert_array_equal(World.gains[mode] @ command, action)
        np.testing.assert_array_equal(apply_action_disturbance(command, key, World.gains[mode], .02),
                                      apply_action_disturbance(action, key, World.gains[0], .02))
    assert not np.array_equal(World.gains[1] @ action, World.gains[1].T @ action)


def test_diagonal_likelihood_and_noise_remain_identical():
    gains = jnp.asarray(((1, 1), (-1, 1), (1, -1), (-1, -1)), jnp.float32)
    matrices = jax.vmap(jnp.diag)(gains)
    actions = jnp.asarray(((.7, -.3), (-.2, .8)))
    predictions = jnp.asarray((((.7, -.3), (.2, .8)), ((.68, -.31), (.19, .79))))
    variance = jnp.asarray((.03, .04))
    vectors = candidate_action_evidence(predictions, actions, gains, variance)
    tensors = candidate_action_evidence(predictions, actions, matrices, variance)
    for left, right in zip(vectors, tensors, strict=True):
        np.testing.assert_allclose(left, right, atol=1e-6)
    for index in range(4):
        key = jax.random.PRNGKey(index)
        np.testing.assert_array_equal(apply_action_disturbance(actions[0], key, gains[index], .02),
                                      apply_action_disturbance(actions[0], key, matrices[index], .02))


def test_non_diagonal_evidence_identifies_executed_channel_mapping():
    action = jnp.asarray(((.6, -.2),))
    predictions = jnp.broadcast_to((World.gains[1] @ action[0])[None, None], (3, 1, 2))
    likelihood, unused, unused = candidate_action_evidence(predictions, action, World.gains, jnp.ones(2)*.002)
    assert int(jnp.argmax(likelihood[0])) == 1


def test_matrix_collection_uses_noise_free_signal_labels():
    actor = Actor()
    collect = run.source.build_collection(World(), nnx.graphdef(actor), nnx.state(actor, nnx.Param))
    obs, following, commands, targets, modes = map(np.asarray, collect(jax.random.PRNGKey(7), 8))
    expected = np.einsum("bij,bj->bi", np.asarray(World.gains)[modes], commands)
    np.testing.assert_allclose(targets, expected, atol=1e-7)
    np.testing.assert_allclose(obs[::4], 0)
    assert np.max(np.abs(following-obs-targets)) > 1e-5


def test_causal_matrix_action_uses_previous_belief_not_hidden_mode():
    actor = Actor()
    evidence = lambda obs, command, following: jnp.asarray((-9., 9., -9., -9.))
    episode = run.common.build_episode(World(), nnx.graphdef(actor), "causal_118003", evidence)
    result = episode(nnx.state(actor, nnx.Param), jax.random.PRNGKey(7), jnp.asarray((3, 2, 0, 1)))
    np.testing.assert_allclose(result[2][0], jnp.full(4, .25))
    np.testing.assert_allclose(result[5][0], (.4, .2))
    np.testing.assert_allclose(result[5][1], (-.2, .4))
    assert int(jnp.argmax(result[2][1])) == 1


def test_conditioned_model_route_receives_belief_without_extra_compensation():
    class ConditionedActor(nnx.Module):
        def deterministic(self, observations):
            assert observations.shape[-1] == 6
            return jnp.broadcast_to(jnp.asarray((.4, .2)), (observations.shape[0], 2))
    actor = ConditionedActor()
    evidence = lambda obs, command, following: jnp.asarray((-9., 9., -9., -9.))
    episode = run.common.build_episode(World(), nnx.graphdef(actor), "conditioned_95121", evidence)
    values = episode(nnx.state(actor, nnx.Param), jax.random.PRNGKey(7), jnp.asarray((3, 2, 0, 1)))
    np.testing.assert_allclose(values[5], np.tile((.4, .2), (4, 1)))
    metrics = run.common.trajectory_metrics(values, 1, "conditioned_95121")
    assert metrics["action_error_energy"] is None


def test_matched_estimator_training_passes_its_frozen_model(monkeypatch):
    calls = []
    monkeypatch.setattr(run.common, "World", lambda *args: World())
    monkeypatch.setattr(run, "evidence", lambda model, gains, suite: model)
    monkeypatch.setattr(run.common, "train_policy", lambda *args, **kwargs: calls.append((args, kwargs)))
    run.train("estimator", "conditioned", p.REACHER_SEEDS[0], p.REACHER_MODELS[1])
    args, kwargs = calls[0]
    assert kwargs["conditioning"][0] == p.REACHER_MODELS[1]
    assert f"model_{p.REACHER_MODELS[1]}" in str(kwargs["run_root"])
    assert kwargs["sampling_device"] == "cpu" and not kwargs["smoke"]


def test_provider_excluded_and_source_budget_includes_warmup():
    assert 88003 not in p.HC_SEEDS and p.PROVIDER_SEED not in p.PERMUTATION_SEEDS
    assert not set(p.PERMUTATION_SEEDS) & set(p.REACHER_SEEDS)
    assert len(p.REACHER_MODELS) == len(p.PERMUTATION_MODELS) == 3
    assert len({p.dataset_dir(seed) for seed in p.PERMUTATION_MODELS}) == 3
    assert 10000 + sum(p.source_samples(i) for i in range(2100)) == p.source_budget()["total_steps"]
    assert 2100 * 250 == p.source_budget()["critic_calls"]
    assert p.source_budget()["actor_opportunities"] == p.source_budget()["critic_calls"] // 2


def test_dag_only_remote_nodes_single_seed_and_compact_results():
    rows = submit.candidates()
    assert len(rows) == 72 and sum(row["vram"] > 0 for row, output in rows) == 44
    assert len({row["signature"] for row, output in rows}) == len(rows)
    outputs = {str(output) for row, output in rows}
    from pathlib import Path
    for row, output in rows:
        assert "local" not in row["allowed_nodes"]
        assert "checkpoints" not in row["result_dir"] and "dataset" not in row["result_dir"]
        assert not any("/dataset" in path for path in row["stage_input_paths"])
        assert not any(Path(path).is_file() for path in row["stage_input_paths"])
        assert str(p.PREREG) in row["wait_for_files"]
        if "/train-" in row["signature"] or row["signature"].endswith("/provider"):
            assert row["cmd"].count("--seed") == 1 and row["resume_managed_by_cmd"]
            assert "smoke/ready.json" in " ".join(row["wait_for_files"])
        if "/audit-" in row["signature"]:
            assert row["vram"] == 0 and set(row["allowed_nodes"]) <= set(p.CPU_NODES)
            assert all(path in row["wait_for_files"] for path in
                       [path + "/bundle.json" for path in row["stage_input_paths"] if path in
                        {str(p.actor("estimator", "conditioned", seed, model))
                         for seed in p.REACHER_SEEDS for model in p.REACHER_MODELS[1:]}])
        if row["signature"].endswith("/aggregate"):
            assert set(row["wait_for_files"]) - {str(p.PREREG)} <= outputs
    source_smoke = next(row for row, output in rows if row["signature"] == submit.PREFIX + "/source/smoke")
    assert len(source_smoke["wait_for_files"]) == 2
    assert source_smoke["allowed_nodes"] == ["jtl110gpu"]


def test_analysis_keeps_model_seeds_as_crossed_factor(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "audit_path", lambda suite, seed: tmp_path / f"{seed}.json")
    monkeypatch.setattr(p, "analysis_path", lambda suite: tmp_path / "analysis.json")
    for index, seed in enumerate(p.REACHER_SEEDS):
        means = {"oracle": 2., "uncompensated": -5., "robust": -1.}
        for model in p.REACHER_MODELS:
            means[f"causal_{model}"] = 1. + index*.1
            means[f"conditioned_{model}"] = 0.
        p.audit_path("estimator", seed).write_text(json.dumps({"protocol": p.PROTOCOL, "suite": "estimator",
            "seed": seed, "runtime_versions": p.RUNTIME, "estimator_seeds": list(p.REACHER_MODELS), "means": means}))
    run.analyze("estimator")
    analysis = json.loads(p.analysis_path("estimator").read_text())
    assert not analysis["historical_cohorts_pooled"]
    for model in p.REACHER_MODELS:
        assert analysis["comparisons"][f"causal_{model}-minus-conditioned_{model}"]["n_policy_seeds"] == 5

