"""Causal timing, estimator resume/export, and the new scheduler dependency DAG."""
import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx, struct

from jax_experiments.analysis import reacher_causal_protocol_v35 as p
from jax_experiments.analysis import run_reacher_causal_v35 as run
from jax_experiments.networks.executed_action_inverse import ExecutedActionInverse


@struct.dataclass
class State:
    obs: jax.Array
    reward: jax.Array
    done: jax.Array


class World:
    sys = jnp.asarray(0)
    horizon, dwell = 4, 1
    gains = jnp.asarray(((1, 1), (-1, 1), (1, -1), (-1, -1)), dtype=jnp.float32)

    @staticmethod
    def reset(sys, key):
        return State(jnp.zeros(2), jnp.asarray(0.0), jnp.asarray(0.0))

    @staticmethod
    def step(state, action):
        return state.replace(obs=state.obs + action, reward=-jnp.sum(action ** 2))


class Actor(nnx.Module):
    def deterministic(self, observations):
        return jnp.broadcast_to(jnp.asarray((0.4, 0.2)), (observations.shape[0], 2))


def small_model(obs, act, seed):
    return ExecutedActionInverse(obs, act, hidden_dim=8, ensemble_size=2, rngs=nnx.Rngs(seed))


def test_provider_and_primary_model_are_fixed_and_disjoint():
    assert p.PROVIDER_SEED not in p.POLICY_SEEDS
    assert p.PRIMARY_MODEL == 95103 and len(p.MODEL_SEEDS) == 3
    assert p.CONTROLS == (0.002, 1.0, 0.98)
    assert not set(p.EVENT_SEEDS) & set(run.common.p.EVENT_SEEDS)


def test_collection_labels_exclude_noise_and_reset_on_native_horizon():
    actor = Actor()
    collect = run.build_collection(World(), nnx.graphdef(actor), nnx.state(actor, nnx.Param))
    obs, following, command, target, modes = map(np.asarray, collect(jax.random.PRNGKey(7), 12))
    np.testing.assert_array_equal(modes, np.arange(12) % 4)
    np.testing.assert_allclose(target, command * np.asarray(World.gains)[modes])
    np.testing.assert_allclose(obs[::4], 0)
    assert np.any(np.abs(following - obs - target) > 1e-5)


def test_causal_command_cannot_use_the_completed_transition_or_true_mode():
    actor = Actor()
    evidence = lambda obs, command, following: jnp.asarray((-9.0, 9.0, -9.0, -9.0))
    episode = run.common.build_episode(World(), nnx.graphdef(actor), "causal_95103", evidence)
    values = episode(nnx.state(actor, nnx.Param), jax.random.PRNGKey(7), jnp.asarray((1, 2, 3, 0)))
    beliefs, commands = np.asarray(values[2]), np.asarray(values[5])
    np.testing.assert_allclose(beliefs[0], np.full(4, 0.25))
    np.testing.assert_allclose(commands[0], (0.4, 0.2))
    np.testing.assert_allclose(commands[1], (-0.4, 0.2))
    assert np.argmax(beliefs[1]) == 1


@pytest.mark.parametrize("route", ("oracle", "native_reference", "robust", "uncompensated", "fixed_1"))
def test_switch_window_mode_error_is_not_applicable_without_estimator(route):
    values = (np.ones(4), np.asarray((0, 1, 2, 3)), np.full((4, 4), 0.25),
              np.zeros(4), np.ones(4, dtype=bool), np.zeros((4, 2)), np.zeros((4, 2)))
    baseline = run.common.trajectory_metrics(values, 1, route)
    assert baseline["return"] == 4 and baseline["switches_observed"] == 3
    assert baseline["mode_accuracy"] is None
    assert all(window["mode_error_rate"] is None for window in baseline["switch_windows"].values())
    causal = run.common.trajectory_metrics(values, 1, "causal_95103")
    assert causal["switch_windows"]["1"]["mode_error_rate"] == 1.0


def test_estimator_checkpoint_restores_optimizer_and_next_update(tmp_path):
    model = small_model(2, 2, 7)
    params = nnx.state(model, nnx.Param)
    optimizer = optax.adamw(1e-4)
    opt_state = optimizer.init(params)
    update = run.training_update(nnx.graphdef(model), optimizer)
    obs = jax.random.normal(jax.random.PRNGKey(2), (8, 2))
    batch = (obs, obs + 0.1, jnp.ones((8, 2)) * 0.2)
    params, opt_state, loss = update(params, opt_state, *batch)
    assert np.isfinite(float(loss))
    path = tmp_path / "train_state.pkl"
    run.common.save_pickle(path, {"next_iteration": 1, "state": run.common._to_numpy_tree((params, opt_state))})
    saved = run.common.read_pickle(path)
    restored = run.common._restore_tree_like((params, opt_state), saved["state"], "estimator", allow_fallback=False)
    first = update(params, opt_state, *batch)
    second = update(*restored, *batch)
    assert saved["next_iteration"] == 1
    for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(second), strict=True):
        np.testing.assert_allclose(a, b, atol=1e-7)


def test_compact_export_restores_identical_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(run.common, "inverse_model", small_model)
    monkeypatch.setattr(p, "model_dir", lambda seed: tmp_path)
    model = small_model(2, 2, p.PRIMARY_MODEL)
    params = nnx.state(model, nnx.Param)
    variance = jnp.ones(2) * 0.05
    run.export_model(tmp_path, params, {
        "model_seed": p.PRIMARY_MODEL, "obs_dim": 2, "act_dim": 2,
        "runtime_versions": p.RUNTIME, "filter_controls": list(p.CONTROLS),
        "residual_variance": variance.tolist()})
    expected = run.common.inverse_evidence(model, params, World.gains, variance)
    restored = run.load_evidence(p.PRIMARY_MODEL, World.gains)
    args = (jnp.zeros(2), jnp.ones(2) * 0.2, jnp.ones(2) * 0.1)
    np.testing.assert_allclose(expected(*args), restored(*args), atol=1e-7)
    assert {path.name for path in tmp_path.iterdir()} == {"inverse_params.npz", "complete.json"}


def test_submission_dag_does_not_stage_or_sync_dataset_arrays():
    from scripts import submit_reacher_causal_v35 as submit

    rows = submit.candidates()
    assert len(rows) == 10
    assert sum(row["vram"] > 0 for row, output in rows) == 6
    outputs = {str(output) for row, output in rows}
    for row, output in rows:
        assert str(p.PREREG) in row["wait_for_files"]
        assert "local" not in row["allowed_nodes"]
        assert str(p.DATA_DIR) not in row["stage_input_paths"]
        assert row["result_dir"] != str(p.DATA_DIR)
        assert "run_reacher_causal_v35" in row["cmd"]
        if "/fit-" in row["signature"]:
            assert row["require_node"] == p.DATA_NODE
            assert str(p.DATA_MANIFEST) in row["wait_for_files"]
            assert not row["reroute_on_node_down"]
            assert row["resume_managed_by_cmd"]
        if "/audit-" in row["signature"]:
            assert row["vram"] == 0 and "JAX_PLATFORMS=cpu" in row["cmd"]
            assert submit.legacy.CPU_EVAL_PYTHON in row["cmd"]
            assert all(str(p.model_dir(seed) / "complete.json") in row["wait_for_files"] for seed in p.MODEL_SEEDS)
        if row["signature"].endswith("/aggregate"):
            assert set(row["wait_for_files"]) - {str(p.PREREG)} <= outputs


def test_analysis_uses_three_policy_seeds_and_rejects_truncated_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "ANALYSIS", tmp_path / "analysis.json")
    monkeypatch.setattr(p, "audit_path", lambda seed: tmp_path / f"{seed}.json")
    primary = f"causal_{p.PRIMARY_MODEL}"
    for index, seed in enumerate(p.POLICY_SEEDS):
        means = {"oracle": 2.0, "robust": 0.0, **{f"fixed_{mode}": -1.0 for mode in range(4)},
                 **{f"causal_{s}": 1 + 0.05 * index for s in p.MODEL_SEEDS}}
        event = {"steps": 200, "switches_observed": 3}
        p.audit_path(seed).write_text(json.dumps({
            "means": means, "events": {arm: [event] * 15 for arm in (primary, "oracle", "robust")},
            "stationary": {arm: [{"return": value}] for arm, value in means.items()}}))
    run.analyze()
    result = json.loads(p.ANALYSIS.read_text())
    assert result["causal_screen"]["passes"]
    assert result["comparisons"][f"{primary}-minus-robust"]["n_policy_seeds"] == 3
    path = p.audit_path(p.POLICY_SEEDS[0])
    row = json.loads(path.read_text())
    row["events"][primary][0]["steps"] = 199
    path.write_text(json.dumps(row))
    run.analyze()
    assert not json.loads(p.ANALYSIS.read_text())["causal_screen"]["passes"]
