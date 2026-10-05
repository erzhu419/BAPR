"""Narrow checks for the new causal comparison and scheduler DAG."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx, struct

from jax_experiments.analysis import mechanism_gap_protocol_v34 as p
from jax_experiments.analysis import run_mechanism_gap_v34 as run
from jax_experiments.common.checkpoint import load_checkpoint, save_checkpoint
from jax_experiments.common.logging import Logger
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.networks.policy import GaussianPolicy


@struct.dataclass
class ToyState:
    obs: jax.Array
    reward: jax.Array
    done: jax.Array


class ToyWorld:
    kind = "hc"
    horizon = 1000
    sys = jnp.asarray(0)
    gains = jnp.asarray(((1,), (-1,), (1,), (-1,)), dtype=jnp.float32)

    @staticmethod
    def reset(unused, key):
        return ToyState(jnp.zeros(1), jnp.asarray(0.0), jnp.asarray(0.0))

    @staticmethod
    def step(state, action):
        return state.replace(obs=state.obs + action, reward=jnp.sum(action))


def test_filter_matches_frozen_numpy_equation():
    from jax_experiments.analysis.regime_polarity_posterior import posterior_update, FilterConfig
    belief = np.asarray((0.1, 0.2, 0.3, 0.4))
    evidence = np.asarray((-2, 0, 1, -1))
    controls = (0.002, 1.0, 0.98)
    expected = posterior_update(belief, evidence, FilterConfig(*controls))
    np.testing.assert_allclose(run.update_belief(jnp.asarray(belief), jnp.asarray(evidence), controls), expected, atol=1e-7)
    instant = run.update_belief(jnp.asarray(belief), jnp.asarray(evidence), controls, temporal=False)
    np.testing.assert_allclose(instant, jax.nn.softmax(jnp.asarray(evidence)), atol=1e-7)


def test_rollout_stores_pre_action_and_post_transition_contexts():
    world = ToyWorld()
    policy = GaussianPolicy(5, 1, 8, rngs=nnx.Rngs(1))
    evidence = lambda observation, action, next_observation: jnp.asarray((-5.0, 5.0, -5.0, -5.0))
    scan = run.build_training_scan(world, nnx.graphdef(policy), evidence, True)
    _, transitions = scan(nnx.state(policy, nnx.Param), world.reset(0, None), jnp.full(4, 0.25),
                          jnp.asarray(0), jax.random.split(jax.random.PRNGKey(4), 3), jnp.asarray((0, 3, 2)))
    obs, unused, reward, next_obs, done = transitions
    np.testing.assert_allclose(obs[0, -4:], np.full(4, 0.25))
    np.testing.assert_allclose(obs[1:, -4:], next_obs[:-1, -4:])
    assert int(jnp.argmax(next_obs[0, -4:])) == 1
    # The second physical mode is 3, but the policy receives inferred mode 1.
    assert int(jnp.argmax(obs[1, -4:])) != 3


def test_conditional_reset_preserves_timeout_and_next_context():
    class TimeoutWorld(ToyWorld):
        kind = "reacher"
        horizon = 2

    world = TimeoutWorld()
    policy = GaussianPolicy(5, 1, 8, rngs=nnx.Rngs(1))
    evidence = lambda observation, action, next_observation: jnp.asarray((-5.0, 5.0, -5.0, -5.0))
    scan = run.build_training_scan(world, nnx.graphdef(policy), evidence, True)
    final, transitions = scan(
        nnx.state(policy, nnx.Param), world.reset(0, None), jnp.full(4, 0.25),
        jnp.asarray(0), jax.random.split(jax.random.PRNGKey(4), 4), jnp.asarray((0, 3, 2, 1)))
    obs, unused, reward, next_obs, done = transitions
    np.testing.assert_allclose(obs[2, :1], jnp.zeros(1))
    np.testing.assert_allclose(obs[2, -4:], jnp.full(4, 0.25))
    assert int(jnp.argmax(next_obs[1, -4:])) == 1
    np.testing.assert_allclose(final[0].obs, jnp.zeros(1))
    assert int(final[2]) == 0


def test_rollout_unroll_preserves_causal_contexts_and_timeout():
    from jax_experiments.analysis.benchmark_mechanism_gap_rollout import compare_outputs

    class TimeoutWorld(ToyWorld):
        kind = "reacher"
        horizon = 2

    world = TimeoutWorld()
    policy = GaussianPolicy(5, 1, 8, rngs=nnx.Rngs(1))
    evidence = lambda observation, action, next_observation: jnp.asarray((-5., 5., -5., -5.))
    inputs = (nnx.state(policy, nnx.Param), world.reset(0, None), jnp.full(4, 0.25),
              jnp.asarray(0), jax.random.split(jax.random.PRNGKey(4), 9),
              jnp.asarray((0, 3, 2, 1, 0, 1, 2, 3, 0)))
    baseline = run.build_training_scan(world, nnx.graphdef(policy), evidence, True)(*inputs)
    unrolled = run.build_training_scan(world, nnx.graphdef(policy), evidence, True, unroll=4)(*inputs)
    assert compare_outputs(baseline, unrolled, True)["equivalent"]
    changed = (unrolled[0], (*unrolled[1][:2], unrolled[1][2] + 0.1, *unrolled[1][3:]))
    assert not compare_outputs(baseline, changed, True)["equivalent"]


def test_sampling_screen_uses_each_native_environment_reward_and_termination(monkeypatch):
    from jax_experiments.analysis import benchmark_mechanism_gap_rollout as bench

    created = []

    class NativeEnv:
        obs_dim, act_dim = 1, 1
        base_sys = jnp.asarray(2.0)
        _tasks = [{"action_gain": [gain]} for gain in (1, -1, 1, -1)]

        def __init__(self, name, **kwargs):
            created.append((name, kwargs))

        @staticmethod
        def _physics_step(sys, old, action):
            return old + sys + action

        @staticmethod
        def _reward_obs(old, new, action):
            return jnp.sum(new - old), new, jnp.asarray(1.0)

        @staticmethod
        def _reset_state(sys, key):
            return None

    monkeypatch.setattr(bench, "StochasticModeEnv", NativeEnv)
    for kind, expected in (("ant", "Ant-v2"), ("hopper", "Hopper-v2"), ("walker", "Walker2d-v2")):
        world = bench.sampling_world(kind, 13)
        state = SimpleNamespace(pipeline_state=jnp.zeros(1))
        state.replace = lambda **kwargs: SimpleNamespace(**kwargs)
        following = world.step(state, jnp.ones(1))
        np.testing.assert_allclose(following.obs, [3.0])
        assert float(following.reward) == 3.0 and float(following.done) == 1.0
        assert created[-1] == (expected, {"family": "actuator_polarity", "backend": "spring", "dwell_steps": 250, "seed": 13})
        assert world.gains.shape == (4, 1)


def test_training_scan_accepts_native_reacher_reset_state():
    world = run.World("reacher", 13)
    world.horizon = 2
    policy = GaussianPolicy(world.obs_dim, world.act_dim, 8, rngs=nnx.Rngs(1))
    scan = run.build_training_scan(world, nnx.graphdef(policy), None, False)
    reset_key, step_key = jax.random.split(jax.random.PRNGKey(4))
    final, transitions = scan(
        nnx.state(policy, nnx.Param), world.reset(world.sys, reset_key), jnp.full(4, 0.25),
        jnp.asarray(0), jax.random.split(step_key, 4), jnp.asarray((0, 3, 2, 1)))
    assert int(final[2]) == 0
    assert all(np.isfinite(np.asarray(value)).all() for value in transitions)


def test_episode_scan_keeps_episode_rngs_and_replay_order():
    from jax_experiments.analysis.benchmark_mechanism_gap_rollout import compare_outputs, build_reacher_episode_scan

    class EpisodeWorld(ToyWorld):
        kind = "reacher"
        horizon = 3

        @staticmethod
        def reset(unused, key):
            return ToyState(jax.random.uniform(key, (1,)), jnp.asarray(0.0), jnp.asarray(0.0))

    world = EpisodeWorld()
    policy = GaussianPolicy(1, 1, 8, rngs=nnx.Rngs(1))
    graph = nnx.graphdef(policy)
    inputs = (nnx.state(policy, nnx.Param), world.reset(0, jax.random.PRNGKey(17)),
              jnp.full(4, 0.25), jnp.asarray(0),
              jax.random.split(jax.random.PRNGKey(4), 12),
              jnp.asarray((0, 3, 2, 1, 0, 1, 2, 3, 0, 3, 1, 2)))
    reference = run.build_training_scan(world, graph, None, False)(*inputs)
    parallel = build_reacher_episode_scan(world, graph)(*inputs)
    assert compare_outputs(reference, parallel, False)["equivalent"]


def test_episode_scan_uses_native_transitions():
    from jax_experiments.analysis.benchmark_mechanism_gap_rollout import compare_outputs, build_reacher_episode_scan

    world = run.World("reacher", 13)
    world.horizon = 8
    policy = GaussianPolicy(world.obs_dim, world.act_dim, 8, rngs=nnx.Rngs(1))
    graph = nnx.graphdef(policy)
    key, reset_key = jax.random.split(jax.random.PRNGKey(4))
    inputs = (nnx.state(policy, nnx.Param), world.reset(world.sys, reset_key),
              jnp.full(4, 0.25), jnp.asarray(0), jax.random.split(key, 32),
              jnp.asarray(run.training_modes(13, 32, 2)))
    baseline = run.build_training_scan(world, graph, None, False)(*inputs)
    parallel = build_reacher_episode_scan(world, graph)(*inputs)
    assert compare_outputs(baseline, parallel, False)["equivalent"]


def test_command_buffer_pair_compares_traces_and_keeps_only_json(tmp_path, monkeypatch):
    from jax_experiments.analysis import benchmark_mechanism_gap_rollout as bench

    calls = []

    def trial(command, env):
        calls.append((command, env))
        output = Path(command[command.index("--output") + 1])
        trajectory = Path(command[command.index("--trajectory") + 1])
        accelerated = len(calls) == 2
        output.write_text(json.dumps({"median_seconds": 1.0 if accelerated else 10.0}))
        np.savez(trajectory, np.zeros(4), np.ones((12, 1)))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(bench.subprocess, "run", trial)
    output = tmp_path / "timing.json"
    bench.command_buffer_pair(output, samples=12, repeats=1)
    data = json.loads(output.read_text())
    assert data["rows"][1]["equivalent"] and data["rows"][1]["speedup"] == 10.0
    assert "WHILE,CONDITIONAL" in calls[1][1]["XLA_FLAGS"]
    assert calls[0][0][3] == "jax_experiments.analysis.benchmark_mechanism_gap_rollout"
    assert list(tmp_path.iterdir()) == [output]


def test_physics_unroll_preserves_native_substep_count_and_reward():
    from jax_experiments.analysis.benchmark_mechanism_gap_rollout import compare_outputs

    for kind, factor in (("hc", 4), ("reacher", 2)):
        baseline = run.World(kind, 13, physics_unroll=1)
        optimized = run.World(kind, 13, physics_unroll=factor)
        state = baseline.reset(baseline.sys, jax.random.PRNGKey(4))
        action = jnp.full(baseline.act_dim, 0.2)
        first = jax.jit(baseline.step)(state, action)
        second = jax.jit(optimized.step)(state, action)
        assert compare_outputs(first, second, False)["equivalent"]


def test_frozen_compact_model_and_actor_load():
    model, params, variance, controls = run.frozen_inverse()
    assert model.ensemble_size == 5
    assert variance.shape == (6,)
    graph, actor = run.load_frozen_actor(p.PROVIDER_SEED)
    command = run.actor_function(graph, actor)(jnp.zeros(17))
    assert command.shape == (6,)
    gains = jnp.asarray(json.loads((p.RELEASE / "estimator/model_manifest.json").read_text())["mode_gain_vectors"])
    for single in (False, True):
        values = run.inverse_evidence(model, params, gains, variance, single)(jnp.zeros(17), command, jnp.ones(17) * 0.01)
        assert values.shape == (4,)
        assert np.isfinite(np.asarray(values)).all()


def test_checkpoint_preserves_iteration_optimizer_replay_and_rng(tmp_path):
    settings = run.config("hc", 13)
    settings.hidden_dim = 8
    identity = {"kind": "test", "seed": 13}
    original = run.ExperimentSAC(5, 1, settings, identity)
    replay = ReplayBuffer(5, 1, capacity=32)
    replay.push_batch_jax(jnp.ones((8, 5)), jnp.zeros((8, 1)), jnp.zeros(8), jnp.zeros((8, 5)), jnp.zeros(8))
    batch = replay.sample_stacked(2, 4, jax.random.PRNGKey(17))
    original.multi_update(batch)
    logger = Logger(str(tmp_path / "logs"))
    save_checkpoint(str(tmp_path / "checkpoint"), original, replay, logger, 0, 8, "sac")
    resumed = run.ExperimentSAC(5, 1, settings, identity)
    restored_replay = ReplayBuffer(5, 1, capacity=32)
    iteration, steps = load_checkpoint(str(tmp_path / "checkpoint"), resumed, restored_replay, logger, "sac")
    assert (iteration, steps, resumed.update_count, restored_replay.size) == (1, 8, 2, 8)
    original.multi_update(batch)
    resumed.multi_update(batch)
    for first, second in zip(jax.tree.leaves(nnx.state(original.policy, nnx.Param)), jax.tree.leaves(nnx.state(resumed.policy, nnx.Param))):
        np.testing.assert_allclose(first, second, atol=1e-7)


def test_cpu_admission_accepts_equal_returns_and_rejects_systematic_bias():
    from jax_experiments.analysis.audit_reacher_cpu_sampling_v34 import return_equivalence

    rewards = np.full((20, 200), -0.5, dtype=np.float32)
    identical = return_equivalence(rewards, rewards.copy())
    assert identical["passed"] and identical["episodes"] == 20
    assert identical["paired_difference_ci95"] == [0.0, 0.0]
    assert not return_equivalence(rewards, rewards - 0.01)["passed"]


def test_cpu_admission_rejects_uncertain_mean_even_when_average_matches():
    from jax_experiments.analysis.audit_reacher_cpu_sampling_v34 import return_equivalence

    rewards = np.full((20, 200), -0.5, dtype=np.float32)
    differences = np.tile(np.asarray((-0.1, 0.1)), 10)[:, None]
    result = return_equivalence(rewards, rewards + differences)
    np.testing.assert_allclose(result["cpu_mean"], result["gpu_mean"], atol=1e-12)
    assert not result["passed"]


def test_reacher_uses_native_reward_and_preserves_oracle_trajectory():
    world = run.World("reacher", 17)
    policy = GaussianPolicy(world.obs_dim, world.act_dim, 8, rngs=nnx.Rngs(17))
    graph, params = nnx.graphdef(policy), nnx.state(policy, nnx.Param)
    # A short functional trace is enough; the production audit checks 200 steps.
    world.horizon, world.dwell = 8, 2
    key, schedule = jax.random.PRNGKey(19), jnp.asarray((0, 1, 3, 2))
    oracle = run.build_episode(world, graph, "oracle")(params, key, schedule)
    native = run.build_episode(world, graph, "native_reference")(params, key, schedule)
    np.testing.assert_allclose(oracle[0], native[0], atol=1e-6)
    np.testing.assert_allclose(oracle[6], native[6], atol=1e-6)
    assert np.isfinite(np.asarray(oracle[0])).all()
    assert np.unique(np.asarray(oracle[1])).size == 4


def test_same_classifier_features_do_not_include_mode_labels():
    features = run.classifier_features(jnp.ones((3, 17)), jnp.zeros((3, 6)), jnp.ones((3, 17)) * 2)
    assert features.shape == (3, 40)
    classifier = run.DirectClassifier(17, 6, 13)
    assert classifier(jnp.ones((3, 17)), jnp.zeros((3, 6)), jnp.ones((3, 17))).shape == (3, 4)


def test_modes_are_reproducible_and_persistent():
    first = run.training_modes(13, 1000, 250)
    np.testing.assert_array_equal(first, run.training_modes(13, 1000, 250))
    assert all(np.unique(first[start:start+250]).size == 1 for start in range(0, 1000, 250))
    assert np.all(first[249::250][:-1] != first[250::250])


def test_scheduler_dag_uses_small_outputs_and_remote_nodes():
    location = p.ROOT / "scripts/submit_mechanism_gap_v34.py"
    spec = importlib.util.spec_from_file_location("submit_v34", location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = module.candidates()
    assert len(rows) == 40
    assert sum(row["vram"] > 0 for row, output in rows) == 20
    assert len({row["signature"] for row, output in rows}) == len(rows)
    for row, output in rows:
        assert "local" not in row["allowed_nodes"]
        assert "checkpoints" not in row["result_dir"]
        assert "identification_data" not in row["result_dir"]
        if row["vram"]:
            assert row["vram"] < 4096
            if not row["signature"].endswith("/smoke"):
                assert str(module.SMOKE_READY) in row["wait_for_files"]
        if row["signature"].endswith("-aggregate"):
            assert row["cpu"] == 1
            assert all(path.endswith("audit.json") for path in row["wait_for_files"][1:])
    fitting = [row for row, output in rows if "identification-fit" in row["signature"]]
    assert all(set(row["allowed_nodes"]) == set(module.GPU_NODES) for row in fitting)


def test_paired_intervals_use_seed_not_episode():
    summary = run.paired_summary([5, 6, 7], [1, 2, 3])
    assert summary["n_policy_seeds"] == 3
    assert summary["mean"] == 4
    assert summary["wins"] == 3
    assert summary["ci95"] == [4, 4]


def test_cpu_replacement_dag_is_isolated_and_waits_for_admission():
    from scripts import submit_mechanism_gap_v34 as submit

    rows = submit.reacher_cpu_candidates()
    assert len(rows) == 10 and sum(row["vram"] > 0 for row, unused in rows) == 6
    for row, output in rows:
        assert "--sampling-device cpu" in row["cmd"]
        assert "_v34_reacher_cpu/" in str(output)
        assert "local" not in row["allowed_nodes"]
        if row["vram"]:
            assert str(p.CPU_ADMISSION) in row["wait_for_files"]
            assert str(p.CPU_ADMISSION.parent) in row["stage_input_paths"]
            assert str(p.CPU_RUN_ROOT) in row["ckpt_dir"]
            assert "node007" not in row["allowed_nodes"]
            assert "JAX_PLATFORMS=cuda,cpu" in row["cmd"]
        else:
            assert submit.CPU_EVAL_PYTHON in row["cmd"]
