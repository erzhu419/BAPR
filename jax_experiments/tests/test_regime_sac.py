import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.algos.regime_sac import RegimeSAC
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.configs.default import Config
from jax_experiments.train import collect_samples, make_algo


def make_config(source="robust"):
    config = Config()
    config.algo = "regime_sac"
    config.env_type = "stochastic_mode"
    config.regime_context_source = source
    config.task_num = 4
    config.test_task_num = 4
    config.hidden_dim = 8
    config.ensemble_size = 2
    config.batch_size = 4
    config.samples_per_iter = 4
    config.start_train_steps = 0
    config.context_warmup_iters = 0
    return config


def tasks():
    return [{"mode_id": mode} for mode in range(4)]


def parameter_arrays(module):
    return [
        np.asarray(value)
        for value in jax.tree.leaves(nnx.state(module, nnx.Param))
    ]


def test_equal_budget_arms_have_identical_initial_parameters():
    robust = RegimeSAC(3, 2, make_config("robust"), seed=17)
    oracle = RegimeSAC(3, 2, make_config("oracle"), seed=17)

    for robust_module, oracle_module in (
            (robust.policy, oracle.policy),
            (robust.critic, oracle.critic),
            (robust.target_critic, oracle.target_critic)):
        robust_values = parameter_arrays(robust_module)
        oracle_values = parameter_arrays(oracle_module)
        assert [value.shape for value in robust_values] == [
            value.shape for value in oracle_values]
        for robust_value, oracle_value in zip(
                robust_values, oracle_values):
            np.testing.assert_array_equal(robust_value, oracle_value)


def test_robust_context_is_zero_and_oracle_context_is_one_hot():
    robust = RegimeSAC(3, 2, make_config("robust"), seed=1)
    oracle = RegimeSAC(3, 2, make_config("oracle"), seed=1)
    robust.set_task_metadata(tasks())
    oracle.set_task_metadata(tasks())

    for mode in range(4):
        np.testing.assert_array_equal(
            np.asarray(robust.context_for_task_id(mode)),
            np.zeros(4, dtype=np.float32))
        np.testing.assert_array_equal(
            np.asarray(oracle.context_for_task_id(mode)),
            np.eye(4, dtype=np.float32)[mode])


def test_make_algo_exposes_direct_regime_sac():
    config = make_config("oracle")
    agent = make_algo("regime_sac", 3, 2, config)
    assert isinstance(agent, RegimeSAC)
    assert agent.belief_dim == config.task_num
    assert not hasattr(agent, "context_net")


class FakeSwitchingEnv:
    def __init__(self):
        self.current_task_id = 0
        self.rollout_chunk_steps = 2
        self.obs_dim = 3
        self.act_dim = 2

    def rollout(self, policy_params, n_steps, rng_key, **kwargs):
        del policy_params, rng_key, kwargs
        mode = self.current_task_id
        obs = jnp.full((n_steps, self.obs_dim), mode, dtype=jnp.float32)
        act = jnp.zeros((n_steps, self.act_dim), dtype=jnp.float32)
        rew = jnp.zeros((n_steps,), dtype=jnp.float32)
        next_obs = obs + 0.5
        done = jnp.zeros((n_steps,), dtype=jnp.float32)
        self.current_task_id = (mode + 1) % 4
        return (obs, act, rew, next_obs, done), []


def test_rollout_replay_context_changes_at_bellman_boundary():
    config = make_config("oracle")
    agent = RegimeSAC(3, 2, config, seed=3)
    agent.set_task_metadata(tasks())
    replay = ReplayBuffer(3, 2, capacity=16, belief_dim=4)

    _, recent = collect_samples(
        agent, FakeSwitchingEnv(), replay, config, n_steps=4,
        current_iter=0)

    expected_context = np.asarray([
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32)
    expected_next = np.asarray([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ], dtype=np.float32)
    np.testing.assert_array_equal(
        np.asarray(replay.belief[:4]), expected_context)
    np.testing.assert_array_equal(
        np.asarray(replay.next_belief[:4]), expected_next)
    np.testing.assert_array_equal(
        np.asarray(replay.task_id[:4]), np.asarray([0, 0, 1, 1]))
    np.testing.assert_array_equal(
        np.asarray(recent["context"]), expected_context)
    np.testing.assert_array_equal(
        np.asarray(recent["next_context"]), expected_next)


def test_conditioned_update_runs_and_reports_finite_metrics():
    config = make_config("oracle")
    agent = RegimeSAC(3, 2, config, seed=7)
    agent.set_task_metadata(tasks())
    context = jnp.eye(4, dtype=jnp.float32)[
        jnp.asarray([[0, 1, 2, 3], [3, 2, 1, 0]])]
    batch = {
        "obs": jnp.zeros((2, 4, 3), dtype=jnp.float32),
        "act": jnp.zeros((2, 4, 2), dtype=jnp.float32),
        "rew": jnp.ones((2, 4, 1), dtype=jnp.float32),
        "next_obs": jnp.full((2, 4, 3), 0.1, dtype=jnp.float32),
        "done": jnp.zeros((2, 4, 1), dtype=jnp.float32),
        "belief": context,
        "next_belief": context,
    }

    metrics = agent.multi_update(batch)

    assert agent.update_count == 2
    assert metrics["regime_context_oracle"] == 1.0
    assert all(np.isfinite(value) for value in metrics.values())


def test_eval_context_override_covers_true_zero_fixed_and_cyclic():
    agent = RegimeSAC(3, 2, make_config("oracle"), seed=9)
    agent.set_task_metadata(tasks())

    agent.set_eval_context_override("true")
    for mode in range(4):
        agent.set_oracle_task_id(mode)
        np.testing.assert_array_equal(
            np.asarray(agent.rollout_context()),
            np.eye(4, dtype=np.float32)[mode])

    agent.set_eval_context_override("zero")
    agent.set_oracle_task_id(2)
    np.testing.assert_array_equal(
        np.asarray(agent.rollout_context()),
        np.zeros(4, dtype=np.float32))

    agent.set_eval_context_override("fixed", 3)
    for mode in range(4):
        agent.set_oracle_task_id(mode)
        np.testing.assert_array_equal(
            np.asarray(agent.rollout_context()),
            np.eye(4, dtype=np.float32)[3])

    agent.set_eval_context_override("cyclic")
    for mode in range(4):
        agent.set_oracle_task_id(mode)
        np.testing.assert_array_equal(
            np.asarray(agent.rollout_context()),
            np.eye(4, dtype=np.float32)[(mode + 1) % 4])

    agent.set_eval_context_override("checkpoint")
    np.testing.assert_array_equal(
        np.asarray(agent.context_for_task_id(1)),
        np.eye(4, dtype=np.float32)[1])


def test_eval_context_override_covers_shuffled_context():
    agent = RegimeSAC(3, 2, make_config("oracle"), seed=10)
    agent.set_task_metadata(tasks())
    mapping = (2, 3, 1, 0)
    agent.set_eval_context_override(
        "shuffled", shuffled_mode_map=mapping)

    for mode in range(4):
        agent.set_oracle_task_id(mode)
        assert agent.eval_context_mode_id() == mapping[mode]
        np.testing.assert_array_equal(
            np.asarray(agent.rollout_context()),
            np.eye(4, dtype=np.float32)[mapping[mode]])


def test_delayed_context_holds_previous_mode_for_exact_action_count():
    agent = RegimeSAC(3, 2, make_config("oracle"), seed=11)
    agent.set_task_metadata(tasks())
    agent.set_eval_context_override("delayed", delay_steps=3)

    agent.reset_eval_context_state()
    agent.set_oracle_task_id(0)
    assert (agent.eval_context_mode_id(),
            agent.eval_context_delay_remaining()) == (0, 0)

    observed = []
    for _ in range(4):
        agent.set_oracle_task_id(2)
        observed.append((
            agent.eval_context_mode_id(),
            agent.eval_context_delay_remaining()))
    assert observed == [(0, 3), (0, 2), (0, 1), (2, 0)]

    agent.reset_eval_context_state()
    agent.set_oracle_task_id(3)
    assert (agent.eval_context_mode_id(),
            agent.eval_context_delay_remaining()) == (3, 0)
