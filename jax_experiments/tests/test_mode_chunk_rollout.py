"""Check block fusion preserves mode clocks, reset boundaries and causal carry."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from types import SimpleNamespace
from brax.envs.base import State
from flax import nnx

from jax_experiments.analysis.benchmark_v39_sampling import compare
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.configs.default import Config
from jax_experiments.envs.stochastic_mode_env import StochasticModeEnv
from jax_experiments.envs.brax_env import _build_core_fns
from jax_experiments.train import collect_samples, make_algo


class ToyEnv(StochasticModeEnv):
    def __init__(self, seed):
        self.family, self.dwell_distribution = "actuator_polarity", "fixed"
        self.rollout_chunk_steps = self.dwell_steps = 6
        self.obs_dim, self.act_dim = 2, 1
        self.base_sys = self._current_sys = jnp.asarray(1.0)
        self._state = None
        self._reset_fn = jax.jit(self._reset_state)
        self.current_task_id = self._step_counter = 0
        self._next_switch_step = self.dwell_steps
        self._switching_enabled = True
        self._eval_mode_sequence = None
        self._tasks = [{} for _ in range(4)]
        self._mode_rng = np.random.RandomState(seed + 73001)
        self._switch_history = [(0, 0)]
        self._activate_mode(0)

    def _activate_mode(self, mode):
        self.current_task_id = int(mode)
        self._set_action_disturbance(gain=(-1.0 if mode % 2 else 1.0), noise_std=0.02)

    @staticmethod
    def _reset_state(sys, key):
        initial = jnp.array([jax.random.uniform(key) * 0.01, 0.0])
        return State(pipeline_state=initial, obs=initial, reward=jnp.float32(0),
                     done=jnp.float32(0), metrics={}, info={})

    @staticmethod
    def _physics_step(sys, state, action):
        return state + jnp.array([action[0] * 0.1, 1.0])

    @staticmethod
    def _reward_obs(old, new, action):
        return new[0], new, (new[1] >= 7).astype(jnp.float32)


def setup(recurrent, seed, baseline=False):
    config = Config(seed=seed, algo="escp" if recurrent else "sac", hidden_dim=8,
                    ensemble_size=2, start_train_steps=0, context_warmup_iters=3,
                    escp_context_mode="recurrent", escp_bottleneck_sigma=0.03,
                    escp_target_mode="twin_min", escp_actor_mode="twin_min",
                    task_num=4, escp_history_length=4)
    agent = make_algo(config.algo, 2, 1, config)
    env = ToyEnv(seed)
    if recurrent:
        graph, unused, non_params = nnx.split(agent.context_net, nnx.Param, ...)
        env.build_rollout_fn(nnx.graphdef(agent.policy), recurrent_context_graphdef=graph,
                             recurrent_context_non_params=non_params)
    else:
        env.build_rollout_fn(nnx.graphdef(agent.policy))
    if baseline:
        env.rollout_mode_chunks = None
    buffer = ReplayBuffer(2, 1, capacity=64)
    return config, agent, env, buffer


@pytest.mark.parametrize("recurrent", [False, True])
@pytest.mark.parametrize("iteration", [0, 5])
def test_fusion_preserves_transitions_replay_resets_and_clock(recurrent, iteration):
    baseline = setup(recurrent, 51, baseline=True)
    fused = setup(recurrent, 51)
    for _ in range(2):
        results = [collect_samples(agent, env, replay, cfg, 24, iteration)
                   for cfg, agent, env, replay in (baseline, fused)]
        assert compare(*results)["equivalent"]
        assert compare(baseline[2]._state, fused[2]._state)["equivalent"]
        assert baseline[2].get_switch_history() == fused[2].get_switch_history()
        assert baseline[2]._mode_rng.get_state()[2] == fused[2]._mode_rng.get_state()[2]
        assert (baseline[3].ptr, baseline[3].size) == (fused[3].ptr, fused[3].size)
        fields = ("obs", "act", "rew", "next_obs", "done", "task_id", "episode_start")
        assert compare([getattr(baseline[3], k) for k in fields],
                       [getattr(fused[3], k) for k in fields])["equivalent"]


def test_partial_block_uses_original_collector():
    config, agent, env, replay = setup(False, 52)
    env.rollout_mode_chunks = lambda *args, **kwargs: pytest.fail("partial blocks cannot be fused")
    rewards, recent = collect_samples(agent, env, replay, config, 25, 5)
    assert recent["obs"].shape == (25, 2)
    assert env._step_counter == 25
    assert len(rewards) == 3


def test_comparison_rejects_changed_action_or_nan():
    reference = {"act": np.array([0.1]), "done": np.array([0.0])}
    assert not compare(reference, {**reference, "act": np.array([0.2])})["equivalent"]
    assert not compare(reference, {**reference, "act": np.array([np.nan])})["equivalent"]


@pytest.mark.parametrize("recurrent", [False, True])
def test_replay_samples_and_updates_are_unchanged(recurrent):
    baseline = setup(recurrent, 53, baseline=True)
    fused = setup(recurrent, 53)
    batches = []
    for cfg, agent, env, replay in (baseline, fused):
        collect_samples(agent, env, replay, cfg, 24, 5)
        key = jax.random.PRNGKey(81)
        if recurrent:
            batch = replay.sample_stacked_sequences(2, 4, 4, rng_key=key)
        else:
            batch = replay.sample_stacked(2, 4, rng_key=key)
        batches.append(batch)
        agent.multi_update(batch, current_iter=5)
    assert compare(*batches)["equivalent"]
    assert baseline[1].update_count == fused[1].update_count == 2
    for name in ("policy", "critic", "target_critic"):
        assert compare(nnx.state(getattr(baseline[1], name), nnx.Param),
                       nnx.state(getattr(fused[1], name), nnx.Param))["equivalent"]


@pytest.mark.parametrize("name,backend,platform,expected", [
    ("HalfCheetah-v2", "spring", "gpu", True),
    ("HalfCheetah-v2", "spring", "cpu", 1),
    ("HalfCheetah-v2", "generalized", "gpu", 1),
    ("Ant-v2", "spring", "gpu", 1),
])
def test_native_gpu_unroll_preserves_frame_count_and_other_backends(monkeypatch, name, backend, platform, expected):
    calls = []
    original = jax.lax.scan

    def scan(*args, **kwargs):
        calls.append(kwargs.get("unroll", 1))
        return original(*args, **kwargs)

    monkeypatch.setattr(jax.lax, "scan", scan)
    native = SimpleNamespace(
        backend=backend, dt=0.05, _debug=False, _n_frames=16,
        sys=SimpleNamespace(gravity=SimpleNamespace(devices=lambda: [SimpleNamespace(platform=platform)])),
        _pipeline=SimpleNamespace(step=lambda sys, state, action, debug: state + action))
    step, unused, unused = _build_core_fns(native, name)
    result = step(None, jnp.array(0.0), jnp.array(1.0))
    assert float(result) == 16.0
    assert calls == [expected]
