import jax.numpy as jnp
import numpy as np
from jax_experiments.analysis import analyze_regime_control_headroom

from jax_experiments.analysis import final_task_sweep
from jax_experiments.analysis import regime_cross_context as cross_protocol
from jax_experiments.analysis import regime_control_headroom as protocol
from jax_experiments.configs.default import Config
from scripts import submit_regime_control_headroom as submitter
from scripts import submit_regime_cross_context as cross_submitter


class FakeDirectAgent:
    uses_regime_context = True
    context_mode = "oracle"

    def __init__(self):
        self.current_mode = 0
        self.mode_calls = []

    def set_eval_task(self, task):
        self.set_oracle_task_id(int(task["mode_id"]))

    def set_oracle_task_id(self, mode_id):
        self.current_mode = int(mode_id)
        self.mode_calls.append(self.current_mode)

    def _build_belief_jax(self):
        return jnp.eye(4, dtype=jnp.float32)[self.current_mode]

    def select_action(self, obs, deterministic=True):
        del obs, deterministic
        return np.asarray([self.current_mode], dtype=np.float32)


class FakeStationaryEnv:
    def __init__(self):
        self.contexts = []

    def set_task(self, task):
        self.task = task

    def eval_rollout(self, policy_params, n_steps, rng_key,
                     context_params=None, belief_vec=None,
                     episode_horizon=None):
        del policy_params, rng_key, context_params, episode_horizon
        self.contexts.append(np.asarray(belief_vec))
        return np.zeros(n_steps), np.zeros(n_steps)


class FakeSwitchingEnv:
    def __init__(self):
        self.current_task_id = 0
        self.step_index = 0

    def reset(self):
        return np.zeros(1, dtype=np.float32)

    def task_id_for_next_step(self):
        return self.current_task_id

    def step(self, action):
        assert int(action[0]) == self.current_task_id
        self.step_index += 1
        if self.step_index % 2 == 0:
            self.current_task_id = 1 - self.current_task_id
        return np.zeros(1, dtype=np.float32), 1.0, False, {}


def test_final_sweep_refreshes_direct_context_for_stationary_tasks():
    agent = FakeDirectAgent()
    env = FakeStationaryEnv()
    config = Config()
    config.max_episode_steps = 2
    tasks = [{"mode_id": 0}, {"mode_id": 2}]
    original = final_task_sweep._eval_policy_state
    final_task_sweep._eval_policy_state = lambda *_: (None, None, None)
    try:
        rows = final_task_sweep.evaluate_task_split(
            agent, env, config, tasks, "test", 1, None, 7)
    finally:
        final_task_sweep._eval_policy_state = original
    assert len(rows) == 2
    np.testing.assert_array_equal(env.contexts[0], np.eye(4)[0])
    np.testing.assert_array_equal(env.contexts[1], np.eye(4)[2])


def test_final_sweep_sets_mode_before_every_switching_action():
    agent = FakeDirectAgent()
    env = FakeSwitchingEnv()
    config = Config()
    config.max_episode_steps = 4
    tasks = [{"mode_id": 0}, {"mode_id": 1}]
    original_select = final_task_sweep._select_eval_switch_sequence
    original_reset = final_task_sweep._reset_eval_switch_schedule
    final_task_sweep._select_eval_switch_sequence = (
        lambda *_: (tasks, [0, 1]))
    final_task_sweep._reset_eval_switch_schedule = lambda *_: None
    try:
        rows, trace = final_task_sweep.evaluate_switching(
            agent, env, config, tasks, 1, 2, 11)
    finally:
        final_task_sweep._select_eval_switch_sequence = original_select
        final_task_sweep._reset_eval_switch_schedule = original_reset
    assert len(rows) == 1
    assert len(trace) == 4
    assert agent.mode_calls == [0, 0, 1, 1]


def test_scheduler_chain_is_unpinned_and_file_gated():
    training = submitter.candidates("training", "high")
    audits = submitter.candidates("audit", "high")
    analysis = submitter.candidates("analysis", "high")
    assert len(training) == 30
    assert len(audits) == 30
    assert len(analysis) == 1
    signatures = [row[0] for row in training + audits + analysis]
    assert len(signatures) == len(set(signatures))
    for _, spec, _ in training:
        assert spec["vram"] == 2048
        assert "allowed_nodes" not in spec
        assert spec["resume_managed_by_cmd"] is True
        assert spec["allow_initial_resume_scan_error"] is False
    for _, spec, _ in audits:
        assert spec["vram"] == 0
        assert spec["allowed_nodes"] == submitter.CPU_NODES
        assert len(spec["wait_for_files"]) == 4
        assert all("bundle" in path for path in spec["wait_for_files"])
    _, aggregate, _ = analysis[0]
    assert len(aggregate["wait_for_files"]) == 30


def test_protocol_budget_and_gate_matrix_are_frozen():
    assert protocol.FAMILY == "structured_channel"
    assert protocol.DWELL_STEPS == 250


def test_preregistered_gate_uses_training_seeds_as_inference_units():
    env_data = {}
    for seed in protocol.TRAINING_SEEDS:
        env_data[seed] = {}
        for role, score in (("robust", 100.0), ("oracle", 120.0)):
            env_data[seed][role] = {}
            for event_seed in protocol.AUDIT_EVENT_SEEDS:
                env_data[seed][role][event_seed] = {
                    "mode_returns": {
                        mode: score + mode for mode in protocol.MODES},
                    "stationary_termination": 0.0,
                    "switching_returns": [
                        score for _ in range(protocol.SWITCHING_EPISODES)],
                    "switching_termination": [
                        0.0 for _ in range(protocol.SWITCHING_EPISODES)],
                }

    result = analyze_regime_control_headroom._analyze_env(env_data)

    assert result["env_gate_pass"] is True
    assert result["mode_wins"] == 4
    assert result["switching_relative_gain"] == 0.2
    assert result["worst_mode_relative_gain"] == 0.2
    assert result["paired_deltas"]["switching"][
        "n_training_seeds"] == 5
    assert result["paired_deltas"]["switching"]["ci95_low"] == 20.0
    assert protocol.FINAL_TOTAL_STEPS == 5_600_000
    assert protocol.FINAL_UPDATE_COUNT == 350_000
    assert len(protocol.ENVS) * len(protocol.ROLES) * len(
        protocol.TRAINING_SEEDS) == 30


def test_cross_context_audits_are_split_cpu_only_and_file_gated():
    audits = cross_submitter.candidates("audit", "high")
    analysis = cross_submitter.candidates("analysis", "high")
    expected = (
        len(protocol.ENVS)
        * len(protocol.TRAINING_SEEDS)
        * len(protocol.AUDIT_EVENT_SEEDS))
    assert len(audits) == expected == 75
    assert len(analysis) == 1
    assert cross_protocol.CASE_LABELS == (
        "robust_model", "true", "zero", "fixed_0",
        "fixed_1", "fixed_2", "fixed_3", "cyclic")
    signatures = [row[0] for row in audits + analysis]
    assert len(signatures) == len(set(signatures))
    for _, spec, _ in audits:
        assert spec["vram"] == 0
        assert spec["cpu"] == 32
        assert "OMP_NUM_THREADS=1" in spec["cmd"]
        assert "xla_cpu_multi_thread_eigen=false" in spec["cmd"]
        assert spec["allowed_nodes"] == cross_submitter.CPU_NODES
        assert len(spec["wait_for_files"]) == 8
        assert len(spec["stage_input_paths"]) == 2
        assert spec["allow_cpu_training"] is True
    _, aggregate, _ = analysis[0]
    assert len(aggregate["wait_for_files"]) == expected
