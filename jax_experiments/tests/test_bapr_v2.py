import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from pathlib import Path
from tempfile import TemporaryDirectory

from jax_experiments.algos.bapr_v2 import (
    BAPRv2,
    _gravity_latent,
    _oracle_context_from_task_ids,
)
from jax_experiments.analysis.final_task_sweep import (
    binary_auc,
    evaluate_switching,
    finite_correlation,
    resolve_bapr_v2_eval_mode,
    switch_detection_metrics,
)
from jax_experiments.analysis.bapr_v2_final_sweep import completed_result_runs
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.configs.default import Config
from jax_experiments.networks.residual_policy import (
    ResidualGaussianPolicy,
    conservative_q_advantage,
)
from jax_experiments.networks.transition_context import TransitionContextEncoder
from jax_experiments.train import (
    _eval_task_id_for_action,
    _select_eval_switch_sequence,
    paired_safe_target,
)


def test_residual_policy_gate_zero_is_base_policy():
    policy = ResidualGaussianPolicy(
        obs_dim=5, act_dim=2, hidden_dim=16, latent_dim=3,
        residual_delta=0.25, rngs=nnx.Rngs(0))
    obs = jnp.ones((4, 5), dtype=jnp.float32)
    context = jnp.concatenate([
        jnp.ones((4, 3), dtype=jnp.float32),
        jnp.zeros((4, 1), dtype=jnp.float32),
    ], axis=-1)
    np.testing.assert_allclose(
        np.asarray(policy.deterministic(obs, context)),
        np.asarray(policy.base_deterministic(obs)),
        atol=1e-6)


def test_direct_policy_gate_zero_is_base_policy():
    policy = ResidualGaussianPolicy(
        obs_dim=5, act_dim=2, hidden_dim=16, latent_dim=3,
        policy_mode="direct", rngs=nnx.Rngs(3))
    obs = jnp.ones((4, 5), dtype=jnp.float32)
    context = jnp.concatenate([
        jnp.ones((4, 3), dtype=jnp.float32),
        jnp.zeros((4, 1), dtype=jnp.float32),
    ], axis=-1)
    np.testing.assert_allclose(
        np.asarray(policy.deterministic(obs, context)),
        np.asarray(policy.base_deterministic(obs)),
        atol=1e-6)


def test_direct_teacher_warmstart_exactly_copies_robust_policy():
    policy = ResidualGaussianPolicy(
        obs_dim=5, act_dim=2, hidden_dim=16, latent_dim=3,
        policy_mode="direct", rngs=nnx.Rngs(31))
    obs = jax.random.normal(jax.random.PRNGKey(32), (4, 5))
    context = jnp.concatenate([
        jax.random.normal(jax.random.PRNGKey(33), (4, 3)),
        jnp.ones((4, 1)),
    ], axis=-1)

    assert policy.warmstart_conditioned_from_base()
    direct_mean, direct_log_std = policy(obs, context)
    base_mean, base_log_std = policy(obs, None)
    np.testing.assert_allclose(direct_mean, base_mean, atol=1e-6)
    np.testing.assert_allclose(direct_log_std, base_log_std, atol=1e-6)


def test_gated_direct_warmstart_is_safe_and_gate_is_continuous():
    policy = ResidualGaussianPolicy(
        obs_dim=5, act_dim=2, hidden_dim=16, latent_dim=3,
        policy_mode="gated_direct", policy_gate_init=-2.0,
        rngs=nnx.Rngs(36))
    obs = jnp.ones((4, 5), dtype=jnp.float32)
    context = jnp.concatenate([
        jnp.ones((4, 3), dtype=jnp.float32),
        jnp.ones((4, 1), dtype=jnp.float32),
    ], axis=-1)
    expected = float(jax.nn.sigmoid(jnp.asarray(-2.0)))
    np.testing.assert_allclose(
        np.asarray(policy.adaptation_strength(obs, context)),
        np.full((4, 1), expected), atol=1e-6)

    assert policy.warmstart_conditioned_from_base()
    np.testing.assert_allclose(
        np.asarray(policy.deterministic(obs, context)),
        np.asarray(policy.base_deterministic(obs)), atol=1e-6)
    zero_context = context.at[:, -1].set(0.0)
    np.testing.assert_allclose(
        np.asarray(policy.adaptation_strength(obs, zero_context)),
        np.zeros((4, 1)), atol=1e-6)


def test_expert_policy_routes_extreme_oracle_latents():
    policy = ResidualGaussianPolicy(
        obs_dim=5, act_dim=2, hidden_dim=16, latent_dim=3,
        policy_mode="expert", num_experts=5, rngs=nnx.Rngs(4))
    obs = jnp.ones((2, 5), dtype=jnp.float32)
    context = jnp.asarray([
        [-1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
    ])
    actions = np.asarray(policy.deterministic(obs, context))
    assert not np.allclose(actions[0], actions[1])


def test_categorical_experts_warmstart_as_exact_robust_copies():
    policy = ResidualGaussianPolicy(
        obs_dim=5, act_dim=2, hidden_dim=16, latent_dim=4,
        policy_mode="categorical_expert", num_experts=4,
        rngs=nnx.Rngs(41))
    obs = jax.random.normal(jax.random.PRNGKey(42), (7, 5))

    assert policy.warmstart_conditioned_from_base()
    base_mean, base_log_std = policy(obs, None)
    for mode in range(4):
        latent = jax.nn.one_hot(
            jnp.full((len(obs),), mode), 4, dtype=jnp.float32)
        context = jnp.concatenate([latent, jnp.ones((len(obs), 1))], axis=-1)
        expert_mean, expert_log_std = policy(obs, context)
        np.testing.assert_allclose(expert_mean, base_mean, atol=1e-6)
        np.testing.assert_allclose(expert_log_std, base_log_std, atol=1e-6)


def test_categorical_experts_route_one_hot_and_mix_posterior():
    policy = ResidualGaussianPolicy(
        obs_dim=3, act_dim=1, hidden_dim=8, latent_dim=4,
        policy_mode="categorical_expert", num_experts=4,
        rngs=nnx.Rngs(43))
    obs = jnp.ones((1, 3), dtype=jnp.float32)
    for mode, head in enumerate(policy.expert_means):
        head.kernel.value = jnp.zeros_like(head.kernel.value)
        head.bias.value = jnp.full_like(head.bias.value, float(mode))
    for head in policy.expert_log_stds:
        head.kernel.value = jnp.zeros_like(head.kernel.value)
        head.bias.value = jnp.zeros_like(head.bias.value)

    means = []
    for mode in range(4):
        latent = jax.nn.one_hot(jnp.asarray([mode]), 4, dtype=jnp.float32)
        context = jnp.concatenate([latent, jnp.ones((1, 1))], axis=-1)
        mean, _ = policy(obs, context)
        means.append(float(mean[0, 0]))
    np.testing.assert_allclose(means, np.arange(4, dtype=np.float32))

    posterior = jnp.asarray([[0.25, 0.25, 0.25, 0.25, 1.0]])
    mixed_mean, _ = policy(obs, posterior)
    np.testing.assert_allclose(float(mixed_mean[0, 0]), 1.5, atol=1e-6)


def test_categorical_experts_require_one_expert_per_latent_coordinate():
    with np.testing.assert_raises_regex(
            ValueError, "num_experts == latent_dim"):
        ResidualGaussianPolicy(
            obs_dim=3, act_dim=1, hidden_dim=8, latent_dim=4,
            policy_mode="categorical_expert", num_experts=3,
            rngs=nnx.Rngs(44))


def test_oracle_context_uses_privileged_latent_and_unit_gate():
    encoder = TransitionContextEncoder(
        obs_dim=5, act_dim=2, latent_dim=3, hidden_dim=8,
        mode="oracle", rngs=nnx.Rngs(1))
    context = encoder.policy_context(
        encoder.initial_state(), jnp.asarray([0.2, -0.3, 0.4]))
    np.testing.assert_allclose(
        np.asarray(context), np.asarray([0.2, -0.3, 0.4, 1.0]),
        atol=1e-6)


def test_pow1p5_oracle_latent_uses_matching_log_base():
    high = {"gravity": np.asarray([0.0, 0.0, -9.81 * 1.5 ** 3])}
    low = {"gravity": np.asarray([0.0, 0.0, -9.81 * 1.5 ** -3])}
    assert np.isclose(
        _gravity_latent(
            high, 2, 3.0, "pow1p5", "task_distribution")[0], 1.0)
    assert np.isclose(
        _gravity_latent(
            low, 2, 3.0, "pow1p5", "task_distribution")[0], -1.0)


def test_oracle_teacher_context_is_rebuilt_from_replay_task_ids():
    task_latents = jnp.asarray([
        [-1.0, 0.0],
        [0.25, 0.5],
        [1.0, -0.5],
    ])
    task_ids = jnp.asarray([[2, 0], [1, 4]])
    context = np.asarray(
        _oracle_context_from_task_ids(task_latents, task_ids))
    np.testing.assert_allclose(
        context,
        np.asarray([
            [[1.0, -0.5, 1.0], [-1.0, 0.0, 1.0]],
            [[0.25, 0.5, 1.0], [0.25, 0.5, 1.0]],
        ]),
    )


def test_teacher_student_schedule_freezes_expected_controller_parts():
    config = Config(
        task_num=4,
        hidden_dim=16,
        ensemble_size=2,
        bapr_v2_training_schedule="teacher_student",
        bapr_v2_base_pretrain_iters=10,
        bapr_v2_teacher_iters=20,
    )
    agent = BAPRv2(5, 2, config, seed=7)

    assert agent.training_stage(9) == "robust"
    assert agent.controller_update_flags(9) == (True, False, True)
    assert agent.rollout_context_source(9) == agent.CONTEXT_ROBUST
    assert agent.training_stage(10) == "teacher"
    assert agent.controller_update_flags(10) == (False, True, True)
    assert agent.rollout_context_source(10) == agent.CONTEXT_ORACLE
    assert agent.training_stage(30) == "student"
    assert agent.controller_update_flags(30) == (False, False, False)
    assert agent.rollout_context_source(30) == agent.CONTEXT_LEARNED

    base_values = jax.tree.leaves(agent._policy_base_mask)
    residual_values = jax.tree.leaves(agent._policy_residual_mask)
    assert any(float(value) == 1.0 for value in base_values)
    assert any(float(value) == 1.0 for value in residual_values)


def test_constrained_deploy_schedule_uses_learned_context_and_resets_replay():
    config = Config(
        task_num=4,
        hidden_dim=16,
        ensemble_size=2,
        bapr_v2_training_schedule="constrained_deploy",
        bapr_v2_base_pretrain_iters=10,
        bapr_v2_teacher_iters=20,
        bapr_v2_student_iters=5,
        bapr_v2_freeze_gate_in_teacher=True,
    )
    agent = BAPRv2(5, 2, config, seed=87)
    assert agent.training_stage(9) == "robust"
    assert agent.training_stage(10) == "teacher"
    assert agent.training_stage(30) == "student"
    assert agent.training_stage(35) == "deployment"
    assert agent.controller_update_flags(35) == (False, True, True)
    assert agent.rollout_context_source(35) == agent.CONTEXT_LEARNED
    assert not agent.train_policy_gate(10)
    assert agent.train_policy_gate(35)
    agent.set_training_iteration(35)
    assert agent.consume_replay_reset_request()
    assert not agent.consume_replay_reset_request()


def test_paired_safe_target_penalizes_termination_risk():
    config = Config(
        bapr_v2_paired_gain_margin=0.0,
        bapr_v2_paired_gain_temperature=0.05,
        bapr_v2_paired_risk_tolerance=0.0,
        bapr_v2_paired_risk_temperature=0.05,
    )
    safe, gain, risk = paired_safe_target(
        100.0, 150.0, 0.0, 0.0, config)
    unsafe, _, unsafe_risk = paired_safe_target(
        100.0, 150.0, 0.0, 1.0, config)
    assert gain > 0.0
    assert risk == 0.0
    assert unsafe_risk == 1.0
    assert safe > 0.4
    assert unsafe < 1e-6


def test_context_training_batch_includes_switch_boundaries():
    config = Config(
        task_num=4,
        hidden_dim=16,
        ensemble_size=2,
        bapr_v2_context_length=8,
        bapr_v2_context_chunks=4,
        bapr_v2_context_burnin=2,
    )
    agent = BAPRv2(3, 2, config, seed=88)
    n_steps = 32
    rollout = {
        "obs": jnp.zeros((n_steps, 3)),
        "act": jnp.zeros((n_steps, 2)),
        "rew": jnp.zeros((n_steps, 1)),
        "next_obs": jnp.zeros((n_steps, 3)),
        "done": jnp.zeros((n_steps, 1)),
        "task_id": jnp.asarray([0] * 8 + [1] * 8 + [2] * 8 + [3] * 8),
    }
    *_, task_ids, switch_chunks = agent._context_training_batch(rollout)
    assert task_ids.shape == (4, 8)
    assert switch_chunks >= 1


def test_teacher_transition_warmstarts_direct_branch_once():
    config = Config(
        task_num=4,
        hidden_dim=16,
        ensemble_size=2,
        bapr_v2_policy_mode="direct",
        bapr_v2_training_schedule="teacher_student",
        bapr_v2_base_pretrain_iters=10,
        bapr_v2_teacher_iters=20,
        bapr_v2_warmstart_conditioned=True,
    )
    agent = BAPRv2(5, 2, config, seed=34)
    assert not agent._conditioned_warmstarted
    agent.set_training_iteration(10)
    assert agent._conditioned_warmstarted

    obs = jnp.ones((3, 5), dtype=jnp.float32)
    context = jnp.concatenate([
        jnp.ones((3, agent.latent_dim), dtype=jnp.float32),
        jnp.ones((3, 1), dtype=jnp.float32),
    ], axis=-1)
    np.testing.assert_allclose(
        np.asarray(agent.policy.deterministic(obs, context)),
        np.asarray(agent.policy.base_deterministic(obs)), atol=1e-6)

    resumed = BAPRv2(5, 2, config, seed=35)
    resumed.load_checkpoint_state(agent.checkpoint_state())
    assert resumed.training_stage() == "teacher"
    assert resumed._conditioned_warmstarted


def test_policy_ladder_can_force_base_or_oracle_from_one_checkpoint():
    config = Config(
        task_num=4,
        hidden_dim=16,
        ensemble_size=2,
        bapr_v2_training_schedule="teacher_student",
        bapr_v2_base_pretrain_iters=10,
        bapr_v2_teacher_iters=20,
        bapr_v2_advantage_gate=True,
    )
    agent = BAPRv2(5, 2, config, seed=8)
    agent.set_training_iteration(30)
    agent.oracle_latent = jnp.asarray(
        [0.4] + [0.0] * (agent.latent_dim - 1), dtype=jnp.float32)

    robust = np.asarray(agent.context_for_source(agent.CONTEXT_ROBUST))
    oracle = np.asarray(agent.context_for_source(agent.CONTEXT_ORACLE))
    np.testing.assert_allclose(robust, np.zeros(agent.context_dim), atol=1e-6)
    np.testing.assert_allclose(
        oracle,
        np.asarray([0.4] + [0.0] * (agent.latent_dim - 1) + [1.0]))

    obs = np.zeros(5, dtype=np.float32)
    assert np.isfinite(agent.select_action(
        obs, deterministic=True, context_source=agent.CONTEXT_ROBUST,
        advantage_enabled=False)).all()
    assert agent._last_context_gate == 0.0
    assert np.isfinite(agent.select_action(
        obs, deterministic=True, context_source=agent.CONTEXT_ORACLE,
        advantage_enabled=False)).all()
    assert agent._last_context_gate == 1.0

    source, enabled = resolve_bapr_v2_eval_mode(
        agent, "checkpoint", "checkpoint")
    assert source == agent.CONTEXT_LEARNED
    assert enabled is True
    assert resolve_bapr_v2_eval_mode(
        agent, "robust", "off") == (agent.CONTEXT_ROBUST, False)


def test_conservative_advantage_requires_positive_ensemble_lcb():
    adaptive = jnp.asarray([[3.0, 2.0], [1.0, 2.0]])
    base = jnp.asarray([[1.0, 2.0], [1.0, 2.0]])
    score = np.asarray(conservative_q_advantage(
        adaptive, base, lcb_scale=1.0))
    np.testing.assert_allclose(score, np.asarray([0.0, 0.0]))
    assert not np.any(score > 0.0)


def test_oracle_eval_peeks_at_upcoming_physics_task_only():
    class Agent:
        context_mode = "oracle"

    class LearnedAgent:
        context_mode = "supervised"

    class Env:
        @staticmethod
        def task_id_for_next_step():
            return 3

    assert _eval_task_id_for_action(Agent(), Env(), 2) == 3
    assert _eval_task_id_for_action(LearnedAgent(), Env(), 2) == 2
    learned = LearnedAgent()
    learned.CONTEXT_ORACLE = 1
    learned.CONTEXT_LEARNED = 2
    assert _eval_task_id_for_action(
        learned, Env(), 2, context_source=learned.CONTEXT_ORACLE) == 3
    assert _eval_task_id_for_action(
        learned, Env(), 2, context_source=learned.CONTEXT_LEARNED) == 2


def test_switch_sequence_uses_farthest_tasks_and_alternates_direction():
    tasks = [
        {"gravity": np.asarray([0.0, 0.0, -9.81])},
        {"gravity": np.asarray([0.0, 0.0, -9.81 * 1.5 ** 3])},
        {"gravity": np.asarray([0.0, 0.0, -9.81 * 1.5 ** -3])},
        {"gravity": np.asarray([0.0, 0.0, -9.81 * 1.5])},
    ]
    forward, forward_indices = _select_eval_switch_sequence(
        object(), tasks, episode_index=0)
    reverse, reverse_indices = _select_eval_switch_sequence(
        object(), tasks, episode_index=1)

    assert forward_indices == [1, 2]
    assert reverse_indices == [2, 1]
    np.testing.assert_allclose(forward[0]["gravity"], tasks[1]["gravity"])
    np.testing.assert_allclose(reverse[0]["gravity"], tasks[2]["gravity"])


def test_switch_auc_uses_event_error_ranking():
    assert np.isclose(binary_auc([True, False, False], [1.0, 0.1, 0.2]), 1.0)


def test_finite_correlation_ignores_nonfinite_pairs():
    assert np.isclose(
        finite_correlation([0.0, 1.0, np.nan], [0.0, 2.0, 3.0]), 1.0)


def test_switch_detection_delay_uses_first_post_switch_threshold_crossing():
    trace = [
        {"episode": 0, "step": 9, "switched": False,
         "context_error": 0.05},
        {"episode": 0, "step": 10, "switched": True,
         "context_error": 0.10},
        {"episode": 0, "step": 11, "switched": False,
         "context_error": 0.25},
    ]
    metrics = switch_detection_metrics(trace, threshold=0.20,
                                       max_delay_steps=50)
    assert metrics["detection_events"] == 1
    assert metrics["detection_rate"] == 1.0
    assert metrics["median_detection_delay"] == 1.0


def test_switching_sweep_continues_after_termination():
    class Agent:
        @staticmethod
        def select_action(obs, deterministic=False):
            del obs, deterministic
            return np.zeros((1,), dtype=np.float32)

    class Env:
        def __init__(self):
            self.current_task_id = 0
            self._step_counter = 0
            self.period = 3

        def set_nonstationary_para(self, tasks, changing_period,
                                   changing_interval):
            del tasks, changing_interval
            self.current_task_id = 0
            self._step_counter = 0
            self.period = changing_period

        def reset(self):
            return np.zeros((1,), dtype=np.float32)

        def task_id_for_next_step(self):
            return ((self._step_counter + 1) // self.period) % 2

        def step(self, action):
            del action
            self._step_counter += 1
            self.current_task_id = (self._step_counter // self.period) % 2
            done = self._step_counter in (2, 5)
            return np.zeros((1,), dtype=np.float32), 1.0, done, {}

    config = Config(max_episode_steps=6)
    rows, trace = evaluate_switching(
        Agent(), Env(), config, [{}, {}], episodes=1,
        period_steps=3, rng_seed=0)
    assert len(trace) == 6
    assert rows[0]["return"] == 6.0
    assert rows[0]["termination_count"] == 2
    assert rows[0]["switch_count"] == 2
    assert rows[0]["switch_sequence_source_indices"] == "0|1"


def test_transition_context_is_causal_and_finite():
    encoder = TransitionContextEncoder(
        obs_dim=5, act_dim=2, latent_dim=3, hidden_dim=8,
        mode="supervised", min_history=4, rngs=nnx.Rngs(2))
    state = encoder.initial_state()
    gates = []
    for _ in range(8):
        context = encoder.policy_context(state, jnp.zeros((3,)))
        gates.append(float(context[-1]))
        state, error, _, _ = encoder.observe(
            state, jnp.zeros((5,)), jnp.zeros((2,)), 0.0,
            jnp.zeros((5,)), 0.0, enable_reset=False)
        assert np.isfinite(float(error))
    assert int(state[2]) == 8
    assert gates[-1] > gates[0]


def test_replay_roundtrip_preserves_next_context():
    buffer = ReplayBuffer(3, 2, capacity=8, belief_dim=4)
    obs = jnp.arange(9, dtype=jnp.float32).reshape(3, 3)
    context = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    next_context = context + 1.0
    buffer.push_batch_jax(
        obs, jnp.zeros((3, 2)), jnp.ones((3, 1)), obs + 0.1,
        jnp.zeros((3, 1)), jnp.arange(3),
        belief=context, next_belief=next_context)
    saved = buffer.to_numpy()
    restored = ReplayBuffer(3, 2, capacity=8, belief_dim=4)
    restored.from_numpy(saved)
    np.testing.assert_allclose(
        np.asarray(restored.belief[:3]), np.asarray(context))
    np.testing.assert_allclose(
        np.asarray(restored.next_belief[:3]), np.asarray(next_context))


def test_replay_clear_keeps_storage_and_resets_logical_size():
    buffer = ReplayBuffer(3, 2, capacity=8, belief_dim=0)
    buffer.push_batch_jax(
        jnp.zeros((3, 3)), jnp.zeros((3, 2)), jnp.zeros((3, 1)),
        jnp.zeros((3, 3)), jnp.zeros((3, 1)))
    storage_shape = buffer.obs.shape
    buffer.clear()
    assert buffer.ptr == 0
    assert buffer.size == 0
    assert buffer.obs.shape == storage_shape


def test_bapr_v2_defaults_keep_mujoco_raw_regularizer_off():
    config = Config()
    assert config.bapr_v2_reg_weight == 0.0
    assert config.bapr_v2_use_fallback is True
    assert config.bapr_v2_policy_mode == "residual"
    assert config.bapr_v2_latent_scale_mode == "legacy_exp"
    assert config.bapr_v2_policy_context_source == "stored"
    assert config.bapr_v2_action_deviation_weight == 0.0


def test_bus_regularization_keeps_proven_positive_shift_sign():
    source = (Path(__file__).resolve().parents[2]
              / "sac_ensemble_bapr.py").read_text()
    assert (
        "target_q_next - self.alpha * next_log_prob + args.weight_reg * reg_norm"
        in source
    )
    assert (
        "q_values_dist = self.soft_q_net(state, new_action, ep_tensor) + args.weight_reg * reg_norm"
        in source
    )


def test_final_sweep_discovers_results_without_training_checkpoint():
    with TemporaryDirectory() as raw:
        tmp_path = Path(raw)
        stationary = tmp_path / "stationary"
        switching = tmp_path / "switching"
        complete = stationary / "v82a_robust_mean_Ant_s0" / "summary.csv"
        complete.parent.mkdir(parents=True)
        complete.write_text("metric_group,split\nstationary,test\n")
        empty = switching / "v82b_robust_mean_Ant_s0" / "summary.csv"
        empty.parent.mkdir(parents=True)
        empty.touch()

        assert completed_result_runs(stationary, switching) == [
            "v82a_robust_mean_Ant_s0"
        ]


if __name__ == "__main__":
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
