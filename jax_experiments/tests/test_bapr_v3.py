import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from types import SimpleNamespace

from jax_experiments.analysis.final_task_sweep import (
    oracle_eval_mode_ids,
    protocol_gravity_latent,
    require_checkpoint_iteration,
    task_policy_latent,
)
from jax_experiments.algos.bapr_v3 import BAPRv3, _mode_latent
from jax_experiments.algos.sac_base import SACBase
from jax_experiments.configs.default import Config
from jax_experiments.envs.brax_env import apply_action_disturbance
from jax_experiments.envs.stochastic_mode_env import (
    MODE_FAMILIES,
    StochasticModeEnv,
)
from jax_experiments.networks.probabilistic_regime_context import (
    ProbabilisticRegimeContext,
    detached_gaussian_variance_loss,
)
from jax_experiments.train import (
    _oracle_latent_for_eval,
    evaluate_stationary,
    evaluate_switching_online,
)


def test_oracle_context_ladder_validates_and_expands_modes():
    agent = SimpleNamespace(
        uses_transition_context=True, CONTEXT_ORACLE=1, latent_dim=4)
    args = SimpleNamespace(
        oracle_context_ladder=True, fixed_oracle_mode_id=None)
    assert oracle_eval_mode_ids(agent, args, 1) == (None, 0, 1, 2, 3)

    args.fixed_oracle_mode_id = 2
    try:
        oracle_eval_mode_ids(agent, args, 1)
    except ValueError as exc:
        assert "cannot be combined" in str(exc)
    else:
        raise AssertionError("ladder accepted a simultaneous fixed mode")

    args.fixed_oracle_mode_id = None
    try:
        oracle_eval_mode_ids(agent, args, 0)
    except ValueError as exc:
        assert "requires" in str(exc)
    else:
        raise AssertionError("ladder accepted a non-oracle context")


def test_action_disturbance_identity_and_reproducibility():
    action = jnp.asarray([0.2, -0.4], dtype=jnp.float32)
    key = jax.random.PRNGKey(4)
    identity = apply_action_disturbance(action, key, 1.0, 0.0)
    np.testing.assert_allclose(identity, action, atol=1e-7)
    first = apply_action_disturbance(action, key, 0.8, 0.1)
    second = apply_action_disturbance(action, key, 0.8, 0.1)
    other = apply_action_disturbance(
        action, jax.random.PRNGKey(5), 0.8, 0.1)
    np.testing.assert_allclose(first, second, atol=1e-7)
    assert not np.allclose(first, other)


def test_action_disturbance_packet_loss_and_burst_events():
    action = jnp.asarray([0.3, -0.6], dtype=jnp.float32)
    key = jax.random.PRNGKey(14)
    dropped = apply_action_disturbance(
        action, key, 1.0, 0.0, packet_loss_prob=1.0)
    np.testing.assert_allclose(dropped, 0.0, atol=1e-7)

    burst = apply_action_disturbance(
        jnp.zeros_like(action), key, 1.0, 0.0,
        burst_prob=1.0, burst_std=0.4)
    repeated = apply_action_disturbance(
        jnp.zeros_like(action), key, 1.0, 0.0,
        burst_prob=1.0, burst_std=0.4)
    np.testing.assert_allclose(burst, repeated, atol=1e-7)
    assert not np.allclose(burst, 0.0)


def test_stochastic_mode_profiles_separate_mean_and_variance_axes():
    variance = MODE_FAMILIES["variance_only"]
    assert {row["gravity_scale"] for row in variance} == {1.0}
    assert {row["action_gain"] for row in variance} == {1.0}
    assert len({row["action_noise_std"] for row in variance}) == 4

    deterministic = MODE_FAMILIES["deterministic_mean"]
    assert {row["action_noise_std"] for row in deterministic} == {0.0}
    assert len({(row["gravity_scale"], row["action_gain"])
                for row in deterministic}) == 4


def test_stochastic_event_profiles_keep_robot_and_nominal_control_fixed():
    packet = MODE_FAMILIES["packet_loss"]
    burst = MODE_FAMILIES["burst_torque"]
    for family in (packet, burst):
        assert {row["gravity_scale"] for row in family} == {1.0}
        assert {row["action_gain"] for row in family} == {1.0}
        assert {row["action_noise_std"] for row in family} == {0.0}
    assert [row["packet_loss_prob"] for row in packet] == [
        0.0, 0.08, 0.18, 0.32]
    assert [row["burst_prob"] for row in burst] == [
        0.0, 0.03, 0.08, 0.15]
    assert [row["burst_std"] for row in burst] == [
        0.0, 0.25, 0.45, 0.70]


def test_structured_channel_profiles_use_equal_severity_distinct_masks():
    profiles = MODE_FAMILIES["structured_channel"]
    assert {row["gravity_scale"] for row in profiles} == {1.0}
    assert {row["action_noise_std"] for row in profiles} == {0.04}
    assert {row["impaired_gain"] for row in profiles} == {0.45}
    assert [row["action_gain_pattern"] for row in profiles] == [
        "low_half", "high_half", "even", "odd"]

    env = StochasticModeEnv(
        "HalfCheetah-v2", family="structured_channel", dwell_steps=500,
        dwell_distribution="fixed", seed=19, backend="spring")
    tasks = env.sample_tasks(env.num_modes)
    gains = [np.asarray(task["action_gain"]) for task in tasks]
    expected = [
        [0.45, 0.45, 0.45, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.45, 0.45, 0.45],
        [0.45, 1.0, 0.45, 1.0, 0.45, 1.0],
        [1.0, 0.45, 1.0, 0.45, 1.0, 0.45],
    ]
    for actual, wanted in zip(gains, expected):
        np.testing.assert_allclose(actual, wanted, atol=1e-7)
        assert np.count_nonzero(np.isclose(actual, 0.45)) == env.act_dim // 2
    assert len({tuple(gain.tolist()) for gain in gains}) == 4


def test_structured_channel_gain_reaches_step_and_switch_configuration():
    env = StochasticModeEnv(
        "Ant-v2", family="structured_channel", dwell_steps=500,
        dwell_distribution="fixed", seed=23, backend="spring")
    tasks = env.sample_tasks(env.num_modes)
    env.set_task(tasks[2])
    env.reset()
    _, _, _, info = env.step(np.full(env.act_dim, 0.2, dtype=np.float32))
    assert info["mode_used"] == 2
    np.testing.assert_allclose(
        info["action_gain"],
        [0.45, 1.0, 0.45, 1.0, 0.45, 1.0, 0.45, 1.0],
        atol=1e-7,
    )
    env.configure_eval_switching(tasks, period_steps=500)
    assert env.current_task_id in range(4)
    assert len(env._eval_mode_sequence) == 2
    assert env._eval_mode_sequence[0] != env._eval_mode_sequence[1]


def test_actuator_polarity_profiles_cover_odd_action_space():
    env = StochasticModeEnv(
        "Hopper-v2", family="actuator_polarity", dwell_steps=250,
        dwell_distribution="fixed", seed=7, backend="spring")
    tasks = env.sample_tasks(4)
    gains = [
        np.asarray(task["action_gain"], dtype=np.float32)
        for task in tasks
    ]
    assert len({tuple(gain.tolist()) for gain in gains}) == 4
    for task, gain in zip(tasks, gains):
        assert task["gravity_scale"] == 1.0
        assert task["action_noise_std"] == 0.02
        assert set(np.unique(gain)) == {-1.0, 1.0}
        assert abs(
            np.count_nonzero(gain < 0.0)
            - np.count_nonzero(gain > 0.0)) <= 1


def test_explicit_eval_mode_sequence_covers_all_modes_in_order():
    env = StochasticModeEnv(
        "HalfCheetah-v2", family="structured_channel", dwell_steps=500,
        dwell_distribution="fixed", seed=29, backend="spring")
    tasks = env.sample_tasks(env.num_modes)
    sequence = [2, 0, 3, 1]
    env.configure_eval_mode_sequence(tasks, sequence, period_steps=7)
    observed = [env.current_task_id]
    for boundary in (7, 14, 21):
        env._step_counter = boundary
        env._check_switch()
        observed.append(env.current_task_id)
    assert observed == sequence
    assert env.get_switch_history() == [
        (0, 2), (7, 0), (14, 3), (21, 1)]


def test_stochastic_event_profile_reaches_sequential_step():
    env = StochasticModeEnv(
        "HalfCheetah-v2", family="packet_loss", dwell_steps=500,
        dwell_distribution="fixed", seed=17, backend="spring")
    task = env.sample_tasks(env.num_modes)[-1]
    env.set_task(task)
    env.reset()
    _, _, _, info = env.step(np.full(env.act_dim, 0.2, dtype=np.float32))
    assert info["mode_used"] == 3
    np.testing.assert_allclose(info["packet_loss_prob"], 0.32, atol=1e-7)
    np.testing.assert_allclose(info["burst_prob"], 0.0, atol=1e-7)
    np.testing.assert_allclose(info["burst_std"], 0.0, atol=1e-7)


def test_stochastic_mode_env_holds_mode_until_dwell_boundary():
    env = StochasticModeEnv(
        "HalfCheetah-v2", family="mean_variance", dwell_steps=500,
        dwell_distribution="fixed", seed=3, backend="spring")
    assert env.current_task_id == 0
    first_sys = env._current_sys
    env._step_counter = 499
    env._check_switch()
    assert env.current_task_id == 0
    assert env._current_sys is first_sys
    env._step_counter = 500
    env._check_switch()
    assert env.current_task_id != 0
    assert env._current_sys is not first_sys


def test_stochastic_mode_fixed_training_never_switches_until_eval_override():
    env = StochasticModeEnv(
        "HalfCheetah-v2", family="mean_variance", dwell_steps=8,
        dwell_distribution="fixed", seed=4, backend="spring",
        fixed_mode_id=2)
    tasks = env.sample_tasks(4)
    env.set_nonstationary_para(tasks)
    assert env.current_task_id == 2
    fixed_sys = env._current_sys
    env._step_counter = 10_000
    env._check_switch()
    assert env.current_task_id == 2
    assert env._current_sys is fixed_sys

    env.configure_eval_switching(tasks, period_steps=8)
    first_mode = env.current_task_id
    env._step_counter = 8
    env._check_switch()
    assert env.current_task_id != first_mode


def test_stochastic_mode_fixed_id_is_validated():
    with np.testing.assert_raises_regex(ValueError, "fixed_mode_id"):
        StochasticModeEnv(
            "HalfCheetah-v2", family="mean_variance", dwell_steps=8,
            dwell_distribution="fixed", seed=4, backend="spring",
            fixed_mode_id=4)


def test_mode_latent_is_full_one_hot_posterior_coordinate():
    np.testing.assert_array_equal(
        _mode_latent({"mode_id": 2}, 4),
        np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32))


def test_final_sweep_uses_mode_coordinates_for_stochastic_protocols():
    task = {"mode_id": 3, "gravity_scale": 1.0}
    config = Config()
    agent = SimpleNamespace(latent_dim=4)
    assert protocol_gravity_latent(task, config) == 3.0
    np.testing.assert_array_equal(
        task_policy_latent(agent, task, config),
        np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32))


def test_final_sweep_baseline_without_latent_uses_physical_mode_coordinate():
    task = {"mode_id": 3, "gravity_scale": 1.0}
    np.testing.assert_array_equal(
        task_policy_latent(SimpleNamespace(), task, Config()),
        np.asarray([3.0], dtype=np.float32))


def test_final_sweep_baseline_without_latent_uses_continuous_coordinate():
    task = {
        "gravity": np.asarray([0.0, 0.0, -12.2625], dtype=np.float32),
    }
    config = Config()
    np.testing.assert_array_equal(
        task_policy_latent(SimpleNamespace(), task, config),
        np.asarray(
            [protocol_gravity_latent(task, config)], dtype=np.float32))


def test_final_sweep_rejects_stale_checkpoint_generation():
    require_checkpoint_iteration(2600, 2600)
    require_checkpoint_iteration(2600, None)
    try:
        require_checkpoint_iteration(2000, 2600)
    except RuntimeError as exc:
        assert "loaded next_iter=2000" in str(exc)
    else:
        raise AssertionError("stale checkpoint was accepted")


def test_probabilistic_context_outputs_finite_mode_statistics():
    model = ProbabilisticRegimeContext(
        obs_dim=5, act_dim=2, num_modes=4, hidden_dim=16,
        ensemble_size=3, likelihood="probabilistic",
        mode="supervised", rngs=nnx.Rngs(7))
    obs = jnp.linspace(-0.2, 0.2, 5)
    action = jnp.asarray([0.1, -0.2])
    next_obs = obs + 0.01
    target = model._target(obs, 0.5, next_obs)
    stats = model.likelihood_statistics(obs, action, target)
    mode_ll, mode_nll, aleatoric, epistemic, means, logvars = stats
    assert mode_ll.shape == (4,)
    assert mode_nll.shape == (4,)
    assert aleatoric.shape == (4,)
    assert epistemic.shape == (4,)
    assert means.shape == (3, 4, 6)
    assert logvars.shape == (3, 4, 6)
    for value in stats:
        assert np.all(np.isfinite(np.asarray(value)))


def test_mode_calibrated_variance_starts_bounded_at_fixed_prior():
    model = ProbabilisticRegimeContext(
        obs_dim=3, act_dim=1, num_modes=4, hidden_dim=8,
        ensemble_size=2, likelihood="probabilistic",
        variance_model="mode_calibrated", fixed_variance=0.02,
        variance_floor=1e-4, variance_ceiling=0.05,
        mode="supervised", rngs=nnx.Rngs(70))
    variances = np.asarray(model.mode_variances())
    assert variances.shape == (4, 4)
    np.testing.assert_allclose(variances, 0.02, rtol=1e-5, atol=1e-7)
    assert float(variances.min()) >= 1e-4
    assert float(variances.max()) <= 0.05


def test_shared_empirical_context_uses_one_mean_model_for_all_modes():
    model = ProbabilisticRegimeContext(
        obs_dim=5, act_dim=2, num_modes=4, hidden_dim=16,
        ensemble_size=3, likelihood="probabilistic",
        variance_model="mode_shared_empirical", fixed_variance=0.02,
        variance_floor=1e-4, variance_ceiling=0.5,
        mode="supervised", rngs=nnx.Rngs(72))
    means, logvars = model.distribution(
        jnp.linspace(-0.2, 0.2, 5), jnp.asarray([0.1, -0.3]))
    assert means.shape == (3, 4, 6)
    assert logvars.shape == means.shape
    for mode in range(1, 4):
        np.testing.assert_allclose(
            means[:, 0, :], means[:, mode, :], atol=1e-7)
    assert model.mode_variances().shape == (4, 6)


def test_inverse_empirical_predicts_action_from_transition_only():
    model = ProbabilisticRegimeContext(
        obs_dim=5, act_dim=2, num_modes=4, hidden_dim=16,
        ensemble_size=3, likelihood="probabilistic",
        variance_model="inverse_empirical", fixed_variance=0.02,
        variance_floor=1e-4, variance_ceiling=0.5,
        mode="supervised", rngs=nnx.Rngs(74))
    obs = jnp.linspace(-0.2, 0.2, 5)
    next_obs = obs + 0.03
    means, logvars = model.inverse_distribution(obs, next_obs)
    assert means.shape == (3, 4, 2)
    assert logvars.shape == means.shape
    for mode in range(1, 4):
        np.testing.assert_allclose(
            means[:, 0, :], means[:, mode, :], atol=1e-7)
    assert model.mode_variances().shape == (4, 2)


def test_inverse_empirical_tracks_action_residual_variance():
    config = Config()
    config.task_num = 2
    config.test_task_num = 2
    config.bapr_v2_latent_dim = 2
    config.hidden_dim = 8
    config.ensemble_size = 2
    config.bapr_v2_context_hidden_dim = 8
    config.bapr_v3_context_ensemble_size = 2
    config.bapr_v3_variance_model = "inverse_empirical"
    config.bapr_v3_variance_floor = 1e-4
    config.bapr_v3_variance_ceiling = 0.5
    config.bapr_v3_variance_ema = 0.5
    config.bapr_v3_instant_classifier_weight = 0.0
    config.bapr_v2_context_burnin = 0
    config.bapr_v2_temporal_weight = 0.0
    agent = BAPRv3(1, 1, config, seed=75)
    agent.set_task_metadata([{"mode_id": 0}, {"mode_id": 1}])
    for layer in (
            agent.context_net.inverse_hidden1,
            agent.context_net.inverse_hidden2,
            agent.context_net.inverse_action):
        layer.kernel.value = jnp.zeros_like(layer.kernel.value)
        layer.bias.value = jnp.zeros_like(layer.bias.value)

    length = 16
    obs = jnp.zeros((2, length, 1), dtype=jnp.float32)
    next_obs = jnp.zeros_like(obs)
    act = jnp.stack([
        jnp.zeros((length, 1), dtype=jnp.float32),
        jnp.full((length, 1), 0.4, dtype=jnp.float32),
    ])
    reward = jnp.zeros((2, length, 1), dtype=jnp.float32)
    done = jnp.zeros_like(reward)
    target = jnp.stack([
        jnp.tile(jnp.asarray([1.0, 0.0]), (length, 1)),
        jnp.tile(jnp.asarray([0.0, 1.0]), (length, 1)),
    ])
    opt_state = agent.context_opt_state
    for _ in range(12):
        params, opt_state, metrics = agent._context_update(
            nnx.state(agent.context_net, nnx.Param), opt_state,
            obs, act, reward, next_obs, done, target)
        nnx.update(agent.context_net, params)
        agent._record_context_update_diagnostics(*metrics[5:])
    variance = np.asarray(agent.context_net.mode_variances()).mean(axis=1)
    assert variance[0] < 0.01
    assert variance[1] > variance[0] * 10.0


def test_inverse_empirical_filter_accumulates_action_variance_evidence():
    model = ProbabilisticRegimeContext(
        obs_dim=1, act_dim=1, num_modes=2, hidden_dim=8,
        ensemble_size=2, likelihood="probabilistic",
        variance_model="inverse_empirical", fixed_variance=0.02,
        variance_floor=1e-4, variance_ceiling=0.5,
        evidence_scale=1.0, hazard_rate=0.01,
        mode="supervised", rngs=nnx.Rngs(76))
    for layer in (
            model.inverse_hidden1, model.inverse_hidden2,
            model.inverse_action):
        layer.kernel.value = jnp.zeros_like(layer.kernel.value)
        layer.bias.value = jnp.zeros_like(layer.bias.value)
    model.mode_logvar_raw.value = model.calibrated_raw_from_variance(
        jnp.asarray([[0.001], [0.10]], dtype=jnp.float32))
    obs = jnp.zeros((1,), dtype=jnp.float32)

    high_state = model.initial_state()
    for _ in range(6):
        high_state, _, _, _ = model.observe(
            high_state, obs, jnp.asarray([0.3]), 0.0, obs, 0.0)
    assert float(high_state[0][1]) > 0.95

    low_state = model.initial_state()
    for _ in range(6):
        low_state, _, _, _ = model.observe(
            low_state, obs, jnp.zeros((1,)), 0.0, obs, 0.0)
    assert float(low_state[0][0]) > 0.95


def test_robust_estimator_rollout_source_breaks_policy_feedback_loop():
    config = Config()
    config.task_num = 2
    config.test_task_num = 2
    config.bapr_v2_latent_dim = 2
    config.hidden_dim = 8
    config.ensemble_size = 2
    config.bapr_v2_context_hidden_dim = 8
    config.bapr_v3_context_ensemble_size = 2
    config.bapr_v2_training_schedule = "teacher_student"
    config.bapr_v2_base_pretrain_iters = 1
    config.bapr_v2_teacher_iters = 1
    config.bapr_v3_estimator_rollout_source = "robust"
    agent = BAPRv3(2, 1, config, seed=77)
    assert agent.rollout_context_source(1) == agent.CONTEXT_ORACLE
    assert agent.rollout_context_source(2) == agent.CONTEXT_ROBUST


def test_mode_empirical_variance_tracks_mode_residual_moments():
    config = Config()
    config.task_num = 2
    config.test_task_num = 2
    config.bapr_v2_latent_dim = 2
    config.hidden_dim = 8
    config.ensemble_size = 2
    config.bapr_v2_context_hidden_dim = 8
    config.bapr_v3_context_ensemble_size = 2
    config.bapr_v3_variance_model = "mode_empirical"
    config.bapr_v3_variance_floor = 1e-4
    config.bapr_v3_variance_ceiling = 0.2
    config.bapr_v3_variance_ema = 0.5
    config.bapr_v3_instant_classifier_weight = 0.0
    config.bapr_v2_context_burnin = 0
    config.bapr_v2_temporal_weight = 0.0
    agent = BAPRv3(1, 1, config, seed=73)
    agent.set_task_metadata([{"mode_id": 0}, {"mode_id": 1}])

    length = 16
    magnitude = float(np.arctanh(0.3))
    signs = jnp.asarray([(-1.0) ** index for index in range(length)])
    obs = jnp.zeros((2, length, 1), dtype=jnp.float32)
    act = jnp.zeros_like(obs)
    next_obs = jnp.stack([
        jnp.zeros((length, 1), dtype=jnp.float32),
        (magnitude * signs).reshape(length, 1),
    ])
    reward = jnp.stack([
        jnp.zeros((length, 1), dtype=jnp.float32),
        (magnitude * config.bapr_v2_reward_scale * signs).reshape(
            length, 1),
    ])
    done = jnp.zeros((2, length, 1), dtype=jnp.float32)
    target = jnp.stack([
        jnp.tile(jnp.asarray([1.0, 0.0]), (length, 1)),
        jnp.tile(jnp.asarray([0.0, 1.0]), (length, 1)),
    ])
    opt_state = agent.context_opt_state
    for _ in range(12):
        params, opt_state, metrics = agent._context_update(
            nnx.state(agent.context_net, nnx.Param), opt_state,
            obs, act, reward, next_obs, done, target)
        nnx.update(agent.context_net, params)
        agent._record_context_update_diagnostics(*metrics[5:])

    per_mode = np.asarray(agent.context_net.mode_variances()).mean(axis=1)
    assert per_mode[0] < 0.02
    assert per_mode[1] > per_mode[0] * 4.0
    assert per_mode[1] <= config.bapr_v3_variance_ceiling
    assert agent._v3_empirical_updates == 12
    assert agent._v3_calibration_error >= 0.0


def test_detached_variance_loss_corrects_both_under_and_over_dispersion():
    residual_sq = jnp.asarray(0.01, dtype=jnp.float32)
    grad = jax.grad(
        lambda logvar: detached_gaussian_variance_loss(
            residual_sq, logvar))
    assert float(grad(jnp.log(1.0))) > 0.0
    assert float(grad(jnp.log(0.001))) < 0.0


def test_calibrated_sticky_posterior_accumulates_variance_evidence():
    model = ProbabilisticRegimeContext(
        obs_dim=1, act_dim=1, num_modes=2, hidden_dim=8,
        ensemble_size=2, likelihood="probabilistic",
        variance_model="mode_calibrated", fixed_variance=0.02,
        variance_floor=1e-4, variance_ceiling=0.2,
        evidence_scale=1.0, hazard_rate=0.01,
        mode="supervised", rngs=nnx.Rngs(71))
    model.decoder_mean.kernel.value = jnp.zeros_like(
        model.decoder_mean.kernel.value)
    model.decoder_mean.bias.value = jnp.zeros_like(
        model.decoder_mean.bias.value)
    desired = jnp.asarray([
        [0.001, 0.001],
        [0.10, 0.10],
    ], dtype=jnp.float32)
    model.mode_logvar_raw.value = model.calibrated_raw_from_variance(
        desired)

    obs = jnp.zeros((1,), dtype=jnp.float32)
    action = jnp.zeros((1,), dtype=jnp.float32)
    large = float(np.arctanh(0.3))
    high_state = model.initial_state()
    for _ in range(6):
        high_state, _, _, _ = model.observe(
            high_state, obs, action, large * model.reward_scale,
            jnp.asarray([large]), 0.0)
    assert float(high_state[0][1]) > 0.95

    low_state = model.initial_state()
    for _ in range(6):
        low_state, _, _, _ = model.observe(
            low_state, obs, action, 0.0, obs, 0.0)
    assert float(low_state[0][0]) > 0.95


def test_context_update_learns_mode_residual_variance_without_inflation():
    config = Config()
    config.task_num = 2
    config.test_task_num = 2
    config.bapr_v2_latent_dim = 2
    config.hidden_dim = 8
    config.ensemble_size = 2
    config.bapr_v2_context_hidden_dim = 8
    config.bapr_v3_context_ensemble_size = 2
    config.bapr_v3_variance_model = "mode_calibrated"
    config.bapr_v3_variance_floor = 1e-4
    config.bapr_v3_variance_ceiling = 0.2
    config.bapr_v3_variance_loss_weight = 1.0
    config.bapr_v3_variance_prior_weight = 0.0
    config.bapr_v2_context_lr = 0.01
    config.bapr_v2_context_burnin = 0
    config.bapr_v2_temporal_weight = 0.0
    agent = BAPRv3(1, 1, config, seed=72)
    agent.set_task_metadata([{"mode_id": 0}, {"mode_id": 1}])

    length = 16
    magnitude = float(np.arctanh(0.3))
    signs = jnp.asarray([(-1.0) ** index for index in range(length)])
    obs = jnp.zeros((2, length, 1), dtype=jnp.float32)
    act = jnp.zeros_like(obs)
    next_obs = jnp.stack([
        jnp.zeros((length, 1), dtype=jnp.float32),
        (magnitude * signs).reshape(length, 1),
    ])
    reward = jnp.stack([
        jnp.zeros((length, 1), dtype=jnp.float32),
        (magnitude * config.bapr_v2_reward_scale * signs).reshape(
            length, 1),
    ])
    done = jnp.zeros((2, length, 1), dtype=jnp.float32)
    target = jnp.stack([
        jnp.tile(jnp.asarray([1.0, 0.0]), (length, 1)),
        jnp.tile(jnp.asarray([0.0, 1.0]), (length, 1)),
    ])
    params = nnx.state(agent.context_net, nnx.Param)
    opt_state = agent.context_opt_state
    for _ in range(60):
        params, opt_state, _ = agent._context_update(
            params, opt_state, obs, act, reward, next_obs, done, target)
    nnx.update(agent.context_net, params)
    per_mode = np.asarray(agent.context_net.mode_variances()).mean(axis=1)
    assert per_mode[0] < 0.02
    assert per_mode[1] > per_mode[0] * 4.0
    assert per_mode[1] < config.bapr_v3_variance_ceiling


def test_sticky_filter_normalizes_and_starts_with_closed_gate():
    model = ProbabilisticRegimeContext(
        obs_dim=3, act_dim=1, num_modes=4, hidden_dim=8,
        ensemble_size=2, likelihood="point", mode="supervised",
        min_history=8, rngs=nnx.Rngs(9))
    state = model.initial_state()
    initial_context = model.policy_context(state, jnp.zeros((4,)))
    np.testing.assert_allclose(initial_context[-1], 0.0, atol=1e-7)
    next_state, surprise, prediction, target = model.observe(
        state, jnp.zeros((3,)), jnp.zeros((1,)), 0.0,
        jnp.ones((3,)) * 0.01, 0.0)
    np.testing.assert_allclose(
        np.sum(np.asarray(next_state[0])), 1.0, atol=1e-6)
    assert np.all(np.asarray(next_state[0]) > 0.0)
    assert np.isfinite(float(surprise))
    assert prediction.shape == target.shape == (4,)


def test_learned_context_ignores_privileged_oracle_latent():
    model = ProbabilisticRegimeContext(
        obs_dim=3, act_dim=1, num_modes=4, hidden_dim=8,
        ensemble_size=2, likelihood="probabilistic", mode="supervised",
        min_history=8, rngs=nnx.Rngs(10))
    state = model.initial_state()
    mode_zero = jnp.asarray([1.0, 0.0, 0.0, 0.0])
    mode_three = jnp.asarray([0.0, 0.0, 0.0, 1.0])

    learned_zero = model.policy_context(state, mode_zero)
    learned_three = model.policy_context(state, mode_three)
    np.testing.assert_allclose(learned_zero, learned_three, atol=1e-7)

    model.mode = "oracle"
    oracle_zero = model.policy_context(state, mode_zero)
    oracle_three = model.policy_context(state, mode_three)
    assert not np.allclose(oracle_zero, oracle_three)


def test_non_oracle_eval_receives_zero_privileged_latent():
    class Agent:
        CONTEXT_ROBUST = 0
        CONTEXT_ORACLE = 1
        CONTEXT_LEARNED = 2

        def __init__(self):
            self.oracle_latent = jnp.asarray([0.0, 0.0, 0.0, 1.0])
            self.set_calls = 0

        def set_eval_task(self, task):
            self.set_calls += 1
            self.oracle_latent = jnp.asarray(task["latent"])

    agent = Agent()
    task = {"latent": [1.0, 0.0, 0.0, 0.0]}
    learned = _oracle_latent_for_eval(
        agent, agent.CONTEXT_LEARNED, task)
    np.testing.assert_allclose(learned, 0.0, atol=1e-7)
    assert agent.set_calls == 0

    oracle = _oracle_latent_for_eval(agent, agent.CONTEXT_ORACLE, task)
    np.testing.assert_allclose(oracle, task["latent"], atol=1e-7)
    assert agent.set_calls == 1


def test_bapr_v3_builds_independent_context_path():
    config = Config()
    config.algo = "bapr_v3"
    config.task_num = 4
    config.test_task_num = 4
    config.bapr_v2_latent_dim = 4
    config.bapr_v2_context_hidden_dim = 16
    config.hidden_dim = 16
    config.ensemble_size = 2
    config.bapr_v3_context_ensemble_size = 2
    config.bapr_v3_variance_model = "inverse_empirical"
    config.bapr_v3_variance_ceiling = 0.5
    agent = BAPRv3(5, 2, config, seed=11)
    assert isinstance(agent.context_net, ProbabilisticRegimeContext)
    assert agent.context_dim == 5
    agent.set_task_metadata([
        {"mode_id": index} for index in range(4)
    ])
    np.testing.assert_allclose(
        np.asarray(agent.task_latents), np.eye(4), atol=1e-7)


def test_deployment_advantage_gate_requires_explicit_opt_in():
    config = Config()
    config.algo = "bapr_v3"
    config.task_num = 4
    config.test_task_num = 4
    config.bapr_v2_latent_dim = 4
    config.bapr_v2_context_hidden_dim = 16
    config.hidden_dim = 16
    config.ensemble_size = 2
    config.bapr_v3_context_ensemble_size = 2
    config.bapr_v2_training_schedule = "constrained_deploy"
    config.bapr_v2_base_pretrain_iters = 1
    config.bapr_v2_teacher_iters = 1
    config.bapr_v2_student_iters = 1
    config.bapr_v2_advantage_gate = True
    agent = BAPRv3(5, 2, config, seed=13)
    assert not agent.advantage_gate_active(1)
    assert agent.advantage_gate_active(2)
    assert agent.advantage_gate_active(3)

    config.bapr_v2_advantage_gate = False
    disabled = BAPRv3(5, 2, config, seed=14)
    assert not disabled.advantage_gate_active(3)


def test_freeze_teacher_preserves_controller_in_deployment():
    config = Config()
    config.task_num = 2
    config.test_task_num = 2
    config.bapr_v2_latent_dim = 2
    config.bapr_v2_context_hidden_dim = 8
    config.hidden_dim = 8
    config.ensemble_size = 2
    config.bapr_v3_context_ensemble_size = 2
    config.bapr_v2_training_schedule = "constrained_deploy"
    config.bapr_v2_base_pretrain_iters = 1
    config.bapr_v2_teacher_iters = 1
    config.bapr_v2_student_iters = 1
    config.bapr_v3_freeze_teacher_after_teacher = True
    frozen = BAPRv3(3, 1, config, seed=15)
    assert frozen.controller_update_flags(1) == (False, True, True)
    assert frozen.controller_update_flags(2) == (False, False, False)
    assert frozen.controller_update_flags(3) == (False, False, False)
    assert not frozen.train_policy_gate(3)

    config.bapr_v3_freeze_teacher_after_teacher = False
    unfrozen = BAPRv3(3, 1, config, seed=16)
    assert unfrozen.controller_update_flags(3) == (False, True, True)
    assert unfrozen.train_policy_gate(3)


def test_bapr_v3_compiled_rollout_and_context_update_are_finite():
    config = Config()
    config.algo = "bapr_v3"
    config.env_name = "HalfCheetah-v2"
    config.task_num = 4
    config.test_task_num = 4
    config.samples_per_iter = 16
    config.max_episode_steps = 8
    config.bapr_v2_latent_dim = 4
    config.bapr_v2_context_hidden_dim = 16
    config.bapr_v2_context_length = 8
    config.bapr_v2_context_chunks = 2
    config.bapr_v2_context_burnin = 2
    config.hidden_dim = 16
    config.ensemble_size = 2
    config.bapr_v3_context_ensemble_size = 2
    config.bapr_v3_variance_model = "mode_calibrated"
    config.bapr_v3_variance_ceiling = 0.05
    env = StochasticModeEnv(
        config.env_name, family="mean_variance", dwell_steps=8,
        dwell_distribution="fixed", seed=13, backend="spring")
    tasks = env.sample_tasks(4)
    env.set_nonstationary_para(tasks)
    agent = BAPRv3(env.obs_dim, env.act_dim, config, seed=13)
    agent.set_task_metadata(tasks)
    env.build_rollout_fn(
        nnx.graphdef(agent.policy),
        transition_context_graphdef=nnx.graphdef(agent.context_net),
        critic_graphdef=nnx.graphdef(agent.critic))
    agent.set_oracle_task_id(0)
    transitions, _, final_state = env.rollout_adaptive(
        nnx.state(agent.policy, nnx.Param),
        nnx.state(agent.context_net, nnx.Param),
        agent.adaptation_state, agent.oracle_latent, 8,
        jax.random.PRNGKey(14),
        critic_params=nnx.state(agent.critic, nnx.Param),
        context_source=agent.CONTEXT_LEARNED)
    assert transitions[0].shape == (8, env.obs_dim)
    assert transitions[1].shape == (8, env.act_dim)
    assert transitions[5].shape == (8, 5)
    for value in transitions:
        assert np.all(np.isfinite(np.asarray(value)))
    agent.adaptation_state = final_state

    task_ids = jnp.zeros((8,), dtype=jnp.int32)
    recent = {
        "obs": transitions[0],
        "act": transitions[1],
        "rew": transitions[2].reshape(-1, 1),
        "next_obs": transitions[3],
        "done": transitions[4].reshape(-1, 1),
        "task_id": task_ids,
    }
    chunks = agent._context_training_batch(recent)
    *transition_chunks, chunk_task_ids, _ = chunks
    targets = agent.task_latents[chunk_task_ids]
    params, _, metrics = agent._context_update(
        nnx.state(agent.context_net, nnx.Param),
        agent.context_opt_state, *transition_chunks, targets)
    assert len(metrics) == 12
    for value in metrics:
        assert np.all(np.isfinite(np.asarray(value)))
    for value in jax.tree.leaves(params):
        assert np.all(np.isfinite(np.asarray(value)))

    for source in (
            agent.CONTEXT_ROBUST,
            agent.CONTEXT_ORACLE,
            agent.CONTEXT_LEARNED):
        stationary = evaluate_stationary(
            agent, env, config, tasks, n_episodes=1,
            context_source=source, advantage_enabled=False,
            record_diagnostics=False)
        switching = evaluate_switching_online(
            agent, env, config, tasks, n_episodes=2, period_steps=4,
            context_source=source, advantage_enabled=False)
        assert np.all(np.isfinite(np.asarray(stationary)))
        assert np.all(np.isfinite(np.asarray(switching)))
        assert switching[2] >= 1.0


def test_standard_sac_uses_same_stochastic_mode_chunk_clock():
    config = Config()
    config.hidden_dim = 16
    config.ensemble_size = 2
    env = StochasticModeEnv(
        "HalfCheetah-v2", family="variance_only", dwell_steps=4,
        dwell_distribution="fixed", seed=17, backend="spring")
    tasks = env.sample_tasks(4)
    env.set_nonstationary_para(tasks)
    agent = SACBase(env.obs_dim, env.act_dim, config, seed=17)
    env.build_rollout_fn(nnx.graphdef(agent.policy))
    policy_params = nnx.state(agent.policy, nnx.Param)
    first, _ = env.rollout(
        policy_params, 4, jax.random.PRNGKey(18))
    first_mode = env.get_switch_history()[0][1]
    assert first_mode == 0
    assert env.current_task_id != 0
    second, _ = env.rollout(
        policy_params, 4, jax.random.PRNGKey(19), continue_state=True)
    assert first[0].shape == second[0].shape == (4, env.obs_dim)
    assert np.all(np.isfinite(np.asarray(first[2])))
    assert np.all(np.isfinite(np.asarray(second[2])))


if __name__ == "__main__":
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
