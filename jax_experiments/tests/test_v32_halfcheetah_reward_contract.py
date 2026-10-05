"""Check the reward contract used by the HalfCheetah Lean specialization."""
import numpy as np
from mujoco.mjx._src import collision_driver

from jax_experiments.envs.stochastic_mode_env import StochasticModeEnv


def test_spring_halfcheetah_return_telescopes_across_mode_switches():
    env = StochasticModeEnv(
        "HalfCheetah-v2", family="actuator_polarity", dwell_steps=8,
        seed=90281, backend="spring")
    tasks = env.sample_tasks(4)
    env.configure_eval_mode_sequence(tasks, [0, 1, 2, 3], period_steps=8)
    env.reset()
    sys = env.env.sys
    assert sys.link_types == "3111111"
    assert not sys.enable_fluid
    mass = np.asarray(sys.link.inertia.mass)
    inertia = np.diagonal(np.asarray(sys.link.inertia.i), axis1=-2, axis2=-1)
    assert np.all(np.isfinite(mass)) and np.all(mass > 0.0)
    assert np.all(np.isfinite(inertia)) and np.all(inertia > 0.0)
    pairs = list(collision_driver.geom_pairs(sys.mj_model))
    assert pairs
    assert all((int(sys.geom_type[a]), int(sys.geom_type[b])) == (0, 3)
               for a, b, _ in pairs)
    assert env.act_dim == 6
    assert np.isclose(float(env.dt), 0.05)
    assert env.env._forward_reward_weight == 1.0
    assert env.env._ctrl_cost_weight == 0.1
    initial_x = float(env._state.pipeline_state.x.pos[0, 0])
    total_return, energy = 0.0, 0.0
    modes = set()
    for step in range(32):
        command = np.sin(np.arange(6, dtype=np.float32) + step / 7.0)
        _, reward, done, info = env.step(command)
        assert not done
        executed = np.asarray(info["executed_action"], dtype=np.float64)
        assert np.all(np.abs(executed) <= 1.0)
        assert np.isfinite(reward)
        state = env._state.pipeline_state
        assert np.all(np.isfinite(np.asarray(state.x_i.pos)))
        assert np.all(np.isfinite(np.asarray(state.xd_i.vel)))
        np.testing.assert_allclose(
            np.linalg.norm(np.asarray(state.x_i.rot), axis=-1), 1.0,
            atol=2e-6, rtol=2e-6)
        total_return += reward
        energy += float(np.sum(executed ** 2))
        modes.add(info["mode_used"])
    final_x = float(env._state.pipeline_state.x.pos[0, 0])
    expected = 20.0 * (final_x - initial_x) - 0.1 * energy
    assert modes == {0, 1, 2, 3}
    assert 0.0 <= energy <= 6 * 32
    np.testing.assert_allclose(total_return, expected, atol=1e-3, rtol=1e-5)
