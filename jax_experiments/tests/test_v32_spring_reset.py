"""Source correspondence for the real-arithmetic HalfCheetah reset certificate."""

import json

import jax
import numpy as np
from jax import numpy as jnp
from brax import actuator
from brax.spring import joints, pipeline

from jax_experiments.envs.stochastic_mode_env import StochasticModeEnv


PARENTS = (-1, 0, 1, 2, 0, 4, 5)
LINK_OFFSETS = np.array([
    [0., 0., 0.7], [-0.5, 0., 0.], [0.16, 0., -0.25], [-0.28, 0., -0.14],
    [0.5, 0., 0.], [-0.14, 0., -0.24], [0.13, 0., -0.18],
])


def _rotate(v, q):
    s, u = q[0], q[1:]
    return 2*np.dot(u, v)*u+(s*s-np.dot(u, u))*v+2*s*np.cross(u, v)


def _reset_formula(q, qd, offset):
    angle, speed = np.empty(7), np.empty(7)
    angle[0], speed[0] = q[2], qd[2]
    for i in range(1, 7):
        angle[i] = angle[PARENTS[i]]+q[i+2]
        speed[i] = speed[PARENTS[i]]+qd[i+2]
    rot = np.column_stack((np.cos(angle/2), np.zeros(7), np.sin(angle/2), np.zeros(7)))
    angular = np.column_stack((np.zeros(7), speed, np.zeros(7)))
    pos, vel, relative = np.zeros((3, 7, 3))
    pos[0], vel[0] = [q[0], 0., 0.7+q[1]], [qd[0], 0., qd[1]]
    for i in range(1, 7):
        parent = PARENTS[i]
        lever = _rotate(LINK_OFFSETS[i], rot[parent])
        pos[i] = pos[parent]+lever
        vel[i] = vel[parent]+np.cross(angular[parent], lever)
        # world_to_joint uses parent.pos-parentAnchor.pos, not its opposite.
        relative[i] = _rotate(2*np.cross(angular[parent], lever),
                              rot[parent]*[1., -1., -1., -1.])
    relative[0] = vel[0]
    com_pos, com_vel = pos.copy(), vel.copy()
    for i in range(7):
        lever = _rotate(offset[i], rot[i])
        com_pos[i] += lever
        com_vel[i] += np.cross(angular[i], lever)
    return pos, rot, vel, angular, com_pos, com_vel, relative


def test_reset_and_first_physics_substep():
    env = StochasticModeEnv("HalfCheetah-v2", family="actuator_polarity", backend="spring")
    sys = env.base_sys
    assert tuple(sys.link_parents) == PARENTS and sys.link_types == "3111111"
    assert not sys.enable_fluid and env.env._n_frames == 16
    assert env.env._reset_noise_scale == 0.1
    np.testing.assert_array_equal(sys.init_q, np.zeros(9))
    np.testing.assert_allclose(sys.link.transform.pos, LINK_OFFSETS, atol=2e-6)
    np.testing.assert_array_equal(sys.link.transform.rot, np.tile([1., 0., 0., 0.], (7, 1)))
    np.testing.assert_array_equal(sys.link.joint.pos, np.zeros((7, 3)))
    np.testing.assert_array_equal(sys.link.joint.rot, np.tile([1., 0., 0., 0.], (7, 1)))
    expected_ang = np.zeros((9, 3))
    expected_ang[2:, 1] = 1.
    expected_vel = np.zeros((9, 3))
    expected_vel[0, 0], expected_vel[1, 2] = 1., 1.
    np.testing.assert_array_equal(sys.dof.motion.ang, expected_ang)
    np.testing.assert_array_equal(sys.dof.motion.vel, expected_vel)
    offset = np.asarray(sys.link.inertia.transform.pos, dtype=np.float64)
    assert np.max(np.linalg.norm(offset, axis=1)) <= 0.2
    assert np.max(np.abs(sys.gravity)) <= 10.
    assert np.max(np.abs(sys.link.constraint_vel_damping)) <= 10.
    dt = float(sys.opt.timestep)
    np.testing.assert_allclose(dt, 1/320, atol=1e-10)
    assert float(sys.vel_damping) == 0.

    @jax.jit
    def force_and_acceleration(state, action):
        tau = actuator.to_tau(sys, action, state.q, state.qd)
        force = joints.resolve(sys, state, tau)
        return force.vel, sys.gravity+force.vel/state.mass[:, None]

    step = jax.jit(lambda state, action: pipeline.step(sys, state, action))
    states = [env._reset_fn(sys, jax.random.PRNGKey(seed)).pipeline_state
              for seed in range(90289, 90301)]
    for sign_q, sign_qd in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        states.append(pipeline.init(sys, jnp.full(9, sign_q*0.1), jnp.full(9, sign_qd*0.1)))
    minima = {key: np.inf for key in ("reset_com_height", "precontact_height", "first_step_height")}
    maxima = {key: 0. for key in ("formula_error", "joint_position_error", "com_velocity",
                                "cache_velocity", "local_force", "acceleration")}
    for state in states:
        q, qd = np.asarray(state.q), np.asarray(state.qd)
        assert np.max(np.abs(q)) <= 0.1+1e-7 and np.max(np.abs(qd)) <= 0.1+1e-7
        expected = _reset_formula(q, qd, offset)
        actual = (state.x.pos, state.x.rot, state.xd.vel, state.xd.ang,
                  state.x_i.pos, state.xd_i.vel, state.jd.vel)
        for got, want in zip(actual, expected):
            error = float(np.max(np.abs(np.asarray(got)-want)))
            maxima["formula_error"] = max(maxima["formula_error"], error)
            np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-6)
        initial_height = np.asarray(state.x_i.pos[:, 2], dtype=np.float64)
        assert np.min(initial_height) >= -1/3
        assert np.max(np.abs(state.x_i.pos)) <= 3.
        np.testing.assert_allclose(np.sum(np.asarray(state.x_i.rot)**2, axis=1), 1., atol=2e-6)
        assert np.max(np.abs(state.xd_i.vel)) <= 1.
        assert np.max(np.abs(state.jd.vel[1:])) <= 8.4
        assert np.max(np.abs(state.j.pos[1:])) <= 2e-6
        assert abs(float(state.j.pos[0, 1])) <= 2e-6 and abs(float(state.jd.vel[0, 1])) <= 2e-6
        mass_inv = 1/np.asarray(state.mass)
        assert np.all(mass_inv >= 0.) and np.max(mass_inv) <= 4.
        local = (-np.asarray(sys.link.constraint_stiffness[1:, None])*np.asarray(state.j.pos[1:])
                 -np.asarray(sys.link.constraint_vel_damping[1:, None])*np.asarray(state.jd.vel[1:]))
        assert np.max(np.abs(local)) <= 84.
        maxima["joint_position_error"] = max(maxima["joint_position_error"], float(np.max(np.abs(state.j.pos[1:]))))
        maxima["com_velocity"] = max(maxima["com_velocity"], float(np.max(np.abs(state.xd_i.vel))))
        maxima["cache_velocity"] = max(maxima["cache_velocity"], float(np.max(np.abs(state.jd.vel[1:]))))
        maxima["local_force"] = max(maxima["local_force"], float(np.max(np.abs(local))))
        minima["reset_com_height"] = min(minima["reset_com_height"], float(np.min(initial_height)))
        for action in (jnp.zeros(6), jnp.ones(6), -jnp.ones(6)):
            force, acceleration = force_and_acceleration(state, action)
            assert np.max(np.abs(force)) <= 3528. and np.max(np.abs(acceleration)) <= 14200.
            pre_height = initial_height+dt*(np.asarray(state.xd_i.vel[:, 2])+dt*np.asarray(acceleration[:, 2]))
            post_height = np.asarray(step(state, action).x_i.pos[:, 2])
            assert np.all(np.isfinite(post_height))
            assert np.min(pre_height) >= -0.5 and np.min(post_height) >= -0.5
            minima["precontact_height"] = min(minima["precontact_height"], float(np.min(pre_height)))
            minima["first_step_height"] = min(minima["first_step_height"], float(np.min(post_height)))
            maxima["acceleration"] = max(maxima["acceleration"], float(np.max(np.abs(acceleration))))
    return {"reset_cases": len(states), "first_physics_steps": 3*len(states), **minima, **maxima}


if __name__ == "__main__":
    print(json.dumps(test_reset_and_first_physics_substep(), indent=2))
