"""Source/formula correspondence for the real-arithmetic Lean components.

Finite tests detect mistranslated formulas; they do not certify reachability.
"""

import numpy as np
from jax import numpy as jnp, tree_util
from brax import actuator, com, kinematics, math
from brax.base import Motion, Transform
from brax.spring import integrator, joints
from mujoco.mjx._src import collision_primitive

from jax_experiments.envs.stochastic_mode_env import StochasticModeEnv


def _close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=3e-3)


def test_halfcheetah_source_component_formulas():
    env = StochasticModeEnv("HalfCheetah-v2", family="actuator_polarity", backend="spring")
    sys = env.env.sys
    assert sys.link_types == "3111111"
    np.testing.assert_array_equal(sys.actuator.gain, np.ones(6))
    np.testing.assert_array_equal(sys.actuator.bias_q, np.zeros(6))
    np.testing.assert_array_equal(sys.actuator.bias_qd, np.zeros(6))
    np.testing.assert_array_equal(sys.actuator.qd_id, np.arange(3, 9))
    np.testing.assert_array_equal(sys.actuator.gear, [120, 90, 60, 120, 100, 100])
    np.testing.assert_array_equal(sys.actuator.ctrl_range, np.tile([-1, 1], (6, 1)))
    assert np.all(np.isneginf(np.asarray(sys.actuator.force_range)[:, 0]))
    assert np.all(np.isposinf(np.asarray(sys.actuator.force_range)[:, 1]))
    np.testing.assert_array_equal(sys.dof.motion.vel[3:], np.zeros((6, 3)))
    np.testing.assert_array_equal(sys.dof.motion.ang[3:], np.tile([0, 1, 0], (6, 1)))
    np.testing.assert_array_equal(sys.dof.motion.vel[:3], [[1, 0, 0], [0, 0, 1], [0, 0, 0]])
    np.testing.assert_array_equal(sys.dof.motion.ang[:3], [[0, 0, 0], [0, 0, 0], [0, 1, 0]])
    assert np.all(np.isneginf(np.asarray(sys.dof.limit[0][:3])))
    assert np.all(np.isposinf(np.asarray(sys.dof.limit[1][:3])))

    rng = np.random.default_rng(90283)
    for case in range(16):
        q = rng.normal(size=4).astype(np.float32)
        q /= np.linalg.norm(q)
        p, v, w = rng.uniform(-3, 3, size=(3, 3)).astype(np.float32)
        u = rng.uniform(-2, 2, size=6).astype(np.float32)
        joint = Transform(pos=jnp.asarray(p), rot=jnp.asarray(q))
        motion = Motion(vel=jnp.asarray(v), ang=jnp.asarray(w))

        # The sourceRotate polynomial and both capsule endpoint signs.
        rotated = np.asarray(math.rotate(jnp.asarray(v), jnp.asarray(q)))
        expected_rotate = (2 * np.dot(q[1:], v) * q[1:]
                           + (q[0] ** 2 - np.dot(q[1:], q[1:])) * v
                           + 2 * q[0] * np.cross(q[1:], v))
        _close(rotated, expected_rotate)
        assert np.max(np.abs(rotated)) <= 14 * np.max(np.abs(v)) + 1e-5
        axis = np.asarray(math.rotate(jnp.array([0., 0., 1.]), jnp.asarray(q)))
        for offset in [-0.3, 0.3]:
            endpoint = p + offset * axis
            dist, point = collision_primitive._plane_sphere(
                jnp.array([0., 0., 1.]), jnp.zeros(3), jnp.asarray(endpoint), 0.05)
            _close(dist, endpoint[2] - 0.05)
            _close(point, [endpoint[0], endpoint[1], (endpoint[2] - 0.05) / 2])

        # q/qd do not affect this motor map, since both bias coefficients vanish.
        tau = actuator.to_tau(sys, jnp.asarray(u), jnp.full(sys.q_size(), 99.),
                              jnp.full(sys.qd_size(), -99.))
        _close(tau, np.r_[np.zeros(3), np.asarray(sys.actuator.gear) * np.clip(u, -1, 1)])
        assert np.max(np.abs(tau)) <= 120

        link_id = case % 6 + 1
        link = sys.link.take(link_id)
        dof = sys.dof.take(jnp.array([link_id + 2]))
        frame, parity = kinematics.link_to_joint_frame(dof.motion)
        _, angles, _ = kinematics.axis_angle_ang(joint, frame, parity)
        angle = float(angles[0])
        lo, hi = float(dof.limit[0][0]), float(dof.limit[1][0])
        error = angle - hi if angle > hi else angle - lo if angle < lo else 0.0
        assert abs(error) <= abs(angle) + abs(lo) + abs(hi) + 1e-6
        a = np.asarray(frame.ang[0])
        child = np.asarray(math.rotate(frame.ang[0], joint.rot))
        force = joints._one_dof(link, joint, motion, dof, tau[link_id + 2])
        _close(force.vel, -float(link.constraint_stiffness) * p
               - float(link.constraint_vel_damping) * v)
        _close(force.ang, -float(link.constraint_stiffness) * np.cross(a, child)
               + float(tau[link_id + 2]) * a - float(link.constraint_ang_damping) * w
               - float(link.constraint_limit_stiffness) * error * a)

        root = sys.link.take(0)
        root_force = joints._three_dof(root, joint, motion, sys.dof.take(jnp.arange(3)), jnp.zeros(3))
        y = jnp.array([0., 1., 0.])
        _close(root_force.vel, [0., -float(root.constraint_stiffness) * p[1]
                               - float(root.constraint_vel_damping) * v[1], 0.])
        _close(root_force.ang, -float(root.constraint_ang_damping) * w
               - float(root.constraint_stiffness) * np.cross(y, math.rotate(y, joint.rot)))

        # Compare com.inv_inertia with the positive outer-product representation.
        transforms = Transform(pos=jnp.zeros((7, 3)), rot=jnp.tile(joint.rot, (7, 1)))
        inertia_inv = np.asarray(com.inv_inertia(sys, transforms)[link_id])
        rotation = math.quat_mul(joint.rot, link.inertia.transform.rot)
        columns = np.stack([math.rotate(jnp.eye(3)[i], rotation) for i in range(3)], axis=1)
        weights = 1 / np.diag(np.asarray(link.inertia.i)) ** (1 - float(sys.spring_inertia_scale))
        outer_form = (columns * weights) @ columns.T
        _close(inertia_inv, outer_form)
        normal, lever = rng.normal(size=(2, 3))
        rn = np.cross(lever, normal)
        triple = np.dot(normal, np.cross(inertia_inv @ rn, lever))
        squares = np.sum(weights * (columns.T @ rn) ** 2)
        _close(triple, squares)
        assert triple >= -1e-5

        # Semi-implicit ordering: acceleration, damping, collision, then position.
        acceleration = Motion(vel=jnp.asarray(w), ang=jnp.asarray(v))
        collision = Motion(vel=jnp.asarray(p / 10), ang=jnp.asarray(w / 10))
        dt = float(sys.opt.timestep)
        batched = tree_util.tree_map(
            lambda x: x[None], (joint, motion + acceleration * dt, collision))
        next_x, next_v = integrator.integrate(sys, *batched)
        next_x, next_v = next_x.take(0), next_v.take(0)
        expected_v = np.exp(float(sys.vel_damping) * dt) * (v + dt * w) + p / 10
        expected_w = np.exp(float(sys.ang_damping) * dt) * (w + dt * v) + w / 10
        _close(next_v.vel, expected_v)
        _close(next_v.ang, expected_w)
        _close(next_x.pos, p + dt * expected_v)
        raw_q = q + np.asarray(math.quat_mul(jnp.asarray(np.r_[0., 0.5 * dt * expected_w]), joint.rot))
        assert np.linalg.norm(raw_q) >= 1 - 2e-6
        _close(next_x.rot, raw_q / np.linalg.norm(raw_q))
        np.testing.assert_allclose(np.linalg.norm(next_x.rot), 1., atol=2e-6)


if __name__ == "__main__":
    test_halfcheetah_source_component_formulas()
    print("PASS: 16 source/formula cases, six revolute joints, planar root, capsule contacts, inertia, motors, integrator")
