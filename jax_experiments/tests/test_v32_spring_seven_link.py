"""Check the frame and seven-link formulas translated in SpringSevenLink.lean."""

import numpy as np
from jax import numpy as jnp
from brax import actuator, com, contact, kinematics, math
from brax.base import Motion, Transform
from brax.spring import collisions, joints, pipeline

from jax_experiments.analysis.audit_v32_spring_contact_rollout import MATRIX_RESOLVE, MATRIX_STEP
from jax_experiments.envs.stochastic_mode_env import StochasticModeEnv


def _rotate(v, q):
    s, u = q[0], q[1:]
    return 2 * np.dot(u, v) * u + (s*s - np.dot(u, u)) * v + 2*s*np.cross(u, v)


def _conjugate(q):
    return q * [1, -1, -1, -1]


def _compose(x, t):
    return x[0] + _rotate(t[0], x[1]), np.asarray(math.quat_mul(x[1], t[1]))


def _local(x, t):
    return _rotate(x[0] - t[0], _conjugate(t[1])), np.asarray(math.quat_mul(_conjugate(t[1]), x[1]))


def _close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=8e-3)


def _capsule_contact_formula(sys, x):
    c = contact.get(sys, x)
    np.testing.assert_array_equal(c.link_idx[0], -np.ones(len(c.dist), dtype=np.int32))
    for geom in np.unique(c.geom2):
        body = int(sys.geom_bodyid[geom]) - 1
        assert int(sys.geom_type[geom]) == 3
        center, rotation = _compose((np.asarray(x.pos[body]), np.asarray(x.rot[body])),
                                    (np.asarray(sys.geom_pos[geom]), np.asarray(sys.geom_quat[geom])))
        axis = _rotate(np.array([0., 0., 1.]), rotation)
        radius, half_length = np.asarray(sys.geom_size[geom])[:2]
        endpoints = center + np.array([-half_length, half_length])[:, None]*axis
        distances = endpoints[:, 2]-radius
        points = endpoints.copy()
        points[:, 2] = distances/2
        ids = np.flatnonzero(np.asarray(c.geom2) == geom)
        assert len(ids) == 2
        actual_order = ids[np.argsort(np.asarray(c.dist)[ids])]
        expected_order = np.argsort(distances)
        _close(c.dist[actual_order], distances[expected_order])
        _close(c.pos[actual_order], points[expected_order])
        _close(c.frame[ids, 0], np.tile([0., 0., 1.], (2, 1)))


def _seven_link_source_assembly(matrix_pipeline):
    env = StochasticModeEnv("HalfCheetah-v2", family="actuator_polarity", backend="spring")
    sys = env.env.sys
    parents = np.array([-1, 0, 1, 2, 0, 4, 5])
    np.testing.assert_array_equal(sys.link_parents, parents)
    assert max(np.count_nonzero(parents == i) for i in range(7)) == 2
    assert not sys.enable_fluid
    initial = pipeline.init(sys, sys.init_q, jnp.zeros(sys.qd_size()))
    rng = np.random.default_rng(90284)
    for _ in range(8):
        pos, vel, ang = rng.uniform(-3, 3, size=(3, 7, 3)).astype(np.float32)
        rot = rng.normal(size=(7, 4)).astype(np.float32)
        rot /= np.linalg.norm(rot, axis=1, keepdims=True)
        x, xd = Transform(pos=jnp.asarray(pos), rot=jnp.asarray(rot)), Motion(vel=jnp.asarray(vel), ang=jnp.asarray(ang))
        if matrix_pipeline:
            _capsule_contact_formula(sys, x)
        x_i, xd_i = com.from_world(sys, x, xd)
        world_x, world_v = com.to_world(sys, x_i, xd_i)
        for i in range(7):
            lever = _rotate(-np.asarray(sys.link.inertia.transform.pos[i]), np.asarray(x_i.rot[i]))
            _close(world_x.pos[i], np.asarray(x_i.pos[i]) + lever)
            _close(world_v.vel[i], np.asarray(xd_i.vel[i]) + np.cross(xd_i.ang[i], lever))
            _close(world_x.rot[i], x_i.rot[i])
            _close(world_v.ang[i], xd_i.ang[i])

        j, jd, ap, ac = kinematics.world_to_joint(sys, x, xd)
        manual_ap, manual_ac = [], []
        for i, parent in enumerate(parents):
            xp = (np.zeros(3), np.array([1., 0., 0., 0.])) if parent < 0 else (pos[parent], rot[parent])
            vp, wp = (np.zeros(3), np.zeros(3)) if parent < 0 else (vel[parent], ang[parent])
            link = (np.asarray(sys.link.transform.pos[i]), np.asarray(sys.link.transform.rot[i]))
            joint = (np.asarray(sys.link.joint.pos[i]), np.asarray(sys.link.joint.rot[i]))
            a_p, a_c = _compose(_compose(xp, link), joint), _compose((pos[i], rot[i]), joint)
            manual_ap.append(a_p)
            manual_ac.append(a_c)
            local = _local(a_c, a_p)
            relative_v = _rotate(vel[i] - vp - np.cross(wp, xp[0] - a_p[0]), _conjugate(a_p[1]))
            relative_w = _rotate(ang[i] - wp, _conjugate(a_p[1]))
            _close(ap.pos[i], a_p[0])
            _close(ap.rot[i], a_p[1])
            _close(ac.pos[i], a_c[0])
            _close(ac.rot[i], a_c[1])
            _close(j.pos[i], local[0])
            _close(j.rot[i], local[1])
            _close(jd.vel[i], relative_v)
            _close(jd.ang[i], relative_w)
            np.testing.assert_allclose(np.linalg.norm([a_p[1], a_c[1], local[1]], axis=1), 1., atol=2e-6)

        commands = jnp.asarray(rng.uniform(-2, 2, size=6), dtype=jnp.float32)
        tau = actuator.to_tau(sys, commands, initial.q, initial.qd)
        local_forces = []
        for i in range(7):
            link, joint, motion = sys.link.take(i), j.take(i), jd.take(i)
            if i == 0:
                force = joints._three_dof(link, joint, motion, sys.dof.take(jnp.arange(3)), jnp.zeros(3))
            else:
                force = joints._one_dof(link, joint, motion, sys.dof.take(jnp.array([i + 2])), tau[i + 2])
                line = np.cross(_rotate(np.array([1., 0., 0.]), np.asarray(joint.rot)), [0., 1., 0.])
                theta = np.angle(complex(np.dot([0., 0., 1.], line),
                                        np.dot(np.cross([0., 0., 1.], line), [0., 1., 0.])))
                frame, parity = kinematics.link_to_joint_frame(sys.dof.take(jnp.array([i + 2])).motion)
                _, angles, _ = kinematics.axis_angle_ang(joint, frame, parity)
                _close(angles[0], theta)
                lo, hi = float(sys.dof.limit[0][i + 2]), float(sys.dof.limit[1][i + 2])
                error = theta - hi if theta > hi else theta - lo if theta < lo else 0.
                axis = np.array([0., 1., 0.])
                expected_w = (-float(link.constraint_stiffness)*np.cross(axis, _rotate(axis, np.asarray(joint.rot)))
                              + float(tau[i + 2])*axis - float(link.constraint_ang_damping)*np.asarray(motion.ang)
                              - float(link.constraint_limit_stiffness)*error*axis)
                _close(force.ang, expected_w)
            local_forces.append(force)

        expected_force = np.zeros((7, 3))
        expected_torque = np.zeros((7, 3))
        for i, parent in enumerate(parents):
            f = _rotate(np.asarray(local_forces[i].vel), manual_ap[i][1])
            t = _rotate(np.asarray(local_forces[i].ang), manual_ap[i][1])
            expected_force[i] += f
            expected_torque[i] += t + np.cross(manual_ac[i][0] - x_i.pos[i], f)
            if parent >= 0:
                expected_force[parent] -= f
                expected_torque[parent] -= t + np.cross(manual_ap[i][0] - x_i.pos[parent], f)
        state = initial.replace(x=x, xd=xd, x_i=x_i, xd_i=xd_i, j=j, jd=jd, a_p=ap, a_c=ac)
        actual_force = joints.resolve(sys, state, tau)
        _close(actual_force.vel, expected_force)
        _close(actual_force.ang, expected_torque)

        # Match the source's rotated diagonal inertia in the assembled acceleration.
        actual_inverse = np.asarray(com.inv_inertia(sys, x))
        expected_angular_acceleration = []
        for i in range(7):
            q = np.asarray(math.quat_mul(rot[i], sys.link.inertia.transform.rot[i]))
            axes = np.stack([_rotate(basis, q) for basis in np.eye(3)], axis=1)
            weights = 1 / np.diag(np.asarray(sys.link.inertia.i[i])) ** (1 - float(sys.spring_inertia_scale))
            expected_angular_acceleration.append((axes*weights) @ (axes.T @ expected_torque[i]))
        _close(np.einsum("ijk,ik->ij", actual_inverse, actual_force.ang), expected_angular_acceleration)
        mass = np.asarray(sys.link.inertia.mass) ** (1 - float(sys.spring_mass_scale))
        _close(np.asarray(sys.gravity) + np.asarray(actual_force.vel)/mass[:, None],
               np.asarray(sys.gravity) + expected_force/mass[:, None])

        # Close the source ordering through integration and cache reconstruction.
        acceleration = Motion(
            vel=sys.gravity + actual_force.vel/jnp.asarray(mass[:, None]),
            ang=jnp.asarray(expected_angular_acceleration))
        dt = float(sys.opt.timestep)
        pre_contact = state.replace(i_inv=jnp.asarray(actual_inverse), xd_i=xd_i + acceleration*dt)
        delta = (MATRIX_RESOLVE if matrix_pipeline else collisions.resolve)(sys, pre_contact)
        nv = np.exp(float(sys.vel_damping)*dt)*(np.asarray(xd_i.vel)+dt*np.asarray(acceleration.vel)) + delta.vel
        nw = np.exp(float(sys.ang_damping)*dt)*(np.asarray(xd_i.ang)+dt*np.asarray(acceleration.ang)) + delta.ang
        new_pos = np.asarray(x_i.pos) + dt*nv
        new_rot = []
        for i in range(7):
            q = np.asarray(x_i.rot[i])
            increment = math.quat_mul(jnp.asarray(np.r_[0., 0.5*dt*nw[i]]), jnp.asarray(q))
            raw = q + np.asarray(increment)
            assert np.linalg.norm(raw) >= 1 - 2e-6
            new_rot.append(raw/np.linalg.norm(raw))
        new_rot = np.asarray(new_rot)
        rebuilt_x, rebuilt_v = [], []
        for i in range(7):
            lever = _rotate(-np.asarray(sys.link.inertia.transform.pos[i]), new_rot[i])
            rebuilt_x.append(new_pos[i] + lever)
            rebuilt_v.append(nv[i] + np.cross(nw[i], lever))
        step = (MATRIX_STEP if matrix_pipeline else pipeline.step)(sys, state, commands)
        _close(step.x_i.pos, new_pos)
        _close(step.x_i.rot, new_rot)
        _close(step.xd_i.vel, nv)
        _close(step.xd_i.ang, nw)
        _close(step.x.pos, rebuilt_x)
        _close(step.xd.vel, rebuilt_v)
        for i, parent in enumerate(parents):
            xp = (np.zeros(3), np.array([1., 0., 0., 0.])) if parent < 0 else (rebuilt_x[parent], new_rot[parent])
            vp, wp = (np.zeros(3), np.zeros(3)) if parent < 0 else (rebuilt_v[parent], nw[parent])
            link = (np.asarray(sys.link.transform.pos[i]), np.asarray(sys.link.transform.rot[i]))
            joint = (np.asarray(sys.link.joint.pos[i]), np.asarray(sys.link.joint.rot[i]))
            a_p, a_c = _compose(_compose(xp, link), joint), _compose((rebuilt_x[i], new_rot[i]), joint)
            relative = _local(a_c, a_p)
            _close(step.a_p.pos[i], a_p[0])
            _close(step.a_c.pos[i], a_c[0])
            _close(step.j.pos[i], relative[0])
            _close(step.j.rot[i], relative[1])
            _close(step.jd.vel[i], _rotate(rebuilt_v[i]-vp-np.cross(wp, xp[0]-a_p[0]), _conjugate(a_p[1])))
            _close(step.jd.ang[i], _rotate(nw[i]-wp, _conjugate(a_p[1])))


def test_seven_link_source_assembly():
    _seven_link_source_assembly(False)


def test_matrix_seven_link_source_assembly():
    _seven_link_source_assembly(True)


if __name__ == "__main__":
    test_seven_link_source_assembly()
    print("PASS: eight seven-link frame/cache, angle, wrench, acceleration and reconstructed-step source comparisons")
