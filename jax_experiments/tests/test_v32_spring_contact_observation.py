"""Compare the ground-contact closure and reduced observation with Brax source."""

import numpy as np
from jax import numpy as jnp
from brax import actuator, com, contact, kinematics, math
from brax.base import Motion, Transform
from brax.spring import collisions, joints, pipeline

from jax_experiments.analysis.audit_v32_spring_contact_rollout import MATRIX_RESOLVE
from jax_experiments.envs.brax_env import _build_core_fns
from jax_experiments.envs.stochastic_mode_env import StochasticModeEnv


def _rotate(v, q):
    s, u = q[0], q[1:]
    return 2*np.dot(u, v)*u + (s*s-np.dot(u, u))*v + 2*s*np.cross(u, v)


def _angle(axis, ref, child):
    return np.angle(complex(np.dot(ref, child), np.dot(np.cross(ref, child), axis)))


def _safe_norm(v):
    return 0. if np.all(np.abs(v) <= 1e-8) else np.linalg.norm(v)


def test_ground_contact_closure():
    env = StochasticModeEnv("HalfCheetah-v2", family="actuator_polarity", backend="spring")
    sys = env.env.sys
    initial = pipeline.init(sys, sys.init_q, jnp.zeros(sys.qd_size()))
    rng = np.random.default_rng(90285)
    coverage = {"active": 0, "inactive": 0, "friction": 0, "safe_zero": 0}
    for case in range(12):
        pos = np.asarray(initial.x.pos).copy()
        pos[:, 2] += 1. if case == 11 else -0.4
        x = initial.x.replace(pos=jnp.asarray(pos))
        world_motion = Motion.zero((7,))
        x_i, _ = com.from_world(sys, x, world_motion)
        if case < 6:
            speed = [0., 0.5e-8, 2e-8, 0.0009, 0.0011, 0.5][case]
            vel = np.tile([speed, 0., -1.], (7, 1)).astype(np.float32)
            ang = np.zeros((7, 3), np.float32)
        else:
            vel, ang = rng.uniform(-2, 2, size=(2, 7, 3)).astype(np.float32)
        inertia = np.asarray(com.inv_inertia(sys, x))
        # Array.take defaults to a flattened gather in this installed resolver.
        contact_inertia = inertia.ravel()[np.arange(7)]
        mass_inv = 1 / np.asarray(initial.mass)
        state = initial.replace(x=x, x_i=x_i, xd_i=Motion(vel=jnp.asarray(vel), ang=jnp.asarray(ang)),
                                i_inv=jnp.asarray(inertia))
        c = contact.get(sys, x)
        assert c is not None
        gathered = jnp.asarray(inertia).take(jnp.asarray(c.link_idx).T)
        assert gathered.shape == (len(c.dist), 2)
        np.testing.assert_array_equal(gathered[:, 1], contact_inertia[np.asarray(c.link_idx[1])])
        np.testing.assert_array_equal(c.link_idx[0], -np.ones(len(c.dist), dtype=np.int32))
        np.testing.assert_allclose(c.frame[:, 0], np.tile([0., 0., 1.], (len(c.dist), 1)), atol=1e-6)
        force, torque, counts = np.zeros((7, 3)), np.zeros((7, 3)), np.zeros(7)
        for j, body in enumerate(np.asarray(c.link_idx[1])):
            n = np.asarray(c.frame[j, 0], dtype=np.float64)
            lever = np.asarray(c.pos[j], dtype=np.float64) - np.asarray(x_i.pos[body], dtype=np.float64)
            point_vel = vel[body].astype(np.float64) + np.cross(ang[body], lever)
            normal_vel = np.dot(n, point_vel)
            angular_mass = contact_inertia[body]*np.dot(np.cross(lever, n), np.cross(lever, n))
            normal_den = mass_inv[body] + angular_mass
            assert normal_den >= mass_inv[body] - 1e-6
            raw = (-(1+float(c.elasticity[j]))*normal_vel
                   - float(sys.baumgarte_erp)/float(sys.opt.timestep)*float(c.dist[j]))/normal_den
            active = float(c.dist[j]) < 0 and normal_vel < 0 and raw > 0
            impulse = raw if active else 0.
            tangent = point_vel - normal_vel*n
            speed = _safe_norm(tangent)
            direction = tangent/(1e-6+speed)
            assert np.max(np.abs(direction)) <= 1 + 1e-6
            tangent_den = mass_inv[body] + contact_inertia[body]*np.dot(np.cross(lever, direction), np.cross(lever, direction))
            assert tangent_den >= mass_inv[body] - 1e-6
            drag = min(speed/tangent_den, float(c.friction[j, 0])*raw) if active and speed > 1e-3 else 0.
            f = impulse*n - drag*direction
            force[body] += f
            torque[body] += np.cross(lever, f)
            counts[body] += active
            coverage["active" if active else "inactive"] += 1
            coverage["friction"] += drag > 0
            coverage["safe_zero"] += speed == 0
        force /= counts[:, None] + 1e-8
        torque /= counts[:, None] + 1e-8
        expected_v = mass_inv[:, None]*force
        expected_w = np.einsum("ijk,ik->ij", inertia, torque)
        actual = collisions.resolve(sys, state)
        np.testing.assert_allclose(actual.vel, expected_v, rtol=6e-5, atol=2e-3)
        np.testing.assert_allclose(actual.ang, expected_w, rtol=6e-5, atol=2e-3)
    assert all(count > 0 for count in coverage.values()), coverage
    return coverage


def test_halfcheetah_reduced_coordinates_and_observation():
    env = StochasticModeEnv("HalfCheetah-v2", family="actuator_polarity", backend="spring")
    sys = env.env.sys
    initial = pipeline.init(sys, sys.init_q, jnp.zeros(sys.qd_size()))
    _, reward_obs, _ = _build_core_fns(env.env, "HalfCheetah-v2")
    rng = np.random.default_rng(90286)
    y, z, x_axis = np.eye(3)[1], np.eye(3)[2], np.eye(3)[0]
    for _ in range(12):
        pos, vel, ang = rng.uniform(-2, 2, size=(3, 7, 3)).astype(np.float32)
        rot = rng.normal(size=(7, 4)).astype(np.float32)
        rot /= np.linalg.norm(rot, axis=1, keepdims=True)
        transforms = Transform(pos=jnp.asarray(pos), rot=jnp.asarray(rot))
        motions = Motion(vel=jnp.asarray(vel), ang=jnp.asarray(ang))
        j, jd, _, _ = kinematics.world_to_joint(sys, transforms, motions)
        expected_q = [float(j.pos[0, 0]), float(j.pos[0, 2])]
        q0 = np.asarray(j.rot[0])
        child_y, child_x = _rotate(y, q0), _rotate(x_axis, q0)
        expected_q.append(_angle(-child_y, child_x, np.cross(child_y, z)))
        root_angular = _rotate(np.asarray(jd.ang[0]), q0*np.array([1., -1., -1., -1.]))
        expected_qd = [float(jd.vel[0, 0]), float(jd.vel[0, 2]), np.dot(child_y, root_angular)]
        for i in range(1, 7):
            qi = np.asarray(j.rot[i])
            expected_q.append(_angle(y, z, np.cross(_rotate(x_axis, qi), y)))
            expected_qd.append(np.dot(_rotate(y, qi), jd.ang[i]))
        q, qd = kinematics.inverse(sys, j, jd)
        np.testing.assert_allclose(q, expected_q, rtol=3e-5, atol=3e-5)
        np.testing.assert_allclose(qd, expected_qd, rtol=3e-5, atol=3e-5)
        post = initial.replace(x=transforms, q=q, qd=qd)
        _, obs, done = reward_obs(initial, post, jnp.zeros(6))
        expected_obs = np.r_[expected_q[1:], expected_qd]
        assert obs.shape == (17,) and float(done) == 0
        np.testing.assert_allclose(obs, expected_obs, rtol=3e-5, atol=3e-5)


def test_flat_contact_inertia_negative_denominator():
    env = StochasticModeEnv("HalfCheetah-v2", family="actuator_polarity", backend="spring")
    sys = env.env.sys
    initial = pipeline.init(sys, sys.init_q, jnp.zeros(sys.qd_size()))
    rot = np.asarray(initial.x.rot).copy()
    rot[0] = [0.8, 0., 0.6, 0.]
    np.testing.assert_allclose(np.linalg.norm(rot, axis=1), 1., atol=2e-6)
    x = initial.x.replace(rot=jnp.asarray(rot))
    x_i, _ = com.from_world(sys, x, Motion.zero((7,)))
    inertia = np.asarray(com.inv_inertia(sys, x))
    scalar = float(inertia.ravel()[6])
    assert scalar < 0
    c = contact.get(sys, x)
    index = int(np.flatnonzero(np.asarray(c.link_idx[1]) == 6)[0])
    original_lever = np.asarray(c.pos[index]) - np.asarray(x_i.pos[6])
    mass_inv = 1/float(initial.mass[6])
    direction = np.array([1/(1+1e-6), 0., 0.])
    critical_z = np.sqrt(mass_inv/(-scalar*direction[0]**2) - original_lever[1]**2)
    pos = np.asarray(x.pos).copy()
    pos[6, 2] += 2*(original_lever[2] - 1.2*critical_z)
    x = x.replace(pos=jnp.asarray(pos))
    x_i, _ = com.from_world(sys, x, Motion.zero((7,)))
    c = contact.get(sys, x)
    lever = np.asarray(c.pos[index]) - np.asarray(x_i.pos[6])
    normal = np.asarray(c.frame[index, 0])
    normal_den = mass_inv + scalar*np.sum(np.cross(lever, normal)**2)
    tangent_den = mass_inv + scalar*np.sum(np.cross(lever, direction)**2)
    matrix_den = mass_inv + np.dot(direction, np.cross(inertia[6] @ np.cross(lever, direction), lever))
    assert float(c.dist[index]) < 0 and normal_den > 0 and tangent_den < 0 and matrix_den >= mass_inv
    velocities = np.tile([1., 0., -1.], (7, 1)).astype(np.float32)
    state = initial.replace(x=x, x_i=x_i, i_inv=jnp.asarray(inertia),
                            xd_i=Motion(vel=jnp.asarray(velocities), ang=jnp.zeros((7, 3))))
    delta = collisions.resolve(sys, state)
    # Negative drag accelerates along +x in this source branch.
    assert float(delta.vel[6, 0]) > 0
    matrix_delta = MATRIX_RESOLVE(sys, state)
    assert float(matrix_delta.vel[6, 0]) <= 0
    return {"scalar": scalar, "normal_den": float(normal_den), "tangent_den": float(tangent_den),
            "matrix_tangent_den": float(matrix_den), "delta_vx": float(delta.vel[6, 0]),
            "matrix_delta_vx": float(matrix_delta.vel[6, 0]), "com_z": float(x_i.pos[6, 2])}


def test_scalar_geometric_denominator_margins():
    env = StochasticModeEnv("HalfCheetah-v2", family="actuator_polarity", backend="spring")
    sys = env.env.sys
    initial = pipeline.init(sys, sys.init_q, jnp.zeros(sys.qd_size()))
    weights = 1/np.diagonal(np.asarray(sys.link.inertia.i), axis1=1, axis2=2)**(
        1-float(sys.spring_inertia_scale))
    mass_inv = 1/np.asarray(initial.mass)
    non_diagonal_bodies = [1, 2, 3, 5, 6]
    assert np.all(weights[0] >= 0) and np.max(weights[0]) <= 9/4
    assert np.min(mass_inv) >= 1/4 and np.min(mass_inv[non_diagonal_bodies]) >= 7/10
    inertia_rot = np.asarray(sys.link.inertia.transform.rot)
    np.testing.assert_allclose(np.linalg.norm(inertia_rot[0]), 1., atol=2e-6)
    capsule_offsets = []
    for geom in np.flatnonzero(np.asarray(sys.geom_type) == 3):
        body = int(sys.geom_bodyid[geom])-1
        radius, half_length = np.asarray(sys.geom_size[geom])[:2]
        assert 0 <= radius <= 1/20
        np.testing.assert_allclose(np.linalg.norm(sys.geom_quat[geom]), 1., atol=2e-6)
        if body in non_diagonal_bodies:
            np.testing.assert_array_equal(sys.geom_pos[geom], sys.link.inertia.transform.pos[body])
            assert abs(half_length) <= 1/5
            capsule_offsets.append(float(half_length))
    rng = np.random.default_rng(90287)
    normal_min, certified_tangent_min = np.inf, np.inf
    normal_cases, certified_tangent_cases = 0, 0
    maximum_normal_geometry_error = 0.
    for case in range(12):
        rot = rng.normal(size=(7, 4)).astype(np.float32)
        rot /= np.linalg.norm(rot, axis=1, keepdims=True)
        com_pos = rng.uniform(-3, 3, size=(7, 3)).astype(np.float32)
        com_pos[:, 2] = rng.uniform(-0.49, 0.2, size=7) if case < 8 else rng.uniform(-30, -2, size=7)
        pos = com_pos-np.stack([_rotate(np.asarray(sys.link.inertia.transform.pos[i]), rot[i]) for i in range(7)])
        x = Transform(pos=jnp.asarray(pos), rot=jnp.asarray(rot))
        x_i, _ = com.from_world(sys, x, Motion.zero((7,)))
        inertia = np.asarray(com.inv_inertia(sys, x))
        scalar = inertia.ravel()[:7]
        c = contact.get(sys, x)
        np.testing.assert_array_equal(c.link_idx[0], -np.ones(len(c.dist), dtype=np.int32))
        np.testing.assert_allclose(c.frame[:, 0], np.tile([0., 0., 1.], (len(c.dist), 1)), atol=1e-6)
        for j, body in enumerate(np.asarray(c.link_idx[1])):
            lever = np.asarray(c.pos[j], np.float64)-np.asarray(x_i.pos[body], np.float64)
            normal = np.asarray(c.frame[j, 0], np.float64)
            normal_den = mass_inv[body]+scalar[body]*np.dot(np.cross(lever, normal), np.cross(lever, normal))
            assert normal_den >= 1/4-2e-6
            normal_min = min(normal_min, normal_den)
            normal_cases += 1
            if body in non_diagonal_bodies:
                geom = int(c.geom2[j])
                q = np.asarray(math.quat_mul(x.rot[body], sys.geom_quat[geom]))
                axis = _rotate(np.array([0., 0., 1.]), q)
                offset = float(sys.geom_size[geom, 1])
                expected = offset**2*(axis[0]**2+axis[1]**2)
                maximum_normal_geometry_error = max(maximum_normal_geometry_error,
                    abs(np.dot(np.cross(lever, normal), np.cross(lever, normal))-expected))
            tangent = rng.uniform(-2, 2, size=3)
            tangent[2] = 0.
            if case in (0, 1):
                tangent = np.array([0., 0., 0.]) if case == 0 else np.array([5e-9, 5e-9, 0.])
            direction = tangent/(1e-6+_safe_norm(tangent))
            assert np.dot(direction, direction) <= 1+1e-6
            in_floor_region = body in (0, 4) or float(x_i.pos[body, 2]) >= -0.5
            if float(c.dist[j]) <= 0 and in_floor_region:
                tangent_den = mass_inv[body]+scalar[body]*np.dot(np.cross(lever, direction), np.cross(lever, direction))
                assert tangent_den >= 1/4-2e-6
                certified_tangent_min = min(certified_tangent_min, tangent_den)
                certified_tangent_cases += 1
    assert normal_cases == 12*len(c.dist) and certified_tangent_cases > 0
    assert maximum_normal_geometry_error <= 2e-6
    return {"root_weight_max": float(np.max(weights[0])),
            "minimum_inverse_mass": float(np.min(mass_inv)),
            "minimum_non_diagonal_inverse_mass": float(np.min(mass_inv[non_diagonal_bodies])),
            "maximum_non_diagonal_half_length": max(capsule_offsets),
            "normal_cases": normal_cases, "normal_denominator_min": float(normal_min),
            "certified_tangent_cases": certified_tangent_cases,
            "certified_tangent_denominator_min": float(certified_tangent_min),
            "maximum_normal_geometry_error": maximum_normal_geometry_error}


def test_scalar_vertical_height_balance():
    env = StochasticModeEnv("HalfCheetah-v2", family="actuator_polarity", backend="spring")
    sys = env.env.sys
    assert not sys.enable_fluid
    dt = float(sys.opt.timestep)
    damping = np.exp(float(sys.vel_damping)*dt)
    rng = np.random.default_rng(90288)
    viable_cases, inactive_links = 0, 0
    minimum_vertical_impulse, maximum_balance_error = np.inf, 0.
    for case in range(12):
        q = np.asarray(sys.init_q).copy()+rng.uniform(-0.1, 0.1, sys.q_size())
        q[1] += (0., -0.45, -0.7)[case % 3]
        qd = rng.uniform(-2, 2, sys.qd_size()).astype(np.float32)
        state = pipeline.init(sys, jnp.asarray(q, dtype=jnp.float32), jnp.asarray(qd))
        action = jnp.asarray(rng.uniform(-1, 1, 6), dtype=jnp.float32)
        inertia = com.inv_inertia(sys, state.x)
        state = state.replace(i_inv=inertia)
        tau = actuator.to_tau(sys, action, state.q, state.qd)
        force = joints.resolve(sys, state, tau)
        opposite_tau = actuator.to_tau(sys, -action, state.q, state.qd)
        opposite_force = joints.resolve(sys, state, opposite_tau)
        np.testing.assert_array_equal(force.vel, opposite_force.vel)
        acceleration = Motion(
            vel=sys.gravity+force.vel/state.mass[:, None],
            ang=jnp.einsum("ijk,ik->ij", inertia, force.ang))
        pre_contact = state.replace(xd_i=state.xd_i+acceleration*dt)
        delta = collisions.resolve(sys, pre_contact)
        vertical_impulse = np.asarray(delta.vel[:, 2], dtype=np.float64)
        assert np.all(np.isfinite(vertical_impulse)) and np.min(vertical_impulse) >= -2e-6
        minimum_vertical_impulse = min(minimum_vertical_impulse, float(np.min(vertical_impulse)))
        inactive_links += int(np.count_nonzero(vertical_impulse == 0.))
        lower = np.asarray(state.x_i.pos[:, 2], dtype=np.float64)+dt*damping*np.asarray(
            pre_contact.xd_i.vel[:, 2], dtype=np.float64)
        expected = lower+dt*vertical_impulse
        post = pipeline.step(sys, state, action)
        actual = np.asarray(post.x_i.pos[:, 2], dtype=np.float64)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
        maximum_balance_error = max(maximum_balance_error, float(np.max(np.abs(actual-expected))))
        viable = lower >= -0.5+2e-6
        assert np.all(actual[viable] >= -0.5-2e-6)
        viable_cases += int(np.count_nonzero(viable))
        # The exact criterion uses post-contact velocity, not just initial height.
        threshold = (-0.5-np.asarray(state.x_i.pos[:, 2], dtype=np.float64))/dt
        margin = np.asarray(post.xd_i.vel[:, 2], dtype=np.float64)-threshold
        away_from_boundary = np.abs(actual+0.5) > 2e-6
        np.testing.assert_array_equal((actual >= -0.5)[away_from_boundary],
                                      (margin >= 0.)[away_from_boundary])
    assert viable_cases > 0 and inactive_links > 0
    return {"source_steps": 12, "link_cases": 84, "action_independence_cases": 12,
            "precontact_viable_cases": viable_cases,
            "inactive_links": inactive_links, "minimum_vertical_collision_increment": minimum_vertical_impulse,
            "maximum_height_balance_error": maximum_balance_error}


if __name__ == "__main__":
    counts = test_ground_contact_closure()
    test_halfcheetah_reduced_coordinates_and_observation()
    obstruction = test_flat_contact_inertia_negative_denominator()
    margins = test_scalar_geometric_denominator_margins()
    heights = test_scalar_vertical_height_balance()
    print(f"PASS: 12 scalar-source contact cases {counts}; 12 inverse/17-coordinate observation cases")
    print(f"SOURCE OBSTRUCTION: unit-rotation maximal state, not asserted reachable from reset: {obstruction}")
    print(f"GEOMETRIC MARGINS: normal unconditional in unit states; tangent conditional on COM floor: {margins}")
    print(f"VERTICAL BALANCE: source integration and position-velocity viability: {heights}")
