"""Check scalar-contact recovery and leaf-joint dynamics against Brax source."""

import json

import jax
import numpy as np
from jax import numpy as jnp
from brax import actuator, com, contact, math
from brax.base import Motion
from brax.spring import collisions, joints, pipeline

from jax_experiments.envs.stochastic_mode_env import StochasticModeEnv


def test_contact_recovery_bound():
    env = StochasticModeEnv("HalfCheetah-v2", family="actuator_polarity", backend="spring")
    sys = env.base_sys
    dt = float(sys.opt.timestep)
    assert float(sys.vel_damping) == 0.
    for rotation in (sys.link.transform.rot[0], sys.link.joint.rot[0]):
        np.testing.assert_allclose(rotation, [1., 0., 0., 0.], atol=1e-7)
    leaf_bodies = (1, 2, 3, 5, 6)
    local_axes = np.asarray(jax.vmap(lambda q: jax.vmap(lambda e: math.rotate(e, q))(
        jnp.eye(3)))(sys.link.inertia.transform.rot))
    axis_residual = np.minimum(np.abs(local_axes[:, :, 1]),
                               np.max(np.abs(local_axes[:, :, [0, 2]]), axis=2))
    assert np.max(axis_residual) < 2e-6
    for offset in (sys.link.inertia.transform.pos, sys.link.transform.pos, sys.link.joint.pos):
        np.testing.assert_allclose(np.asarray(offset)[:, 1], 0., atol=2e-6)
    for rotation in (sys.link.transform.rot, sys.link.joint.rot):
        np.testing.assert_allclose(np.asarray(rotation)[:, [1, 3]], 0., atol=2e-6)

    @jax.jit
    def advance(state, action):
        state = state.replace(i_inv=com.inv_inertia(sys, state.x))
        force = joints.resolve(sys, state, actuator.to_tau(sys, action, state.q, state.qd))
        acceleration = Motion(vel=sys.gravity+force.vel/state.mass[:, None],
                              ang=jnp.einsum("ijk,ik->ij", state.i_inv, force.ang))
        pre = state.replace(xd_i=state.xd_i+acceleration*dt)
        return pre, contact.get(sys, state.x), collisions.resolve(sys, pre), pipeline.step(sys, state, action), force

    rng = np.random.default_rng(90310)
    steps, active_cases, separating_cases = 0, 0, 0
    maximum_balance_error = 0.
    maximum_planar_residual, maximum_zero_scalar = 0., 0.
    minimum_recovery_slack, minimum_height = np.inf, np.inf
    minimum_contact_barrier_margin = np.inf
    maximum_joint_sum_error, maximum_pre_momentum_error = 0., 0.
    maximum_post_momentum_error, maximum_foot_decomposition_error = 0., 0.
    minimum_freefall_slack, minimum_relative_budget_slack = np.inf, np.inf
    def check_planarity(state):
        return float(max(np.max(np.abs(np.asarray(state.x_i.pos)[:, 1])),
                         np.max(np.abs(np.asarray(state.x_i.rot)[:, [1, 3]])),
                         np.max(np.abs(np.asarray(state.xd_i.vel)[:, 1])),
                         np.max(np.abs(np.asarray(state.xd_i.ang)[:, [0, 2]]))))
    for case in range(4):
        state = env._reset_fn(sys, jax.random.PRNGKey(90310+case)).pipeline_state
        assert check_planarity(state) < 2e-6
        mass = np.asarray(state.mass, dtype=np.float64)
        total_mass = float(np.sum(mass))
        assert np.all(mass > 0.)
        initial_momentum = float(np.dot(mass, np.asarray(state.xd_i.vel[:, 2], dtype=np.float64)))
        coefficients = np.array([-np.sum(mass[1:4]), -np.sum(mass[2:4]), -mass[3],
                                 total_mass-np.sum(mass[4:]), total_mass-np.sum(mass[5:]),
                                 total_mass-mass[6]])
        assert np.all(np.abs(coefficients) <= total_mass)
        for substep in range(64):
            action = (jnp.zeros(6), jnp.ones(6), -jnp.ones(6),
                      jnp.array(rng.uniform(-1, 1, 6), dtype=jnp.float32))[case]
            pre, contacts, delta, post, force = advance(state, action)
            np.testing.assert_array_equal(np.asarray(post.mass), np.asarray(state.mass))
            old_vz = np.asarray(state.xd_i.vel[:, 2], dtype=np.float64)
            pre_vz = np.asarray(pre.xd_i.vel[:, 2], dtype=np.float64)
            post_vz = np.asarray(post.xd_i.vel[:, 2], dtype=np.float64)
            old_momentum, pre_momentum, post_momentum = (
                float(np.dot(mass, vz)) for vz in (old_vz, pre_vz, post_vz))
            gravity_drift = dt*total_mass*float(sys.gravity[2])
            contact_momentum = float(np.dot(mass, np.asarray(delta.vel[:, 2], dtype=np.float64)))
            joint_sum_error = abs(float(np.sum(np.asarray(force.vel[:, 2], dtype=np.float64))))
            pre_momentum_error = abs(pre_momentum-old_momentum-gravity_drift)
            post_momentum_error = abs(post_momentum-old_momentum-gravity_drift-contact_momentum)
            edges = pre_vz[1:]-pre_vz[np.array([0, 1, 2, 0, 4, 5])]
            correction = float(np.dot(coefficients, edges))
            decomposition_error = abs(total_mass*pre_vz[6]-pre_momentum-correction)
            freefall_slack = post_momentum-initial_momentum-(substep+1)*gravity_drift
            relative_slack = pre_vz[6]-(pre_momentum/total_mass-np.sum(np.abs(edges)))
            assert joint_sum_error < .03
            assert pre_momentum_error < 2e-4 and post_momentum_error < 2e-4
            assert decomposition_error < 1e-8 and contact_momentum >= -2e-6
            assert freefall_slack >= -2e-4 and relative_slack >= -2e-6
            maximum_joint_sum_error = max(maximum_joint_sum_error, joint_sum_error)
            maximum_pre_momentum_error = max(maximum_pre_momentum_error, pre_momentum_error)
            maximum_post_momentum_error = max(maximum_post_momentum_error, post_momentum_error)
            maximum_foot_decomposition_error = max(maximum_foot_decomposition_error, decomposition_error)
            minimum_freefall_slack = min(minimum_freefall_slack, freefall_slack)
            minimum_relative_budget_slack = min(minimum_relative_budget_slack, relative_slack)
            np.testing.assert_allclose(np.sum(np.asarray(state.x_i.rot)**2, axis=1), 1., atol=2e-6)
            np.testing.assert_allclose(contacts.frame[:, 0], np.tile([0., 0., 1.], (len(contacts.dist), 1)), atol=1e-6)
            np.testing.assert_array_equal(contacts.link_idx[0], -np.ones(len(contacts.dist), dtype=int))
            body = np.asarray(contacts.link_idx[1])
            mass_inv = 1/np.asarray(pre.mass, dtype=np.float64)
            scalar = np.asarray(pre.i_inv, dtype=np.float64).ravel()[body]
            zero_scalar = np.asarray(pre.i_inv, dtype=np.float64).ravel()[[1, 3, 5]]
            np.testing.assert_allclose(zero_scalar, 0., atol=2e-6)
            maximum_zero_scalar = max(maximum_zero_scalar, float(np.max(np.abs(zero_scalar))))
            maximum_planar_residual = max(maximum_planar_residual, check_planarity(post))
            assert maximum_planar_residual < 2e-6
            lever = np.asarray(contacts.pos, dtype=np.float64)-np.asarray(pre.x_i.pos, dtype=np.float64)[body]
            normal = np.asarray(contacts.frame[:, 0], dtype=np.float64)
            point_v = (np.asarray(pre.xd_i.vel, dtype=np.float64)[body]
                       +np.cross(np.asarray(pre.xd_i.ang, dtype=np.float64)[body], lever))
            nv = np.einsum("ij,ij->i", normal, point_v)
            den = mass_inv[body]+scalar*np.sum(np.cross(lever, normal)**2, axis=1)
            dist = np.asarray(contacts.dist, dtype=np.float64)
            elasticity = np.asarray(contacts.elasticity, dtype=np.float64)
            k = float(sys.baumgarte_erp)/dt
            assert np.all(elasticity >= 0.) and k >= 0.
            raw = (-(1+elasticity)*nv-k*dist)/den
            active = (dist < 0.) & (nv < 0.) & (raw > 0.)
            initial_z = np.asarray(state.x_i.pos[:, 2], dtype=np.float64)
            pre_z = initial_z+dt*np.asarray(pre.xd_i.vel[:, 2], dtype=np.float64)
            post_z = np.asarray(post.x_i.pos[:, 2], dtype=np.float64)
            reconstructed_z = pre_z+dt*np.asarray(delta.vel[:, 2], dtype=np.float64)
            np.testing.assert_allclose(post_z, reconstructed_z, rtol=2e-5, atol=2e-6)
            maximum_balance_error = max(maximum_balance_error, float(np.max(np.abs(post_z-reconstructed_z))))
            for b in leaf_bodies:
                indices = np.flatnonzero(body == b)
                assert len(indices) == (0 if b in (2, 5) else 2)
                assert np.all(den[indices] >= 0.25-2e-6)
                assert np.all(den[indices] <= mass_inv[b]+0.09+2e-6)
                selected = indices[(dist[indices] < 0.) & (nv[indices] < 0.)]
                np.testing.assert_array_equal(active[indices], (dist[indices] < 0.) & (nv[indices] < 0.))
                count = len(selected)
                expected_delta = mass_inv[b]*np.sum(raw[selected])/(count+1e-8)
                np.testing.assert_allclose(delta.vel[b, 2], expected_delta, rtol=2e-4, atol=2e-5)
                lower = mass_inv[b]/(count+1e-8)*np.sum(-nv[selected]-k*dist[selected])/(mass_inv[b]+0.09)
                if len(indices):
                    margin = pre_z[b]+dt*lower+0.5
                    assert margin >= 0.
                    minimum_contact_barrier_margin = min(minimum_contact_barrier_margin, float(margin))
                if count:
                    slack = float(delta.vel[b, 2])-lower
                    assert lower >= 0. and slack >= -2e-5-2e-4*abs(lower)
                    assert post_z[b] >= pre_z[b]+dt*lower-2e-6
                    minimum_recovery_slack = min(minimum_recovery_slack, slack)
                    active_cases += count
                separating_cases += int(np.count_nonzero((dist[indices] < 0.) & (nv[indices] >= 0.)))
            minimum_height = min(minimum_height, float(np.min(post_z[list(leaf_bodies)])))
            state = post
            steps += 1
    assert active_cases > 0 and separating_cases > 0
    return {"normal_reset_cases": 4, "source_substeps": steps,
            "leaf_candidate_counts": {str(b): 0 if b in (2, 5) else 2 for b in leaf_bodies},
            "approaching_leaf_contacts": active_cases, "separating_leaf_contacts": separating_cases,
            "maximum_vertical_balance_error": maximum_balance_error,
            "maximum_local_axis_residual": float(np.max(axis_residual)),
            "maximum_planar_residual": maximum_planar_residual,
            "maximum_zero_scalar_entry": maximum_zero_scalar,
            "minimum_recovery_slack": minimum_recovery_slack,
            "minimum_contact_barrier_margin": minimum_contact_barrier_margin,
            "minimum_leaf_com_height": minimum_height,
            "maximum_joint_vertical_sum_error": maximum_joint_sum_error,
            "maximum_precontact_momentum_error": maximum_pre_momentum_error,
            "maximum_postcontact_momentum_error": maximum_post_momentum_error,
            "maximum_foot_momentum_decomposition_error": maximum_foot_decomposition_error,
            "minimum_momentum_freefall_slack": minimum_freefall_slack,
            "minimum_relative_budget_slack": minimum_relative_budget_slack}


def test_paired_foot_recovery_bound():
    env = StochasticModeEnv("HalfCheetah-v2", family="actuator_polarity", backend="spring")
    sys = env.base_sys
    dt = float(sys.opt.timestep)
    k = float(sys.baumgarte_erp)/dt
    assert dt == 1/320 and k >= 32. and float(sys.vel_damping) == 0.
    base = env._reset_fn(sys, jax.random.PRNGKey(90310)).pipeline_state

    @jax.jit
    def resolve(z, vz, angular, root_pitch):
        root_rotation = jnp.array([jnp.cos(root_pitch/2), 0., jnp.sin(root_pitch/2), 0.])
        xi = base.x_i.replace(pos=base.x_i.pos.at[6, 2].set(z),
                              rot=base.x_i.rot.at[0].set(root_rotation))
        vi = base.xd_i.replace(vel=base.xd_i.vel.at[6, 2].set(vz),
                              ang=base.xd_i.ang.at[6].set(jnp.array([0., angular, 0.])))
        x, xd = com.to_world(sys, xi, vi)
        state = base.replace(x_i=xi, xd_i=vi, x=x, xd=xd, i_inv=com.inv_inertia(sys, x))
        return state, contact.get(sys, x), collisions.resolve(sys, state)

    cases, deep_cases = 0, 0
    active_counts, scalar_signs = set(), set()
    maximum_pair_sum_error = 0.
    minimum_recovery_slack, minimum_height_margin = np.inf, np.inf
    for root_pitch in (-0.6, 0.6):
        for z in (-0.5, -0.35, -0.201, -0.2, -0.1, 0.2):
            for vz in (-64., -0.1, 1.):
                for angular in (-2000., 0., 2000.):
                    state, contacts, delta = resolve(z, vz, angular, root_pitch)
                    selected = np.flatnonzero(np.asarray(contacts.link_idx[1]) == 6)
                    assert len(selected) == 2
                    lever = np.asarray(contacts.pos, dtype=np.float64)[selected]-np.asarray(state.x_i.pos[6])
                    nv = vz+np.cross(np.asarray(state.xd_i.ang[6], dtype=np.float64), lever)[:, 2]
                    pair_error = abs(float(np.sum(nv))-2*vz)
                    assert pair_error < 2e-4
                    maximum_pair_sum_error = max(maximum_pair_sum_error, pair_error)
                    dist = np.asarray(contacts.dist, dtype=np.float64)[selected]
                    np.testing.assert_allclose((lever[0]+lever[1])[:2], 0., atol=1e-7)
                    assert np.all(dist <= z+0.2+2e-6)
                    active = (dist < 0.) & (nv < 0.)
                    count = int(np.count_nonzero(active))
                    active_counts.add(count)
                    mass_inv = float(1/state.mass[6])
                    assert mass_inv >= 0.7
                    scalar = float(np.asarray(state.i_inv).ravel()[6])
                    scalar_signs.add(int(np.sign(scalar)))
                    lower = mass_inv/(count+1e-8)*np.sum(-nv[active]-k*dist[active])/(mass_inv+0.09)
                    if z < -0.2 and vz < 0.:
                        angular_free = 0.87*(-vz+k*(-z-0.2))
                        assert count >= 1 and np.all(dist < 0.)
                        assert lower >= angular_free-2e-4
                        assert vz+float(delta.vel[6, 2]) >= .13*vz+.87*k*(-z-.2)-2e-4
                        assert vz+float(delta.vel[6, 2]) >= -8.32-2e-4
                        minimum_recovery_slack = min(minimum_recovery_slack, lower-angular_free)
                        deep_cases += 1
                    margin = z+dt*vz+dt*lower+0.5
                    assert margin >= -2e-6
                    assert z+dt*(vz+float(delta.vel[6, 2])) >= -0.5-2e-6
                    minimum_height_margin = min(minimum_height_margin, margin)
                    cases += 1
    assert active_counts == {0, 1, 2} and scalar_signs == {-1, 1}
    return {"source_formula_cases": cases, "deep_downward_cases": deep_cases,
            "active_contact_counts": sorted(active_counts),
            "root_scalar_signs": sorted(scalar_signs),
            "maximum_pair_velocity_sum_error": maximum_pair_sum_error,
            "minimum_angular_free_recovery_slack": minimum_recovery_slack,
            "minimum_paired_height_margin": minimum_height_margin}


def test_foot_joint_speed_balance():
    env = StochasticModeEnv("HalfCheetah-v2", family="actuator_polarity", backend="spring")
    sys = env.base_sys
    dt = float(sys.opt.timestep)
    stiffness = float(sys.link.constraint_stiffness[6])
    damping = float(sys.link.constraint_vel_damping[6])
    assert dt == 1/320 and 0. <= stiffness <= 20000. and 0. <= damping <= 10.
    np.testing.assert_allclose(sys.link.joint.pos[6], 0., atol=1e-7)
    np.testing.assert_allclose(sys.link.joint.pos[5], 0., atol=1e-7)
    assert np.all(np.linalg.norm(np.asarray(sys.link.inertia.transform.pos)[[5, 6]], axis=1) <= .2)
    assert np.linalg.norm(np.asarray(sys.link.transform.pos[6])) <= 1/3
    assert float(sys.gravity[2]) >= -10.

    @jax.jit
    def audit(initial, commands):
        def step(state, index):
            action = commands[index//16]
            state = state.replace(i_inv=com.inv_inertia(sys, state.x))
            force = joints.resolve(sys, state, actuator.to_tau(sys, action, state.q, state.qd))
            mass_inv = 1/state.mass[6]
            weight = dt*mass_inv*damping
            lever = math.rotate(sys.link.transform.pos[6], state.x.rot[5])
            error = state.x.pos[6]-state.x.pos[5]-lever
            relative = state.xd.vel[6]-state.xd.vel[5]+jnp.cross(state.xd.ang[5], lever)
            bias = relative-(state.xd_i.vel[6]-state.xd_i.vel[5])
            angular_bias = (
                jnp.cross(state.xd_i.ang[6], math.rotate(-sys.link.inertia.transform.pos[6], state.x.rot[6]))
                -jnp.cross(state.xd_i.ang[5], math.rotate(-sys.link.inertia.transform.pos[5], state.x.rot[5]))
                +jnp.cross(state.xd_i.ang[5], lever))
            expected_force = -stiffness*error-damping*relative
            defect = dt*mass_inv*stiffness*error[2]+weight*bias[2]-dt*sys.gravity[2]
            predicted_pre = (1-weight)*state.xd_i.vel[6, 2]+weight*state.xd_i.vel[5, 2]-defect
            pre = state.xd_i.vel[6, 2]+dt*(sys.gravity[2]+force.vel[6, 2]*mass_inv)
            joint_slack = ((1-weight)*(state.xd_i.vel[6, 2]+64)
                           +weight*(state.xd_i.vel[5, 2]+64)-defect)
            source_certificate = ((state.xd_i.vel[6, 2] >= -50)
                                  & (state.xd_i.vel[5, 2] >= -50)
                                  & (error[2] <= .01) & (bias[2] <= 80))
            post = pipeline.step(sys, state, action)
            upstream_lever = math.rotate(sys.link.transform.pos[5], state.x.rot[4])
            upstream_error = state.x.pos[5]-state.x.pos[4]-upstream_lever
            upstream_relative = state.xd.vel[5]-state.xd.vel[4]+jnp.cross(state.xd.ang[4], upstream_lever)
            upstream_force = (-sys.link.constraint_stiffness[5]*upstream_error
                              -sys.link.constraint_vel_damping[5]*upstream_relative)
            acceleration = Motion(vel=sys.gravity+force.vel/state.mass[:, None],
                                  ang=jnp.einsum("ijk,ik->ij", state.i_inv, force.ang))
            delta = collisions.resolve(sys, state.replace(xd_i=state.xd_i+acceleration*dt))
            pair_mass = mass_inv+1/state.mass[5]
            gamma, chi = dt*pair_mass*damping, dt*dt*pair_mass*stiffness
            kick = delta.vel[6, 2]-dt*(pair_mass*damping*bias[2]+upstream_force[2]/state.mass[5])
            relative_speed = state.xd_i.vel[6, 2]-state.xd_i.vel[5, 2]
            post_relative_speed = post.xd_i.vel[6, 2]-post.xd_i.vel[5, 2]
            predicted_relative = (1-gamma)*relative_speed-dt*pair_mass*stiffness*error[2]+kick
            post_lever = math.rotate(sys.link.transform.pos[6], post.x.rot[5])
            post_error = post.x.pos[6]-post.x.pos[5]-post_lever
            geometry = error[2]-(state.x_i.pos[6, 2]-state.x_i.pos[5, 2])
            post_geometry = post_error[2]-(post.x_i.pos[6, 2]-post.x_i.pos[5, 2])
            rotation_remainder = post_geometry-geometry
            predicted_error = error[2]+dt*post_relative_speed+rotation_remainder
            finite = (jnp.all(jnp.isfinite(post.x_i.pos)) & jnp.all(jnp.isfinite(post.xd_i.vel))
                      & jnp.isfinite(pre) & jnp.isfinite(defect))
            deep = (state.x_i.pos[6, 2] < -.2) & (pre < 0.)
            values = jnp.array([
                pre, post.x_i.pos[6, 2], joint_slack,
                jnp.max(jnp.abs(expected_force-force.vel[6])), jnp.abs(predicted_pre-pre),
                jnp.max(jnp.abs(bias-angular_bias)), jnp.abs(joint_slack-(pre+64)),
                error[2], bias[2], weight, mass_inv, source_certificate,
                source_certificate & ((pre < -2001/32-2e-4) | (defect > 401/32+2e-4)),
                deep, jnp.where(deep, post.xd_i.vel[6, 2]+8.32, jnp.inf),
                (~finite) | (pre < -64.) | (post.x_i.pos[6, 2] < -.5),
                jnp.max(jnp.abs(upstream_force-expected_force-force.vel[5])),
                jnp.abs(predicted_relative-post_relative_speed), jnp.abs(predicted_error-post_error[2]),
                jnp.abs(rotation_remainder), jnp.maximum(jnp.abs(geometry), jnp.abs(post_geometry)),
                jnp.max(jnp.abs(delta.vel[5])), gamma, chi, jnp.abs(kick),
                error[2], post_error[2], rotation_remainder, kick])
            return post, values

        _, values = jax.lax.scan(step, initial, jnp.arange(commands.shape[0]*16))
        second_order_error = (values[1:, 26]-(2-values[1:, 22]-values[1:, 23])*values[1:, 25]
                              +(1-values[1:, 22])*values[:-1, 25]-dt*values[1:, 28]-values[1:, 27]
                              +(1-values[1:, 22])*values[:-1, 27])
        return (jnp.min(values, axis=0), jnp.max(values, axis=0), jnp.sum(values[:, 11:14], axis=0),
                jnp.max(jnp.abs(second_order_error)))

    rng = np.random.default_rng(90310)
    rows = []
    for case, name in enumerate(("zero", "positive", "negative", "random")):
        commands = (np.zeros((1000, 6)), np.ones((1000, 6)), -np.ones((1000, 6)),
                    rng.uniform(-1, 1, (1000, 6)))[case]
        q = jax.random.uniform(jax.random.PRNGKey(90310+case), (9,), minval=-.1, maxval=.1)
        qd = jax.random.uniform(jax.random.PRNGKey(91310+case), (9,), minval=-.1, maxval=.1)
        lo, hi, counts, second_order_error = map(
            np.asarray, audit(pipeline.init(sys, q, qd), jnp.asarray(commands, dtype=jnp.float32)))
        assert hi[15] == 0 and hi[12] == 0
        assert lo[9] >= 0. and hi[9] <= .125 and hi[10] <= 4.
        assert hi[3] < .02 and np.max(hi[4:7]) < 4e-4
        assert lo[14] >= -2e-4
        assert hi[16] < .02 and np.max(hi[17:19]) < 4e-4 and second_order_error < 4e-4
        assert hi[19] <= 22/15+2e-6 and hi[20] <= 11/15+2e-6 and hi[21] == 0.
        assert lo[22] > 0. and hi[22] <= .25 and lo[23] > 0. and hi[23] <= 25/16
        np.testing.assert_allclose(lo[22:24], hi[22:24], rtol=1e-6, atol=1e-7)
        root_modulus = float(np.max(np.abs(np.roots([1., -(2-hi[22]-hi[23]), 1-hi[22]]))))
        assert root_modulus < 1.
        rows.append({"command": name, "physics_substeps": 16000,
                     "minimum_pre_foot_velocity": float(lo[0]), "minimum_foot_height": float(lo[1]),
                     "minimum_joint_speed_slack": float(lo[2]),
                     "maximum_joint_force_formula_error": float(hi[3]),
                     "maximum_pre_velocity_formula_error": float(hi[4]),
                     "maximum_angular_bias_formula_error": float(hi[5]),
                     "maximum_joint_slack_identity_error": float(hi[6]),
                     "maximum_signed_joint_height_error": float(hi[7]),
                     "maximum_signed_velocity_bias": float(hi[8]),
                     "source_certificate_substeps": int(counts[0]), "deep_downward_substeps": int(counts[2]),
                     "maximum_parent_force_formula_error": float(hi[16]),
                     "maximum_relative_speed_recurrence_error": float(hi[17]),
                     "maximum_anchor_recurrence_error": float(hi[18]),
                     "maximum_second_order_recurrence_error": float(second_order_error),
                     "maximum_rotation_remainder": float(hi[19]), "maximum_anchor_geometry": float(hi[20]),
                     "maximum_pair_kick": float(hi[24]), "pair_damping": float(hi[22]),
                     "pair_spring_strength": float(hi[23]), "homogeneous_root_modulus": root_modulus})
    return {"source_substeps": 64000, "cases": rows}


if __name__ == "__main__":
    print(json.dumps({"normal_resets": test_contact_recovery_bound(),
                      "paired_foot_formulas": test_paired_foot_recovery_bound(),
                      "foot_joint_speed": test_foot_joint_speed_balance()}, indent=2))
