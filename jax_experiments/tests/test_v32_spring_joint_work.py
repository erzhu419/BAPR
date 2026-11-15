"""Check source joint work and the discrete kinetic-energy balance."""

import json

import jax
import numpy as np
from jax import numpy as jnp
from brax import actuator, com, contact, kinematics, math
from brax.base import Motion
from brax.spring import collisions, joints, pipeline

from jax_experiments.envs.stochastic_mode_env import StochasticModeEnv
from jax_experiments.tests.test_v32_spring_seven_link import _rotate


def _contact_energy(sys, contacts, pos, pre_v, pre_w, inverse, mass, alpha):
    """Retain the source's scalar denominator and true-matrix final inertia."""
    bodies = np.asarray(contacts.link_idx[1], dtype=np.int32)
    np.testing.assert_array_equal(contacts.link_idx[0], -np.ones(len(bodies), dtype=np.int32))
    normal = np.asarray(contacts.frame[:, 0], dtype=np.float64)
    np.testing.assert_allclose(normal, np.tile([0., 0., 1.], (len(bodies), 1)))
    lever = np.asarray(contacts.pos, dtype=np.float64)-pos[bodies]
    point_v = pre_v[bodies]+np.cross(pre_w[bodies], lever)
    normal_v = np.einsum("ij,ij->i", point_v, normal)
    eta = 1/mass[bodies]
    scalar = inverse.ravel()[bodies]
    denominator = eta+scalar*np.sum(np.cross(lever, normal)**2, axis=1)
    stabilization = float(sys.baumgarte_erp)/float(sys.opt.timestep)
    distance = np.asarray(contacts.dist, dtype=np.float64)
    elasticity = np.asarray(contacts.elasticity, dtype=np.float64)
    impulse = (-(1+elasticity)*normal_v-stabilization*distance)/denominator
    active = (distance < 0) & (normal_v < 0) & (impulse > 0)
    tangent = point_v-normal_v[:, None]*normal
    speed = np.linalg.norm(tangent, axis=1)
    speed[np.all(np.isclose(tangent, 0., atol=1e-8, rtol=1e-5), axis=1)] = 0.
    direction = tangent/(1e-6+speed[:, None])
    tangent_denominator = eta+scalar*np.sum(np.cross(lever, direction)**2, axis=1)
    drag = np.minimum(speed/tangent_denominator,
                      np.asarray(contacts.friction[:, 0], dtype=np.float64)*impulse)
    forces = impulse[:, None]*normal*active[:, None]
    forces -= drag[:, None]*direction*(active & (speed > 1e-3))[:, None]
    torques = np.cross(lever, forces)
    dv, dw = np.zeros_like(pos), np.zeros_like(pos)
    individual_energy, defect = 0., 0.
    multiple_contacts = 0
    for i in range(7):
        ids = (bodies == i) & active
        count = np.count_nonzero(ids)
        scale = 1/(count+1e-8)
        f, torque = forces[ids], torques[ids, 1]
        average_f, average_torque = scale*np.sum(f, axis=0), scale*np.sum(torque)
        lam, inv_mass = inverse[i, 1, 1], 1/mass[i]
        dv[i] = inv_mass*average_f
        dw[i, 1] = lam*average_torque
        individual_energy += scale*np.sum(f@pre_v[i]+alpha*pre_w[i, 1]*torque+
                                           inv_mass/2*np.sum(f*f, axis=1)+lam/2*torque**2)
        defect += inv_mass/2*(scale*np.sum(f*f)-np.dot(average_f, average_f))
        defect += lam/2*(scale*np.sum(torque**2)-average_torque**2)
        multiple_contacts += count > 1
    return dv, dw, individual_energy, defect, multiple_contacts


def test_joint_work():
    env = StochasticModeEnv("HalfCheetah-v2", family="actuator_polarity", backend="spring")
    sys = env.base_sys
    dt = float(sys.opt.timestep)
    parents = np.array([-1, 0, 1, 2, 0, 4, 5])
    np.testing.assert_array_equal(sys.link_parents, parents)
    np.testing.assert_array_equal(sys.link.joint.pos, np.zeros((7, 3)))
    assert float(sys.vel_damping) == 0.
    alpha = np.exp(float(sys.ang_damping)*dt)
    stiffness = np.asarray(sys.link.constraint_stiffness, dtype=np.float64)
    damping = np.asarray(sys.link.constraint_vel_damping, dtype=np.float64)
    angular_damping = np.asarray(sys.link.constraint_ang_damping, dtype=np.float64)
    limit_stiffness = np.asarray(sys.link.constraint_limit_stiffness, dtype=np.float64)
    com_offsets = np.asarray(sys.link.inertia.transform.pos, dtype=np.float64)
    child_local = -com_offsets[1:]
    parent_local = -com_offsets[parents[1:]]+np.asarray(sys.link.transform.pos[1:], dtype=np.float64)
    assert np.max(np.linalg.norm(child_local, axis=1)) <= .2
    assert np.max(np.linalg.norm(parent_local, axis=1)) <= .7

    @jax.jit
    def advance(state, action):
        inverse = com.inv_inertia(sys, state.x)
        state = state.replace(i_inv=inverse)
        tau = actuator.to_tau(sys, action, state.q, state.qd)
        local = [joints._three_dof(sys.link.take(0), state.j.take(0), state.jd.take(0),
                                  sys.dof.take(jnp.arange(3)), jnp.zeros(3))]
        for i in range(1, 7):
            local.append(joints._one_dof(sys.link.take(i), state.j.take(i), state.jd.take(i),
                                        sys.dof.take(jnp.array([i+2])), tau[i+2]))
        world_force = jax.vmap(math.rotate)(jnp.stack([f.vel for f in local]), state.a_p.rot)
        world_torque = jax.vmap(math.rotate)(jnp.stack([f.ang for f in local]), state.a_p.rot)
        cache_velocity = jax.vmap(math.rotate)(state.jd.vel, state.a_p.rot)
        force = joints.resolve(sys, state, tau)
        acceleration = Motion(vel=sys.gravity+force.vel/state.mass[:, None],
                              ang=jnp.einsum("ijk,ik->ij", inverse, force.ang))
        pre = state.xd_i+acceleration*dt
        delta = collisions.resolve(sys, state.replace(xd_i=pre))
        limit_error = []
        for i in range(1, 7):
            dof = sys.dof.take(jnp.array([i+2]))
            frame, parity = kinematics.link_to_joint_frame(dof.motion)
            _, (angle, _, _), _ = kinematics.axis_angle_ang(state.j.take(i), frame, parity)
            lo, hi = dof.limit
            limit_error.append(jnp.where(angle > hi[0], angle-hi[0],
                                        jnp.where(angle < lo[0], angle-lo[0], 0)))
        return (pipeline.step(sys, state, action), force, world_force, world_torque,
                cache_velocity, acceleration, pre, delta, inverse, tau, jnp.stack(limit_error),
                contact.get(sys, state.x))

    def array(x):
        return np.asarray(x, dtype=np.float64)

    def parent(x):
        return np.concatenate([x, np.zeros((1, 3))])[parents]

    maxima = np.zeros(10)
    bound_errors = np.zeros(5)
    minimum_contact_defect = np.inf
    minimum_elastic_upper_slack = np.inf
    multiple_contact_cases = 0
    minimum_linear_work = np.inf
    maximum_linear_work = -np.inf
    maximum_contact_work = -np.inf
    positive_linear_cases = 0
    positive_contact_cases = 0
    minimum_current_energy_slack = np.inf
    maximum_current_prefix_slack = 0.
    minimum_kinetic_speed_slack = np.inf
    kinetic_speed_cases = 0
    minimum_prefix_kinetic_speed_slack = np.inf
    prefix_kinetic_speed_cases = 0
    steps = 0
    rng = np.random.default_rng(90310)
    for case in range(4):
        state = env._reset_fn(sys, jax.random.PRNGKey(90310+case)).pipeline_state
        for substep in range(64):
            action = (jnp.zeros(6), jnp.ones(6), -jnp.ones(6),
                      jnp.asarray(rng.uniform(-1, 1, 6), dtype=jnp.float32))[case]
            post, force, fw, tw, cached, acceleration, pre, delta, inverse, tau, limit_error, contacts = advance(state, action)
            pos, vel, ang = array(state.x_i.pos), array(state.xd_i.vel), array(state.xd_i.ang)
            child_lever = array(state.a_c.pos)-pos
            parent_lever = array(state.a_p.pos)-parent(pos)
            anchor_velocity = vel+np.cross(ang, child_lever)-parent(vel)-np.cross(parent(ang), parent_lever)
            offset = array(state.a_p.pos)-parent(array(state.x.pos))
            shift = np.cross(parent(ang), offset)
            shift_error = np.max(np.abs(array(cached)-anchor_velocity-2*shift))
            force_work = np.sum(array(force.vel)*vel)+np.sum(array(force.ang)*ang)
            anchor_work = np.sum(array(fw)*anchor_velocity)+np.sum(array(tw)*(ang-parent(ang)))
            work_error = abs(force_work-anchor_work)
            error = array(state.a_c.pos)-array(state.a_p.pos)
            potential_rate = np.sum(stiffness[1:, None]*error[1:]*anchor_velocity[1:])
            linear_work = np.sum(array(fw)[1:]*anchor_velocity[1:])+potential_rate
            completed = np.sum(damping[1:, None]*(shift[1:]**2-(anchor_velocity[1:]+shift[1:])**2))
            linear_error = abs(linear_work-completed)
            shift_budget = np.sum(damping[1:, None]*shift[1:]**2)
            np.testing.assert_allclose(array(cached), anchor_velocity+2*shift, rtol=3e-5, atol=2e-5)
            np.testing.assert_allclose(force_work, anchor_work, rtol=3e-5, atol=.05)
            np.testing.assert_allclose(linear_work, completed, rtol=3e-5, atol=.05)
            relative_pitch = array(state.jd.ang)[1:, 1]
            angular_work = np.sum(array(tw)[1:]*(ang-parent(ang))[1:])
            angular_formula = np.sum(array(tau)[3:]*relative_pitch-
                                     angular_damping[1:]*relative_pitch**2-
                                     limit_stiffness[1:]*array(limit_error)*relative_pitch)
            angular_error = abs(angular_work-angular_formula)
            np.testing.assert_allclose(angular_work, angular_formula, rtol=3e-5, atol=.05)
            assert linear_work <= shift_budget+.05
            lever_bound = np.sum(damping[1:]*.25*ang[parents[1:], 1]**2)
            assert shift_budget <= lever_bound+2e-5

            mass = array(state.mass)
            pitch_inertia = 1/array(inverse)[:, 1, 1]
            assert np.min(pitch_inertia) > 0.

            def kinetic(v, w):
                return .5*np.sum(mass[:, None]*v*v)+.5*np.dot(pitch_inertia, w[:, 1]**2)

            if substep == 0:
                current_energy_prefix = kinetic(vel, ang)+.5*np.sum(stiffness[1:, None]*error[1:]**2)
            pre_v, pre_w = array(pre.vel), array(pre.ang)
            acc_v, acc_w = array(acceleration.vel), array(acceleration.ang)
            euler_correction = .5*dt*dt*(np.sum(mass[:, None]*acc_v*acc_v)+
                                       np.dot(pitch_inertia, acc_w[:, 1]**2))
            euler = dt*(force_work+np.sum(mass[:, None]*vel*array(sys.gravity)))+euler_correction
            euler_error = abs(kinetic(pre_v, pre_w)-kinetic(vel, ang)-euler)
            damped_w = alpha*pre_w
            damping_work = .5*(alpha*alpha-1)*np.dot(pitch_inertia, pre_w[:, 1]**2)
            dv, dw = array(delta.vel), array(delta.ang)
            contact_work = np.sum(mass[:, None]*(pre_v*dv+.5*dv*dv))
            contact_work += np.dot(pitch_inertia, damped_w[:, 1]*dw[:, 1]+.5*dw[:, 1]**2)
            formula_v, formula_w, contact_upper, contact_defect, multiple = _contact_energy(
                sys, contacts, pos, pre_v, pre_w, array(inverse), mass, alpha)
            np.testing.assert_allclose(dv, formula_v, rtol=3e-5, atol=2e-5)
            np.testing.assert_allclose(dw, formula_w, rtol=3e-5, atol=2e-5)
            np.testing.assert_allclose(contact_work, contact_upper-contact_defect, rtol=3e-5, atol=2e-5)
            assert contact_defect >= -2e-10
            assert contact_work <= contact_upper+2e-5
            minimum_contact_defect = min(minimum_contact_defect, contact_defect)
            multiple_contact_cases += multiple
            post_v, post_w = array(post.xd_i.vel), array(post.xd_i.ang)
            balance = kinetic(post_v, post_w)-kinetic(vel, ang)
            np.testing.assert_allclose(kinetic(pre_v, pre_w)-kinetic(vel, ang), euler, rtol=3e-5, atol=.05)
            np.testing.assert_allclose(balance, euler+damping_work+contact_work, rtol=3e-5, atol=.05)
            balance_error = abs(balance-euler-damping_work-contact_work)
            next_anchor_error = array(post.a_c.pos)-array(post.a_p.pos)
            anchor_increment = next_anchor_error[1:]-error[1:]
            anchor_remainder = anchor_increment-dt*anchor_velocity[1:]
            elastic_change = .5*np.sum(stiffness[1:, None]*
                                      (next_anchor_error[1:]**2-error[1:]**2))
            elastic_remainder = np.sum(stiffness[1:, None]*(error[1:]*anchor_remainder+
                                                           .5*anchor_increment**2))
            elastic_error = abs(elastic_change-dt*potential_rate-elastic_remainder)
            np.testing.assert_allclose(elastic_change, dt*potential_rate+elastic_remainder,
                                       rtol=3e-5, atol=2e-5)
            old_rot, next_rot = array(state.x_i.rot), array(post.x_i.rot)

            def lever_remainder(local, ids):
                before = np.array([_rotate(v, old_rot[i]) for v, i in zip(local, ids)])
                after = np.array([_rotate(v, next_rot[i]) for v, i in zip(local, ids)])
                remainder = after-before-dt*np.cross(post_w[ids], before)
                half_turn = dt*post_w[ids, 1]/2
                left = (1+half_turn**2)*np.sum(remainder**2, axis=1)
                right = 4*half_turn**4*np.sum(local**2, axis=1)
                np.testing.assert_allclose(left, right, rtol=1e-3, atol=2e-10)
                radius = dt**2/2*post_w[ids, 1]**2*np.linalg.norm(local, axis=1)
                assert np.all(np.linalg.norm(remainder, axis=1) <= radius+1e-6)
                return before, remainder, np.max(np.abs(left-right))

            lc, rc, error_c = lever_remainder(child_local, np.arange(1, 7))
            lp, rp, error_p = lever_remainder(parent_local, parents[1:])
            np.testing.assert_allclose(child_lever[1:], lc, rtol=3e-5, atol=1e-6)
            np.testing.assert_allclose(parent_lever[1:], lp, rtol=3e-5, atol=1e-6)
            kick = (post_v-vel)[1:]-(post_v-vel)[parents[1:]]
            kick += np.cross((post_w-ang)[1:], lc)-np.cross((post_w-ang)[parents[1:]], lp)
            anchor_formula = dt*kick+rc-rp
            np.testing.assert_allclose(anchor_remainder, anchor_formula, rtol=3e-5, atol=1e-6)
            taylor_budget = dt*np.abs(kick)+dt**2/2*(.2*post_w[1:, 1]**2+
                                                   .7*post_w[parents[1:], 1]**2)[:, None]
            assert np.all(np.abs(anchor_remainder) <= taylor_budget+1e-6)
            elastic_upper = np.sum(stiffness[1:, None]*(np.abs(error[1:])*taylor_budget+
                                                       .5*anchor_increment**2))
            elastic_slack = elastic_upper-elastic_remainder
            assert elastic_slack >= -2e-5
            minimum_elastic_upper_slack = min(minimum_elastic_upper_slack, elastic_slack)
            bound_errors = np.maximum(bound_errors, [np.max(np.abs(dv-formula_v)), np.max(np.abs(dw-formula_w)),
                abs(contact_work-contact_upper+contact_defect), max(error_c, error_p),
                np.max(np.abs(anchor_remainder-anchor_formula))])
            root_work = np.sum(array(fw)[0]*anchor_velocity[0])+np.dot(array(tw)[0], ang[0])
            mechanical_rhs = dt*(root_work+completed+angular_formula+
                                  np.sum(mass[:, None]*vel*array(sys.gravity)))
            mechanical_rhs += euler_correction+damping_work+contact_work+elastic_remainder
            mechanical_error = abs(balance+elastic_change-mechanical_rhs)
            np.testing.assert_allclose(balance+elastic_change, mechanical_rhs, rtol=3e-5, atol=.05)
            mechanical_upper = mechanical_rhs-contact_work-elastic_remainder+contact_upper+elastic_upper
            assert balance+elastic_change <= mechanical_upper+.05
            linear_kick_budget = dt*np.abs(acc_v)+np.abs(dv)
            angular_kick_budget = abs(alpha-1)*np.abs(ang[:, 1])+alpha*dt*np.abs(acc_w[:, 1])+np.abs(dw[:, 1])
            pitch_budget = alpha*(np.abs(ang[:, 1])+dt*np.abs(acc_w[:, 1]))+np.abs(dw[:, 1])
            assert np.all(np.abs(post_v-vel) <= linear_kick_budget+2e-5)
            assert np.all(np.abs(post_w[:, 1]-ang[:, 1]) <= angular_kick_budget+2e-5)
            assert np.all(np.abs(post_w[:, 1]) <= pitch_budget+2e-5)
            current_kick = linear_kick_budget[1:]+linear_kick_budget[parents[1:]]
            current_kick += (.2*angular_kick_budget[1:]+.7*angular_kick_budget[parents[1:]])[:, None]
            assert np.all(np.abs(kick) <= current_kick+2e-5)
            current_taylor = dt*current_kick+dt**2/2*(.2*pitch_budget[1:]**2+
                                                    .7*pitch_budget[parents[1:]]**2)[:, None]
            current_increment = dt*np.abs(anchor_velocity[1:])+current_taylor
            assert np.all(taylor_budget <= current_taylor+1e-6)
            assert np.all(np.abs(anchor_increment) <= current_increment+1e-6)
            current_elastic = np.sum(stiffness[1:, None]*(np.abs(error[1:])*current_taylor+
                                                         .5*current_increment**2))
            assert elastic_upper <= current_elastic+2e-5
            current_gain = mechanical_rhs-contact_work-elastic_remainder+contact_upper+current_elastic
            current_slack = current_gain-balance-elastic_change
            assert current_slack >= -.05
            prefix_pre_kinetic = current_energy_prefix-.5*np.sum(stiffness[1:, None]*error[1:]**2)+euler
            assert kinetic(pre_v, pre_w) <= prefix_pre_kinetic+.05
            prefix_kinetic_slack = 4096-2/mass[6]*prefix_pre_kinetic
            minimum_prefix_kinetic_speed_slack = min(minimum_prefix_kinetic_speed_slack, prefix_kinetic_slack)
            prefix_kinetic_speed_cases += prefix_kinetic_slack >= 0
            current_energy_prefix += current_gain
            actual_energy = kinetic(post_v, post_w)+.5*np.sum(stiffness[1:, None]*next_anchor_error[1:]**2)
            assert actual_energy <= current_energy_prefix+.05
            minimum_current_energy_slack = min(minimum_current_energy_slack, current_slack)
            maximum_current_prefix_slack = max(maximum_current_prefix_slack, current_energy_prefix-actual_energy)
            kinetic_budget = 2/mass[6]*kinetic(pre_v, pre_w)
            assert pre_v[6, 2]**2 <= kinetic_budget+2e-5
            kinetic_slack = 4096-kinetic_budget
            minimum_kinetic_speed_slack = min(minimum_kinetic_speed_slack, kinetic_slack)
            kinetic_speed_cases += kinetic_slack >= 0
            next_inverse = array(com.inv_inertia(sys, post.x))[:, 1, 1]
            inertia_error = np.max(np.abs(next_inverse-array(inverse)[:, 1, 1]))
            np.testing.assert_allclose(next_inverse, array(inverse)[:, 1, 1], rtol=3e-6, atol=2e-6)
            maxima = np.maximum(maxima, [shift_error, work_error, linear_error,
                                         euler_error, balance_error, inertia_error, shift_budget,
                                         angular_error, elastic_error, mechanical_error])
            minimum_linear_work = min(minimum_linear_work, linear_work)
            maximum_linear_work = max(maximum_linear_work, linear_work)
            maximum_contact_work = max(maximum_contact_work, contact_work)
            positive_linear_cases += linear_work > .05
            positive_contact_cases += contact_work > .05
            state = post
            steps += 1

    @jax.jit
    def contact_only(state):
        return collisions.resolve(sys, state), contact.get(sys, state.x)

    base = env._reset_fn(sys, jax.random.PRNGKey(90320)).pipeline_state
    two_contact_cases = 0
    for pitch_speed in (-15., 0., 15.):
        frame = base.x_i.replace(pos=base.x_i.pos.at[6, 2].set(-.3))
        motion = Motion.zero((7,)).replace(vel=jnp.zeros((7, 3)).at[6].set(jnp.array([.8, 0., -2.])),
                                          ang=jnp.zeros((7, 3)).at[6, 1].set(pitch_speed))
        world_frame, world_motion = com.to_world(sys, frame, motion)
        inverse = com.inv_inertia(sys, world_frame)
        source = base.replace(x_i=frame, xd_i=motion, x=world_frame, xd=world_motion, i_inv=inverse)
        delta, contacts = contact_only(source)
        mass, pos = array(source.mass), array(frame.pos)
        pre_v, pre_w = array(motion.vel), array(motion.ang)
        fv, fw, upper, defect, multiple = _contact_energy(sys, contacts, pos, pre_v, pre_w,
                                                        array(inverse), mass, alpha)
        dv, dw = array(delta.vel), array(delta.ang)
        np.testing.assert_allclose(dv, fv, rtol=3e-5, atol=2e-5)
        np.testing.assert_allclose(dw, fw, rtol=3e-5, atol=2e-5)
        actual = np.sum(mass[:, None]*(pre_v*dv+.5*dv*dv))
        actual += np.dot(1/array(inverse)[:, 1, 1], alpha*pre_w[:, 1]*dw[:, 1]+.5*dw[:, 1]**2)
        np.testing.assert_allclose(actual, upper-defect, rtol=3e-5, atol=2e-5)
        assert multiple == 1 and defect > 0.
        assert actual <= upper+2e-5
        two_contact_cases += 1
        bound_errors[:3] = np.maximum(bound_errors[:3], [np.max(np.abs(dv-fv)),
            np.max(np.abs(dw-fw)), abs(actual-upper+defect)])
    return {"normal_reset_cases": 4, "source_substeps": steps,
            "maximum_cache_shift_error": float(maxima[0]), "maximum_wrench_power_error": float(maxima[1]),
            "maximum_linear_work_identity_error": float(maxima[2]), "maximum_euler_energy_error": float(maxima[3]),
            "maximum_full_kinetic_balance_error": float(maxima[4]), "maximum_pitch_inverse_inertia_change": float(maxima[5]),
            "maximum_frame_shift_work_budget": float(maxima[6]), "minimum_linear_work_plus_potential_rate": float(minimum_linear_work),
            "maximum_angular_work_error": float(maxima[7]), "maximum_elastic_increment_error": float(maxima[8]),
            "maximum_mechanical_balance_error": float(maxima[9]),
            "maximum_contact_linear_increment_error": float(bound_errors[0]),
            "maximum_contact_angular_increment_error": float(bound_errors[1]),
            "maximum_contact_average_energy_error": float(bound_errors[2]),
            "maximum_rotation_remainder_square_error": float(bound_errors[3]),
            "maximum_anchor_taylor_identity_error": float(bound_errors[4]),
            "minimum_contact_average_defect": float(minimum_contact_defect),
            "multiple_contact_body_substeps": int(multiple_contact_cases),
            "constructed_two_contact_cases": two_contact_cases,
            "minimum_elastic_upper_slack": float(minimum_elastic_upper_slack),
            "minimum_current_energy_step_slack": float(minimum_current_energy_slack),
            "maximum_current_energy_prefix_slack": float(maximum_current_prefix_slack),
            "minimum_kinetic_speed_budget_slack": float(minimum_kinetic_speed_slack),
            "kinetic_speed_certificate_substeps": int(kinetic_speed_cases),
            "minimum_prefix_kinetic_speed_budget_slack": float(minimum_prefix_kinetic_speed_slack),
            "prefix_kinetic_speed_certificate_substeps": int(prefix_kinetic_speed_cases),
            "maximum_linear_work_plus_potential_rate": float(maximum_linear_work), "positive_linear_work_cases": int(positive_linear_cases),
            "maximum_contact_kinetic_work": float(maximum_contact_work), "positive_contact_work_cases": int(positive_contact_cases)}


if __name__ == "__main__":
    print(json.dumps(test_joint_work(), indent=2))
