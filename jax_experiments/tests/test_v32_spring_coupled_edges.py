"""Compare the complete six-edge translation model with the Brax spring update."""

import json

import jax
import numpy as np
from jax import numpy as jnp
from brax import actuator, com, math
from brax.base import Motion
from brax.spring import collisions, joints, pipeline

from jax_experiments.envs.stochastic_mode_env import StochasticModeEnv


def _modal_energy(c, gamma, error, scaled_speed):
    return (c*error**2+(gamma-c)*error*scaled_speed
            +(1-gamma)*scaled_speed**2)


def test_coupled_edge_translation():
    env = StochasticModeEnv("HalfCheetah-v2", family="actuator_polarity", backend="spring")
    sys = env.base_sys
    dt = float(sys.opt.timestep)
    stiffness = float(sys.link.constraint_stiffness[1])
    damping = float(sys.link.constraint_vel_damping[1])
    assert dt == 1/320 and float(sys.vel_damping) == 0.
    assert 0. < stiffness <= 20000. and 0. < damping <= 10.
    assert stiffness == 20000. and damping == 10.
    np.testing.assert_array_equal(sys.link.constraint_stiffness[1:], np.full(6, stiffness))
    np.testing.assert_array_equal(sys.link.constraint_vel_damping[1:], np.full(6, damping))
    np.testing.assert_array_equal(sys.link.joint.pos, np.zeros((7, 3)))
    for rotation in (sys.link.transform.rot[0], sys.link.joint.rot[0]):
        np.testing.assert_allclose(rotation, [1., 0., 0., 0.], atol=1e-7)
    assert np.max(np.linalg.norm(np.asarray(sys.link.inertia.transform.pos), axis=1)) <= .2
    assert np.max(np.linalg.norm(np.asarray(sys.link.transform.pos)[1:], axis=1)) <= .5
    parents = np.array([0, 1, 2, 0, 4, 5])
    incidence = np.zeros((6, 7))
    incidence[np.arange(6), np.arange(1, 7)] = 1.
    incidence[np.arange(6), parents] = -1.
    degree = np.array([2, 2, 2, 1, 2, 2, 1])

    @jax.jit
    def advance(state, action):
        state = state.replace(i_inv=com.inv_inertia(sys, state.x))
        force = joints.resolve(sys, state, actuator.to_tau(sys, action, state.q, state.qd))
        acceleration = Motion(vel=sys.gravity+force.vel/state.mass[:, None],
                              ang=jnp.einsum("ijk,ik->ij", state.i_inv, force.ang))
        delta = collisions.resolve(sys, state.replace(xd_i=state.xd_i+acceleration*dt))
        parent = jnp.asarray(parents)
        lever = jax.vmap(math.rotate)(sys.link.transform.pos[1:], state.x.rot[parent])
        error = state.x.pos[1:]-state.x.pos[parent]-lever
        relative = state.xd.vel[1:]-state.xd.vel[parent]+jnp.cross(state.xd.ang[parent], lever)
        edge_force = -stiffness*error-damping*relative
        com_lever = jax.vmap(math.rotate)(-sys.link.inertia.transform.pos, state.x_i.rot)
        angular_bias = (jnp.cross(state.xd_i.ang[1:], com_lever[1:])
                        -jnp.cross(state.xd_i.ang[parent], com_lever[parent])
                        +jnp.cross(state.xd_i.ang[parent], lever))
        post = pipeline.step(sys, state, action)
        post_lever = jax.vmap(math.rotate)(sys.link.transform.pos[1:], post.x.rot[parent])
        post_error = post.x.pos[1:]-post.x.pos[parent]-post_lever
        post_com_lever = jax.vmap(math.rotate)(-sys.link.inertia.transform.pos, post.x_i.rot)
        return (post, force, acceleration, delta, error, relative, edge_force,
                angular_bias, post_error, com_lever, lever, post_com_lever, post_lever)

    rng = np.random.default_rng(90310)
    maxima = np.zeros(14)
    minimum_bias_slack = np.inf
    minimum_contact_slack = np.inf
    minimum_rotation_slack = np.inf
    minimum_input_slack = np.inf
    minimum_contact_response = np.inf
    maximum_foot_response_error = 0.
    minimum_signed_foot_lower = np.inf
    minimum_support_foot_lower = np.inf
    minimum_exact_angular_lower = np.inf
    minimum_pre_foot_speed = np.inf
    signed_floor_cases = 0
    support_floor_cases = 0
    exact_angular_floor_cases = 0
    maximum_modal_source_error = 0.
    maximum_modal_homogeneous_error = 0.
    minimum_modal_step_slack = np.inf
    minimum_modal_prefix_slack = np.inf
    maximum_modal_energy = 0.
    maximum_modal_input_energy = 0.
    maximum_modal_envelope = 0.
    minimum_modal_coercivity_slack = np.inf
    minimum_modal_upper_slack = np.inf
    minimum_current_input_slack = np.inf
    maximum_modal_foot_identity_error = 0.
    maximum_modal_center_history_error = 0.
    minimum_modal_output_slack = np.inf
    minimum_modal_norm_step_slack = np.inf
    minimum_modal_norm_prefix_slack = np.inf
    minimum_energy_foot_lower = np.inf
    minimum_norm_foot_lower = np.inf
    energy_foot_floor_cases = 0
    norm_foot_floor_cases = 0
    previous = None
    steps = 0
    eigenvalues = None
    root_modulus = None
    for case in range(4):
        state = env._reset_fn(sys, jax.random.PRNGKey(90310+case)).pipeline_state
        eta = 1/np.asarray(state.mass, dtype=np.float64)
        assert np.all(eta > 0.) and np.all(degree*eta <= 4.)
        operator = (incidence*eta)@incidence.T
        damped = np.eye(6)-dt*damping*operator
        spring = dt*dt*stiffness*operator
        transient = np.block([[np.eye(6)-spring, damped], [-spring, damped]])
        pre_speed_map = np.concatenate([-dt*stiffness*operator, damped/dt], axis=1)
        mass = 1/eta
        beta = np.array([-sum(mass[1:4]), -sum(mass[2:4]), -mass[3],
                         sum(mass)-sum(mass[4:]), sum(mass)-sum(mass[5:]), sum(mass)-mass[6]])
        contact_map = dt*np.concatenate([incidence, incidence])
        geometry_map = np.concatenate([np.eye(6), np.zeros((6, 6))])
        bias_map = -dt*dt*damping*np.concatenate([operator, operator])
        foot_row = beta@pre_speed_map/sum(mass)
        current_bias_row = -dt*damping*(beta@operator)/sum(mass)
        powers = [np.eye(12)]
        for _ in range(64):
            contact_response = (mass+beta@pre_speed_map@powers[-1]@contact_map)/sum(mass)
            minimum_contact_response = min(minimum_contact_response, np.min(contact_response))
            powers.append(transient@powers[-1])
        eigenvalues, eigenvectors = np.linalg.eigh(operator)
        assert np.min(eigenvalues) > 0. and np.max(eigenvalues) <= 8.
        modal_c = dt*dt*stiffness*eigenvalues
        modal_gamma = dt*damping*eigenvalues
        np.testing.assert_allclose(modal_gamma, 4*modal_c/25, atol=1e-16)
        modal_contraction = 1-modal_gamma/2
        modal_gain = 1+2/modal_gamma
        modal_envelope = None
        modal_norm_contraction = np.sqrt(1-modal_gamma)
        modal_norm_envelope = None
        modal_output_factor = (1-modal_gamma)/(1-841*modal_c/2500)
        modal_foot_weights = eigenvectors.T@beta/sum(mass)
        root_modulus = max(float(np.max(np.abs(np.roots([
            1., -(2-dt*lam*damping-dt*dt*lam*stiffness), 1-dt*lam*damping]))))
            for lam in eigenvalues)
        assert root_modulus < 1.
        previous = None
        inputs = []
        allowances = []
        angular_inputs = []
        contact_inputs = []
        geometry_bounds = []
        bias_bounds = []
        contact_supports = []
        initial_transient = None
        initial_mean_speed = np.dot(mass, np.asarray(state.xd_i.vel[:, 2], dtype=np.float64))/sum(mass)
        for _ in range(64):
            action = (jnp.zeros(6), jnp.ones(6), -jnp.ones(6),
                      jnp.asarray(rng.uniform(-1, 1, 6), dtype=jnp.float32))[case]
            (post, force, acceleration, delta, anchor_error, relative,
             edge_force, angular_bias, post_anchor_error, com_lever, link_lever,
             post_com_lever, post_link_lever) = advance(state, action)
            np.testing.assert_array_equal(post.mass, state.mass)
            np.testing.assert_allclose(np.sum(np.asarray(state.x_i.rot)**2, axis=1), 1., atol=2e-6)
            y = incidence@np.asarray(state.x_i.pos[:, 2], dtype=np.float64)
            speed = incidence@np.asarray(state.xd_i.vel[:, 2], dtype=np.float64)
            post_y = incidence@np.asarray(post.x_i.pos[:, 2], dtype=np.float64)
            post_speed = incidence@np.asarray(post.xd_i.vel[:, 2], dtype=np.float64)
            anchor_error = np.asarray(anchor_error[:, 2], dtype=np.float64)
            relative = np.asarray(relative[:, 2], dtype=np.float64)
            edge_force = np.asarray(edge_force[:, 2], dtype=np.float64)
            geometry, bias = anchor_error-y, relative-speed
            post_anchor_error = np.asarray(post_anchor_error[:, 2], dtype=np.float64)
            post_geometry = post_anchor_error-post_y
            angular = np.asarray(state.xd_i.ang, dtype=np.float64)
            post_angular = np.asarray(post.xd_i.ang, dtype=np.float64)
            np.testing.assert_allclose(angular[:, [0, 2]], 0., atol=2e-6)
            bias_error = np.max(np.abs(bias-np.asarray(angular_bias[:, 2], dtype=np.float64)))
            bias_slack = .9*np.max(np.abs(angular[:, 1]))-np.max(np.abs(bias))
            contact = np.asarray(delta.vel[:, 2], dtype=np.float64)
            assert np.min(contact) >= -2e-6
            edge_contact = incidence@contact
            support = np.dot(np.asarray(state.mass, dtype=np.float64), contact)
            contact_slack = 8*support-np.sum(np.abs(edge_contact))
            assert bias_error < 2e-5 and min(bias_slack, contact_slack) >= -2e-5
            minimum_bias_slack = min(minimum_bias_slack, bias_slack)
            minimum_contact_slack = min(minimum_contact_slack, contact_slack)
            rho = post_geometry-geometry
            rotation_bound = dt*(.2*np.abs(post_angular[1:, 1])+.7*np.abs(post_angular[parents, 1]))
            rotation_slack = np.min(rotation_bound-np.abs(rho))
            assert rotation_slack >= -2e-6
            minimum_rotation_slack = min(minimum_rotation_slack, rotation_slack)
            current = np.concatenate([anchor_error, dt*speed])
            expected = np.concatenate([post_anchor_error, dt*post_speed])
            if initial_transient is None:
                initial_transient = current
            pre_speed = incidence@(np.asarray(state.xd_i.vel[:, 2], dtype=np.float64)
                                   +dt*np.asarray(acceleration.vel[:, 2], dtype=np.float64))
            pre_foot = float(state.xd_i.vel[6, 2]+dt*acceleration.vel[6, 2])
            bias_bound = .2*np.abs(angular[1:, 1])+.7*np.abs(angular[parents, 1])
            foot_base = initial_mean_speed+(len(inputs)+1)*dt*float(sys.gravity[2])
            foot_base += foot_row@powers[len(inputs)]@initial_transient
            signed_lower = foot_base-np.dot(np.abs(current_bias_row), bias_bound)
            support_lower = signed_lower
            exact_angular_lower = foot_base+current_bias_row@bias
            foot_response = exact_angular_lower
            for index in range(len(inputs)):
                projected = foot_row@powers[len(inputs)-1-index]
                geometry_row = projected@geometry_map
                bias_row = projected@bias_map
                contact_row = mass/sum(mass)+projected@contact_map
                angular_cost = np.dot(np.abs(geometry_row), geometry_bounds[index])
                angular_cost += np.dot(np.abs(bias_row), bias_bounds[index])
                contact_lower = np.min(eta*contact_row)*contact_supports[index]
                signed_lower += -angular_cost+contact_row@contact_inputs[index]
                support_lower += -angular_cost+contact_lower
                exact_angular_lower += projected@angular_inputs[index]+contact_lower
                foot_response += projected@angular_inputs[index]+contact_row@contact_inputs[index]
            foot_response_error = abs(pre_foot-foot_response)
            assert foot_response_error < 5e-4, (case, len(inputs), pre_foot, foot_response, foot_response_error)
            assert max(signed_lower, support_lower, exact_angular_lower) <= pre_foot+5e-4
            assert support_lower <= signed_lower+2e-12
            maximum_foot_response_error = max(maximum_foot_response_error, foot_response_error)
            minimum_signed_foot_lower = min(minimum_signed_foot_lower, signed_lower)
            minimum_support_foot_lower = min(minimum_support_foot_lower, support_lower)
            minimum_exact_angular_lower = min(minimum_exact_angular_lower, exact_angular_lower)
            minimum_pre_foot_speed = min(minimum_pre_foot_speed, pre_foot)
            signed_floor_cases += signed_lower >= -64.
            support_floor_cases += support_lower >= -64.
            exact_angular_floor_cases += exact_angular_lower >= -64.
            pre_observation_error = np.max(np.abs(pre_speed-pre_speed_map@current+dt*damping*(operator@bias)))
            pre_bound = np.linalg.norm(pre_speed_map@powers[len(inputs)], 1)*np.sum(np.abs(initial_transient))
            for index, item in enumerate(allowances):
                pre_bound += np.linalg.norm(pre_speed_map@powers[len(inputs)-1-index], 1)*item
            pre_bound += (56/5)*dt*damping*np.sum(np.abs(angular[:, 1]))
            assert pre_observation_error < 2e-5 and np.sum(np.abs(pre_speed)) <= pre_bound+2e-5
            kick = -dt*dt*damping*(operator@bias)+dt*edge_contact
            forcing = np.concatenate([rho+kick, kick])
            transient_error = np.max(np.abs(expected-transient@current-forcing))
            modal_error, modal_speed = eigenvectors.T@anchor_error, eigenvectors.T@(dt*speed)
            modal_q, modal_r = eigenvectors.T@(rho+kick), eigenvectors.T@kick
            homogeneous_error = (1-modal_c)*modal_error+(1-modal_gamma)*modal_speed
            homogeneous_speed = -modal_c*modal_error+(1-modal_gamma)*modal_speed
            actual_error = eigenvectors.T@post_anchor_error
            actual_speed = eigenvectors.T@(dt*post_speed)
            modal_source_error = max(np.max(np.abs(actual_error-homogeneous_error-modal_q)),
                                     np.max(np.abs(actual_speed-homogeneous_speed-modal_r)))
            modal_energy = _modal_energy(modal_c, modal_gamma, modal_error, modal_speed)
            actual_energy = _modal_energy(modal_c, modal_gamma, actual_error, actual_speed)
            input_energy = _modal_energy(modal_c, modal_gamma, modal_q, modal_r)
            homogeneous_energy = _modal_energy(modal_c, modal_gamma,
                                               homogeneous_error, homogeneous_speed)
            homogeneous_identity_error = np.max(np.abs(
                homogeneous_energy-(1-modal_gamma)*modal_energy))
            diagonal_energy = modal_c*modal_error**2+modal_speed**2
            coercivity_slack = np.min(modal_energy-diagonal_energy/4)
            upper_slack = np.min(2*diagonal_energy-modal_energy)
            modal_step_slack = np.min(modal_contraction*modal_energy+modal_gain*input_energy
                                     -actual_energy)
            if modal_envelope is None:
                modal_envelope = modal_energy.copy()
                modal_norm_envelope = np.sqrt(modal_energy)
            modal_bias = eigenvectors.T@bias
            projected_pre_foot = (np.dot(mass, np.asarray(state.xd_i.vel[:, 2]))/sum(mass)
                                  +dt*float(sys.gravity[2])
                                  +np.dot(modal_foot_weights,
                                          homogeneous_speed/dt-modal_gamma*modal_bias))
            modal_foot_identity_error = abs(pre_foot-projected_pre_foot)
            modal_output_slack = np.min(modal_output_factor*modal_energy-homogeneous_speed**2)
            mean_pre = np.dot(mass, np.asarray(state.xd_i.vel[:, 2]))/sum(mass)+dt*float(sys.gravity[2])
            current_bias_cost = np.dot(modal_foot_weights, modal_gamma*modal_bias)
            history_mean_pre = (initial_mean_speed+(len(inputs)+1)*dt*float(sys.gravity[2])
                                +sum(contact_supports)/sum(mass))
            center_history_error = abs(mean_pre-history_mean_pre)
            assert center_history_error < 5e-4
            maximum_modal_center_history_error = max(maximum_modal_center_history_error,
                                                     center_history_error)
            energy_foot_lower = history_mean_pre-current_bias_cost-np.dot(
                np.abs(modal_foot_weights), np.sqrt(modal_output_factor*modal_envelope))/dt
            norm_foot_lower = history_mean_pre-current_bias_cost-np.dot(
                np.abs(modal_foot_weights)*np.sqrt(modal_output_factor), modal_norm_envelope)/dt
            assert modal_foot_identity_error < 5e-4 and modal_output_slack >= -1e-12, (
                case, len(inputs), modal_foot_identity_error, modal_output_slack,
                pre_foot, projected_pre_foot)
            assert max(energy_foot_lower, norm_foot_lower) <= pre_foot+5e-4
            maximum_modal_foot_identity_error = max(maximum_modal_foot_identity_error,
                                                   modal_foot_identity_error)
            minimum_modal_output_slack = min(minimum_modal_output_slack, modal_output_slack)
            minimum_energy_foot_lower = min(minimum_energy_foot_lower, energy_foot_lower)
            minimum_norm_foot_lower = min(minimum_norm_foot_lower, norm_foot_lower)
            energy_foot_floor_cases += energy_foot_lower >= -64.
            norm_foot_floor_cases += norm_foot_lower >= -64.
            modal_norm_step_slack = np.min(modal_norm_contraction*np.sqrt(modal_energy)
                                          +np.sqrt(input_energy)-np.sqrt(actual_energy))
            modal_norm_envelope = modal_norm_contraction*modal_norm_envelope+np.sqrt(input_energy)
            modal_norm_prefix_slack = np.min(modal_norm_envelope-np.sqrt(actual_energy))
            assert min(modal_norm_step_slack, modal_norm_prefix_slack) >= -2e-6
            minimum_modal_norm_step_slack = min(minimum_modal_norm_step_slack, modal_norm_step_slack)
            minimum_modal_norm_prefix_slack = min(minimum_modal_norm_prefix_slack,
                                                 modal_norm_prefix_slack)
            modal_envelope = modal_contraction*modal_envelope+modal_gain*input_energy
            modal_prefix_slack = np.min(modal_envelope-actual_energy)
            assert modal_source_error < 2e-6 and homogeneous_identity_error < 1e-12
            assert min(coercivity_slack, upper_slack, np.min(input_energy)) >= -1e-12
            assert min(modal_step_slack, modal_prefix_slack) >= -2e-6
            maximum_modal_source_error = max(maximum_modal_source_error, modal_source_error)
            maximum_modal_homogeneous_error = max(maximum_modal_homogeneous_error,
                                                  homogeneous_identity_error)
            minimum_modal_step_slack = min(minimum_modal_step_slack, modal_step_slack)
            minimum_modal_prefix_slack = min(minimum_modal_prefix_slack, modal_prefix_slack)
            minimum_modal_coercivity_slack = min(minimum_modal_coercivity_slack, coercivity_slack)
            minimum_modal_upper_slack = min(minimum_modal_upper_slack, upper_slack)
            maximum_modal_energy = max(maximum_modal_energy, np.max(actual_energy))
            maximum_modal_input_energy = max(maximum_modal_input_energy, np.max(input_energy))
            maximum_modal_envelope = max(maximum_modal_envelope, np.max(modal_envelope))
            allowance = (7/5)*dt*np.sum(np.abs(post_angular[:, 1]))
            allowance += (112/5)*dt*dt*damping*np.sum(np.abs(angular[:, 1]))+16*dt*support
            input_slack = allowance-np.sum(np.abs(forcing))
            assert transient_error < 2e-6 and input_slack >= -2e-6
            current_pitch_bound = (abs(np.exp(float(sys.ang_damping)*dt))
                                   *(np.abs(angular[:, 1])+dt*np.abs(np.asarray(acceleration.ang[:, 1])))
                                   +np.abs(np.asarray(delta.ang[:, 1])))
            current_allowance = (7/5)*dt*np.sum(current_pitch_bound)
            current_allowance += (112/5)*dt*dt*damping*np.sum(np.abs(angular[:, 1]))+16*dt*support
            current_input_slack = current_allowance-np.sum(np.abs(forcing))
            assert current_input_slack >= -2e-6
            minimum_current_input_slack = min(minimum_current_input_slack, current_input_slack)
            minimum_input_slack = min(minimum_input_slack, input_slack)
            inputs.append(forcing)
            allowances.append(allowance)
            angular_inputs.append(geometry_map@rho+bias_map@bias)
            contact_inputs.append(contact)
            geometry_bounds.append(rotation_bound)
            bias_bounds.append(bias_bound)
            contact_supports.append(support)
            response = powers[len(inputs)]@initial_transient
            for index, item in enumerate(inputs):
                response += powers[len(inputs)-1-index]@item
            response_error = np.max(np.abs(expected-response))
            assert response_error < 2e-5
            gain_bound = np.linalg.norm(powers[len(inputs)], 1)
            gain_bound *= np.sum(np.abs(initial_transient))
            for index, item in enumerate(allowances):
                gain_bound += np.linalg.norm(powers[len(inputs)-1-index], 1)*item
            assert np.sum(np.abs(response)) <= gain_bound+2e-12
            rotated = np.asarray(com_lever, dtype=np.float64)
            rotated_link = np.asarray(link_lever, dtype=np.float64)
            rotated_levers = np.concatenate([rotated[1:], rotated[parents], rotated_link])
            owners = np.concatenate([np.arange(1, 7), parents, parents])
            half_pitch = dt*post_angular[owners, 1]/2
            post_rotated = np.asarray(post_com_lever, dtype=np.float64)
            post_rotated_link = np.asarray(post_link_lever, dtype=np.float64)
            next_rotated = np.concatenate([post_rotated[1:], post_rotated[parents], post_rotated_link])
            lever_delta = next_rotated[:, 2]-rotated_levers[:, 2]
            rotation_identity_error = np.max(np.abs(
                (1+half_pitch**2)*lever_delta+2*half_pitch*rotated_levers[:, 0]
                +2*half_pitch**2*rotated_levers[:, 2]))
            assert rotation_identity_error < 2e-6
            source_input = -dt*(operator@(stiffness*geometry+damping*bias))+edge_contact
            force_error = np.max(np.abs(incidence.T@edge_force-np.asarray(force.vel[:, 2], dtype=np.float64)))
            acceleration_error = np.max(np.abs(
                operator@edge_force-incidence@np.asarray(acceleration.vel[:, 2], dtype=np.float64)))
            speed_error = np.max(np.abs(post_speed-speed-dt*(operator@edge_force)-edge_contact))
            position_error = np.max(np.abs(post_y-y-dt*post_speed))
            forced_error = np.max(np.abs(post_speed-speed+dt*damping*(operator@speed)
                                        +dt*stiffness*(operator@y)-source_input))
            second_error = 0.
            if previous is not None:
                second_error = np.max(np.abs(post_y-2*y+previous+(dt*damping+dt*dt*stiffness)*(operator@y)
                                            -dt*damping*(operator@previous)-dt*source_input))
            assert force_error < .05 and acceleration_error < .1
            assert max(speed_error, position_error, forced_error, second_error) < 5e-4
            assert np.max(np.abs(geometry)) <= .9+2e-6
            maxima = np.maximum(maxima, [force_error, acceleration_error, speed_error, position_error,
                                         forced_error, second_error, np.max(np.abs(geometry)), bias_error,
                                         transient_error, response_error, rotation_identity_error, gain_bound,
                                         pre_observation_error, pre_bound])
            previous, state = y, post
            steps += 1
    return {"normal_reset_cases": 4, "source_substeps": steps,
            "maximum_joint_force_assembly_error": float(maxima[0]),
            "maximum_edge_acceleration_error": float(maxima[1]),
            "maximum_edge_speed_error": float(maxima[2]),
            "maximum_edge_position_error": float(maxima[3]),
            "maximum_forced_speed_error": float(maxima[4]),
            "maximum_coupled_second_order_error": float(maxima[5]),
            "maximum_edge_geometry": float(maxima[6]),
            "maximum_angular_bias_formula_error": float(maxima[7]),
            "minimum_angular_bias_bound_slack": float(minimum_bias_slack),
            "minimum_contact_edge_budget_slack": float(minimum_contact_slack),
            "maximum_anchor_transient_error": float(maxima[8]),
            "maximum_finite_forced_response_error": float(maxima[9]),
            "maximum_pitch_increment_identity_error": float(maxima[10]),
            "maximum_finite_response_l1_bound": float(maxima[11]),
            "maximum_pre_speed_observation_error": float(maxima[12]),
            "maximum_pre_speed_transient_l1_bound": float(maxima[13]),
            "minimum_geometry_increment_bound_slack": float(minimum_rotation_slack),
            "minimum_transient_input_bound_slack": float(minimum_input_slack),
            "minimum_contact_to_prefoot_response": float(minimum_contact_response),
            "maximum_signed_foot_response_error": float(maximum_foot_response_error),
            "minimum_signed_foot_lower": float(minimum_signed_foot_lower),
            "minimum_support_foot_lower": float(minimum_support_foot_lower),
            "minimum_exact_angular_foot_lower": float(minimum_exact_angular_lower),
            "minimum_pre_foot_speed": float(minimum_pre_foot_speed),
            "signed_angular_floor_cases": int(signed_floor_cases),
            "support_angular_floor_cases": int(support_floor_cases),
            "exact_angular_floor_cases": int(exact_angular_floor_cases),
            "modal_source_updates": 6*steps,
            "maximum_modal_source_error": float(maximum_modal_source_error),
            "maximum_modal_homogeneous_identity_error": float(maximum_modal_homogeneous_error),
            "minimum_modal_coercivity_slack": float(minimum_modal_coercivity_slack),
            "minimum_modal_upper_bound_slack": float(minimum_modal_upper_slack),
            "minimum_modal_forced_step_slack": float(minimum_modal_step_slack),
            "minimum_modal_forced_prefix_slack": float(minimum_modal_prefix_slack),
            "minimum_current_transient_input_bound_slack": float(minimum_current_input_slack),
            "maximum_modal_foot_identity_error": float(maximum_modal_foot_identity_error),
            "maximum_modal_center_history_error": float(maximum_modal_center_history_error),
            "minimum_modal_output_energy_slack": float(minimum_modal_output_slack),
            "minimum_modal_norm_step_slack": float(minimum_modal_norm_step_slack),
            "minimum_modal_norm_prefix_slack": float(minimum_modal_norm_prefix_slack),
            "minimum_energy_envelope_foot_lower": float(minimum_energy_foot_lower),
            "minimum_norm_envelope_foot_lower": float(minimum_norm_foot_lower),
            "energy_envelope_foot_floor_cases": int(energy_foot_floor_cases),
            "norm_envelope_foot_floor_cases": int(norm_foot_floor_cases),
            "maximum_actual_modal_energy": float(maximum_modal_energy),
            "maximum_modal_input_energy": float(maximum_modal_input_energy),
            "maximum_geometric_modal_envelope": float(maximum_modal_envelope),
            "modal_gamma_range": [float(np.min(modal_gamma)), float(np.max(modal_gamma))],
            "edge_eigenvalues": eigenvalues.tolist(), "maximum_homogeneous_root_modulus": root_modulus}


if __name__ == "__main__":
    print(json.dumps(test_coupled_edge_translation(), indent=2))
