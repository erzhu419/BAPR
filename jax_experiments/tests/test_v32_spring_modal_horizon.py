"""Check the modal foot certificate over the supported 1000-action horizon."""

import json

import jax
import numpy as np
from jax import numpy as jnp
from brax import actuator, com, math
from brax.base import Motion
from brax.spring import collisions, joints, pipeline

from jax_experiments.envs.stochastic_mode_env import StochasticModeEnv
from jax_experiments.tests.test_v32_spring_coupled_edges import _modal_energy


def _source_rollout(sys):
    dt = float(sys.opt.timestep)
    parent = jnp.array([0, 1, 2, 0, 4, 5])

    def anchor_geometry(state):
        lever = jax.vmap(math.rotate)(sys.link.transform.pos[1:], state.x.rot[parent])
        error = state.x.pos[1:, 2]-state.x.pos[parent, 2]-lever[:, 2]
        y = state.x_i.pos[1:, 2]-state.x_i.pos[parent, 2]
        return error, error-y

    @jax.jit
    def rollout(initial, actions):
        def advance(state, action):
            refreshed = state.replace(i_inv=com.inv_inertia(sys, state.x))
            force = joints.resolve(sys, refreshed, actuator.to_tau(sys, action, state.q, state.qd))
            acceleration = Motion(vel=sys.gravity+force.vel/state.mass[:, None],
                                  ang=jnp.einsum("ijk,ik->ij", refreshed.i_inv, force.ang))
            pre = state.xd_i+acceleration*dt
            delta = collisions.resolve(sys, refreshed.replace(xd_i=pre))
            post = pipeline.step(sys, state, action)
            error, geometry = anchor_geometry(state)
            _, next_geometry = anchor_geometry(post)
            speed = state.xd_i.vel[1:, 2]-state.xd_i.vel[parent, 2]
            # The joint cache stores local velocity; unrotate its linear part.
            relative = jax.vmap(math.rotate)(refreshed.jd.vel[1:],
                                            refreshed.a_p.rot[1:])[:, 2]
            values = (error, speed, relative-speed, next_geometry-geometry,
                      delta.vel[:, 2], pre.vel[6, 2],
                      jnp.dot(state.mass, state.xd_i.vel[:, 2])/jnp.sum(state.mass),
                      state.x_i.pos[6, 2])
            return post, values

        return jax.lax.scan(advance, initial, jnp.repeat(actions, 16, axis=0))[1]

    return rollout


def test_modal_foot_horizon():
    env = StochasticModeEnv("HalfCheetah-v2", family="actuator_polarity", backend="spring")
    sys = env.base_sys
    dt = float(sys.opt.timestep)
    stiffness, damping = 20000., 10.
    assert dt == 1/320 and float(sys.vel_damping) == 0.
    np.testing.assert_array_equal(sys.link.constraint_stiffness[1:], np.full(6, stiffness))
    np.testing.assert_array_equal(sys.link.constraint_vel_damping[1:], np.full(6, damping))
    assert env.env._n_frames == 16
    parents = np.array([0, 1, 2, 0, 4, 5])
    incidence = np.zeros((6, 7))
    incidence[np.arange(6), np.arange(1, 7)] = 1.
    incidence[np.arange(6), parents] = -1.
    rollout = _source_rollout(sys)

    rng = np.random.default_rng(90310)
    rows = []
    for case, name in enumerate(("zero", "positive", "negative", "random_held")):
        state = env._reset_fn(sys, jax.random.PRNGKey(90310+case)).pipeline_state
        actions = (np.full((1000, 6), (0., 1., -1.)[case]) if case < 3
                   else rng.uniform(-1, 1, (1000, 6)))
        values = jax.device_get(rollout(state, jnp.asarray(actions, dtype=jnp.float32)))
        error, speed, bias, rho, contact, prefoot, mean, height = (
            np.asarray(item, dtype=np.float64) for item in values)
        assert all(np.all(np.isfinite(item)) for item in values)
        mass = np.asarray(state.mass, dtype=np.float64)
        operator = (incidence/mass)@incidence.T
        eigenvalues, eigenvectors = np.linalg.eigh(operator)
        c, gamma = dt*dt*stiffness*eigenvalues, dt*damping*eigenvalues
        weights = np.array([-sum(mass[1:4]), -sum(mass[2:4]), -mass[3],
                            sum(mass)-sum(mass[4:]), sum(mass)-sum(mass[5:]),
                            sum(mass)-mass[6]])/sum(mass)
        modal_weights = weights@eigenvectors
        e, v = error@eigenvectors, dt*speed@eigenvectors
        b = bias@eigenvectors
        kick = -dt*dt*damping*(bias@operator.T)+dt*(contact@incidence.T)
        q, r = (rho+kick)@eigenvectors, kick@eigenvectors
        output = -c*e+(1-gamma)*v
        center = mean+dt*float(sys.gravity[2])-np.sum(modal_weights*gamma*b, axis=1)
        predicted = center+output@modal_weights/dt
        output_error = float(np.max(np.abs(prefoot-predicted)))
        assert output_error < 5e-4, output_error
        norm = np.sqrt(_modal_energy(c, gamma, e, v))
        input_norm = np.sqrt(_modal_energy(c, gamma, q, r))
        response = norm[0].copy()
        block_response = norm[0].copy()
        lower = np.zeros(len(prefoot))
        block_lower = np.zeros(len(prefoot))
        gain = np.abs(modal_weights)*np.sqrt((1-gamma)/(1-841*c/2500))/dt
        current_lower = center-norm@gain
        homogeneous_e, homogeneous_v = (1-c)*e+(1-gamma)*v, -c*e+(1-gamma)*v
        polar = (c*homogeneous_e*q+(gamma-c)/2*(homogeneous_e*r+homogeneous_v*q)
                 +(1-gamma)*homogeneous_v*r)
        signed_gain = -gamma*_modal_energy(c, gamma, e, v)+2*polar+_modal_energy(c, gamma, q, r)
        energy_gain_error = float(np.max(np.abs(
            _modal_energy(c, gamma, e[1:], v[1:])-
            _modal_energy(c, gamma, e[:-1], v[:-1])-signed_gain[:-1])))
        denominator = norm[:-1]+norm[1:]
        norm_gain = np.divide(signed_gain[:-1], denominator,
                              out=np.zeros_like(denominator), where=denominator > 0.)
        norm_gain_error = float(np.max(np.abs(norm[1:]-norm[:-1]-norm_gain)))
        bias_change = (b[1:]-b[:-1])@(modal_weights*gamma)
        center_gain = dt*float(sys.gravity[2])+contact[:-1]@mass/sum(mass)-bias_change
        local_drift = center_gain-norm_gain@gain
        drift_error = float(np.max(np.abs(current_lower[1:]-current_lower[:-1]-local_drift)))
        assert energy_gain_error < 2e-6 and norm_gain_error < 2e-6
        assert drift_error < 5e-4
        for t in range(len(prefoot)):
            if t % 16 == 0:
                block_response = norm[t].copy()
            lower[t] = center[t]-gain@response
            block_lower[t] = center[t]-gain@block_response
            assert np.min(response-norm[t]) >= -2e-6
            assert np.min(block_response-norm[t]) >= -2e-6
            assert np.max(block_response-response) <= 2e-6
            response = np.sqrt(1-gamma)*response+input_norm[t]
            block_response = np.sqrt(1-gamma)*block_response+input_norm[t]
        assert np.max(lower-prefoot) <= 5e-4
        assert np.max(block_lower-prefoot) <= 5e-4
        assert np.max(current_lower-prefoot) <= 5e-4
        assert np.all(current_lower >= -64.)
        assert np.all(block_lower >= -64.)
        failures = np.flatnonzero(lower < -64.)
        row = {"command": name, "physics_substeps": len(prefoot),
               "first_norm_floor_failure": int(failures[0]) if len(failures) else None,
               "norm_floor_cases": int(np.sum(lower >= -64.)),
               "minimum_norm_lower": float(np.min(lower)),
               "minimum_current_norm_lower": float(np.min(current_lower)),
               "current_norm_floor_cases": int(np.sum(current_lower >= -64.)),
               "minimum_action_block_norm_lower": float(np.min(block_lower)),
               "action_block_norm_floor_cases": int(np.sum(block_lower >= -64.)),
               "minimum_actual_prefoot_speed": float(np.min(prefoot)),
               "actual_speed_floor_cases": int(np.sum(prefoot >= -64.)),
               "minimum_front_foot_height": float(np.min(height)),
               "maximum_foot_identity_error": output_error,
               "maximum_signed_energy_gain_error": energy_gain_error,
               "maximum_signed_norm_gain_error": norm_gain_error,
               "maximum_local_drift_identity_error": drift_error,
               "negative_input_polar_terms": int(np.sum(polar[:-1] < 0.)),
               "signed_modal_updates": int(polar[:-1].size)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    return {"purpose": "source reachability diagnostic, not a performance experiment", "rows": rows}


def test_modal_corner_commands():
    """Audit every constant corner of the six-motor command box, not all policies."""
    env = StochasticModeEnv("HalfCheetah-v2", family="actuator_polarity", backend="spring")
    sys, dt = env.base_sys, float(env.base_sys.opt.timestep)
    assert dt == 1/320 and env.env._n_frames == 16
    state = env._reset_fn(sys, jax.random.PRNGKey(90310)).pipeline_state
    mass = np.asarray(state.mass, dtype=np.float64)
    incidence = np.zeros((6, 7))
    incidence[np.arange(6), np.arange(1, 7)] = 1.
    incidence[np.arange(6), [0, 1, 2, 0, 4, 5]] = -1.
    operator = (incidence/mass)@incidence.T
    eigenvalues, eigenvectors = np.linalg.eigh(operator)
    c, gamma = dt*dt*20000*eigenvalues, dt*10*eigenvalues
    weights = np.array([-sum(mass[1:4]), -sum(mass[2:4]), -mass[3],
                        sum(mass)-sum(mass[4:]), sum(mass)-sum(mass[5:]),
                        sum(mass)-mass[6]])/sum(mass)
    modal_weights = weights@eigenvectors
    gain = np.abs(modal_weights)*np.sqrt((1-gamma)/(1-841*c/2500))/dt
    rollout = _source_rollout(sys)
    rows = []
    for code in range(64):
        command = np.array([1. if code & (1 << j) else -1. for j in range(6)])
        values = jax.device_get(rollout(state, jnp.tile(jnp.asarray(command), (1000, 1))))
        error, speed, bias, rho, contact, prefoot, mean, _ = (
            np.asarray(item, dtype=np.float64) for item in values)
        assert all(np.all(np.isfinite(item)) for item in values), code
        e, v, b = error@eigenvectors, dt*speed@eigenvectors, bias@eigenvectors
        kick = -dt*dt*10*(bias@operator.T)+dt*(contact@incidence.T)
        q, r = (rho+kick)@eigenvectors, kick@eigenvectors
        energy = _modal_energy(c, gamma, e, v)
        input_energy = _modal_energy(c, gamma, q, r)
        norm, input_norm = np.sqrt(energy), np.sqrt(input_energy)
        center = mean+dt*float(sys.gravity[2])-np.sum(modal_weights*gamma*b, axis=1)
        current_lower = center-norm@gain
        predicted = center+(-c*e+(1-gamma)*v)@modal_weights/dt
        identity_error = float(np.max(np.abs(predicted-prefoot)))
        assert identity_error < 5e-4, (code, identity_error)
        assert np.max(current_lower-prefoot) < 5e-4
        block_response = norm[0].copy()
        block_lower = np.zeros(len(prefoot))
        for t in range(len(prefoot)):
            if t % 16 == 0:
                block_response = norm[t].copy()
            block_lower[t] = center[t]-gain@block_response
            assert np.min(block_response-norm[t]) >= -2e-6
            block_response = np.sqrt(1-gamma)*block_response+input_norm[t]
        assert np.max(block_lower-prefoot) < 5e-4
        current_failures = np.flatnonzero(current_lower < -64.)
        block_failures = np.flatnonzero(block_lower < -64.)
        row = {"code": code, "command": command.astype(int).tolist(),
               "minimum_current_lower": float(np.min(current_lower)),
               "minimum_block_lower": float(np.min(block_lower)),
               "minimum_actual_speed": float(np.min(prefoot)),
               "current_floor_failures": int(len(current_failures)),
               "block_floor_failures": int(len(block_failures)),
               "actual_floor_failures": int(np.sum(prefoot < -64.)),
               "maximum_identity_error": identity_error}
        if len(current_failures) or len(block_failures):
            row["first_current_failure"] = int(current_failures[0]) if len(current_failures) else None
            row["first_block_failure"] = int(block_failures[0]) if len(block_failures) else None
            print(json.dumps(row), flush=True)
        rows.append(row)
        if code % 16 == 15:
            print(json.dumps({"completed_corner_profiles": code+1}), flush=True)
    summary = {"purpose": "constant-command source diagnostic, not universal verification",
               "reset_key": 90310, "profiles": len(rows), "physics_substeps": 16000*len(rows),
               "current_floor_failures": sum(row["current_floor_failures"] for row in rows),
               "block_floor_failures": sum(row["block_floor_failures"] for row in rows),
               "actual_floor_failures": sum(row["actual_floor_failures"] for row in rows),
               "worst_current": min(rows, key=lambda row: row["minimum_current_lower"]),
               "worst_block": min(rows, key=lambda row: row["minimum_block_lower"])}
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    test_modal_foot_horizon()
