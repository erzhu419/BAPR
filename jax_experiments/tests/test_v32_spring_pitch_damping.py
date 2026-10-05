"""Check the source pitch recurrence without dropping lever/contact forcing."""

import json

import jax
import numpy as np
from jax import numpy as jnp
from brax import actuator, com, kinematics, scan
from brax.base import Motion, Transform
from brax.spring import collisions, joints, pipeline

from jax_experiments.envs.stochastic_mode_env import StochasticModeEnv


def test_pitch_damping():
    # Check the real-arithmetic factorization at higher diagnostic precision.
    jax.config.update("jax_enable_x64", True)
    env = StochasticModeEnv("HalfCheetah-v2", family="actuator_polarity", backend="spring")
    sys = jax.tree.map(lambda x: x.astype(jnp.float64)
                       if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating) else x,
                       env.base_sys)
    dt = float(sys.opt.timestep)
    assert dt == 1/320 and env.env._n_frames == 16
    parents = np.array([-1, 0, 1, 2, 0, 4, 5])
    np.testing.assert_array_equal(sys.link_parents, parents)
    incidence = np.zeros((6, 7))
    incidence[np.arange(6), np.arange(1, 7)] = 1.
    incidence[np.arange(6), parents[1:]] = -1.
    damping = np.asarray(sys.link.constraint_ang_damping, dtype=np.float64)
    assert np.all((damping >= 0) & (damping <= 10))
    laplacian = incidence.T@np.diag(damping[1:])@incidence
    laplacian[0, 0] += damping[0]
    rho = np.exp(float(sys.ang_damping)*dt)
    initial = env._reset_fn(sys, jax.random.PRNGKey(90310)).pipeline_state
    assert initial.xd_i.ang.dtype == jnp.float64
    lam = np.asarray(initial.i_inv, dtype=np.float64)[:, 1, 1]
    assert np.all((lam > 0) & (lam <= 4))
    mix = np.eye(7)-dt*lam[:, None]*laplacian
    assert np.min(mix) >= 0 and np.max(mix.sum(axis=1)) <= 1.
    assert np.max(dt*lam*np.diag(laplacian)) < 1.
    pitch_matrix = rho*mix

    @jax.jit
    def rollout(state, actions):
        def advance(state, action):
            inverse = com.inv_inertia(sys, state.x)
            state = state.replace(i_inv=inverse)
            tau = actuator.to_tau(sys, action, state.q, state.qd)
            # Retain the installed resolver's batching and force evaluation order.
            def joint_force(typ, link, joint, motion, dof, tau):
                def reshape(x):
                    return x.reshape((joint.pos.shape[0], -1)+x.shape[1:])

                tau, dof = jax.tree.map(reshape, (tau, dof))
                return jax.vmap({"1": joints._one_dof, "3": joints._three_dof}[typ])(
                    link, joint, motion, dof, tau)

            local = scan.link_types(sys, joint_force, "llldd", "l",
                                    sys.link, state.j, state.jd, sys.dof, tau)
            control = []
            for i in range(1, 7):
                dof = sys.dof.take(jnp.array([i+2]))
                frame, parity = kinematics.link_to_joint_frame(dof.motion)
                _, (angle, _, _), _ = kinematics.axis_angle_ang(state.j.take(i), frame, parity)
                lo, hi = dof.limit
                limit_error = jnp.where(angle > hi[0], angle-hi[0],
                                        jnp.where(angle < lo[0], angle-lo[0], 0))
                control.append(tau[i+2]-sys.link.constraint_limit_stiffness[i]*limit_error)
            world = Transform.create(rot=state.a_p.rot).vmap().do(local)
            world_force, world_torque = world.vel, world.ang
            child_lever = state.a_c.pos-state.x_i.pos
            parent_pos = jnp.concatenate([state.x_i.pos, jnp.zeros((1, 3))])[parents]
            parent_lever = state.a_p.pos-parent_pos
            shifted = jnp.cross(child_lever, world_force)[:, 1]
            incoming = jnp.zeros(7).at[parents[1:]].add(
                jnp.cross(parent_lever[1:], world_force[1:])[:, 1])
            lever = shifted-incoming
            force = joints.resolve(sys, state, tau)
            acceleration = Motion(vel=sys.gravity+force.vel/state.mass[:, None],
                                  ang=jnp.einsum("ijk,ik->ij", inverse, force.ang))
            delta = collisions.resolve(sys, state.replace(xd_i=state.xd_i+acceleration*dt))
            post = pipeline.step(sys, state, action)
            return post, (state.xd_i.ang[:, 1], post.xd_i.ang[:, 1], jnp.stack(control),
                          lever, force.ang[:, 1], delta.ang[:, 1], inverse[:, 1, 1],
                          world_torque[:, 1], state.jd.ang[:, 1])

        return jax.lax.scan(advance, state, jnp.repeat(actions, 16, axis=0))[1]

    rng = np.random.default_rng(90310)
    results = []
    for case, name in enumerate(("zero", "positive", "negative", "random_held")):
        state = env._reset_fn(sys, jax.random.PRNGKey(90310+case)).pipeline_state
        actions = (np.full((1000, 6), (0., 1., -1.)[case]) if case < 3
                   else rng.uniform(-1, 1, (1000, 6)))
        values = jax.device_get(rollout(state, jnp.asarray(actions, dtype=jnp.float64)))
        old, new, control, lever, torque, contact, observed_lam, world_torque, cached_pitch = (
            np.asarray(v, dtype=np.float64) for v in values)
        assert all(np.all(np.isfinite(v)) for v in values)
        np.testing.assert_allclose(observed_lam, np.broadcast_to(lam, observed_lam.shape),
                                   rtol=1e-10, atol=1e-10)
        forcing_torque = control@incidence+lever
        expected_torque = forcing_torque-old@laplacian.T
        local = np.column_stack([-damping[0]*old[:, 0], control-damping[1:]*(old@incidence.T)])
        assembled = world_torque[:, 1:]@incidence+lever
        assembled[:, 0] += world_torque[:, 0]
        np.testing.assert_allclose(world_torque, local, rtol=1e-10, atol=1e-9)
        np.testing.assert_allclose(cached_pitch[:, 1:], old@incidence.T, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(torque, expected_torque, rtol=1e-10, atol=1e-9)
        source = rho*dt*lam*forcing_torque+contact
        predicted = old@pitch_matrix.T+source
        np.testing.assert_allclose(new, predicted, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(old[1:], new[:-1], rtol=0, atol=0)

        signed = np.asarray(state.xd_i.ang, dtype=np.float64)[:, 1].copy()
        envelope = np.max(np.abs(signed))
        signed_error, envelope_slack, maximum_envelope = 0., np.inf, envelope
        for t, b in enumerate(source):
            signed = pitch_matrix@signed+b
            envelope = abs(rho)*envelope+np.max(np.abs(b))
            signed_error = max(signed_error, float(np.max(np.abs(signed-new[t]))))
            envelope_slack = min(envelope_slack, envelope-float(np.max(np.abs(new[t]))))
            maximum_envelope = max(maximum_envelope, envelope)
        assert signed_error < 1e-7, (name, signed_error)
        assert envelope_slack > -1e-10, (name, envelope_slack)
        result = {"command": name, "precision": "float64", "physics_substeps": len(old),
                  "maximum_damping_step": float(np.max(dt*lam*np.diag(laplacian))),
                  "maximum_torque_split_error": float(np.max(np.abs(torque-expected_torque))),
                  "maximum_assembly_error": float(np.max(np.abs(assembled-torque))),
                  "maximum_pitch_step_error": float(np.max(np.abs(new-predicted))),
                  "maximum_signed_trajectory_error": signed_error,
                  "maximum_actual_pitch_speed": float(np.max(np.abs(new))),
                  "maximum_absolute_forcing_envelope": maximum_envelope,
                  "minimum_envelope_slack": envelope_slack,
                  "maximum_shifted_linear_torque": float(np.max(np.abs(lever))),
                  "maximum_contact_pitch_increment": float(np.max(np.abs(contact)))}
        print(json.dumps(result), flush=True)
        results.append(result)
    return results


if __name__ == "__main__":
    test_pitch_damping()
