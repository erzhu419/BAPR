"""Check the bounded primitive and exact angular-bias endpoint correction."""

import json

import jax
import numpy as np
from jax import numpy as jnp
from brax import math
from brax.spring import pipeline

from jax_experiments.envs.stochastic_mode_env import StochasticModeEnv


def test_angular_primitive():
    env = StochasticModeEnv("HalfCheetah-v2", family="actuator_polarity", backend="spring")
    sys = env.base_sys
    dt = float(sys.opt.timestep)
    assert dt == 1/320 and env.env._n_frames == 16
    parents = jnp.array([0, 1, 2, 0, 4, 5])
    offsets = sys.link.inertia.transform.pos
    levers = (-offsets[1:], -offsets[parents], sys.link.transform.pos[1:])
    assert max(np.max(np.linalg.norm(np.asarray(v), axis=1)) for v in levers[:2]) <= .2
    assert np.max(np.linalg.norm(np.asarray(levers[2]), axis=1)) <= .5
    np.testing.assert_array_equal(sys.link.joint.pos, np.zeros((7, 3)))

    def rotated(state):
        return (jax.vmap(math.rotate)(levers[0], state.x.rot[1:])[:, 2],
                jax.vmap(math.rotate)(levers[1], state.x.rot[parents])[:, 2],
                jax.vmap(math.rotate)(levers[2], state.x.rot[parents])[:, 2])

    @jax.jit
    def rollout(initial, actions):
        def advance(state, action):
            post = pipeline.step(sys, state, action)
            old, new = rotated(state), rotated(post)
            primitive0 = old[0]-old[1]+old[2]
            primitive1 = new[0]-new[1]+new[2]
            omega = post.xd_i.ang[:, 1]
            child_factor, parent_factor = (dt*omega[1:]/2)**2, (dt*omega[parents]/2)**2
            correction = (child_factor*(old[0]+new[0])-parent_factor*(old[1]+new[1])
                          +parent_factor*(old[2]+new[2]))
            speed = post.xd_i.vel[1:, 2]-post.xd_i.vel[parents, 2]
            bias = jax.vmap(math.rotate)(post.jd.vel[1:], post.a_p.rot[1:])[:, 2]-speed
            budget = dt**2/2*(.2*omega[1:]**2+.7*omega[parents]**2)
            geometry0, geometry1 = old[0]-old[1]-old[2], new[0]-new[1]-new[2]
            return post, (primitive0, primitive1, correction, bias, budget, omega,
                          geometry0, geometry1)

        return jax.lax.scan(advance, initial, jnp.repeat(actions, 16, axis=0))[1]

    # Use the actual translational response rows, including the current bias term.
    state = env._reset_fn(sys, jax.random.PRNGKey(90310)).pipeline_state
    mass = np.asarray(state.mass, dtype=np.float64)
    incidence = np.zeros((6, 7))
    incidence[np.arange(6), np.arange(1, 7)] = 1.
    incidence[np.arange(6), np.asarray(parents)] = -1.
    operator = (incidence/mass)@incidence.T
    spring, damper = dt**2*20000*operator, np.eye(6)-dt*10*operator
    transient = np.block([[np.eye(6)-spring, damper], [-spring, damper]])
    weights = np.array([-sum(mass[1:4]), -sum(mass[2:4]), -mass[3],
                        sum(mass)-sum(mass[4:]), sum(mass)-sum(mass[5:]),
                        sum(mass)-mass[6]])/sum(mass)
    row = weights@np.concatenate([-dt*20000*operator, damper/dt], axis=1)
    lag_rows = []
    for _ in range(16000):
        lag_rows.append(row)
        row = row@transient
    lag_rows = np.asarray(lag_rows)
    rng = np.random.default_rng(90310)
    rows = []
    for case, name in enumerate(("zero", "positive", "negative", "random_held")):
        state = env._reset_fn(sys, jax.random.PRNGKey(90310+case)).pipeline_state
        actions = (np.full((1000, 6), (0., 1., -1.)[case]) if case < 3
                   else rng.uniform(-1, 1, (1000, 6)))
        values = jax.device_get(rollout(state, jnp.asarray(actions, dtype=jnp.float32)))
        old, new, correction, bias, budget, omega, geometry0, geometry1 = (
            np.asarray(v, dtype=np.float64) for v in values)
        assert all(np.all(np.isfinite(v)) for v in values)
        identity_error = float(np.max(np.abs(dt*bias-(new-old-correction))))
        assert identity_error < 2e-6, (name, identity_error)
        assert np.max(np.abs(correction)-budget) < 2e-6
        assert np.max(np.abs(old)) <= .9+2e-6 and np.max(np.abs(new)) <= .9+2e-6
        assert np.max(np.abs(geometry0)) <= .9+2e-6 and np.max(np.abs(geometry1)) <= .9+2e-6
        np.testing.assert_array_equal(new[:-1], old[1:])
        windows = []
        for t in (16, 256, 4096, 16000):
            r = np.zeros((t+1, 6))
            r[:t-1] = -dt*10*((lag_rows[:t-1, :6]+lag_rows[:t-1, 6:])@operator)[::-1]
            r[t-1:] = -10*weights@operator
            difference = np.diff(r, axis=0)
            boundary = float(r[t]@new[t-1]-r[0]@old[0])
            variation_term = float(np.sum(difference*new[:t]))
            correction_term = float(np.sum(r[:t]*correction[:t]))
            weighted_bias = float(dt*np.sum(r[:t]*bias[:t]))
            error = abs(weighted_bias-(boundary-variation_term-correction_term))
            new_bound = float(.9*(np.sum(np.abs(r[t]))+np.sum(np.abs(r[0]))
                                    +np.sum(np.abs(difference)))
                              +np.sum(np.abs(r[:t])*budget[:t]))
            old_bound = float(dt*np.sum(np.abs(r[:t])*(.2*np.abs(omega[:t, 1:])
                                       +.7*np.abs(omega[:t, np.asarray(parents)]))))
            assert error < 5e-4, (name, t, error)
            assert abs(weighted_bias) <= new_bound+5e-4
            windows.append({"substeps": t, "weighted_identity_error": error,
                            "absolute_weighted_bias": abs(weighted_bias),
                            "primitive_bound": new_bound, "absolute_pitch_bound": old_bound})
        result = {"command": name, "physics_substeps": len(old),
                  "maximum_endpoint_identity_error": identity_error,
                  "maximum_primitive": float(max(np.max(np.abs(old)), np.max(np.abs(new)))),
                  "maximum_correction": float(np.max(np.abs(correction))), "windows": windows}
        prefix_checks = []
        for t in (16, 256, 4096, 15999):
            response = np.vstack([lag_rows[:t][::-1], lag_rows[0]])
            rE, rB = response[:, :6], (response[:, :6]+response[:, 6:])@operator
            dE, dB = np.diff(rE, axis=0), np.diff(rB, axis=0)
            geometry_term = float(np.sum(rE[:t]*(geometry1[1:t+1]-geometry1[:t])))
            direct = geometry_term-dt**2*10*float(np.sum(rB[:t]*bias[:t]))
            g_parts = float(rE[t]@geometry1[t]-rE[0]@geometry1[0]
                            -np.sum(dE*geometry1[1:t+1]))
            b_parts = float(rB[t]@new[t-1]-rB[0]@old[0]
                            -np.sum(dB*new[:t])-np.sum(rB[:t]*correction[:t]))
            g_variation = np.sum(np.abs(rE[t]))+np.sum(np.abs(rE[0]))+np.sum(np.abs(dE))
            b_variation = np.sum(np.abs(rB[t]))+np.sum(np.abs(rB[0]))+np.sum(np.abs(dB))
            cost = float(.9*g_variation+abs(dt*10)*(.9*b_variation
                         +np.sum(np.abs(rB[:t])*budget[:t])))
            error = abs(direct-(g_parts-dt*10*b_parts))
            assert error < 5e-4 and abs(direct) <= cost+5e-4
            prefix_checks.append({"substeps": t, "full_input_identity_error": error,
                                  "absolute_angular_input": abs(direct), "primitive_input_bound": cost})
        result["input_prefix_checks"] = prefix_checks
        rows.append(result)
        print(json.dumps(result), flush=True)
    return rows


if __name__ == "__main__":
    test_angular_primitive()
