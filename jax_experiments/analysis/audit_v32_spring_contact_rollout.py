"""Read-only contact diagnostics on the released V32 canonical actor."""
from __future__ import annotations

import argparse
import ast
import importlib.metadata
import inspect
import json
from pathlib import Path
import pickle
import socket
import sys
from types import FunctionType, SimpleNamespace

import jax
from jax import numpy as jnp
import numpy as np
from brax import com, contact, math
from brax.spring import collisions, pipeline
from flax import nnx

from jax_experiments.analysis import (
    regime_polarity_action_compensation_power_confirmation_v32 as protocol,
)
from jax_experiments.common.checkpoint import (
    _patch_flax_variablestate_unpickle, _restore_tree_like,
)
from jax_experiments.envs.brax_env import apply_action_disturbance
from jax_experiments.envs.stochastic_mode_env import StochasticModeEnv
from jax_experiments.networks.policy import GaussianPolicy


ROOT = Path(__file__).resolve().parents[2]
COUNT_FIELDS = (
    "contact_slots", "penetrating_contacts", "active_normal_contacts",
    "active_friction_contacts", "negative_scalar_contacts",
    "normal_below_inverse_mass", "nonpositive_normal_denominators",
    "nonpositive_tangent_denominators", "nonpositive_active_friction_denominators",
    "nonfinite_substeps",
)
MIN_FIELDS = ("min_normal_denominator", "min_tangent_denominator")
MAX_FIELDS = ("max_collision_delta_difference", "max_contact_lever",
              "max_scalar_source_formula_error", "max_matrix_source_formula_error")


def matrix_contact_resolver():
    """Change only the inertia gather in a private copy of the installed resolver."""
    tree = ast.parse(inspect.getsource(collisions.resolve))
    assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "i_inv" for t in n.targets)]
    assert len(assignments) == 1
    assignments[0].value = ast.parse(
        "state.i_inv.take(link_idx, axis=0) * (link_idx > -1)[..., None, None]",
        mode="eval").body
    namespace = dict(collisions.resolve.__globals__)
    exec(compile(ast.fix_missing_locations(tree), "<matrix-contact-audit>", "exec"), namespace)
    return namespace["resolve"]


MATRIX_RESOLVE = matrix_contact_resolver()
MATRIX_STEP = FunctionType(pipeline.step.__code__,
                          {**pipeline.step.__globals__,
                           "collisions": SimpleNamespace(resolve=MATRIX_RESOLVE)},
                          "matrix_contact_step", pipeline.step.__defaults__)


def contact_statistics(sys, state, matrix_pipeline):
    c = contact.get(sys, state.x)
    body = c.link_idx[1]
    lever = c.pos - state.x_i.pos[body]
    velocity = state.xd_i.vel[body] + jnp.cross(state.xd_i.ang[body], lever)
    normal = c.frame[:, 0]
    normal_velocity = jnp.sum(normal * velocity, axis=-1)
    tangent = velocity - normal_velocity[:, None] * normal
    speed = jax.vmap(math.safe_norm)(tangent)
    direction = tangent / (1e-6 + speed[:, None])
    inertia = com.inv_inertia(sys, state.x)
    scalar = inertia.take(body)
    mass_inv = 1 / state.mass[body]

    def denominator(d, matrix):
        cross = jnp.cross(lever, d)
        angular = (jnp.einsum("nij,nj->ni", inertia[body], cross)
                   if matrix else scalar[:, None] * cross)
        return mass_inv + jnp.sum(d * jnp.cross(angular, lever), axis=-1)

    normal_den = denominator(normal, False)
    tangent_den = denominator(direction, False)
    raw = (-(1+c.elasticity)*normal_velocity
           - sys.baumgarte_erp/sys.opt.timestep*c.dist) / normal_den
    active = (c.dist < 0) & (normal_velocity < 0) & (raw > 0)
    friction_active = active & (speed > 1e-3)

    def increments(matrix):
        nd, td = denominator(normal, matrix), denominator(direction, matrix)
        impulse = (-(1+c.elasticity)*normal_velocity
                   - sys.baumgarte_erp/sys.opt.timestep*c.dist) / nd
        mask = (c.dist < 0) & (normal_velocity < 0) & (impulse > 0)
        drag = jnp.minimum(speed/td, c.friction[:, 0]*impulse)
        force = (impulse*mask)[:, None]*normal - (
            drag*mask*(speed > 1e-3))[:, None]*direction
        counts = jax.ops.segment_sum(mask.astype(jnp.float32), body, 7)[:, None] + 1e-8
        force_sum = jax.ops.segment_sum(force, body, 7)/counts
        torque_sum = jax.ops.segment_sum(jnp.cross(lever, force), body, 7)/counts
        return force_sum/state.mass[:, None], jnp.einsum("nij,nj->ni", inertia, torque_sum)

    actual = increments(False)
    matrix = increments(True)
    refreshed = state.replace(i_inv=inertia)
    scalar_source = collisions.resolve(sys, refreshed)
    matrix_source = MATRIX_RESOLVE(sys, refreshed)
    scalar_error = jnp.maximum(jnp.max(jnp.abs(actual[0]-scalar_source.vel)),
                               jnp.max(jnp.abs(actual[1]-scalar_source.ang)))
    matrix_error = jnp.maximum(jnp.max(jnp.abs(matrix[0]-matrix_source.vel)),
                               jnp.max(jnp.abs(matrix[1]-matrix_source.ang)))
    if matrix_pipeline:
        normal_den = denominator(normal, True)
        tangent_den = denominator(direction, True)
        raw = (-(1+c.elasticity)*normal_velocity
               - sys.baumgarte_erp/sys.opt.timestep*c.dist) / normal_den
        active = (c.dist < 0) & (normal_velocity < 0) & (raw > 0)
        friction_active = active & (speed > 1e-3)
    finite = jnp.all(jnp.isfinite(normal_den)) & jnp.all(jnp.isfinite(tangent_den))
    finite &= jnp.all(jnp.isfinite(state.q)) & jnp.all(jnp.isfinite(state.qd))
    resolved = matrix if matrix_pipeline else actual
    finite &= jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in resolved]))
    difference = jnp.maximum(jnp.max(jnp.abs(actual[0]-matrix[0])),
                             jnp.max(jnp.abs(actual[1]-matrix[1])))
    counts = (jnp.array(c.dist.size), jnp.sum(c.dist < 0), jnp.sum(active),
              jnp.sum(friction_active), jnp.sum(scalar < 0),
              jnp.sum((c.dist < 0) & (normal_den < mass_inv)),
              jnp.sum(normal_den <= 0), jnp.sum(tangent_den <= 0),
              jnp.sum(friction_active & (tangent_den <= 0)), (~finite).astype(jnp.int32))
    return {
        **dict(zip(COUNT_FIELDS, counts)),
        "min_normal_denominator": jnp.min(normal_den),
        "min_tangent_denominator": jnp.min(tangent_den),
        "max_collision_delta_difference": difference,
        "max_contact_lever": jnp.max(jnp.abs(lever)),
        "max_scalar_source_formula_error": scalar_error,
        "max_matrix_source_formula_error": matrix_error,
    }


def summarize(values):
    return {
        **{k: jnp.sum(values[k]) for k in COUNT_FIELDS},
        **{k: jnp.min(values[k]) for k in MIN_FIELDS},
        **{k: jnp.max(values[k]) for k in MAX_FIELDS},
    }


def audit(policy_seed: int, event_seed: int, steps: int):
    directory = ROOT / "reproducibility/v32/reference_policies" / f"seed_{policy_seed}" / "mode_0"
    config = json.loads((directory / "logs/protocol_signature.json").read_text())["config"]
    policy = GaussianPolicy(17, 6, config["hidden_dim"], rngs=nnx.Rngs(0))
    _patch_flax_variablestate_unpickle()
    with (directory / "policy/policy_params.pkl").open("rb") as handle:
        raw = pickle.load(handle)
    nnx.update(policy, _restore_tree_like(nnx.state(policy, nnx.Param), raw,
                                        "V32 compact canonical actor", allow_fallback=False))
    graphdef, params = nnx.split(policy, nnx.Param)
    env = StochasticModeEnv("HalfCheetah-v2", family="actuator_polarity", backend="spring")
    sys_model = env.env.sys
    gains = jnp.asarray([t["action_gain"] for t in env.sample_tasks(4)])
    schedule = jnp.asarray(protocol.SWITCHING_SCHEDULES[event_seed])
    rng, reset_key = jax.random.split(jax.random.PRNGKey(event_seed))
    initial = env._reset_fn(sys_model, reset_key)
    ground = contact.get(sys_model, initial.pipeline_state.x)
    assert np.all(np.asarray(ground.link_idx[0]) == -1)
    assert initial.obs.shape == (17,) and sys_model.num_links() == 7

    def rollout(parameters, compensated, matrix_pipeline):
        actor = nnx.merge(graphdef, parameters)

        def outer(carry, step):
            state, key = carry
            key, noise_key = jax.random.split(key)
            mode = schedule[(step//250) % len(schedule)]
            reference = actor.deterministic(state.obs[None])[0]
            command = jnp.where(compensated, gains[mode]*gains[0]*reference, reference)
            executed = apply_action_disturbance(command, noise_key, gains[mode], jnp.array(0.02))

            def inner(ps, _):
                stats = contact_statistics(sys_model, ps, matrix_pipeline)
                step_fn = MATRIX_STEP if matrix_pipeline else pipeline.step
                return step_fn(sys_model, ps, executed, False), stats

            post, values = jax.lax.scan(inner, state.pipeline_state, None, length=env.env._n_frames)
            reward, observation, done = env._reward_obs(state.pipeline_state, post, executed)
            updated = state.replace(pipeline_state=post, obs=observation, reward=reward, done=done)
            return (updated, key), {**summarize(values), "reward": reward}

        _, values = jax.lax.scan(outer, (initial, rng), jnp.arange(steps))
        return {**summarize(values), "return": jnp.sum(values["reward"])}

    run = jax.jit(rollout, static_argnums=2)
    rows = {}
    for matrix_pipeline in (False, True):
        for route, compensated in (("true_mode_compensation", True), ("no_compensation", False)):
            name = f"{'matrix' if matrix_pipeline else 'original'}_{route}"
            row = jax.device_get(run(params, jnp.array(compensated), matrix_pipeline))
            rows[name] = {k: int(v) if k in COUNT_FIELDS else float(v) if np.isfinite(v) else None
                          for k, v in row.items()}
            print(json.dumps({"arm": name, **rows[name]}), flush=True)
    frozen = json.loads((ROOT / "reproducibility/v32/audits" / f"seed_{policy_seed}"
                         / "audit.json").read_text())["switching_holdout"][str(event_seed)]
    return {
        "purpose": "retrospective source diagnostic; not an additional performance comparison",
        "policy_seed": policy_seed, "event_seed": event_seed, "steps": steps,
        "physics_substeps_per_action": env.env._n_frames,
        "runtime": {"python": sys.executable, "host": socket.gethostname(),
                    "brax": importlib.metadata.version("brax"), "jax": importlib.metadata.version("jax")},
        "arms": rows,
        "matrix_comparison": "private resolver and pipeline copies; original modules remain unchanged",
        "historical_replay": "not exact: current runtime and instrumented fused rollout",
        "frozen_first_episode_returns": {
            "true_mode_compensation": frozen[protocol.ORACLE_COMPENSATION_ARM]["returns"][0],
            "no_compensation": frozen[protocol.NO_COMPENSATION_ARM]["returns"][0],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-seed", type=int, default=88003, choices=protocol.TRAINING_SEEDS)
    parser.add_argument("--event-seed", type=int, default=197201, choices=protocol.SWITCHING_EVENT_SEEDS)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = audit(args.policy_seed, args.event_seed, args.steps)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
