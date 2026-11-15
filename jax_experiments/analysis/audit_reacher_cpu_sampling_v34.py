"""Remote-only CPU sampler audit; exports JSON, not trajectories/checkpoints."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import time
from pathlib import Path

import brax
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from scipy.stats import t as student_t

from jax_experiments.analysis import run_mechanism_gap_v34 as run
from jax_experiments.envs.brax_env import apply_action_disturbance
from jax_experiments.networks.policy import GaussianPolicy

SEEDS = (92101, 92103, 92107)
RETURN_MARGIN = 0.01


def return_equivalence(gpu_rewards, cpu_rewards):
    gpu = np.asarray(gpu_rewards).reshape(-1, 200).sum(axis=1).astype(np.float64)
    cpu = np.asarray(cpu_rewards).reshape(-1, 200).sum(axis=1).astype(np.float64)
    difference = cpu - gpu
    radius = student_t.ppf(0.975, len(difference) - 1) * difference.std(ddof=1) / np.sqrt(len(difference))
    interval = [float(difference.mean() - radius), float(difference.mean() + radius)]
    margin = RETURN_MARGIN * max(1.0, abs(float(gpu.mean())))
    return {"episodes": len(difference), "gpu_mean": float(gpu.mean()), "cpu_mean": float(cpu.mean()),
            "paired_difference_ci95": interval, "equivalence_margin": margin,
            "passed": bool(interval[0] > -margin and interval[1] < margin)}


def audit(output):
    if jax.__version__ != "0.4.38" or brax.__version__ != "0.10.5":
        raise ValueError("This replacement protocol freezes JAX 0.4.38 and Brax 0.10.5")
    gpu, cpu = jax.devices("gpu")[0], jax.devices("cpu")[0]
    world_gpu = run.World("reacher", SEEDS[0])
    with jax.default_device(cpu):
        world_cpu = run.World("reacher", SEEDS[0])
    fields = ("gravity", "dt", "link.inertia.mass", "dof.damping", "actuator.gear")

    def physics(world):
        values = {}
        for field in fields:
            value = world.env if field == "dt" else world.sys
            for name in field.split("."):
                value = getattr(value, name)
            values[field] = np.asarray(value)
        return values

    first_physics, second_physics = physics(world_gpu), physics(world_cpu)
    physics_match = all(np.array_equal(first_physics[name], second_physics[name]) for name in fields)
    print(f"AUDIT physics_match={physics_match}", flush=True)
    policy = GaussianPolicy(world_gpu.obs_dim, world_gpu.act_dim, 256, rngs=nnx.Rngs(SEEDS[0]))
    graph = nnx.graphdef(policy)
    gpu_scan = run.build_training_scan(world_gpu, graph, None, False)
    with jax.default_device(cpu):
        cpu_scan = run.build_training_scan(world_cpu, graph, None, False)

    @jax.jit
    def observed_states(params, state, keys, modes):
        def step(current, pair):
            key, mode = pair
            action_key, noise_key, unused = jax.random.split(key, 3)
            command = nnx.merge(graph, params).sample(current.obs[None], action_key)[0][0]
            executed = apply_action_disturbance(command, noise_key, world_gpu.gains[mode], 0.02)
            return world_gpu.step(current, executed), (current, executed)
        return jax.lax.scan(step, state, (keys, modes))[1]

    def single_steps(world):
        @jax.jit
        def evaluate(states, actions):
            def step(pair):
                following = world.step(*pair)
                return following.obs, following.reward, following.done
            return jax.lax.map(step, (states, actions))
        return evaluate

    gpu_steps = single_steps(world_gpu)
    with jax.default_device(cpu):
        cpu_steps = single_steps(world_cpu)
    cases = []
    for seed in SEEDS:
        params = nnx.state(GaussianPolicy(world_gpu.obs_dim, world_gpu.act_dim, 256, rngs=nnx.Rngs(seed)), nnx.Param)
        key, reset_key = jax.random.split(jax.random.PRNGKey(seed))
        state = world_gpu.reset(world_gpu.sys, reset_key)
        keys = jax.random.split(key, 4000)
        for arm in ("reference", "robust"):
            print(f"AUDIT seed={seed} arm={arm} phase=single_step", flush=True)
            modes = jnp.zeros(4000, jnp.int32) if arm == "reference" else jnp.asarray(run.training_modes(seed, 4000, 50))
            states, actions = observed_states(params, state, keys[:200], modes[:200])
            gpu_values = gpu_steps(states, actions)
            cpu_values = cpu_steps(jax.device_put(states, cpu), jax.device_put(actions, cpu))
            single = {name: {"max_absolute_difference": float(np.max(np.abs(np.asarray(a) - np.asarray(b)))),
                             "passed": bool(np.allclose(a, b, atol=1e-5, rtol=1e-5))}
                      for name, a, b in zip(("next_observation", "reward", "done"), gpu_values, cpu_values, strict=True)}
            print(f"AUDIT seed={seed} arm={arm} single_step={single}", flush=True)
            inputs = (params, state, jnp.full(4, 0.25), jnp.asarray(0), keys, modes)
            timing, rewards = {}, {}
            for label, device, scan in (("gpu", gpu, gpu_scan), ("cpu", cpu, cpu_scan)):
                print(f"AUDIT seed={seed} arm={arm} phase={label}_compile", flush=True)
                placed = jax.device_put(inputs, device)
                jax.block_until_ready(placed)
                compiled = scan.lower(*placed).compile()
                print(f"AUDIT seed={seed} arm={arm} phase={label}_rollout", flush=True)
                start = time.perf_counter()
                result = compiled(*placed)
                delivered = jax.device_put(result[1], gpu)
                jax.block_until_ready(delivered)
                timing[label] = time.perf_counter() - start
                rewards[label] = np.asarray(delivered[2])
                if not all(np.isfinite(np.asarray(value)).all() for value in delivered):
                    raise ValueError("nonfinite CPU/GPU validation trajectory")
                if np.any(np.asarray(delivered[4]) != 0):
                    raise ValueError("native Reacher unexpectedly terminated")
            returns = return_equivalence(rewards["gpu"], rewards["cpu"])
            case = {"seed": seed, "arm": arm, "single_step": single, "return_equivalence": returns,
                    "warm_seconds": timing, "speedup": timing["gpu"] / timing["cpu"],
                    "passed": bool(all(value["passed"] for value in single.values()) and returns["passed"])}
            cases.append(case)
            print(f"AUDIT seed={seed} arm={arm} passed={case['passed']} "
                  f"single={single} returns={returns} seconds={timing}", flush=True)
    admitted = bool(physics_match and all(row["passed"] for row in cases))
    smoke = None
    if admitted:
        run.p.RUN_ROOT, run.p.BUNDLE_ROOT = run.p.CPU_RUN_ROOT, run.p.CPU_BUNDLE_ROOT
        smoke = run.train_policy("reacher", "reference", 95009, smoke=True, sampling_device="cpu")
    payload = {"passed": admitted,
               "jax_version": jax.__version__, "brax_version": brax.__version__,
               "physics_match": physics_match, "physics": {name: value.tolist() for name, value in first_physics.items()},
               "horizon": 200, "dwell": 50, "noise_std": 0.02, "training_updates": 0,
               "validation_policy_seeds": list(SEEDS), "cases": cases,
               "gpu_learner_smoke": smoke,
               "runtime_versions": {name: version(name) for name in
                                    ("jax", "jaxlib", "brax", "flax", "optax", "mujoco", "numpy", "scipy")},
               "closed_loop_bitwise_equivalence_required": False,
               "scope": "fresh matched CPU-sampling comparison, not checkpoint backend substitution"}
    run.write_json(output, payload)
    print(f"CPU_SAMPLER_AUDIT_COMPLETE passed={payload['passed']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)


if __name__ == "__main__":
    main()
