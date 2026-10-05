"""Rollout timing with identical inputs; no training or checkpoint exports."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import brax
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.analysis import run_mechanism_gap_v34 as run
from jax_experiments.envs.brax_env import apply_action_disturbance
from jax_experiments.envs.stochastic_mode_env import StochasticModeEnv
from jax_experiments.networks.policy import GaussianPolicy


def compare_outputs(reference, candidate, conditioned):
    first, tree = jax.tree.flatten(reference)
    second, other_tree = jax.tree.flatten(candidate)
    if tree != other_tree:
        raise ValueError("rollout output structure changed")
    equivalent, maximum = True, 0.0
    for a, b in zip(first, second, strict=True):
        a, b = np.asarray(a), np.asarray(b)
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise ValueError("non-finite benchmark trajectory")
        maximum = max(maximum, float(np.max(np.abs(a - b))) if a.size else 0.0)
        equivalent &= bool(np.allclose(a, b, atol=1e-5, rtol=1e-5))
    posterior_matches = True
    if conditioned:
        posterior_matches = bool(np.array_equal(
            np.argmax(reference[1][3][:, -4:], axis=-1),
            np.argmax(candidate[1][3][:, -4:], axis=-1)))
    return {"equivalent": equivalent and posterior_matches,
            "max_absolute_difference": maximum,
            "posterior_argmax_matches": posterior_matches}


def transition_differences(reference, candidate):
    return {name: {"max_absolute_difference": float(np.max(np.abs(np.asarray(first) - np.asarray(second)))),
                   "equivalent": bool(np.allclose(first, second, atol=1e-5, rtol=1e-5))}
            for name, first, second in zip(
                ("observation", "action", "reward", "next_observation", "done"),
                reference[1], candidate[1], strict=True)}


def timed_call(scan, inputs):
    start = time.perf_counter()
    result = scan(*inputs)
    # Include delivery back to the learner device for CPU rollout timing.
    result = jax.device_put(result, jax.devices("gpu")[0])
    jax.block_until_ready(result)
    return result, time.perf_counter() - start


def sampling_world(kind, seed):
    if kind in ("hc", "reacher"):
        return run.World(kind, seed)
    env = StochasticModeEnv(
        {"ant": "Ant-v2", "hopper": "Hopper-v2", "walker": "Walker2d-v2"}[kind],
        family="actuator_polarity", backend="spring", dwell_steps=250, seed=seed)

    def step(state, action):
        old = state.pipeline_state
        new = env._physics_step(env.base_sys, old, action)
        reward, observation, done = env._reward_obs(old, new, action)
        return state.replace(pipeline_state=new, obs=observation, reward=reward, done=done)

    return SimpleNamespace(
        kind=kind, env=env, obs_dim=env.obs_dim, act_dim=env.act_dim,
        gains=jnp.asarray([task["action_gain"] for task in env._tasks], dtype=jnp.float32),
        sys=env.base_sys, reset=env._reset_state, step=step, horizon=1000, dwell=250)


def cpu_sampling_pair(kind, conditioned, output, samples=4000, repeats=3):
    if (jax.__version__, brax.__version__) != ("0.4.38", "0.10.5"):
        raise ValueError("Compare sampling devices on JAX 0.4.38 / Brax 0.10.5")
    gpu, cpu = jax.devices("gpu")[0], jax.devices("cpu")[0]
    seed = 92101
    with jax.default_device(gpu):
        world = sampling_world(kind, seed)
        policy = GaussianPolicy(world.obs_dim + (4 if conditioned else 0), world.act_dim,
                                256, rngs=nnx.Rngs(seed))
        graph, params = nnx.graphdef(policy), nnx.state(policy, nnx.Param)
        key, reset_key = jax.random.split(jax.random.PRNGKey(seed))
        inputs = (params, world.reset(world.sys, reset_key), jnp.full(4, 0.25),
                  jnp.asarray(0), jax.random.split(key, samples),
                  jnp.asarray(run.training_modes(seed, samples, world.dwell)))
    scans, rows, reference = [], [], None
    for label, device in (("gpu", gpu), ("cpu", cpu)):
        print(f"SAMPLING_PAIR kind={kind} conditioned={conditioned} backend={label} phase=compile", flush=True)
        with jax.default_device(device):
            candidate_world = sampling_world(kind, seed)
            evidence = None
            if conditioned:
                model, saved, variance, unused = run.frozen_inverse()
                evidence = run.inverse_evidence(model, saved, candidate_world.gains, variance)
            scan = run.build_training_scan(candidate_world, graph, evidence, conditioned)
        placed = jax.device_put(inputs, device)
        jax.block_until_ready(placed)
        started = time.perf_counter()
        compiled = scan.lower(*placed).compile()
        compilation = time.perf_counter() - started
        result, first = timed_call(compiled, placed)
        if reference is None:
            reference = result
        comparison = compare_outputs(reference, result, conditioned)
        rows.append({"label": label, "compile_seconds": compilation, "first_call_seconds": first,
                     "warm_seconds": [], **comparison,
                     "terminal_steps": int(np.count_nonzero(np.asarray(result[1][4]))),
                     "transition_differences": transition_differences(reference, result)})
        scans.append((device, compiled, placed))
        print(f"SAMPLING_PAIR backend={label} first={first:.3f}s equivalent={comparison['equivalent']}", flush=True)
    for repetition in range(repeats):
        order = range(2) if repetition % 2 == 0 else reversed(range(2))
        for index in order:
            device, compiled, placed = scans[index]
            started = time.perf_counter()
            # A real iteration refreshes sampler actor parameters after GPU updates.
            refreshed = (jax.device_put(params, device), *placed[1:])
            result = compiled(*refreshed)
            jax.block_until_ready(jax.device_put(result[1], gpu))
            elapsed = time.perf_counter() - started
            rows[index]["warm_seconds"].append(elapsed)
            print(f"SAMPLING_PAIR backend={rows[index]['label']} repeat={repetition + 1} seconds={elapsed:.3f}", flush=True)
    baseline = statistics.median(rows[0]["warm_seconds"])
    for row in rows:
        row.update(median_seconds=statistics.median(row["warm_seconds"]))
        row["speedup"] = baseline / row["median_seconds"]
    payload = {"kind": kind, "conditioned": conditioned, "samples": samples,
               "repeats": repeats, "seed": seed, "device": gpu.device_kind,
               "jax_version": jax.__version__, "brax_version": brax.__version__,
               "training_updates": 0, "rows": rows,
               "scope": "Runtime screen with an untrained actor, not backend admission or a performance comparison"}
    run.write_json(output, payload)
    print(f"CPU_SAMPLING_PAIR_COMPLETE {output}", flush=True)
    return payload


def build_reacher_episode_scan(world, graph):
    """Native scalar physics; reset control flow only at episode boundaries."""
    @jax.jit
    def scan(params, state, belief, episode_step, keys, modes):
        horizon = world.horizon
        episodes = keys.shape[0] // horizon

        def episode(inputs):
            index, episode_keys, episode_modes = inputs
            prior_reset_key = jax.random.split(keys[index * horizon - 1], 3)[2]
            initial = jax.lax.cond(index == 0, lambda: state,
                                   lambda: world.reset(world.sys, prior_reset_key))

            def step(current, inputs):
                key, mode = inputs
                action_key, noise_key, unused = jax.random.split(key, 3)
                command = nnx.merge(graph, params).sample(current.obs[None], action_key)[0][0]
                executed = apply_action_disturbance(command, noise_key, world.gains[mode], 0.02)
                following = world.step(current, executed)
                return following, (current.obs, command, following.reward, following.obs, following.done)

            return jax.lax.scan(step, initial, (episode_keys, episode_modes))[1]

        transitions = jax.lax.map(episode, (
            jnp.arange(episodes), keys.reshape(episodes, horizon, -1), modes.reshape(episodes, horizon)))
        transitions = jax.tree.map(lambda value: value.reshape((-1, *value.shape[2:])), transitions)
        final = world.reset(world.sys, jax.random.split(keys[-1], 3)[2])
        return (final, jnp.full(4, 0.25), jnp.asarray(0)), transitions

    return scan


def sequential_trial(output, trajectory, samples, repeats):
    seed = 92101
    world = run.World("reacher", seed, physics_unroll=1)
    policy = GaussianPolicy(world.obs_dim, world.act_dim, 256, rngs=nnx.Rngs(seed))
    key, reset_key = jax.random.split(jax.random.PRNGKey(seed))
    inputs = (nnx.state(policy, nnx.Param), world.reset(world.sys, reset_key), jnp.full(4, 0.25),
              jnp.asarray(0), jax.random.split(key, samples),
              jnp.asarray(run.training_modes(seed, samples, world.dwell)))
    scan = run.build_training_scan(world, nnx.graphdef(policy), None, False)
    result, first = timed_call(scan, inputs)
    print(f"SEQUENTIAL_TRIAL first={first:.6f}s", flush=True)
    warm = []
    for repetition in range(repeats):
        warm.append(timed_call(scan, inputs)[1])
        print(f"SEQUENTIAL_TRIAL repeat={repetition + 1} seconds={warm[-1]:.6f}", flush=True)
    np.savez(trajectory, *[np.asarray(value) for value in jax.tree.leaves(result)])
    run.write_json(output, {"first_call_seconds": first, "warm_seconds": warm,
                           "median_seconds": statistics.median(warm),
                           "jax_version": jax.__version__, "brax_version": brax.__version__})
    print(f"SEQUENTIAL_TRIAL_COMPLETE median={statistics.median(warm):.6f}s", flush=True)


def command_buffer_pair(output, samples=4000, repeats=3):
    # Separate processes are necessary: this JAX version cannot override the
    # repeated command-buffer enum through per-function compiler_options.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="command-buffer-", dir=output.parent) as temporary:
        directory = Path(temporary)
        timings, traces = [], []
        for label in ("baseline", "command_buffer"):
            print(f"SEQUENTIAL_TRIAL_START {label} samples={samples}", flush=True)
            env = dict(os.environ)
            if label == "command_buffer":
                env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + (
                    " --xla_gpu_enable_command_buffer=FUSION,CUBLAS,WHILE,CONDITIONAL")
            timing, trajectory = directory / f"{label}.json", directory / f"{label}.npz"
            process = subprocess.run([
                sys.executable, "-u", "-m", "jax_experiments.analysis.benchmark_mechanism_gap_rollout", "--kind", "reacher",
                "--output", str(timing), "--trajectory", str(trajectory),
                "--samples", str(samples), "--repeats", str(repeats)], env=env)
            if process.returncode:
                run.write_json(output, {"kind": "reacher", "status": "unsupported",
                                       "exit_code": process.returncode, "training_updates": 0})
                return
            timings.append({"label": label, **json.loads(timing.read_text())})
            with np.load(trajectory) as archive:
                traces.append([archive[f"arr_{index}"] for index in range(len(archive.files))])
        comparison = compare_outputs(traces[0], traces[1], False)
        timings[1].update(comparison, speedup=timings[0]["median_seconds"] / timings[1]["median_seconds"])
        run.write_json(output, {"kind": "reacher", "samples": samples, "seed": 92101,
                               "training_updates": 0, "rows": timings})
        print(f"BENCHMARK_COMPLETE {output} {comparison}", flush=True)


def benchmark(kind, conditioned, output, samples=4000, repeats=3):
    if jax.default_backend() != "gpu":
        raise RuntimeError("This is a GPU timing benchmark; CPU timing is not representative")
    seed = 92101
    world = run.World(kind, seed, physics_unroll=1)
    policy = GaussianPolicy(world.obs_dim + (4 if conditioned else 0), world.act_dim,
                            256, rngs=nnx.Rngs(seed))
    graph, params = nnx.graphdef(policy), nnx.state(policy, nnx.Param)
    evidence = None
    if conditioned:
        model, saved, variance, unused = run.frozen_inverse()
        evidence = run.inverse_evidence(model, saved, world.gains, variance)
    key, reset_key = jax.random.split(jax.random.PRNGKey(seed))
    inputs = (params, world.reset(world.sys, reset_key), jnp.full(4, 0.25),
              jnp.asarray(0), jax.random.split(key, samples),
              jnp.asarray(run.training_modes(seed, samples, world.dwell)))
    jax.block_until_ready(inputs)
    inner_factor = 4 if kind == "hc" else 2
    settings = (("baseline", 1, 1), ("outer_unroll", 4, 1),
                ("physics_unroll", 1, inner_factor)) if kind == "hc" else (
                    ("baseline", 1, 1), ("episode_boundaries", 1, 1))
    scans, rows, reference = [], [], None
    for label, outer, inner in settings:
        device = jax.devices("gpu")[0]
        with jax.default_device(device):
            candidate_world = run.World(kind, seed, physics_unroll=inner)
            scan = (build_reacher_episode_scan(candidate_world, graph) if label == "episode_boundaries"
                    else run.build_training_scan(candidate_world, graph, evidence, conditioned, unroll=outer))
        placed_inputs = jax.device_put(inputs, device)
        jax.block_until_ready(placed_inputs)
        result, first_seconds = timed_call(scan, placed_inputs)
        if reference is None:
            reference = result
        comparison = compare_outputs(reference, result, conditioned)
        row = {"label": label, "execution_backend": device.platform,
               "outer_unroll": outer, "physics_unroll": inner,
               "first_call_seconds": first_seconds, "warm_seconds": [], **comparison,
               "transition_differences": transition_differences(reference, result)}
        rows.append(row)
        scans.append((scan, placed_inputs))
        print(f"{label}: first={first_seconds:.3f}s equivalent={comparison['equivalent']} "
              f"max_delta={comparison['max_absolute_difference']:.6g}", flush=True)
    # Alternate order to avoid giving one lowering all of the idle-GPU samples.
    for repetition in range(repeats):
        order = range(len(scans)) if repetition % 2 == 0 else reversed(range(len(scans)))
        for index in order:
            unused, seconds = timed_call(*scans[index])
            rows[index]["warm_seconds"].append(seconds)
            print(f"{rows[index]['label']}: repeat={repetition + 1} seconds={seconds:.3f}", flush=True)
    baseline = statistics.median(rows[0]["warm_seconds"])
    for row in rows:
        row["median_seconds"] = statistics.median(row["warm_seconds"])
        row["speedup"] = baseline / row["median_seconds"]
    payload = {"kind": kind, "conditioned": conditioned, "samples": samples,
               "repeats": repeats, "seed": seed, "device": jax.devices()[0].device_kind,
               "jax_version": jax.__version__, "brax_version": brax.__version__,
               "training_updates": 0, "rows": rows}
    run.write_json(output, payload)
    print(f"BENCHMARK_COMPLETE {output}", flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("hc", "reacher", "ant", "hopper", "walker"), required=True)
    parser.add_argument("--conditioned", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--command-buffer-pair", action="store_true")
    parser.add_argument("--cpu-sampling-pair", action="store_true")
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.conditioned and args.kind != "hc":
        parser.error("the frozen conditioned estimator is specific to HalfCheetah")
    if args.kind not in ("hc", "reacher") and not args.cpu_sampling_pair:
        parser.error("Ant/Hopper/Walker are supported by --cpu-sampling-pair")
    if args.cpu_sampling_pair:
        cpu_sampling_pair(args.kind, args.conditioned, args.output, args.samples, args.repeats)
    elif args.command_buffer_pair:
        command_buffer_pair(args.output, args.samples, args.repeats)
    elif args.trajectory:
        sequential_trial(args.output, args.trajectory, args.samples, args.repeats)
    else:
        benchmark(args.kind, args.conditioned, args.output, args.samples, args.repeats)


if __name__ == "__main__":
    main()
