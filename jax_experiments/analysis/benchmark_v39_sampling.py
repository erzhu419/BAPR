"""Time real trained-policy collection and check identical trajectories; JSON only."""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.analysis import regularized_comparison_protocol_v39 as p
from jax_experiments.analysis import run_regularized_comparison_v39 as run
from jax_experiments.common.checkpoint import _patch_flax_variablestate_unpickle
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.train import collect_samples, make_env


def compare(reference, candidate):
    first, tree = jax.tree.flatten(reference)
    second, other = jax.tree.flatten(candidate)
    if tree != other:
        return {"equivalent": False, "reason": "different rollout structure"}
    maximum, exact, equivalent = 0.0, True, True
    for a, b in zip(first, second, strict=True):
        a, b = np.asarray(a), np.asarray(b)
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            return {"equivalent": False, "reason": "nonfinite trajectory"}
        exact &= bool(np.array_equal(a, b))
        equivalent &= bool(np.allclose(a, b, atol=1e-5, rtol=1e-5))
        if a.size:
            maximum = max(maximum, float(np.max(np.abs(a.astype(float) - b.astype(float)))))
    return {"equivalent": equivalent, "exact": exact, "max_absolute_difference": maximum}


def build_env(config, agent, label):
    env = make_env(config)
    env.set_nonstationary_para(env.sample_tasks(4), config.changing_period, config.changing_interval)
    native = env.env
    unroll = True if label.endswith("unroll16") else 1

    def physics_step(sys, state, action):
        def frame(current, unused):
            return native._pipeline.step(sys, current, action, native._debug), None
        return jax.lax.scan(frame, state, (), native._n_frames, unroll=unroll)[0]

    env._physics_step = physics_step
    if getattr(agent, "uses_recurrent_context", False):
        graph, unused, non_params = nnx.split(agent.context_net, nnx.Param, ...)
        env.build_rollout_fn(nnx.graphdef(agent.policy), recurrent_context_graphdef=graph,
                             recurrent_context_non_params=non_params)
    else:
        env.build_rollout_fn(nnx.graphdef(agent.policy))
    if label == "baseline":
        env.rollout_mode_chunks = None
    return env


def snapshot_clock(env):
    return {"rng": env._mode_rng.get_state(), "mode": env.current_task_id,
            "steps": env._step_counter, "next": env._next_switch_step,
            "history": env.get_switch_history()}


def restore_clock(env, snapshot):
    env._mode_rng.set_state(snapshot["rng"])
    env._activate_mode(snapshot["mode"])
    env._step_counter, env._next_switch_step = snapshot["steps"], snapshot["next"]
    env._switch_history = list(snapshot["history"])


def benchmark(arm, seed, output, repeats=3):
    p.check_runtime()
    if jax.default_backend() != "gpu":
        raise ValueError("production collection timing requires the GPU runtime")
    _patch_flax_variablestate_unpickle()
    config, agent = run.load_agent(arm, seed)
    replay = ReplayBuffer(agent.obs_dim, agent.act_dim, capacity=config.replay_size)
    jax.block_until_ready(replay.obs)
    settings = ("baseline", "fused_chunks", "fused_chunks_unroll16")
    rows, calls, reference = [], [], None
    for label in settings:
        env = build_env(config, agent, label)
        clock = snapshot_clock(env)

        def call(env=env, clock=clock):
            restore_clock(env, clock)
            replay.size, replay.ptr = 4000, 4000
            started = time.perf_counter()
            rewards, recent = collect_samples(agent, env, replay, config, 4000, current_iter=2100)
            jax.block_until_ready((recent, env._state, replay.obs, replay.episode_start))
            elapsed = time.perf_counter() - started
            return (rewards, recent, env._state, env.get_switch_history()), elapsed

        result, first = call()
        if label != "baseline" and not hasattr(env, "_mode_chunk_scan"):
            raise RuntimeError("staged collector did not execute fused mode blocks")
        if reference is None:
            reference = jax.device_get(result)
        row = {"label": label, "first_call_seconds": first, "warm_seconds": [],
               **compare(reference, result)}
        print(f"COLLECTION {arm} {label} first={first:.3f}s {row}", flush=True)
        rows.append(row)
        calls.append(call)
    for repetition in range(repeats):
        order = range(len(calls)) if repetition % 2 == 0 else reversed(range(len(calls)))
        for index in order:
            result, seconds = calls[index]()
            check = compare(reference, result)
            rows[index]["equivalent"] &= check["equivalent"]
            rows[index]["exact"] &= check.get("exact", False)
            rows[index]["warm_seconds"].append(seconds)
            print(f"COLLECTION {arm} {settings[index]} repeat={repetition+1} seconds={seconds:.3f}", flush=True)
    baseline = statistics.median(rows[0]["warm_seconds"])
    for row in rows:
        row.update(median_seconds=statistics.median(row["warm_seconds"]))
        row["speedup"] = baseline / row["median_seconds"]
    run.write(output, {"arm": arm, "seed": seed, "samples": 4000, "repeats": repeats,
                      "runtime_versions": p.check_runtime(), "device": jax.devices()[0].device_kind,
                      "training_updates": 0, "rows": rows,
                      "scope": "Same trained actor, mode clock, random keys, replay write and final simulator state."})
    print(f"COLLECTION_BENCHMARK_COMPLETE {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("sac", "escp", "resac_full"), required=True)
    parser.add_argument("--seed", type=int, choices=p.SEEDS, default=p.SEEDS[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    benchmark(args.arm, args.seed, args.output, args.repeats)


if __name__ == "__main__":
    main()
