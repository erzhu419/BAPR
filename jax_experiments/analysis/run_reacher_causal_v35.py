"""Independent Reacher calibration and causal compensation; compact exports only."""
from __future__ import annotations

import argparse
import json
import time
from functools import partial
from importlib.metadata import version

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from jax_experiments.analysis import reacher_causal_protocol_v35 as p
from jax_experiments.analysis import run_mechanism_gap_v34 as common
from jax_experiments.networks.policy import GaussianPolicy


def require_runtime():
    actual = {name: version(name) for name in p.RUNTIME}
    if actual != p.RUNTIME:
        raise ValueError(f"Reacher runtime differs from the admitted group: {actual}")
    return actual


def load_actor(directory):
    metadata = json.loads((directory / "bundle.json").read_text())
    expected = (common.p.REACHER_ITERS, common.p.REACHER_ITERS * common.p.SAMPLES,
                common.p.REACHER_ITERS * common.p.UPDATES)
    if (metadata["next_iteration"], metadata["total_steps"], metadata["update_count"]) != expected:
        raise ValueError("incomplete Reacher actor")
    if metadata["sampling_device"] != "cpu" or metadata["runtime_versions"] != p.RUNTIME:
        raise ValueError("actor does not belong to the matched CPU-sampling group")
    model = GaussianPolicy(metadata["policy_input_dim"], metadata["act_dim"], 256,
                           rngs=nnx.Rngs(metadata["seed"]))
    params = common._restore_tree_like(
        nnx.state(model, nnx.Param), common.read_pickle(directory / "policy_params.pkl"),
        "Reacher actor", allow_fallback=False)
    return nnx.graphdef(model), params, metadata


def provider():
    common.p.RUN_ROOT = p.PROVIDER_RUN_ROOT
    common.p.BUNDLE_ROOT = p.PROVIDER_BUNDLE_ROOT
    common.train_policy("reacher", "reference", p.PROVIDER_SEED, sampling_device="cpu")


def build_collection(world, graph, params):
    @partial(jax.jit, static_argnums=1)
    def collect(key, count):
        reset_key, rollout_key = jax.random.split(key)

        def step(state, pair):
            index, step_key = pair
            explore_key, mix_key, noise_key, reset_key = jax.random.split(step_key, 4)
            mode = (index // world.dwell) % 4
            nominal = nnx.merge(graph, params).deterministic(state.obs[None])[0]
            compensated = common.compensated_command(nominal, world.gains, mode)
            exploratory = jnp.clip(compensated + 0.15 * jax.random.normal(explore_key, nominal.shape), -1, 1)
            command = jnp.where(jax.random.uniform(mix_key) < 0.2,
                                jax.random.uniform(explore_key, nominal.shape, minval=-1, maxval=1), exploratory)
            executed = common.apply_action_disturbance(command, noise_key, world.gains[mode], 0.02)
            following = world.step(state, executed)
            restart = (following.done > 0.5) | ((index + 1) % world.horizon == 0)
            outgoing = jax.lax.cond(restart, lambda: world.reset(world.sys, reset_key), lambda: following)
            # Training labels are expected actuator signals, not sampled noise.
            return outgoing, (state.obs, following.obs, command,
                              common.nominal_signal(command, world.gains[mode]), mode)

        return jax.lax.scan(step, world.reset(world.sys, reset_key),
                            (jnp.arange(count), jax.random.split(rollout_key, count)))[1]

    return collect


def collect_data():
    if p.DATA_MANIFEST.is_file():
        return
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        world = common.World("reacher", p.PROVIDER_SEED)
        graph, params, unused = load_actor(p.PROVIDER_BUNDLE)
        collect = build_collection(world, graph, params)
    p.DATA_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for name, count, seed in (("train", p.TRAIN_ROWS, p.TRAIN_EVENT_SEED),
                               ("validation", p.VALIDATION_ROWS, p.VALIDATION_EVENT_SEED)):
        print(f"COLLECT split={name} rows={count} backend=cpu", flush=True)
        with jax.default_device(cpu):
            values = collect(jax.random.PRNGKey(seed), count)
        data = dict(zip(("obs", "next_obs", "action", "target", "mode"), map(np.asarray, values), strict=True))
        if not all(np.isfinite(value).all() for value in data.values()):
            raise ValueError("nonfinite calibration transitions")
        np.savez_compressed(p.DATA_DIR / f"{name}.npz", **data)
        print(f"COLLECT_COMPLETE split={name} seconds={time.perf_counter()-started:.1f}", flush=True)
    common.write_json(p.DATA_MANIFEST, {
        "provider_seed": p.PROVIDER_SEED, "excluded_policy_seeds": list(p.POLICY_SEEDS),
        "train_rows": p.TRAIN_ROWS, "validation_rows": p.VALIDATION_ROWS,
        "obs_dim": world.obs_dim, "act_dim": world.act_dim, "horizon": world.horizon,
        "dwell": world.dwell, "noise_std": 0.02, "runtime_versions": require_runtime(),
        "calibration_interactions": p.TRAIN_ROWS + p.VALIDATION_ROWS,
        "elapsed_seconds": time.perf_counter() - started,
    })


def training_update(graph, optimizer):
    @jax.jit
    def update(params, opt_state, obs, next_obs, target):
        def loss(candidate):
            predictions = nnx.merge(graph, candidate).predict(obs, next_obs)
            return jnp.mean(jnp.square(predictions - target[None]))
        value, gradient = jax.value_and_grad(loss)(params)
        change, opt_state = optimizer.update(gradient, opt_state, params)
        return optax.apply_updates(params, change), opt_state, value
    return update


def calibration(params, graph, data):
    residuals = []
    for begin in range(0, len(data["obs"]), 1024):
        predictions = nnx.merge(graph, params).predict(
            jnp.asarray(data["obs"][begin:begin+1024]), jnp.asarray(data["next_obs"][begin:begin+1024]))
        residuals.append(np.asarray(predictions).mean(axis=0) - data["target"][begin:begin+1024])
    squared = np.mean(np.concatenate(residuals) ** 2, axis=0)
    return squared, np.clip(squared, 1e-4, 0.25)


def export_model(directory, params, metadata):
    directory.mkdir(parents=True, exist_ok=True)
    leaves = jax.tree.leaves(params)
    np.savez_compressed(directory / "inverse_params.npz", **{
        f"arr_{index}": np.asarray(value) for index, value in enumerate(leaves)})
    common.write_json(directory / "complete.json", metadata)


def fit(seed):
    directory = p.model_dir(seed)
    if (directory / "complete.json").is_file():
        return
    metadata = json.loads(p.DATA_MANIFEST.read_text())
    with np.load(p.DATA_DIR / "train.npz") as archive:
        train = dict(archive)
    with np.load(p.DATA_DIR / "validation.npz") as archive:
        validation = dict(archive)
    model = common.inverse_model(metadata["obs_dim"], metadata["act_dim"], seed)
    graph, params = nnx.graphdef(model), nnx.state(model, nnx.Param)
    optimizer = optax.chain(optax.clip_by_global_norm(10), optax.adamw(1e-4, weight_decay=1e-5))
    opt_state = optimizer.init(params)
    update = training_update(graph, optimizer)
    checkpoint = p.checkpoint_dir(seed) / "train_state.pkl"
    start = 0
    if checkpoint.is_file():
        saved = common.read_pickle(checkpoint)
        if saved["seed"] != seed or saved["provider_seed"] != p.PROVIDER_SEED:
            raise ValueError("estimator checkpoint belongs to another calibration run")
        params, opt_state = common._restore_tree_like((params, opt_state), saved["state"],
                                                     "Reacher estimator", allow_fallback=False)
        start = saved["next_iteration"]
        print(f"RESUME estimator update={start}/{p.MODEL_UPDATES}", flush=True)
    started = time.perf_counter()
    loss = None
    for iteration in range(start, p.MODEL_UPDATES):
        indices = np.random.default_rng(np.random.SeedSequence([seed, iteration])).integers(
            len(train["obs"]), size=p.BATCH_SIZE)
        params, opt_state, loss = update(params, opt_state, *[
            jnp.asarray(train[name][indices]) for name in ("obs", "next_obs", "target")])
        if not np.isfinite(float(loss)):
            raise ValueError(f"nonfinite inverse loss at update {iteration}")
        if (iteration + 1) % 500 == 0 or iteration + 1 == p.MODEL_UPDATES:
            common.save_pickle(checkpoint, {
                "seed": seed, "provider_seed": p.PROVIDER_SEED, "next_iteration": iteration + 1,
                "iteration": iteration, "total_steps": iteration + 1,
                "state": common._to_numpy_tree((params, opt_state))})
            print(f"Iter {iteration+1}/{p.MODEL_UPDATES} | Time: {time.perf_counter()-started:.1f}s | loss={float(loss):.6f}", flush=True)
    mse, variance = calibration(params, graph, validation)
    if not np.isfinite(mse).all():
        raise ValueError("nonfinite inverse calibration")
    export_model(directory, params, {
        **metadata, "model_seed": seed, "updates": p.MODEL_UPDATES, "batch_size": p.BATCH_SIZE,
        "inverse_heads": 5, "parameters": sum(value.size for value in jax.tree.leaves(params)),
        "validation_mse": mse.tolist(), "residual_variance": variance.tolist(),
        "filter_controls": list(p.CONTROLS), "runtime_versions": require_runtime(),
        "elapsed_this_attempt_seconds": time.perf_counter() - started,
    })


def load_evidence(seed, gains, single_head=False):
    directory = p.model_dir(seed)
    metadata = json.loads((directory / "complete.json").read_text())
    if metadata["runtime_versions"] != p.RUNTIME or metadata["filter_controls"] != list(p.CONTROLS):
        raise ValueError("estimator runtime or filter protocol differs")
    model = common.inverse_model(metadata["obs_dim"], metadata["act_dim"], seed)
    template = nnx.state(model, nnx.Param)
    leaves, tree = jax.tree.flatten(template)
    with np.load(directory / "inverse_params.npz") as archive:
        saved = [jnp.asarray(archive[f"arr_{index}"]) for index in range(len(leaves))]
    if [value.shape for value in leaves] != [value.shape for value in saved]:
        raise ValueError("estimator parameter layout differs")
    return common.inverse_evidence(model, jax.tree.unflatten(tree, saved), gains,
                                   jnp.asarray(metadata["residual_variance"]), single_head)


def audit(seed, *, protocol=p, conditioned_arms=()):
    output = protocol.audit_path(seed)
    if output.is_file():
        return
    world = common.World("reacher", seed)
    actors = {arm: load_actor(protocol.screen_bundle(arm, seed))
              for arm in ("reference", "robust", *conditioned_arms)}
    routes = [("oracle", "reference", None, True), ("native_reference", "reference", None, True),
              ("robust", "robust", None, True), ("uncompensated", "reference", None, True)]
    routes.extend((f"fixed_{mode}", "reference", None, True) for mode in range(4))
    routes.extend((f"causal_{model}", "reference", load_evidence(model, world.gains), True) for model in protocol.MODEL_SEEDS)
    routes.extend((("no_filter", "reference", load_evidence(protocol.PRIMARY_MODEL, world.gains), False),
                   ("single_head", "reference", load_evidence(protocol.PRIMARY_MODEL, world.gains, True), True)))
    routes.extend((arm, arm, load_evidence(protocol.PRIMARY_MODEL, world.gains), arm != "conditioned_no_filter")
                  for arm in conditioned_arms)
    events = common.evaluate_routes(world, actors, routes, event_seeds=protocol.EVENT_SEEDS, episodes=protocol.EPISODES)
    differences = [a["return"] - b["return"] for a, b in zip(events["oracle"], events["native_reference"], strict=True)]
    if not np.allclose(differences, 0, atol=1e-3):
        raise ValueError("oracle/native return equivalence changed")
    stationary = {}
    for route, actor, evidence, temporal in routes:
        if (route not in ("oracle", "native_reference", "robust", *conditioned_arms)
                and not route.startswith("causal_") and not (conditioned_arms and route == "no_filter")):
            continue
        graph, params = actors[actor][:2]
        episode = common.build_episode(world, graph, route, evidence, temporal)
        for mode in range(4):
            for index in range(protocol.EPISODES):
                values = episode(params, jax.random.PRNGKey(protocol.STATIONARY_EVENT_SEED + 100 * mode + index),
                                 jnp.full(4, mode, jnp.int32))
                row = common.trajectory_metrics(values, world.dwell, route)
                row.update(mode=mode, episode=index)
                stationary.setdefault(route, []).append(row)
    common.write_json(output, {
        "seed": seed, "primary_model_seed": protocol.PRIMARY_MODEL, "runtime_versions": require_runtime(),
        "events": events, "stationary": stationary,
        "means": {name: float(np.mean([e["return"] for e in rows])) for name, rows in events.items()},
        "budgets": {name: value[2] for name, value in actors.items()},
        "estimator_cost": {str(s): json.loads((p.model_dir(s) / "complete.json").read_text()) for s in protocol.MODEL_SEEDS},
        "oracle_native_max_abs_return_difference": float(np.max(np.abs(differences))),
    })


def analyze(*, protocol=p):
    rows = [json.loads(protocol.audit_path(seed).read_text()) for seed in protocol.POLICY_SEEDS]
    primary = f"causal_{protocol.PRIMARY_MODEL}"
    means = {arm: [row["means"][arm] for row in rows] for arm in rows[0]["means"]}
    comparisons = {f"{primary}-minus-{arm}": common.paired_summary(means[primary], values)
                   for arm, values in means.items() if arm != primary}
    headroom = np.asarray(means["oracle"]) - np.asarray(means["robust"])
    gains = np.asarray(means[primary]) - np.asarray(means["robust"])
    valid = all(e["steps"] == common.p.REACHER_HORIZON and e["switches_observed"] == 3
                for row in rows for arm in (primary, "oracle", "robust") for e in row["events"][arm])
    stationary_means = {arm: float(np.mean([e["return"] for row in rows for e in row["stationary"][arm]]))
                        for arm in rows[0]["stationary"]}
    positive = comparisons[f"{primary}-minus-robust"]["ci95"][0] > 0
    fixed = all(comparisons[f"{primary}-minus-fixed_{mode}"]["mean"] > 0 for mode in range(4))
    common.write_json(protocol.ANALYSIS, {
        "policy_seeds": list(protocol.POLICY_SEEDS), "primary_model_seed": protocol.PRIMARY_MODEL,
        "means": {arm: float(np.mean(values)) for arm, values in means.items()}, "seed_means": means,
        "comparisons": comparisons, "stationary_means": stationary_means,
        "headroom_recovery_by_seed": [float(g / h) if h > 0 else None for g, h in zip(gains, headroom, strict=True)],
        "causal_screen": {"positive_paired_ci_vs_robust": positive, "beats_all_fixed_means": fixed,
                          "valid_switch_windows": valid, "passes": bool(positive and fixed and valid)},
        "inference_unit": f"policy seed (n={len(protocol.POLICY_SEEDS)}); estimator repeats are not extra policy seeds",
    })


def smoke():
    if jax.default_backend() != "gpu":
        raise ValueError("estimator training smoke requires GPU")
    world = common.World("reacher", p.PROVIDER_SEED)
    model = common.inverse_model(world.obs_dim, world.act_dim, p.PRIMARY_MODEL)
    graph, params = nnx.graphdef(model), nnx.state(model, nnx.Param)
    optimizer = optax.adamw(1e-4)
    opt_state = optimizer.init(params)
    update = training_update(graph, optimizer)
    obs = jax.random.normal(jax.random.PRNGKey(7), (p.BATCH_SIZE, world.obs_dim))
    target = jnp.zeros((p.BATCH_SIZE, world.act_dim))
    for unused in range(2):
        params, opt_state, loss = update(params, opt_state, obs, obs + 0.01, target)
    if not np.isfinite(float(loss)):
        raise ValueError("nonfinite estimator smoke")
    common.write_json(p.SMOKE, {"passed": True, "runtime_versions": require_runtime(),
                               "obs_dim": world.obs_dim, "act_dim": world.act_dim,
                               "model_updates": 2, "loss": float(loss)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "provider", "collect", "fit", "audit", "analyze"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    require_runtime()
    if args.mode in ("smoke", "provider", "fit") and jax.default_backend() != "gpu":
        raise ValueError("training requires a GPU learner")
    if args.mode == "fit":
        if args.seed not in p.MODEL_SEEDS:
            parser.error("fit seed must be preregistered")
        fit(args.seed)
    elif args.mode == "audit":
        if args.seed not in p.POLICY_SEEDS:
            parser.error("audit seed must be preregistered")
        audit(args.seed)
    else:
        {"smoke": smoke, "provider": provider, "collect": collect_data, "analyze": analyze}[args.mode]()
    print(f"REACHER_CAUSAL_COMPLETE mode={args.mode}", flush=True)


if __name__ == "__main__":
    main()
