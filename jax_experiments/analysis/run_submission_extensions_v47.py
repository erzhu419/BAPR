"""Independent model-seed audits and the frozen non-diagonal actuator study."""
import argparse
from functools import partial
from importlib.metadata import version
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.analysis import submission_extensions_protocol_v47 as p
from jax_experiments.analysis import run_mechanism_gap_v34 as common
from jax_experiments.analysis import run_reacher_causal_v35 as source
from jax_experiments.analysis import reacher_information_confirmation_protocol_v38 as confirmation

read = lambda path: json.loads(path.read_text())


def check_runtime():
    actual = {name: version(name) for name in p.RUNTIME}
    if actual != p.RUNTIME:
        raise ValueError(f"extension runtime differs: {actual}")
    return actual


def configure_permutation():
    common.World = partial(common.World, transforms=p.TRANSFORMS)
    source.p = SimpleNamespace(
        RUNTIME=p.RUNTIME, PROVIDER_SEED=p.PROVIDER_SEED, POLICY_SEEDS=p.PERMUTATION_SEEDS,
        PROVIDER_BUNDLE=p.actor("permutation", "reference", p.PROVIDER_SEED),
        DATA_DIR=p.RUN_ROOT / "permutation/dataset",
        DATA_MANIFEST=p.RUN_ROOT / "permutation/dataset/complete.json",
        TRAIN_ROWS=178000, VALIDATION_ROWS=82000, TRAIN_EVENT_SEED=p.TRAIN_EVENT_SEED,
        VALIDATION_EVENT_SEED=p.VALIDATION_EVENT_SEED,
        MODEL_UPDATES=4500, BATCH_SIZE=256, CONTROLS=p.CONTROLS,
        model_dir=p.model_dir,
        checkpoint_dir=lambda seed: p.RUN_ROOT / "permutation/estimators" / f"seed_{seed}/checkpoints")


def evidence(model, gains, suite):
    return source.load_evidence(model, gains)


def hc_evidence(seed, gains):
    directory = p.hc_model_dir(seed)
    metadata = read(directory / "complete.json")
    model = common.inverse_model(17, 6, seed)
    template = nnx.state(model, nnx.Param)
    leaves, tree = jax.tree.flatten(template)
    with np.load(directory / "inverse_params.npz") as archive:
        saved = [jnp.asarray(archive[f"arr_{index}"]) for index in range(len(leaves))]
    if [value.shape for value in leaves] != [value.shape for value in saved]:
        raise ValueError("historical inverse model parameter layout differs")
    return common.inverse_evidence(model, jax.tree.unflatten(tree, saved), gains,
                                   jnp.asarray(metadata["residual_variance"]))


def train(suite, arm, seed, model, smoke=False):
    root = p.RUN_ROOT / suite
    bundle = p.BUNDLE_ROOT / suite
    if suite == "estimator":
        if arm != "conditioned" or model not in p.REACHER_MODELS[1:]:
            raise ValueError("only the two missing matched estimator-conditioned arms are trained")
        root, bundle = root / f"model_{model}", bundle / f"model_{model}"
    if smoke:
        root, bundle = p.RUN_ROOT / "smoke" / suite / str(model), p.BUNDLE_ROOT / "smoke" / suite / str(model)
    conditioned = None
    if arm == "conditioned":
        with jax.default_device(jax.devices("cpu")[0]):
            world = common.World("reacher", seed)
            conditioned = (evidence(model, world.gains, suite), p.CONTROLS, True)
    common.train_policy("reacher", arm, seed, sampling_device="cpu", run_root=root,
                        bundle_root=bundle, protocol_name=f"{p.PROTOCOL}/{suite}/model-{model}",
                        conditioning=conditioned, smoke=smoke)


def load_reacher(suite, arm, seed, model=None):
    if suite == "estimator" and (arm != "conditioned" or model == p.REACHER_MODELS[0]):
        directory = confirmation.screen_bundle(arm, seed)
    else:
        directory = p.actor(suite, arm, seed, model if suite == "estimator" else None)
    graph, params, metadata = source.load_actor(directory)
    if metadata["arm"] != arm or metadata["seed"] != seed:
        raise ValueError("controller identity differs from the audit")
    return graph, params, metadata


def stationary(world, actors, routes):
    rows = {}
    for route, actor, scores, temporal in routes:
        graph, params = actors[actor][:2]
        episode = common.build_episode(world, graph, route, scores, temporal)
        for mode in range(4):
            for index in range(5):
                key = jax.random.PRNGKey(p.STATIONARY_EVENT_SEED * 100 + mode * 5 + index)
                values = episode(params, key, jnp.full(4, mode))
                row = common.trajectory_metrics(values, world.dwell, route)
                row.update(mode=mode, episode=index)
                rows.setdefault(route, []).append(row)
    return rows


def audit(suite, seed):
    output = p.audit_path(suite, seed)
    if output.exists():
        return
    kind = "hc" if suite == "hc-estimator" else "reacher"
    world = common.World(kind, seed)
    actors = {"reference": (common.load_frozen_actor(seed) if kind == "hc"
                            else load_reacher(suite, "reference", seed))}
    routes = [("oracle", "reference", None, True), ("native_reference", "reference", None, True),
              ("uncompensated", "reference", None, True)]
    models = p.HC_MODELS if kind == "hc" else p.REACHER_MODELS if suite == "estimator" else p.PERMUTATION_MODELS
    if kind != "hc":
        actors["robust"] = load_reacher(suite, "robust", seed)
        routes.append(("robust", "robust", None, True))
    for model in models:
        scores = (hc_evidence(model, world.gains) if kind == "hc"
                  else evidence(model, world.gains, suite))
        routes.append((f"causal_{model}", "reference", scores, True))
        if suite == "estimator" or (suite == "permutation" and model == models[0]):
            actor = f"conditioned_{model}"
            actors[actor] = load_reacher(suite, "conditioned", seed, model)
            # The route name controls augmented observation, not the actor key.
            routes.append((f"conditioned_{model}", actor, scores, True))
    if suite == "permutation":
        routes.extend((f"fixed_{mode}", "reference", None, True) for mode in range(4))
    events = common.evaluate_routes(world, actors, routes, event_seeds=p.EVENT_SEEDS, episodes=5)
    stationary_rows = stationary(world, actors, routes)
    oracle = np.asarray([row["return"] for row in events["oracle"]])
    native = np.asarray([row["return"] for row in events["native_reference"]])
    if not np.allclose(oracle, native, atol=1e-3, rtol=0):
        raise ValueError("paired true-mode compensation/native equivalence failed")
    if any(row["steps"] != world.horizon or row["switches_observed"] != 3 or row["termination"]
           for episodes in events.values() for row in episodes):
        raise ValueError("evaluation lost its full switching windows")
    common.write_json(output, {"protocol": p.PROTOCOL, "suite": suite, "seed": seed,
        "runtime_versions": check_runtime(), "events": events, "stationary": stationary_rows,
        "transforms": np.asarray(world.gains).tolist(), "estimator_seeds": list(models),
        "budgets": {arm: actor[2] for arm, actor in actors.items() if len(actor) == 3},
        "oracle_native_max_return_error": float(np.max(np.abs(oracle-native))),
        "means": {arm: float(np.mean([row["return"] for row in episodes])) for arm, episodes in events.items()}})


def analyze(suite):
    seeds = p.HC_SEEDS if suite == "hc-estimator" else p.REACHER_SEEDS if suite == "estimator" else p.PERMUTATION_SEEDS
    rows = [read(p.audit_path(suite, seed)) for seed in seeds]
    if any(row["seed"] != seed or row["suite"] != suite or row["protocol"] != p.PROTOCOL
           or row["runtime_versions"] != p.RUNTIME for seed, row in zip(seeds, rows, strict=True)):
        raise ValueError("mixed extension audits")
    values = {arm: [row["means"][arm] for row in rows] for arm in rows[0]["means"]}
    models = rows[0]["estimator_seeds"]
    comparisons = {}
    for model in models:
        for baseline in ("oracle", "uncompensated", "robust", f"conditioned_{model}"):
            if baseline in values:
                comparisons[f"causal_{model}-minus-{baseline}"] = common.paired_summary(values[f"causal_{model}"], values[baseline])
    if suite == "permutation":
        for mode in range(4):
            comparisons[f"oracle-minus-fixed_{mode}"] = common.paired_summary(values["oracle"], values[f"fixed_{mode}"])
    common.write_json(p.analysis_path(suite), {"protocol": p.PROTOCOL, "suite": suite,
        "policy_seeds": list(seeds), "estimator_seeds": models, "seed_means": values,
        "means": {arm: float(np.mean(row)) for arm, row in values.items()}, "comparisons": comparisons,
        "historical_cohorts_pooled": False,
        "inference_unit": "policy seeds; estimator seeds are a crossed factor, not independent episodes",
        "scope": ("retrospective independent calibration audit; not the original two-stage estimator recipe"
                  if suite == "hc-estimator" else "frozen-estimator-seed robustness"
                  if suite == "estimator" else "predefined actuator-coordinate permutation extension")})


def smoke(suite):
    if suite == "permutation":
        result = common.train_policy("reacher", "reference", p.PROVIDER_SEED, smoke=True,
            sampling_device="cpu", run_root=p.RUN_ROOT / "smoke/permutation", bundle_root=p.BUNDLE_ROOT / "smoke/permutation")
        cpu = jax.devices("cpu")[0]
        with jax.default_device(cpu):
            world = common.World("reacher", p.PROVIDER_SEED)
            agent = common.SACBase(11, 2, common.config("reacher", p.PROVIDER_SEED), p.PROVIDER_SEED)
            graph, params = nnx.graphdef(agent.policy), nnx.state(agent.policy, nnx.Param)
            key, schedule = jax.random.PRNGKey(101), jnp.asarray([0, 1, 3, 2])
            oracle = common.build_episode(world, graph, "oracle")(params, key, schedule)
            native = common.build_episode(world, graph, "native_reference")(params, key, schedule)
            if not np.array_equal(np.asarray(oracle[-1]), np.asarray(native[-1])):
                raise ValueError("matrix compensation changed paired observations")
    else:
        result = {"models": list(p.REACHER_MODELS)}
        for model in p.REACHER_MODELS:
            with jax.default_device(jax.devices("cpu")[0]):
                world = common.World("reacher", p.REACHER_SEEDS[0])
                scores = evidence(model, world.gains, suite)
                if not np.isfinite(np.asarray(scores(jnp.zeros(11), jnp.zeros(2), jnp.zeros(11)))).all():
                    raise ValueError("nonfinite independent estimator scores")
        for model in p.REACHER_MODELS[1:]:
            train(suite, "conditioned", p.REACHER_SEEDS[0], model, smoke=True)
    common.write_json(p.smoke(suite), {"passed": True, "protocol": p.PROTOCOL,
                                     "runtime_versions": check_runtime(), "result": result})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "train", "collect", "fit", "audit", "analyze"))
    parser.add_argument("--suite", choices=("estimator", "hc-estimator", "permutation"), required=True)
    parser.add_argument("--arm", choices=("reference", "robust", "conditioned"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model", type=int)
    args = parser.parse_args()
    check_runtime()
    if args.suite == "permutation":
        configure_permutation()
    if args.mode in ("train", "fit", "smoke") and jax.default_backend() != "gpu":
        raise ValueError("learning/smoke requires the GPU scheduler pool")
    if args.mode == "train":
        train(args.suite, args.arm, args.seed, args.model)
    elif args.mode == "collect":
        source.collect_data()
    elif args.mode == "fit":
        # Recreate the same prescribed rows on the learner node, not on local disk.
        source.p.DATA_DIR = p.dataset_dir(args.model)
        source.p.DATA_MANIFEST = source.p.DATA_DIR / "complete.json"
        if not all((source.p.DATA_DIR / f"{split}.npz").exists() for split in ("train", "validation")):
            source.p.DATA_MANIFEST.unlink(missing_ok=True)
        source.collect_data()
        source.fit(args.model)
    elif args.mode == "audit":
        audit(args.suite, args.seed)
    elif args.mode == "analyze":
        analyze(args.suite)
    else:
        smoke(args.suite)
    print(f"EXTENSION_COMPLETE suite={args.suite} mode={args.mode}", flush=True)


if __name__ == "__main__":
    main()
