"""Train and strictly evaluate the frozen Torch-equation alignment contrast."""
import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys

import jax
import numpy as np
from flax import nnx
from scipy.stats import t

from jax_experiments.analysis import resac_torch_alignment_protocol_v41 as p
from jax_experiments.analysis import run_resac_norm_causality_v40 as old
from jax_experiments.analysis.check_resac_torch_parity_v41 import check as check_parity
from jax_experiments.common.checkpoint import _patch_flax_variablestate_unpickle, _to_numpy_tree
from jax_experiments.train import make_algo, make_env

read, write = old.read, old.write


def training_command(arm, seed):
    directory = p.run_dir(arm, seed)
    command = old.training_command("per_head", p.old.SEEDS[0])
    for name, value in {**p.settings(arm), "seed": seed, "save_root": directory.parent,
                        "run_name": directory.name}.items():
        if name in {"resac_adaptive_beta", "use_ema_eval", "use_ema_rollout"}:
            continue
        old.previous.set_option(command, name, value)
    return command


def publish(arm, seed):
    directory, destination = p.run_dir(arm, seed), p.bundle_dir(arm, seed)
    config, agent, budget = old.checkpoint_agent(directory)
    signature = read(directory / "logs/protocol_signature.json")
    actor_updates = float(np.load(directory / "logs/actor_update_rate.npy").sum()) * config.updates_per_iter
    metadata = {"identity": p.identity(arm, seed), "budget": budget,
        "actor_update_count": int(actor_updates), "runtime_versions": p.check_runtime(),
        "norm_definition": p.NORM_DEFINITION, "config": signature["config"],
        "host": signature["host"], "python": sys.executable}
    if actor_updates != metadata["actor_update_count"]:
        raise ValueError("fractional actor update counter")
    p.validate(metadata, arm, seed)
    parameters = {"policy": _to_numpy_tree(nnx.state(agent.policy, nnx.Param))}
    if not all(np.isfinite(value).all() for value in jax.tree.leaves(parameters)):
        raise ValueError("nonfinite final policy")
    diagnostics = old.diagnostics(directory, config, agent)
    (destination / "logs").mkdir(parents=True, exist_ok=True)
    with (destination / "parameters.pkl").open("wb") as stream:
        pickle.dump(parameters, stream, protocol=pickle.HIGHEST_PROTOCOL)
    shutil.copy2(directory / "logs/protocol_signature.json", destination / "logs/protocol_signature.json")
    write(destination / "diagnostics.json", diagnostics)
    write(destination / "complete.json", metadata)


def train(arm, seed):
    complete = p.bundle_dir(arm, seed) / "complete.json"
    if complete.exists():
        p.validate(read(complete), arm, seed)
        return
    signature = p.run_dir(arm, seed) / "logs/protocol_signature.json"
    if signature.exists():
        config = read(signature)["config"]
        if config["seed"] != seed or any(config[key] != value for key, value in p.settings(arm).items()):
            raise ValueError("checkpoint belongs to another alignment treatment")
    subprocess.run(training_command(arm, seed), cwd=p.ROOT,
                   env=dict(os.environ, PYTHONPATH=str(p.ROOT)), check=True)
    publish(arm, seed)
    print(f"COMPLETE {arm}/{seed}: {p.budget()}", flush=True)


def load_actor(arm, seed):
    directory = p.bundle_dir(arm, seed)
    p.validate(read(directory / "complete.json"), arm, seed)
    config = old.previous.final_task_sweep.load_config(directory)
    env = make_env(config)
    try:
        agent = make_algo(config.algo, env.obs_dim, env.act_dim, config)
        with (directory / "parameters.pkl").open("rb") as stream:
            old.previous.restore(agent.policy, pickle.load(stream)["policy"], arm + " policy")
        return config, old.previous.evaluate.corrected_audit.AgentRuntime(agent, config)
    finally:
        env.close()


def audit(seed):
    _patch_flax_variablestate_unpickle()
    evaluator = old.previous.evaluate
    evaluator.protocol = p
    controllers = {arm: load_actor(arm, seed) for arm in p.ARMS}
    ignored = {"algo", "ensemble_size", "alpha", "resac_critic_actor_ratio", "resac_clip_norm",
               "weight_reg", "save_root", "run_name"}
    common = [{key: value for key, value in vars(config).items() if key not in ignored}
              for config, runtime in controllers.values()]
    if any(config != common[0] for config in common[1:]):
        raise ValueError("alignment arms differ outside registered controls")
    stationary, switching = {}, {}
    for event in p.STATIONARY_HOLDOUT_EVENT_SEEDS:
        stationary[str(event)] = {arm: {str(mode): evaluator._stationary_runtime(config, runtime, event, mode)
                                      for mode in p.MODES} for arm, (config, runtime) in controllers.items()}
        print(f"STATIONARY event={event} complete", flush=True)
    for event in p.SWITCHING_EVENT_SEEDS:
        rows = {arm: evaluator._switching_runtime(config, runtime, event)
                for arm, (config, runtime) in controllers.items()}
        if any(row["episode_sequences"] != rows[p.ARMS[0]]["episode_sequences"] for row in rows.values()):
            raise ValueError("alignment arms saw different switching schedules")
        switching[str(event)] = rows
        print(f"SWITCHING event={event} complete", flush=True)
    write(p.audit_path(seed), {"protocol": p.PROTOCOL, "seed": seed, "budget": p.budget(),
        "runtime_versions": p.check_runtime(), "stationary": stationary, "switching": switching,
        "actor_updates": {arm: p.actor_updates(arm) for arm in p.ARMS},
        "diagnostics": {arm: read(p.bundle_dir(arm, seed) / "diagnostics.json") for arm in p.ARMS}})


def analyze():
    audits = [read(p.audit_path(seed)) for seed in p.SEEDS]
    for seed, row in zip(p.SEEDS, audits):
        if (row["protocol"] != p.PROTOCOL or row["seed"] != seed or row["budget"] != p.budget()
                or row["runtime_versions"] != p.RUNTIME):
            raise ValueError("mixed or incomplete alignment audits")
    values = {arm: np.asarray([np.mean([event[arm]["returns"] for event in row["switching"].values()])
                             for row in audits]) for arm in p.ARMS}
    comparisons = {}
    for left, right in (("aligned_full", "fork"), ("aligned_full", "aligned_no_shift"),
                        ("aligned_no_shift", "fork")):
        delta = values[left] - values[right]
        radius = float(t.ppf(0.975, len(delta) - 1) * delta.std(ddof=1) / np.sqrt(len(delta)))
        comparisons[left + "-minus-" + right] = {"mean": float(delta.mean()),
            "ci95": [float(delta.mean() - radius), float(delta.mean() + radius)],
            "per_seed": dict(zip(map(str, p.SEEDS), delta.tolist())), "wins": int((delta > 0).sum())}
    write(p.ANALYSIS, {"protocol": p.PROTOCOL, "seeds": list(p.SEEDS), "budget": p.budget(),
        "actor_updates": {arm: p.actor_updates(arm) for arm in p.ARMS},
        "switching": {arm: value.tolist() for arm, value in values.items()}, "comparisons": comparisons,
        "stationary": {arm: [float(np.mean([mode["return_mean"] for event in row["stationary"].values()
                         for mode in event[arm].values()])) for row in audits] for arm in p.ARMS},
        "diagnostics": {str(row["seed"]): row["diagnostics"] for row in audits},
        "scope": "Bundled Torch-equation FP32 alignment diagnostic; fixed environment/budget, not a paper reproduction or BAPR confirmation."})
    print(json.dumps(comparisons, indent=2), flush=True)


def smoke():
    parity = check_parity(p.REFERENCE)
    rows = {}
    for arm in ("aligned_full", "aligned_no_shift"):
        directory = p.RUN_ROOT / "smoke" / arm
        command = training_command(arm, p.SEEDS[0])
        for name, value in {"save_root": directory.parent, "run_name": directory.name, "max_iters": 2,
                            "start_train_steps": 0, "initial_random_steps": 0,
                            "log_interval": 100, "eval_episodes": 1}.items():
            old.previous.set_option(command, name, value)
        subprocess.run(command, cwd=p.ROOT, check=True)
        config, agent, budget = old.checkpoint_agent(directory)
        if budget != {"next_iteration": 2, "total_steps": 8000, "update_count": 500}:
            raise ValueError("incomplete Torch-alignment smoke")
        metrics = {name: np.load(directory / "logs" / f"{name}.npy").tolist()
                   for name in ("critic_loss", "policy_loss", "reg_bonus_mean", "actor_update_rate")}
        if not all(np.isfinite(value).all() for value in metrics.values()):
            raise ValueError("nonfinite Torch-alignment smoke")
        if metrics["actor_update_rate"] != [0.5, 0.5]:
            raise ValueError("incorrect Torch actor update frequency")
        if ((arm == "aligned_full" and min(metrics["reg_bonus_mean"]) <= 0)
                or (arm == "aligned_no_shift" and any(metrics["reg_bonus_mean"]))):
            raise ValueError("incorrect norm-shift intervention")
        rows[arm] = metrics
        del agent
        jax.clear_caches()
    write(p.SMOKE, {"protocol": p.PROTOCOL, "runtime_versions": p.check_runtime(),
                    "parity": parity, "arms": rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "train", "audit", "analyze"))
    parser.add_argument("--arm", choices=p.ARMS)
    parser.add_argument("--seed", type=int, choices=p.SEEDS)
    args = parser.parse_args()
    p.check_runtime()
    if args.mode == "train":
        train(args.arm, args.seed)
    elif args.mode == "audit":
        audit(args.seed)
    else:
        globals()[args.mode]()


if __name__ == "__main__":
    main()
