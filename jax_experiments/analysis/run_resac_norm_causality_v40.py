"""Train matched norm interventions and audit final actors; compact outputs only."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from scipy.stats import t

from jax_experiments.analysis import resac_norm_causality_protocol_v40 as p
from jax_experiments.analysis import run_regularized_comparison_v39 as previous
from jax_experiments.analysis.diagnose_resac_norm_lcb import disagreement
from jax_experiments.algos.resac import regularization_bonus
from jax_experiments.common.checkpoint import _patch_flax_variablestate_unpickle, _to_numpy_tree, load_checkpoint
from jax_experiments.common.logging import Logger
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.train import make_algo, make_env

read, write = previous.read, previous.write


def training_command(arm, seed):
    directory = p.run_dir(arm, seed)
    command = previous.training_command("resac_full", p.previous.SEEDS[0])
    options = {"seed": seed, "save_root": directory.parent, "run_name": directory.name,
               "max_iters": 2100, "weight_reg": p.settings(arm)["weight_reg"],
               "resac_reg_norm_mode": p.settings(arm)["resac_reg_norm_mode"]}
    for name, value in options.items():
        previous.set_option(command, name, value)
    return command


def checkpoint_agent(directory):
    _patch_flax_variablestate_unpickle()
    config = previous.final_task_sweep.load_config(directory)
    env = make_env(config)
    try:
        agent = make_algo(config.algo, env.obs_dim, env.act_dim, config)
        with tempfile.TemporaryDirectory() as temporary:
            iteration, steps = load_checkpoint(str(directory / "checkpoints"), agent,
                ReplayBuffer(env.obs_dim, env.act_dim, capacity=1), Logger(temporary),
                config.algo, load_replay_buffer=False)
        return config, agent, {"next_iteration": iteration, "total_steps": steps,
                               "update_count": int(agent.update_count)}
    finally:
        env.close()


def diagnostics(directory, config, agent):
    with np.load(directory / "checkpoints/replay_buffer.npz") as replay:
        indices = np.linspace(0, int(replay["size"]) - 1, 256, dtype=int)
        obs, act = (jnp.asarray(replay[name][indices]) for name in ("obs", "act"))
        reward = replay["rew"][indices]
    bonus = np.asarray(regularization_bonus(agent.target_critic, config.weight_reg,
                                           config.resac_reg_norm_mode))
    timeline = {}
    for name in ("q_mean", "q_std_mean", "reg_bonus_mean", "reg_bonus_std", "critic_loss",
                 "policy_loss", "alpha", "collect_time", "train_time", "eval_reward"):
        values = np.load(directory / "logs" / f"{name}.npy")
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite {name} in {directory}")
        positions = sorted({i for i in (0, 1, 10, 50, 100, 250, 500, 1000, len(values) - 1)
                            if i < len(values)})
        timeline[name] = {"count": len(values), "samples": [[i, float(values[i])] for i in positions]}
    return {"sample_count": 256, "head_bonus": bonus.tolist(),
            "replay_disagreement": disagreement(agent.critic(obs, act)),
            "sample_reward_abs_mean": float(abs(reward).mean()), "timeline": timeline}


def publish(arm, seed):
    directory, destination = p.run_dir(arm, seed), p.bundle_dir(arm, seed)
    config, agent, budget = checkpoint_agent(directory)
    signature = read(directory / "logs/protocol_signature.json")
    metadata = {"identity": p.identity(arm, seed), "budget": budget,
                "runtime_versions": p.check_runtime(), "norm_definition": p.NORM_DEFINITION,
                "config": signature["config"], "host": signature["host"], "python": sys.executable}
    p.validate(metadata, arm, seed)
    parameters = {"policy": _to_numpy_tree(nnx.state(agent.policy, nnx.Param))}
    if not all(np.isfinite(value).all() for value in jax.tree.leaves(parameters)):
        raise ValueError("nonfinite final policy")
    probe = diagnostics(directory, config, agent)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "parameters.pkl").open("wb") as stream:
        pickle.dump(parameters, stream, protocol=pickle.HIGHEST_PROTOCOL)
    (destination / "logs").mkdir(exist_ok=True)
    shutil.copy2(directory / "logs/protocol_signature.json", destination / "logs/protocol_signature.json")
    write(destination / "diagnostics.json", probe)
    write(destination / "complete.json", metadata)


def train(arm, seed):
    complete = p.bundle_dir(arm, seed) / "complete.json"
    if complete.exists():
        p.validate(read(complete), arm, seed)
        return
    directory = p.run_dir(arm, seed)
    signature = directory / "logs/protocol_signature.json"
    if signature.exists():
        config = read(signature)["config"]
        if config["seed"] != seed or any(config[key] != value for key, value in p.settings(arm).items()):
            raise ValueError("resume checkpoint belongs to different norm treatment")
    environment = dict(os.environ, PYTHONPATH=str(p.ROOT))
    subprocess.run(training_command(arm, seed), cwd=p.ROOT, env=environment, check=True)
    publish(arm, seed)
    print(f"COMPLETE {arm}/{seed}: {p.budget()}", flush=True)


def load_actor(arm, seed):
    directory = p.bundle_dir(arm, seed)
    metadata = read(directory / "complete.json")
    p.validate(metadata, arm, seed)
    config = previous.final_task_sweep.load_config(directory)
    env = make_env(config)
    try:
        agent = make_algo(config.algo, env.obs_dim, env.act_dim, config)
        with (directory / "parameters.pkl").open("rb") as stream:
            payload = pickle.load(stream)
        previous.restore(agent.policy, payload["policy"], f"{arm} policy")
        return config, previous.evaluate.corrected_audit.AgentRuntime(agent, config)
    finally:
        env.close()


def audit(seed):
    _patch_flax_variablestate_unpickle()
    previous.evaluate.protocol = p
    controllers = {arm: load_actor(arm, seed) for arm in p.ARMS}
    configs = [vars(config) for config, runtime in controllers.values()]
    ignored = {"weight_reg", "resac_reg_norm_mode", "save_root", "run_name"}
    common = [{key: value for key, value in config.items() if key not in ignored} for config in configs]
    if any(config != common[0] for config in common[1:]):
        raise ValueError("intervention arms differ outside the norm treatment")
    stationary, switching = {}, {}
    for event in p.STATIONARY_HOLDOUT_EVENT_SEEDS:
        stationary[str(event)] = {arm: {str(mode): previous.evaluate._stationary_runtime(
            config, runtime, event, mode) for mode in p.MODES}
            for arm, (config, runtime) in controllers.items()}
        print(f"STATIONARY event={event} complete", flush=True)
    for event in p.SWITCHING_EVENT_SEEDS:
        rows = {arm: previous.evaluate._switching_runtime(config, runtime, event)
                for arm, (config, runtime) in controllers.items()}
        if any(row["episode_sequences"] != rows[p.ARMS[0]]["episode_sequences"] for row in rows.values()):
            raise ValueError("norm interventions saw different switching schedules")
        switching[str(event)] = rows
        print(f"SWITCHING event={event} complete", flush=True)
    write(p.audit_path(seed), {"protocol": p.PROTOCOL, "seed": seed,
          "runtime_versions": p.check_runtime(), "budget": p.budget(),
          "stationary": stationary, "switching": switching,
          "diagnostics": {arm: read(p.bundle_dir(arm, seed) / "diagnostics.json") for arm in p.ARMS}})


def analyze():
    audits = [read(p.audit_path(seed)) for seed in p.SEEDS]
    for seed, row in zip(p.SEEDS, audits, strict=True):
        if (row["protocol"] != p.PROTOCOL or row["seed"] != seed
                or row["runtime_versions"] != p.RUNTIME or row["budget"] != p.budget()):
            raise ValueError("incomplete or mixed norm-causality audits")
    values = {arm: np.asarray([np.mean([event[arm]["returns"] for event in row["switching"].values()])
                             for row in audits]) for arm in p.ARMS}
    comparisons = {}
    for left, right in (("shared", "per_head"), ("no_shift", "per_head"), ("shared", "no_shift")):
        delta = values[left] - values[right]
        radius = float(t.ppf(0.975, len(delta) - 1) * delta.std(ddof=1) / np.sqrt(len(delta)))
        comparisons[f"{left}-minus-{right}"] = {"mean": float(delta.mean()),
            "ci95": [float(delta.mean() - radius), float(delta.mean() + radius)],
            "per_seed": dict(zip(map(str, p.SEEDS), delta.tolist())), "wins": int((delta > 0).sum())}
    write(p.ANALYSIS, {"protocol": p.PROTOCOL, "seeds": list(p.SEEDS), "budget": p.budget(),
          "switching": {arm: values[arm].tolist() for arm in p.ARMS}, "comparisons": comparisons,
          "stationary": {arm: [float(np.mean([mode["return_mean"] for event in row["stationary"].values()
                             for mode in event[arm].values()])) for row in audits] for arm in p.ARMS},
          "diagnostics": {str(row["seed"]): row["diagnostics"] for row in audits},
          "scope": "Three fresh development seeds; norm intervention only, not original-paper reproduction or BAPR confirmation."})
    print(json.dumps(comparisons, indent=2), flush=True)


def smoke():
    rows = {}
    for arm in p.ARMS:
        directory = p.RUN_ROOT / "smoke" / arm
        command = training_command(arm, p.SEEDS[0])
        for name, value in {"save_root": directory.parent, "run_name": directory.name,
                            "max_iters": 2, "start_train_steps": 0, "initial_random_steps": 0,
                            "log_interval": 100, "eval_episodes": 1}.items():
            previous.set_option(command, name, value)
        subprocess.run(command, cwd=p.ROOT, check=True)
        config, agent, budget = checkpoint_agent(directory)
        if budget != {"next_iteration": 2, "total_steps": 8000, "update_count": 500}:
            raise ValueError("incomplete norm-intervention smoke")
        metrics = {name: np.load(directory / "logs" / f"{name}.npy").tolist()
                   for name in ("critic_loss", "policy_loss", "reg_bonus_mean", "reg_bonus_std")}
        if not all(np.isfinite(value).all() for value in metrics.values()):
            raise ValueError("nonfinite norm-intervention smoke")
        if arm == "no_shift":
            if any(metrics["reg_bonus_mean"]) or any(metrics["reg_bonus_std"]):
                raise ValueError("no-shift control has a norm bonus")
        elif min(metrics["reg_bonus_mean"]) <= 0 or (arm == "shared" and any(metrics["reg_bonus_std"])):
            raise ValueError("positive shared/per-head norm intervention failed")
        rows[arm] = metrics
        del agent
        jax.clear_caches()
    write(p.SMOKE, {"protocol": p.PROTOCOL, "runtime_versions": p.check_runtime(), "arms": rows})


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
