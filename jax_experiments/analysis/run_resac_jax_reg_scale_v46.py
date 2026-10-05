"""Run the positive-sign regularization-scale follow-up and strict audits."""
import argparse
import json
import subprocess

import jax
import numpy as np

from jax_experiments.analysis import resac_jax_reg_scale_protocol_v46 as p
from jax_experiments.analysis import run_resac_update_density_v42 as shared
from jax_experiments.analysis import run_resac_jax_reward_transfer_v45 as v45


def training_command(arm, seed):
    return shared.training_command(arm, seed, protocol=p)


def diagnostics(directory, config, agent):
    return v45.diagnostics(directory, config, agent)


def smoke():
    rows = {}
    for arm in p.ARMS:
        directory = p.RUN_ROOT / "smoke" / arm
        command = training_command(arm, p.SEEDS[0])
        for name, value in {"save_root": directory.parent, "run_name": directory.name, "max_iters": 2,
                            "start_train_steps": 0, "initial_random_steps": 0,
                            "log_interval": 100, "eval_episodes": 1}.items():
            shared.common.set_option(command, name, value)
        subprocess.run(command, cwd=p.ROOT, check=True)
        config, agent, budget = shared.checkpoints.checkpoint_agent(directory)
        if budget != {"next_iteration": 2, "total_steps": 8000, "update_count": 8000}:
            raise ValueError("incomplete regularization-scale smoke")
        metrics = {name: np.load(directory / "logs" / f"{name}.npy") for name in
                   ("critic_loss", "policy_loss", "reg_bonus_mean", "actor_update_rate", "train_time", "collect_time")}
        if any(value.shape != (2,) or not np.isfinite(value).all() for value in metrics.values()):
            raise ValueError("nonfinite/incomplete regularization-scale smoke")
        if not np.array_equal(metrics["actor_update_rate"], [0.5, 0.5]):
            raise ValueError("incorrect actor phase/count")
        if metrics["reg_bonus_mean"].min() <= 0:
            raise ValueError("positive-sign regularization bonus missing")
        metrics.update(v45.reward_metrics(directory, config, 2))
        v45.check_resume(agent, config)
        rows[arm] = {name: value.tolist() for name, value in metrics.items()}
        del agent
        jax.clear_caches()
    shared.write(p.SMOKE, {"protocol": p.PROTOCOL, "runtime_versions": p.check_runtime(),
                           "resume_next_update_equal": True, "arms": rows})


def _paired_summary(values):
    return shared.paired_summary(np.asarray(values, dtype=float), protocol=p)


def summarize(audits):
    baseline = shared.read(p.V45_ANALYSIS)
    for seed, row in zip(p.SEEDS, audits, strict=True):
        if (row["protocol"] != p.PROTOCOL or row["seed"] != seed or row["runtime_versions"] != p.RUNTIME
                or row["budgets"] != {arm: p.budget(arm) for arm in p.ARMS}
                or row["actor_updates"] != {arm: p.actor_updates(arm) for arm in p.ARMS}):
            raise ValueError("mixed/incomplete regularization-scale audits")
    switching = {arm: np.asarray([np.mean([event[arm]["returns"] for event in row["switching"].values()])
                                 for row in audits]) for arm in p.ARMS}
    stationary = {arm: [float(np.mean([mode["return_mean"] for event in row["stationary"].values()
                       for mode in event[arm].values()])) for row in audits] for arm in p.ARMS}
    v45_switch = {arm: np.asarray(values, dtype=float) for arm, values in baseline["switching"].items()}
    v45_stationary = {arm: np.asarray(values, dtype=float) for arm, values in baseline["stationary"].items()}
    contrasts = {}
    for arm in p.ARMS:
        label = arm.removeprefix("bus_reward_")
        contrasts[f"{label}_minus_v45_bus_full"] = _paired_summary(switching[arm] - v45_switch["bus_reward_full"])
        contrasts[f"{label}_minus_v45_bus_no_shift"] = _paired_summary(switching[arm] - v45_switch["bus_reward_no_shift"])
    contrasts["reg001_minus_reg003"] = _paired_summary(switching[p.ARMS[0]] - switching[p.ARMS[1]])
    return {"protocol": p.PROTOCOL, "seeds": list(p.SEEDS),
        "budgets": {arm: p.budget(arm) for arm in p.ARMS},
        "actor_updates": {arm: p.actor_updates(arm) for arm in p.ARMS},
        "switching": {arm: value.tolist() for arm, value in switching.items()},
        "stationary": {arm: value for arm, value in stationary.items()},
        "v45_baseline_switching": baseline["switching"],
        "v45_baseline_stationary": baseline["stationary"],
        "contrasts": contrasts,
        "diagnostics": {str(row["seed"]): row["diagnostics"] for row in audits},
        "scope": "Positive-sign reward-standardized regularization-scale pilot; V45 compact baselines are fixed and paired by seed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "train", "audit", "analyze"))
    parser.add_argument("--arm", choices=p.ARMS)
    parser.add_argument("--seed", type=int, choices=p.SEEDS)
    args = parser.parse_args()
    p.check_runtime()
    if args.mode == "train":
        shared.train(args.arm, args.seed, protocol=p, diagnostics_fn=diagnostics)
    elif args.mode == "audit":
        shared.audit(args.seed, protocol=p)
    elif args.mode == "analyze":
        result = summarize([shared.read(p.audit_path(seed)) for seed in p.SEEDS])
        shared.write(p.ANALYSIS, result)
        print(json.dumps(result["contrasts"], indent=2), flush=True)
    else:
        smoke()


if __name__ == "__main__":
    main()
