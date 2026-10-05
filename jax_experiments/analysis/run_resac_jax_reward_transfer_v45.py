"""Train reward-by-shift arms, then audit raw returns in frozen strict windows."""
import argparse
import json
import subprocess
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.analysis import resac_jax_reward_transfer_protocol_v45 as p
from jax_experiments.analysis import run_resac_update_density_v42 as shared
from jax_experiments.common.checkpoint import load_checkpoint, save_checkpoint
from jax_experiments.common.logging import Logger
from jax_experiments.common.replay_buffer import ReplayBuffer

REWARD_METRICS = ("raw_reward_mean", "raw_reward_std", "training_reward_mean", "training_reward_std")


def training_command(arm, seed):
    return shared.training_command(arm, seed, protocol=p)


def reward_metrics(directory, config, count):
    rows = {name: np.load(directory / "logs" / f"{name}.npy") for name in REWARD_METRICS}
    if any(value.shape != (count,) or not np.isfinite(value).all() for value in rows.values()):
        raise ValueError("incomplete/nonfinite reward-processing logs")
    if config.resac_reward_transform == "bus_standardized":
        if (abs(rows["training_reward_mean"]).max() > 1e-4
                or not np.allclose(rows["training_reward_std"], 10.0, atol=1e-4, rtol=0)):
            raise ValueError("bus reward transform did not reach its declared batch moments")
    elif not all(np.array_equal(rows[f"raw_reward_{stat}"], rows[f"training_reward_{stat}"])
                 for stat in ("mean", "std")):
        raise ValueError("raw reward control was transformed")
    return rows


def diagnostics(directory, config, agent):
    result = shared.checkpoints.diagnostics(directory, config, agent)
    rows = reward_metrics(directory, config, config.max_iters)
    positions = sorted({0, 1, 10, 50, 100, config.max_iters - 1})
    result["reward_processing"] = {"transform": config.resac_reward_transform,
        "timeline": {name: {"count": len(value), "samples": [[i, float(value[i])] for i in positions]}
                     for name, value in rows.items()}}
    return result


def check_resume(agent, config):
    rng = np.random.default_rng(17)
    batch = {name: jnp.asarray(rng.normal(size=(3, 6, size)), dtype=jnp.float32)
             for name, size in (("obs", agent.obs_dim), ("act", agent.act_dim),
                                ("rew", 1), ("next_obs", agent.obs_dim))}
    batch["done"] = jnp.zeros((3, 6, 1))
    agent.multi_update(batch)
    with tempfile.TemporaryDirectory() as temporary:
        replay = ReplayBuffer(agent.obs_dim, agent.act_dim, capacity=1)
        save_checkpoint(temporary, agent, replay, Logger(temporary + "/logs"), 1, 8000, "resac_torch")
        restored = shared.make_algo("resac_torch", agent.obs_dim, agent.act_dim, config)
        load_checkpoint(temporary, restored, replay, Logger(temporary + "/restored"), "resac_torch")
        next_batch = {name: value[:1] for name, value in batch.items()}
        expected, actual = agent.multi_update(next_batch), restored.multi_update(next_batch)
        if expected != actual or expected["actor_update_rate"] != 0:
            raise ValueError("resume changed the next update or odd actor phase")
        for name in ("policy", "critic", "target_critic"):
            for left, right in zip(jax.tree.leaves(nnx.state(getattr(agent, name), nnx.Param)),
                                   jax.tree.leaves(nnx.state(getattr(restored, name), nnx.Param)), strict=True):
                np.testing.assert_array_equal(left, right)
        for name in ("log_alpha", "policy_opt_state", "critic_opt_state", "alpha_opt_state"):
            for left, right in zip(jax.tree.leaves(getattr(agent, name)),
                                   jax.tree.leaves(getattr(restored, name)), strict=True):
                np.testing.assert_array_equal(left, right)


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
            raise ValueError("incomplete reward-transfer smoke")
        metrics = {name: np.load(directory / "logs" / f"{name}.npy") for name in
                   ("critic_loss", "policy_loss", "reg_bonus_mean", "actor_update_rate", "train_time", "collect_time")}
        if any(value.shape != (2,) or not np.isfinite(value).all() for value in metrics.values()):
            raise ValueError("nonfinite/incomplete reward-transfer smoke")
        if not np.array_equal(metrics["actor_update_rate"], [0.5, 0.5]):
            raise ValueError("incorrect actor phase/count")
        if ((config.weight_reg > 0 and metrics["reg_bonus_mean"].min() <= 0)
                or (config.weight_reg == 0 and metrics["reg_bonus_mean"].any())):
            raise ValueError("incorrect positive/zero norm intervention")
        metrics.update(reward_metrics(directory, config, 2))
        rows[arm] = {name: value.tolist() for name, value in metrics.items()}
        check_resume(agent, config)
        print(f"SMOKE {arm}: {json.dumps(rows[arm])}", flush=True)
        del agent
        jax.clear_caches()
    shared.write(p.SMOKE, {"protocol": p.PROTOCOL, "runtime_versions": p.check_runtime(),
                           "resume_next_update_equal": True, "arms": rows})


def summarize(audits):
    for seed, row in zip(p.SEEDS, audits, strict=True):
        if (row["protocol"] != p.PROTOCOL or row["seed"] != seed or row["runtime_versions"] != p.RUNTIME
                or row["budgets"] != {arm: p.budget(arm) for arm in p.ARMS}
                or row["actor_updates"] != {arm: p.actor_updates(arm) for arm in p.ARMS}):
            raise ValueError("mixed/incomplete reward-transfer audits")
    switching = {arm: np.asarray([np.mean([event[arm]["returns"] for event in row["switching"].values()])
                                 for row in audits]) for arm in p.ARMS}
    stationary = {arm: [float(np.mean([mode["return_mean"] for event in row["stationary"].values()
                       for mode in event[arm].values()])) for row in audits] for arm in p.ARMS}
    delta = lambda left, right: switching[left] - switching[right]
    raw_gap = delta("raw_full", "raw_no_shift")
    bus_gap = delta("bus_reward_full", "bus_reward_no_shift")
    contrasts = {"raw_full_minus_zero": raw_gap, "bus_full_minus_zero": bus_gap,
                 "reward_gain_full": delta("bus_reward_full", "raw_full"),
                 "reward_gain_zero": delta("bus_reward_no_shift", "raw_no_shift"),
                 "reward_by_shift_interaction": bus_gap - raw_gap}
    return {"protocol": p.PROTOCOL, "seeds": list(p.SEEDS),
        "budgets": {arm: p.budget(arm) for arm in p.ARMS},
        "actor_updates": {arm: p.actor_updates(arm) for arm in p.ARMS},
        "switching": {arm: value.tolist() for arm, value in switching.items()}, "stationary": stationary,
        "contrasts": {name: shared.paired_summary(value, protocol=p) for name, value in contrasts.items()},
        "diagnostics": {str(row["seed"]): row["diagnostics"] for row in audits},
        "scope": "Reward-only JAX transfer pilot, raw strict returns; not native Torch reproduction or full-budget BAPR confirmation."}


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
