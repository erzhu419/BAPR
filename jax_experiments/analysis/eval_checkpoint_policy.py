"""Evaluate saved BAPR checkpoints with online and/or EMA policy.

This is intentionally separate from training: it reconstructs the run config,
loads params.pkl, skips replay-buffer restore, and calls train.evaluate().
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from flax import nnx

from jax_experiments.configs.default import Config
from jax_experiments.common.checkpoint import load_checkpoint
from jax_experiments.common.logging import Logger
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.train import evaluate, make_algo, make_env


RUN_RE = re.compile(
    r"^(?P<tag>.+)_bapr_(?P<env>Ant|HalfCheetah|Hopper|Walker2d)_dw"
    r"(?P<dwell>\d+)_s(?P<seed>\d+)$"
)


def parse_run_name(name: str):
    match = RUN_RE.match(name)
    if not match:
        raise ValueError(f"Unsupported run name: {name}")
    data = match.groupdict()
    data["dwell"] = int(data["dwell"])
    data["seed"] = int(data["seed"])
    return data


def apply_tag_config(config: Config, tag: str) -> None:
    """Recreate the BAPR flags used by the v52-v54 clean sweeps."""
    config.algo = "bapr"
    config.env_type = "discrete_mode"
    config.brax_backend = "spring"
    config.max_iters = 800
    config.eval_episodes = 5
    config.log_interval = 5
    config.save_interval = 100
    config.critic_target_mode = "independent"
    config.bapr_adaptation_mode = "gate"
    config.actor_objective = "mean"
    config.weight_reg = 0.003
    config.beta_ood = 0.003
    config.penalty_scale = 0.5
    config.belief_conditioned = False

    config.bapr_recent_frac_cap = 0.15
    config.bapr_recent_frac_floor = 0.015
    config.bapr_recent_disagreement_gate = True
    config.bapr_recent_qstd_threshold = 5.0
    config.bapr_recent_qstd_ratio_threshold = 0.02

    config.bapr_reg_disagreement_gate = True
    config.bapr_reg_warmup_iters = 0
    config.bapr_reg_max_iters = 70
    config.bapr_reg_latch = True
    config.bapr_reg_require_both = True
    config.bapr_reg_qstd_threshold = 5.0
    config.bapr_reg_qstd_ratio_threshold = 0.02
    config.bapr_reg_emergency_gate = True
    config.bapr_reg_emergency_scale = 0.2
    config.bapr_reg_emergency_qstd_threshold = 8.0
    config.bapr_reg_emergency_qstd_ratio_threshold = 0.025

    if "v52" in tag:
        config.bapr_recent_floor_mode = "always"
        config.bapr_recent_floor_ratio_low = 0.02
        config.bapr_recent_floor_ratio_high = 0.04
        config.bapr_recent_floor_mid_frac = 0.015
        config.bapr_recent_floor_extreme_frac = 0.0
    elif "v53" in tag:
        config.bapr_recent_floor_mode = "ratio_schedule"
        config.bapr_recent_floor_ratio_low = 0.02
        config.bapr_recent_floor_ratio_high = 0.025
        config.bapr_recent_floor_mid_frac = 0.0
        config.bapr_recent_floor_extreme_frac = 0.0
    elif "v54" in tag:
        config.bapr_recent_floor_mode = "ratio_schedule"
        config.bapr_recent_floor_ratio_low = 0.02
        config.bapr_recent_floor_ratio_high = 0.04
        config.bapr_recent_floor_mid_frac = 0.005
        config.bapr_recent_floor_extreme_frac = 0.0


def evaluate_run(run_dir: Path, policies: list[str], n_episodes: int):
    parsed = parse_run_name(run_dir.name)
    config = Config()
    apply_tag_config(config, parsed["tag"])
    config.env_name = f"{parsed['env']}-v2"
    config.seed = parsed["seed"]
    config.discrete_mean_dwell_iters = parsed["dwell"]
    config.run_name = run_dir.name
    config.save_root = str(run_dir.parent)
    config.eval_episodes = n_episodes

    train_env = make_env(config, seed_offset=0)
    train_env.sample_tasks(config.task_num)
    test_tasks = train_env.sample_tasks(config.test_task_num)

    eval_env = make_env(config, seed_offset=1000)
    agent = make_algo("bapr", train_env.obs_dim, train_env.act_dim, config)

    policy_graphdef = nnx.graphdef(agent.policy)
    context_graphdef = nnx.graphdef(agent.context_net)
    eval_env.build_rollout_fn(policy_graphdef, context_graphdef)

    replay_buffer = ReplayBuffer(
        train_env.obs_dim,
        train_env.act_dim,
        capacity=1,
        belief_dim=getattr(agent, "belief_dim", 0),
    )
    logger = Logger(str(run_dir / "logs"))
    iteration, total_steps = load_checkpoint(
        str(run_dir / "checkpoints"),
        agent,
        replay_buffer,
        logger,
        "bapr",
        load_replay_buffer=False,
    )

    rows = []
    for policy in policies:
        config.use_ema_eval = policy == "ema"
        mean, std = evaluate(agent, eval_env, config, test_tasks, n_episodes)
        rows.append({
            "tag": parsed["tag"],
            "env": parsed["env"],
            "seed": parsed["seed"],
            "policy": policy,
            "eval_mean": mean,
            "eval_std": std,
            "checkpoint_next_iter": iteration,
            "checkpoint_total_steps": total_steps,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--policies", nargs="+", default=["online", "ema"],
                        choices=["online", "ema"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    fieldnames = [
        "tag", "env", "seed", "policy", "eval_mean", "eval_std",
        "checkpoint_next_iter", "checkpoint_total_steps",
    ]
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fh = args.out.open("w", newline="")
    else:
        import sys
        fh = sys.stdout
    with fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for idx, run_dir in enumerate(args.runs, start=1):
            print(f"[{idx}/{len(args.runs)}] evaluating {run_dir.name}",
                  flush=True)
            rows = evaluate_run(run_dir, args.policies, args.episodes)
            writer.writerows(rows)
            fh.flush()


if __name__ == "__main__":
    main()
