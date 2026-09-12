"""Paired deterministic evaluation for the legacy bus controllers."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch


def _seed_episode(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_bus_policy_paired(
        trainer, environment_factory: Callable[[], object],
        episode_seeds: Iterable[int], *, deterministic: bool = True) -> dict:
    """Evaluate one controller on paired exogenous bus event streams."""
    seeds = [int(seed) for seed in episode_seeds]
    policy = trainer.policy_net
    if hasattr(policy, "_warmup_active"):
        policy._warmup_active = False

    episode_rows = []
    for seed in seeds:
        _seed_episode(seed)
        environment = environment_factory()
        environment.reset()
        state_dict, reward_dict, _ = environment.initialize_state(
            render=False)
        action_dict = {
            key: None for key in range(environment.max_agent_num)}
        episode_reward = 0.0
        done = False

        while not done:
            for key in state_dict:
                agent_states = state_dict[key]
                if len(agent_states) == 1:
                    if action_dict[key] is None:
                        state_input = np.asarray(agent_states[0])
                        action_dict[key] = policy.get_action(
                            torch.from_numpy(state_input).float(),
                            deterministic=deterministic)
                elif len(agent_states) == 2:
                    if agent_states[0][1] != agent_states[1][1]:
                        episode_reward += float(reward_dict[key])
                    state_dict[key] = agent_states[1:]
                    state_input = np.asarray(state_dict[key][0])
                    action_dict[key] = policy.get_action(
                        torch.from_numpy(state_input).float(),
                        deterministic=deterministic)
            state_dict, reward_dict, done = environment.step(action_dict)

        episode_rows.append({
            "seed": seed,
            "reward": float(episode_reward),
            "mode_switch_count": int(environment.mode_switch_count),
            "mode_history": [
                [str(mode), int(timestamp)]
                for mode, timestamp in environment.mode_history
            ],
        })
        close = getattr(environment, "close", None)
        if callable(close):
            close()

    rewards = np.asarray(
        [row["reward"] for row in episode_rows], dtype=np.float64)
    if rewards.size == 0:
        raise ValueError("paired bus evaluation requires at least one episode")
    worst_count = max(1, int(np.ceil(0.10 * rewards.size)))
    return {
        "schema": "bapr.bus-paired-eval.v1",
        "deterministic_policy": bool(deterministic),
        "episode_count": int(rewards.size),
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards, ddof=0)),
        "reward_worst_10pct": float(np.mean(np.sort(rewards)[:worst_count])),
        "episodes": episode_rows,
    }


def write_bus_evaluation(path: str | Path, result: dict,
                         metadata: dict | None = None) -> None:
    output = dict(result)
    output["metadata"] = dict(metadata or {})
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    temporary.replace(target)
