"""Strict paired audit for one family of independent specialists."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.analysis import (
    bapr_v3_independent_specialists as protocol,
)
from jax_experiments.analysis.final_task_sweep import (
    episode_returns,
    load_config,
)
from jax_experiments.common.checkpoint import load_checkpoint
from jax_experiments.common.logging import Logger
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.train import (
    _reset_eval_switch_schedule,
    _select_eval_switch_sequence,
    make_algo,
    make_env,
)


SCHEMA = "bapr.v3-independent-specialist-audit-group.v1"
CONTROLLERS = (
    "robust", "dynamic_oracle",
    "fixed_mode_0", "fixed_mode_1", "fixed_mode_2", "fixed_mode_3",
)
EPISODES_PER_TASK = 5
SWITCHING_EPISODES = 5
SWITCHING_PERIOD_STEPS = 500


def _load_agent(bundle_dir: Path):
    config = load_config(bundle_dir)
    config.stochastic_mode_fixed_id = -1
    env = make_env(config, seed_offset=0)
    agent = make_algo(config.algo, env.obs_dim, env.act_dim, config)
    replay = ReplayBuffer(
        env.obs_dim, env.act_dim, capacity=1,
        belief_dim=getattr(agent, "belief_dim", 0))
    with tempfile.TemporaryDirectory() as temporary:
        logger = Logger(temporary)
        next_iteration, total_steps = load_checkpoint(
            str(bundle_dir / "checkpoints"), agent, replay, logger,
            config.algo, load_replay_buffer=False)
    if (next_iteration != protocol.FINAL_NEXT_ITERATION
            or total_steps != protocol.FINAL_TOTAL_STEPS):
        raise ValueError(
            f"bundle checkpoint has wrong budget: {bundle_dir}, "
            f"next_iter={next_iteration}, steps={total_steps}")
    if hasattr(env, "close"):
        env.close()
    return config, agent, nnx.state(agent.policy, nnx.Param)


def _controller_policy_states(family: str):
    protocol.validate_family_bundles(family)
    config, robust, robust_state = _load_agent(
        protocol.robust_bundle_dir(family))
    agents = {"robust": robust}
    states = {"robust": robust_state}
    for mode in protocol.MODES:
        _, agent, state = _load_agent(
            protocol.specialist_bundle_dir(family, mode))
        agents[f"fixed_mode_{mode}"] = agent
        states[f"fixed_mode_{mode}"] = state
    return config, agents, states


def _return_record(
    rewards: np.ndarray, dones: np.ndarray, episodes: int, horizon: int,
) -> dict[str, Any]:
    returns, terminated, steps = episode_returns(
        rewards, dones, episodes, horizon)
    return {
        "returns": [float(value) for value in returns],
        "terminated": [bool(value) for value in terminated],
        "steps": [int(value) for value in steps],
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns)),
        "terminated_rate": float(np.mean(terminated)),
    }


def evaluate_stationary(
    config, policy_graphdef, policy_states: dict[str, Any], event_seed: int,
) -> dict[str, Any]:
    horizon = int(config.max_episode_steps)
    rows: dict[str, dict[str, Any]] = {}
    physical_controllers = (
        "robust", "fixed_mode_0", "fixed_mode_1",
        "fixed_mode_2", "fixed_mode_3")
    for controller in physical_controllers:
        env = make_env(config, seed_offset=event_seed)
        tasks = env.sample_tasks(4)
        env.build_rollout_fn(policy_graphdef)
        mode_rows = {}
        for mode, task in enumerate(tasks):
            env.set_task(task)
            key = jax.random.fold_in(
                jax.random.PRNGKey(20260716 + event_seed), mode)
            rewards, dones = env.eval_rollout(
                policy_states[controller],
                EPISODES_PER_TASK * horizon, key,
                episode_horizon=horizon)
            mode_rows[str(mode)] = _return_record(
                np.asarray(rewards), np.asarray(dones),
                EPISODES_PER_TASK, horizon)
        rows[controller] = mode_rows
        if hasattr(env, "close"):
            env.close()

    rows["dynamic_oracle"] = {
        str(mode): rows[f"fixed_mode_{mode}"][str(mode)]
        for mode in protocol.MODES
    }
    return rows


def _base_action_fn(policy_graphdef):
    @jax.jit
    def action(policy_state, observation):
        policy = nnx.merge(policy_graphdef, policy_state)
        return policy.base_deterministic(observation[None])[0]

    return action


def evaluate_switching_controller(
    config, tasks, policy_states: dict[str, Any], action_fn,
    controller: str, event_seed: int,
    controller_map: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    horizon = int(config.max_episode_steps)
    env = make_env(config, seed_offset=event_seed)
    rows = []
    trace_summary = []
    for episode in range(SWITCHING_EPISODES):
        switch_tasks, source_indices = _select_eval_switch_sequence(
            env, tasks, episode)
        _reset_eval_switch_schedule(
            env, switch_tasks, config, SWITCHING_PERIOD_STEPS)
        base_key = 20260716 + event_seed * 100_000 + episode * 10_000
        env.rng = jax.random.PRNGKey(base_key)
        observation = env.reset()
        total_return = 0.0
        termination_count = 0
        first_done_step = horizon
        switches = 0
        previous_mode = int(env.current_task_id)
        selected_modes = []
        physics_modes = []
        for step in range(horizon):
            physics_mode = int(env.task_id_for_next_step())
            if controller == "robust":
                policy_key = "robust"
                selected_mode = -1
            elif controller == "dynamic_oracle":
                policy_key = f"fixed_mode_{physics_mode}"
                selected_mode = physics_mode
            elif controller == "dynamic_control_oracle":
                if controller_map is None:
                    raise ValueError(
                        "dynamic_control_oracle requires controller_map")
                selected_mode = int(controller_map[physics_mode])
                policy_key = f"fixed_mode_{selected_mode}"
            else:
                selected_mode = int(controller.removeprefix("fixed_mode_"))
                policy_key = controller
            action = np.asarray(action_fn(
                policy_states[policy_key],
                jnp.asarray(observation, dtype=jnp.float32)))

            # Pair actuator events by global event/episode/step, independent of
            # controller-specific simulator resets after termination.
            env.rng = jax.random.PRNGKey(base_key + step * 2 + 1)
            observation, reward, done, info = env.step(action)
            if int(info["mode_used"]) != physics_mode:
                raise RuntimeError("action was applied under the wrong mode")
            total_return += float(reward)
            selected_modes.append(selected_mode)
            physics_modes.append(physics_mode)
            current_mode = int(env.current_task_id)
            if current_mode != previous_mode:
                switches += 1
            previous_mode = current_mode
            if done:
                termination_count += 1
                first_done_step = min(first_done_step, step + 1)
                env.rng = jax.random.PRNGKey(base_key + step * 2 + 2)
                observation = env.reset()
        aligned = []
        for selected, physics in zip(selected_modes, physics_modes):
            if selected < 0:
                continue
            expected = (
                int(controller_map[physics])
                if controller == "dynamic_control_oracle"
                and controller_map is not None
                else physics
            )
            aligned.append(selected == expected)
        rows.append({
            "episode": episode,
            "return": total_return,
            "termination_count": termination_count,
            "first_done_step": first_done_step,
            "switch_count": switches,
            "sequence_source_indices": [int(value) for value in source_indices],
        })
        trace_summary.append({
            "episode": episode,
            "steps": len(physics_modes),
            "physics_mode_counts": {
                str(mode): int(physics_modes.count(mode))
                for mode in protocol.MODES
            },
            "selected_mode_counts": {
                str(mode): int(selected_modes.count(mode))
                for mode in (-1, *protocol.MODES)
                if selected_modes.count(mode)
            },
            "selection_alignment": (
                float(np.mean(aligned)) if aligned else None),
        })
    if hasattr(env, "close"):
        env.close()
    returns = [float(row["return"]) for row in rows]
    return {
        "episodes": rows,
        "trace_summary": trace_summary,
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns)),
    }


def evaluate_switching(
    config, policy_graphdef, policy_states: dict[str, Any], event_seed: int,
) -> dict[str, Any]:
    template = make_env(config, seed_offset=event_seed)
    tasks = template.sample_tasks(4)
    if hasattr(template, "close"):
        template.close()
    action_fn = _base_action_fn(policy_graphdef)
    return {
        controller: evaluate_switching_controller(
            config, tasks, policy_states, action_fn,
            controller, event_seed)
        for controller in CONTROLLERS
    }


def validate_group(payload: dict[str, Any]) -> None:
    if (payload.get("schema") != SCHEMA
            or payload.get("status") != "complete"):
        raise ValueError("invalid independent specialist audit schema")
    if set(payload.get("stationary") or {}) != set(CONTROLLERS):
        raise ValueError("stationary controller set is incomplete")
    if set(payload.get("switching") or {}) != set(CONTROLLERS):
        raise ValueError("switching controller set is incomplete")
    stationary = payload["stationary"]
    for controller in CONTROLLERS:
        if set(stationary[controller]) != {"0", "1", "2", "3"}:
            raise ValueError(f"stationary rows incomplete for {controller}")
        for mode in protocol.MODES:
            record = stationary[controller][str(mode)]
            if (len(record.get("returns") or []) != EPISODES_PER_TASK
                    or not all(math.isfinite(float(value))
                               for value in record["returns"])):
                raise ValueError("stationary return record is invalid")
    for mode in protocol.MODES:
        if (stationary["dynamic_oracle"][str(mode)]
                != stationary[f"fixed_mode_{mode}"][str(mode)]):
            raise ValueError("dynamic stationary row is not the exact diagonal")
    for controller, record in payload["switching"].items():
        episodes = record.get("episodes") or []
        if (len(episodes) != SWITCHING_EPISODES
                or not all(math.isfinite(float(row["return"]))
                           for row in episodes)):
            raise ValueError(f"switching record invalid for {controller}")
    for row in payload["switching"]["dynamic_oracle"]["trace_summary"]:
        if row.get("selection_alignment") != 1.0:
            raise ValueError("dynamic specialist did not track physics mode")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("legacy", "stochastic_headroom", "structured_channel"),
        default="legacy",
    )
    parser.add_argument("--env")
    parser.add_argument("--family", required=True)
    parser.add_argument("--event-seed", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.profile == "stochastic_headroom":
        if not args.env:
            parser.error("--env is required for stochastic_headroom")
        protocol.configure_stochastic_headroom(args.env)
    elif args.profile == "structured_channel":
        if not args.env:
            parser.error("--env is required for structured_channel")
        protocol.configure_structured_channel_headroom(args.env)
    elif args.env and args.env != protocol.ENV:
        parser.error(f"legacy profile only supports {protocol.ENV}")
    protocol._require_family(args.family)
    return args


def main() -> None:
    args = parse_args()
    output = args.out_dir.resolve()
    result_path = output / "group.json"
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        validate_group(payload)
        print(f"Complete valid specialist audit already exists: {output}")
        return

    bundles = protocol.validate_family_bundles(args.family)
    config, agents, policy_states = _controller_policy_states(args.family)
    graphdef = nnx.graphdef(agents["robust"].policy)
    stationary = evaluate_stationary(
        config, graphdef, policy_states, args.event_seed)
    switching = evaluate_switching(
        config, graphdef, policy_states, args.event_seed)
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "family": args.family,
        "env": protocol.ENV,
        "training_seed": protocol.SEED,
        "event_seed": int(args.event_seed),
        "event_seed_role": "paired evaluation stream; not policy seed",
        "episodes_per_task": EPISODES_PER_TASK,
        "switching_episodes": SWITCHING_EPISODES,
        "switching_period_steps": SWITCHING_PERIOD_STEPS,
        "bundle_manifest_sha256": {
            name: protocol.fork_protocol.sha256_file(
                (protocol.family_bundle_root(args.family) / name
                 / protocol.BUNDLE_MANIFEST))
            for name in bundles
        },
        "stationary": stationary,
        "switching": switching,
    }
    validate_group(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.tmp.", dir=output.parent))
    try:
        protocol.write_json_atomic(temporary / "group.json", payload)
        if output.exists():
            shutil.rmtree(output)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"INDEPENDENT SPECIALIST AUDIT COMPLETE: {output}", flush=True)


if __name__ == "__main__":
    main()
