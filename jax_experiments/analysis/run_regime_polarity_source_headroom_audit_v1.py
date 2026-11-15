"""Cross-evaluate independent source controllers and a true-mode selector."""
from __future__ import annotations

import argparse
import copy
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

from jax_experiments.analysis import final_task_sweep
from jax_experiments.analysis import regime_polarity_source_headroom_v1 as protocol
from jax_experiments.analysis.run_regime_polarity_source_controller_v1 import (
    validate_bundle,
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


ARMS = (*protocol.ROLES, "dynamic_oracle")
EVENT_SCHEMA = "bapr.regime-polarity-source-controller-event.v1"


def _load_controller(role: str, seed: int):
    validate_bundle(role, seed)
    directory = protocol.bundle_dir(role, seed)
    config = final_task_sweep.load_config(directory)
    config.stochastic_mode_fixed_id = -1
    env = make_env(config, seed_offset=0)
    tasks = env.sample_tasks(len(protocol.MODES))
    agent = make_algo(config.algo, env.obs_dim, env.act_dim, config)
    if hasattr(agent, "set_task_metadata"):
        agent.set_task_metadata(tasks)
    replay = ReplayBuffer(
        env.obs_dim, env.act_dim, capacity=1,
        belief_dim=getattr(agent, "belief_dim", 0))
    with tempfile.TemporaryDirectory() as temporary:
        logger = Logger(temporary)
        next_iteration, total_steps = load_checkpoint(
            str(directory / "checkpoints"), agent, replay, logger,
            config.algo, load_replay_buffer=False)
    if (next_iteration != protocol.MAX_ITERS
            or total_steps != protocol.FINAL_TOTAL_STEPS
            or int(agent.update_count) != protocol.FINAL_UPDATE_COUNT):
        raise ValueError(f"stale source controller: {directory}")
    if hasattr(env, "close"):
        env.close()
    return {
        "config": config,
        "agent": agent,
        "policy_graphdef": nnx.graphdef(agent.policy),
        "policy_params": nnx.state(agent.policy, nnx.Param),
        "context_graphdef": (
            nnx.graphdef(agent.context_net)
            if hasattr(agent, "context_net") else None),
        "context_params": (
            nnx.state(agent.context_net, nnx.Param)
            if hasattr(agent, "context_net") else None),
    }


def _action_fn(controller):
    policy_graphdef = controller["policy_graphdef"]
    context_graphdef = controller["context_graphdef"]

    @jax.jit
    def action(policy_params, context_params, observation):
        policy = nnx.merge(policy_graphdef, policy_params)
        obs = jnp.asarray(observation, dtype=jnp.float32)[None]
        if context_graphdef is None:
            return policy.deterministic(obs)[0]
        context = nnx.merge(context_graphdef, context_params)(obs)
        return policy.deterministic(obs, context)[0]

    return action


def _record(rewards, dones) -> dict[str, Any]:
    returns, terminated, steps = final_task_sweep.episode_returns(
        np.asarray(rewards), np.asarray(dones),
        protocol.EPISODES_PER_TASK, protocol.MAX_EPISODE_STEPS)
    return {
        "returns": [float(value) for value in returns],
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "terminated": [bool(value) for value in terminated],
        "terminated_rate": float(np.mean(terminated)),
        "steps": [int(value) for value in steps],
    }


def _stationary(controllers, seed: int, event_seed: int) -> dict:
    rows: dict[str, dict[str, Any]] = {}
    for role in protocol.ROLES:
        source = controllers[role]
        role_rows = {}
        for mode in protocol.MODES:
            config = copy.deepcopy(source["config"])
            config.stochastic_mode_fixed_id = int(mode)
            env = make_env(config, seed_offset=event_seed - seed)
            tasks = env.sample_tasks(len(protocol.MODES))
            env.set_nonstationary_para(tasks)
            env.set_task(tasks[mode])
            env.build_rollout_fn(
                source["policy_graphdef"], source["context_graphdef"])
            key = jax.random.PRNGKey(event_seed * 100 + mode)
            rewards, dones = env.eval_rollout(
                source["policy_params"],
                protocol.EPISODES_PER_TASK * protocol.MAX_EPISODE_STEPS,
                key,
                context_params=source["context_params"],
                episode_horizon=protocol.MAX_EPISODE_STEPS,
            )
            role_rows[str(mode)] = _record(rewards, dones)
            if hasattr(env, "close"):
                env.close()
        rows[role] = role_rows
    rows["dynamic_oracle"] = {
        str(mode): rows[f"specialist_{mode}"][str(mode)]
        for mode in protocol.MODES
    }
    return rows


def _switching_arm(
    controllers,
    seed: int,
    event_seed: int,
    arm: str,
) -> dict[str, Any]:
    reference = controllers["robust_sac"]
    config = copy.deepcopy(reference["config"])
    config.stochastic_mode_fixed_id = -1
    env = make_env(config, seed_offset=event_seed - seed)
    tasks = env.sample_tasks(len(protocol.MODES))
    action_fns = {
        role: _action_fn(controller)
        for role, controller in controllers.items()
    }
    episodes = []
    for episode in range(protocol.SWITCHING_EPISODES):
        sequence, source_indices = _select_eval_switch_sequence(
            env, tasks, episode)
        _reset_eval_switch_schedule(
            env, sequence, config, protocol.DWELL_STEPS)
        base_key = event_seed * 1_000_000 + episode * 10_000
        env.rng = jax.random.PRNGKey(base_key)
        observation = env.reset()
        total_return = 0.0
        termination_count = 0
        selected_modes = []
        physics_modes = []
        for step in range(protocol.MAX_EPISODE_STEPS):
            mode = int(env.task_id_for_next_step())
            role = f"specialist_{mode}" if arm == "dynamic_oracle" else arm
            source = controllers[role]
            action = np.asarray(action_fns[role](
                source["policy_params"], source["context_params"],
                observation), dtype=np.float32)
            env.rng = jax.random.PRNGKey(base_key + step * 2 + 1)
            observation, reward, done, info = env.step(action)
            if int(info["mode_used"]) != mode:
                raise RuntimeError("source-headroom action used wrong physics mode")
            total_return += float(reward)
            physics_modes.append(mode)
            selected_modes.append(
                mode if arm == "dynamic_oracle"
                else protocol.role_fixed_mode(arm))
            if done:
                termination_count += 1
                env.rng = jax.random.PRNGKey(base_key + step * 2 + 2)
                observation = env.reset()
        episodes.append({
            "episode": episode,
            "return": total_return,
            "termination_count": termination_count,
            "sequence_source_indices": [int(value) for value in source_indices],
            "selection_alignment": (
                float(np.mean(np.equal(selected_modes, physics_modes)))
                if arm == "dynamic_oracle" else None),
        })
    if hasattr(env, "close"):
        env.close()
    returns = [row["return"] for row in episodes]
    return {
        "episodes": episodes,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "termination_count_mean": float(np.mean([
            row["termination_count"] for row in episodes])),
    }


def _source_records(seed: int) -> dict[str, Any]:
    return {
        role: protocol.file_record(protocol.bundle_manifest(role, seed))
        for role in protocol.ROLES
    }


def _event_identity(seed: int, event_seed: int) -> dict[str, Any]:
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "benchmark_role": "policy_by_mode_source_controller_audit",
        "env": protocol.ENV,
        "family": protocol.FAMILY,
        "training_seed": protocol.require_training_seed(seed),
        "event_seed": protocol.require_event_seed(event_seed),
        "arms": list(ARMS),
        "privileged_arm": "dynamic_oracle",
    }


def _validate_event(payload: dict, seed: int, event_seed: int) -> None:
    if (payload.get("schema") != EVENT_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("identity") != _event_identity(seed, event_seed)
            or payload.get("source_bundles") != _source_records(seed)
            or set(payload.get("stationary") or {}) != set(ARMS)
            or set(payload.get("switching") or {}) != set(ARMS)):
        raise ValueError("invalid source-headroom event payload")
    for arm in ARMS:
        stationary = payload["stationary"][arm]
        if set(stationary) != {str(mode) for mode in protocol.MODES}:
            raise ValueError("source-headroom stationary matrix is incomplete")
        for row in stationary.values():
            returns = row.get("returns") or []
            if (len(returns) != protocol.EPISODES_PER_TASK
                    or not all(math.isfinite(float(value)) for value in returns)):
                raise ValueError("source-headroom stationary returns are invalid")
        episodes = payload["switching"][arm].get("episodes") or []
        if (len(episodes) != protocol.SWITCHING_EPISODES
                or not all(math.isfinite(float(row["return"])) for row in episodes)):
            raise ValueError("source-headroom switching returns are invalid")
    if any(
            row["selection_alignment"] != 1.0
            for row in payload["switching"]["dynamic_oracle"]["episodes"]):
        raise ValueError("dynamic source oracle did not follow true mode")


def validate_audit(seed: int) -> dict:
    destination = protocol.audit_dir(seed)
    manifest = protocol.read_json(destination / "audit_manifest.json")
    expected_identity = {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "training_seed": protocol.require_training_seed(seed),
        "event_seeds": list(protocol.EVENT_SEEDS),
    }
    if (manifest.get("schema") != protocol.AUDIT_SCHEMA
            or manifest.get("status") != "complete"
            or manifest.get("identity") != expected_identity
            or manifest.get("source_bundles") != _source_records(seed)):
        raise ValueError(f"invalid source-headroom audit: {destination}")
    records = manifest.get("event_files") or {}
    expected = {
        str(event_seed): protocol.file_record(
            protocol.audit_event_result(seed, event_seed))
        for event_seed in protocol.EVENT_SEEDS
    }
    if records != expected:
        raise ValueError("source-headroom event records changed")
    for event_seed in protocol.EVENT_SEEDS:
        _validate_event(
            protocol.read_json(protocol.audit_event_result(seed, event_seed)),
            seed, event_seed)
    return manifest


def run(seed: int) -> None:
    seed = protocol.require_training_seed(seed)
    for role in protocol.ROLES:
        validate_bundle(role, seed)
    destination = protocol.audit_dir(seed)
    if protocol.audit_manifest(seed).is_file():
        try:
            validate_audit(seed)
        except (KeyError, OSError, TypeError, ValueError):
            pass
        else:
            print(f"SOURCE HEADROOM AUDIT ALREADY COMPLETE: {destination}")
            return
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp.", dir=destination.parent))
    try:
        controllers = {
            role: _load_controller(role, seed) for role in protocol.ROLES
        }
        event_files = {}
        for event_seed in protocol.EVENT_SEEDS:
            payload = {
                "schema": EVENT_SCHEMA,
                "status": "complete",
                "identity": _event_identity(seed, event_seed),
                "source_bundles": _source_records(seed),
                "stationary": _stationary(controllers, seed, event_seed),
                "switching": {
                    arm: _switching_arm(
                        controllers, seed, event_seed, arm)
                    for arm in ARMS
                },
            }
            _validate_event(payload, seed, event_seed)
            relative = Path(f"event_seed_{event_seed}/results.json")
            protocol.write_json_atomic(temporary / relative, payload)
            event_files[str(event_seed)] = protocol.file_record(
                temporary / relative)
            print(
                f"SOURCE HEADROOM EVENT COMPLETE: seed={seed} "
                f"event={event_seed}", flush=True)
        protocol.write_json_atomic(temporary / "audit_manifest.json", {
            "schema": protocol.AUDIT_SCHEMA,
            "status": "complete",
            "identity": {
                "protocol_version": protocol.PROTOCOL_VERSION,
                "training_seed": seed,
                "event_seeds": list(protocol.EVENT_SEEDS),
            },
            "source_bundles": _source_records(seed),
            "event_files": event_files,
        })
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_audit(seed)
    print(f"SOURCE HEADROOM AUDIT COMPLETE: {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for scheduler input staging")
    run(args.seed)


if __name__ == "__main__":
    main()
