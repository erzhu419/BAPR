"""Strict return audit for one BAPR-v4 development event stream."""
from __future__ import annotations

import argparse
import copy
import json
import math
import tempfile
from pathlib import Path

import jax
import numpy as np

from jax_experiments.analysis import bapr_v4_persistent_option as protocol
from jax_experiments.analysis import (
    run_bapr_v3_utility_aware_router_audit as baseline_audit,
)
from jax_experiments.common.checkpoint import load_checkpoint
from jax_experiments.common.logging import Logger
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.train import make_algo, make_env, _reset_eval_switch_schedule


def source_records(candidate_protocol=protocol):
    paths = (
        Path(__file__).resolve(), Path(candidate_protocol.__file__).resolve())
    return {
        str(path.relative_to(candidate_protocol.ROOT)):
        candidate_protocol.file_record(path)
        for path in paths
    }


def _source_id(agent, source: str) -> int:
    return {
        "robust": agent.CONTEXT_ROBUST,
        "oracle_persistent": agent.CONTEXT_ORACLE,
        "learned_persistent": agent.CONTEXT_LEARNED,
    }[source]


def _selected_option(agent, source: str, physics_mode: int) -> int:
    if source == "robust":
        return -1
    if source == "oracle_persistent":
        return int(physics_mode)
    return int(agent.adaptation_state[5])


def _option_summary(options, physics_modes):
    selected = np.asarray(options, dtype=np.int32)
    modes = np.asarray(physics_modes, dtype=np.int32)
    switches = int(np.count_nonzero(selected[1:] != selected[:-1]))
    active = selected >= 0
    return {
        "active_rate": float(np.mean(active)),
        "physical_mode_accuracy": float(np.mean(selected == modes)),
        "robust_rate": float(np.mean(~active)),
        "option_switches": switches,
        "selected_option_counts": {
            str(option): int(np.count_nonzero(selected == option))
            for option in (-1, 0, 1, 2, 3)
        },
    }


def _load_agent(candidate_protocol=protocol):
    config = candidate_protocol.configure("formal")
    probe_config = copy.deepcopy(config)
    probe_config.seed = 0
    env = make_env(probe_config, seed_offset=0)
    agent = make_algo(
        candidate_protocol.ALGO_NAME, env.obs_dim, env.act_dim, config)
    replay = ReplayBuffer(
        env.obs_dim, env.act_dim, capacity=1,
        belief_dim=agent.belief_dim)
    with tempfile.TemporaryDirectory(prefix="bapr-v4-audit-logger-") as root:
        logger = Logger(root)
        start_iteration, total_steps = load_checkpoint(
            str(candidate_protocol.checkpoint_dir("formal")), agent, replay,
            logger, candidate_protocol.ALGO_NAME, load_replay_buffer=False)
    env.close()
    if (start_iteration != candidate_protocol.FINAL_NEXT_ITERATION
            or total_steps != candidate_protocol.FINAL_TOTAL_STEPS):
        raise RuntimeError(
            "BAPR-v4 audit requires the exact final checkpoint: "
            f"iter={start_iteration}, steps={total_steps}")
    agent.set_training_iteration(start_iteration - 1)
    return config, agent


def _act(agent, source: str, observation, task, physics_mode: int):
    source_id = _source_id(agent, source)
    if source == "oracle_persistent":
        agent.set_eval_task(task)
    action = agent.select_action(
        observation, deterministic=True, context_source=source_id,
        advantage_enabled=False)
    if not np.all(np.isfinite(action)):
        raise RuntimeError("BAPR-v4 emitted a non-finite action")
    selected = _selected_option(agent, source, physics_mode)
    return np.asarray(action), selected


def evaluate_stationary(config, agent, event_seed: int,
                        candidate_protocol=protocol):
    horizon = int(config.max_episode_steps)
    output = {source: {} for source in candidate_protocol.CONTEXT_SOURCES}
    for source in candidate_protocol.CONTEXT_SOURCES:
        for mode in range(4):
            run_config = copy.deepcopy(config)
            run_config.seed = 0
            run_config.stochastic_mode_fixed_id = mode
            env = make_env(run_config, seed_offset=event_seed)
            tasks = env.sample_tasks(4)
            agent.set_task_metadata(tasks)
            env.set_nonstationary_para(tasks)
            env.set_task(tasks[mode])
            episodes = []
            for episode in range(baseline_audit.EPISODES_PER_TASK):
                base_key = (
                    20260718 + event_seed * 100_000
                    + mode * 10_000 + episode * 1000)
                env.rng = jax.random.PRNGKey(base_key)
                observation = env.reset()
                agent.reset_adaptation()
                total_return = 0.0
                options = []
                physics_modes = []
                terminated = False
                steps = horizon
                for step in range(horizon):
                    action, selected = _act(
                        agent, source, observation, tasks[mode], mode)
                    env.rng = jax.random.PRNGKey(base_key + step * 2 + 1)
                    next_observation, reward, done, info = env.step(action)
                    if int(info["mode_used"]) != mode:
                        raise RuntimeError("BAPR-v4 stationary mode changed")
                    agent.observe_transition(
                        observation, action, reward, next_observation, done)
                    options.append(selected)
                    physics_modes.append(mode)
                    total_return += float(reward)
                    observation = next_observation
                    if done:
                        terminated = True
                        steps = step + 1
                        break
                episodes.append({
                    "episode": episode,
                    "return": total_return,
                    "terminated": terminated,
                    "steps": steps,
                    "options": _option_summary(options, physics_modes),
                })
            values = [float(row["return"]) for row in episodes]
            output[source][str(mode)] = {
                "episodes": episodes,
                "returns": values,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "terminated_rate": float(np.mean([
                    bool(row["terminated"]) for row in episodes])),
            }
            env.close()
    return output


def _configure_switching(env, tasks, kind: str, episode: int):
    if kind == "slow_pair":
        _reset_eval_switch_schedule(
            env, tasks, None, baseline_audit.SLOW_DWELL_STEPS)
        return tuple(int(value) for value in env._eval_mode_sequence)
    sequence = baseline_audit.FULL_CYCLE_SEQUENCES[
        episode % len(baseline_audit.FULL_CYCLE_SEQUENCES)]
    env.configure_eval_mode_sequence(
        tasks, sequence, baseline_audit.FULL_CYCLE_DWELL_STEPS)
    return sequence


def evaluate_switching(config, agent, event_seed: int, kind: str,
                       candidate_protocol=protocol):
    horizon = int(config.max_episode_steps)
    output = {}
    kind_offset = 0 if kind == "slow_pair" else 500_000
    for source in candidate_protocol.CONTEXT_SOURCES:
        run_config = copy.deepcopy(config)
        run_config.seed = 0
        run_config.stochastic_mode_fixed_id = -1
        env = make_env(run_config, seed_offset=event_seed)
        tasks = env.sample_tasks(4)
        agent.set_task_metadata(tasks)
        episodes = []
        for episode in range(baseline_audit.SWITCHING_EPISODES):
            sequence = _configure_switching(env, tasks, kind, episode)
            base_key = (
                20260718 + event_seed * 100_000
                + kind_offset + episode * 10_000)
            env.rng = jax.random.PRNGKey(base_key)
            observation = env.reset()
            agent.reset_adaptation()
            total_return = 0.0
            termination_count = 0
            first_done_step = horizon
            options = []
            physics_modes = []
            for step in range(horizon):
                physics_mode = int(env.task_id_for_next_step())
                action, selected = _act(
                    agent, source, observation,
                    tasks[physics_mode], physics_mode)
                env.rng = jax.random.PRNGKey(base_key + step * 2 + 1)
                next_observation, reward, done, info = env.step(action)
                if int(info["mode_used"]) != physics_mode:
                    raise RuntimeError(
                        "BAPR-v4 switching action used the wrong mode")
                agent.observe_transition(
                    observation, action, reward, next_observation, done)
                options.append(selected)
                physics_modes.append(physics_mode)
                total_return += float(reward)
                observation = next_observation
                if done:
                    termination_count += 1
                    first_done_step = min(first_done_step, step + 1)
                    env.rng = jax.random.PRNGKey(base_key + step * 2 + 2)
                    observation = env.reset()
            episodes.append({
                "episode": episode,
                "return": total_return,
                "termination_count": termination_count,
                "first_done_step": first_done_step,
                "configured_mode_sequence": list(sequence),
                "physics_mode_counts": {
                    str(mode): int(physics_modes.count(mode))
                    for mode in range(4)
                },
                "options": _option_summary(options, physics_modes),
            })
        values = [float(row["return"]) for row in episodes]
        output[source] = {
            "episodes": episodes,
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
        env.close()
    return output


def validate_group(payload, candidate_protocol=protocol):
    if (payload.get("schema") != candidate_protocol.AUDIT_SCHEMA
            or payload.get("status") != "complete"
            or int(payload.get("event_seed", -1))
            not in candidate_protocol.DEVELOPMENT_EVENT_SEEDS
            or set(payload.get("stationary") or {})
            != set(candidate_protocol.CONTEXT_SOURCES)
            or set(payload.get("switching") or {})
            != {"slow_pair", "full_cycle"}):
        raise ValueError("invalid BAPR-v4 audit group identity")
    for source in candidate_protocol.CONTEXT_SOURCES:
        if set(payload["stationary"][source]) != {"0", "1", "2", "3"}:
            raise ValueError("incomplete BAPR-v4 stationary modes")
        for record in payload["stationary"][source].values():
            if (len(record.get("episodes") or [])
                    != baseline_audit.EPISODES_PER_TASK):
                raise ValueError("incomplete BAPR-v4 stationary episodes")
        for kind in ("slow_pair", "full_cycle"):
            episodes = payload["switching"][kind][source].get("episodes") or []
            if (len(episodes) != baseline_audit.SWITCHING_EPISODES
                    or not all(math.isfinite(float(row["return"]))
                               for row in episodes)):
                raise ValueError("invalid BAPR-v4 switching episodes")


def run(event_seed: int, candidate_protocol=protocol):
    result_path = candidate_protocol.audit_group_path(event_seed)
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        validate_group(payload, candidate_protocol)
        print(f"Complete option audit exists: {result_path.parent}")
        return result_path
    config, agent = _load_agent(candidate_protocol)
    stationary = evaluate_stationary(
        config, agent, event_seed, candidate_protocol)
    switching = {
        kind: evaluate_switching(
            config, agent, event_seed, kind, candidate_protocol)
        for kind in ("slow_pair", "full_cycle")
    }
    payload = {
        "schema": candidate_protocol.AUDIT_SCHEMA,
        "status": "complete",
        "family": "structured_channel",
        "env": "HalfCheetah-v2",
        "training_seed": candidate_protocol.TRAINING_SEED,
        "event_seed": int(event_seed),
        "checkpoint": candidate_protocol.checkpoint_summary("formal"),
        "checkpoint_params": candidate_protocol.file_record(
            candidate_protocol.checkpoint_dir("formal") / "params.pkl"),
        "source_files": source_records(candidate_protocol),
        "context_sources": list(candidate_protocol.CONTEXT_SOURCES),
        "episodes_per_task": baseline_audit.EPISODES_PER_TASK,
        "switching_episodes": baseline_audit.SWITCHING_EPISODES,
        "slow_dwell_steps": baseline_audit.SLOW_DWELL_STEPS,
        "full_cycle_dwell_steps": baseline_audit.FULL_CYCLE_DWELL_STEPS,
        "stationary": stationary,
        "switching": switching,
    }
    validate_group(payload, candidate_protocol)
    candidate_protocol.write_json_atomic(result_path, payload)
    print(f"OPTION AUDIT COMPLETE: {result_path.parent}", flush=True)
    return result_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.event_seed)


if __name__ == "__main__":
    main()
