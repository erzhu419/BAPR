"""Calibrate and strictly audit one independent BAPR-v8 training seed."""
from __future__ import annotations

import argparse
import copy
import json
import math
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.analysis import bapr_v3_learned_control_router as router_protocol
from jax_experiments.analysis import bapr_v3_utility_aware_router as utility
from jax_experiments.analysis import bapr_v8_seed_validation as protocol
from jax_experiments.analysis.final_task_sweep import load_config
from jax_experiments.analysis.run_bapr_v3_learned_control_router_audit import (
    MODEL_CONFIG,
)
from jax_experiments.analysis.run_bapr_v3_utility_aware_router_audit import (
    FULL_CYCLE_SEQUENCES,
    UtilityRouterRuntime,
)
from jax_experiments.analysis.run_bapr_v8_seed_controller import validate_bundle
from jax_experiments.analysis.train_bapr_v3_learned_control_router import make_model
from jax_experiments.common.checkpoint import load_checkpoint
from jax_experiments.common.logging import Logger
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.train import make_algo, make_env, _reset_eval_switch_schedule


CALIBRATION_EPISODES = 3
AUDIT_EPISODES = 5
SLOW_DWELL_STEPS = 500
FULL_CYCLE_DWELL_STEPS = 250
BANK_CODES = (protocol.ROBUST_CONTROLLER, *protocol.MODES)


def _bundle_role(code: int) -> tuple[str, int | None]:
    if int(code) == protocol.ROBUST_CONTROLLER:
        return "sac", None
    return "specialist", int(code)


def _load_agent(seed: int, role: str, mode: int | None = None):
    bundle = protocol.bundle_dir(seed, role, mode)
    config = load_config(bundle)
    config.stochastic_mode_fixed_id = -1
    env = make_env(config, seed_offset=0)
    agent = make_algo(config.algo, env.obs_dim, env.act_dim, config)
    replay = ReplayBuffer(
        env.obs_dim, env.act_dim, capacity=1,
        belief_dim=getattr(agent, "belief_dim", 0))
    with tempfile.TemporaryDirectory() as temporary:
        logger = Logger(temporary)
        next_iteration, total_steps = load_checkpoint(
            str(bundle / "checkpoints"), agent, replay, logger,
            config.algo, load_replay_buffer=False)
    if (next_iteration != protocol.MAX_ITERS
            or total_steps != protocol.FINAL_TOTAL_STEPS):
        raise ValueError(f"bundle has wrong budget: {bundle}")
    if hasattr(env, "close"):
        env.close()
    return config, agent


def load_agents(seed: int, include_baselines: bool = True):
    seed = protocol.require_seed(seed)
    validate_bundle(seed, "sac", None)
    for mode in protocol.MODES:
        validate_bundle(seed, "specialist", mode)
    if include_baselines:
        validate_bundle(seed, "escp", None)
        validate_bundle(seed, "resac", None)
    config, robust = _load_agent(seed, "sac")
    agents = {"sac": robust}
    for mode in protocol.MODES:
        _, agents[f"specialist_mode_{mode}"] = _load_agent(
            seed, "specialist", mode)
    if include_baselines:
        _, agents["escp"] = _load_agent(seed, "escp")
        _, agents["resac"] = _load_agent(seed, "resac")
    return config, agents


def _router(seed: int, obs_dim: int, act_dim: int, table: dict[str, Any]):
    del seed  # The frozen estimator is part of the preregistered algorithm.
    manifest = protocol.read_json(protocol.ROUTER_MANIFEST)
    if (manifest.get("status") != "complete"
            or manifest.get("family") != protocol.FAMILY
            or manifest.get("env") != protocol.ENV
            or manifest.get("model_config") != MODEL_CONFIG
            or manifest.get("parameter_file")
            != protocol.file_record(protocol.ROUTER_PARAMS)):
        raise ValueError("frozen router snapshot is invalid")
    base = router_protocol.RouterConfig.from_dict(manifest["router_config"])
    config = utility.decision_config(base, protocol.DECISION_VARIANT)
    model = make_model(obs_dim, act_dim, seed=20260718, router_config=config)
    template = nnx.state(model, nnx.Param)
    params = router_protocol.load_parameter_state(
        protocol.ROUTER_PARAMS, template, manifest["parameter_leaves"])
    nnx.update(model, params)
    graphdef = nnx.graphdef(model)

    @jax.jit
    def observe(state, obs, action, reward, next_obs, done):
        current = nnx.merge(graphdef, params)
        return current.observe(
            state, obs, action, reward, next_obs, done,
            enable_reset=False, stop_variance_grad=True)[0]

    return model, observe, config, tuple(
        int(value) for value in table["oracle_controller_map"])


def _action(agents: dict[str, Any], controller: str | int,
            observation: np.ndarray) -> np.ndarray:
    if isinstance(controller, str):
        key = controller
    elif int(controller) in (
            protocol.FALLBACK_CONTROLLER, protocol.ROBUST_CONTROLLER):
        key = "sac"
    else:
        key = f"specialist_mode_{int(controller)}"
    agent = agents[key]
    action = np.asarray(agent.select_action(
        np.asarray(observation, dtype=np.float32), deterministic=True),
        dtype=np.float32)
    expected_shape = (int(agent.act_dim),)
    if action.shape != expected_shape or not np.all(np.isfinite(action)):
        raise RuntimeError(
            f"invalid action from {key}: shape={action.shape}, "
            f"expected={expected_shape}, finite={np.all(np.isfinite(action))}")
    return action


def _stationary_episode(config, agent, mode: int, event_seed: int,
                        episode: int) -> dict[str, Any]:
    run_config = copy.deepcopy(config)
    run_config.stochastic_mode_fixed_id = int(mode)
    env = make_env(run_config, seed_offset=event_seed)
    tasks = env.sample_tasks(4)
    env.set_nonstationary_para(tasks)
    env.set_task(tasks[mode])
    horizon = int(config.max_episode_steps)
    base_key = 20260720 + event_seed * 100_000 + mode * 10_000 + episode * 1000
    env.rng = jax.random.PRNGKey(base_key)
    observation = env.reset()
    total = 0.0
    terminated = False
    steps = horizon
    for step in range(horizon):
        action = np.asarray(agent.select_action(
            observation, deterministic=True))
        env.rng = jax.random.PRNGKey(base_key + step * 2 + 1)
        observation, reward, done, info = env.step(action)
        if int(info["mode_used"]) != mode:
            raise RuntimeError("stationary rollout changed physical mode")
        total += float(reward)
        if done:
            terminated = True
            steps = step + 1
            break
    if hasattr(env, "close"):
        env.close()
    return {
        "return": total,
        "terminated": terminated,
        "steps": steps,
    }


def _nondominated(mean_matrix: dict[int, list[float]]) -> list[int]:
    kept = []
    for controller in BANK_CODES:
        values = np.asarray(mean_matrix[controller], dtype=np.float64)
        dominated = False
        for competitor in BANK_CODES:
            if competitor == controller:
                continue
            other = np.asarray(mean_matrix[competitor], dtype=np.float64)
            if np.all(other >= values) and np.any(other > values):
                dominated = True
                break
        if not dominated:
            kept.append(int(controller))
    if protocol.ROBUST_CONTROLLER not in kept:
        kept.append(protocol.ROBUST_CONTROLLER)
    return sorted(set(kept))


def calibrate(seed: int) -> Path:
    output = protocol.calibration_path(seed)
    if output.is_file():
        validate_calibration(protocol.read_json(output), seed)
        print(f"BAPR V8 CALIBRATION ALREADY COMPLETE: {output}", flush=True)
        return output
    config, agents = load_agents(seed, include_baselines=False)
    per_mode: dict[int, dict[int, list[float]]] = {
        mode: {controller: [] for controller in BANK_CODES}
        for mode in protocol.MODES
    }
    for event_seed in protocol.CALIBRATION_EVENT_SEEDS:
        for mode in protocol.MODES:
            for controller in BANK_CODES:
                role, specialist_mode = _bundle_role(controller)
                key = "sac" if role == "sac" else (
                    f"specialist_mode_{specialist_mode}")
                for episode in range(CALIBRATION_EPISODES):
                    row = _stationary_episode(
                        config, agents[key], mode, event_seed, episode)
                    per_mode[mode][controller].append(float(row["return"]))

    means = {
        controller: [
            float(np.mean(per_mode[mode][controller]))
            for mode in protocol.MODES
        ]
        for controller in BANK_CODES
    }
    nondominated = _nondominated(means)
    oracle_map = [
        int(max(BANK_CODES, key=lambda code: (
            means[code][mode], -code)))
        for mode in protocol.MODES
    ]
    payload = {
        "schema": protocol.CALIBRATION_SCHEMA,
        "status": "complete",
        "training_seed": seed,
        "family": protocol.FAMILY,
        "env": protocol.ENV,
        "calibration_event_seeds": list(protocol.CALIBRATION_EVENT_SEEDS),
        "episodes_per_mode": CALIBRATION_EPISODES,
        "nondominated_controllers": nondominated,
        "oracle_controller_map": oracle_map,
        "robust_controller": protocol.ROBUST_CONTROLLER,
        "fallback_controller": protocol.FALLBACK_CONTROLLER,
        "decision_variant": protocol.DECISION_VARIANT,
        "router_manifest": protocol.file_record(protocol.ROUTER_MANIFEST),
        "router_params": protocol.file_record(protocol.ROUTER_PARAMS),
        "bundle_manifests": {
            path.parent.name: protocol.file_record(path)
            for path in protocol.bundle_paths_for_seed(seed)
        },
        "rows": {
            str(mode): {
                "mean_returns": {
                    str(controller): means[controller][mode]
                    for controller in BANK_CODES
                },
                "oracle_controller": oracle_map[mode],
            }
            for mode in protocol.MODES
        },
    }
    validate_calibration(payload, seed)
    protocol.write_json_atomic(output, payload)
    print(f"BAPR V8 CALIBRATION COMPLETE: {output}", flush=True)
    return output


def validate_calibration(payload: dict[str, Any], seed: int) -> None:
    if (payload.get("schema") != protocol.CALIBRATION_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("training_seed") != seed
            or payload.get("family") != protocol.FAMILY
            or payload.get("env") != protocol.ENV
            or payload.get("decision_variant") != protocol.DECISION_VARIANT
            or payload.get("calibration_event_seeds")
            != list(protocol.CALIBRATION_EVENT_SEEDS)
            or payload.get("episodes_per_mode") != CALIBRATION_EPISODES
            or len(payload.get("oracle_controller_map") or []) != 4
            or protocol.ROBUST_CONTROLLER
            not in set(payload.get("nondominated_controllers") or [])):
        raise ValueError("invalid BAPR-v8 calibration identity")
    if (payload.get("router_manifest")
            != protocol.file_record(protocol.ROUTER_MANIFEST)
            or payload.get("router_params")
            != protocol.file_record(protocol.ROUTER_PARAMS)):
        raise ValueError("BAPR-v8 calibration router snapshot changed")
    expected_bundles = {
        path.parent.name: protocol.file_record(path)
        for path in protocol.bundle_paths_for_seed(seed)
    }
    if payload.get("bundle_manifests") != expected_bundles:
        raise ValueError("BAPR-v8 calibration controller bundles changed")
    oracle_map = [int(value) for value in payload["oracle_controller_map"]]
    nondominated = {
        int(value) for value in payload["nondominated_controllers"]}
    if (not set(oracle_map).issubset(set(BANK_CODES))
            or not nondominated.issubset(set(BANK_CODES))):
        raise ValueError("invalid BAPR-v8 calibration controller ids")
    for mode in protocol.MODES:
        means = (payload.get("rows") or {}).get(str(mode), {}).get(
            "mean_returns") or {}
        if (set(map(int, means)) != set(BANK_CODES)
                or not all(math.isfinite(float(value))
                           for value in means.values())):
            raise ValueError("invalid BAPR-v8 calibration matrix")


def _routing_summary(decisions: list[int], modes: list[int],
                     oracle_map: tuple[int, ...]) -> dict[str, Any]:
    metrics = utility.routing_metrics(
        np.asarray(decisions, dtype=np.int32),
        np.asarray(modes, dtype=np.int32), oracle_map, burnin=32)
    counts = Counter(decisions)
    metrics["selected_controller_counts"] = {
        str(key): int(value) for key, value in sorted(counts.items())}
    return metrics


def _choose(controller: str, physics_mode: int, router, oracle_map):
    if controller == "sac":
        return protocol.ROBUST_CONTROLLER, None
    if controller == "oracle":
        return int(oracle_map[physics_mode]), None
    if controller == "bapr":
        return router.decision()
    return controller, None


def evaluate_stationary(seed: int, event_seed: int, config, agents,
                        table, router_parts):
    model, observe, router_config, oracle_map = router_parts
    horizon = int(config.max_episode_steps)
    output = {controller: {} for controller in protocol.CONTROLLERS}
    for controller in protocol.CONTROLLERS:
        for mode in protocol.MODES:
            run_config = copy.deepcopy(config)
            run_config.stochastic_mode_fixed_id = mode
            env = make_env(run_config, seed_offset=event_seed)
            tasks = env.sample_tasks(4)
            env.set_nonstationary_para(tasks)
            env.set_task(tasks[mode])
            episodes = []
            route_rows = []
            for episode in range(AUDIT_EPISODES):
                base_key = (
                    20260720 + event_seed * 100_000
                    + mode * 10_000 + episode * 1000)
                env.rng = jax.random.PRNGKey(base_key)
                observation = env.reset()
                router = UtilityRouterRuntime(
                    model, observe, table, router_config)
                total = 0.0
                terminated = False
                steps = horizon
                decisions: list[int] = []
                modes: list[int] = []
                for step in range(horizon):
                    selected, _ = _choose(
                        controller, mode, router, oracle_map)
                    action = _action(agents, selected, observation)
                    env.rng = jax.random.PRNGKey(base_key + step * 2 + 1)
                    next_observation, reward, done, info = env.step(action)
                    if int(info["mode_used"]) != mode:
                        raise RuntimeError("stationary audit changed mode")
                    if controller == "bapr":
                        router.observe(
                            observation, action, reward,
                            next_observation, done)
                        decisions.append(int(selected))
                        modes.append(mode)
                    total += float(reward)
                    observation = next_observation
                    if done:
                        terminated = True
                        steps = step + 1
                        break
                episodes.append({
                    "episode": episode, "return": total,
                    "terminated": terminated, "steps": steps})
                if controller == "bapr":
                    route_rows.append(_routing_summary(
                        decisions, modes, oracle_map))
            returns = [row["return"] for row in episodes]
            output[controller][str(mode)] = {
                "episodes": episodes,
                "returns": returns,
                "mean": float(np.mean(returns)),
                "std": float(np.std(returns)),
                "terminated_rate": float(np.mean([
                    row["terminated"] for row in episodes])),
                "routing": route_rows,
            }
            if hasattr(env, "close"):
                env.close()
    return output


def _configure_switching(env, tasks, kind: str, episode: int):
    if kind == "slow_pair":
        _reset_eval_switch_schedule(env, tasks, None, SLOW_DWELL_STEPS)
        return tuple(int(value) for value in env._eval_mode_sequence)
    sequence = FULL_CYCLE_SEQUENCES[episode % len(FULL_CYCLE_SEQUENCES)]
    env.configure_eval_mode_sequence(tasks, sequence, FULL_CYCLE_DWELL_STEPS)
    return sequence


def evaluate_switching(seed: int, event_seed: int, config, agents, table,
                       router_parts, kind: str):
    del seed
    model, observe, router_config, oracle_map = router_parts
    horizon = int(config.max_episode_steps)
    output = {}
    kind_offset = 0 if kind == "slow_pair" else 500_000
    for controller in protocol.CONTROLLERS:
        run_config = copy.deepcopy(config)
        run_config.stochastic_mode_fixed_id = -1
        env = make_env(run_config, seed_offset=event_seed)
        tasks = env.sample_tasks(4)
        episodes = []
        route_rows = []
        for episode in range(AUDIT_EPISODES):
            sequence = _configure_switching(env, tasks, kind, episode)
            base_key = (
                20260720 + event_seed * 100_000
                + kind_offset + episode * 10_000)
            env.rng = jax.random.PRNGKey(base_key)
            observation = env.reset()
            router = UtilityRouterRuntime(model, observe, table, router_config)
            total = 0.0
            termination_count = 0
            first_done_step = horizon
            decisions: list[int] = []
            modes: list[int] = []
            for step in range(horizon):
                physics_mode = int(env.task_id_for_next_step())
                selected, _ = _choose(
                    controller, physics_mode, router, oracle_map)
                action = _action(agents, selected, observation)
                env.rng = jax.random.PRNGKey(base_key + step * 2 + 1)
                next_observation, reward, done, info = env.step(action)
                if int(info["mode_used"]) != physics_mode:
                    raise RuntimeError("switching action used wrong mode")
                if controller == "bapr":
                    router.observe(
                        observation, action, reward, next_observation, done)
                    decisions.append(int(selected))
                    modes.append(physics_mode)
                total += float(reward)
                observation = next_observation
                if done:
                    termination_count += 1
                    first_done_step = min(first_done_step, step + 1)
                    env.rng = jax.random.PRNGKey(base_key + step * 2 + 2)
                    observation = env.reset()
            episodes.append({
                "episode": episode,
                "return": total,
                "termination_count": termination_count,
                "first_done_step": first_done_step,
                "configured_mode_sequence": list(sequence),
                "physics_mode_counts": {
                    str(mode): int(modes.count(mode))
                    if controller == "bapr" else None
                    for mode in protocol.MODES
                },
            })
            if controller == "bapr":
                route_rows.append(_routing_summary(
                    decisions, modes, oracle_map))
        if hasattr(env, "close"):
            env.close()
        returns = [row["return"] for row in episodes]
        output[controller] = {
            "episodes": episodes,
            "mean": float(np.mean(returns)),
            "std": float(np.std(returns)),
            "termination_rate": float(np.mean([
                row["termination_count"] > 0 for row in episodes])),
            "routing": route_rows,
        }
    return output


def validate_audit(payload: dict[str, Any], seed: int,
                   event_seed: int) -> None:
    if (payload.get("schema") != protocol.AUDIT_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("training_seed") != seed
            or payload.get("event_seed") != event_seed
            or payload.get("family") != protocol.FAMILY
            or payload.get("env") != protocol.ENV
            or payload.get("decision_variant") != protocol.DECISION_VARIANT
            or set(payload.get("stationary") or {})
            != set(protocol.CONTROLLERS)
            or set(payload.get("switching") or {})
            != {"slow_pair", "full_cycle"}):
        raise ValueError("invalid BAPR-v8 audit identity")
    if (payload.get("utility_table")
            != protocol.file_record(protocol.calibration_path(seed))
            or payload.get("router_manifest")
            != protocol.file_record(protocol.ROUTER_MANIFEST)
            or payload.get("router_params")
            != protocol.file_record(protocol.ROUTER_PARAMS)):
        raise ValueError("BAPR-v8 audit dependency changed")
    expected_bundles = {
        path.parent.name: protocol.file_record(path)
        for path in protocol.bundle_paths_for_seed(seed)
    }
    if payload.get("bundle_manifests") != expected_bundles:
        raise ValueError("BAPR-v8 audit controller bundles changed")
    for controller in protocol.CONTROLLERS:
        if set(payload["stationary"][controller]) != set(map(str, protocol.MODES)):
            raise ValueError("incomplete stationary audit")
        for record in payload["stationary"][controller].values():
            if (len(record.get("returns") or []) != AUDIT_EPISODES
                    or not all(math.isfinite(float(value))
                               for value in record["returns"])):
                raise ValueError("invalid stationary returns")
        for kind in ("slow_pair", "full_cycle"):
            record = payload["switching"][kind][controller]
            if (len(record.get("episodes") or []) != AUDIT_EPISODES
                    or not math.isfinite(float(record.get("mean")))):
                raise ValueError("invalid switching returns")


def audit(seed: int, event_seed: int) -> Path:
    output = protocol.audit_path(seed, event_seed)
    if output.is_file():
        validate_audit(protocol.read_json(output), seed, event_seed)
        print(f"BAPR V8 AUDIT ALREADY COMPLETE: {output}", flush=True)
        return output
    table = protocol.read_json(protocol.calibration_path(seed))
    validate_calibration(table, seed)
    config, agents = load_agents(seed, include_baselines=True)
    router_parts = _router(
        seed, agents["sac"].obs_dim, agents["sac"].act_dim, table)
    stationary = evaluate_stationary(
        seed, event_seed, config, agents, table, router_parts)
    switching = {
        kind: evaluate_switching(
            seed, event_seed, config, agents, table, router_parts, kind)
        for kind in ("slow_pair", "full_cycle")
    }
    payload = {
        "schema": protocol.AUDIT_SCHEMA,
        "status": "complete",
        "training_seed": seed,
        "event_seed": event_seed,
        "event_seed_role": "sealed independent-training-seed evaluation",
        "family": protocol.FAMILY,
        "env": protocol.ENV,
        "decision_variant": protocol.DECISION_VARIANT,
        "episodes_per_task": AUDIT_EPISODES,
        "slow_dwell_steps": SLOW_DWELL_STEPS,
        "full_cycle_dwell_steps": FULL_CYCLE_DWELL_STEPS,
        "oracle_controller_map": list(router_parts[-1]),
        "utility_table": protocol.file_record(protocol.calibration_path(seed)),
        "router_manifest": protocol.file_record(protocol.ROUTER_MANIFEST),
        "router_params": protocol.file_record(protocol.ROUTER_PARAMS),
        "bundle_manifests": {
            path.parent.name: protocol.file_record(path)
            for path in protocol.bundle_paths_for_seed(seed)
        },
        "stationary": stationary,
        "switching": switching,
    }
    validate_audit(payload, seed, event_seed)
    protocol.write_json_atomic(output, payload)
    print(f"BAPR V8 AUDIT COMPLETE: {output}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("calibration", "audit"),
                        required=True)
    parser.add_argument("--seed", type=int, choices=protocol.TRAINING_SEEDS,
                        required=True)
    parser.add_argument("--event-seed", type=int,
                        choices=protocol.EVALUATION_EVENT_SEEDS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.phase == "calibration":
        if args.event_seed is not None:
            parser.error("calibration does not accept --event-seed")
        calibrate(args.seed)
    else:
        if args.event_seed is None:
            parser.error("audit requires --event-seed")
        audit(args.seed, args.event_seed)


if __name__ == "__main__":
    main()
