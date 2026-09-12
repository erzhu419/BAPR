"""Strict final audit of frozen v5 BAPR and matched v18 baselines."""
from __future__ import annotations

import argparse
import copy
import hashlib
import math
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
from flax import nnx

from jax_experiments.analysis import final_task_sweep
from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_model_v5 as v5_model,
)
from jax_experiments.analysis import (
    regime_polarity_v5_final_comparison_v18 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_corrected_audit_v2 as corrected_audit,
)
from jax_experiments.analysis import (
    run_regime_polarity_fresh_bank_estimator_confirmation_audit_v17 as frozen_audit,
)
from jax_experiments.analysis import (
    run_regime_polarity_v5_final_baseline_v18 as trainer,
)
from jax_experiments.common.checkpoint import (
    _patch_flax_variablestate_unpickle,
    _restore_tree_like,
)
from jax_experiments.train import make_algo, make_env


Action = Callable[[np.ndarray], np.ndarray]


def _mean(values) -> float:
    return float(np.mean([float(value) for value in values]))


def _load_pickle(path: Path) -> dict[str, Any]:
    _patch_flax_variablestate_unpickle()
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a parameter dictionary: {path}")
    return payload


def _load_frozen_controllers(seed: int) -> dict[str, dict[str, Any]]:
    seed = protocol.require_training_seed(seed)
    frozen_audit._bind_policy_audit()
    v5_model.load_model(17, 6)
    return frozen_audit.policy_audit._load_controllers(seed)


def _load_sac_replica(
    seed: int,
    slot: int,
    reference: dict[str, Any],
) -> dict[str, Any]:
    trainer.validate_bundle("sac_replica", seed, slot)
    directory = protocol.sac_bundle_dir(seed, slot)
    config = final_task_sweep.load_config(directory)
    expected = trainer.expected_config("sac_replica", seed, slot)
    mismatches = {
        key: {"actual": getattr(config, key, None), "expected": value}
        for key, value in expected.items()
        if getattr(config, key, None) != value
    }
    if mismatches:
        raise ValueError(f"loaded v18 SAC config changed: {mismatches}")
    env = make_env(config, seed_offset=0)
    try:
        tasks = env.sample_tasks(len(protocol.MODES))
        agent = make_algo("sac", env.obs_dim, env.act_dim, config)
        if hasattr(agent, "set_task_metadata"):
            agent.set_task_metadata(tasks)
        payload = _load_pickle(
            directory / "runtime" / protocol.EVAL_PARAMS_NAME)
        if payload.get("identity") != protocol.identity(
                "sac_replica", seed, slot):
            raise ValueError("v18 SAC evaluation identity changed")
        params = _restore_tree_like(
            nnx.state(agent.policy, nnx.Param),
            payload["policy"],
            "v18 SAC replica policy",
            allow_fallback=False,
        )
        return {
            "config": config,
            "agent": agent,
            "policy_graphdef": nnx.graphdef(agent.policy),
            "policy_params": params,
            "context_graphdef": None,
            "context_params": None,
        }
    finally:
        if hasattr(env, "close"):
            env.close()


def _load_sac5_controllers(
    seed: int,
    frozen_controllers: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    controllers = {"robust_sac": frozen_controllers["robust_sac"]}
    reference = frozen_controllers["robust_sac"]
    for slot in protocol.SAC_REPLICA_SLOTS:
        controllers[f"sac_replica_{slot}"] = _load_sac_replica(
            seed, slot, reference)
    return controllers


def _load_baseline_runtime(method: str, seed: int):
    method = protocol.require_method(method)
    seed = protocol.require_training_seed(seed)
    trainer.validate_bundle(method, seed)
    directory = protocol.baseline_bundle_dir(method, seed)
    config = final_task_sweep.load_config(directory)
    expected = trainer.expected_config(method, seed)
    mismatches = {
        key: {"actual": getattr(config, key, None), "expected": value}
        for key, value in expected.items()
        if getattr(config, key, None) != value
    }
    if mismatches:
        raise ValueError(f"loaded v18 baseline config changed: {mismatches}")
    env = make_env(config, seed_offset=0)
    try:
        tasks = env.sample_tasks(len(protocol.MODES))
        agent = make_algo(protocol.algo_for(method), env.obs_dim, env.act_dim, config)
        if hasattr(agent, "set_task_metadata"):
            agent.set_task_metadata(tasks)
        payload = _load_pickle(
            directory / "runtime" / protocol.EVAL_PARAMS_NAME)
        if payload.get("identity") != protocol.identity(method, seed):
            raise ValueError("v18 baseline evaluation identity changed")
        nnx.update(agent.policy, _restore_tree_like(
            nnx.state(agent.policy, nnx.Param),
            payload["policy"],
            "v18 baseline policy",
            allow_fallback=False,
        ))
        if method == "escp_recurrent":
            nnx.update(agent.context_net, _restore_tree_like(
                nnx.state(agent.context_net, nnx.Param),
                payload["context_net"],
                "v18 ESCP context",
                allow_fallback=False,
            ))
            agent.load_checkpoint_state(payload["custom_agent_state"])
        else:
            nnx.update(agent.ema_policy, _restore_tree_like(
                nnx.state(agent.ema_policy, nnx.Param),
                payload["ema_policy"],
                "v18 RE-SAC EMA policy",
                allow_fallback=False,
            ))
        agent.update_count = int(payload["update_count"])
        if agent.update_count != protocol.FINAL_UPDATE_COUNT:
            raise ValueError("v18 baseline evaluation update count changed")
    finally:
        if hasattr(env, "close"):
            env.close()
    return config, corrected_audit.AgentRuntime(agent, config)


def _action_stacks(
    frozen_controllers: dict[str, dict[str, Any]],
    sac5_controllers: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    stack_fn = frozen_audit.policy_audit.utility_audit._stack_from_controllers
    return stack_fn(frozen_controllers), stack_fn(sac5_controllers)


def _stationary_direct(
    config,
    action: Action,
    event_seed: int,
    mode: int,
    episodes: int,
) -> dict[str, Any]:
    run_config = copy.deepcopy(config)
    run_config.stochastic_mode_fixed_id = int(mode)
    env = make_env(
        run_config, seed_offset=int(event_seed) - int(config.seed))
    tasks = env.sample_tasks(len(protocol.MODES))
    env.set_nonstationary_para(tasks)
    env.set_task(tasks[int(mode)])
    returns = []
    terminations = []
    try:
        for _ in range(int(episodes)):
            observation = env.reset()
            episode_return = 0.0
            terminated = False
            for _ in range(protocol.MAX_EPISODE_STEPS):
                next_observation, reward, done, info = env.step(
                    action(observation))
                if int(info["mode_used"]) != int(mode):
                    raise RuntimeError("v18 stationary mode changed")
                episode_return += float(reward)
                observation = next_observation
                if done:
                    terminated = True
                    observation = env.reset()
            returns.append(float(episode_return))
            terminations.append(float(terminated))
    finally:
        if hasattr(env, "close"):
            env.close()
    return {
        "returns": returns,
        "return_mean": _mean(returns),
        "terminated_rate": _mean(terminations),
        "total_actions": int(episodes) * protocol.MAX_EPISODE_STEPS,
    }


def _stationary_runtime(
    config,
    runtime: corrected_audit.AgentRuntime,
    event_seed: int,
    mode: int,
) -> dict[str, Any]:
    run_config = copy.deepcopy(config)
    run_config.stochastic_mode_fixed_id = int(mode)
    env = make_env(
        run_config, seed_offset=int(event_seed) - int(config.seed))
    tasks = env.sample_tasks(len(protocol.MODES))
    env.set_nonstationary_para(tasks)
    env.set_task(tasks[int(mode)])
    returns = []
    terminations = []
    try:
        for _ in range(protocol.STATIONARY_EPISODES_PER_MODE):
            observation = env.reset()
            runtime.reset()
            episode_return = 0.0
            terminated = False
            for _ in range(protocol.MAX_EPISODE_STEPS):
                action = runtime.select_action(observation)
                next_observation, reward, done, info = env.step(action)
                if int(info["mode_used"]) != int(mode):
                    raise RuntimeError("v18 baseline stationary mode changed")
                runtime.observe_done(done)
                episode_return += float(reward)
                observation = next_observation
                if done:
                    terminated = True
                    observation = env.reset()
            returns.append(float(episode_return))
            terminations.append(float(terminated))
    finally:
        if hasattr(env, "close"):
            env.close()
    return {
        "returns": returns,
        "return_mean": _mean(returns),
        "terminated_rate": _mean(terminations),
        "total_actions": (
            protocol.STATIONARY_EPISODES_PER_MODE
            * protocol.MAX_EPISODE_STEPS),
    }


def _mapping_controller(mapping: dict[str, Any], mode: int) -> str:
    selected = mapping[str(int(mode))]
    if isinstance(selected, dict):
        selected = selected["controller"]
    return str(selected)


def _routed_stationary(
    stack: dict[str, Any],
    mapping: dict[str, Any],
    route: str,
    event_seed: int,
    mode: int,
) -> dict[str, Any]:
    if route not in {"true_mode", "v5_posterior"}:
        raise ValueError(f"unsupported v18 stationary route {route!r}")
    config = copy.deepcopy(stack["config"])
    config.stochastic_mode_fixed_id = int(mode)
    env = make_env(
        config, seed_offset=int(event_seed) - int(config.seed))
    tasks = env.sample_tasks(len(protocol.MODES))
    env.set_nonstationary_para(tasks)
    env.set_task(tasks[int(mode)])
    actions = stack["actions"]
    reference = stack.get("reference_agent")
    if reference is None:
        obs_dim = int(env.obs_dim)
        act_dim = int(env.act_dim)
    else:
        obs_dim = int(reference.obs_dim)
        act_dim = int(reference.act_dim)
    estimator = (
        v5_model.make_estimator(obs_dim, act_dim)
        if route == "v5_posterior" else None)
    returns = []
    terminations = []
    correct = 0
    routed = 0
    posterior_rows = []
    try:
        for _ in range(protocol.STATIONARY_EPISODES_PER_MODE):
            observation = env.reset()
            estimator_state = (
                estimator.initial_state() if estimator is not None else None)
            episode_return = 0.0
            terminated = False
            for _ in range(protocol.MAX_EPISODE_STEPS):
                if estimator is None:
                    route_mode = int(mode)
                else:
                    posterior = np.asarray(
                        estimator.probabilities(estimator_state),
                        dtype=np.float64,
                    )
                    posterior_rows.append(posterior.copy())
                    route_mode = int(np.argmax(posterior))
                controller = _mapping_controller(mapping, route_mode)
                action = actions[controller](observation)
                next_observation, reward, done, info = env.step(action)
                if int(info["mode_used"]) != int(mode):
                    raise RuntimeError("v18 routed stationary mode changed")
                if estimator is not None:
                    estimator_state, _, _, _ = estimator.step(
                        estimator_state,
                        observation,
                        action,
                        reward,
                        next_observation,
                    )
                correct += int(route_mode == int(mode))
                routed += 1
                episode_return += float(reward)
                observation = next_observation
                if done:
                    terminated = True
                    observation = env.reset()
            returns.append(float(episode_return))
            terminations.append(float(terminated))
    finally:
        if hasattr(env, "close"):
            env.close()
    result = {
        "returns": returns,
        "return_mean": _mean(returns),
        "terminated_rate": _mean(terminations),
        "total_actions": int(routed),
        "routing_mode_accuracy": float(correct / max(routed, 1)),
    }
    if posterior_rows:
        posterior = np.asarray(posterior_rows, dtype=np.float64)
        target = np.eye(len(protocol.MODES), dtype=np.float64)[int(mode)]
        result["posterior_metrics"] = {
            "mode_accuracy": float(correct / max(routed, 1)),
            "brier_score": float(np.mean(np.sum(
                (posterior - target[None, :]) ** 2, axis=1))),
        }
    return result


def _aggregate_calibration(
    events: dict[str, Any], roles: tuple[str, ...],
) -> dict[str, Any]:
    matrix: dict[str, Any] = {role: {} for role in roles}
    for role in roles:
        for mode in protocol.MODES:
            rows = [
                events[str(event_seed)][role][str(mode)]
                for event_seed in protocol.CALIBRATION_EVENT_SEEDS
            ]
            matrix[role][str(mode)] = {
                "mean": _mean(row["return_mean"] for row in rows),
                "terminated_rate": _mean(
                    row["terminated_rate"] for row in rows),
                "event_returns": {
                    str(event_seed): float(row["return_mean"])
                    for event_seed, row in zip(
                        protocol.CALIBRATION_EVENT_SEEDS, rows)
                },
            }
    return matrix


def _bapr_calibration(
    stack: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    events: dict[str, Any] = {}
    for event_seed in protocol.CALIBRATION_EVENT_SEEDS:
        events[str(event_seed)] = {
            "robust_sac": {},
            "matching_specialist": {},
        }
        for mode in protocol.MODES:
            events[str(event_seed)]["robust_sac"][str(mode)] = (
                _stationary_direct(
                    stack["config"], stack["actions"]["robust_sac"],
                    event_seed, mode, protocol.CALIBRATION_EPISODES_PER_MODE))
            events[str(event_seed)]["matching_specialist"][str(mode)] = (
                _stationary_direct(
                    stack["config"], stack["actions"][f"specialist_{mode}"],
                    event_seed, mode, protocol.CALIBRATION_EPISODES_PER_MODE))
    matrix = _aggregate_calibration(
        events, ("robust_sac", "matching_specialist"))
    mapping = {}
    for mode in protocol.MODES:
        robust = matrix["robust_sac"][str(mode)]
        specialist = matrix["matching_specialist"][str(mode)]
        gain = (
            (specialist["mean"] - robust["mean"]) / abs(robust["mean"])
            if robust["mean"] != 0.0 else float("-inf"))
        event_wins = sum(
            specialist["event_returns"][str(event_seed)]
            > robust["event_returns"][str(event_seed)]
            for event_seed in protocol.CALIBRATION_EVENT_SEEDS)
        use_specialist = bool(
            gain >= protocol.MIN_CALIBRATION_GAIN
            and event_wins == len(protocol.CALIBRATION_EVENT_SEEDS)
            and specialist["terminated_rate"] == 0.0)
        mapping[str(mode)] = {
            "controller": (
                f"specialist_{mode}" if use_specialist else "robust_sac"),
            "relative_gain": float(gain),
            "event_wins": int(event_wins),
            "specialist_terminated_rate": float(
                specialist["terminated_rate"]),
        }
    return events, matrix, mapping


def _sac5_calibration(
    stack: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, str]]:
    roles = ("robust_sac", *(
        f"sac_replica_{slot}" for slot in protocol.SAC_REPLICA_SLOTS))
    events: dict[str, Any] = {}
    for event_seed in protocol.CALIBRATION_EVENT_SEEDS:
        events[str(event_seed)] = {role: {} for role in roles}
        for role in roles:
            for mode in protocol.MODES:
                events[str(event_seed)][role][str(mode)] = _stationary_direct(
                    stack["config"], stack["actions"][role], event_seed, mode,
                    protocol.CALIBRATION_EPISODES_PER_MODE)
    matrix = _aggregate_calibration(events, roles)

    def score(role: str, mode: int | None = None) -> float:
        rows = (
            [matrix[role][str(mode)]] if mode is not None
            else [matrix[role][str(value)] for value in protocol.MODES])
        mean_return = _mean(row["mean"] for row in rows)
        terminated = _mean(row["terminated_rate"] for row in rows)
        return float(mean_return - 1_000_000.0 * terminated)

    best_static = max(roles, key=lambda role: score(role))
    mode_map = {
        str(mode): max(roles, key=lambda role, m=mode: score(role, m))
        for mode in protocol.MODES
    }
    return events, matrix, best_static, mode_map


def _switching_bank(
    stack: dict[str, Any],
    mapping: dict[str, Any],
    route: str,
    event_seed: int,
    *,
    fixed_controller: str | None = None,
) -> dict[str, Any]:
    if route not in {"fixed", "true_mode", "v5_posterior"}:
        raise ValueError(f"unsupported v18 switching route {route!r}")
    event_seed = protocol.require_switching_event_seed(event_seed)
    config = copy.deepcopy(stack["config"])
    config.stochastic_mode_fixed_id = -1
    env = make_env(config, seed_offset=event_seed - int(config.seed))
    tasks = env.sample_tasks(len(protocol.MODES))
    actions = stack["actions"]
    estimator = (
        v5_model.make_estimator(int(env.obs_dim), int(env.act_dim))
        if route == "v5_posterior" else None)
    returns = []
    terminated = []
    trace: list[int] = []
    mode_counts = {str(mode): 0 for mode in protocol.MODES}
    episode_sequences = []
    posterior_rows = []
    posterior_labels = []
    route_correct = 0
    route_actions = 0
    wrong_controller_actions = 0
    total_actions = 0
    try:
        configure = getattr(env, "configure_eval_mode_sequence", None)
        if not callable(configure):
            raise RuntimeError("v18 requires explicit mode schedules")
        for episode in range(protocol.SWITCHING_EPISODES):
            sequence = protocol.switching_sequence(event_seed, episode)
            episode_sequences.append(list(sequence))
            configure(tasks, sequence, protocol.DWELL_STEPS)
            observation = env.reset()
            estimator_state = (
                estimator.initial_state() if estimator is not None else None)
            episode_return = 0.0
            episode_terminated = False
            for _ in range(protocol.MAX_EPISODE_STEPS):
                mode = int(env.task_id_for_next_step())
                trace.append(mode)
                mode_counts[str(mode)] += 1
                if route == "fixed":
                    if fixed_controller is None:
                        raise ValueError("fixed route needs a controller")
                    controller = fixed_controller
                    route_mode = None
                elif route == "true_mode":
                    route_mode = mode
                    controller = _mapping_controller(mapping, route_mode)
                else:
                    posterior = np.asarray(
                        estimator.probabilities(estimator_state),
                        dtype=np.float64)
                    posterior_rows.append(posterior.copy())
                    posterior_labels.append(mode)
                    route_mode = int(np.argmax(posterior))
                    controller = _mapping_controller(mapping, route_mode)
                action = actions[controller](observation)
                next_observation, reward, done, info = env.step(action)
                if int(info["mode_used"]) != mode:
                    raise RuntimeError("v18 switching action used a wrong mode")
                if estimator is not None:
                    estimator_state, _, _, _ = estimator.step(
                        estimator_state, observation, action, reward,
                        next_observation)
                if route_mode is not None:
                    route_correct += int(route_mode == mode)
                    route_actions += 1
                    wrong_controller_actions += int(
                        controller != _mapping_controller(mapping, mode))
                total_actions += 1
                episode_return += float(reward)
                observation = next_observation
                if done:
                    episode_terminated = True
                    observation = env.reset()
            returns.append(float(episode_return))
            terminated.append(float(episode_terminated))
    finally:
        if hasattr(env, "close"):
            env.close()
    row: dict[str, Any] = {
        "returns": returns,
        "return_mean": _mean(returns),
        "terminated_rate": _mean(terminated),
        "total_actions": int(total_actions),
        "mode_trace_sha256": hashlib.sha256(bytes(trace)).hexdigest(),
        "mode_counts": mode_counts,
        "base_schedule": list(protocol.SWITCHING_SCHEDULES[event_seed]),
        "episode_sequences": episode_sequences,
        "routing_mode_accuracy": (
            float(route_correct / route_actions) if route_actions else None),
        "controller_mismatch_action_fraction": float(
            wrong_controller_actions / max(total_actions, 1)),
    }
    if posterior_rows:
        posterior = np.asarray(posterior_rows, dtype=np.float64)
        labels = np.asarray(posterior_labels, dtype=np.int32)
        target = np.eye(len(protocol.MODES), dtype=np.float64)[labels]
        row["posterior_metrics"] = {
            "mode_accuracy": float(
                np.mean(np.argmax(posterior, axis=1) == labels)),
            "brier_score": float(
                np.mean(np.sum((posterior - target) ** 2, axis=1))),
        }
    return row


def _switching_runtime(
    config,
    runtime: corrected_audit.AgentRuntime,
    event_seed: int,
) -> dict[str, Any]:
    event_seed = protocol.require_switching_event_seed(event_seed)
    run_config = copy.deepcopy(config)
    run_config.stochastic_mode_fixed_id = -1
    env = make_env(
        run_config, seed_offset=event_seed - int(config.seed))
    tasks = env.sample_tasks(len(protocol.MODES))
    returns = []
    terminations = []
    trace: list[int] = []
    mode_counts = {str(mode): 0 for mode in protocol.MODES}
    episode_sequences = []
    try:
        configure = getattr(env, "configure_eval_mode_sequence", None)
        if not callable(configure):
            raise RuntimeError("v18 baseline requires explicit mode schedules")
        for episode in range(protocol.SWITCHING_EPISODES):
            sequence = protocol.switching_sequence(event_seed, episode)
            episode_sequences.append(list(sequence))
            configure(tasks, sequence, protocol.DWELL_STEPS)
            observation = env.reset()
            runtime.reset()
            episode_return = 0.0
            terminated = False
            for _ in range(protocol.MAX_EPISODE_STEPS):
                mode = int(env.task_id_for_next_step())
                trace.append(mode)
                mode_counts[str(mode)] += 1
                action = runtime.select_action(observation)
                next_observation, reward, done, info = env.step(action)
                if int(info["mode_used"]) != mode:
                    raise RuntimeError("v18 baseline switching mode changed")
                runtime.observe_done(done)
                episode_return += float(reward)
                observation = next_observation
                if done:
                    terminated = True
                    observation = env.reset()
            returns.append(float(episode_return))
            terminations.append(float(terminated))
    finally:
        if hasattr(env, "close"):
            env.close()
    return {
        "returns": returns,
        "return_mean": _mean(returns),
        "terminated_rate": _mean(terminations),
        "total_actions": len(trace),
        "mode_trace_sha256": hashlib.sha256(bytes(trace)).hexdigest(),
        "mode_counts": mode_counts,
        "base_schedule": list(protocol.SWITCHING_SCHEDULES[event_seed]),
        "episode_sequences": episode_sequences,
        "routing_mode_accuracy": None,
        "controller_mismatch_action_fraction": 0.0,
    }


def _stationary_holdout(
    bapr_stack: dict[str, Any],
    sac5_stack: dict[str, Any],
    bapr_map: dict[str, Any],
    sac5_static: str,
    sac5_map: dict[str, str],
    baseline_runtimes: dict[str, tuple[Any, corrected_audit.AgentRuntime]],
) -> dict[str, Any]:
    events = {}
    for event_seed in protocol.STATIONARY_HOLDOUT_EVENT_SEEDS:
        arms = {arm: {} for arm in protocol.STATIONARY_ARMS}
        for mode in protocol.MODES:
            key = str(mode)
            arms["robust_sac"][key] = _stationary_direct(
                bapr_stack["config"], bapr_stack["actions"]["robust_sac"],
                event_seed, mode, protocol.STATIONARY_EPISODES_PER_MODE)
            arms["bapr_true_mode_oracle"][key] = _routed_stationary(
                bapr_stack, bapr_map, "true_mode", event_seed, mode)
            arms[protocol.PRIMARY_ARM][key] = _routed_stationary(
                bapr_stack, bapr_map, "v5_posterior", event_seed, mode)
            for method in protocol.TRAINED_METHODS:
                config, runtime = baseline_runtimes[method]
                arms[method][key] = _stationary_runtime(
                    config, runtime, event_seed, mode)
            arms["sac5_best_static"][key] = _stationary_direct(
                sac5_stack["config"], sac5_stack["actions"][sac5_static],
                event_seed, mode, protocol.STATIONARY_EPISODES_PER_MODE)
            arms["sac5_true_mode_oracle"][key] = _routed_stationary(
                sac5_stack, sac5_map, "true_mode", event_seed, mode)
            arms[protocol.EQUAL_POLICY_BUDGET_BASELINE][key] = (
                _routed_stationary(
                    sac5_stack, sac5_map, "v5_posterior", event_seed, mode))
        events[str(event_seed)] = arms
        print(f"v18 stationary event={event_seed} complete", flush=True)
    return events


def _switching_holdout(
    bapr_stack: dict[str, Any],
    sac5_stack: dict[str, Any],
    bapr_map: dict[str, Any],
    sac5_static: str,
    sac5_map: dict[str, str],
    baseline_runtimes: dict[str, tuple[Any, corrected_audit.AgentRuntime]],
) -> dict[str, Any]:
    events = {}
    event_hashes = set()
    for event_seed in protocol.SWITCHING_EVENT_SEEDS:
        rows = {
            "robust_sac": _switching_bank(
                bapr_stack, bapr_map, "fixed", event_seed,
                fixed_controller="robust_sac"),
            "bapr_true_mode_oracle": _switching_bank(
                bapr_stack, bapr_map, "true_mode", event_seed),
            protocol.PRIMARY_ARM: _switching_bank(
                bapr_stack, bapr_map, "v5_posterior", event_seed),
            "escp_recurrent": _switching_runtime(
                *baseline_runtimes["escp_recurrent"], event_seed),
            "resac_b0": _switching_runtime(
                *baseline_runtimes["resac_b0"], event_seed),
            "sac5_best_static": _switching_bank(
                sac5_stack, sac5_map, "fixed", event_seed,
                fixed_controller=sac5_static),
            "sac5_true_mode_oracle": _switching_bank(
                sac5_stack, sac5_map, "true_mode", event_seed),
            protocol.EQUAL_POLICY_BUDGET_BASELINE: _switching_bank(
                sac5_stack, sac5_map, "v5_posterior", event_seed),
        }
        hashes = {str(row["mode_trace_sha256"]) for row in rows.values()}
        if len(hashes) != 1:
            raise RuntimeError("v18 switching arms used different mode streams")
        event_hashes.update(hashes)
        events[str(event_seed)] = rows
        print(f"v18 switching event={event_seed} complete", flush=True)
    if len(event_hashes) != len(protocol.SWITCHING_EVENT_SEEDS):
        raise RuntimeError("v18 switching streams are not distinct")
    return events


def _identity(seed: int) -> dict[str, Any]:
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "benchmark_role": "frozen_v5_final_equal_policy_budget_audit",
        "training_seed": protocol.require_training_seed(seed),
        "calibration_event_seeds": list(protocol.CALIBRATION_EVENT_SEEDS),
        "stationary_holdout_event_seeds": list(
            protocol.STATIONARY_HOLDOUT_EVENT_SEEDS),
        "switching_event_seeds": list(protocol.SWITCHING_EVENT_SEEDS),
        "switching_schedules": {
            str(key): list(value)
            for key, value in protocol.SWITCHING_SCHEDULES.items()
        },
        "stationary_episodes_per_mode": (
            protocol.STATIONARY_EPISODES_PER_MODE),
        "switching_episodes": protocol.SWITCHING_EPISODES,
        "arms": list(protocol.SWITCHING_ARMS),
    }


def evaluate(seed: int) -> dict[str, Any]:
    seed = protocol.require_training_seed(seed)
    frozen_controllers = _load_frozen_controllers(seed)
    sac5_controllers = _load_sac5_controllers(seed, frozen_controllers)
    bapr_stack, sac5_stack = _action_stacks(
        frozen_controllers, sac5_controllers)
    bapr_stack["reference_agent"] = frozen_controllers["robust_sac"]["agent"]
    sac5_stack["reference_agent"] = sac5_controllers["robust_sac"]["agent"]
    baseline_runtimes = {
        method: _load_baseline_runtime(method, seed)
        for method in protocol.TRAINED_METHODS
    }
    bapr_events, bapr_matrix, bapr_map = _bapr_calibration(bapr_stack)
    sac5_events, sac5_matrix, sac5_static, sac5_map = _sac5_calibration(
        sac5_stack)
    return {
        "schema": protocol.AUDIT_SCHEMA,
        "status": "complete",
        "identity": _identity(seed),
        "registration": protocol.file_record(protocol.REGISTRATION_PATH),
        "frozen_policy_bundles": protocol.frozen_policy_records(seed),
        "new_baseline_bundles": protocol.new_bundle_records(seed),
        "v5_estimator": {
            "manifest": protocol.file_record(v5_model.MODEL_MANIFEST),
            "parameters": protocol.file_record(v5_model.MODEL_PATH),
        },
        "calibration": {
            "bapr_events": bapr_events,
            "bapr_matrix": bapr_matrix,
            "bapr_utility_map": bapr_map,
            "sac5_events": sac5_events,
            "sac5_matrix": sac5_matrix,
            "sac5_best_static": sac5_static,
            "sac5_mode_map": sac5_map,
        },
        "stationary_holdout": _stationary_holdout(
            bapr_stack, sac5_stack, bapr_map, sac5_static, sac5_map,
            baseline_runtimes),
        "switching_holdout": _switching_holdout(
            bapr_stack, sac5_stack, bapr_map, sac5_static, sac5_map,
            baseline_runtimes),
    }


def validate_result(payload: dict[str, Any], seed: int) -> None:
    seed = protocol.require_training_seed(seed)
    if (
        payload.get("schema") != protocol.AUDIT_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("identity") != _identity(seed)
        or payload.get("registration")
        != protocol.file_record(protocol.REGISTRATION_PATH)
        or payload.get("frozen_policy_bundles")
        != protocol.frozen_policy_records(seed)
        or payload.get("new_baseline_bundles")
        != protocol.new_bundle_records(seed)
    ):
        raise ValueError("invalid v18 final audit identity")
    calibration = payload.get("calibration") or {}
    if (
        set(calibration.get("bapr_utility_map") or {})
        != {str(mode) for mode in protocol.MODES}
        or set(calibration.get("sac5_mode_map") or {})
        != {str(mode) for mode in protocol.MODES}
        or calibration.get("sac5_best_static") not in {
            "robust_sac", *(
                f"sac_replica_{slot}"
                for slot in protocol.SAC_REPLICA_SLOTS)}
    ):
        raise ValueError("invalid v18 calibration selection")
    stationary = payload.get("stationary_holdout") or {}
    if set(stationary) != {
        str(seed) for seed in protocol.STATIONARY_HOLDOUT_EVENT_SEEDS
    }:
        raise ValueError("incomplete v18 stationary holdout")
    for event in stationary.values():
        if set(event) != set(protocol.STATIONARY_ARMS):
            raise ValueError("incomplete v18 stationary arms")
        for arm in protocol.STATIONARY_ARMS:
            if set(event[arm]) != {str(mode) for mode in protocol.MODES}:
                raise ValueError("incomplete v18 stationary modes")
            for row in event[arm].values():
                returns = row.get("returns") or []
                if (
                    len(returns) != protocol.STATIONARY_EPISODES_PER_MODE
                    or not all(math.isfinite(float(value)) for value in returns)
                    or int(row.get("total_actions", -1))
                    != (protocol.STATIONARY_EPISODES_PER_MODE
                        * protocol.MAX_EPISODE_STEPS)
                ):
                    raise ValueError("invalid v18 stationary rollout")
    switching = payload.get("switching_holdout") or {}
    if set(switching) != {
        str(seed) for seed in protocol.SWITCHING_EVENT_SEEDS
    }:
        raise ValueError("incomplete v18 switching holdout")
    event_hashes = set()
    expected_count = (
        protocol.SWITCHING_EPISODES * protocol.MAX_EPISODE_STEPS
        // len(protocol.MODES))
    for event_seed, event in switching.items():
        if set(event) != set(protocol.SWITCHING_ARMS):
            raise ValueError("incomplete v18 switching arms")
        hashes = set()
        for arm, row in event.items():
            returns = row.get("returns") or []
            if (
                len(returns) != protocol.SWITCHING_EPISODES
                or not all(math.isfinite(float(value)) for value in returns)
                or int(row.get("total_actions", -1))
                != protocol.SWITCHING_EPISODES * protocol.MAX_EPISODE_STEPS
                or row.get("mode_counts") != {
                    str(mode): expected_count for mode in protocol.MODES}
                or row.get("base_schedule")
                != list(protocol.SWITCHING_SCHEDULES[int(event_seed)])
            ):
                raise ValueError("invalid v18 switching rollout")
            hashes.add(str(row.get("mode_trace_sha256") or ""))
            if arm in {
                protocol.PRIMARY_ARM,
                protocol.EQUAL_POLICY_BUDGET_BASELINE,
            }:
                posterior = row.get("posterior_metrics") or {}
                if not all(math.isfinite(float(posterior.get(key, math.nan)))
                           for key in ("mode_accuracy", "brier_score")):
                    raise ValueError("v18 posterior metrics are missing")
        if len(hashes) != 1 or "" in hashes:
            raise ValueError("v18 switching streams differ across arms")
        if event["bapr_true_mode_oracle"]["routing_mode_accuracy"] != 1.0:
            raise ValueError("v18 BAPR oracle routed a wrong mode")
        if event["sac5_true_mode_oracle"]["routing_mode_accuracy"] != 1.0:
            raise ValueError("v18 SAC5 oracle routed a wrong mode")
        event_hashes.update(hashes)
    if len(event_hashes) != len(protocol.SWITCHING_EVENT_SEEDS):
        raise ValueError("v18 switching event traces are not distinct")


def validate_audit(seed: int) -> dict[str, Any]:
    seed = protocol.require_training_seed(seed)
    manifest = protocol.read_json(protocol.audit_manifest(seed))
    payload = protocol.read_json(protocol.audit_result(seed))
    validate_result(payload, seed)
    if (
        manifest.get("schema") != protocol.AUDIT_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("identity") != _identity(seed)
        or manifest.get("registration")
        != protocol.file_record(protocol.REGISTRATION_PATH)
        or manifest.get("frozen_policy_bundles")
        != protocol.frozen_policy_records(seed)
        or manifest.get("new_baseline_bundles")
        != protocol.new_bundle_records(seed)
        or manifest.get("audit")
        != protocol.file_record(protocol.audit_result(seed))
    ):
        raise ValueError("invalid v18 final audit manifest")
    return manifest


def run(seed: int) -> None:
    seed = protocol.require_training_seed(seed)
    protocol.validate_registration()
    destination = protocol.audit_dir(seed)
    if protocol.audit_manifest(seed).is_file():
        try:
            validate_audit(seed)
        except (KeyError, OSError, TypeError, ValueError):
            pass
        else:
            print(f"V18 FINAL AUDIT ALREADY COMPLETE: seed={seed}", flush=True)
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
        payload = evaluate(seed)
        protocol.write_json_atomic(temporary / "audit.json", payload)
        protocol.write_json_atomic(
            temporary / "audit_manifest.json",
            {
                "schema": protocol.AUDIT_SCHEMA,
                "status": "complete",
                "identity": _identity(seed),
                "registration": protocol.file_record(
                    protocol.REGISTRATION_PATH),
                "frozen_policy_bundles": protocol.frozen_policy_records(seed),
                "new_baseline_bundles": protocol.new_bundle_records(seed),
                "audit": protocol.file_record(temporary / "audit.json"),
            },
        )
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_audit(seed)
    print(f"V18 FINAL AUDIT COMPLETE: seed={seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.seed)


if __name__ == "__main__":
    main()
