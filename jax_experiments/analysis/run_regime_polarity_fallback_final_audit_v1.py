"""Strict final audit for one BAPR or baseline model seed."""
from __future__ import annotations

import argparse
import copy
import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.analysis import final_task_sweep
from jax_experiments.analysis import (
    regime_polarity_fallback_final_comparison_v1 as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_final_confirmation as robust_source,
)
from jax_experiments.analysis import (
    regime_polarity_policy_distillation_control_model as model_lib,
)
from jax_experiments.analysis import (
    run_regime_polarity_fallback_final_baseline_v1 as baseline_runner,
)
from jax_experiments.analysis import (
    train_regime_polarity_posterior as controller_loader,
)
from jax_experiments.common.causal_fallback import CausalFallbackPolicy
from jax_experiments.common.checkpoint import load_checkpoint
from jax_experiments.common.logging import Logger
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.train import (
    _reset_eval_switch_schedule,
    _select_eval_switch_sequence,
    make_algo,
    make_env,
)


def _load_student(seed: int, obs_dim: int, act_dim: int):
    seed = protocol.require_student_seed(seed)
    variant = "mode_heads"
    manifest_path = protocol.model_manifest(variant, seed)
    parameter_path = protocol.model_path(variant, seed)
    manifest = protocol.read_json(manifest_path)
    if (manifest.get("schema") != protocol.MODEL_SCHEMA
            or manifest.get("status") != "complete"
            or manifest.get("identity")
            != protocol.model_identity(variant, seed)
            or manifest.get("parameter_file")
            != protocol.file_record(parameter_path)):
        raise ValueError(f"invalid final student model: {manifest_path}")
    model = model_lib.make_student(variant, obs_dim, act_dim, seed)
    params = protocol.load_parameter_state(
        parameter_path,
        nnx.state(model, nnx.Param),
        manifest["parameter_leaves"],
    )
    return model, params, manifest


def _robust_action_fn(agent, params):
    graphdef = nnx.graphdef(agent.policy)
    context = jnp.zeros((len(protocol.MODES),), dtype=jnp.float32)

    @jax.jit
    def action(observation):
        policy = nnx.merge(graphdef, params)
        return policy.deterministic(
            jnp.asarray(observation, dtype=jnp.float32)[None],
            context[None],
        )[0]

    return lambda observation: np.asarray(
        action(observation), dtype=np.float32)


def _load_bapr(seed: int):
    config, robust_agent, robust_params = controller_loader._load_controller(
        robust_source, protocol.ROBUST_SEED, "robust")
    student, student_params, manifest = _load_student(
        seed, robust_agent.obs_dim, robust_agent.act_dim)
    student_action = model_lib.build_student_action(student)

    def adaptive_action(observation, posterior):
        return np.asarray(student_action(
            student_params,
            jnp.asarray(observation, dtype=jnp.float32),
            jnp.asarray(posterior, dtype=jnp.float32),
        ), dtype=np.float32)

    runtime = CausalFallbackPolicy(
        _robust_action_fn(robust_agent, robust_params),
        adaptive_action,
        model_lib.make_estimator(robust_agent.obs_dim, robust_agent.act_dim),
        protocol.FALLBACK_CONFIG,
    )
    return config, runtime, manifest


def _load_baseline(role: str, seed: int):
    role = protocol.require_baseline_role(role)
    seed = protocol.require_training_seed(seed)
    baseline_runner.validate_bundle(role, seed)
    directory = protocol.baseline_bundle_dir(role, seed)
    config = final_task_sweep.load_config(directory)
    expected = baseline_runner.expected_config(role, seed)
    mismatches = {
        key: {"actual": getattr(config, key, None), "expected": value}
        for key, value in expected.items()
        if getattr(config, key, None) != value
    }
    if mismatches:
        raise ValueError(f"loaded final baseline config changed: {mismatches}")
    env = make_env(config, seed_offset=0)
    agent = make_algo(config.algo, env.obs_dim, env.act_dim, config)
    if hasattr(agent, "set_task_metadata"):
        agent.set_task_metadata(env.sample_tasks(config.task_num))
    replay = ReplayBuffer(
        env.obs_dim,
        env.act_dim,
        capacity=1,
        belief_dim=getattr(agent, "belief_dim", 0),
    )
    with tempfile.TemporaryDirectory() as log_dir:
        logger = Logger(log_dir)
        next_iteration, total_steps = load_checkpoint(
            str(directory / "checkpoints"),
            agent,
            replay,
            logger,
            config.algo,
            load_replay_buffer=False,
        )
    if (int(next_iteration) != protocol.MAX_ITERS
            or int(total_steps) != protocol.FINAL_TOTAL_STEPS):
        raise ValueError(f"stale final baseline checkpoint: {directory}")
    if hasattr(env, "close"):
        env.close()
    return config, lambda observation: np.asarray(
        agent.select_action(observation, deterministic=True),
        dtype=np.float32,
    )


def _stationary(
    config,
    action_fn: Callable[[np.ndarray], np.ndarray] | None,
    runtime: CausalFallbackPolicy | None,
    event_seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for mode in protocol.MODES:
        run_config = copy.deepcopy(config)
        run_config.stochastic_mode_fixed_id = int(mode)
        env = make_env(
            run_config,
            seed_offset=int(event_seed) - int(config.seed),
        )
        tasks = env.sample_tasks(len(protocol.MODES))
        env.set_nonstationary_para(tasks)
        env.set_task(tasks[int(mode)])
        returns = []
        terminations = []
        fallback_fractions = []
        try:
            for _ in range(protocol.AUDIT_EPISODES_PER_TASK):
                observation = env.reset()
                if runtime is not None:
                    runtime.reset()
                episode_return = 0.0
                terminated = False
                for _ in range(protocol.MAX_EPISODE_STEPS):
                    action = (
                        runtime.select_action(observation)
                        if runtime is not None else action_fn(observation))
                    next_observation, reward, done, info = env.step(action)
                    if int(info["mode_used"]) != int(mode):
                        raise RuntimeError("stationary final mode changed")
                    if runtime is not None:
                        runtime.observe_transition(
                            observation, action, reward, next_observation)
                    episode_return += float(reward)
                    observation = next_observation
                    if done:
                        terminated = True
                        observation = env.reset()
                returns.append(float(episode_return))
                terminations.append(float(terminated))
                if runtime is not None:
                    fallback_fractions.append(
                        runtime.fallback_action_fraction)
        finally:
            if hasattr(env, "close"):
                env.close()
        row: dict[str, Any] = {
            "mode": int(mode),
            "returns": returns,
            "return_mean": float(np.mean(returns)),
            "terminated_rate": float(np.mean(terminations)),
        }
        if fallback_fractions:
            row["fallback_action_fraction"] = float(
                np.mean(fallback_fractions))
        rows.append(row)
    return rows


def _switching(
    config,
    action_fn: Callable[[np.ndarray], np.ndarray] | None,
    runtime: CausalFallbackPolicy | None,
    event_seed: int,
) -> dict[str, Any]:
    run_config = copy.deepcopy(config)
    run_config.stochastic_mode_fixed_id = -1
    env = make_env(
        run_config,
        seed_offset=int(event_seed) - int(config.seed),
    )
    tasks = env.sample_tasks(len(protocol.MODES))
    returns = []
    terminations = []
    fallback_actions = 0
    total_actions = 0
    trigger_counts = []
    trace = []
    try:
        for episode in range(protocol.AUDIT_SWITCHING_EPISODES):
            sequence, _ = _select_eval_switch_sequence(env, tasks, episode)
            _reset_eval_switch_schedule(
                env, sequence, run_config, protocol.DWELL_STEPS)
            observation = env.reset()
            if runtime is not None:
                runtime.reset()
            episode_return = 0.0
            terminated = False
            for _ in range(protocol.MAX_EPISODE_STEPS):
                physical_mode = int(env.task_id_for_next_step())
                trace.append(physical_mode)
                action = (
                    runtime.select_action(observation)
                    if runtime is not None else action_fn(observation))
                next_observation, reward, done, info = env.step(action)
                if int(info["mode_used"]) != physical_mode:
                    raise RuntimeError("switching final mode was not causal")
                if runtime is not None:
                    runtime.observe_transition(
                        observation, action, reward, next_observation)
                episode_return += float(reward)
                observation = next_observation
                total_actions += 1
                if done:
                    terminated = True
                    observation = env.reset()
            returns.append(float(episode_return))
            terminations.append(float(terminated))
            if runtime is not None:
                fallback_actions += runtime.fallback_action_count
                trigger_counts.append(runtime.gate.state.trigger_count)
    finally:
        if hasattr(env, "close"):
            env.close()
    row: dict[str, Any] = {
        "returns": returns,
        "return_mean": float(np.mean(returns)),
        "terminated_rate": float(np.mean(terminations)),
        "mode_trace_sha256": hashlib.sha256(bytes(trace)).hexdigest(),
        "total_actions": int(total_actions),
    }
    if runtime is not None:
        row.update({
            "fallback_action_fraction": float(
                fallback_actions / max(total_actions, 1)),
            "trigger_count_mean": float(np.mean(trigger_counts)),
        })
    return row


def evaluate(method: str, seed: int) -> dict[str, Any]:
    seed = protocol.require_training_seed(seed)
    if method == "bapr":
        config, runtime, _ = _load_bapr(seed)
        action_fn = None
    else:
        method = protocol.require_baseline_role(method)
        config, action_fn = _load_baseline(method, seed)
        runtime = None
    events = []
    for event_seed in protocol.FINAL_EVENT_SEEDS:
        events.append({
            "event_seed": int(event_seed),
            "stationary": _stationary(
                config, action_fn, runtime, event_seed),
            "switching": _switching(
                config, action_fn, runtime, event_seed),
        })
        print(
            f"final audit method={method} seed={seed} "
            f"event={event_seed} complete",
            flush=True,
        )
    return {
        "schema": protocol.AUDIT_SCHEMA,
        "status": "complete",
        "identity": protocol.audit_identity(method, seed),
        "registration": protocol.FROZEN_REGISTRATION_RECORD,
        "events": events,
    }


def validate_result(payload: dict[str, Any], method: str, seed: int) -> None:
    if (payload.get("schema") != protocol.AUDIT_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("identity")
            != protocol.audit_identity(method, seed)
            or payload.get("registration")
            != protocol.FROZEN_REGISTRATION_RECORD):
        raise ValueError("invalid final audit identity")
    events = payload.get("events") or []
    if [row.get("event_seed") for row in events] \
            != list(protocol.FINAL_EVENT_SEEDS):
        raise ValueError("final audit event set changed")
    for event in events:
        stationary = event.get("stationary") or []
        switching = event.get("switching") or {}
        if ([row.get("mode") for row in stationary] != list(protocol.MODES)
                or any(len(row.get("returns") or [])
                       != protocol.AUDIT_EPISODES_PER_TASK
                       for row in stationary)
                or len(switching.get("returns") or [])
                != protocol.AUDIT_SWITCHING_EPISODES
                or int(switching.get("total_actions", -1))
                != (protocol.AUDIT_SWITCHING_EPISODES
                    * protocol.MAX_EPISODE_STEPS)):
            raise ValueError("incomplete strict final audit")


def validate_manifest(method: str, seed: int) -> dict[str, Any]:
    directory = protocol.audit_dir(method, seed)
    manifest = protocol.read_json(directory / "audit_manifest.json")
    results = protocol.read_json(directory / "results.json")
    validate_result(results, method, seed)
    if (manifest.get("schema") != protocol.AUDIT_SCHEMA
            or manifest.get("status") != "complete"
            or manifest.get("identity")
            != protocol.audit_identity(method, seed)
            or manifest.get("result_file")
            != protocol.file_record(directory / "results.json")):
        raise ValueError(f"invalid final audit manifest: {directory}")
    return manifest


def run(method: str, seed: int) -> None:
    seed = protocol.require_training_seed(seed)
    if method != "bapr":
        method = protocol.require_baseline_role(method)
    protocol.validate_registration()
    destination = protocol.audit_dir(method, seed)
    if (destination / "audit_manifest.json").is_file():
        validate_manifest(method, seed)
        print(f"FINAL AUDIT ALREADY COMPLETE: {destination}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp.", dir=destination.parent))
    try:
        payload = evaluate(method, seed)
        protocol.write_json_atomic(temporary / "results.json", payload)
        manifest = {
            "schema": protocol.AUDIT_SCHEMA,
            "status": "complete",
            "identity": protocol.audit_identity(method, seed),
            "registration": protocol.FROZEN_REGISTRATION_RECORD,
            "result_file": protocol.file_record(temporary / "results.json"),
        }
        protocol.write_json_atomic(
            temporary / "audit_manifest.json", manifest)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_manifest(method, seed)
    print(f"FINAL AUDIT COMPLETE: method={method} seed={seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", choices=("bapr", *protocol.BASELINE_ROLES), required=True)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.method, args.seed)


if __name__ == "__main__":
    main()
