"""Strict event-matched audit for one corrected polarity baseline."""
from __future__ import annotations

import argparse
import copy
import hashlib
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
from jax_experiments.analysis import (
    regime_polarity_corrected_baselines_v2 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_corrected_baseline_v2 as trainer,
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


class AgentRuntime:
    """One causal sequential policy interface for both corrected methods."""

    def __init__(self, agent, config):
        self.agent = agent
        self.recurrent = bool(getattr(
            agent, "uses_recurrent_context", False))
        self._ema_action = None
        if bool(getattr(config, "use_ema_eval", False)):
            if not hasattr(agent, "ema_policy"):
                raise ValueError("EMA evaluation requested without EMA policy")
            graphdef = nnx.graphdef(agent.ema_policy)
            params = nnx.state(agent.ema_policy, nnx.Param)

            @jax.jit
            def action(observation):
                policy = nnx.merge(graphdef, params)
                return policy.deterministic(observation[None])[0]

            self._ema_action = action

    def reset(self):
        if self.recurrent:
            self.agent.reset_recurrent_context()

    def select_action(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        if self._ema_action is not None:
            return np.asarray(
                self._ema_action(jnp.asarray(observation)),
                dtype=np.float32)
        return np.asarray(
            self.agent.select_action(observation, deterministic=True),
            dtype=np.float32)

    def observe_done(self, done):
        if self.recurrent:
            self.agent.finish_recurrent_step(done)


def _load_runtime(method: str, seed: int):
    trainer.validate_bundle(method, seed)
    directory = protocol.bundle_dir(method, seed)
    config = final_task_sweep.load_config(directory)
    expected = trainer.expected_config(method, seed)
    mismatches = {
        key: {"actual": getattr(config, key, None), "expected": value}
        for key, value in expected.items()
        if getattr(config, key, None) != value
    }
    if mismatches:
        raise ValueError(f"loaded corrected config changed: {mismatches}")
    env = make_env(config, seed_offset=0)
    train_tasks = env.sample_tasks(config.task_num)
    agent = make_algo(config.algo, env.obs_dim, env.act_dim, config)
    if hasattr(agent, "set_task_metadata"):
        agent.set_task_metadata(train_tasks)
    replay = ReplayBuffer(env.obs_dim, env.act_dim, capacity=1)
    with tempfile.TemporaryDirectory() as log_dir:
        logger = Logger(log_dir)
        next_iteration, total_steps = load_checkpoint(
            str(directory / "checkpoints"), agent, replay, logger,
            config.algo, load_replay_buffer=False)
    if (int(next_iteration) != protocol.MAX_ITERS
            or int(total_steps) != protocol.FINAL_TOTAL_STEPS
            or int(agent.update_count) != protocol.FINAL_UPDATE_COUNT):
        raise ValueError(f"stale corrected checkpoint: {directory}")
    if hasattr(env, "close"):
        env.close()
    return config, AgentRuntime(agent, config)


def _stationary(config, runtime: AgentRuntime, event_seed: int):
    rows = []
    for mode in protocol.MODES:
        run_config = copy.deepcopy(config)
        run_config.stochastic_mode_fixed_id = int(mode)
        env = make_env(
            run_config,
            seed_offset=int(event_seed) - int(config.seed))
        tasks = env.sample_tasks(len(protocol.MODES))
        env.set_nonstationary_para(tasks)
        env.set_task(tasks[int(mode)])
        returns = []
        terminations = []
        try:
            for _ in range(protocol.AUDIT_EPISODES_PER_TASK):
                observation = env.reset()
                runtime.reset()
                episode_return = 0.0
                terminated = False
                for _ in range(protocol.MAX_EPISODE_STEPS):
                    action = runtime.select_action(observation)
                    next_observation, reward, done, info = env.step(action)
                    if int(info["mode_used"]) != int(mode):
                        raise RuntimeError("stationary corrected mode changed")
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
        rows.append({
            "mode": int(mode),
            "returns": returns,
            "return_mean": float(np.mean(returns)),
            "terminated_rate": float(np.mean(terminations)),
        })
    return rows


def _switching(config, runtime: AgentRuntime, event_seed: int):
    run_config = copy.deepcopy(config)
    run_config.stochastic_mode_fixed_id = -1
    env = make_env(
        run_config,
        seed_offset=int(event_seed) - int(config.seed))
    tasks = env.sample_tasks(len(protocol.MODES))
    returns = []
    terminations = []
    trace = []
    try:
        for episode in range(protocol.AUDIT_SWITCHING_EPISODES):
            sequence, _ = _select_eval_switch_sequence(env, tasks, episode)
            _reset_eval_switch_schedule(
                env, sequence, run_config, protocol.DWELL_STEPS)
            observation = env.reset()
            runtime.reset()
            episode_return = 0.0
            terminated = False
            for _ in range(protocol.MAX_EPISODE_STEPS):
                physical_mode = int(env.task_id_for_next_step())
                trace.append(physical_mode)
                action = runtime.select_action(observation)
                next_observation, reward, done, info = env.step(action)
                if int(info["mode_used"]) != physical_mode:
                    raise RuntimeError("switching corrected mode was not causal")
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
        "return_mean": float(np.mean(returns)),
        "terminated_rate": float(np.mean(terminations)),
        "mode_trace_sha256": hashlib.sha256(bytes(trace)).hexdigest(),
        "total_actions": int(len(trace)),
    }


def evaluate(method: str, seed: int) -> dict[str, Any]:
    config, runtime = _load_runtime(method, seed)
    events = []
    for event_seed in protocol.EVENT_SEEDS:
        events.append({
            "event_seed": int(event_seed),
            "stationary": _stationary(config, runtime, event_seed),
            "switching": _switching(config, runtime, event_seed),
        })
        print(
            f"corrected audit method={method} seed={seed} "
            f"event={event_seed} complete", flush=True)
    return {
        "schema": protocol.AUDIT_SCHEMA,
        "status": "complete",
        "identity": protocol.audit_identity(method, seed),
        "registration": protocol.file_record(protocol.REGISTRATION_PATH),
        "events": events,
    }


def validate_result(payload: dict[str, Any], method: str, seed: int) -> None:
    if (payload.get("schema") != protocol.AUDIT_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("identity")
            != protocol.audit_identity(method, seed)
            or payload.get("registration")
            != protocol.file_record(protocol.REGISTRATION_PATH)):
        raise ValueError("invalid corrected audit identity")
    events = payload.get("events") or []
    if [row.get("event_seed") for row in events] \
            != list(protocol.EVENT_SEEDS):
        raise ValueError("corrected audit event set changed")
    for event in events:
        stationary = event.get("stationary") or []
        switching = event.get("switching") or {}
        if ([row.get("mode") for row in stationary]
                != list(protocol.MODES)
                or any(len(row.get("returns") or [])
                       != protocol.AUDIT_EPISODES_PER_TASK
                       for row in stationary)
                or len(switching.get("returns") or [])
                != protocol.AUDIT_SWITCHING_EPISODES
                or int(switching.get("total_actions", -1))
                != (protocol.AUDIT_SWITCHING_EPISODES
                    * protocol.MAX_EPISODE_STEPS)):
            raise ValueError("incomplete corrected strict audit")


def validate_manifest(method: str, seed: int) -> dict[str, Any]:
    directory = protocol.audit_dir(method, seed)
    manifest = protocol.read_json(directory / "audit_manifest.json")
    results = protocol.read_json(directory / "results.json")
    validate_result(results, method, seed)
    if (manifest.get("schema") != protocol.AUDIT_SCHEMA
            or manifest.get("status") != "complete"
            or manifest.get("identity")
            != protocol.audit_identity(method, seed)
            or manifest.get("registration")
            != protocol.file_record(protocol.REGISTRATION_PATH)
            or manifest.get("result_file")
            != protocol.file_record(directory / "results.json")):
        raise ValueError(f"invalid corrected audit manifest: {directory}")
    return manifest


def run(method: str, seed: int) -> None:
    method = protocol.require_trained_method(method)
    seed = protocol.require_seed(seed)
    protocol.validate_registration()
    destination = protocol.audit_dir(method, seed)
    if protocol.audit_manifest(method, seed).is_file():
        validate_manifest(method, seed)
        print(f"CORRECTED AUDIT ALREADY COMPLETE: {destination}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp.", dir=destination.parent))
    try:
        payload = evaluate(method, seed)
        protocol.write_json_atomic(temporary / "results.json", payload)
        protocol.write_json_atomic(temporary / "audit_manifest.json", {
            "schema": protocol.AUDIT_SCHEMA,
            "status": "complete",
            "identity": protocol.audit_identity(method, seed),
            "registration": protocol.file_record(
                protocol.REGISTRATION_PATH),
            "result_file": protocol.file_record(
                temporary / "results.json"),
        })
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_manifest(method, seed)
    print(f"CORRECTED AUDIT COMPLETE: method={method} seed={seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", choices=protocol.TRAINED_METHODS, required=True)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS,
        type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.method, args.seed)


if __name__ == "__main__":
    main()
