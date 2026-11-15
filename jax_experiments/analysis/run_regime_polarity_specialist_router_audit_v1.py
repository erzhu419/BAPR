"""Audit causal routing over frozen independent specialists."""
from __future__ import annotations

import argparse
import copy
import hashlib
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import jax
import numpy as np

from jax_experiments.analysis import (
    regime_polarity_policy_distillation_control_model as model_lib,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_router_v1 as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_source_headroom_v1 as source,
)
from jax_experiments.analysis import (
    run_regime_polarity_source_headroom_audit_v1 as source_audit,
)
from jax_experiments.train import (
    _reset_eval_switch_schedule,
    _select_eval_switch_sequence,
    make_env,
)


def _load_stack(seed: int) -> dict[str, Any]:
    controllers = {
        role: source_audit._load_controller(role, seed)
        for role in source.ROLES
    }
    actions = {}
    for role, controller in controllers.items():
        action_fn = source_audit._action_fn(controller)

        def action(observation, *, _fn=action_fn, _source=controller):
            return np.asarray(
                _fn(
                    _source["policy_params"],
                    _source["context_params"],
                    observation,
                ),
                dtype=np.float32,
            )

        actions[role] = action
    reference = controllers["robust_sac"]
    return {
        "config": reference["config"],
        "actions": actions,
        "estimator_factory": lambda: model_lib.make_estimator(
            reference["agent"].obs_dim, reference["agent"].act_dim),
    }


def _router(stack: dict[str, Any], arm: str):
    if arm not in protocol.LEARNED_ARMS:
        return None
    reduction = "map" if arm == "posterior_map_fallback" else "soft"
    return protocol.CausalSpecialistRouter(
        stack["actions"]["robust_sac"],
        tuple(
            stack["actions"][f"specialist_{mode}"]
            for mode in protocol.MODES
        ),
        stack["estimator_factory"](),
        reduction,
    )


def _posterior_metrics(rows, labels) -> dict[str, float] | None:
    if not rows:
        return None
    posterior = np.asarray(rows, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    target = np.eye(len(protocol.MODES), dtype=np.float64)[labels]
    return {
        "mode_accuracy": float(np.mean(np.argmax(posterior, axis=1) == labels)),
        "brier_score": float(np.mean(np.sum((posterior - target) ** 2, axis=1))),
    }


def _select_action(stack, router, arm: str, observation, mode: int):
    if arm == "robust_sac":
        return stack["actions"]["robust_sac"](observation), "robust", -1
    if arm == "dynamic_oracle":
        return (
            stack["actions"][f"specialist_{mode}"](observation),
            "specialist",
            int(mode),
        )
    step = router.select_action(observation)
    return step.action, step.source, step.selected_mode


def _stationary_arm(
    stack: dict[str, Any], arm: str, event_seed: int
) -> dict[str, Any]:
    rows = {}
    for mode in protocol.MODES:
        config = copy.deepcopy(stack["config"])
        config.stochastic_mode_fixed_id = int(mode)
        env = make_env(
            config, seed_offset=int(event_seed) - int(config.seed))
        tasks = env.sample_tasks(len(protocol.MODES))
        env.set_nonstationary_para(tasks)
        env.set_task(tasks[int(mode)])
        router = _router(stack, arm)
        returns = []
        terminated = []
        posterior_rows = []
        posterior_labels = []
        fallback_actions = 0
        total_actions = 0
        adaptive_correct = 0
        adaptive_actions = 0
        try:
            for _ in range(protocol.STATIONARY_EPISODES):
                observation = env.reset()
                if router is not None:
                    router.reset()
                episode_return = 0.0
                episode_terminated = False
                for _ in range(protocol.MAX_EPISODE_STEPS):
                    if router is not None:
                        posterior_rows.append(router.posterior.copy())
                        posterior_labels.append(int(mode))
                    action, source_name, selected_mode = _select_action(
                        stack, router, arm, observation, int(mode))
                    next_observation, reward, done, info = env.step(action)
                    if int(info["mode_used"]) != int(mode):
                        raise RuntimeError("stationary specialist mode changed")
                    if router is not None:
                        router.observe_transition(
                            observation, action, reward, next_observation)
                    fallback_actions += int(source_name == "robust")
                    if source_name.startswith("specialist"):
                        adaptive_correct += int(selected_mode == int(mode))
                        adaptive_actions += 1
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
            "return_mean": float(np.mean(returns)),
            "terminated_rate": float(np.mean(terminated)),
            "total_actions": int(total_actions),
        }
        metrics = _posterior_metrics(posterior_rows, posterior_labels)
        if metrics is not None:
            row.update({
                "posterior_metrics": metrics,
                "fallback_action_fraction": float(
                    fallback_actions / max(total_actions, 1)),
                "adaptive_mode_accuracy": float(
                    adaptive_correct / max(adaptive_actions, 1)),
            })
        rows[str(mode)] = row
    return rows


def _switching_arm(
    stack: dict[str, Any], arm: str, event_seed: int
) -> dict[str, Any]:
    config = copy.deepcopy(stack["config"])
    config.stochastic_mode_fixed_id = -1
    env = make_env(config, seed_offset=int(event_seed) - int(config.seed))
    tasks = env.sample_tasks(len(protocol.MODES))
    router = _router(stack, arm)
    returns = []
    terminated = []
    trace = []
    posterior_rows = []
    posterior_labels = []
    fallback_actions = 0
    adaptive_correct = 0
    adaptive_actions = 0
    trigger_counts = []
    total_actions = 0
    try:
        for episode in range(protocol.SWITCHING_EPISODES):
            sequence, _ = _select_eval_switch_sequence(env, tasks, episode)
            _reset_eval_switch_schedule(
                env, sequence, config, protocol.DWELL_STEPS)
            observation = env.reset()
            if router is not None:
                router.reset()
            episode_return = 0.0
            episode_terminated = False
            for _ in range(protocol.MAX_EPISODE_STEPS):
                mode = int(env.task_id_for_next_step())
                trace.append(mode)
                if router is not None:
                    posterior_rows.append(router.posterior.copy())
                    posterior_labels.append(mode)
                action, source_name, selected_mode = _select_action(
                    stack, router, arm, observation, mode)
                next_observation, reward, done, info = env.step(action)
                if int(info["mode_used"]) != mode:
                    raise RuntimeError("switching specialist mode misaligned")
                if router is not None:
                    router.observe_transition(
                        observation, action, reward, next_observation)
                fallback_actions += int(source_name == "robust")
                if source_name.startswith("specialist"):
                    adaptive_correct += int(selected_mode == mode)
                    adaptive_actions += 1
                total_actions += 1
                episode_return += float(reward)
                observation = next_observation
                if done:
                    episode_terminated = True
                    observation = env.reset()
            returns.append(float(episode_return))
            terminated.append(float(episode_terminated))
            if router is not None:
                trigger_counts.append(router.gate.state.trigger_count)
    finally:
        if hasattr(env, "close"):
            env.close()
    row: dict[str, Any] = {
        "returns": returns,
        "return_mean": float(np.mean(returns)),
        "terminated_rate": float(np.mean(terminated)),
        "total_actions": int(total_actions),
        "mode_trace_sha256": hashlib.sha256(bytes(trace)).hexdigest(),
    }
    metrics = _posterior_metrics(posterior_rows, posterior_labels)
    if metrics is not None:
        row.update({
            "posterior_metrics": metrics,
            "fallback_action_fraction": float(
                fallback_actions / max(total_actions, 1)),
            "adaptive_mode_accuracy": float(
                adaptive_correct / max(adaptive_actions, 1)),
            "trigger_count_mean": float(np.mean(trigger_counts)),
        })
    if arm == "dynamic_oracle":
        row["adaptive_mode_accuracy"] = 1.0
    return row


def _identity(seed: int, event_seed: int) -> dict[str, Any]:
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "benchmark_role": "causal_independent_specialist_router_development",
        "env": protocol.ENV,
        "family": protocol.FAMILY,
        "training_seed": protocol.require_training_seed(seed),
        "event_seed": protocol.require_event_seed(event_seed),
        "arms": list(protocol.ARMS),
        "online_inputs": [
            "observation", "commanded_action", "reward", "next_observation"
        ],
        "privileged_arm": "dynamic_oracle",
    }


def evaluate(seed: int, event_seed: int) -> dict[str, Any]:
    seed = protocol.require_training_seed(seed)
    event_seed = protocol.require_event_seed(event_seed)
    stack = _load_stack(seed)
    return {
        "schema": protocol.EVENT_SCHEMA,
        "status": "complete",
        "identity": _identity(seed, event_seed),
        "source_bundles": protocol.source_records(seed),
        "estimator": protocol.estimator_records(),
        "fallback_config": protocol.FALLBACK_CONFIG.to_dict(),
        "stationary": {
            arm: _stationary_arm(stack, arm, event_seed)
            for arm in protocol.ARMS
        },
        "switching": {
            arm: _switching_arm(stack, arm, event_seed)
            for arm in protocol.ARMS
        },
    }


def validate_event(payload: dict, seed: int, event_seed: int) -> None:
    if (
        payload.get("schema") != protocol.EVENT_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("identity") != _identity(seed, event_seed)
        or payload.get("source_bundles") != protocol.source_records(seed)
        or payload.get("estimator") != protocol.estimator_records()
        or payload.get("fallback_config")
        != protocol.FALLBACK_CONFIG.to_dict()
        or set(payload.get("stationary") or {}) != set(protocol.ARMS)
        or set(payload.get("switching") or {}) != set(protocol.ARMS)
    ):
        raise ValueError("invalid specialist-router event payload")
    traces = set()
    for arm in protocol.ARMS:
        stationary = payload["stationary"][arm]
        if set(stationary) != {str(mode) for mode in protocol.MODES}:
            raise ValueError("specialist-router stationary matrix is incomplete")
        for row in stationary.values():
            returns = row.get("returns") or []
            if (
                len(returns) != protocol.STATIONARY_EPISODES
                or not all(math.isfinite(float(value)) for value in returns)
                or int(row.get("total_actions", -1))
                != protocol.STATIONARY_EPISODES * protocol.MAX_EPISODE_STEPS
            ):
                raise ValueError("invalid specialist-router stationary result")
        switching = payload["switching"][arm]
        returns = switching.get("returns") or []
        if (
            len(returns) != protocol.SWITCHING_EPISODES
            or not all(math.isfinite(float(value)) for value in returns)
            or int(switching.get("total_actions", -1))
            != protocol.SWITCHING_EPISODES * protocol.MAX_EPISODE_STEPS
        ):
            raise ValueError("invalid specialist-router switching result")
        traces.add(str(switching.get("mode_trace_sha256") or ""))
    if len(traces) != 1 or "" in traces:
        raise ValueError("specialist-router arms used different mode streams")
    if payload["switching"]["dynamic_oracle"]["adaptive_mode_accuracy"] != 1.0:
        raise ValueError("dynamic specialist oracle did not follow true mode")


def validate_audit(seed: int) -> dict[str, Any]:
    seed = protocol.require_training_seed(seed)
    manifest = protocol.read_json(protocol.audit_manifest(seed))
    expected_files = {
        str(event_seed): protocol.file_record(
            protocol.event_result(seed, event_seed))
        for event_seed in protocol.EVENT_SEEDS
    }
    if (
        manifest.get("schema") != protocol.AUDIT_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("identity")
        != {
            "protocol_version": protocol.PROTOCOL_VERSION,
            "training_seed": seed,
            "event_seeds": list(protocol.EVENT_SEEDS),
        }
        or manifest.get("event_files") != expected_files
    ):
        raise ValueError("invalid specialist-router audit manifest")
    for event_seed in protocol.EVENT_SEEDS:
        validate_event(
            protocol.read_json(protocol.event_result(seed, event_seed)),
            seed,
            event_seed,
        )
    return manifest


def run(seed: int) -> None:
    seed = protocol.require_training_seed(seed)
    destination = protocol.audit_dir(seed)
    if protocol.audit_manifest(seed).is_file():
        try:
            validate_audit(seed)
        except (KeyError, OSError, TypeError, ValueError):
            pass
        else:
            print(f"SPECIALIST ROUTER AUDIT ALREADY COMPLETE: {destination}")
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
        event_files = {}
        for event_seed in protocol.EVENT_SEEDS:
            result_path = (
                temporary / f"event_seed_{event_seed}" / "results.json")
            protocol.write_json_atomic(
                result_path, evaluate(seed, event_seed))
            event_files[str(event_seed)] = protocol.file_record(result_path)
        protocol.write_json_atomic(
            temporary / "audit_manifest.json",
            {
                "schema": protocol.AUDIT_SCHEMA,
                "status": "complete",
                "identity": {
                    "protocol_version": protocol.PROTOCOL_VERSION,
                    "training_seed": seed,
                    "event_seeds": list(protocol.EVENT_SEEDS),
                },
                "source_bundles": protocol.source_records(seed),
                "estimator": protocol.estimator_records(),
                "event_files": event_files,
            },
        )
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_audit(seed)
    print(f"SPECIALIST ROUTER AUDIT COMPLETE: seed={seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.seed)


if __name__ == "__main__":
    main()
