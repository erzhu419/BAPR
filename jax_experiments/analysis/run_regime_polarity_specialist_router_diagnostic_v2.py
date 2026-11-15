"""Run switching-only specialist-router causal diagnostics."""
from __future__ import annotations

import argparse
import copy
import hashlib
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_specialist_router_diagnostic_v2 as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_router_v1 as parent,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_router_audit_v1 as parent_audit,
)
from jax_experiments.train import (
    _reset_eval_switch_schedule,
    _select_eval_switch_sequence,
    make_env,
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


def _identity(seed: int, event_seed: int) -> dict[str, Any]:
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "benchmark_role": "specialist_router_failure_decomposition",
        "env": protocol.ENV,
        "family": protocol.FAMILY,
        "training_seed": protocol.require_training_seed(seed),
        "event_seed": protocol.require_event_seed(event_seed),
        "arms": list(protocol.ARMS),
        "privileged_arms": list(protocol.PRIVILEGED_ARMS),
    }


def _switching_arm(stack, arm: str, event_seed: int) -> dict[str, Any]:
    config = copy.deepcopy(stack["config"])
    config.stochastic_mode_fixed_id = -1
    env = make_env(config, seed_offset=int(event_seed) - int(config.seed))
    tasks = env.sample_tasks(len(protocol.MODES))
    actions = stack["actions"]
    current_router = None
    debounced_router = None
    estimator = None
    estimator_state = None
    if arm in {"posterior_map_fallback", "true_mode_current_gate"}:
        current_router = parent.CausalSpecialistRouter(
            actions["robust_sac"],
            tuple(actions[f"specialist_{mode}"] for mode in protocol.MODES),
            stack["estimator_factory"](),
            "map",
        )
    elif arm == "posterior_debounced_option":
        debounced_router = protocol.DebouncedSpecialistOption(
            actions["robust_sac"],
            tuple(actions[f"specialist_{mode}"] for mode in protocol.MODES),
            stack["estimator_factory"](),
        )
    elif arm == "posterior_map_no_gate":
        estimator = stack["estimator_factory"]()

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
            if current_router is not None:
                current_router.reset()
            if debounced_router is not None:
                debounced_router.reset()
            if estimator is not None:
                estimator_state = estimator.initial_state()
            previous_mode = None
            warmup_remaining = 0
            episode_return = 0.0
            episode_terminated = False
            for _ in range(protocol.MAX_EPISODE_STEPS):
                mode = int(env.task_id_for_next_step())
                trace.append(mode)
                if previous_mode is None or mode != previous_mode:
                    warmup_remaining = protocol.WARMUP_STEPS
                previous_mode = mode

                if arm == "robust_sac":
                    action = actions["robust_sac"](observation)
                    source_name, selected_mode = "robust", -1
                elif arm == "dynamic_oracle":
                    action = actions[f"specialist_{mode}"](observation)
                    source_name, selected_mode = "specialist", mode
                elif arm == "true_mode_robust10":
                    if warmup_remaining > 0:
                        action = actions["robust_sac"](observation)
                        source_name, selected_mode = "robust", -1
                    else:
                        action = actions[f"specialist_{mode}"](observation)
                        source_name, selected_mode = "specialist", mode
                    warmup_remaining = max(warmup_remaining - 1, 0)
                elif current_router is not None:
                    posterior_rows.append(current_router.posterior.copy())
                    posterior_labels.append(mode)
                    step = current_router.select_action(observation)
                    action = step.action
                    source_name = step.source
                    selected_mode = step.selected_mode
                    if (
                        arm == "true_mode_current_gate"
                        and source_name != "robust"
                    ):
                        action = actions[f"specialist_{mode}"](observation)
                        selected_mode = mode
                elif debounced_router is not None:
                    posterior_rows.append(debounced_router.posterior.copy())
                    posterior_labels.append(mode)
                    step = debounced_router.select_action(observation)
                    action = step.action
                    source_name = step.source
                    selected_mode = step.selected_mode
                else:
                    posterior = np.asarray(
                        estimator.probabilities(estimator_state),
                        dtype=np.float64,
                    )
                    posterior_rows.append(posterior.copy())
                    posterior_labels.append(mode)
                    selected_mode = int(np.argmax(posterior))
                    action = actions[f"specialist_{selected_mode}"](observation)
                    source_name = "specialist"

                next_observation, reward, done, info = env.step(action)
                if int(info["mode_used"]) != mode:
                    raise RuntimeError("specialist diagnostic mode misaligned")
                if current_router is not None:
                    current_router.observe_transition(
                        observation, action, reward, next_observation)
                elif debounced_router is not None:
                    debounced_router.observe_transition(
                        observation, action, reward, next_observation)
                elif estimator is not None:
                    estimator_state, _, _, _ = estimator.step(
                        estimator_state,
                        observation,
                        action,
                        reward,
                        next_observation,
                    )

                fallback_actions += int(source_name == "robust")
                if source_name == "specialist":
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
            if current_router is not None:
                trigger_counts.append(current_router.gate.state.trigger_count)
            elif debounced_router is not None:
                trigger_counts.append(debounced_router.trigger_count)
    finally:
        if hasattr(env, "close"):
            env.close()

    row: dict[str, Any] = {
        "returns": returns,
        "return_mean": float(np.mean(returns)),
        "terminated_rate": float(np.mean(terminated)),
        "total_actions": int(total_actions),
        "mode_trace_sha256": hashlib.sha256(bytes(trace)).hexdigest(),
        "fallback_action_fraction": float(
            fallback_actions / max(total_actions, 1)),
        "adaptive_mode_accuracy": (
            float(adaptive_correct / adaptive_actions)
            if adaptive_actions else None
        ),
    }
    metrics = _posterior_metrics(posterior_rows, posterior_labels)
    if metrics is not None:
        row["posterior_metrics"] = metrics
    if trigger_counts:
        row["trigger_count_mean"] = float(np.mean(trigger_counts))
    return row


def evaluate(seed: int, event_seed: int) -> dict[str, Any]:
    seed = protocol.require_training_seed(seed)
    event_seed = protocol.require_event_seed(event_seed)
    stack = parent_audit._load_stack(seed)
    return {
        "schema": protocol.EVENT_SCHEMA,
        "status": "complete",
        "identity": _identity(seed, event_seed),
        "source_bundles": protocol.source_records(seed),
        "estimator": protocol.estimator_records(),
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
        or set(payload.get("switching") or {}) != set(protocol.ARMS)
    ):
        raise ValueError("invalid specialist diagnostic event")
    traces = set()
    for arm in protocol.ARMS:
        row = payload["switching"][arm]
        values = row.get("returns") or []
        if (
            len(values) != protocol.SWITCHING_EPISODES
            or not all(math.isfinite(float(value)) for value in values)
            or int(row.get("total_actions", -1))
            != protocol.SWITCHING_EPISODES * protocol.MAX_EPISODE_STEPS
        ):
            raise ValueError("invalid specialist diagnostic result")
        traces.add(str(row.get("mode_trace_sha256") or ""))
    if len(traces) != 1 or "" in traces:
        raise ValueError("diagnostic arms used different mode streams")
    for arm in protocol.PRIVILEGED_ARMS:
        accuracy = payload["switching"][arm]["adaptive_mode_accuracy"]
        if accuracy is not None and accuracy != 1.0:
            raise ValueError("privileged specialist arm used a wrong mode")


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
        raise ValueError("invalid specialist diagnostic audit")
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
            print(f"SPECIALIST DIAGNOSTIC ALREADY COMPLETE: {destination}")
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
        records = {}
        for event_seed in protocol.EVENT_SEEDS:
            result = temporary / f"event_seed_{event_seed}" / "results.json"
            protocol.write_json_atomic(result, evaluate(seed, event_seed))
            records[str(event_seed)] = protocol.file_record(result)
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
                "event_files": records,
            },
        )
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_audit(seed)
    print(f"SPECIALIST DIAGNOSTIC COMPLETE: seed={seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.seed)


if __name__ == "__main__":
    main()
