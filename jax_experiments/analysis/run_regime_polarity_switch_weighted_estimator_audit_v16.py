"""Audit v16 switch-weighted evidence with plain causal MAP routing."""
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
    regime_polarity_conflict_fallback_confirmation_v15 as parent,
)
from jax_experiments.analysis import (
    regime_polarity_frozen_estimator_transfer_v13 as transfer_parent,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_system_id_v5 as estimator_parent,
)
from jax_experiments.analysis import (
    regime_polarity_switch_weighted_estimator_model_v16 as model_lib,
)
from jax_experiments.analysis import (
    regime_polarity_switch_weighted_estimator_v16 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_frozen_estimator_transfer_audit_v13 as stack_loader,
)
from jax_experiments.train import make_env


def _identity(seed: int, event_seed: int) -> dict[str, Any]:
    event_seed = protocol.require_audit_event_seed(event_seed)
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "benchmark_role": "switch_weighted_estimator_development_audit",
        "env": protocol.ENV,
        "family": protocol.FAMILY,
        "policy_seed": protocol.require_policy_seed(seed),
        "event_seed": event_seed,
        "base_schedule": list(protocol.SWITCHING_SCHEDULES[event_seed]),
        "arms": list(protocol.ARMS),
        "dwell_steps": protocol.DWELL_STEPS,
        "max_episode_steps": protocol.MAX_EPISODE_STEPS,
        "switching_episodes": protocol.AUDIT_EPISODES,
        "switch_window_steps": protocol.SWITCH_WINDOW_STEPS,
    }


def _load_stack(seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    seed = protocol.require_policy_seed(seed)
    transfer_parent.validate_registration()
    stack_loader.validate_audit(seed)
    return stack_loader._load_stack(seed)


def _controller_for_mode(mapping: dict[str, Any], mode: int) -> str:
    return str(mapping[str(int(mode))]["controller"])


def _switching_arm(
    stack: dict[str, Any],
    mapping: dict[str, Any],
    arm: str,
    event_seed: int,
) -> dict[str, Any]:
    arm = protocol.require_arm(arm)
    event_seed = protocol.require_audit_event_seed(event_seed)
    config = copy.deepcopy(stack["config"])
    config.stochastic_mode_fixed_id = -1
    env = make_env(config, seed_offset=event_seed - int(config.seed))
    tasks = env.sample_tasks(len(protocol.MODES))
    actions = stack["actions"]
    if arm == "frozen_v5_posterior_map":
        estimator = stack["estimator_factory"]()
    elif arm == "switch_weighted_v16_posterior_map":
        reference = stack["estimator_factory"]()
        estimator = model_lib.make_estimator(
            17,
            int(np.asarray(reference.gain_vectors).shape[1]),
        )
    else:
        estimator = None

    returns = []
    terminated = []
    trace: list[int] = []
    mode_counts = {str(mode): 0 for mode in protocol.MODES}
    episode_sequences = []
    posterior_rows = []
    posterior_labels = []
    route_correct = 0
    route_actions = 0
    switch_route_correct = 0
    switch_route_actions = 0
    stable_route_correct = 0
    stable_route_actions = 0
    wrong_specialist_actions = 0
    controller_mismatch_actions = 0
    mapped_robust_actions = 0
    delayed_actions = 0
    total_actions = 0
    try:
        configure = getattr(env, "configure_eval_mode_sequence", None)
        if not callable(configure):
            raise RuntimeError("v16 audit requires explicit mode schedules")
        for episode in range(protocol.AUDIT_EPISODES):
            sequence = protocol.switching_sequence(event_seed, episode)
            episode_sequences.append(list(sequence))
            configure(tasks, sequence, protocol.DWELL_STEPS)
            observation = env.reset()
            estimator_state = (
                estimator.initial_state() if estimator is not None else None)
            previous_mode = None
            switch_age = 0
            delay_remaining = 0
            episode_return = 0.0
            episode_terminated = False
            for _ in range(protocol.MAX_EPISODE_STEPS):
                mode = int(env.task_id_for_next_step())
                trace.append(mode)
                mode_counts[str(mode)] += 1
                if previous_mode is None or mode != previous_mode:
                    switch_age = 0
                    delay_remaining = protocol.DELAY_STEPS
                else:
                    switch_age += 1
                previous_mode = mode
                in_switch_window = switch_age < protocol.SWITCH_WINDOW_STEPS

                route_mode = None
                if arm == "robust_sac":
                    controller = "robust_sac"
                elif arm == "true_mode_safe_utility":
                    route_mode = mode
                    controller = _controller_for_mode(mapping, route_mode)
                elif arm == "delayed_oracle_4_safe_utility":
                    if delay_remaining > 0:
                        controller = "robust_sac"
                        delay_remaining -= 1
                        delayed_actions += 1
                    else:
                        route_mode = mode
                        controller = _controller_for_mode(mapping, route_mode)
                else:
                    posterior = np.asarray(
                        estimator.probabilities(estimator_state),
                        dtype=np.float64,
                    )
                    posterior_rows.append(posterior.copy())
                    posterior_labels.append(mode)
                    route_mode = int(np.argmax(posterior))
                    controller = _controller_for_mode(mapping, route_mode)

                action = actions[controller](observation)
                next_observation, reward, done, info = env.step(action)
                if int(info["mode_used"]) != mode:
                    raise RuntimeError("v16 switching action used the wrong mode")
                if estimator is not None:
                    estimator_state, _, _, _ = estimator.step(
                        estimator_state,
                        observation,
                        action,
                        reward,
                        next_observation,
                    )

                if route_mode is not None:
                    correct = int(route_mode == mode)
                    route_correct += correct
                    route_actions += 1
                    if in_switch_window:
                        switch_route_correct += correct
                        switch_route_actions += 1
                    else:
                        stable_route_correct += correct
                        stable_route_actions += 1
                    wrong_specialist_actions += int(
                        controller != "robust_sac" and route_mode != mode)
                controller_mismatch_actions += int(
                    controller != _controller_for_mode(mapping, mode))
                mapped_robust_actions += int(controller == "robust_sac")
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
        "mode_trace_sha256": hashlib.sha256(bytes(trace)).hexdigest(),
        "mode_counts": mode_counts,
        "base_schedule": list(protocol.SWITCHING_SCHEDULES[event_seed]),
        "episode_sequences": episode_sequences,
        "routing_mode_accuracy": (
            float(route_correct / route_actions) if route_actions else None),
        "switch_window_routing_accuracy": (
            float(switch_route_correct / switch_route_actions)
            if switch_route_actions else None),
        "stable_routing_accuracy": (
            float(stable_route_correct / stable_route_actions)
            if stable_route_actions else None),
        "wrong_specialist_action_fraction": float(
            wrong_specialist_actions / max(total_actions, 1)),
        "controller_mismatch_action_fraction": float(
            controller_mismatch_actions / max(total_actions, 1)),
        "mapped_robust_action_fraction": float(
            mapped_robust_actions / max(total_actions, 1)),
        "delayed_action_fraction": float(
            delayed_actions / max(total_actions, 1)),
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


def evaluate(seed: int, event_seed: int) -> dict[str, Any]:
    seed = protocol.require_policy_seed(seed)
    event_seed = protocol.require_audit_event_seed(event_seed)
    stack, mapping = _load_stack(seed)
    return {
        "schema": protocol.EVENT_SCHEMA,
        "status": "complete",
        "identity": _identity(seed, event_seed),
        "registration": protocol.file_record(protocol.REGISTRATION_PATH),
        "model": protocol.model_records(),
        "parent_model": estimator_parent.estimator_records(),
        "parent_audit": protocol.file_record(parent.audit_manifest(seed)),
        "transfer_parent_audit": protocol.file_record(
            transfer_parent.audit_manifest(seed)),
        "policy_bank": protocol.policy_bank_records(seed),
        "utility_map": mapping,
        "switching": {
            arm: _switching_arm(stack, mapping, arm, event_seed)
            for arm in protocol.ARMS
        },
    }


def validate_event(payload: dict, seed: int, event_seed: int) -> None:
    parent_payload = protocol.read_json(
        parent.event_result(seed, parent.EVENT_SEEDS[0]))
    if (
        payload.get("schema") != protocol.EVENT_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("identity") != _identity(seed, event_seed)
        or payload.get("registration")
        != protocol.file_record(protocol.REGISTRATION_PATH)
        or payload.get("model") != protocol.model_records()
        or payload.get("parent_model") != estimator_parent.estimator_records()
        or payload.get("parent_audit")
        != protocol.file_record(parent.audit_manifest(seed))
        or payload.get("transfer_parent_audit")
        != protocol.file_record(transfer_parent.audit_manifest(seed))
        or payload.get("policy_bank") != protocol.policy_bank_records(seed)
        or payload.get("utility_map") != parent_payload["utility_map"]
        or set(payload.get("switching") or {}) != set(protocol.ARMS)
    ):
        raise ValueError("invalid v16 switch-weighted estimator event")
    expected_mode_count = (
        protocol.AUDIT_EPISODES * protocol.MAX_EPISODE_STEPS
        // len(protocol.MODES)
    )
    traces = set()
    for arm in protocol.ARMS:
        row = payload["switching"][arm]
        if (
            len(row.get("returns") or []) != protocol.AUDIT_EPISODES
            or not all(math.isfinite(float(value)) for value in row["returns"])
            or int(row.get("total_actions", -1))
            != protocol.AUDIT_EPISODES * protocol.MAX_EPISODE_STEPS
            or row.get("mode_counts") != {
                str(mode): expected_mode_count for mode in protocol.MODES
            }
            or row.get("base_schedule")
            != list(protocol.SWITCHING_SCHEDULES[int(event_seed)])
        ):
            raise ValueError("invalid v16 switching rollout")
        traces.add(str(row.get("mode_trace_sha256") or ""))
    if len(traces) != 1 or "" in traces:
        raise ValueError("v16 arms used different switching streams")
    for arm in (
        "true_mode_safe_utility", "delayed_oracle_4_safe_utility",
    ):
        if payload["switching"][arm]["routing_mode_accuracy"] != 1.0:
            raise ValueError("v16 privileged arm routed a wrong mode")
    for arm in (
        "frozen_v5_posterior_map", "switch_weighted_v16_posterior_map",
    ):
        row = payload["switching"][arm]
        posterior = row.get("posterior_metrics") or {}
        for key in (
            "routing_mode_accuracy", "switch_window_routing_accuracy",
            "stable_routing_accuracy",
        ):
            if not math.isfinite(float(row.get(key, math.nan))):
                raise ValueError("v16 routing metrics are missing")
        if not (
            math.isfinite(float(posterior.get("mode_accuracy", math.nan)))
            and math.isfinite(float(posterior.get("brier_score", math.nan)))
        ):
            raise ValueError("v16 posterior metrics are missing")


def validate_audit(seed: int) -> dict[str, Any]:
    seed = protocol.require_policy_seed(seed)
    manifest = protocol.read_json(protocol.audit_manifest(seed))
    expected = {
        str(event_seed): protocol.file_record(
            protocol.event_result(seed, event_seed))
        for event_seed in protocol.AUDIT_EVENT_SEEDS
    }
    if (
        manifest.get("schema") != protocol.AUDIT_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("identity")
        != {
            "protocol_version": protocol.PROTOCOL_VERSION,
            "policy_seed": seed,
            "event_seeds": list(protocol.AUDIT_EVENT_SEEDS),
            "switching_schedules": {
                str(key): list(protocol.SWITCHING_SCHEDULES[key])
                for key in protocol.AUDIT_EVENT_SEEDS
            },
        }
        or manifest.get("registration")
        != protocol.file_record(protocol.REGISTRATION_PATH)
        or manifest.get("model") != protocol.model_records()
        or manifest.get("parent_audit")
        != protocol.file_record(parent.audit_manifest(seed))
        or manifest.get("policy_bank") != protocol.policy_bank_records(seed)
        or manifest.get("event_files") != expected
    ):
        raise ValueError("invalid v16 estimator audit manifest")
    traces = set()
    for event_seed in protocol.AUDIT_EVENT_SEEDS:
        payload = protocol.read_json(protocol.event_result(seed, event_seed))
        validate_event(payload, seed, event_seed)
        traces.add(payload["switching"]["robust_sac"]["mode_trace_sha256"])
    if len(traces) != len(protocol.AUDIT_EVENT_SEEDS):
        raise ValueError("v16 audit event streams are not distinct")
    return manifest


def run(seed: int) -> None:
    seed = protocol.require_policy_seed(seed)
    protocol.validate_registration()
    model_lib.load_model(17, 6)
    destination = protocol.audit_dir(seed)
    if protocol.audit_manifest(seed).is_file():
        try:
            validate_audit(seed)
        except (KeyError, OSError, TypeError, ValueError):
            pass
        else:
            print(f"V16 ESTIMATOR AUDIT ALREADY COMPLETE: seed={seed}")
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
        for event_seed in protocol.AUDIT_EVENT_SEEDS:
            result = temporary / f"event_seed_{event_seed}" / "results.json"
            protocol.write_json_atomic(result, evaluate(seed, event_seed))
            event_files[str(event_seed)] = protocol.file_record(result)
            print(
                f"v16 estimator seed={seed} event={event_seed} complete",
                flush=True,
            )
        protocol.write_json_atomic(
            temporary / "audit_manifest.json",
            {
                "schema": protocol.AUDIT_SCHEMA,
                "status": "complete",
                "identity": {
                    "protocol_version": protocol.PROTOCOL_VERSION,
                    "policy_seed": seed,
                    "event_seeds": list(protocol.AUDIT_EVENT_SEEDS),
                    "switching_schedules": {
                        str(key): list(protocol.SWITCHING_SCHEDULES[key])
                        for key in protocol.AUDIT_EVENT_SEEDS
                    },
                },
                "registration": protocol.file_record(
                    protocol.REGISTRATION_PATH),
                "model": protocol.model_records(),
                "parent_audit": protocol.file_record(
                    parent.audit_manifest(seed)),
                "policy_bank": protocol.policy_bank_records(seed),
                "event_files": event_files,
            },
        )
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_audit(seed)
    print(f"V16 ESTIMATOR AUDIT COMPLETE: seed={seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", choices=protocol.AUDIT_POLICY_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.seed)


if __name__ == "__main__":
    main()
