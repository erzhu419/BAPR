"""Audit frozen v5/v16 estimators on fresh v17 policy banks."""
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
    regime_polarity_fresh_bank_estimator_confirmation_v17 as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_model_v5 as v5_model,
)
from jax_experiments.analysis import (
    regime_polarity_switch_weighted_estimator_model_v16 as v16_model,
)
from jax_experiments.analysis import (
    run_regime_polarity_robust_source_v17 as source_runner,
)
from jax_experiments.analysis import (
    run_regime_polarity_robust_warmstart_confirmation_audit_v12 as policy_audit,
)
from jax_experiments.analysis import (
    run_regime_polarity_robust_warmstart_specialist_v17 as producer,
)
from jax_experiments.train import make_env


def _bind_policy_audit() -> None:
    policy_audit.protocol = protocol
    policy_audit.source_runner = source_runner
    policy_audit.producer = producer
    policy_audit._bind()


def _identity(seed: int) -> dict[str, Any]:
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "benchmark_role": "fresh_policy_bank_estimator_confirmation",
        "training_seed": protocol.require_training_seed(seed),
        "calibration_event_seeds": list(protocol.CALIBRATION_EVENT_SEEDS),
        "stationary_holdout_event_seeds": list(
            protocol.STATIONARY_HOLDOUT_EVENT_SEEDS),
        "switching_event_seeds": list(protocol.SWITCHING_EVENT_SEEDS),
        "switching_schedules": {
            str(key): list(value)
            for key, value in protocol.SWITCHING_SCHEDULES.items()
        },
        "arms": list(protocol.ARMS),
        "switch_window_steps": protocol.SWITCH_WINDOW_STEPS,
    }


def _controller_for_mode(mapping: dict[str, Any], mode: int) -> str:
    return str(mapping[str(int(mode))]["controller"])


def _make_estimator(arm: str, obs_dim: int, act_dim: int):
    if arm == "frozen_v5_posterior_map":
        return v5_model.make_estimator(obs_dim, act_dim)
    if arm == "switch_weighted_v16_posterior_map":
        return v16_model.make_estimator(obs_dim, act_dim)
    return None


def _switching_arm(
    stack: dict[str, Any],
    mapping: dict[str, Any],
    arm: str,
    event_seed: int,
    obs_dim: int,
    act_dim: int,
) -> dict[str, Any]:
    if arm not in protocol.ARMS:
        raise ValueError(f"unsupported v17 switching arm {arm!r}")
    event_seed = protocol.require_switching_event_seed(event_seed)
    config = copy.deepcopy(stack["config"])
    config.stochastic_mode_fixed_id = -1
    env = make_env(config, seed_offset=event_seed - int(config.seed))
    tasks = env.sample_tasks(len(protocol.MODES))
    actions = stack["actions"]
    estimator = _make_estimator(arm, obs_dim, act_dim)

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
            raise RuntimeError("v17 requires explicit mode schedules")
        for episode in range(protocol.SWITCHING_EPISODES):
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
                in_switch_window = (
                    switch_age < protocol.SWITCH_WINDOW_STEPS)

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
                    raise RuntimeError("v17 switching action used the wrong mode")
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


def _switching_events(
    controllers: dict[str, dict[str, Any]],
    utility_map: dict[str, Any],
) -> dict[str, Any]:
    stack = policy_audit.utility_audit._stack_from_controllers(controllers)
    reference = controllers["robust_sac"]["agent"]
    obs_dim = int(reference.obs_dim)
    act_dim = int(reference.act_dim)
    events = {}
    trace_hashes = set()
    for event_seed in protocol.SWITCHING_EVENT_SEEDS:
        rows = {
            arm: _switching_arm(
                stack, utility_map, arm, event_seed, obs_dim, act_dim)
            for arm in protocol.ARMS
        }
        hashes = {str(row["mode_trace_sha256"]) for row in rows.values()}
        if len(hashes) != 1:
            raise RuntimeError("v17 switching arms used different mode streams")
        trace_hashes.update(hashes)
        events[str(event_seed)] = rows
    if len(trace_hashes) != len(protocol.SWITCHING_EVENT_SEEDS):
        raise RuntimeError("v17 switching events did not produce distinct traces")
    return events


def validate_audit(seed: int) -> dict[str, Any]:
    seed = protocol.require_training_seed(seed)
    manifest = protocol.read_json(protocol.audit_manifest(seed))
    payload = protocol.read_json(protocol.audit_result(seed))
    if (
        manifest.get("schema") != protocol.AUDIT_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("identity") != _identity(seed)
        or manifest.get("registration")
        != protocol.file_record(protocol.REGISTRATION_PATH)
        or manifest.get("source_bundle_manifest")
        != protocol.file_record(protocol.source_manifest(seed))
        or manifest.get("specialist_bundles") != protocol.bundle_records(seed)
        or manifest.get("estimators") != protocol.estimator_records()
        or manifest.get("audit")
        != protocol.file_record(protocol.audit_result(seed))
        or payload.get("schema") != protocol.AUDIT_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("identity") != _identity(seed)
        or payload.get("source_bundle_manifest")
        != protocol.file_record(protocol.source_manifest(seed))
        or payload.get("specialist_bundles") != protocol.bundle_records(seed)
        or payload.get("estimators") != protocol.estimator_records()
    ):
        raise ValueError("invalid v17 fresh-bank estimator audit")

    for split_name, event_seeds in (
        ("calibration_events", protocol.CALIBRATION_EVENT_SEEDS),
        ("stationary_holdout", protocol.STATIONARY_HOLDOUT_EVENT_SEEDS),
    ):
        split = payload.get(split_name) or {}
        if set(split) != {str(value) for value in event_seeds}:
            raise ValueError(f"v17 {split_name} is incomplete")
        for event in split.values():
            for role in ("robust_sac", "matching_specialist"):
                if set(event.get(role) or {}) != {
                    str(mode) for mode in protocol.MODES
                }:
                    raise ValueError("v17 stationary modes are incomplete")
                for row in event[role].values():
                    values = row.get("returns") or []
                    if (
                        len(values) != protocol.EPISODES_PER_TASK
                        or not all(math.isfinite(float(value)) for value in values)
                        or int(row.get("total_actions", -1))
                        != protocol.EPISODES_PER_TASK
                        * protocol.MAX_EPISODE_STEPS
                    ):
                        raise ValueError("invalid v17 stationary rollout")

    switching = payload.get("switching_holdout") or {}
    if set(switching) != {
        str(value) for value in protocol.SWITCHING_EVENT_SEEDS
    }:
        raise ValueError("v17 switching holdout is incomplete")
    event_hashes = set()
    expected_mode_count = (
        protocol.SWITCHING_EPISODES * protocol.MAX_EPISODE_STEPS
        // len(protocol.MODES)
    )
    for event_seed, event in switching.items():
        if set(event) != set(protocol.ARMS):
            raise ValueError("v17 switching arms are incomplete")
        arm_hashes = set()
        for arm, row in event.items():
            values = row.get("returns") or []
            if (
                len(values) != protocol.SWITCHING_EPISODES
                or not all(math.isfinite(float(value)) for value in values)
                or int(row.get("total_actions", -1))
                != protocol.SWITCHING_EPISODES
                * protocol.MAX_EPISODE_STEPS
                or row.get("mode_counts") != {
                    str(mode): expected_mode_count for mode in protocol.MODES
                }
                or row.get("base_schedule")
                != list(protocol.SWITCHING_SCHEDULES[int(event_seed)])
            ):
                raise ValueError("invalid v17 switching rollout")
            arm_hashes.add(str(row.get("mode_trace_sha256") or ""))
            if arm in {
                "frozen_v5_posterior_map",
                "switch_weighted_v16_posterior_map",
            }:
                posterior = row.get("posterior_metrics") or {}
                for key in (
                    "routing_mode_accuracy",
                    "switch_window_routing_accuracy",
                    "stable_routing_accuracy",
                ):
                    if not math.isfinite(float(row.get(key, math.nan))):
                        raise ValueError("v17 routing metrics are missing")
                if not (
                    math.isfinite(float(
                        posterior.get("mode_accuracy", math.nan)))
                    and math.isfinite(float(
                        posterior.get("brier_score", math.nan)))
                ):
                    raise ValueError("v17 posterior metrics are missing")
        if len(arm_hashes) != 1 or "" in arm_hashes:
            raise ValueError("v17 switching arms used different streams")
        if event["true_mode_safe_utility"]["routing_mode_accuracy"] != 1.0:
            raise ValueError("v17 true-mode oracle routed a wrong mode")
        if event["delayed_oracle_4_safe_utility"]["routing_mode_accuracy"] != 1.0:
            raise ValueError("v17 delayed oracle routed a wrong mode")
        event_hashes.update(arm_hashes)
    if len(event_hashes) != len(protocol.SWITCHING_EVENT_SEEDS):
        raise ValueError("v17 switching event traces are not distinct")
    return manifest


def run(seed: int) -> None:
    seed = protocol.require_training_seed(seed)
    protocol.validate_registration()
    _bind_policy_audit()
    v5_model.load_model(17, 6)
    v16_model.load_model(17, 6)
    destination = protocol.audit_dir(seed)
    if protocol.audit_manifest(seed).is_file():
        try:
            validate_audit(seed)
        except (KeyError, OSError, TypeError, ValueError):
            pass
        else:
            print(f"V17 ESTIMATOR AUDIT ALREADY COMPLETE: seed={seed}")
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
        controllers = policy_audit._load_controllers(seed)
        calibration_events = policy_audit.base._stationary_events(
            controllers, seed, protocol.CALIBRATION_EVENT_SEEDS)
        calibration_matrix = policy_audit.base._aggregate_calibration(
            calibration_events)
        utility_map = policy_audit.base._utility_map(calibration_matrix)
        payload = {
            "schema": protocol.AUDIT_SCHEMA,
            "status": "complete",
            "identity": _identity(seed),
            "registration": protocol.file_record(protocol.REGISTRATION_PATH),
            "source_bundle_manifest": protocol.file_record(
                protocol.source_manifest(seed)),
            "specialist_bundles": protocol.bundle_records(seed),
            "estimators": protocol.estimator_records(),
            "calibration_events": calibration_events,
            "calibration_matrix": calibration_matrix,
            "utility_map": utility_map,
            "stationary_holdout": policy_audit.base._stationary_events(
                controllers,
                seed,
                protocol.STATIONARY_HOLDOUT_EVENT_SEEDS,
            ),
            "switching_holdout": _switching_events(
                controllers, utility_map),
        }
        protocol.write_json_atomic(temporary / "audit.json", payload)
        protocol.write_json_atomic(
            temporary / "audit_manifest.json",
            {
                "schema": protocol.AUDIT_SCHEMA,
                "status": "complete",
                "identity": _identity(seed),
                "registration": protocol.file_record(
                    protocol.REGISTRATION_PATH),
                "source_bundle_manifest": protocol.file_record(
                    protocol.source_manifest(seed)),
                "specialist_bundles": protocol.bundle_records(seed),
                "estimators": protocol.estimator_records(),
                "audit": protocol.file_record(temporary / "audit.json"),
            },
        )
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_audit(seed)
    print(f"V17 ESTIMATOR AUDIT COMPLETE: seed={seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.seed)


if __name__ == "__main__":
    main()
