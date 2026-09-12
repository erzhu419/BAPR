"""Audit the frozen v5 estimator on the confirmed v12 policy banks."""
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
    regime_polarity_frozen_estimator_transfer_v13 as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_robust_warmstart_confirmation_v12 as parent,
)
from jax_experiments.analysis import (
    run_regime_polarity_robust_warmstart_confirmation_audit_v12 as parent_audit,
)
from jax_experiments.analysis import (
    run_regime_polarity_robust_warmstart_confirmation_audit_v12_fix1 as parent_fix,
)
from jax_experiments.analysis import (
    run_regime_polarity_robust_warmstart_specialist_v12 as producer,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_safe_utility_audit_v8 as utility_audit,
)
from jax_experiments.train import make_env


def _identity(seed: int, event_seed: int) -> dict[str, Any]:
    event_seed = protocol.require_event_seed(event_seed)
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "benchmark_role": "frozen_estimator_transfer_on_confirmed_policy_bank",
        "env": protocol.ENV,
        "family": protocol.FAMILY,
        "training_seed": protocol.require_training_seed(seed),
        "event_seed": event_seed,
        "base_schedule": list(protocol.SWITCHING_SCHEDULES[event_seed]),
        "arms": list(protocol.ARMS),
        "delay_steps": protocol.DELAY_STEPS,
        "dwell_steps": protocol.DWELL_STEPS,
        "max_episode_steps": protocol.MAX_EPISODE_STEPS,
        "switching_episodes": protocol.SWITCHING_EPISODES,
    }


def _load_stack(seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    seed = protocol.require_training_seed(seed)
    parent.validate_registration()
    parent_fix.validate_fix_registration()
    parent_audit.validate_audit(seed)

    frozen_validate = producer.validate_bundle

    def compatible_validate_bundle(
        variant: str, requested_seed: int, mode: int,
    ) -> dict[str, Any]:
        parent.require_variant(variant)
        return frozen_validate(requested_seed, mode)

    producer.validate_bundle = compatible_validate_bundle
    try:
        controllers = parent_audit._load_controllers(seed)
    finally:
        producer.validate_bundle = frozen_validate

    frozen_audit = parent.read_json(parent.audit_result(seed))
    utility_map = frozen_audit["utility_map"]
    return utility_audit._stack_from_controllers(controllers), utility_map


def _controller_for_mode(mapping: dict[str, Any], mode: int) -> str:
    return str(mapping[str(int(mode))]["controller"])


def _switching_arm(
    stack: dict[str, Any],
    mapping: dict[str, Any],
    arm: str,
    event_seed: int,
) -> dict[str, Any]:
    arm = protocol.require_arm(arm)
    event_seed = protocol.require_event_seed(event_seed)
    config = copy.deepcopy(stack["config"])
    config.stochastic_mode_fixed_id = -1
    env = make_env(config, seed_offset=event_seed - int(config.seed))
    tasks = env.sample_tasks(len(protocol.MODES))
    actions = stack["actions"]
    estimator = (
        stack["estimator_factory"]()
        if arm == protocol.PRIMARY_ARM else None
    )

    returns: list[float] = []
    terminated: list[float] = []
    trace: list[int] = []
    mode_counts = {str(mode): 0 for mode in protocol.MODES}
    episode_sequences: list[list[int]] = []
    posterior_rows: list[np.ndarray] = []
    posterior_labels: list[int] = []
    routing_correct = 0
    routing_actions = 0
    mapped_robust_actions = 0
    delayed_actions = 0
    regime_onsets = 0
    total_actions = 0
    try:
        configure = getattr(env, "configure_eval_mode_sequence", None)
        if not callable(configure):
            raise RuntimeError(
                "v13 requires explicit configure_eval_mode_sequence support")
        for episode in range(protocol.SWITCHING_EPISODES):
            sequence = protocol.switching_sequence(event_seed, episode)
            episode_sequences.append(list(sequence))
            configure(tasks, sequence, protocol.DWELL_STEPS)
            observation = env.reset()
            estimator_state = (
                estimator.initial_state() if estimator is not None else None
            )
            previous_mode = None
            delay_remaining = 0
            episode_return = 0.0
            episode_terminated = False
            for _ in range(protocol.MAX_EPISODE_STEPS):
                mode = int(env.task_id_for_next_step())
                trace.append(mode)
                mode_counts[str(mode)] += 1
                if previous_mode is None or mode != previous_mode:
                    delay_remaining = protocol.DELAY_STEPS
                    regime_onsets += 1
                previous_mode = mode

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
                    raise RuntimeError("v13 switching action used the wrong mode")
                if estimator is not None:
                    estimator_state, _, _, _ = estimator.step(
                        estimator_state,
                        observation,
                        action,
                        reward,
                        next_observation,
                    )

                if route_mode is not None:
                    routing_correct += int(route_mode == mode)
                    routing_actions += 1
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
            float(routing_correct / routing_actions)
            if routing_actions else None
        ),
        "mapped_robust_action_fraction": float(
            mapped_robust_actions / max(total_actions, 1)),
        "delayed_action_fraction": float(
            delayed_actions / max(total_actions, 1)),
        "regime_onset_count": int(regime_onsets),
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
    seed = protocol.require_training_seed(seed)
    event_seed = protocol.require_event_seed(event_seed)
    stack, utility_map = _load_stack(seed)
    return {
        "schema": protocol.EVENT_SCHEMA,
        "status": "complete",
        "identity": _identity(seed, event_seed),
        "registration": protocol.file_record(protocol.REGISTRATION_PATH),
        "parent_registration": protocol.file_record(parent.REGISTRATION_PATH),
        "parent_fix_registration": protocol.file_record(
            parent_fix.FIX_REGISTRATION_PATH),
        "policy_bank": protocol.policy_bank_records(seed),
        "estimator": protocol.estimator_records(),
        "utility_map": utility_map,
        "switching": {
            arm: _switching_arm(stack, utility_map, arm, event_seed)
            for arm in protocol.ARMS
        },
    }


def validate_event(payload: dict, seed: int, event_seed: int) -> None:
    parent_payload = parent.read_json(parent.audit_result(seed))
    if (
        payload.get("schema") != protocol.EVENT_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("identity") != _identity(seed, event_seed)
        or payload.get("registration")
        != protocol.file_record(protocol.REGISTRATION_PATH)
        or payload.get("parent_registration")
        != protocol.file_record(parent.REGISTRATION_PATH)
        or payload.get("parent_fix_registration")
        != protocol.file_record(parent_fix.FIX_REGISTRATION_PATH)
        or payload.get("policy_bank") != protocol.policy_bank_records(seed)
        or payload.get("estimator") != protocol.estimator_records()
        or payload.get("utility_map") != parent_payload["utility_map"]
        or set(payload.get("switching") or {}) != set(protocol.ARMS)
    ):
        raise ValueError("invalid v13 frozen-estimator event")

    expected_mode_count = (
        protocol.SWITCHING_EPISODES * protocol.MAX_EPISODE_STEPS
        // len(protocol.MODES)
    )
    traces = set()
    for arm in protocol.ARMS:
        row = payload["switching"][arm]
        if (
            len(row.get("returns") or []) != protocol.SWITCHING_EPISODES
            or not all(math.isfinite(float(value)) for value in row["returns"])
            or int(row.get("total_actions", -1))
            != protocol.SWITCHING_EPISODES * protocol.MAX_EPISODE_STEPS
            or row.get("mode_counts") != {
                str(mode): expected_mode_count for mode in protocol.MODES
            }
            or row.get("base_schedule")
            != list(protocol.SWITCHING_SCHEDULES[int(event_seed)])
        ):
            raise ValueError("invalid v13 frozen-estimator rollout")
        traces.add(str(row.get("mode_trace_sha256") or ""))
    if len(traces) != 1 or "" in traces:
        raise ValueError("v13 arms used different switching streams")
    for arm in (
        "true_mode_safe_utility", "delayed_oracle_4_safe_utility",
    ):
        if payload["switching"][arm]["routing_mode_accuracy"] != 1.0:
            raise ValueError("v13 privileged arm routed a wrong mode")
    posterior = payload["switching"][protocol.PRIMARY_ARM].get(
        "posterior_metrics") or {}
    if not (
        math.isfinite(float(posterior.get("mode_accuracy", math.nan)))
        and math.isfinite(float(posterior.get("brier_score", math.nan)))
    ):
        raise ValueError("v13 posterior metrics are missing")


def validate_audit(seed: int) -> dict[str, Any]:
    seed = protocol.require_training_seed(seed)
    manifest = protocol.read_json(protocol.audit_manifest(seed))
    expected = {
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
            "switching_schedules": {
                str(key): list(value)
                for key, value in protocol.SWITCHING_SCHEDULES.items()
            },
        }
        or manifest.get("registration")
        != protocol.file_record(protocol.REGISTRATION_PATH)
        or manifest.get("policy_bank") != protocol.policy_bank_records(seed)
        or manifest.get("estimator") != protocol.estimator_records()
        or manifest.get("event_files") != expected
    ):
        raise ValueError("invalid v13 frozen-estimator audit manifest")
    traces = set()
    for event_seed in protocol.EVENT_SEEDS:
        payload = protocol.read_json(protocol.event_result(seed, event_seed))
        validate_event(payload, seed, event_seed)
        traces.add(payload["switching"]["robust_sac"]["mode_trace_sha256"])
    if len(traces) != len(protocol.EVENT_SEEDS):
        raise ValueError("v13 event streams are not distinct")
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
            print(f"V13 FROZEN ESTIMATOR AUDIT ALREADY COMPLETE: seed={seed}")
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
            result = temporary / f"event_seed_{event_seed}" / "results.json"
            protocol.write_json_atomic(result, evaluate(seed, event_seed))
            event_files[str(event_seed)] = protocol.file_record(result)
            print(
                f"v13 frozen estimator seed={seed} event={event_seed} complete",
                flush=True,
            )
        protocol.write_json_atomic(
            temporary / "audit_manifest.json",
            {
                "schema": protocol.AUDIT_SCHEMA,
                "status": "complete",
                "identity": {
                    "protocol_version": protocol.PROTOCOL_VERSION,
                    "training_seed": seed,
                    "event_seeds": list(protocol.EVENT_SEEDS),
                    "switching_schedules": {
                        str(key): list(value)
                        for key, value in protocol.SWITCHING_SCHEDULES.items()
                    },
                },
                "registration": protocol.file_record(
                    protocol.REGISTRATION_PATH),
                "policy_bank": protocol.policy_bank_records(seed),
                "estimator": protocol.estimator_records(),
                "event_files": event_files,
            },
        )
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_audit(seed)
    print(f"V13 FROZEN ESTIMATOR AUDIT COMPLETE: seed={seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.seed)


if __name__ == "__main__":
    main()
