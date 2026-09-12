"""Evaluate causal delay controls on one frozen v9 safe-utility bank."""
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
    regime_polarity_safe_utility_causal_delay_v10 as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_safe_utility_confirmation_v9 as parent,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_expected_action_confirmation_audit_v6 as source_audit,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_safe_utility_audit_v8 as utility_audit,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_safe_utility_confirmation_audit_v9 as parent_audit,
)
from jax_experiments.train import (
    _reset_eval_switch_schedule,
    _select_eval_switch_sequence,
    make_env,
)


def _bind() -> None:
    parent_audit._bind()


def _identity(seed: int, event_seed: int) -> dict[str, Any]:
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "benchmark_role": "safe_utility_causal_delay_diagnostic",
        "env": protocol.ENV,
        "family": protocol.FAMILY,
        "training_seed": protocol.require_training_seed(seed),
        "event_seed": protocol.require_event_seed(event_seed),
        "arms": list(protocol.ARMS),
    }


def _load_stack(seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    _bind()
    parent.validate_registration()
    parent_audit.validate_audit(seed)
    controllers = {
        role: source_audit._load_controller(role, seed)
        for role in protocol.ROLES
    }
    calibration = parent.read_json(parent.calibration_result(seed))
    parent_audit.validate_calibration(calibration, seed)
    return utility_audit._stack_from_controllers(controllers), calibration


def _controller_for_mode(mapping: dict[str, Any], mode: int) -> str:
    return str(mapping[str(int(mode))]["controller"])


def _switching_arm(
    stack: dict[str, Any],
    mapping: dict[str, Any],
    arm: str,
    event_seed: int,
) -> dict[str, Any]:
    arm = protocol.require_arm(arm)
    delay = protocol.delay_for_arm(arm)
    config = copy.deepcopy(stack["config"])
    config.stochastic_mode_fixed_id = -1
    env = make_env(config, seed_offset=int(event_seed) - int(config.seed))
    tasks = env.sample_tasks(len(protocol.MODES))
    actions = stack["actions"]
    estimator = (
        stack["estimator_factory"]()
        if arm == "posterior_map_safe_utility" else None
    )

    returns = []
    terminated = []
    trace = []
    posterior_rows = []
    posterior_labels = []
    routing_correct = 0
    routing_actions = 0
    mapped_robust_actions = 0
    delayed_actions = 0
    regime_onsets = 0
    total_actions = 0
    try:
        for episode in range(protocol.SWITCHING_EPISODES):
            sequence, _ = _select_eval_switch_sequence(env, tasks, episode)
            _reset_eval_switch_schedule(
                env, sequence, config, protocol.DWELL_STEPS)
            observation = env.reset()
            estimator_state = (
                estimator.initial_state() if estimator is not None else None
            )
            previous_mode = None
            stale_mode = None
            delay_remaining = 0
            episode_return = 0.0
            episode_terminated = False
            for _ in range(protocol.MAX_EPISODE_STEPS):
                mode = int(env.task_id_for_next_step())
                trace.append(mode)
                if previous_mode is None or mode != previous_mode:
                    stale_mode = previous_mode
                    delay_remaining = delay
                    regime_onsets += 1
                previous_mode = mode

                route_mode = None
                if arm == "robust_sac":
                    controller = "robust_sac"
                elif arm == "true_mode_safe_utility":
                    route_mode = mode
                    controller = _controller_for_mode(mapping, route_mode)
                elif arm == "posterior_map_safe_utility":
                    posterior = np.asarray(
                        estimator.probabilities(estimator_state),
                        dtype=np.float64,
                    )
                    posterior_rows.append(posterior.copy())
                    posterior_labels.append(mode)
                    route_mode = int(np.argmax(posterior))
                    controller = _controller_for_mode(mapping, route_mode)
                elif arm.startswith("stale_delay_") and delay_remaining > 0:
                    delayed_actions += 1
                    if stale_mode is None:
                        controller = "robust_sac"
                    else:
                        route_mode = int(stale_mode)
                        controller = _controller_for_mode(mapping, route_mode)
                    delay_remaining -= 1
                elif arm.startswith("robust_handoff_") and delay_remaining > 0:
                    delayed_actions += 1
                    controller = "robust_sac"
                    delay_remaining -= 1
                else:
                    route_mode = mode
                    controller = _controller_for_mode(mapping, route_mode)

                action = actions[controller](observation)
                next_observation, reward, done, info = env.step(action)
                if int(info["mode_used"]) != mode:
                    raise RuntimeError("v10 causal-delay mode misaligned")
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
    stack, calibration = _load_stack(seed)
    return {
        "schema": protocol.EVENT_SCHEMA,
        "status": "complete",
        "identity": _identity(seed, event_seed),
        "registration": protocol.file_record(protocol.REGISTRATION_PATH),
        "parent_registration": protocol.file_record(parent.REGISTRATION_PATH),
        "parent_audit": protocol.file_record(parent.audit_manifest(seed)),
        "parent_calibration": protocol.file_record(
            parent.calibration_result(seed)),
        "utility_map": calibration["utility_map"],
        "switching": {
            arm: _switching_arm(
                stack, calibration["utility_map"], arm, event_seed)
            for arm in protocol.ARMS
        },
    }


def validate_event(payload: dict, seed: int, event_seed: int) -> None:
    if (
        payload.get("schema") != protocol.EVENT_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("identity") != _identity(seed, event_seed)
        or payload.get("registration")
        != protocol.file_record(protocol.REGISTRATION_PATH)
        or payload.get("parent_registration")
        != protocol.file_record(parent.REGISTRATION_PATH)
        or payload.get("parent_audit")
        != protocol.file_record(parent.audit_manifest(seed))
        or payload.get("parent_calibration")
        != protocol.file_record(parent.calibration_result(seed))
        or set(payload.get("switching") or {}) != set(protocol.ARMS)
    ):
        raise ValueError("invalid v10 causal-delay event")
    traces = set()
    for arm in protocol.ARMS:
        row = payload["switching"][arm]
        if (
            len(row.get("returns") or []) != protocol.SWITCHING_EPISODES
            or not all(math.isfinite(float(x)) for x in row["returns"])
            or int(row.get("total_actions", -1))
            != protocol.SWITCHING_EPISODES * protocol.MAX_EPISODE_STEPS
        ):
            raise ValueError("invalid v10 causal-delay rollout")
        traces.add(str(row.get("mode_trace_sha256") or ""))
    if len(traces) != 1 or "" in traces:
        raise ValueError("v10 arms used different switching streams")
    if payload["switching"]["true_mode_safe_utility"][
        "routing_mode_accuracy"
    ] != 1.0:
        raise ValueError("v10 true-mode safe oracle routed a wrong mode")


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
        }
        or manifest.get("registration")
        != protocol.file_record(protocol.REGISTRATION_PATH)
        or manifest.get("parent_audit")
        != protocol.file_record(parent.audit_manifest(seed))
        or manifest.get("event_files") != expected
    ):
        raise ValueError("invalid v10 causal-delay audit manifest")
    for event_seed in protocol.EVENT_SEEDS:
        validate_event(
            protocol.read_json(protocol.event_result(seed, event_seed)),
            seed,
            event_seed,
        )
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
            print(f"V10 CAUSAL-DELAY AUDIT ALREADY COMPLETE: seed={seed}")
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
                f"v10 causal-delay seed={seed} event={event_seed} complete",
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
                },
                "registration": protocol.file_record(
                    protocol.REGISTRATION_PATH),
                "parent_audit": protocol.file_record(
                    parent.audit_manifest(seed)),
                "event_files": event_files,
            },
        )
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_audit(seed)
    print(f"V10 CAUSAL-DELAY AUDIT COMPLETE: seed={seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.seed)


if __name__ == "__main__":
    main()
