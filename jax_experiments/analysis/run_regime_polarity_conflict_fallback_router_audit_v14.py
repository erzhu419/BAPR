"""Evaluate causal evidence-conflict fallback routers on frozen policy banks."""
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
    regime_polarity_conflict_fallback_router_v14 as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_frozen_estimator_transfer_v13 as parent,
)
from jax_experiments.analysis import (
    run_regime_polarity_frozen_estimator_transfer_audit_v13 as parent_audit,
)
from jax_experiments.train import make_env


def _identity(seed: int, event_seed: int) -> dict[str, Any]:
    event_seed = protocol.require_event_seed(event_seed)
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "benchmark_role": "causal_evidence_conflict_fallback_development",
        "env": protocol.ENV,
        "family": protocol.FAMILY,
        "training_seed": protocol.require_training_seed(seed),
        "event_seed": event_seed,
        "base_schedule": list(protocol.SWITCHING_SCHEDULES[event_seed]),
        "arms": list(protocol.ARMS),
        "dwell_steps": protocol.DWELL_STEPS,
        "max_episode_steps": protocol.MAX_EPISODE_STEPS,
        "switching_episodes": protocol.SWITCHING_EPISODES,
    }


def _load_stack(seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    seed = protocol.require_training_seed(seed)
    parent.validate_registration()
    parent_audit.validate_audit(seed)
    return parent_audit._load_stack(seed)


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
    confirm_steps = protocol.confirm_steps_for_arm(arm)
    is_candidate = arm in protocol.CANDIDATE_ARMS
    uses_estimator = arm == "posterior_map_safe_utility" or is_candidate

    config = copy.deepcopy(stack["config"])
    config.stochastic_mode_fixed_id = -1
    env = make_env(config, seed_offset=event_seed - int(config.seed))
    tasks = env.sample_tasks(len(protocol.MODES))
    actions = stack["actions"]
    estimator = stack["estimator_factory"]() if uses_estimator else None

    returns: list[float] = []
    terminated: list[float] = []
    trace: list[int] = []
    mode_counts = {str(mode): 0 for mode in protocol.MODES}
    episode_sequences: list[list[int]] = []
    posterior_rows: list[np.ndarray] = []
    posterior_labels: list[int] = []
    routing_correct = 0
    routing_actions = 0
    wrong_specialist_actions = 0
    mapped_robust_actions = 0
    candidate_fallback_actions = 0
    delayed_actions = 0
    evidence_conflicts = 0
    fallback_exits = 0
    regime_onsets = 0
    total_actions = 0
    try:
        configure = getattr(env, "configure_eval_mode_sequence", None)
        if not callable(configure):
            raise RuntimeError(
                "v14 requires explicit configure_eval_mode_sequence support")
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

            active_mode = None
            in_fallback = bool(is_candidate)
            fallback_min_remaining = (
                protocol.MIN_FALLBACK_ACTIONS_AFTER_CONFLICT
                if is_candidate else 0
            )
            candidate_mode = None
            confirmation_count = 0

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
                used_candidate_fallback = False
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
                elif arm == "posterior_map_safe_utility":
                    posterior = np.asarray(
                        estimator.probabilities(estimator_state),
                        dtype=np.float64,
                    )
                    posterior_rows.append(posterior.copy())
                    posterior_labels.append(mode)
                    route_mode = int(np.argmax(posterior))
                    controller = _controller_for_mode(mapping, route_mode)
                elif in_fallback or active_mode is None:
                    controller = "robust_sac"
                    used_candidate_fallback = True
                    candidate_fallback_actions += 1
                    posterior_rows.append(np.asarray(
                        estimator.probabilities(estimator_state),
                        dtype=np.float64,
                    ))
                    posterior_labels.append(mode)
                else:
                    route_mode = int(active_mode)
                    controller = _controller_for_mode(mapping, route_mode)
                    posterior_rows.append(np.asarray(
                        estimator.probabilities(estimator_state),
                        dtype=np.float64,
                    ))
                    posterior_labels.append(mode)

                action = actions[controller](observation)
                next_observation, reward, done, info = env.step(action)
                if int(info["mode_used"]) != mode:
                    raise RuntimeError("v14 switching action used the wrong mode")

                evidence = None
                if estimator is not None:
                    estimator_state, evidence, _, _ = estimator.step(
                        estimator_state,
                        observation,
                        action,
                        reward,
                        next_observation,
                    )

                if is_candidate:
                    posterior_after = np.asarray(
                        estimator.probabilities(estimator_state),
                        dtype=np.float64,
                    )
                    evidence_mode = int(np.argmax(np.asarray(evidence)))
                    posterior_mode = int(np.argmax(posterior_after))
                    active_evidence = (
                        float(np.asarray(evidence)[int(active_mode)])
                        if active_mode is not None else None
                    )
                    alternative_evidence = (
                        float(np.max(np.delete(
                            np.asarray(evidence), int(active_mode))))
                        if active_mode is not None else None
                    )
                    conflict_margin = (
                        alternative_evidence - active_evidence
                        if active_mode is not None else float("-inf")
                    )

                    if (
                        not in_fallback
                        and evidence_mode != active_mode
                        and conflict_margin
                        >= protocol.CONFLICT_LOG_LIKELIHOOD_MARGIN
                    ):
                        in_fallback = True
                        fallback_min_remaining = (
                            protocol.MIN_FALLBACK_ACTIONS_AFTER_CONFLICT)
                        candidate_mode = None
                        confirmation_count = 0
                        evidence_conflicts += 1

                    if used_candidate_fallback and fallback_min_remaining > 0:
                        fallback_min_remaining -= 1

                    if in_fallback:
                        agrees = bool(
                            evidence_mode == posterior_mode
                            and float(np.max(posterior_after))
                            >= protocol.POSTERIOR_EXIT_CONFIDENCE
                        )
                        if agrees:
                            if candidate_mode == posterior_mode:
                                confirmation_count += 1
                            else:
                                candidate_mode = posterior_mode
                                confirmation_count = 1
                        else:
                            candidate_mode = None
                            confirmation_count = 0
                        if (
                            fallback_min_remaining == 0
                            and confirmation_count >= confirm_steps
                        ):
                            active_mode = int(posterior_mode)
                            in_fallback = False
                            candidate_mode = None
                            confirmation_count = 0
                            fallback_exits += 1

                if route_mode is not None:
                    routing_correct += int(route_mode == mode)
                    routing_actions += 1
                    wrong_specialist_actions += int(
                        controller != "robust_sac" and route_mode != mode)
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
        "wrong_specialist_action_fraction": float(
            wrong_specialist_actions / max(total_actions, 1)),
        "mapped_robust_action_fraction": float(
            mapped_robust_actions / max(total_actions, 1)),
        "candidate_fallback_action_fraction": float(
            candidate_fallback_actions / max(total_actions, 1)),
        "delayed_action_fraction": float(
            delayed_actions / max(total_actions, 1)),
        "evidence_conflict_count": int(evidence_conflicts),
        "fallback_exit_count": int(fallback_exits),
        "regime_onset_count": int(regime_onsets),
        "confirmation_steps": int(confirm_steps),
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
        "parent_audit": protocol.file_record(parent.audit_manifest(seed)),
        "policy_bank": protocol.policy_bank_records(seed),
        "estimator": protocol.estimator_records(),
        "utility_map": utility_map,
        "switching": {
            arm: _switching_arm(stack, utility_map, arm, event_seed)
            for arm in protocol.ARMS
        },
    }


def validate_event(payload: dict, seed: int, event_seed: int) -> None:
    parent_payload = parent.read_json(parent.event_result(seed, parent.EVENT_SEEDS[0]))
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
        or payload.get("policy_bank") != protocol.policy_bank_records(seed)
        or payload.get("estimator") != protocol.estimator_records()
        or payload.get("utility_map") != parent_payload["utility_map"]
        or set(payload.get("switching") or {}) != set(protocol.ARMS)
    ):
        raise ValueError("invalid v14 conflict-fallback event")

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
            or int(row.get("confirmation_steps", -1))
            != protocol.confirm_steps_for_arm(arm)
        ):
            raise ValueError("invalid v14 conflict-fallback rollout")
        traces.add(str(row.get("mode_trace_sha256") or ""))
    if len(traces) != 1 or "" in traces:
        raise ValueError("v14 arms used different switching streams")
    for arm in (
        "true_mode_safe_utility", "delayed_oracle_4_safe_utility",
    ):
        if payload["switching"][arm]["routing_mode_accuracy"] != 1.0:
            raise ValueError("v14 privileged arm routed a wrong mode")
    for arm in ("posterior_map_safe_utility", *protocol.CANDIDATE_ARMS):
        posterior = payload["switching"][arm].get("posterior_metrics") or {}
        if not (
            math.isfinite(float(posterior.get("mode_accuracy", math.nan)))
            and math.isfinite(float(posterior.get("brier_score", math.nan)))
        ):
            raise ValueError("v14 posterior metrics are missing")


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
        or manifest.get("parent_audit")
        != protocol.file_record(parent.audit_manifest(seed))
        or manifest.get("policy_bank") != protocol.policy_bank_records(seed)
        or manifest.get("estimator") != protocol.estimator_records()
        or manifest.get("event_files") != expected
    ):
        raise ValueError("invalid v14 conflict-fallback audit manifest")
    traces = set()
    for event_seed in protocol.EVENT_SEEDS:
        payload = protocol.read_json(protocol.event_result(seed, event_seed))
        validate_event(payload, seed, event_seed)
        traces.add(payload["switching"]["robust_sac"]["mode_trace_sha256"])
    if len(traces) != len(protocol.EVENT_SEEDS):
        raise ValueError("v14 event streams are not distinct")
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
            print(f"V14 CONFLICT FALLBACK AUDIT ALREADY COMPLETE: seed={seed}")
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
                f"v14 conflict fallback seed={seed} event={event_seed} complete",
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
                "parent_audit": protocol.file_record(
                    parent.audit_manifest(seed)),
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
    print(f"V14 CONFLICT FALLBACK AUDIT COMPLETE: seed={seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.seed)


if __name__ == "__main__":
    main()
