"""Audit V24 Ant switch-recovery specialists on frozen holdouts."""
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
from flax import nnx

from jax_experiments.analysis import (
    regime_polarity_ant_switch_recovery_v24 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_ant_switch_recovery_v24 as producer,
)
from jax_experiments.analysis import (
    run_regime_polarity_robust_warmstart_specialist_audit_v11 as base,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_safe_utility_audit_v8 as utility_audit,
)
from jax_experiments.train import make_env


SWITCHING_ARMS = (
    "robust_sac",
    "dynamic_specialist_oracle",
    "true_mode_safe_utility",
    "true_mode_safe_transient_fallback",
)


def _bind() -> None:
    base.protocol = protocol
    base.producer = producer
    utility_audit.protocol = protocol


def _load_source(seed: int) -> dict[str, Any]:
    config, agent = producer._load_source(seed)
    return {
        "config": config,
        "agent": agent,
        "policy_graphdef": nnx.graphdef(agent.policy),
        "policy_params": nnx.state(agent.policy, nnx.Param),
        "context_graphdef": None,
        "context_params": None,
    }


def _load_controllers(
    variant: str, seed: int,
) -> dict[str, dict[str, Any]]:
    _bind()
    robust = _load_source(seed)
    controllers = {"robust_sac": robust}
    for mode in protocol.MODES:
        controllers[f"specialist_{mode}"] = base._load_specialist(
            variant, seed, mode, robust)
    return controllers


def _switching_arm(
    raw_stack: dict[str, Any],
    safe_stack: dict[str, Any],
    arm: str,
    event_seed: int,
) -> dict[str, Any]:
    if arm not in SWITCHING_ARMS:
        raise ValueError(f"unsupported V24 switching arm {arm!r}")
    event_seed = protocol.require_switching_event_seed(event_seed)
    config = copy.deepcopy(raw_stack["config"])
    config.stochastic_mode_fixed_id = -1
    env = make_env(config, seed_offset=event_seed - int(config.seed))
    tasks = env.sample_tasks(len(protocol.MODES))
    returns = []
    terminated = []
    trace: list[int] = []
    mode_counts = {str(mode): 0 for mode in protocol.MODES}
    episode_sequences = []
    fallback_actions = 0
    adaptive_actions = 0
    total_actions = 0
    try:
        configure = getattr(env, "configure_eval_mode_sequence", None)
        if not callable(configure):
            raise RuntimeError("V24 requires explicit mode sequences")
        for episode in range(protocol.SWITCHING_EPISODES):
            sequence = protocol.switching_sequence(event_seed, episode)
            episode_sequences.append(list(sequence))
            configure(tasks, sequence, protocol.DWELL_STEPS)
            observation = env.reset()
            episode_return = 0.0
            episode_terminated = False
            previous_mode = None
            fallback_remaining = 0
            for _ in range(protocol.MAX_EPISODE_STEPS):
                mode = int(env.task_id_for_next_step())
                if mode != previous_mode:
                    fallback_remaining = protocol.TRANSIENT_FALLBACK_STEPS
                    previous_mode = mode
                trace.append(mode)
                mode_counts[str(mode)] += 1
                if arm == "robust_sac":
                    action = raw_stack["actions"]["robust_sac"](observation)
                    fallback_actions += 1
                elif (
                    arm == "true_mode_safe_transient_fallback"
                    and fallback_remaining > 0
                ):
                    action = raw_stack["actions"]["robust_sac"](observation)
                    fallback_actions += 1
                elif arm == "dynamic_specialist_oracle":
                    action = raw_stack["actions"][f"specialist_{mode}"](
                        observation)
                    adaptive_actions += 1
                else:
                    action = safe_stack["actions"][f"specialist_{mode}"](
                        observation)
                    adaptive_actions += 1
                next_observation, reward, done, info = env.step(action)
                if int(info["mode_used"]) != mode:
                    raise RuntimeError("V24 action used the wrong physics mode")
                total_actions += 1
                episode_return += float(reward)
                observation = next_observation
                fallback_remaining = max(0, fallback_remaining - 1)
                if done:
                    episode_terminated = True
                    observation = env.reset()
            returns.append(float(episode_return))
            terminated.append(float(episode_terminated))
    finally:
        if hasattr(env, "close"):
            env.close()
    return {
        "returns": returns,
        "return_mean": float(np.mean(returns)),
        "terminated_rate": float(np.mean(terminated)),
        "total_actions": int(total_actions),
        "mode_trace_sha256": hashlib.sha256(bytes(trace)).hexdigest(),
        "mode_counts": mode_counts,
        "base_schedule": list(protocol.SWITCHING_SCHEDULES[event_seed]),
        "episode_sequences": episode_sequences,
        "fallback_action_fraction": float(
            fallback_actions / max(total_actions, 1)),
        "adaptive_action_fraction": float(
            adaptive_actions / max(total_actions, 1)),
        "transient_fallback_steps": (
            protocol.TRANSIENT_FALLBACK_STEPS
            if arm == "true_mode_safe_transient_fallback" else 0
        ),
    }


def _switching_events(
    controllers: dict[str, dict[str, Any]],
    utility_map: dict[str, Any],
) -> dict[str, Any]:
    raw = utility_audit._stack_from_controllers(controllers)
    safe = base._safe_stack(raw, utility_map)
    events = {}
    event_hashes = set()
    for event_seed in protocol.SWITCHING_EVENT_SEEDS:
        rows = {
            arm: _switching_arm(raw, safe, arm, event_seed)
            for arm in SWITCHING_ARMS
        }
        hashes = {str(row["mode_trace_sha256"]) for row in rows.values()}
        if len(hashes) != 1:
            raise RuntimeError("V24 switching arms used different streams")
        event_hashes.update(hashes)
        events[str(event_seed)] = rows
    if len(event_hashes) != len(protocol.SWITCHING_EVENT_SEEDS):
        raise RuntimeError("V24 switching event traces are not distinct")
    return events


def _identity(variant: str, seed: int) -> dict[str, Any]:
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "benchmark_role": "ant_switch_recovery_development_audit",
        "variant": protocol.require_variant(variant),
        "training_seed": protocol.require_training_seed(seed),
        "transient_fallback_steps": protocol.TRANSIENT_FALLBACK_STEPS,
        "calibration_event_seeds": list(protocol.CALIBRATION_EVENT_SEEDS),
        "stationary_holdout_event_seeds": list(
            protocol.STATIONARY_HOLDOUT_EVENT_SEEDS),
        "switching_event_seeds": list(protocol.SWITCHING_EVENT_SEEDS),
        "switching_schedules": {
            str(key): list(value)
            for key, value in protocol.SWITCHING_SCHEDULES.items()
        },
    }


def validate_audit(variant: str, seed: int) -> dict[str, Any]:
    variant = protocol.require_variant(variant)
    seed = protocol.require_training_seed(seed)
    manifest = protocol.read_json(protocol.audit_manifest(variant, seed))
    payload = protocol.read_json(protocol.audit_result(variant, seed))
    identity = _identity(variant, seed)
    if (
        manifest.get("schema") != protocol.AUDIT_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("identity") != identity
        or manifest.get("registration")
        != protocol.file_record(protocol.REGISTRATION_PATH)
        or manifest.get("source_bundle_manifest")
        != protocol.file_record(protocol.source_manifest(seed))
        or manifest.get("specialist_bundles")
        != protocol.bundle_records(variant, seed)
        or manifest.get("audit")
        != protocol.file_record(protocol.audit_result(variant, seed))
        or payload.get("schema") != protocol.AUDIT_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("identity") != identity
    ):
        raise ValueError("invalid V24 audit")
    for split_name, event_seeds in (
        ("calibration_events", protocol.CALIBRATION_EVENT_SEEDS),
        ("stationary_holdout", protocol.STATIONARY_HOLDOUT_EVENT_SEEDS),
    ):
        split = payload.get(split_name) or {}
        if set(split) != {str(value) for value in event_seeds}:
            raise ValueError(f"V24 {split_name} is incomplete")
        for event in split.values():
            for role in ("robust_sac", "matching_specialist"):
                if set(event.get(role) or {}) != {
                    str(mode) for mode in protocol.MODES
                }:
                    raise ValueError("V24 stationary modes are incomplete")
                for row in event[role].values():
                    if (
                        len(row.get("returns") or [])
                        != protocol.EPISODES_PER_TASK
                        or not all(math.isfinite(float(value))
                                   for value in row["returns"])
                        or int(row.get("total_actions", -1))
                        != protocol.EPISODES_PER_TASK
                        * protocol.MAX_EPISODE_STEPS
                    ):
                        raise ValueError("invalid V24 stationary rollout")
    switching = payload.get("switching_holdout") or {}
    if set(switching) != {
        str(value) for value in protocol.SWITCHING_EVENT_SEEDS
    }:
        raise ValueError("V24 switching holdout is incomplete")
    expected_mode_count = (
        protocol.SWITCHING_EPISODES * protocol.MAX_EPISODE_STEPS
        // len(protocol.MODES)
    )
    hashes = set()
    for event_seed, event in switching.items():
        if set(event) != set(SWITCHING_ARMS):
            raise ValueError("V24 switching arms are incomplete")
        event_hashes = set()
        for row in event.values():
            if (
                len(row.get("returns") or [])
                != protocol.SWITCHING_EPISODES
                or not all(math.isfinite(float(value))
                           for value in row["returns"])
                or int(row.get("total_actions", -1))
                != protocol.SWITCHING_EPISODES
                * protocol.MAX_EPISODE_STEPS
                or row.get("mode_counts") != {
                    str(mode): expected_mode_count for mode in protocol.MODES
                }
                or row.get("base_schedule")
                != list(protocol.SWITCHING_SCHEDULES[int(event_seed)])
            ):
                raise ValueError("invalid V24 switching rollout")
            event_hashes.add(str(row.get("mode_trace_sha256") or ""))
        if len(event_hashes) != 1 or "" in event_hashes:
            raise ValueError("V24 switching arms used different streams")
        hashes.update(event_hashes)
    if len(hashes) != len(protocol.SWITCHING_EVENT_SEEDS):
        raise ValueError("V24 switching event streams are not distinct")
    return manifest


def run(variant: str, seed: int) -> None:
    variant = protocol.require_variant(variant)
    seed = protocol.require_training_seed(seed)
    protocol.validate_registration()
    destination = protocol.audit_dir(variant, seed)
    if protocol.audit_manifest(variant, seed).is_file():
        try:
            validate_audit(variant, seed)
        except (KeyError, OSError, TypeError, ValueError):
            pass
        else:
            print(f"V24 AUDIT ALREADY COMPLETE: {variant} seed={seed}")
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
        controllers = _load_controllers(variant, seed)
        calibration_events = base._stationary_events(
            controllers, seed, protocol.CALIBRATION_EVENT_SEEDS)
        calibration_matrix = base._aggregate_calibration(calibration_events)
        utility_map = base._utility_map(calibration_matrix)
        payload = {
            "schema": protocol.AUDIT_SCHEMA,
            "status": "complete",
            "identity": _identity(variant, seed),
            "registration": protocol.file_record(protocol.REGISTRATION_PATH),
            "source_bundle_manifest": protocol.file_record(
                protocol.source_manifest(seed)),
            "specialist_bundles": protocol.bundle_records(variant, seed),
            "calibration_events": calibration_events,
            "calibration_matrix": calibration_matrix,
            "utility_map": utility_map,
            "stationary_holdout": base._stationary_events(
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
                "identity": _identity(variant, seed),
                "registration": protocol.file_record(
                    protocol.REGISTRATION_PATH),
                "source_bundle_manifest": protocol.file_record(
                    protocol.source_manifest(seed)),
                "specialist_bundles": protocol.bundle_records(variant, seed),
                "audit": protocol.file_record(temporary / "audit.json"),
            },
        )
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_audit(variant, seed)
    print(f"V24 AUDIT COMPLETE: {variant} seed={seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=protocol.VARIANTS, required=True)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.variant, args.seed)


if __name__ == "__main__":
    main()
