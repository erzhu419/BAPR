"""Audit the frozen v12 actor-only policy bank on new data and schedules."""
from __future__ import annotations

import argparse
import copy
import hashlib
import math
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any

import jax
import numpy as np
from flax import nnx

from jax_experiments.analysis import final_task_sweep
from jax_experiments.analysis import (
    regime_polarity_robust_warmstart_confirmation_v12 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_robust_source_v12 as source_runner,
)
from jax_experiments.analysis import (
    run_regime_polarity_robust_warmstart_specialist_audit_v11 as base,
)
from jax_experiments.analysis import (
    run_regime_polarity_robust_warmstart_specialist_v12 as producer,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_safe_utility_audit_v8 as utility_audit,
)
from jax_experiments.common.checkpoint import (
    _patch_flax_variablestate_unpickle,
    _restore_tree_like,
)
from jax_experiments.train import make_algo, make_env


def _bind() -> None:
    base.protocol = protocol
    base.producer = producer
    base._bind()
    utility_audit.protocol = protocol


def _load_source(seed: int) -> dict[str, Any]:
    seed = protocol.require_training_seed(seed)
    source_runner.validate_bundle(seed)
    directory = protocol.source_bundle(seed)
    config = final_task_sweep.load_config(directory)
    config.stochastic_mode_fixed_id = -1
    env = make_env(config, seed_offset=0)
    try:
        tasks = env.sample_tasks(len(protocol.MODES))
        agent = make_algo("sac", env.obs_dim, env.act_dim, config)
        if hasattr(agent, "set_task_metadata"):
            agent.set_task_metadata(tasks)
        template = nnx.state(agent.policy, nnx.Param)
        _patch_flax_variablestate_unpickle()
        with (directory / "policy" / protocol.POLICY_NAME).open("rb") as handle:
            raw = pickle.load(handle)
        params = _restore_tree_like(
            template,
            raw,
            "v12 robust source policy parameters",
            allow_fallback=False,
        )
        return {
            "config": config,
            "agent": agent,
            "policy_graphdef": nnx.graphdef(agent.policy),
            "policy_params": params,
            "context_graphdef": None,
            "context_params": None,
        }
    finally:
        if hasattr(env, "close"):
            env.close()


def _load_controllers(seed: int) -> dict[str, dict[str, Any]]:
    _bind()
    robust = _load_source(seed)
    controllers = {"robust_sac": robust}
    for mode in protocol.MODES:
        controllers[f"specialist_{mode}"] = base._load_specialist(
            "actor_only", seed, mode, robust)
    return controllers


def _switching_arm(
    stack: dict[str, Any], arm: str, event_seed: int,
) -> dict[str, Any]:
    if arm not in {
        "robust_sac", "dynamic_specialist_oracle", "true_mode_safe_utility"
    }:
        raise ValueError(f"unsupported v12 switching arm {arm!r}")
    event_seed = protocol.require_switching_event_seed(event_seed)
    config = copy.deepcopy(stack["config"])
    config.stochastic_mode_fixed_id = -1
    env = make_env(config, seed_offset=event_seed - int(config.seed))
    tasks = env.sample_tasks(len(protocol.MODES))
    actions = stack["actions"]
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
            raise RuntimeError(
                "v12 requires explicit configure_eval_mode_sequence support")
        for episode in range(protocol.SWITCHING_EPISODES):
            sequence = protocol.switching_sequence(event_seed, episode)
            episode_sequences.append(list(sequence))
            configure(tasks, sequence, protocol.DWELL_STEPS)
            observation = env.reset()
            episode_return = 0.0
            episode_terminated = False
            for _ in range(protocol.MAX_EPISODE_STEPS):
                mode = int(env.task_id_for_next_step())
                trace.append(mode)
                mode_counts[str(mode)] += 1
                if arm == "robust_sac":
                    action = actions["robust_sac"](observation)
                    fallback_actions += 1
                else:
                    action = actions[f"specialist_{mode}"](observation)
                    adaptive_actions += 1
                next_observation, reward, done, info = env.step(action)
                if int(info["mode_used"]) != mode:
                    raise RuntimeError("v12 switching action used the wrong mode")
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
        "adaptive_mode_accuracy": (
            1.0 if adaptive_actions else None
        ),
    }


def _switching_events(
    controllers: dict[str, dict[str, Any]],
    utility_map: dict[str, Any],
) -> dict[str, Any]:
    stack = utility_audit._stack_from_controllers(controllers)
    safe = base._safe_stack(stack, utility_map)
    events = {}
    trace_hashes = set()
    for event_seed in protocol.SWITCHING_EVENT_SEEDS:
        rows = {
            "robust_sac": _switching_arm(stack, "robust_sac", event_seed),
            "dynamic_specialist_oracle": _switching_arm(
                stack, "dynamic_specialist_oracle", event_seed),
            "true_mode_safe_utility": _switching_arm(
                safe, "true_mode_safe_utility", event_seed),
        }
        hashes = {str(row["mode_trace_sha256"]) for row in rows.values()}
        if len(hashes) != 1:
            raise RuntimeError("v12 switching arms used different mode streams")
        trace_hashes.update(hashes)
        events[str(event_seed)] = rows
    if len(trace_hashes) != len(protocol.SWITCHING_EVENT_SEEDS):
        raise RuntimeError("v12 switching events did not produce distinct traces")
    return events


def _identity(seed: int) -> dict[str, Any]:
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "benchmark_role": "actor_only_warmstart_policy_bank_confirmation",
        "training_seed": protocol.require_training_seed(seed),
        "calibration_event_seeds": list(protocol.CALIBRATION_EVENT_SEEDS),
        "stationary_holdout_event_seeds": list(
            protocol.STATIONARY_HOLDOUT_EVENT_SEEDS),
        "switching_event_seeds": list(protocol.SWITCHING_EVENT_SEEDS),
        "switching_schedules": {
            str(key): list(value)
            for key, value in protocol.SWITCHING_SCHEDULES.items()
        },
    }


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
        or manifest.get("audit")
        != protocol.file_record(protocol.audit_result(seed))
        or payload.get("schema") != protocol.AUDIT_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("identity") != _identity(seed)
        or payload.get("source_bundle_manifest")
        != protocol.file_record(protocol.source_manifest(seed))
        or payload.get("specialist_bundles") != protocol.bundle_records(seed)
    ):
        raise ValueError("invalid v12 warm-start audit")
    for split_name, event_seeds in (
        ("calibration_events", protocol.CALIBRATION_EVENT_SEEDS),
        ("stationary_holdout", protocol.STATIONARY_HOLDOUT_EVENT_SEEDS),
    ):
        split = payload.get(split_name) or {}
        if set(split) != {str(value) for value in event_seeds}:
            raise ValueError(f"v12 {split_name} is incomplete")
        for event in split.values():
            for role in ("robust_sac", "matching_specialist"):
                if set(event.get(role) or {}) != {
                    str(mode) for mode in protocol.MODES
                }:
                    raise ValueError("v12 stationary modes are incomplete")
                for row in event[role].values():
                    values = row.get("returns") or []
                    if (
                        len(values) != protocol.EPISODES_PER_TASK
                        or not all(math.isfinite(float(value)) for value in values)
                        or int(row.get("total_actions", -1))
                        != protocol.EPISODES_PER_TASK
                        * protocol.MAX_EPISODE_STEPS
                    ):
                        raise ValueError("invalid v12 stationary rollout")
    switching = payload.get("switching_holdout") or {}
    if set(switching) != {
        str(value) for value in protocol.SWITCHING_EVENT_SEEDS
    }:
        raise ValueError("v12 switching holdout is incomplete")
    event_hashes = set()
    expected_mode_count = (
        protocol.SWITCHING_EPISODES * protocol.MAX_EPISODE_STEPS
        // len(protocol.MODES)
    )
    for event_seed, event in switching.items():
        if set(event) != {
            "robust_sac",
            "dynamic_specialist_oracle",
            "true_mode_safe_utility",
        }:
            raise ValueError("v12 switching arms are incomplete")
        arm_hashes = set()
        for row in event.values():
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
                raise ValueError("invalid v12 switching rollout")
            arm_hashes.add(str(row.get("mode_trace_sha256") or ""))
        if len(arm_hashes) != 1 or "" in arm_hashes:
            raise ValueError("v12 switching arms used different streams")
        event_hashes.update(arm_hashes)
    if len(event_hashes) != len(protocol.SWITCHING_EVENT_SEEDS):
        raise ValueError("v12 switching event traces are not distinct")
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
            print(f"V12 CONFIRMATION AUDIT ALREADY COMPLETE: seed={seed}")
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
        controllers = _load_controllers(seed)
        calibration_events = base._stationary_events(
            controllers, seed, protocol.CALIBRATION_EVENT_SEEDS)
        calibration_matrix = base._aggregate_calibration(calibration_events)
        utility_map = base._utility_map(calibration_matrix)
        payload = {
            "schema": protocol.AUDIT_SCHEMA,
            "status": "complete",
            "identity": _identity(seed),
            "registration": protocol.file_record(protocol.REGISTRATION_PATH),
            "source_bundle_manifest": protocol.file_record(
                protocol.source_manifest(seed)),
            "specialist_bundles": protocol.bundle_records(seed),
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
                "identity": _identity(seed),
                "registration": protocol.file_record(
                    protocol.REGISTRATION_PATH),
                "source_bundle_manifest": protocol.file_record(
                    protocol.source_manifest(seed)),
                "specialist_bundles": protocol.bundle_records(seed),
                "audit": protocol.file_record(temporary / "audit.json"),
            },
        )
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_audit(seed)
    print(f"V12 CONFIRMATION AUDIT COMPLETE: seed={seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.seed)


if __name__ == "__main__":
    main()
