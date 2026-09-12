"""Audit V26 joint Ant controllers on frozen stationary and switch streams."""
from __future__ import annotations

import argparse
import copy
import math
import pickle
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
    regime_polarity_ant_joint_mode_risk_v26 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_ant_joint_mode_risk_v26 as producer,
)
from jax_experiments.analysis import (
    run_regime_polarity_ant_switch_recovery_audit_v24 as switching_base,
)
from jax_experiments.common.checkpoint import (
    _patch_flax_variablestate_unpickle,
    _restore_tree_like,
)
from jax_experiments.networks.policy import GaussianPolicy
from jax_experiments.train import make_env


SWITCHING_ARMS = switching_base.SWITCHING_ARMS


def _load_source(seed: int) -> dict[str, Any]:
    config, agent = producer._load_source(seed)
    return {
        "config": config,
        "agent": agent,
        "policy_graphdef": nnx.graphdef(agent.policy),
        "policy_params": nnx.state(agent.policy, nnx.Param),
        "belief_vec": None,
        "direct_context": False,
    }


def _load_joint(variant: str, seed: int, reference: dict[str, Any]):
    producer.validate_bundle(variant, seed)
    model = GaussianPolicy(
        int(reference["agent"].obs_dim),
        int(reference["agent"].act_dim),
        int(reference["config"].hidden_dim),
        ep_dim=len(protocol.MODES),
        n_layers=2,
        rngs=nnx.Rngs(seed + 326_000),
    )
    path = protocol.bundle_dir(variant, seed) / "policy" / protocol.POLICY_NAME
    _patch_flax_variablestate_unpickle()
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    params = _restore_tree_like(
        nnx.state(model, nnx.Param),
        raw,
        "V26 joint policy",
        allow_fallback=False,
    )
    return {
        "config": copy.deepcopy(reference["config"]),
        "agent": reference["agent"],
        "policy_graphdef": nnx.graphdef(model),
        "policy_params": params,
        "direct_context": True,
    }


def _controllers(variant: str, seed: int) -> dict[str, dict[str, Any]]:
    robust = _load_source(seed)
    joint = _load_joint(variant, seed, robust)
    controllers = {"robust_sac": robust}
    for mode in protocol.MODES:
        controller = dict(joint)
        controller["belief_vec"] = jax.nn.one_hot(
            mode, len(protocol.MODES), dtype=jnp.float32)
        controllers[f"specialist_{mode}"] = controller
    return controllers


def _stationary_mode(
    controller: dict[str, Any],
    training_seed: int,
    event_seed: int,
    mode: int,
) -> dict[str, Any]:
    mode = protocol.require_mode(mode)
    config = copy.deepcopy(controller["config"])
    config.stochastic_mode_fixed_id = mode
    env = make_env(config, seed_offset=int(event_seed) - int(training_seed))
    try:
        tasks = env.sample_tasks(len(protocol.MODES))
        env.set_nonstationary_para(tasks)
        env.set_task(tasks[mode])
        env.build_rollout_fn(
            controller["policy_graphdef"],
            direct_policy_context=controller["direct_context"],
        )
        rewards, dones = env.eval_rollout(
            controller["policy_params"],
            protocol.EPISODES_PER_TASK * protocol.MAX_EPISODE_STEPS,
            jax.random.PRNGKey(int(event_seed) * 100 + mode),
            belief_vec=controller.get("belief_vec"),
            episode_horizon=protocol.MAX_EPISODE_STEPS,
        )
    finally:
        if hasattr(env, "close"):
            env.close()
    returns, terminated, steps = final_task_sweep.episode_returns(
        np.asarray(rewards),
        np.asarray(dones),
        protocol.EPISODES_PER_TASK,
        protocol.MAX_EPISODE_STEPS,
    )
    return {
        "returns": [float(value) for value in returns],
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "terminated": [bool(value) for value in terminated],
        "terminated_rate": float(np.mean(terminated)),
        "steps": [int(value) for value in steps],
        "total_actions": protocol.EPISODES_PER_TASK * protocol.MAX_EPISODE_STEPS,
    }


def _stationary_events(
    controllers: dict[str, dict[str, Any]],
    seed: int,
    event_seeds: tuple[int, ...],
) -> dict[str, Any]:
    events = {}
    for event_seed in event_seeds:
        events[str(event_seed)] = {
            "robust_sac": {
                str(mode): _stationary_mode(
                    controllers["robust_sac"], seed, event_seed, mode)
                for mode in protocol.MODES
            },
            "matching_specialist": {
                str(mode): _stationary_mode(
                    controllers[f"specialist_{mode}"],
                    seed,
                    event_seed,
                    mode,
                )
                for mode in protocol.MODES
            },
        }
    return events


def _aggregate_calibration(events: dict[str, Any]) -> dict[str, Any]:
    matrix = {"robust_sac": {}, "matching_specialist": {}}
    for role in matrix:
        for mode in protocol.MODES:
            rows = [
                events[str(event_seed)][role][str(mode)]
                for event_seed in protocol.CALIBRATION_EVENT_SEEDS
            ]
            matrix[role][str(mode)] = {
                "mean": float(np.mean([row["return_mean"] for row in rows])),
                "terminated_rate": float(np.mean(
                    [row["terminated_rate"] for row in rows])),
                "event_returns": {
                    str(event_seed): float(row["return_mean"])
                    for event_seed, row in zip(
                        protocol.CALIBRATION_EVENT_SEEDS, rows)
                },
            }
    return matrix


def _utility_map(matrix: dict[str, Any]) -> dict[str, Any]:
    mapping = {}
    for mode in protocol.MODES:
        robust = matrix["robust_sac"][str(mode)]
        candidate = matrix["matching_specialist"][str(mode)]
        relative_gain = (
            (candidate["mean"] - robust["mean"]) / abs(robust["mean"])
            if robust["mean"] != 0.0 else float("-inf")
        )
        event_wins = sum(
            candidate["event_returns"][str(event_seed)]
            > robust["event_returns"][str(event_seed)]
            for event_seed in protocol.CALIBRATION_EVENT_SEEDS
        )
        enabled = bool(
            relative_gain >= protocol.MIN_CALIBRATION_GAIN
            and event_wins == len(protocol.CALIBRATION_EVENT_SEEDS)
            and candidate["terminated_rate"] == 0.0
        )
        mapping[str(mode)] = {
            "controller": f"specialist_{mode}" if enabled else "robust_sac",
            "relative_gain": float(relative_gain),
            "event_wins": int(event_wins),
            "specialist_terminated_rate": float(candidate["terminated_rate"]),
        }
    return mapping


def _action_stack(
    controllers: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    actions = {}
    for role, controller in controllers.items():
        graphdef = controller["policy_graphdef"]
        context = controller.get("belief_vec")

        @jax.jit
        def action(params, observation, *, _graphdef=graphdef, _context=context):
            policy = nnx.merge(_graphdef, params)
            obs = jnp.asarray(observation, dtype=jnp.float32)[None]
            if _context is None:
                return policy.deterministic(obs)[0]
            return policy.deterministic(obs, _context[None, :])[0]

        def wrapped(observation, *, _fn=action, _controller=controller):
            return np.asarray(
                _fn(_controller["policy_params"], observation),
                dtype=np.float32,
            )

        actions[role] = wrapped
    return {
        "config": controllers["robust_sac"]["config"],
        "actions": actions,
    }


def _safe_stack(
    stack: dict[str, Any], utility_map: dict[str, Any],
) -> dict[str, Any]:
    actions = dict(stack["actions"])
    for mode in protocol.MODES:
        selected = str(utility_map[str(mode)]["controller"])
        actions[f"specialist_{mode}"] = stack["actions"][selected]
    return {"config": stack["config"], "actions": actions}


def _switching_events(
    controllers: dict[str, dict[str, Any]],
    utility_map: dict[str, Any],
) -> dict[str, Any]:
    switching_base.protocol = protocol
    raw = _action_stack(controllers)
    safe = _safe_stack(raw, utility_map)
    events = {}
    event_hashes = set()
    for event_seed in protocol.SWITCHING_EVENT_SEEDS:
        rows = {
            arm: switching_base._switching_arm(
                raw, safe, arm, event_seed)
            for arm in SWITCHING_ARMS
        }
        hashes = {str(row["mode_trace_sha256"]) for row in rows.values()}
        if len(hashes) != 1:
            raise RuntimeError("V26 switching arms used different streams")
        event_hashes.update(hashes)
        events[str(event_seed)] = rows
    if len(event_hashes) != len(protocol.SWITCHING_EVENT_SEEDS):
        raise RuntimeError("V26 switching event traces are not distinct")
    return events


def _identity(variant: str, seed: int) -> dict[str, Any]:
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "benchmark_role": "ant_joint_mode_risk_development_audit",
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
        or manifest.get("controller_bundle")
        != protocol.file_record(protocol.bundle_manifest(variant, seed))
        or manifest.get("audit")
        != protocol.file_record(protocol.audit_result(variant, seed))
        or payload.get("schema") != protocol.AUDIT_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("identity") != identity
    ):
        raise ValueError("invalid V26 audit")
    for split_name, event_seeds in (
        ("calibration_events", protocol.CALIBRATION_EVENT_SEEDS),
        ("stationary_holdout", protocol.STATIONARY_HOLDOUT_EVENT_SEEDS),
    ):
        split = payload.get(split_name) or {}
        if set(split) != {str(value) for value in event_seeds}:
            raise ValueError(f"V26 {split_name} is incomplete")
        for event in split.values():
            for role in ("robust_sac", "matching_specialist"):
                if set(event.get(role) or {}) != {
                    str(mode) for mode in protocol.MODES
                }:
                    raise ValueError("V26 stationary modes are incomplete")
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
                        raise ValueError("invalid V26 stationary rollout")
    switching = payload.get("switching_holdout") or {}
    if set(switching) != {
        str(value) for value in protocol.SWITCHING_EVENT_SEEDS
    }:
        raise ValueError("V26 switching holdout is incomplete")
    expected_mode_count = (
        protocol.SWITCHING_EPISODES * protocol.MAX_EPISODE_STEPS
        // len(protocol.MODES)
    )
    hashes = set()
    for event_seed, event in switching.items():
        if set(event) != set(SWITCHING_ARMS):
            raise ValueError("V26 switching arms are incomplete")
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
                raise ValueError("invalid V26 switching rollout")
            event_hashes.add(str(row.get("mode_trace_sha256") or ""))
        if len(event_hashes) != 1 or "" in event_hashes:
            raise ValueError("V26 switching arms used different streams")
        hashes.update(event_hashes)
    if len(hashes) != len(protocol.SWITCHING_EVENT_SEEDS):
        raise ValueError("V26 switching event streams are not distinct")
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
            print(f"V26 AUDIT ALREADY COMPLETE: {variant} seed={seed}")
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
        controllers = _controllers(variant, seed)
        calibration_events = _stationary_events(
            controllers, seed, protocol.CALIBRATION_EVENT_SEEDS)
        calibration_matrix = _aggregate_calibration(calibration_events)
        utility_map = _utility_map(calibration_matrix)
        payload = {
            "schema": protocol.AUDIT_SCHEMA,
            "status": "complete",
            "identity": _identity(variant, seed),
            "registration": protocol.file_record(protocol.REGISTRATION_PATH),
            "source_bundle_manifest": protocol.file_record(
                protocol.source_manifest(seed)),
            "controller_bundle": protocol.file_record(
                protocol.bundle_manifest(variant, seed)),
            "calibration_events": calibration_events,
            "calibration_matrix": calibration_matrix,
            "utility_map": utility_map,
            "stationary_holdout": _stationary_events(
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
                "controller_bundle": protocol.file_record(
                    protocol.bundle_manifest(variant, seed)),
                "audit": protocol.file_record(temporary / "audit.json"),
            },
        )
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_audit(variant, seed)
    print(f"V26 AUDIT COMPLETE: {variant} seed={seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=protocol.VARIANTS, required=True)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for checkpoint-safe execution")
    run(args.variant, args.seed)


if __name__ == "__main__":
    main()
