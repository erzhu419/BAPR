"""Audit robust-warm-started specialists without a learned estimator."""
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
import numpy as np

from jax_experiments.analysis import final_task_sweep
from jax_experiments.analysis import (
    regime_polarity_robust_warmstart_specialist_v11 as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_safe_utility_confirmation_v9 as parent,
)
from jax_experiments.analysis import (
    run_regime_polarity_robust_warmstart_specialist_v11 as producer,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_capacity_diagnostic_v7 as capacity_audit,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_expected_action_confirmation_audit_v6 as source_audit,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_router_diagnostic_v2 as switching_audit,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_safe_utility_audit_v8 as utility_audit,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_safe_utility_confirmation_audit_v9 as parent_audit,
)
from jax_experiments.common.checkpoint import (
    _patch_flax_variablestate_unpickle,
    _restore_tree_like,
)
from jax_experiments.train import make_env


def _bind() -> None:
    parent_audit._bind()
    capacity_audit.protocol = protocol
    switching_audit.protocol = protocol
    utility_audit.protocol = protocol


def _load_specialist(
    variant: str,
    seed: int,
    mode: int,
    reference: dict[str, Any],
) -> dict[str, Any]:
    producer.validate_bundle(variant, seed, mode)
    path = (
        protocol.bundle_dir(variant, seed, mode)
        / "policy" / protocol.POLICY_NAME
    )
    _patch_flax_variablestate_unpickle()
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    params = _restore_tree_like(
        reference["policy_params"],
        raw,
        "v11 policy parameters",
        allow_fallback=False,
    )
    return {
        "config": copy.deepcopy(reference["config"]),
        "agent": reference["agent"],
        "policy_graphdef": reference["policy_graphdef"],
        "policy_params": params,
        "context_graphdef": None,
        "context_params": None,
    }


def _load_controllers(variant: str, seed: int) -> dict[str, dict[str, Any]]:
    _bind()
    parent.validate_registration()
    robust = source_audit._load_controller("robust_sac", seed)
    controllers = {"robust_sac": robust}
    for mode in protocol.MODES:
        controllers[f"specialist_{mode}"] = _load_specialist(
            variant, seed, mode, robust)
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
            controller["policy_graphdef"], controller["context_graphdef"])
        rewards, dones = env.eval_rollout(
            controller["policy_params"],
            protocol.EPISODES_PER_TASK * protocol.MAX_EPISODE_STEPS,
            jax.random.PRNGKey(int(event_seed) * 100 + mode),
            context_params=controller["context_params"],
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
        specialist = matrix["matching_specialist"][str(mode)]
        relative_gain = (
            (specialist["mean"] - robust["mean"]) / abs(robust["mean"])
            if robust["mean"] != 0.0 else float("-inf")
        )
        event_wins = sum(
            specialist["event_returns"][str(event_seed)]
            > robust["event_returns"][str(event_seed)]
            for event_seed in protocol.CALIBRATION_EVENT_SEEDS
        )
        enabled = bool(
            relative_gain >= protocol.MIN_CALIBRATION_GAIN
            and event_wins == len(protocol.CALIBRATION_EVENT_SEEDS)
            and specialist["terminated_rate"] == 0.0
        )
        mapping[str(mode)] = {
            "controller": f"specialist_{mode}" if enabled else "robust_sac",
            "relative_gain": float(relative_gain),
            "event_wins": int(event_wins),
            "specialist_terminated_rate": float(
                specialist["terminated_rate"]),
        }
    return mapping


def _safe_stack(stack: dict[str, Any], mapping: dict[str, Any]) -> dict[str, Any]:
    actions = dict(stack["actions"])
    for mode in protocol.MODES:
        selected = str(mapping[str(mode)]["controller"])
        actions[f"specialist_{mode}"] = stack["actions"][selected]
    return {"config": stack["config"], "actions": actions}


def _switching_events(
    controllers: dict[str, dict[str, Any]],
    utility_map: dict[str, Any],
) -> dict[str, Any]:
    stack = utility_audit._stack_from_controllers(controllers)
    safe = _safe_stack(stack, utility_map)
    events = {}
    for event_seed in protocol.SWITCHING_EVENT_SEEDS:
        rows = {
            "robust_sac": switching_audit._switching_arm(
                stack, "robust_sac", event_seed),
            "dynamic_specialist_oracle": switching_audit._switching_arm(
                stack, "dynamic_oracle", event_seed),
            "true_mode_safe_utility": switching_audit._switching_arm(
                safe, "dynamic_oracle", event_seed),
        }
        traces = {str(row["mode_trace_sha256"]) for row in rows.values()}
        if len(traces) != 1:
            raise RuntimeError("v11 switching arms used different mode streams")
        events[str(event_seed)] = rows
    return events


def _identity(variant: str, seed: int) -> dict[str, Any]:
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "benchmark_role": "robust_warmstart_specialist_development_audit",
        "variant": protocol.require_variant(variant),
        "training_seed": protocol.require_training_seed(seed),
        "calibration_event_seeds": list(protocol.CALIBRATION_EVENT_SEEDS),
        "stationary_holdout_event_seeds": list(
            protocol.STATIONARY_HOLDOUT_EVENT_SEEDS),
        "switching_event_seeds": list(protocol.SWITCHING_EVENT_SEEDS),
    }


def validate_audit(variant: str, seed: int) -> dict[str, Any]:
    variant = protocol.require_variant(variant)
    seed = protocol.require_training_seed(seed)
    manifest = protocol.read_json(protocol.audit_manifest(variant, seed))
    payload = protocol.read_json(protocol.audit_result(variant, seed))
    if (
        manifest.get("schema") != protocol.AUDIT_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("identity") != _identity(variant, seed)
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
        or payload.get("identity") != _identity(variant, seed)
        or payload.get("source_bundle_manifest")
        != protocol.file_record(protocol.source_manifest(seed))
        or payload.get("specialist_bundles")
        != protocol.bundle_records(variant, seed)
    ):
        raise ValueError("invalid v11 warm-start audit")
    for split_name, event_seeds in (
        ("calibration_events", protocol.CALIBRATION_EVENT_SEEDS),
        ("stationary_holdout", protocol.STATIONARY_HOLDOUT_EVENT_SEEDS),
    ):
        split = payload.get(split_name) or {}
        if set(split) != {str(value) for value in event_seeds}:
            raise ValueError(f"v11 {split_name} is incomplete")
        for event in split.values():
            for role in ("robust_sac", "matching_specialist"):
                if set(event.get(role) or {}) != {
                    str(mode) for mode in protocol.MODES
                }:
                    raise ValueError("v11 stationary modes are incomplete")
                for row in event[role].values():
                    values = row.get("returns") or []
                    if (
                        len(values) != protocol.EPISODES_PER_TASK
                        or not all(math.isfinite(float(value)) for value in values)
                        or int(row.get("total_actions", -1))
                        != protocol.EPISODES_PER_TASK
                        * protocol.MAX_EPISODE_STEPS
                    ):
                        raise ValueError("invalid v11 stationary rollout")
    switching = payload.get("switching_holdout") or {}
    if set(switching) != {
        str(value) for value in protocol.SWITCHING_EVENT_SEEDS
    }:
        raise ValueError("v11 switching holdout is incomplete")
    for event in switching.values():
        if set(event) != {
            "robust_sac",
            "dynamic_specialist_oracle",
            "true_mode_safe_utility",
        }:
            raise ValueError("v11 switching arms are incomplete")
        traces = set()
        for row in event.values():
            values = row.get("returns") or []
            if (
                len(values) != protocol.SWITCHING_EPISODES
                or not all(math.isfinite(float(value)) for value in values)
                or int(row.get("total_actions", -1))
                != protocol.SWITCHING_EPISODES
                * protocol.MAX_EPISODE_STEPS
            ):
                raise ValueError("invalid v11 switching rollout")
            traces.add(str(row.get("mode_trace_sha256") or ""))
        if len(traces) != 1 or "" in traces:
            raise ValueError("v11 switching streams differ")
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
            print(
                f"V11 AUDIT ALREADY COMPLETE: {variant} seed={seed}",
                flush=True,
            )
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
            "specialist_bundles": protocol.bundle_records(variant, seed),
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
                "specialist_bundles": protocol.bundle_records(variant, seed),
                "audit": protocol.file_record(temporary / "audit.json"),
            },
        )
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_audit(variant, seed)
    print(f"V11 AUDIT COMPLETE: {variant} seed={seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=protocol.VARIANTS, required=True)
    parser.add_argument("--seed", choices=protocol.TRAINING_SEEDS,
                        type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.variant, args.seed)


if __name__ == "__main__":
    main()
