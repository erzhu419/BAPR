"""Strict stationary and switching audit of the V33 training recipes."""
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
    regime_polarity_canonical_training_ablation_v33 as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_model_v5 as estimator_model,
)
from jax_experiments.analysis import (
    run_regime_polarity_action_compensation_audit_v28 as compensation,
)
from jax_experiments.analysis import (
    run_regime_polarity_canonical_training_direct_v33 as direct_runner,
)
from jax_experiments.analysis import (
    run_regime_polarity_canonical_training_warmstart_v33 as warmstart_runner,
)
from jax_experiments.analysis import (
    run_regime_polarity_v5_final_comparison_audit_v18 as rollout,
)
from jax_experiments.common.checkpoint import (
    _patch_flax_variablestate_unpickle,
    _restore_tree_like,
)
from jax_experiments.train import make_algo, make_env


def arm(recipe: str, route: str) -> str:
    recipe = protocol.require_recipe(recipe)
    if recipe not in protocol.CANONICAL_RECIPES:
        raise ValueError(f"recipe {recipe!r} has no compensation route")
    if route not in {"no_compensation", "oracle", "causal"}:
        raise ValueError(f"unknown V33 route {route!r}")
    return f"{recipe}_{route}"


def _bind_rollouts() -> None:
    rollout.protocol = protocol
    compensation.protocol = protocol
    compensation.base = rollout
    compensation.v5_model = estimator_model


def _load_policy_tree(path: Path):
    _patch_flax_variablestate_unpickle()
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (dict, nnx.State)):
        raise ValueError(f"expected policy parameter tree: {path}")
    return payload


def _load_controller(recipe: str, seed: int) -> dict[str, Any]:
    recipe = protocol.require_recipe(recipe)
    seed = protocol.require_training_seed(seed)
    if recipe == protocol.WARMSTART_RECIPE:
        warmstart_runner.validate_bundle(seed)
    else:
        direct_runner.validate_bundle(recipe, seed)
    directory = protocol.recipe_bundle_dir(recipe, seed)
    config = final_task_sweep.load_config(directory)
    # Training-only canonicalization must never leak into evaluation.
    config.stochastic_mode_fixed_id = -1
    config.stochastic_mode_canonical_reference_id = -1
    env = make_env(config, seed_offset=0)
    try:
        agent = make_algo("sac", env.obs_dim, env.act_dim, config)
        raw = _load_policy_tree(directory / "policy" / protocol.POLICY_NAME)
        params = _restore_tree_like(
            nnx.state(agent.policy, nnx.Param),
            raw,
            f"V33 {recipe} policy",
            allow_fallback=False,
        )
        graphdef = nnx.graphdef(agent.policy)
    finally:
        if hasattr(env, "close"):
            env.close()

    @jax.jit
    def apply(policy_params, observation):
        policy = nnx.merge(graphdef, policy_params)
        obs = jnp.asarray(observation, dtype=jnp.float32)[None]
        return policy.deterministic(obs)[0]

    def action(observation):
        return np.asarray(apply(params, observation), dtype=np.float32)

    return {"config": config, "action": action}


def _candidate_stack(controller: dict[str, Any]) -> dict[str, Any]:
    return {
        "config": controller["config"],
        "actions": {f"specialist_{protocol.REFERENCE_MODE}": controller["action"]},
    }


def _stationary(seed: int, controllers: dict[str, dict[str, Any]]):
    events: dict[str, Any] = {}
    for event_seed in protocol.STATIONARY_HOLDOUT_EVENT_SEEDS:
        rows = {name: {} for name in protocol.ARMS}
        robust = controllers[protocol.ROBUST_RECIPE]
        for mode in protocol.MODES:
            key = str(mode)
            rows[protocol.ROBUST_ARM][key] = rollout._stationary_direct(
                robust["config"], robust["action"], event_seed, mode,
                protocol.STATIONARY_EPISODES_PER_MODE)
            for recipe in protocol.CANONICAL_RECIPES:
                controller = controllers[recipe]
                stack = _candidate_stack(controller)
                rows[arm(recipe, "no_compensation")][key] = (
                    rollout._stationary_direct(
                        controller["config"], controller["action"],
                        event_seed, mode,
                        protocol.STATIONARY_EPISODES_PER_MODE))
                rows[arm(recipe, "oracle")][key] = (
                    compensation._compensated_stationary(
                        stack, protocol.REFERENCE_MODE, "true_mode",
                        event_seed, mode))
                rows[arm(recipe, "causal")][key] = (
                    compensation._compensated_stationary(
                        stack, protocol.REFERENCE_MODE, "v5_posterior",
                        event_seed, mode))
        events[str(event_seed)] = rows
        print(f"V33 stationary seed={seed} event={event_seed} complete", flush=True)
    return events


def _switching(seed: int, controllers: dict[str, dict[str, Any]]):
    events: dict[str, Any] = {}
    equivalence: dict[str, Any] = {
        recipe: {} for recipe in protocol.CANONICAL_RECIPES
    }
    for event_seed in protocol.SWITCHING_EVENT_SEEDS:
        robust = controllers[protocol.ROBUST_RECIPE]
        robust_stack = {
            "config": robust["config"],
            "actions": {protocol.ROBUST_ARM: robust["action"]},
        }
        rows = {
            protocol.ROBUST_ARM: rollout._switching_bank(
                robust_stack, {}, "fixed", event_seed,
                fixed_controller=protocol.ROBUST_ARM),
        }
        for recipe in protocol.CANONICAL_RECIPES:
            controller = controllers[recipe]
            stack = _candidate_stack(controller)
            reference = f"specialist_{protocol.REFERENCE_MODE}"
            rows[arm(recipe, "no_compensation")] = rollout._switching_bank(
                stack, {}, "fixed", event_seed,
                fixed_controller=reference)
            rows[arm(recipe, "oracle")] = compensation._compensated_switching(
                stack, protocol.REFERENCE_MODE, "true_mode", event_seed)
            rows[arm(recipe, "causal")] = compensation._compensated_switching(
                stack, protocol.REFERENCE_MODE, "v5_posterior", event_seed)
            native = rollout._stationary_direct(
                controller["config"], controller["action"], event_seed,
                protocol.REFERENCE_MODE, protocol.SWITCHING_EPISODES)
            oracle_returns = np.asarray(
                rows[arm(recipe, "oracle")]["returns"], dtype=np.float64)
            native_returns = np.asarray(native["returns"], dtype=np.float64)
            error = float(np.max(np.abs(oracle_returns - native_returns)))
            equivalence[recipe][str(event_seed)] = {
                "native_reference_returns": native["returns"],
                "oracle_compensation_returns": rows[arm(recipe, "oracle")][
                    "returns"
                ],
                "max_abs_return_error": error,
                "pass": bool(error <= protocol.EXACT_RETURN_ATOL),
            }
        hashes = {str(row["mode_trace_sha256"]) for row in rows.values()}
        if len(hashes) != 1:
            raise RuntimeError("V33 switching arms used different mode streams")
        events[str(event_seed)] = rows
        print(f"V33 switching seed={seed} event={event_seed} complete", flush=True)
    return events, equivalence


def identity(seed: int) -> dict[str, Any]:
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "benchmark_role": "canonical_controller_training_ablation",
        "training_seed": protocol.require_training_seed(seed),
        "recipes": list(protocol.RECIPES),
        "reference_mode": protocol.REFERENCE_MODE,
        "stationary_event_seeds": list(protocol.STATIONARY_HOLDOUT_EVENT_SEEDS),
        "switching_event_seeds": list(protocol.SWITCHING_EVENT_SEEDS),
        "switching_schedules": {
            str(key): list(value)
            for key, value in protocol.SWITCHING_SCHEDULES.items()
        },
    }


def evaluate(seed: int) -> dict[str, Any]:
    seed = protocol.require_training_seed(seed)
    _bind_rollouts()
    controllers = {
        recipe: _load_controller(recipe, seed) for recipe in protocol.RECIPES
    }
    stationary = _stationary(seed, controllers)
    switching, equivalence = _switching(seed, controllers)
    return {
        "schema": protocol.AUDIT_SCHEMA,
        "status": "complete",
        "identity": identity(seed),
        "registration": protocol.file_record(protocol.REGISTRATION_PATH),
        "frozen_inputs": protocol.frozen_input_records(seed),
        "evaluation_config": {
            "stochastic_mode_fixed_id": -1,
            "stochastic_mode_canonical_reference_id": -1,
            "frozen_v5_estimator": True,
        },
        "stationary_holdout": stationary,
        "switching_holdout": switching,
        "oracle_equivalence": equivalence,
    }


def validate_result(payload: dict[str, Any], seed: int) -> None:
    if (
        payload.get("schema") != protocol.AUDIT_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("identity") != identity(seed)
        or payload.get("registration")
        != protocol.file_record(protocol.REGISTRATION_PATH)
        or payload.get("frozen_inputs") != protocol.frozen_input_records(seed)
        or payload.get("evaluation_config", {}).get(
            "stochastic_mode_canonical_reference_id") != -1
    ):
        raise ValueError("invalid V33 audit identity or evaluation boundary")
    stationary = payload.get("stationary_holdout") or {}
    if set(stationary) != {
        str(value) for value in protocol.STATIONARY_HOLDOUT_EVENT_SEEDS
    }:
        raise ValueError("incomplete V33 stationary events")
    for event in stationary.values():
        if set(event) != set(protocol.ARMS):
            raise ValueError("incomplete V33 stationary arms")
        for rows in event.values():
            if set(rows) != {str(mode) for mode in protocol.MODES}:
                raise ValueError("incomplete V33 stationary modes")
            for row in rows.values():
                returns = row.get("returns") or []
                if (
                    len(returns) != protocol.STATIONARY_EPISODES_PER_MODE
                    or not all(math.isfinite(float(value)) for value in returns)
                ):
                    raise ValueError("invalid V33 stationary rollout")
    switching = payload.get("switching_holdout") or {}
    if set(switching) != {
        str(value) for value in protocol.SWITCHING_EVENT_SEEDS
    }:
        raise ValueError("incomplete V33 switching events")
    expected_actions = protocol.SWITCHING_EPISODES * protocol.MAX_EPISODE_STEPS
    for event in switching.values():
        if set(event) != set(protocol.ARMS):
            raise ValueError("incomplete V33 switching arms")
        hashes = set()
        for row in event.values():
            returns = row.get("returns") or []
            if (
                len(returns) != protocol.SWITCHING_EPISODES
                or not all(math.isfinite(float(value)) for value in returns)
                or int(row.get("total_actions", -1)) != expected_actions
            ):
                raise ValueError("invalid V33 switching rollout")
            hashes.add(str(row.get("mode_trace_sha256") or ""))
        if len(hashes) != 1 or "" in hashes:
            raise ValueError("V33 switching arms differ in event stream")
    equivalence = payload.get("oracle_equivalence") or {}
    if set(equivalence) != set(protocol.CANONICAL_RECIPES):
        raise ValueError("incomplete V33 oracle equivalence")
    for rows in equivalence.values():
        if set(rows) != {str(seed) for seed in protocol.SWITCHING_EVENT_SEEDS}:
            raise ValueError("incomplete V33 equivalence events")
        if not all(row.get("pass") is True for row in rows.values()):
            raise ValueError("V33 oracle trajectory equivalence failed")


def validate_audit(seed: int) -> dict[str, Any]:
    seed = protocol.require_training_seed(seed)
    manifest = protocol.read_json(protocol.audit_manifest(seed))
    payload = protocol.read_json(protocol.audit_result(seed))
    validate_result(payload, seed)
    if (
        manifest.get("schema") != protocol.AUDIT_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("identity") != identity(seed)
        or manifest.get("registration")
        != protocol.file_record(protocol.REGISTRATION_PATH)
        or manifest.get("frozen_inputs") != protocol.frozen_input_records(seed)
        or manifest.get("audit") != protocol.file_record(protocol.audit_result(seed))
    ):
        raise ValueError("invalid V33 audit manifest")
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
            print(f"V33 AUDIT ALREADY COMPLETE: seed={seed}", flush=True)
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
        payload = evaluate(seed)
        validate_result(payload, seed)
        protocol.write_json_atomic(temporary / "audit.json", payload)
        protocol.write_json_atomic(
            temporary / "audit_manifest.json",
            {
                "schema": protocol.AUDIT_SCHEMA,
                "status": "complete",
                "identity": identity(seed),
                "registration": protocol.file_record(protocol.REGISTRATION_PATH),
                "frozen_inputs": protocol.frozen_input_records(seed),
                "audit": protocol.file_record(temporary / "audit.json"),
            },
        )
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_audit(seed)
    print(f"V33 AUDIT COMPLETE: seed={seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for idempotent execution")
    run(args.seed)


if __name__ == "__main__":
    main()
