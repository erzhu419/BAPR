"""Evaluate one posterior-conditioned residual variant on one dev stream."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.analysis import (
    bapr_v3_learned_control_router as estimator,
)
from jax_experiments.analysis import bapr_v3_posterior_residual as protocol
from jax_experiments.analysis import (
    run_bapr_v3_independent_specialist_audit as specialist_audit,
)
from jax_experiments.analysis import (
    run_bapr_v3_learned_control_router_audit as estimator_audit,
)
from jax_experiments.analysis import (
    run_bapr_v3_utility_aware_router_audit as base_audit,
)
from jax_experiments.train import make_env


CONTROLLERS = (
    "robust",
    "dynamic_utility_oracle",
    "learned_utility_router",
    "posterior_residual",
)
ADAPTIVE_CONTROLLERS = (
    "learned_utility_router",
    "posterior_residual",
)


def source_records() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(),
        Path(protocol.__file__).resolve(),
        Path(base_audit.__file__).resolve(),
        Path(protocol.utility.__file__).resolve(),
        Path(specialist_audit.__file__).resolve(),
        Path(estimator_audit.__file__).resolve(),
    )
    return {
        str(path.relative_to(protocol.ROOT)): protocol.file_record(path)
        for path in paths
    }


def primary_endpoints() -> dict[str, float]:
    return {
        "min_full_cycle_mean_gain": protocol.MIN_FULL_CYCLE_MEAN_GAIN,
        "stationary_mean_margin": protocol.STATIONARY_MEAN_MARGIN,
        "stationary_per_seed_margin":
            protocol.STATIONARY_PER_SEED_MARGIN,
        "termination_rate_margin": protocol.TERMINATION_RATE_MARGIN,
        "min_adaptation_rate": protocol.MIN_ADAPTATION_RATE,
    }


def _policy_action(action_fn, policy_states, key: str, observation):
    return np.asarray(action_fn(
        policy_states[key],
        jnp.asarray(observation, dtype=jnp.float32)), dtype=np.float32)


def _action_and_diagnostics(
    controller: str,
    physics_mode: int,
    router: base_audit.UtilityRouterRuntime,
    oracle_map: tuple[int, ...],
    policy_states: dict[str, Any],
    action_fn,
    table: dict[str, Any],
    cap: float,
    observation,
):
    robust_action = _policy_action(
        action_fn, policy_states, "robust", observation)
    if controller == "robust":
        return robust_action, protocol.utility.ROBUST_CONTROLLER, None
    if controller == "dynamic_utility_oracle":
        selected = int(oracle_map[physics_mode])
        return (
            _policy_action(
                action_fn, policy_states,
                base_audit._policy_key(selected), observation),
            selected,
            None,
        )
    if controller == "learned_utility_router":
        selected, diagnostics = router.decision()
        action = _policy_action(
            action_fn, policy_states,
            base_audit._policy_key(selected), observation)
        strength = float(selected not in (
            protocol.utility.FALLBACK_CONTROLLER,
            protocol.utility.ROBUST_CONTROLLER,
        ))
    elif controller == "posterior_residual":
        selected, strength, diagnostics = (
            protocol.posterior_residual_decision(
                np.asarray(router.state[0]), table, cap))
        if selected == protocol.utility.ROBUST_CONTROLLER:
            specialist_action = robust_action
        else:
            specialist_action = _policy_action(
                action_fn, policy_states,
                base_audit._policy_key(selected), observation)
        action = protocol.blend_action(
            robust_action, specialist_action, strength)
    else:
        raise ValueError(f"unknown residual-screen controller {controller}")
    diagnostics = dict(diagnostics)
    diagnostics.update({
        "adaptation_strength": float(strength),
        "action_delta_l2": float(np.linalg.norm(action - robust_action)),
        "selected_controller": float(selected),
    })
    return action, int(selected), diagnostics


def _routing_row(
    decisions, physics_modes, oracle_map, diagnostics, episode: int,
):
    row = base_audit._routing_summary(
        decisions, physics_modes, oracle_map)
    row.update({
        "episode": int(episode),
        "confidence_mean": float(np.mean([
            value["confidence"] for value in diagnostics])),
        "best_expected_advantage_mean": float(np.mean([
            value["best_expected_advantage"] for value in diagnostics])),
        "adaptation_strength_mean": float(np.mean([
            value["adaptation_strength"] for value in diagnostics])),
        "adaptation_positive_rate": float(np.mean([
            value["adaptation_strength"] > 0.0 for value in diagnostics])),
        "action_delta_l2_mean": float(np.mean([
            value["action_delta_l2"] for value in diagnostics])),
        "action_delta_l2_max": float(np.max([
            value["action_delta_l2"] for value in diagnostics])),
    })
    return row


def evaluate_stationary(
    config,
    policy_states: dict[str, Any],
    action_fn,
    model,
    observe,
    table,
    router_config,
    oracle_map: tuple[int, ...],
    event_seed: int,
    cap: float,
):
    horizon = int(config.max_episode_steps)
    output = {controller: {} for controller in CONTROLLERS}
    for controller in CONTROLLERS:
        for mode in range(4):
            run_config = copy.deepcopy(config)
            run_config.stochastic_mode_fixed_id = mode
            env = make_env(run_config, seed_offset=event_seed)
            tasks = env.sample_tasks(4)
            env.set_nonstationary_para(tasks)
            env.set_task(tasks[mode])
            episodes = []
            route_rows = []
            for episode in range(base_audit.EPISODES_PER_TASK):
                base_key = (
                    20260718 + event_seed * 100_000
                    + mode * 10_000 + episode * 1000)
                env.rng = jax.random.PRNGKey(base_key)
                observation = env.reset()
                router = base_audit.UtilityRouterRuntime(
                    model, observe, table, router_config)
                total_return = 0.0
                decisions = []
                physics_modes = []
                diagnostic_rows = []
                terminated = False
                steps = horizon
                for step in range(horizon):
                    action, selected, diagnostics = _action_and_diagnostics(
                        controller, mode, router, oracle_map, policy_states,
                        action_fn, table, cap, observation)
                    env.rng = jax.random.PRNGKey(base_key + step * 2 + 1)
                    next_observation, reward, done, info = env.step(action)
                    if int(info["mode_used"]) != mode:
                        raise RuntimeError("stationary residual audit changed mode")
                    if controller in ADAPTIVE_CONTROLLERS:
                        router.observe(
                            observation, action, reward,
                            next_observation, done)
                        diagnostic_rows.append(diagnostics)
                    decisions.append(int(selected))
                    physics_modes.append(mode)
                    total_return += float(reward)
                    observation = next_observation
                    if done:
                        terminated = True
                        steps = step + 1
                        break
                episodes.append({
                    "episode": episode,
                    "return": total_return,
                    "terminated": terminated,
                    "steps": steps,
                })
                if controller in ADAPTIVE_CONTROLLERS:
                    route_rows.append(_routing_row(
                        decisions, physics_modes, oracle_map,
                        diagnostic_rows, episode))
            output[controller][str(mode)] = base_audit._stationary_record(
                episodes, route_rows)
            if hasattr(env, "close"):
                env.close()
    return output


def evaluate_switching_kind(
    config,
    policy_states: dict[str, Any],
    action_fn,
    model,
    observe,
    table,
    router_config,
    oracle_map: tuple[int, ...],
    event_seed: int,
    kind: str,
    cap: float,
):
    horizon = int(config.max_episode_steps)
    output = {}
    kind_offset = 0 if kind == "slow_pair" else 500_000
    for controller in CONTROLLERS:
        run_config = copy.deepcopy(config)
        run_config.stochastic_mode_fixed_id = -1
        env = make_env(run_config, seed_offset=event_seed)
        tasks = env.sample_tasks(4)
        episodes = []
        route_rows = []
        for episode in range(base_audit.SWITCHING_EPISODES):
            sequence = base_audit._configure_switching(
                env, tasks, kind, episode)
            base_key = (
                20260718 + event_seed * 100_000
                + kind_offset + episode * 10_000)
            env.rng = jax.random.PRNGKey(base_key)
            observation = env.reset()
            router = base_audit.UtilityRouterRuntime(
                model, observe, table, router_config)
            total_return = 0.0
            termination_count = 0
            first_done_step = horizon
            decisions = []
            physics_modes = []
            diagnostic_rows = []
            for step in range(horizon):
                physics_mode = int(env.task_id_for_next_step())
                action, selected, diagnostics = _action_and_diagnostics(
                    controller, physics_mode, router, oracle_map,
                    policy_states, action_fn, table, cap, observation)
                env.rng = jax.random.PRNGKey(base_key + step * 2 + 1)
                next_observation, reward, done, info = env.step(action)
                if int(info["mode_used"]) != physics_mode:
                    raise RuntimeError(
                        "switching residual action used the wrong mode")
                if controller in ADAPTIVE_CONTROLLERS:
                    router.observe(
                        observation, action, reward, next_observation, done)
                    diagnostic_rows.append(diagnostics)
                decisions.append(int(selected))
                physics_modes.append(physics_mode)
                total_return += float(reward)
                observation = next_observation
                if done:
                    termination_count += 1
                    first_done_step = min(first_done_step, step + 1)
                    env.rng = jax.random.PRNGKey(base_key + step * 2 + 2)
                    observation = env.reset()
            episodes.append({
                "episode": episode,
                "return": total_return,
                "termination_count": termination_count,
                "first_done_step": first_done_step,
                "configured_mode_sequence": list(sequence),
                "physics_mode_counts": {
                    str(mode): int(physics_modes.count(mode))
                    for mode in range(4)
                },
            })
            if controller in ADAPTIVE_CONTROLLERS:
                row = _routing_row(
                    decisions, physics_modes, oracle_map,
                    diagnostic_rows, episode)
                row["physics_mode_counts"] = {
                    str(mode): int(physics_modes.count(mode))
                    for mode in range(4)
                }
                route_rows.append(row)
        if hasattr(env, "close"):
            env.close()
        returns = [float(row["return"]) for row in episodes]
        output[controller] = {
            "episodes": episodes,
            "mean": float(np.mean(returns)),
            "std": float(np.std(returns)),
            "routing": route_rows,
        }
    return output


def validate_group(payload: dict[str, Any]) -> None:
    event_seed = int(payload.get("event_seed", -1))
    variant = str(payload.get("residual_variant"))
    try:
        expected_cap = protocol.residual_cap(variant)
    except ValueError as exc:
        raise ValueError("invalid residual-screen identity") from exc
    if (payload.get("schema") != protocol.GROUP_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("family") != protocol.FAMILY
            or payload.get("env") != protocol.ENV
            or payload.get("decision_variant") != protocol.DECISION_VARIANT
            or event_seed not in protocol.DEVELOPMENT_EVENT_SEEDS
            or float(payload.get("residual_cap", -1.0)) != expected_cap
            or payload.get("primary_endpoints") != primary_endpoints()
            or payload.get("source_files") != source_records()
            or set(payload.get("stationary") or {}) != set(CONTROLLERS)
            or set(payload.get("switching") or {})
            != {"slow_pair", "full_cycle"}):
        raise ValueError("invalid residual-screen group identity")
    if not math.isfinite(float(payload.get("advantage_scale", math.nan))):
        raise ValueError("invalid residual advantage scale")
    for controller in CONTROLLERS:
        if set(payload["stationary"][controller]) != {"0", "1", "2", "3"}:
            raise ValueError("incomplete residual-screen stationary modes")
        for record in payload["stationary"][controller].values():
            episodes = record.get("episodes") or []
            if (len(episodes) != base_audit.EPISODES_PER_TASK
                    or not all(math.isfinite(float(row["return"]))
                               for row in episodes)):
                raise ValueError("invalid residual-screen stationary episodes")
        for kind in ("slow_pair", "full_cycle"):
            record = payload["switching"][kind][controller]
            episodes = record.get("episodes") or []
            if (len(episodes) != base_audit.SWITCHING_EPISODES
                    or not all(math.isfinite(float(row["return"]))
                               for row in episodes)):
                raise ValueError("invalid residual-screen switching episodes")
    residual_sections = list(
        payload["stationary"]["posterior_residual"].values())
    residual_sections += [
        payload["switching"][kind]["posterior_residual"]
        for kind in ("slow_pair", "full_cycle")
    ]
    for section in residual_sections:
        for row in section.get("routing") or []:
            strength = float(row["adaptation_strength_mean"])
            if (not math.isfinite(strength)
                    or not 0.0 <= strength <= expected_cap + 1e-7):
                raise ValueError("residual strength exceeded its frozen cap")


def run(variant: str, event_seed: int) -> Path:
    protocol.configure()
    cap = protocol.residual_cap(variant)
    result_path = protocol.group_path(variant, event_seed)
    output = result_path.parent
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        validate_group(payload)
        print(f"Complete posterior-residual screen exists: {output}")
        return result_path

    table = protocol.utility.load_utility_table()
    advantage_scale = protocol.residual_advantage_scale(table)
    estimator_manifest = estimator.load_manifest()
    oracle_map = tuple(int(value) for value in table["oracle_controller_map"])
    bundles = (
        protocol.utility.control.specialist_protocol.validate_family_bundles(
            protocol.FAMILY))
    bundle_root = (
        protocol.utility.control.specialist_protocol.family_bundle_root(
            protocol.FAMILY))
    bundle_hashes = {
        name: estimator.sha256_file(
            bundle_root / name
            / protocol.utility.control.specialist_protocol.BUNDLE_MANIFEST)
        for name in bundles
    }
    if bundle_hashes != table["bundle_manifest_sha256"]:
        raise RuntimeError("controller bank changed after utility freeze")

    source_before = (
        protocol.utility.control.fork_protocol.current_source_manifest())
    config, agents, policy_states = specialist_audit._controller_policy_states(
        protocol.FAMILY)
    action_fn = specialist_audit._base_action_fn(
        nnx.graphdef(agents["robust"].policy))
    base_router_config = estimator.RouterConfig.from_dict(
        estimator_manifest["router_config"])
    router_config = protocol.utility.decision_config(
        base_router_config, protocol.DECISION_VARIANT)
    model, router_config, observe = estimator_audit.load_router(
        estimator_manifest, agents["robust"].obs_dim,
        agents["robust"].act_dim,
        router_config_override=router_config)
    stationary = evaluate_stationary(
        config, policy_states, action_fn, model, observe, table,
        router_config, oracle_map, event_seed, cap)
    switching = {
        kind: evaluate_switching_kind(
            config, policy_states, action_fn, model, observe, table,
            router_config, oracle_map, event_seed, kind, cap)
        for kind in ("slow_pair", "full_cycle")
    }
    source_after = (
        protocol.utility.control.fork_protocol.current_source_manifest())
    if source_after["sha256"] != source_before["sha256"]:
        raise RuntimeError("source changed during posterior-residual screen")

    payload = {
        "schema": protocol.GROUP_SCHEMA,
        "status": "complete",
        "family": protocol.FAMILY,
        "env": protocol.ENV,
        "training_seed": 0,
        "event_seed": int(event_seed),
        "event_seed_role": "posterior residual development stream",
        "decision_variant": protocol.DECISION_VARIANT,
        "residual_variant": variant,
        "residual_cap": cap,
        "advantage_scale": advantage_scale,
        "advantage_scale_rule": (
            "median positive specialist-over-robust headroom in the "
            "calibration-frozen utility table"),
        "primary_endpoints": primary_endpoints(),
        "oracle_controller_map": list(oracle_map),
        "utility_table_file": protocol.file_record(
            protocol.utility.TABLE_PATH),
        "estimator_manifest_file": protocol.file_record(
            estimator.MANIFEST_PATH),
        "estimator_parameter_file": protocol.file_record(
            estimator.MODEL_PATH),
        "bundle_manifest_sha256": bundle_hashes,
        "source_snapshot_sha256": source_before["sha256"],
        "source_files": source_records(),
        "router_config": router_config.to_dict(),
        "episodes_per_task": base_audit.EPISODES_PER_TASK,
        "switching_episodes": base_audit.SWITCHING_EPISODES,
        "slow_dwell_steps": base_audit.SLOW_DWELL_STEPS,
        "full_cycle_dwell_steps": base_audit.FULL_CYCLE_DWELL_STEPS,
        "stationary": stationary,
        "switching": switching,
    }
    validate_group(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.tmp.", dir=output.parent))
    try:
        protocol.write_json_atomic(temporary / "group.json", payload)
        if output.exists():
            shutil.rmtree(output)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"POSTERIOR RESIDUAL SCREEN COMPLETE: {output}", flush=True)
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=tuple(protocol.RESIDUAL_VARIANTS), required=True)
    parser.add_argument("--event-seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.variant, args.event_seed)


if __name__ == "__main__":
    main()
