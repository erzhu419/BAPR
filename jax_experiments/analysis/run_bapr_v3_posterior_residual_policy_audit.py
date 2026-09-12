"""Strict development audit for the nonlinear posterior residual policy."""
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
    bapr_v3_posterior_residual_policy as protocol,
)
from jax_experiments.analysis import (
    run_bapr_v3_independent_specialist_audit as specialist_audit,
)
from jax_experiments.analysis import (
    run_bapr_v3_utility_aware_router_audit as base_audit,
)
from jax_experiments.train import make_env


STRENGTH_RTOL = 5e-6
STRENGTH_ATOL = 1e-6


def residual_strengths_agree(model_strength: float,
                             reference_strength: float) -> bool:
    """Allow float32/float64 utility arithmetic noise, not protocol drift."""
    return bool(np.isclose(
        model_strength,
        reference_strength,
        rtol=STRENGTH_RTOL,
        atol=STRENGTH_ATOL,
    ))


def source_records() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(),
        Path(protocol.__file__).resolve(),
        Path(base_audit.__file__).resolve(),
    )
    return {
        str(path.relative_to(protocol.ROOT)): protocol.file_record(path)
        for path in paths
    }


def primary_endpoints() -> dict[str, float]:
    return {
        "min_full_cycle_mean_gain":
            protocol.screen.MIN_FULL_CYCLE_MEAN_GAIN,
        "stationary_mean_margin":
            protocol.screen.STATIONARY_MEAN_MARGIN,
        "stationary_per_seed_margin":
            protocol.screen.STATIONARY_PER_SEED_MARGIN,
        "termination_rate_margin":
            protocol.screen.TERMINATION_RATE_MARGIN,
        "min_adaptation_rate": protocol.screen.MIN_ADAPTATION_RATE,
    }


def make_residual_action(model, params, obs_mean, obs_std, table):
    graphdef = nnx.graphdef(model)
    matrix, robust_index, specialist_indices, scale = (
        protocol.utility_constants(table))
    matrix = jnp.asarray(matrix)
    mean = jnp.asarray(obs_mean)
    std = jnp.asarray(obs_std)

    @jax.jit
    def action(observation, posterior, robust_action):
        current = nnx.merge(graphdef, params)
        strength = protocol.posterior_strength(
            posterior, matrix, robust_index, specialist_indices, scale)
        selected = current(
            observation, posterior, robust_action, strength, mean, std)
        return selected, strength

    return action


def _base_action(action_fn, policy_states, key: str, observation):
    return np.asarray(action_fn(
        policy_states[key],
        jnp.asarray(observation, dtype=jnp.float32)), dtype=np.float32)


def _residual_decision(
    router,
    table,
    model_action,
    action_fn,
    policy_states,
    observation,
):
    posterior = np.asarray(router.state[0], dtype=np.float32)
    robust_action = _base_action(
        action_fn, policy_states, "robust", observation)
    action, strength = model_action(
        jnp.asarray(observation, dtype=jnp.float32),
        jnp.asarray(posterior, dtype=jnp.float32),
        jnp.asarray(robust_action, dtype=jnp.float32))
    action = np.asarray(action, dtype=np.float32)
    strength = float(strength)
    selected, expected_strength, diagnostics = (
        protocol.screen.posterior_residual_decision(
            posterior, table, cap=1.0))
    if not residual_strengths_agree(strength, expected_strength):
        raise RuntimeError(
            "training and audit residual strengths disagree: "
            f"model={strength:.12g}, reference={expected_strength:.12g}, "
            f"posterior={posterior.tolist()}")
    diagnostics = dict(diagnostics)
    diagnostics.update({
        "adaptation_strength": strength,
        "action_delta_l2": float(np.linalg.norm(action - robust_action)),
        "selected_controller": float(selected),
    })
    return action, int(selected), diagnostics


def _routing_row(decisions, physics_modes, oracle_map, diagnostics, episode):
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
    policy_states,
    action_fn,
    router_model,
    observe,
    table,
    router_config,
    oracle_map,
    model_action,
    event_seed,
):
    horizon = int(config.max_episode_steps)
    output = {}
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
                router_model, observe, table, router_config)
            total_return = 0.0
            decisions = []
            physics_modes = []
            diagnostics = []
            terminated = False
            steps = horizon
            for step in range(horizon):
                action, selected, row = _residual_decision(
                    router, table, model_action, action_fn,
                    policy_states, observation)
                env.rng = jax.random.PRNGKey(base_key + step * 2 + 1)
                next_observation, reward, done, info = env.step(action)
                if int(info["mode_used"]) != mode:
                    raise RuntimeError("stationary nonlinear audit changed mode")
                router.observe(
                    observation, action, reward, next_observation, done)
                decisions.append(selected)
                physics_modes.append(mode)
                diagnostics.append(row)
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
            route_rows.append(_routing_row(
                decisions, physics_modes, oracle_map, diagnostics, episode))
        output[str(mode)] = base_audit._stationary_record(
            episodes, route_rows)
        if hasattr(env, "close"):
            env.close()
    return output


def evaluate_switching_kind(
    config,
    policy_states,
    action_fn,
    router_model,
    observe,
    table,
    router_config,
    oracle_map,
    model_action,
    event_seed,
    kind,
):
    horizon = int(config.max_episode_steps)
    run_config = copy.deepcopy(config)
    run_config.stochastic_mode_fixed_id = -1
    env = make_env(run_config, seed_offset=event_seed)
    tasks = env.sample_tasks(4)
    episodes = []
    route_rows = []
    kind_offset = 0 if kind == "slow_pair" else 500_000
    for episode in range(base_audit.SWITCHING_EPISODES):
        sequence = base_audit._configure_switching(env, tasks, kind, episode)
        base_key = (
            20260718 + event_seed * 100_000
            + kind_offset + episode * 10_000)
        env.rng = jax.random.PRNGKey(base_key)
        observation = env.reset()
        router = base_audit.UtilityRouterRuntime(
            router_model, observe, table, router_config)
        total_return = 0.0
        termination_count = 0
        first_done_step = horizon
        decisions = []
        physics_modes = []
        diagnostics = []
        for step in range(horizon):
            physics_mode = int(env.task_id_for_next_step())
            action, selected, row = _residual_decision(
                router, table, model_action, action_fn,
                policy_states, observation)
            env.rng = jax.random.PRNGKey(base_key + step * 2 + 1)
            next_observation, reward, done, info = env.step(action)
            if int(info["mode_used"]) != physics_mode:
                raise RuntimeError("switching nonlinear audit used wrong mode")
            router.observe(
                observation, action, reward, next_observation, done)
            decisions.append(selected)
            physics_modes.append(physics_mode)
            diagnostics.append(row)
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
                str(mode): int(physics_modes.count(mode)) for mode in range(4)
            },
        })
        route = _routing_row(
            decisions, physics_modes, oracle_map, diagnostics, episode)
        route["physics_mode_counts"] = {
            str(mode): int(physics_modes.count(mode)) for mode in range(4)
        }
        route_rows.append(route)
    if hasattr(env, "close"):
        env.close()
    returns = [float(row["return"]) for row in episodes]
    return {
        "episodes": episodes,
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns)),
        "routing": route_rows,
    }


def validate_group(payload: dict[str, Any]) -> None:
    event_seed = int(payload.get("event_seed", -1))
    if (payload.get("schema") != protocol.GROUP_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("family") != protocol.FAMILY
            or payload.get("env") != protocol.ENV
            or payload.get("decision_variant") != protocol.DECISION_VARIANT
            or event_seed not in protocol.DEVELOPMENT_RETURN_EVENT_SEEDS
            or payload.get("primary_endpoints") != primary_endpoints()
            or payload.get("source_files") != source_records()
            or set(payload.get("stationary") or {}) != {"0", "1", "2", "3"}
            or set(payload.get("switching") or {})
            != {"slow_pair", "full_cycle"}
            or payload.get("policy_manifest_file")
            != protocol.file_record(protocol.MANIFEST_PATH)
            or payload.get("policy_parameter_file")
            != protocol.file_record(protocol.MODEL_PATH)):
        raise ValueError("invalid nonlinear residual audit identity")
    reference = protocol.screen.utility.audit_group_path(
        "validation", event_seed, protocol.DECISION_VARIANT)
    if payload.get("reference_baseline_file") != protocol.file_record(reference):
        raise ValueError("nonlinear residual baseline reference changed")
    for record in payload["stationary"].values():
        episodes = record.get("episodes") or []
        if (len(episodes) != base_audit.EPISODES_PER_TASK
                or not all(math.isfinite(float(row["return"]))
                           for row in episodes)):
            raise ValueError("invalid nonlinear stationary episodes")
    for kind in ("slow_pair", "full_cycle"):
        episodes = payload["switching"][kind].get("episodes") or []
        if (len(episodes) != base_audit.SWITCHING_EPISODES
                or not all(math.isfinite(float(row["return"]))
                           for row in episodes)):
            raise ValueError("invalid nonlinear switching episodes")
    for section in (
            *payload["stationary"].values(),
            payload["switching"]["slow_pair"],
            payload["switching"]["full_cycle"]):
        for row in section.get("routing") or []:
            strength = float(row["adaptation_strength_mean"])
            if not 0.0 <= strength <= 1.0 + 1e-7:
                raise ValueError("invalid nonlinear residual strength")


def run(event_seed: int) -> Path:
    protocol.configure()
    result_path = protocol.audit_group_path(event_seed)
    output = result_path.parent
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        validate_group(payload)
        print(f"Complete nonlinear residual audit exists: {output}")
        return result_path

    table = protocol.screen.utility.load_utility_table()
    policy_model, policy_params, obs_mean, obs_std, manifest = (
        protocol.load_policy())
    oracle_map = tuple(int(value) for value in table["oracle_controller_map"])
    reference_path = protocol.screen.utility.audit_group_path(
        "validation", event_seed, protocol.DECISION_VARIANT)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    base_audit.validate_group(reference)

    source_before = (
        protocol.screen.utility.control.fork_protocol.current_source_manifest())
    config, agents, policy_states = specialist_audit._controller_policy_states(
        protocol.FAMILY)
    action_fn = specialist_audit._base_action_fn(
        nnx.graphdef(agents["robust"].policy))
    estimator_manifest = protocol.estimator.load_manifest()
    base_router_config = protocol.estimator.RouterConfig.from_dict(
        estimator_manifest["router_config"])
    router_config = protocol.screen.utility.decision_config(
        base_router_config, protocol.DECISION_VARIANT)
    from jax_experiments.analysis import (
        run_bapr_v3_learned_control_router_audit as estimator_audit,
    )
    router_model, router_config, observe = estimator_audit.load_router(
        estimator_manifest, agents["robust"].obs_dim,
        agents["robust"].act_dim,
        router_config_override=router_config)
    model_action = make_residual_action(
        policy_model, policy_params, obs_mean, obs_std, table)
    stationary = evaluate_stationary(
        config, policy_states, action_fn, router_model, observe, table,
        router_config, oracle_map, model_action, event_seed)
    switching = {
        kind: evaluate_switching_kind(
            config, policy_states, action_fn, router_model, observe, table,
            router_config, oracle_map, model_action, event_seed, kind)
        for kind in ("slow_pair", "full_cycle")
    }
    source_after = (
        protocol.screen.utility.control.fork_protocol.current_source_manifest())
    if source_after["sha256"] != source_before["sha256"]:
        raise RuntimeError("source changed during nonlinear residual audit")

    payload = {
        "schema": protocol.GROUP_SCHEMA,
        "status": "complete",
        "family": protocol.FAMILY,
        "env": protocol.ENV,
        "training_seed": 0,
        "event_seed": int(event_seed),
        "event_seed_role": "nonlinear residual development return stream",
        "decision_variant": protocol.DECISION_VARIANT,
        "primary_endpoints": primary_endpoints(),
        "oracle_controller_map": list(oracle_map),
        "reference_baseline_file": protocol.file_record(reference_path),
        "policy_manifest_file": protocol.file_record(protocol.MANIFEST_PATH),
        "policy_parameter_file": protocol.file_record(protocol.MODEL_PATH),
        "policy_selected_stage": manifest["selected_stage"],
        "source_snapshot_sha256": source_before["sha256"],
        "source_files": source_records(),
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
    print(f"NONLINEAR RESIDUAL AUDIT COMPLETE: {output}", flush=True)
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.event_seed)


if __name__ == "__main__":
    main()
