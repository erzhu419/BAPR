"""Audit posterior utility fallback filling on one development stream."""
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
from jax_experiments.analysis import (
    bapr_v3_posterior_discrete_selector as protocol,
)
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


def source_records() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(),
        Path(protocol.__file__).resolve(),
        Path(base_audit.__file__).resolve(),
        Path(protocol.prior.utility.__file__).resolve(),
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
        "stationary_per_seed_margin": protocol.STATIONARY_PER_SEED_MARGIN,
        "termination_rate_margin": protocol.TERMINATION_RATE_MARGIN,
        "min_adaptation_rate": protocol.MIN_ADAPTATION_RATE,
    }


def _policy_action(action_fn, policy_states, selected: int, observation):
    return np.asarray(action_fn(
        policy_states[base_audit._policy_key(selected)],
        jnp.asarray(observation, dtype=jnp.float32)), dtype=np.float32)


def _decision(router, table, action_fn, policy_states, observation):
    posterior = np.asarray(router.state[0], dtype=np.float32)
    hard_selected, hard_diagnostics = router.decision()
    utility_selected, diagnostics = (
        protocol.select_expected_utility_controller(posterior, table))
    hard_fallback = bool(hard_diagnostics["fallback"])
    selected = int(utility_selected if hard_fallback else hard_selected)
    if selected == protocol.prior.utility.FALLBACK_CONTROLLER:
        raise RuntimeError("posterior-discrete selector emitted fallback code")
    robust_action = _policy_action(
        action_fn, policy_states, protocol.prior.utility.ROBUST_CONTROLLER,
        observation)
    action = _policy_action(action_fn, policy_states, selected, observation)
    diagnostics = dict(diagnostics)
    diagnostics.update({
        "hard_fallback": float(hard_fallback),
        "fallback_replaced": float(
            hard_fallback
            and selected != protocol.prior.utility.ROBUST_CONTROLLER),
        "adaptation_strength": float(
            selected != protocol.prior.utility.ROBUST_CONTROLLER),
        "action_delta_l2": float(np.linalg.norm(action - robust_action)),
        "selected_controller": float(selected),
    })
    return action, selected, diagnostics


def _routing_row(decisions, physics_modes, oracle_map, diagnostics, episode):
    row = base_audit._routing_summary(decisions, physics_modes, oracle_map)
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
        "hard_fallback_rate": float(np.mean([
            value["hard_fallback"] for value in diagnostics])),
        "fallback_replacement_rate": float(np.mean([
            value["fallback_replaced"] for value in diagnostics])),
    })
    return row


def evaluate_stationary(
    config, policy_states, action_fn, router_model, observe, table,
    router_config, oracle_map, event_seed,
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
                action, selected, row = _decision(
                    router, table, action_fn, policy_states, observation)
                env.rng = jax.random.PRNGKey(base_key + step * 2 + 1)
                next_observation, reward, done, info = env.step(action)
                if int(info["mode_used"]) != mode:
                    raise RuntimeError("stationary discrete audit changed mode")
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
    config, policy_states, action_fn, router_model, observe, table,
    router_config, oracle_map, event_seed, kind,
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
            action, selected, row = _decision(
                router, table, action_fn, policy_states, observation)
            env.rng = jax.random.PRNGKey(base_key + step * 2 + 1)
            next_observation, reward, done, info = env.step(action)
            if int(info["mode_used"]) != physics_mode:
                raise RuntimeError("switching discrete audit used wrong mode")
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
            or event_seed not in protocol.DEVELOPMENT_EVENT_SEEDS
            or payload.get("primary_endpoints") != primary_endpoints()
            or payload.get("source_files") != source_records()
            or set(payload.get("stationary") or {}) != {"0", "1", "2", "3"}
            or set(payload.get("switching") or {})
            != {"slow_pair", "full_cycle"}):
        raise ValueError("invalid posterior-discrete audit identity")
    reference = protocol.prior.utility.audit_group_path(
        "validation", event_seed, protocol.DECISION_VARIANT)
    if payload.get("reference_baseline_file") != protocol.file_record(reference):
        raise ValueError("posterior-discrete baseline reference changed")
    for record in payload["stationary"].values():
        episodes = record.get("episodes") or []
        if (len(episodes) != base_audit.EPISODES_PER_TASK
                or not all(math.isfinite(float(row["return"]))
                           for row in episodes)):
            raise ValueError("invalid posterior-discrete stationary episodes")
    for kind in ("slow_pair", "full_cycle"):
        episodes = payload["switching"][kind].get("episodes") or []
        if (len(episodes) != base_audit.SWITCHING_EPISODES
                or not all(math.isfinite(float(row["return"]))
                           for row in episodes)):
            raise ValueError("invalid posterior-discrete switching episodes")


def run(event_seed: int) -> Path:
    protocol.configure()
    result_path = protocol.group_path(event_seed)
    output = result_path.parent
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        validate_group(payload)
        print(f"Complete posterior-discrete audit exists: {output}")
        return result_path

    table = protocol.prior.utility.load_utility_table()
    oracle_map = tuple(int(value) for value in table["oracle_controller_map"])
    reference_path = protocol.prior.utility.audit_group_path(
        "validation", event_seed, protocol.DECISION_VARIANT)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    base_audit.validate_group(reference)
    bundles = (
        protocol.prior.utility.control.specialist_protocol
        .validate_family_bundles(protocol.FAMILY))
    bundle_root = (
        protocol.prior.utility.control.specialist_protocol
        .family_bundle_root(protocol.FAMILY))
    bundle_hashes = {
        name: estimator.sha256_file(
            bundle_root / name
            / protocol.prior.utility.control.specialist_protocol
            .BUNDLE_MANIFEST)
        for name in bundles
    }
    if bundle_hashes != table["bundle_manifest_sha256"]:
        raise RuntimeError("controller bank changed after utility freeze")

    source_before = (
        protocol.prior.utility.control.fork_protocol.current_source_manifest())
    config, agents, policy_states = specialist_audit._controller_policy_states(
        protocol.FAMILY)
    action_fn = specialist_audit._base_action_fn(
        nnx.graphdef(agents["robust"].policy))
    estimator_manifest = estimator.load_manifest()
    base_router_config = estimator.RouterConfig.from_dict(
        estimator_manifest["router_config"])
    router_config = protocol.prior.utility.decision_config(
        base_router_config, protocol.DECISION_VARIANT)
    router_model, router_config, observe = estimator_audit.load_router(
        estimator_manifest, agents["robust"].obs_dim,
        agents["robust"].act_dim,
        router_config_override=router_config)
    stationary = evaluate_stationary(
        config, policy_states, action_fn, router_model, observe, table,
        router_config, oracle_map, event_seed)
    switching = {
        kind: evaluate_switching_kind(
            config, policy_states, action_fn, router_model, observe, table,
            router_config, oracle_map, event_seed, kind)
        for kind in ("slow_pair", "full_cycle")
    }
    source_after = (
        protocol.prior.utility.control.fork_protocol.current_source_manifest())
    if source_after["sha256"] != source_before["sha256"]:
        raise RuntimeError("source changed during posterior-discrete audit")

    payload = {
        "schema": protocol.GROUP_SCHEMA,
        "status": "complete",
        "family": protocol.FAMILY,
        "env": protocol.ENV,
        "training_seed": 0,
        "event_seed": int(event_seed),
        "event_seed_role": "posterior-discrete development return stream",
        "decision_variant": protocol.DECISION_VARIANT,
        "selection_rule": (
            "preserve eligible hard-CUSUM decisions; on hard fallback choose "
            "the frozen controller with maximum posterior expected utility"),
        "primary_endpoints": primary_endpoints(),
        "oracle_controller_map": list(oracle_map),
        "reference_baseline_file": protocol.file_record(reference_path),
        "utility_table_file": protocol.file_record(
            protocol.prior.utility.TABLE_PATH),
        "estimator_manifest_file": protocol.file_record(
            estimator.MANIFEST_PATH),
        "estimator_parameter_file": protocol.file_record(estimator.MODEL_PATH),
        "bundle_manifest_sha256": bundle_hashes,
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
    print(f"POSTERIOR DISCRETE AUDIT COMPLETE: {output}", flush=True)
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.event_seed)


if __name__ == "__main__":
    main()
