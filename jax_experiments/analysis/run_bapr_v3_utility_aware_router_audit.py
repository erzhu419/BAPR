"""Strict audit for posterior-utility routing with explicit robust control."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.analysis import (
    bapr_v3_learned_control_router as estimator,
)
from jax_experiments.analysis import bapr_v3_utility_aware_router as protocol
from jax_experiments.analysis import (
    run_bapr_v3_independent_specialist_audit as specialist_audit,
)
from jax_experiments.analysis import (
    run_bapr_v3_learned_control_router_audit as estimator_audit,
)
from jax_experiments.train import make_env, _reset_eval_switch_schedule


CONTROLLERS = ("robust", "dynamic_utility_oracle", "learned_utility_router")
EPISODES_PER_TASK = 5
SWITCHING_EPISODES = 5
SLOW_DWELL_STEPS = 500
FULL_CYCLE_DWELL_STEPS = 250
FULL_CYCLE_SEQUENCES = (
    (0, 1, 2, 3),
    (3, 2, 1, 0),
    (0, 2, 1, 3),
    (1, 3, 0, 2),
    (2, 0, 3, 1),
)


class UtilityRouterRuntime:
    def __init__(self, model, observe, table, router_config):
        self.model = model
        self.observe_fn = observe
        self.table = table
        self.config = router_config
        self.reset()

    def reset(self):
        self.state = self.model.initial_state()

    def decision(self):
        return protocol.select_utility_controller(
            self.state[0], int(self.state[-1]), self.table, self.config)

    def observe(self, obs, action, reward, next_obs, done):
        self.state = self.observe_fn(
            self.state,
            jnp.asarray(obs, dtype=jnp.float32),
            jnp.asarray(action, dtype=jnp.float32),
            jnp.asarray(reward, dtype=jnp.float32),
            jnp.asarray(next_obs, dtype=jnp.float32),
            jnp.asarray(done, dtype=jnp.float32),
        )


def _policy_key(controller: int) -> str:
    if controller in (
            protocol.FALLBACK_CONTROLLER, protocol.ROBUST_CONTROLLER):
        return "robust"
    return f"fixed_mode_{int(controller)}"


def _controller_decision(
    controller: str,
    physics_mode: int,
    router: UtilityRouterRuntime,
    oracle_map: tuple[int, ...],
):
    if controller == "robust":
        return protocol.ROBUST_CONTROLLER, None
    if controller == "dynamic_utility_oracle":
        return int(oracle_map[physics_mode]), None
    return router.decision()


def _routing_summary(decisions, physics_modes, oracle_map):
    metrics = protocol.routing_metrics(
        np.asarray(decisions, dtype=np.int32),
        np.asarray(physics_modes, dtype=np.int32),
        oracle_map,
        burnin=32,
    )
    counts = Counter(int(value) for value in decisions)
    metrics["selected_controller_counts"] = {
        str(controller): int(count)
        for controller, count in sorted(counts.items())
    }
    return metrics


def _stationary_record(episodes, routing):
    returns = [float(row["return"]) for row in episodes]
    return {
        "episodes": episodes,
        "returns": returns,
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns)),
        "terminated_rate": float(np.mean([
            bool(row["terminated"]) for row in episodes])),
        "steps_mean": float(np.mean([
            int(row["steps"]) for row in episodes])),
        "routing": routing,
    }


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
            for episode in range(EPISODES_PER_TASK):
                base_key = (
                    20260718 + event_seed * 100_000
                    + mode * 10_000 + episode * 1000)
                env.rng = jax.random.PRNGKey(base_key)
                observation = env.reset()
                router = UtilityRouterRuntime(
                    model, observe, table, router_config)
                total_return = 0.0
                decisions = []
                physics_modes = []
                diagnostic_rows = []
                terminated = False
                steps = horizon
                for step in range(horizon):
                    selected, diagnostics = _controller_decision(
                        controller, mode, router, oracle_map)
                    action = np.asarray(action_fn(
                        policy_states[_policy_key(selected)],
                        jnp.asarray(observation, dtype=jnp.float32)))
                    env.rng = jax.random.PRNGKey(base_key + step * 2 + 1)
                    next_observation, reward, done, info = env.step(action)
                    if int(info["mode_used"]) != mode:
                        raise RuntimeError("stationary utility audit changed mode")
                    if controller == "learned_utility_router":
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
                if controller == "learned_utility_router":
                    row = _routing_summary(
                        decisions, physics_modes, oracle_map)
                    row.update({
                        "episode": episode,
                        "confidence_mean": float(np.mean([
                            value["confidence"]
                            for value in diagnostic_rows])),
                        "best_expected_advantage_mean": float(np.mean([
                            value["best_expected_advantage"]
                            for value in diagnostic_rows])),
                    })
                    route_rows.append(row)
            output[controller][str(mode)] = _stationary_record(
                episodes, route_rows)
            if hasattr(env, "close"):
                env.close()
    return output


def _configure_switching(env, tasks, kind: str, episode: int):
    if kind == "slow_pair":
        _reset_eval_switch_schedule(
            env, tasks, None, SLOW_DWELL_STEPS)
        return tuple(int(value) for value in env._eval_mode_sequence)
    sequence = FULL_CYCLE_SEQUENCES[episode % len(FULL_CYCLE_SEQUENCES)]
    env.configure_eval_mode_sequence(
        tasks, sequence, FULL_CYCLE_DWELL_STEPS)
    return sequence


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
        for episode in range(SWITCHING_EPISODES):
            sequence = _configure_switching(env, tasks, kind, episode)
            base_key = (
                20260718 + event_seed * 100_000
                + kind_offset + episode * 10_000)
            env.rng = jax.random.PRNGKey(base_key)
            observation = env.reset()
            router = UtilityRouterRuntime(
                model, observe, table, router_config)
            total_return = 0.0
            termination_count = 0
            first_done_step = horizon
            decisions = []
            physics_modes = []
            diagnostic_rows = []
            for step in range(horizon):
                physics_mode = int(env.task_id_for_next_step())
                selected, diagnostics = _controller_decision(
                    controller, physics_mode, router, oracle_map)
                action = np.asarray(action_fn(
                    policy_states[_policy_key(selected)],
                    jnp.asarray(observation, dtype=jnp.float32)))
                env.rng = jax.random.PRNGKey(base_key + step * 2 + 1)
                next_observation, reward, done, info = env.step(action)
                if int(info["mode_used"]) != physics_mode:
                    raise RuntimeError(
                        "switching utility action used the wrong mode")
                if controller == "learned_utility_router":
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
            if controller == "learned_utility_router":
                row = _routing_summary(
                    decisions, physics_modes, oracle_map)
                row.update({
                    "episode": episode,
                    "confidence_mean": float(np.mean([
                        value["confidence"] for value in diagnostic_rows])),
                    "best_expected_advantage_mean": float(np.mean([
                        value["best_expected_advantage"]
                        for value in diagnostic_rows])),
                    "physics_mode_counts": {
                        str(mode): int(physics_modes.count(mode))
                        for mode in range(4)
                    },
                })
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


def validate_group(payload):
    role = str(payload.get("role"))
    event_seed = int(payload.get("event_seed", -1))
    decision_variant = str(payload.get(
        "decision_variant", protocol.BASELINE_DECISION_VARIANT))
    if (payload.get("schema") != protocol.AUDIT_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("family") != protocol.FAMILY
            or payload.get("env") != protocol.ENV
            or role not in ("validation", "holdout")
            or decision_variant not in protocol.DECISION_VARIANTS
            or event_seed not in protocol.event_seeds(role)
            or set(payload.get("stationary") or {}) != set(CONTROLLERS)
            or set(payload.get("switching") or {})
            != {"slow_pair", "full_cycle"}):
        raise ValueError("invalid utility-router audit identity")
    oracle_map = tuple(int(value) for value in
                       payload.get("oracle_controller_map", []))
    if len(oracle_map) != 4:
        raise ValueError("utility-router oracle map is incomplete")
    for controller in CONTROLLERS:
        if set(payload["stationary"][controller]) != {"0", "1", "2", "3"}:
            raise ValueError("stationary utility modes are incomplete")
        for record in payload["stationary"][controller].values():
            values = record.get("returns") or []
            if (len(values) != EPISODES_PER_TASK
                    or not all(math.isfinite(float(value))
                               for value in values)):
                raise ValueError("invalid stationary utility returns")
        for kind in ("slow_pair", "full_cycle"):
            record = payload["switching"][kind][controller]
            episodes = record.get("episodes") or []
            if (len(episodes) != SWITCHING_EPISODES
                    or not all(math.isfinite(float(row["return"]))
                               for row in episodes)):
                raise ValueError("invalid switching utility returns")
            if kind == "full_cycle":
                for episode in episodes:
                    if any(int(episode["physics_mode_counts"][str(mode)]) <= 0
                           for mode in range(4)):
                        raise ValueError(
                            "full-cycle audit did not visit every mode")
    valid_decisions = {
        protocol.FALLBACK_CONTROLLER,
        *protocol.ALL_CONTROLLERS,
    }
    route_sections = list(
        payload["stationary"]["learned_utility_router"].values())
    route_sections += [
        payload["switching"][kind]["learned_utility_router"]
        for kind in ("slow_pair", "full_cycle")
    ]
    for section in route_sections:
        for row in section["routing"]:
            selected = set(int(value) for value in
                           row["selected_controller_counts"])
            if not selected.issubset(valid_decisions):
                raise ValueError("router selected an invalid controller")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("validation", "holdout"),
                        required=True)
    parser.add_argument("--event-seed", type=int, required=True)
    parser.add_argument(
        "--decision-variant", choices=tuple(protocol.DECISION_VARIANTS),
        default=protocol.BASELINE_DECISION_VARIANT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol.configure()
    if args.event_seed not in protocol.event_seeds(args.role):
        raise ValueError(
            f"event seed {args.event_seed} is not sealed for {args.role}")
    output = args.out_dir.resolve()
    result_path = output / "group.json"
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        validate_group(payload)
        print(f"Complete utility-router audit exists: {output}")
        return

    table = protocol.load_utility_table()
    estimator_manifest = estimator.load_manifest()
    oracle_map = tuple(int(value) for value in
                       table["oracle_controller_map"])
    bundles = protocol.control.specialist_protocol.validate_family_bundles(
        protocol.FAMILY)
    bundle_root = protocol.control.specialist_protocol.family_bundle_root(
        protocol.FAMILY)
    bundle_hashes = {
        name: estimator.sha256_file(
            bundle_root / name
            / protocol.control.specialist_protocol.BUNDLE_MANIFEST)
        for name in bundles
    }
    if bundle_hashes != table["bundle_manifest_sha256"]:
        raise RuntimeError("controller bank changed after utility freeze")

    source_before = protocol.control.fork_protocol.current_source_manifest()
    config, agents, policy_states = specialist_audit._controller_policy_states(
        protocol.FAMILY)
    action_fn = specialist_audit._base_action_fn(
        nnx.graphdef(agents["robust"].policy))
    base_router_config = estimator.RouterConfig.from_dict(
        estimator_manifest["router_config"])
    router_config = protocol.decision_config(
        base_router_config, args.decision_variant)
    model, router_config, observe = estimator_audit.load_router(
        estimator_manifest, agents["robust"].obs_dim,
        agents["robust"].act_dim,
        router_config_override=router_config)
    stationary = evaluate_stationary(
        config, policy_states, action_fn, model, observe, table,
        router_config, oracle_map, args.event_seed)
    switching = {
        kind: evaluate_switching_kind(
            config, policy_states, action_fn, model, observe, table,
            router_config, oracle_map, args.event_seed, kind)
        for kind in ("slow_pair", "full_cycle")
    }
    source_after = protocol.control.fork_protocol.current_source_manifest()
    if source_after["sha256"] != source_before["sha256"]:
        raise RuntimeError("source changed during utility-router audit")

    payload = {
        "schema": protocol.AUDIT_SCHEMA,
        "status": "complete",
        "family": protocol.FAMILY,
        "env": protocol.ENV,
        "training_seed": 0,
        "role": args.role,
        "decision_variant": args.decision_variant,
        "event_seed": int(args.event_seed),
        "event_seed_role": (
            "utility decision validation stream"
            if args.role == "validation" else
            "untouched sealed utility-router holdout stream"),
        "oracle_controller_map": list(oracle_map),
        "utility_table_file": estimator.file_record(protocol.TABLE_PATH),
        "estimator_manifest_file": estimator.file_record(
            estimator.MANIFEST_PATH),
        "estimator_parameter_file": estimator.file_record(
            estimator.MODEL_PATH),
        "bundle_manifest_sha256": bundle_hashes,
        "source_snapshot_sha256": source_before["sha256"],
        "router_config": router_config.to_dict(),
        "episodes_per_task": EPISODES_PER_TASK,
        "switching_episodes": SWITCHING_EPISODES,
        "slow_dwell_steps": SLOW_DWELL_STEPS,
        "full_cycle_dwell_steps": FULL_CYCLE_DWELL_STEPS,
        "stationary": stationary,
        "switching": switching,
    }
    validate_group(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.tmp.", dir=output.parent))
    try:
        estimator.write_json_atomic(temporary / "group.json", payload)
        if output.exists():
            shutil.rmtree(output)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"UTILITY ROUTER AUDIT COMPLETE: {output}", flush=True)


if __name__ == "__main__":
    main()
