"""Strict paired holdout audit for the learned control-equivalence router."""
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

from jax_experiments.analysis import bapr_v3_learned_control_router as protocol
from jax_experiments.analysis import (
    run_bapr_v3_independent_specialist_audit as specialist_audit,
)
from jax_experiments.analysis.train_bapr_v3_learned_control_router import (
    MODEL_CONFIG,
    make_model,
)
from jax_experiments.train import (
    _reset_eval_switch_schedule,
    _select_eval_switch_sequence,
    make_env,
)


CONTROLLERS = ("robust", "dynamic_control_oracle", "learned_router")
EPISODES_PER_TASK = 5
SWITCHING_EPISODES = 5
SWITCHING_PERIOD_STEPS = 500


def load_router(
    manifest,
    obs_dim: int,
    act_dim: int,
    router_config_override=None,
):
    if manifest.get("model_config") != MODEL_CONFIG:
        raise ValueError("learned-router architecture changed after training")
    router_config = (
        protocol.RouterConfig.from_dict(manifest["router_config"])
        if router_config_override is None else router_config_override)
    model = make_model(
        obs_dim, act_dim, seed=20260718, router_config=router_config)
    template = nnx.state(model, nnx.Param)
    params = protocol.load_parameter_state(
        protocol.MODEL_PATH, template, manifest["parameter_leaves"])
    nnx.update(model, params)
    graphdef = nnx.graphdef(model)

    @jax.jit
    def observe(state, obs, action, reward, next_obs, done):
        current = nnx.merge(graphdef, params)
        return current.observe(
            state, obs, action, reward, next_obs, done,
            enable_reset=False, stop_variance_grad=True)[0]

    return model, router_config, observe


class RouterRuntime:
    def __init__(self, model, router_config, observe, controller_map):
        self.model = model
        self.config = router_config
        self.observe_fn = observe
        self.controller_map = tuple(controller_map)
        self.reset()

    def reset(self):
        self.state = self.model.initial_state()
        self.previous_controller = -1

    def decision(self):
        selected, diagnostics = protocol.select_controller(
            self.state[0], int(self.state[-1]), self.controller_map,
            self.config, self.previous_controller)
        self.previous_controller = selected
        return selected, diagnostics

    def observe(self, obs, action, reward, next_obs, done):
        self.state = self.observe_fn(
            self.state,
            jnp.asarray(obs, dtype=jnp.float32),
            jnp.asarray(action, dtype=jnp.float32),
            jnp.asarray(reward, dtype=jnp.float32),
            jnp.asarray(next_obs, dtype=jnp.float32),
            jnp.asarray(done, dtype=jnp.float32),
        )


def _routing_summary(decisions, physics_modes, controller_map):
    metrics = protocol.routing_metrics(
        np.asarray(decisions, dtype=np.int32),
        np.asarray(physics_modes, dtype=np.int32),
        controller_map,
        burnin=32,
    )
    metrics["selected_controller_counts"] = {
        str(controller): int(decisions.count(controller))
        for controller in (-1, 0, 2, 3)
        if decisions.count(controller)
    }
    return metrics


def _stationary_record(episode_rows, route_rows):
    returns = [float(row["return"]) for row in episode_rows]
    return {
        "episodes": episode_rows,
        "returns": returns,
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns)),
        "terminated_rate": float(np.mean([
            bool(row["terminated"]) for row in episode_rows])),
        "steps_mean": float(np.mean([
            int(row["steps"]) for row in episode_rows])),
        "routing": route_rows,
    }


def evaluate_stationary(
    config,
    policy_states: dict[str, Any],
    action_fn,
    model,
    router_config,
    observe,
    controller_map,
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
            episode_rows = []
            route_rows = []
            for episode in range(EPISODES_PER_TASK):
                base_key = (
                    20260718 + event_seed * 100_000
                    + mode * 10_000 + episode * 1000)
                env.rng = jax.random.PRNGKey(base_key)
                observation = env.reset()
                router = RouterRuntime(
                    model, router_config, observe, controller_map)
                total_return = 0.0
                decisions = []
                physics_modes = []
                confidence = []
                margin = []
                terminated = False
                steps = horizon
                for step in range(horizon):
                    if controller == "robust":
                        selected = -1
                    elif controller == "dynamic_control_oracle":
                        selected = int(controller_map[mode])
                    else:
                        selected, diagnostics = router.decision()
                        confidence.append(float(diagnostics["confidence"]))
                        margin.append(float(diagnostics["margin"]))
                    policy_key = (
                        "robust" if selected < 0
                        else f"fixed_mode_{selected}")
                    action = np.asarray(action_fn(
                        policy_states[policy_key],
                        jnp.asarray(observation, dtype=jnp.float32)))
                    env.rng = jax.random.PRNGKey(base_key + step * 2 + 1)
                    next_observation, reward, done, info = env.step(action)
                    if int(info["mode_used"]) != mode:
                        raise RuntimeError("stationary router audit changed mode")
                    if controller == "learned_router":
                        router.observe(
                            observation, action, reward, next_observation, done)
                    decisions.append(int(selected))
                    physics_modes.append(mode)
                    total_return += float(reward)
                    observation = next_observation
                    if done:
                        terminated = True
                        steps = step + 1
                        break
                episode_rows.append({
                    "episode": episode,
                    "return": total_return,
                    "terminated": terminated,
                    "steps": steps,
                })
                if controller == "learned_router":
                    row = _routing_summary(
                        decisions, physics_modes, controller_map)
                    row.update({
                        "episode": episode,
                        "confidence_mean": (
                            float(np.mean(confidence)) if confidence else 0.0),
                        "margin_mean": (
                            float(np.mean(margin)) if margin else 0.0),
                    })
                    route_rows.append(row)
            output[controller][str(mode)] = _stationary_record(
                episode_rows, route_rows)
            if hasattr(env, "close"):
                env.close()
    return output


def evaluate_switching(
    config,
    policy_states: dict[str, Any],
    action_fn,
    model,
    router_config,
    observe,
    controller_map,
    event_seed: int,
):
    horizon = int(config.max_episode_steps)
    output = {}
    for controller in CONTROLLERS:
        run_config = copy.deepcopy(config)
        run_config.stochastic_mode_fixed_id = -1
        env = make_env(run_config, seed_offset=event_seed)
        tasks = env.sample_tasks(4)
        episodes = []
        route_rows = []
        for episode in range(SWITCHING_EPISODES):
            switch_tasks, source_indices = _select_eval_switch_sequence(
                env, tasks, episode)
            _reset_eval_switch_schedule(
                env, switch_tasks, run_config, SWITCHING_PERIOD_STEPS)
            base_key = 20260718 + event_seed * 100_000 + episode * 10_000
            env.rng = jax.random.PRNGKey(base_key)
            observation = env.reset()
            router = RouterRuntime(
                model, router_config, observe, controller_map)
            total_return = 0.0
            termination_count = 0
            first_done_step = horizon
            decisions = []
            physics_modes = []
            confidence = []
            margin = []
            for step in range(horizon):
                physics_mode = int(env.task_id_for_next_step())
                if controller == "robust":
                    selected = -1
                elif controller == "dynamic_control_oracle":
                    selected = int(controller_map[physics_mode])
                else:
                    selected, diagnostics = router.decision()
                    confidence.append(float(diagnostics["confidence"]))
                    margin.append(float(diagnostics["margin"]))
                policy_key = (
                    "robust" if selected < 0
                    else f"fixed_mode_{selected}")
                action = np.asarray(action_fn(
                    policy_states[policy_key],
                    jnp.asarray(observation, dtype=jnp.float32)))
                env.rng = jax.random.PRNGKey(base_key + step * 2 + 1)
                next_observation, reward, done, info = env.step(action)
                if int(info["mode_used"]) != physics_mode:
                    raise RuntimeError("switching action used the wrong mode")
                if controller == "learned_router":
                    router.observe(
                        observation, action, reward, next_observation, done)
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
                "sequence_source_indices": [
                    int(value) for value in source_indices],
            })
            if controller == "learned_router":
                row = _routing_summary(
                    decisions, physics_modes, controller_map)
                row.update({
                    "episode": episode,
                    "confidence_mean": float(np.mean(confidence)),
                    "margin_mean": float(np.mean(margin)),
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
    if (payload.get("schema") != protocol.AUDIT_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("family") != protocol.FAMILY
            or payload.get("env") != protocol.ENV
            or int(payload.get("event_seed", -1))
            not in protocol.HOLDOUT_EVENT_SEEDS
            or set(payload.get("stationary") or {}) != set(CONTROLLERS)
            or set(payload.get("switching") or {}) != set(CONTROLLERS)):
        raise ValueError("invalid learned-router audit identity")
    mapping = tuple(int(value) for value in payload.get("controller_map", []))
    if len(mapping) != 4:
        raise ValueError("learned-router audit has invalid controller map")
    for controller in CONTROLLERS:
        if set(payload["stationary"][controller]) != {"0", "1", "2", "3"}:
            raise ValueError("stationary modes are incomplete")
        for record in payload["stationary"][controller].values():
            values = record.get("returns") or []
            if (len(values) != EPISODES_PER_TASK
                    or not all(math.isfinite(float(value)) for value in values)):
                raise ValueError("invalid stationary return record")
        episodes = payload["switching"][controller].get("episodes") or []
        if (len(episodes) != SWITCHING_EPISODES
                or not all(math.isfinite(float(row["return"]))
                           for row in episodes)):
            raise ValueError("invalid switching return record")
    for section in (
            payload["stationary"]["learned_router"].values()):
        for row in section["routing"]:
            selected = set(int(value) for value in
                           row["selected_controller_counts"])
            if not selected.issubset({-1, 0, 2, 3}):
                raise ValueError("router selected a non-promoted controller")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event-seed", type=int,
        choices=protocol.HOLDOUT_EVENT_SEEDS, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol.configure()
    output = args.out_dir.resolve()
    result_path = output / "group.json"
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        validate_group(payload)
        print(f"Complete learned-router audit exists: {output}")
        return

    manifest = protocol.load_manifest()
    if not manifest.get("validation_gate_pass"):
        raise RuntimeError("router validation gate failed; holdout is sealed")
    controller_map, _ = protocol.control.load_controller_map()
    bundles = protocol.control.specialist_protocol.validate_family_bundles(
        protocol.FAMILY)
    bundle_root = protocol.control.specialist_protocol.family_bundle_root(
        protocol.FAMILY)
    bundle_hashes = {
        name: protocol.sha256_file(
            bundle_root / name
            / protocol.control.specialist_protocol.BUNDLE_MANIFEST)
        for name in bundles
    }
    if bundle_hashes != manifest["bundle_manifest_sha256"]:
        raise RuntimeError("specialist bank changed after router training")

    source_before = protocol.control.fork_protocol.current_source_manifest()
    config, agents, policy_states = specialist_audit._controller_policy_states(
        protocol.FAMILY)
    action_fn = specialist_audit._base_action_fn(
        nnx.graphdef(agents["robust"].policy))
    model, router_config, observe = load_router(
        manifest, agents["robust"].obs_dim, agents["robust"].act_dim)
    stationary = evaluate_stationary(
        config, policy_states, action_fn, model, router_config, observe,
        controller_map, args.event_seed)
    switching = evaluate_switching(
        config, policy_states, action_fn, model, router_config, observe,
        controller_map, args.event_seed)
    source_after = protocol.control.fork_protocol.current_source_manifest()
    if source_after["sha256"] != source_before["sha256"]:
        raise RuntimeError("source changed during learned-router audit")

    payload = {
        "schema": protocol.AUDIT_SCHEMA,
        "status": "complete",
        "family": protocol.FAMILY,
        "env": protocol.ENV,
        "training_seed": 0,
        "event_seed": int(args.event_seed),
        "event_seed_role": (
            "untouched paired holdout sealed before estimator training"),
        "controller_map": list(controller_map),
        "router_config": router_config.to_dict(),
        "router_manifest_file": protocol.file_record(protocol.MANIFEST_PATH),
        "router_parameter_file": protocol.file_record(protocol.MODEL_PATH),
        "bundle_manifest_sha256": bundle_hashes,
        "source_snapshot_sha256": source_before["sha256"],
        "episodes_per_task": EPISODES_PER_TASK,
        "switching_episodes": SWITCHING_EPISODES,
        "switching_period_steps": SWITCHING_PERIOD_STEPS,
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
    print(f"LEARNED CONTROL ROUTER AUDIT COMPLETE: {output}", flush=True)


if __name__ == "__main__":
    main()
