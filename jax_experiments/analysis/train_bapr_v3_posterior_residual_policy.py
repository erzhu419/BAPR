"""Train one switch-matched nonlinear posterior residual policy."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from jax_experiments.analysis import (
    bapr_v3_learned_control_router as estimator,
)
from jax_experiments.analysis import (
    bapr_v3_posterior_residual_policy as protocol,
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
    paths = (Path(__file__).resolve(), Path(protocol.__file__).resolve())
    return {
        str(path.relative_to(protocol.ROOT)): protocol.file_record(path)
        for path in paths
    }


def _policy_action(action_fn, policy_states, key: str, observation):
    return np.asarray(action_fn(
        policy_states[key],
        jnp.asarray(observation, dtype=jnp.float32)), dtype=np.float32)


def _empty_rows():
    return {key: [] for key in (
        "obs", "posterior", "robust_action", "teacher_action", "mode")}


def _finalize_rows(rows: dict[str, list[Any]]) -> dict[str, np.ndarray]:
    payload = {
        "obs": np.asarray(rows["obs"], dtype=np.float32),
        "posterior": np.asarray(rows["posterior"], dtype=np.float32),
        "robust_action": np.asarray(rows["robust_action"], dtype=np.float32),
        "teacher_action": np.asarray(rows["teacher_action"], dtype=np.float32),
        "mode": np.asarray(rows["mode"], dtype=np.int32),
    }
    size = len(payload["obs"])
    if size <= 0 or any(len(value) != size for value in payload.values()):
        raise ValueError("invalid nonlinear residual dataset")
    return payload


def _append_rows(left: dict[str, np.ndarray], right: dict[str, np.ndarray]):
    return {key: np.concatenate([left[key], right[key]], axis=0)
            for key in left}


def make_model_action(
    model,
    params,
    obs_mean,
    obs_std,
    utility_matrix,
    robust_index,
    specialist_indices,
    advantage_scale,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    graphdef = nnx.graphdef(model)
    mean = jnp.asarray(obs_mean)
    std = jnp.asarray(obs_std)
    matrix = jnp.asarray(utility_matrix)

    @jax.jit
    def action(observation, posterior, robust_action):
        current = nnx.merge(graphdef, params)
        strength = protocol.posterior_strength(
            posterior, matrix, robust_index,
            specialist_indices, advantage_scale)
        return current(
            observation, posterior, robust_action, strength, mean, std)

    def wrapped(observation, posterior, robust_action):
        return np.asarray(action(
            jnp.asarray(observation, dtype=jnp.float32),
            jnp.asarray(posterior, dtype=jnp.float32),
            jnp.asarray(robust_action, dtype=jnp.float32)), dtype=np.float32)

    return wrapped


def collect_switch_matched_dataset(
    config,
    policy_states,
    action_fn,
    router_model,
    observe,
    table,
    router_config,
    oracle_map,
    event_seeds,
    behaviors,
    model_action=None,
):
    rows = _empty_rows()
    horizon = int(config.max_episode_steps)
    expected_horizon = (
        len(protocol.FULL_CYCLE_SEQUENCES[0])
        * protocol.FULL_CYCLE_DWELL_STEPS)
    if horizon != expected_horizon:
        raise ValueError(
            f"switch-matched residual training requires horizon "
            f"{expected_horizon}, got {horizon}")
    for event_seed in event_seeds:
        for behavior in behaviors:
            run_config = copy.deepcopy(config)
            run_config.stochastic_mode_fixed_id = -1
            env = make_env(run_config, seed_offset=int(event_seed))
            tasks = env.sample_tasks(4)
            for episode, sequence in enumerate(protocol.FULL_CYCLE_SEQUENCES):
                env.configure_eval_mode_sequence(
                    tasks, sequence, protocol.FULL_CYCLE_DWELL_STEPS)
                base_key = (
                    20260719 + int(event_seed) * 100_000
                    + episode * 10_000)
                env.rng = jax.random.PRNGKey(base_key)
                observation = env.reset()
                router = base_audit.UtilityRouterRuntime(
                    router_model, observe, table, router_config)
                for step in range(horizon):
                    physics_mode = int(env.task_id_for_next_step())
                    posterior = np.asarray(router.state[0], dtype=np.float32)
                    robust_action = _policy_action(
                        action_fn, policy_states, "robust", observation)
                    teacher_controller = int(oracle_map[physics_mode])
                    teacher_action = _policy_action(
                        action_fn, policy_states,
                        base_audit._policy_key(teacher_controller), observation)
                    if behavior == "robust":
                        action = robust_action
                    elif behavior == "dynamic_utility_oracle":
                        action = teacher_action
                    elif behavior == "learned_utility_router":
                        selected, _ = router.decision()
                        action = _policy_action(
                            action_fn, policy_states,
                            base_audit._policy_key(selected), observation)
                    elif behavior == "posterior_residual_policy":
                        if model_action is None:
                            raise ValueError("DAgger behavior requires model action")
                        action = model_action(
                            observation, posterior, robust_action)
                    else:
                        raise ValueError(f"unknown collection behavior {behavior}")
                    rows["obs"].append(np.asarray(observation))
                    rows["posterior"].append(posterior)
                    rows["robust_action"].append(robust_action)
                    rows["teacher_action"].append(teacher_action)
                    rows["mode"].append(physics_mode)
                    env.rng = jax.random.PRNGKey(base_key + step * 2 + 1)
                    next_observation, reward, done, info = env.step(action)
                    if int(info["mode_used"]) != physics_mode:
                        raise RuntimeError("residual training action used wrong mode")
                    router.observe(
                        observation, action, reward, next_observation, done)
                    observation = next_observation
                    if done:
                        env.rng = jax.random.PRNGKey(
                            base_key + step * 2 + 2)
                        observation = env.reset()
            if hasattr(env, "close"):
                env.close()
            print(
                f"  collected event_seed={event_seed} behavior={behavior} "
                f"rows={len(rows['obs'])}", flush=True)
    return _finalize_rows(rows)


def build_optimizer(model, utility_values):
    graphdef = nnx.graphdef(model)
    optimizer = optax.chain(
        optax.clip_by_global_norm(10.0),
        optax.adam(float(protocol.MODEL_CONFIG["learning_rate"])),
    )
    utility_matrix, robust_index, specialist_indices, advantage_scale = (
        utility_values)
    matrix = jnp.asarray(utility_matrix)
    robust_index = int(robust_index)
    specialist_indices = tuple(specialist_indices)
    advantage_scale = float(advantage_scale)
    teacher_weight = float(protocol.MODEL_CONFIG["teacher_loss_weight"])
    anchor_weight = float(protocol.MODEL_CONFIG["entropy_anchor_weight"])

    def loss_fn(params, batch, obs_mean, obs_std):
        current = nnx.merge(graphdef, params)
        strength = protocol.posterior_strength(
            batch["posterior"], matrix, robust_index,
            specialist_indices, advantage_scale)
        predicted = current(
            batch["obs"], batch["posterior"], batch["robust_action"],
            strength, obs_mean, obs_std)
        teacher_loss = jnp.mean(jnp.square(
            predicted - batch["teacher_action"]))
        entropy = protocol.normalized_entropy(batch["posterior"])
        anchor_loss = jnp.mean(
            entropy[..., None]
            * jnp.square(predicted - batch["robust_action"]))
        total = teacher_weight * teacher_loss + anchor_weight * anchor_loss
        return total, (teacher_loss, anchor_loss,
                       jnp.mean(strength), jnp.mean(jnp.abs(
                           predicted - batch["robust_action"])))

    @jax.jit
    def update(params, opt_state, batch, obs_mean, obs_std):
        (loss, metrics), gradients = jax.value_and_grad(
            loss_fn, has_aux=True)(params, batch, obs_mean, obs_std)
        updates, next_opt_state = optimizer.update(
            gradients, opt_state, params)
        next_params = optax.apply_updates(params, updates)
        return next_params, next_opt_state, (loss,) + metrics

    @jax.jit
    def evaluate(params, batch, obs_mean, obs_std):
        return loss_fn(params, batch, obs_mean, obs_std)

    return optimizer, update, evaluate


def _jax_dataset(dataset):
    return {key: jnp.asarray(value) for key, value in dataset.items()
            if key != "mode"}


def evaluate_model(evaluate, params, dataset, obs_mean, obs_std):
    (loss, metrics) = evaluate(
        params, _jax_dataset(dataset),
        jnp.asarray(obs_mean), jnp.asarray(obs_std))
    values = np.asarray((loss,) + metrics, dtype=np.float64)
    names = (
        "total_loss", "teacher_loss", "entropy_anchor_loss",
        "strength_mean", "action_delta_abs_mean")
    result = {name: float(value) for name, value in zip(names, values)}
    protocol.validate_finite_metrics(result)
    return result


def train_model(model, dataset, validation, obs_mean, obs_std,
                utility_values, seed: int, updates_count: int,
                initial_params=None):
    optimizer, update, evaluate = build_optimizer(model, utility_values)
    params = (
        nnx.state(model, nnx.Param)
        if initial_params is None else initial_params)
    opt_state = optimizer.init(params)
    rng = np.random.default_rng(int(seed) + 20260719)
    batch_size = int(protocol.MODEL_CONFIG["batch_size"])
    size = len(dataset["obs"])
    for step in range(int(updates_count)):
        indices = rng.integers(0, size, size=batch_size)
        batch = {
            key: jnp.asarray(value[indices])
            for key, value in dataset.items() if key != "mode"
        }
        params, opt_state, metrics = update(
            params, opt_state, batch,
            jnp.asarray(obs_mean), jnp.asarray(obs_std))
        if (step + 1) % 1000 == 0:
            values = np.asarray(metrics, dtype=np.float64)
            print(
                f"  train seed={seed} step={step + 1}/{updates_count} "
                f"loss={values[0]:.6f} teacher={values[1]:.6f} "
                f"anchor={values[2]:.6f}", flush=True)
    return params, evaluate_model(
        evaluate, params, validation, obs_mean, obs_std)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    del args
    protocol.configure()
    if protocol.MANIFEST_PATH.is_file():
        protocol.load_manifest()
        print(f"Complete nonlinear residual policy exists: {protocol.MODEL_ROOT}")
        return

    table = protocol.screen.utility.load_utility_table()
    utility_values = protocol.utility_constants(table)
    estimator_manifest = estimator.load_manifest()
    oracle_map = tuple(int(value) for value in table["oracle_controller_map"])
    bundles = (
        protocol.screen.utility.control.specialist_protocol
        .validate_family_bundles(protocol.FAMILY))
    bundle_root = (
        protocol.screen.utility.control.specialist_protocol
        .family_bundle_root(protocol.FAMILY))
    bundle_hashes = {
        name: estimator.sha256_file(
            bundle_root / name
            / protocol.screen.utility.control.specialist_protocol
            .BUNDLE_MANIFEST)
        for name in bundles
    }
    if bundle_hashes != table["bundle_manifest_sha256"]:
        raise RuntimeError("controller bank changed after utility freeze")

    source_before = (
        protocol.screen.utility.control.fork_protocol.current_source_manifest())
    config, agents, policy_states = specialist_audit._controller_policy_states(
        protocol.FAMILY)
    action_fn = specialist_audit._base_action_fn(
        nnx.graphdef(agents["robust"].policy))
    base_router_config = estimator.RouterConfig.from_dict(
        estimator_manifest["router_config"])
    router_config = protocol.screen.utility.decision_config(
        base_router_config, protocol.DECISION_VARIANT)
    router_model, router_config, observe = estimator_audit.load_router(
        estimator_manifest, agents["robust"].obs_dim,
        agents["robust"].act_dim,
        router_config_override=router_config)

    train_data = collect_switch_matched_dataset(
        config, policy_states, action_fn, router_model, observe, table,
        router_config, oracle_map, protocol.TRAIN_EVENT_SEEDS,
        protocol.BEHAVIOR_CONTROLLERS)
    validation_data = collect_switch_matched_dataset(
        config, policy_states, action_fn, router_model, observe, table,
        router_config, oracle_map, protocol.MODEL_SELECTION_EVENT_SEEDS,
        protocol.BEHAVIOR_CONTROLLERS)
    obs_mean = np.mean(train_data["obs"], axis=0).astype(np.float32)
    obs_std = np.maximum(
        np.std(train_data["obs"], axis=0), 1e-4).astype(np.float32)

    candidates = {}
    candidate_params = {}
    for seed in protocol.MODEL_CONFIG["initialization_seeds"]:
        model = protocol.make_policy(
            agents["robust"].obs_dim, agents["robust"].act_dim, seed)
        params, metrics = train_model(
            model, train_data, validation_data, obs_mean, obs_std,
            utility_values, int(seed),
            int(protocol.MODEL_CONFIG["initial_updates"]))
        candidates[str(seed)] = metrics
        candidate_params[int(seed)] = params
    selected_seed = min(
        candidate_params,
        key=lambda seed: candidates[str(seed)]["total_loss"])
    selected_model = protocol.make_policy(
        agents["robust"].obs_dim, agents["robust"].act_dim, selected_seed)
    initial_params = candidate_params[selected_seed]
    model_action = make_model_action(
        selected_model, initial_params, obs_mean, obs_std, *utility_values)
    dagger_data = collect_switch_matched_dataset(
        config, policy_states, action_fn, router_model, observe, table,
        router_config, oracle_map, protocol.TRAIN_EVENT_SEEDS,
        ("posterior_residual_policy",), model_action=model_action)
    combined = _append_rows(train_data, dagger_data)
    dagger_params, dagger_metrics = train_model(
        selected_model, combined, validation_data, obs_mean, obs_std,
        utility_values, selected_seed + 10_000,
        int(protocol.MODEL_CONFIG["dagger_updates"]),
        initial_params=initial_params)
    initial_metrics = candidates[str(selected_seed)]
    if dagger_metrics["total_loss"] <= initial_metrics["total_loss"]:
        final_params = dagger_params
        selected_stage = "dagger1"
        selected_metrics = dagger_metrics
    else:
        final_params = initial_params
        selected_stage = "initial"
        selected_metrics = initial_metrics

    source_after = (
        protocol.screen.utility.control.fork_protocol.current_source_manifest())
    if source_after["sha256"] != source_before["sha256"]:
        raise RuntimeError("source changed during nonlinear residual training")
    protocol.MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    metadata = estimator.save_parameter_state(
        protocol.MODEL_PATH, final_params)
    protocol.save_normalization(
        protocol.NORMALIZATION_PATH, obs_mean, obs_std)
    payload = {
        "schema": protocol.MODEL_SCHEMA,
        "status": "complete",
        "family": protocol.FAMILY,
        "env": protocol.ENV,
        "training_seed": 0,
        "decision_variant": protocol.DECISION_VARIANT,
        "train_event_seeds": list(protocol.TRAIN_EVENT_SEEDS),
        "model_selection_event_seeds": list(
            protocol.MODEL_SELECTION_EVENT_SEEDS),
        "development_return_event_seeds": list(
            protocol.DEVELOPMENT_RETURN_EVENT_SEEDS),
        "sealed_confirmation_event_seeds": list(
            protocol.SEALED_CONFIRMATION_EVENT_SEEDS),
        "behavior_controllers": list(protocol.BEHAVIOR_CONTROLLERS),
        "full_cycle_sequences": [
            list(row) for row in protocol.FULL_CYCLE_SEQUENCES],
        "full_cycle_dwell_steps": protocol.FULL_CYCLE_DWELL_STEPS,
        "model_config": protocol.MODEL_CONFIG,
        "obs_dim": int(agents["robust"].obs_dim),
        "act_dim": int(agents["robust"].act_dim),
        "train_rows": int(len(train_data["obs"])),
        "model_selection_rows": int(len(validation_data["obs"])),
        "dagger_rows": int(len(dagger_data["obs"])),
        "initial_candidates": candidates,
        "selected_initialization_seed": int(selected_seed),
        "dagger_validation_metrics": dagger_metrics,
        "selected_stage": selected_stage,
        "selected_validation_metrics": selected_metrics,
        "parameter_metadata": metadata,
        "parameter_file": protocol.file_record(protocol.MODEL_PATH),
        "normalization_file": protocol.file_record(
            protocol.NORMALIZATION_PATH),
        "utility_table_file": protocol.file_record(
            protocol.screen.utility.TABLE_PATH),
        "estimator_manifest_file": protocol.file_record(
            estimator.MANIFEST_PATH),
        "estimator_parameter_file": protocol.file_record(
            estimator.MODEL_PATH),
        "bundle_manifest_sha256": bundle_hashes,
        "router_config": router_config.to_dict(),
        "utility_advantage_scale": float(utility_values[-1]),
        "source_snapshot_sha256": source_before["sha256"],
        "source_files": source_records(),
    }
    protocol.write_json_atomic(protocol.MANIFEST_PATH, payload)
    protocol.load_manifest()
    print(
        "NONLINEAR POSTERIOR RESIDUAL TRAINING COMPLETE: "
        f"seed={selected_seed}, stage={selected_stage}, "
        f"validation_loss={selected_metrics['total_loss']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
