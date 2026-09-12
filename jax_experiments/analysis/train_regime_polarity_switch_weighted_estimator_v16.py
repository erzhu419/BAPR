"""Fine-tune inverse evidence on switch-local v12 policy-bank trajectories."""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_model_v5 as parent_model,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_system_id_v5 as parent_protocol,
)
from jax_experiments.analysis import (
    regime_polarity_switch_weighted_estimator_model_v16 as model_lib,
)
from jax_experiments.analysis import (
    regime_polarity_switch_weighted_estimator_v16 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_frozen_estimator_transfer_audit_v13 as stack_loader,
)
from jax_experiments.train import make_env


def _source_records() -> dict:
    paths = (
        Path(__file__).resolve(),
        Path(protocol.__file__).resolve(),
        Path(model_lib.__file__).resolve(),
        protocol.ROOT / "jax_experiments/networks/executed_action_inverse.py",
        protocol.ROOT / "jax_experiments/envs/brax_env.py",
        protocol.ROOT / "jax_experiments/envs/stochastic_mode_env.py",
        protocol.REGISTRATION_PATH,
    )
    return {
        str(path.relative_to(protocol.ROOT)): protocol.file_record(path)
        for path in paths
    }


def _parent_model_records() -> dict:
    return {
        "manifest": protocol.file_record(parent_protocol.MODEL_MANIFEST),
        "parameters": protocol.file_record(parent_protocol.MODEL_PATH),
    }


def _policy_records() -> dict:
    return {
        str(seed): protocol.policy_bank_records(seed)
        for seed in (
            *protocol.TRAIN_POLICY_SEEDS,
            *protocol.VALIDATION_POLICY_SEEDS,
        )
    }


def _collect_one(
    stack: dict,
    mapping: dict,
    event_seed: int,
    episode: int,
    *,
    steps: int | None = None,
) -> dict:
    event_seed = protocol.require_event_seed(event_seed)
    horizon = protocol.MAX_EPISODE_STEPS if steps is None else int(steps)
    config = copy.deepcopy(stack["config"])
    config.stochastic_mode_fixed_id = -1
    env = make_env(config, seed_offset=event_seed - int(config.seed))
    tasks = env.sample_tasks(len(protocol.MODES))
    configure = getattr(env, "configure_eval_mode_sequence", None)
    if not callable(configure):
        raise RuntimeError("v16 training requires explicit mode schedules")
    sequence = protocol.switching_sequence(event_seed, episode)
    configure(tasks, sequence, protocol.DWELL_STEPS)
    estimator = stack["estimator_factory"]()
    gains = np.asarray(estimator.gain_vectors, dtype=np.float32)
    rows = {key: [] for key in (
        "obs", "act", "next_obs", "target_action", "mode_id",
        "sample_weight", "switch_age",
    )}
    observation = env.reset()
    estimator_state = estimator.initial_state()
    previous_mode = None
    switch_age = 0
    try:
        for _ in range(horizon):
            mode = int(env.task_id_for_next_step())
            if previous_mode is None or mode != previous_mode:
                switch_age = 0
            else:
                switch_age += 1
            previous_mode = mode
            posterior = np.asarray(
                estimator.probabilities(estimator_state), dtype=np.float64)
            route_mode = int(np.argmax(posterior))
            controller = str(mapping[str(route_mode)]["controller"])
            action = np.asarray(
                stack["actions"][controller](observation), dtype=np.float32)
            next_observation, reward, done, info = env.step(action)
            if int(info["mode_used"]) != mode:
                raise RuntimeError("v16 collection used a misaligned mode")
            estimator_state, _, _, _ = estimator.step(
                estimator_state,
                observation,
                action,
                reward,
                next_observation,
            )
            rows["obs"].append(np.asarray(observation, dtype=np.float32))
            rows["act"].append(action)
            rows["next_obs"].append(
                np.asarray(next_observation, dtype=np.float32))
            rows["target_action"].append(np.clip(
                action * gains[mode], -1.0, 1.0).astype(np.float32))
            rows["mode_id"].append(mode)
            rows["sample_weight"].append(
                protocol.SWITCH_SAMPLE_WEIGHT
                if switch_age < protocol.SWITCH_WINDOW_STEPS else 1.0)
            rows["switch_age"].append(switch_age)
            observation = next_observation
            if done:
                observation = env.reset()
    finally:
        if hasattr(env, "close"):
            env.close()
    return {
        "obs": np.asarray(rows["obs"], dtype=np.float32),
        "act": np.asarray(rows["act"], dtype=np.float32),
        "next_obs": np.asarray(rows["next_obs"], dtype=np.float32),
        "target_action": np.asarray(rows["target_action"], dtype=np.float32),
        "mode_id": np.asarray(rows["mode_id"], dtype=np.int32),
        "sample_weight": np.asarray(rows["sample_weight"], dtype=np.float32),
        "switch_age": np.asarray(rows["switch_age"], dtype=np.int32),
        "identity": {
            "policy_seed": int(config.seed),
            "event_seed": int(event_seed),
            "episode": int(episode),
            "base_schedule": list(protocol.SWITCHING_SCHEDULES[event_seed]),
        },
    }


def _collect(policy_seeds, event_seeds) -> list[dict]:
    sequences = []
    for seed in policy_seeds:
        stack, mapping = stack_loader._load_stack(seed)
        for event_seed in event_seeds:
            for episode in range(protocol.TRAINING_EPISODES):
                sequences.append(_collect_one(
                    stack, mapping, event_seed, episode))
            print(
                f"v16 data seed={seed} event={event_seed} complete",
                flush=True,
            )
    return sequences


def _flatten(sequences: list[dict]) -> dict[str, np.ndarray]:
    return {
        key: np.concatenate([row[key] for row in sequences], axis=0)
        for key in (
            "obs", "act", "next_obs", "target_action", "mode_id",
            "sample_weight", "switch_age",
        )
    }


def _build_update(model, gains, residual_variance):
    graphdef = nnx.graphdef(model)
    optimizer = optax.chain(
        optax.clip_by_global_norm(10.0),
        optax.adamw(
            protocol.MODEL_CONFIG["learning_rate"],
            weight_decay=protocol.MODEL_CONFIG["weight_decay"],
        ),
    )
    optimizer_state = optimizer.init(nnx.state(model, nnx.Param))
    gains = jnp.asarray(gains, dtype=jnp.float32)
    variance = jnp.clip(
        jnp.asarray(residual_variance, dtype=jnp.float32), 1e-6, 1.0)

    @jax.jit
    def update(
        params,
        opt_state,
        obs,
        next_obs,
        commanded_action,
        target_action,
        mode_id,
        sample_weight,
    ):
        def loss_fn(candidate_params):
            current = nnx.merge(graphdef, candidate_params)
            predicted = current.predict_head_batches(obs, next_obs)
            squared = jnp.mean(jnp.square(predicted - target_action), axis=-1)
            candidate_actions = jnp.clip(
                commanded_action[:, :, None, :] * gains[None, None, :, :],
                -1.0,
                1.0,
            )
            error = candidate_actions - predicted[:, :, None, :]
            logits = -0.5 * jnp.mean(
                jnp.square(error) / variance[None, None, None, :],
                axis=-1,
            )
            classification = optax.softmax_cross_entropy_with_integer_labels(
                logits, mode_id)
            normalizer = jnp.sum(sample_weight)
            mse = jnp.sum(sample_weight * squared) / normalizer
            cross_entropy = (
                jnp.sum(sample_weight * classification) / normalizer)
            loss = mse + (
                protocol.CLASSIFICATION_LOSS_WEIGHT * cross_entropy)
            accuracy = jnp.sum(
                sample_weight * (jnp.argmax(logits, axis=-1) == mode_id)
            ) / normalizer
            return loss, (
                jnp.sqrt(mse), cross_entropy, accuracy,
                jnp.mean(jnp.var(predicted, axis=0)),
            )

        (loss, metrics), gradients = jax.value_and_grad(
            loss_fn, has_aux=True)(params)
        updates, next_optimizer_state = optimizer.update(
            gradients, opt_state, params)
        next_params = optax.apply_updates(params, updates)
        next_params = jax.tree.map(
            lambda value: jnp.nan_to_num(value), next_params)
        return next_params, next_optimizer_state, (loss,) + metrics

    return update, optimizer_state


def _sample_batch(dataset, rng):
    indices = rng.integers(
        0,
        len(dataset["obs"]),
        size=(protocol.MODEL_CONFIG["ensemble_size"], protocol.BATCH_SIZE),
    )
    return tuple(jnp.asarray(dataset[key][indices]) for key in (
        "obs", "next_obs", "act", "target_action", "mode_id",
        "sample_weight",
    ))


def _prediction_rows(prediction_fn, params, sequences):
    return [
        (
            sequence,
            np.asarray(prediction_fn(
                params,
                jnp.asarray(sequence["obs"]),
                jnp.asarray(sequence["next_obs"]),
            )),
        )
        for sequence in sequences
    ]


def _residual_variance(prediction_rows):
    residuals = [
        np.square(np.mean(predicted, axis=0) - sequence["target_action"])
        for sequence, predicted in prediction_rows
    ]
    variance = np.mean(np.concatenate(residuals, axis=0), axis=0)
    return np.clip(
        variance,
        protocol.MODEL_CONFIG["variance_floor"],
        protocol.MODEL_CONFIG["variance_ceiling"],
    ).astype(np.float32)


def _inverse_metrics(prediction_rows):
    squared = []
    switch_squared = []
    stable_squared = []
    for sequence, predicted in prediction_rows:
        error = np.square(
            np.mean(predicted, axis=0) - sequence["target_action"])
        switch = sequence["switch_age"] < protocol.SWITCH_WINDOW_STEPS
        squared.append(error)
        switch_squared.append(error[switch])
        stable_squared.append(error[~switch])
    return {
        "rmse": float(np.sqrt(np.mean(np.concatenate(squared, axis=0)))),
        "switch_window_rmse": float(np.sqrt(np.mean(
            np.concatenate(switch_squared, axis=0)))),
        "stable_rmse": float(np.sqrt(np.mean(
            np.concatenate(stable_squared, axis=0)))),
    }


def _causal_evidence_metrics(
    evidence_fn,
    params,
    residual_variance,
    sequences,
) -> dict:
    gains = jnp.asarray(protocol.mode_gain_vectors(6), dtype=jnp.float32)
    config = protocol.FilterConfig.from_dict(protocol.FILTER_CONFIG)
    correct = []
    switch_correct = []
    stable_correct = []
    wrong_run_lengths = []
    for sequence in sequences:
        evidence, _, _ = evidence_fn(
            params,
            jnp.asarray(sequence["obs"]),
            jnp.asarray(sequence["next_obs"]),
            jnp.asarray(sequence["act"]),
            gains,
            jnp.asarray(residual_variance, dtype=jnp.float32),
        )
        state = np.full((len(protocol.MODES),), 1.0 / len(protocol.MODES))
        predicted = []
        for row in np.asarray(evidence):
            predicted.append(int(np.argmax(state)))
            state = protocol.posterior_update(state, row, config)
        predicted = np.asarray(predicted, dtype=np.int32)
        labels = sequence["mode_id"]
        match = predicted == labels
        switch = sequence["switch_age"] < protocol.SWITCH_WINDOW_STEPS
        correct.append(match)
        switch_correct.append(match[switch])
        stable_correct.append(match[~switch])
        run = 0
        for value in match:
            if value:
                if run:
                    wrong_run_lengths.append(run)
                run = 0
            else:
                run += 1
        if run:
            wrong_run_lengths.append(run)
    return {
        "action_time_mode_accuracy": float(np.mean(
            np.concatenate(correct))),
        "switch_window_mode_accuracy": float(np.mean(
            np.concatenate(switch_correct))),
        "stable_mode_accuracy": float(np.mean(
            np.concatenate(stable_correct))),
        "maximum_wrong_run": int(max(wrong_run_lengths, default=0)),
        "mean_wrong_run": float(np.mean(wrong_run_lengths))
        if wrong_run_lengths else 0.0,
    }


def _write_scheduler_checkpoint(update_index: int) -> None:
    payload = {
        "iteration": max(int(update_index) - 1, 0),
        "next_iteration": int(update_index),
        "total_steps": int(update_index),
        "algo": "switch_weighted_expected_action_estimator_v16",
    }
    protocol.CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=".train_state.", suffix=".pkl", dir=protocol.CHECKPOINT_ROOT)
    try:
        with os.fdopen(handle, "wb") as output:
            pickle.dump(payload, output, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_name, protocol.TRAIN_STATE_PKL)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _checkpoint_payload(update_index, model, optimizer_state, rng, history):
    metadata = protocol.save_parameter_state(
        protocol.TRAIN_STATE_NPZ,
        (nnx.state(model, nnx.Param), optimizer_state),
    )
    protocol.write_json_atomic(protocol.TRAIN_STATE_JSON, {
        "schema": protocol.TRAIN_STATE_SCHEMA,
        "status": "in_progress",
        "update_index": int(update_index),
        "parameter_file": protocol.file_record(protocol.TRAIN_STATE_NPZ),
        "parameter_leaves": metadata,
        "rng_state": rng.bit_generator.state,
        "model_config": protocol.MODEL_CONFIG,
        "source_files": _source_records(),
        "parent_model": _parent_model_records(),
        "policy_records": _policy_records(),
        "history": history,
    })
    _write_scheduler_checkpoint(update_index)


def _restore_checkpoint(model, optimizer_state):
    if not protocol.TRAIN_STATE_JSON.is_file():
        return None
    payload = protocol.read_json(protocol.TRAIN_STATE_JSON)
    if (
        payload.get("schema") != protocol.TRAIN_STATE_SCHEMA
        or payload.get("status") != "in_progress"
        or payload.get("model_config") != protocol.MODEL_CONFIG
        or payload.get("source_files") != _source_records()
        or payload.get("parent_model") != _parent_model_records()
        or payload.get("policy_records") != _policy_records()
        or payload.get("parameter_file")
        != protocol.file_record(protocol.TRAIN_STATE_NPZ)
    ):
        raise ValueError("stale v16 estimator train state")
    state = protocol.load_parameter_state(
        protocol.TRAIN_STATE_NPZ,
        (nnx.state(model, nnx.Param), optimizer_state),
        payload["parameter_leaves"],
    )
    return payload, state


def _initialized_models():
    parent, _, gains, variance, parent_manifest = parent_model.load_model(17, 6)
    model = model_lib.make_model(17, 6, protocol.MODEL_SEED)
    nnx.update(model, nnx.state(parent, nnx.Param))
    return parent, model, gains, variance, parent_manifest


def smoke() -> None:
    stack, mapping = stack_loader._load_stack(protocol.TRAIN_POLICY_SEEDS[0])
    sequence = _collect_one(
        stack,
        mapping,
        protocol.TRAIN_EVENT_SEEDS[0],
        0,
        steps=16,
    )
    dataset = _flatten([sequence])
    _, model, gains, variance, _ = _initialized_models()
    update, optimizer_state = _build_update(model, gains, variance)
    batch = _sample_batch(dataset, np.random.default_rng(protocol.MODEL_SEED))
    params, _, metrics = update(
        nnx.state(model, nnx.Param), optimizer_state, *batch)
    if not all(
        np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree.leaves(params)
    ):
        raise ValueError("v16 smoke produced nonfinite parameters")
    if not all(np.isfinite(float(value)) for value in metrics):
        raise ValueError("v16 smoke produced nonfinite metrics")
    print(
        "V16 SWITCH-WEIGHTED ESTIMATOR SMOKE COMPLETE: "
        f"loss={float(metrics[0]):.6f} rmse={float(metrics[1]):.6f} "
        f"ce={float(metrics[2]):.6f} accuracy={float(metrics[3]):.6f}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=protocol.TRAIN_UPDATES)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if min(args.updates, args.checkpoint_interval) <= 0:
        parser.error("training controls must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.smoke:
        smoke()
        return
    protocol.validate_registration()
    if protocol.MODEL_MANIFEST.is_file():
        model_lib.load_model(17, 6)
        print(f"V16 ESTIMATOR MODEL ALREADY COMPLETE: {protocol.MODEL_ROOT}")
        return
    if protocol.MODEL_ROOT.exists() and not args.resume:
        raise RuntimeError(
            f"partial v16 estimator root requires --resume: {protocol.MODEL_ROOT}")
    protocol.MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    parent, model, gains, parent_variance, parent_manifest = _initialized_models()
    update, optimizer_state = _build_update(model, gains, parent_variance)
    rng = np.random.default_rng(protocol.MODEL_SEED)
    history = []
    start_update = 0
    restored = _restore_checkpoint(
        model, optimizer_state) if args.resume else None
    if restored is not None:
        payload, (params, optimizer_state) = restored
        nnx.update(model, params)
        start_update = int(payload["update_index"])
        history = list(payload.get("history") or [])
        rng.bit_generator.state = payload["rng_state"]
        print(f"Resumed v16 estimator at update {start_update}", flush=True)
    else:
        _checkpoint_payload(0, model, optimizer_state, rng, history)

    train_sequences = _collect(
        protocol.TRAIN_POLICY_SEEDS, protocol.TRAIN_EVENT_SEEDS)
    validation_sequences = _collect(
        protocol.VALIDATION_POLICY_SEEDS, protocol.VALIDATION_EVENT_SEEDS)
    train_dataset = _flatten(train_sequences)

    for update_index in range(start_update, args.updates):
        batch = _sample_batch(train_dataset, rng)
        params, optimizer_state, metrics = update(
            nnx.state(model, nnx.Param), optimizer_state, *batch)
        nnx.update(model, params)
        completed = update_index + 1
        if completed == 1 or completed % 50 == 0:
            row = {
                "update": int(completed),
                "loss": float(metrics[0]),
                "weighted_rmse": float(metrics[1]),
                "weighted_cross_entropy": float(metrics[2]),
                "weighted_mode_accuracy": float(metrics[3]),
                "ensemble_variance": float(metrics[4]),
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        if completed % args.checkpoint_interval == 0:
            _checkpoint_payload(
                completed, model, optimizer_state, rng, history)

    params = nnx.state(model, nnx.Param)
    prediction_fn = model_lib.build_prediction_fn(model)
    train_predictions = _prediction_rows(
        prediction_fn, params, train_sequences)
    validation_predictions = _prediction_rows(
        prediction_fn, params, validation_sequences)
    residual_variance = _residual_variance(train_predictions)
    evidence_fn = model_lib.build_evidence_fn(model)

    parent_params = nnx.state(parent, nnx.Param)
    parent_evidence_fn = parent_model.build_evidence_fn(parent)
    parent_metrics = _causal_evidence_metrics(
        parent_evidence_fn,
        parent_params,
        np.asarray(parent_variance),
        validation_sequences,
    )
    validation_metrics = _causal_evidence_metrics(
        evidence_fn,
        params,
        residual_variance,
        validation_sequences,
    )
    parameter_leaves = protocol.save_parameter_state(
        protocol.MODEL_PATH, params)
    manifest = {
        "schema": protocol.MODEL_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "env": protocol.ENV,
        "family": protocol.FAMILY,
        "model_seed": protocol.MODEL_SEED,
        "model_config": protocol.MODEL_CONFIG,
        "filter_config": protocol.FILTER_CONFIG,
        "train_policy_seeds": list(protocol.TRAIN_POLICY_SEEDS),
        "validation_policy_seeds": list(protocol.VALIDATION_POLICY_SEEDS),
        "train_event_seeds": list(protocol.TRAIN_EVENT_SEEDS),
        "validation_event_seeds": list(protocol.VALIDATION_EVENT_SEEDS),
        "training_config": {
            "episodes_per_event": protocol.TRAINING_EPISODES,
            "updates": int(args.updates),
            "batch_size_per_head": protocol.BATCH_SIZE,
            "switch_window_steps": protocol.SWITCH_WINDOW_STEPS,
            "switch_sample_weight": protocol.SWITCH_SAMPLE_WEIGHT,
            "classification_loss_weight": (
                protocol.CLASSIFICATION_LOSS_WEIGHT),
        },
        "training_data_policy": (
            "frozen-v5 causal MAP trajectories; true mode used only offline "
            "for executed-action targets and mode-evidence labels"
        ),
        "mode_gain_vectors": np.asarray(gains).tolist(),
        "residual_variance": residual_variance.tolist(),
        "training_inverse_metrics": _inverse_metrics(train_predictions),
        "validation_inverse_metrics": _inverse_metrics(
            validation_predictions),
        "parent_validation_evidence_metrics": parent_metrics,
        "validation_evidence_metrics": validation_metrics,
        "training_history": history,
        "registration": protocol.file_record(protocol.REGISTRATION_PATH),
        "initialization": _parent_model_records(),
        "policy_records": _policy_records(),
        "source_files": _source_records(),
        "parameter_file": protocol.file_record(protocol.MODEL_PATH),
        "parameter_leaves": parameter_leaves,
        "parent_model_manifest": parent_manifest,
    }
    protocol.write_json_atomic(protocol.MODEL_MANIFEST, manifest)
    model_lib.load_model(17, 6)
    if protocol.CHECKPOINT_ROOT.exists():
        shutil.rmtree(protocol.CHECKPOINT_ROOT)
    print(
        "V16 SWITCH-WEIGHTED ESTIMATOR MODEL COMPLETE: "
        f"switch_acc={validation_metrics['switch_window_mode_accuracy']:.6f} "
        f"parent_switch_acc="
        f"{parent_metrics['switch_window_mode_accuracy']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
