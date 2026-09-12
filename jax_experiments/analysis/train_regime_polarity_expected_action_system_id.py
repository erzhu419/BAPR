"""Train inverse system ID without realized executed-action labels."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from jax_experiments.analysis import regime_polarity_headroom as exploratory
from jax_experiments.analysis import (
    regime_polarity_expected_action_system_id as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_expected_action_system_id_model as model_lib,
)
from jax_experiments.analysis import train_regime_polarity_posterior as common
from jax_experiments.analysis import (
    train_regime_polarity_inverse_system_id as v3_trainer,
)
from jax_experiments.train import (
    _reset_eval_switch_schedule,
    _select_eval_switch_sequence,
    make_env,
)


MODEL_SEED = 20_260_732
CHECKPOINT_ROOT = protocol.MODEL_ROOT / "checkpoints"
TRAIN_STATE_JSON = CHECKPOINT_ROOT / "train_state.json"
TRAIN_STATE_NPZ = CHECKPOINT_ROOT / "train_state.npz"


def _source_records():
    paths = (
        Path(__file__).resolve(),
        Path(protocol.__file__).resolve(),
        Path(model_lib.__file__).resolve(),
        Path(v3_trainer.__file__).resolve(),
        protocol.ROOT
        / "jax_experiments/networks/executed_action_inverse.py",
        protocol.ROOT / "jax_experiments/envs/brax_env.py",
        protocol.ROOT / "jax_experiments/envs/stochastic_mode_env.py",
    )
    return {
        str(path.relative_to(protocol.ROOT)): protocol.file_record(path)
        for path in paths
    }


def _sequence(transitions, labels, identity):
    obs, act, rew, next_obs, done = (
        np.asarray(value) for value in transitions)
    labels = np.asarray(labels, dtype=np.int32)
    if labels.shape != (len(obs),):
        raise ValueError("transition labels have the wrong length")
    gains = protocol.mode_gain_vectors(act.shape[-1])
    target_action = np.clip(
        act * gains[labels],
        -1.0,
        1.0,
    ).astype(np.float32)
    return {
        "obs": obs.astype(np.float32),
        "act": act.astype(np.float32),
        "rew": rew.astype(np.float32),
        "next_obs": next_obs.astype(np.float32),
        "done": done.astype(np.float32),
        "target_action": target_action,
        "mode_id": labels,
        "identity": identity,
    }


def _collect_stationary(controller_seeds, event_seeds, steps):
    sequences = []
    for controller_seed in controller_seeds:
        for role in protocol.ROLES:
            config, agent, policy_state = common._load_controller(
                exploratory, controller_seed, role)
            graphdef = nnx.graphdef(agent.policy)
            for event_seed in event_seeds:
                for mode in protocol.MODES:
                    run_config = copy.deepcopy(config)
                    run_config.stochastic_mode_fixed_id = int(mode)
                    env = make_env(
                        run_config,
                        seed_offset=int(event_seed) - int(config.seed),
                    )
                    tasks = env.sample_tasks(len(protocol.MODES))
                    env.set_nonstationary_para(tasks)
                    env.set_task(tasks[mode])
                    env.build_rollout_fn(
                        graphdef, direct_policy_context=True)
                    key = jax.random.PRNGKey(
                        int(event_seed) * 100_000
                        + int(controller_seed) * 100
                        + (0 if role == "robust" else 10)
                        + int(mode)
                    )
                    transitions, _ = env.rollout(
                        policy_state,
                        int(steps),
                        key,
                        belief_vec=jnp.asarray(
                            common._policy_context(role, mode)),
                    )
                    if len(transitions) != 5:
                        raise ValueError(
                            "expected-action collection exposed extra fields")
                    sequences.append(_sequence(
                        transitions,
                        np.full((steps,), mode, dtype=np.int32),
                        {
                            "kind": "stationary",
                            "controller_seed": int(controller_seed),
                            "role": role,
                            "event_seed": int(event_seed),
                            "mode": int(mode),
                        },
                    ))
                    if hasattr(env, "close"):
                        env.close()
                    print(
                        "expected-action data stationary "
                        f"seed={controller_seed} role={role} "
                        f"event={event_seed} mode={mode}",
                        flush=True,
                    )
    return sequences


def _collect_switching(controller_seeds, event_seeds, episodes):
    sequences = []
    chunks = protocol.MAX_EPISODE_STEPS // protocol.DWELL_STEPS
    if chunks * protocol.DWELL_STEPS != protocol.MAX_EPISODE_STEPS:
        raise ValueError("switching horizon must divide into complete dwells")
    for controller_seed in controller_seeds:
        for role in protocol.ROLES:
            config, agent, policy_state = common._load_controller(
                exploratory, controller_seed, role)
            graphdef = nnx.graphdef(agent.policy)
            for event_seed in event_seeds:
                run_config = copy.deepcopy(config)
                run_config.stochastic_mode_fixed_id = -1
                env = make_env(
                    run_config,
                    seed_offset=int(event_seed) - int(config.seed),
                )
                tasks = env.sample_tasks(len(protocol.MODES))
                env.build_rollout_fn(
                    graphdef, direct_policy_context=True)
                for episode in range(int(episodes)):
                    switch_tasks, _ = _select_eval_switch_sequence(
                        env, tasks, episode)
                    _reset_eval_switch_schedule(
                        env,
                        switch_tasks,
                        run_config,
                        protocol.DWELL_STEPS,
                    )
                    pieces = []
                    labels = []
                    for chunk in range(chunks):
                        mode = int(env.current_task_id)
                        key = jax.random.PRNGKey(
                            int(event_seed) * 1_000_000
                            + int(controller_seed) * 10_000
                            + (0 if role == "robust" else 5_000)
                            + episode * 100
                            + chunk
                        )
                        transitions, _ = env.rollout(
                            policy_state,
                            protocol.DWELL_STEPS,
                            key,
                            belief_vec=jnp.asarray(
                                common._policy_context(role, mode)),
                            continue_state=chunk > 0,
                        )
                        if len(transitions) != 5:
                            raise ValueError(
                                "expected-action collection exposed "
                                "extra fields")
                        pieces.append(tuple(
                            np.asarray(value) for value in transitions))
                        labels.append(np.full(
                            (protocol.DWELL_STEPS,),
                            mode,
                            dtype=np.int32,
                        ))
                    joined = tuple(np.concatenate(
                        [piece[index] for piece in pieces], axis=0)
                        for index in range(5))
                    sequences.append(_sequence(
                        joined,
                        np.concatenate(labels),
                        {
                            "kind": "switching",
                            "controller_seed": int(controller_seed),
                            "role": role,
                            "event_seed": int(event_seed),
                            "episode": int(episode),
                        },
                    ))
                if hasattr(env, "close"):
                    env.close()
                print(
                    "expected-action data switching "
                    f"seed={controller_seed} role={role} event={event_seed}",
                    flush=True,
                )
    return sequences


def _flatten(sequences):
    return {
        key: np.concatenate([row[key] for row in sequences], axis=0)
        for key in ("obs", "act", "next_obs", "target_action", "mode_id")
    }


def _build_update(model):
    graphdef = nnx.graphdef(model)
    optimizer = optax.chain(
        optax.clip_by_global_norm(10.0),
        optax.adamw(
            protocol.MODEL_CONFIG["learning_rate"],
            weight_decay=protocol.MODEL_CONFIG["weight_decay"],
        ),
    )
    optimizer_state = optimizer.init(nnx.state(model, nnx.Param))

    @jax.jit
    def update(params, opt_state, obs, next_obs, target_action):
        def loss_fn(candidate):
            current = nnx.merge(graphdef, candidate)
            predicted = current.predict_head_batches(obs, next_obs)
            residual = predicted - target_action
            loss = jnp.mean(jnp.square(residual))
            return loss, (
                jnp.sqrt(jnp.mean(jnp.square(residual))),
                jnp.mean(jnp.var(predicted, axis=0)),
            )

        (loss, metrics), gradients = jax.value_and_grad(
            loss_fn, has_aux=True)(params)
        updates, next_opt_state = optimizer.update(
            gradients, opt_state, params)
        next_params = optax.apply_updates(params, updates)
        next_params = jax.tree.map(
            lambda value: jnp.nan_to_num(value), next_params)
        return next_params, next_opt_state, (loss,) + metrics

    return update, optimizer_state


def _sample_batch(dataset, rng):
    indices = rng.integers(
        0,
        len(dataset["obs"]),
        size=(
            protocol.MODEL_CONFIG["ensemble_size"],
            protocol.BATCH_SIZE,
        ),
    )
    return tuple(jnp.asarray(dataset[key][indices]) for key in (
        "obs", "next_obs", "target_action"))


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
        np.square(
            np.mean(predicted, axis=0) - sequence["target_action"])
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
    absolute = []
    ensemble_variance = []
    for sequence, predicted in prediction_rows:
        target = sequence["target_action"]
        mean_prediction = np.mean(predicted, axis=0)
        squared.append(np.square(mean_prediction - target))
        absolute.append(np.abs(mean_prediction - target))
        ensemble_variance.append(np.var(predicted, axis=0))
    squared = np.concatenate(squared, axis=0)
    absolute = np.concatenate(absolute, axis=0)
    ensemble_variance = np.concatenate(ensemble_variance, axis=0)
    return {
        "rmse": float(np.sqrt(np.mean(squared))),
        "mae": float(np.mean(absolute)),
        "per_action_rmse": [
            float(value) for value in np.sqrt(np.mean(squared, axis=0))
        ],
        "ensemble_variance_mean": float(np.mean(ensemble_variance)),
    }


def _checkpoint_payload(update_index, model, optimizer_state, rng, history):
    metadata = protocol.save_parameter_state(
        TRAIN_STATE_NPZ,
        (nnx.state(model, nnx.Param), optimizer_state),
    )
    protocol.write_json_atomic(TRAIN_STATE_JSON, {
        "schema": (
            "bapr.regime-polarity-expected-action-system-id-train-state.v4"),
        "status": "in_progress",
        "update_index": int(update_index),
        "parameter_file": protocol.file_record(TRAIN_STATE_NPZ),
        "parameter_leaves": metadata,
        "rng_state": rng.bit_generator.state,
        "model_config": protocol.MODEL_CONFIG,
        "source_files": _source_records(),
        "history": history,
    })


def _restore_checkpoint(model, optimizer_state):
    if not TRAIN_STATE_JSON.is_file():
        return None
    payload = protocol.read_json(TRAIN_STATE_JSON)
    if (payload.get("schema")
            != "bapr.regime-polarity-expected-action-system-id-train-state.v4"
            or payload.get("status") != "in_progress"
            or payload.get("model_config") != protocol.MODEL_CONFIG
            or payload.get("source_files") != _source_records()
            or payload.get("parameter_file")
            != protocol.file_record(TRAIN_STATE_NPZ)):
        raise ValueError(
            "stale or incompatible expected-action train state")
    state = protocol.load_parameter_state(
        TRAIN_STATE_NPZ,
        (nnx.state(model, nnx.Param), optimizer_state),
        payload["parameter_leaves"],
    )
    return payload, state


def _source_bundle_records():
    records = {}
    for split, seeds in (
        ("train", protocol.TRAIN_CONTROLLER_SEEDS),
        ("validation", protocol.VALIDATION_CONTROLLER_SEEDS),
    ):
        for seed in seeds:
            for role in protocol.ROLES:
                path = exploratory.bundle_manifest(
                    protocol.ENV, role, seed)
                common._validate_bundle(exploratory, seed, role)
                records[f"{split}/{role}/seed_{seed}"] = (
                    protocol.file_record(path))
    return records


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--updates", type=int, default=protocol.TRAIN_UPDATES)
    parser.add_argument(
        "--stationary-steps", type=int, default=protocol.STATIONARY_STEPS)
    parser.add_argument(
        "--switching-episodes", type=int,
        default=protocol.SWITCHING_EPISODES)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(
            args.updates,
            args.stationary_steps,
            args.switching_episodes,
            args.checkpoint_interval) <= 0:
        parser.error("all training controls must be positive")
    return args


def main():
    args = parse_args()
    if protocol.MODEL_MANIFEST.is_file():
        model_lib.load_model(17, 6)
        print(
            "EXPECTED-ACTION SYSTEM ID ALREADY COMPLETE: "
            f"{protocol.MODEL_ROOT}",
            flush=True,
        )
        return
    if protocol.MODEL_ROOT.exists() and not args.resume:
        raise RuntimeError(
            f"partial expected-action root requires --resume: "
            f"{protocol.MODEL_ROOT}")
    protocol.MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    _, reference_agent, _ = common._load_controller(
        exploratory, protocol.TRAIN_CONTROLLER_SEEDS[0], "robust")
    model = model_lib.make_model(
        reference_agent.obs_dim, reference_agent.act_dim, MODEL_SEED)
    update_fn, optimizer_state = _build_update(model)
    rng = np.random.default_rng(MODEL_SEED)
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
        print(
            f"Resumed expected-action system ID at update {start_update}",
            flush=True,
        )
    else:
        _checkpoint_payload(0, model, optimizer_state, rng, history)

    train_sequences = (
        _collect_stationary(
            protocol.TRAIN_CONTROLLER_SEEDS,
            protocol.TRAIN_EVENT_SEEDS,
            args.stationary_steps,
        )
        + _collect_switching(
            protocol.TRAIN_CONTROLLER_SEEDS,
            protocol.TRAIN_EVENT_SEEDS,
            args.switching_episodes,
        )
    )
    validation_sequences = (
        _collect_stationary(
            protocol.VALIDATION_CONTROLLER_SEEDS,
            protocol.VALIDATION_EVENT_SEEDS,
            args.stationary_steps,
        )
        + _collect_switching(
            protocol.VALIDATION_CONTROLLER_SEEDS,
            protocol.VALIDATION_EVENT_SEEDS,
            args.switching_episodes,
        )
    )
    train_dataset = _flatten(train_sequences)

    for update_index in range(start_update, args.updates):
        batch = _sample_batch(train_dataset, rng)
        params, optimizer_state, metrics = update_fn(
            nnx.state(model, nnx.Param), optimizer_state, *batch)
        nnx.update(model, params)
        completed = update_index + 1
        if completed == 1 or completed % 50 == 0:
            row = {
                "update": int(completed),
                "loss": float(metrics[0]),
                "rmse": float(metrics[1]),
                "ensemble_variance": float(metrics[2]),
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
    gains = jnp.asarray(
        protocol.mode_gain_vectors(reference_agent.act_dim),
        dtype=jnp.float32,
    )
    variance_jax = jnp.asarray(residual_variance, dtype=jnp.float32)
    evidence_fn = model_lib.build_evidence_fn(model)
    prepared = [
        (
            sequence,
            v3_trainer._sequence_evidence(
                evidence_fn,
                params,
                sequence,
                gains,
                variance_jax,
            )[0],
        )
        for sequence in validation_sequences
    ]
    selected, candidates = v3_trainer._select_filter(prepared)
    parameter_leaves = protocol.save_parameter_state(
        protocol.MODEL_PATH, params)
    manifest = {
        "schema": protocol.MODEL_SCHEMA,
        "status": "complete",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "env": protocol.ENV,
        "family": protocol.FAMILY,
        "model_seed": MODEL_SEED,
        "model_config": protocol.MODEL_CONFIG,
        "training_controller_seeds": list(
            protocol.TRAIN_CONTROLLER_SEEDS),
        "validation_controller_seeds": list(
            protocol.VALIDATION_CONTROLLER_SEEDS),
        "test_controller_seeds": list(
            protocol.TEST_CONTROLLER_SEEDS),
        "train_event_seeds": list(protocol.TRAIN_EVENT_SEEDS),
        "validation_event_seeds": list(
            protocol.VALIDATION_EVENT_SEEDS),
        "test_event_seeds": list(protocol.TEST_EVENT_SEEDS),
        "training_data_policy": (
            "target_action is computed only as "
            "clip(gain(true_training_mode) * commanded_action); rollout "
            "never requests or records realized executed_action"),
        "online_data_policy": (
            "consecutive observations and commanded action only"),
        "training_config": {
            "updates": int(args.updates),
            "batch_size_per_head": protocol.BATCH_SIZE,
            "stationary_steps": int(args.stationary_steps),
            "switching_episodes": int(args.switching_episodes),
        },
        "mode_gain_vectors": np.asarray(gains).tolist(),
        "residual_variance": residual_variance.tolist(),
        "training_inverse_metrics": _inverse_metrics(train_predictions),
        "validation_inverse_metrics": _inverse_metrics(
            validation_predictions),
        "filter_config": selected["filter_config"],
        "validation_gate_pass": selected["validation_gate_pass"],
        "validation": selected,
        "validation_candidates": candidates,
        "training_history": history,
        "source_bundles": _source_bundle_records(),
        "source_files": _source_records(),
        "parameter_file": protocol.file_record(protocol.MODEL_PATH),
        "parameter_leaves": parameter_leaves,
    }
    protocol.write_json_atomic(protocol.MODEL_MANIFEST, manifest)
    model_lib.load_model(reference_agent.obs_dim, reference_agent.act_dim)
    if CHECKPOINT_ROOT.exists():
        shutil.rmtree(CHECKPOINT_ROOT)
    print(
        "EXPECTED-ACTION SYSTEM ID COMPLETE: "
        f"validation_gate="
        f"{'PASS' if selected['validation_gate_pass'] else 'FAIL'} "
        f"filter={selected['filter_config']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
