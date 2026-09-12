"""Train a causal recurrent filter over frozen probabilistic mode emissions."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from jax_experiments.analysis import bapr_v3_sequence_router as protocol
from jax_experiments.analysis import bapr_v3_utility_aware_router as utility
from jax_experiments.analysis import (
    run_bapr_v3_independent_specialist_audit as specialist_audit,
)
from jax_experiments.analysis import train_bapr_v3_learned_control_router as emission_train
from jax_experiments.networks.causal_mode_filter import CausalModeFilter
from jax_experiments.train import make_env


TRAIN_STATE_SCHEMA = "bapr.v3-sequence-router-train-state.v1"
BEHAVIOR_CONTROLLERS = (
    "robust", "fixed_mode_0", "fixed_mode_2", "fixed_mode_3")


def source_records() -> dict[str, dict[str, Any]]:
    paths = (
        Path(__file__).resolve(),
        Path(protocol.__file__).resolve(),
        Path(CausalModeFilter.__module__.replace(".", "/") + ".py"),
    )
    output = {}
    for path in paths:
        resolved = path if path.is_absolute() else protocol.ROOT / path
        resolved = resolved.resolve()
        output[str(resolved.relative_to(protocol.ROOT))] = (
            protocol.file_record(resolved))
    return output


def make_sequence_model(seed: int) -> CausalModeFilter:
    return CausalModeFilter(
        **protocol.MODEL_CONFIG,
        rngs=nnx.Rngs(seed),
    )


def load_frozen_emission_model(obs_dim: int, act_dim: int):
    manifest = protocol.emission.load_manifest()
    model = emission_train.make_model(obs_dim, act_dim, seed=20260718)
    template = nnx.state(model, nnx.Param)
    params = protocol.emission.load_parameter_state(
        protocol.emission.MODEL_PATH,
        template,
        manifest["parameter_leaves"],
    )
    nnx.update(model, params)
    return model, manifest


def collect_full_cycle_sequences(
    config,
    policy_graphdef,
    policy_states: dict[str, Any],
    event_seeds: tuple[int, ...],
):
    sequences = []
    horizon = int(config.max_episode_steps)
    period = protocol.FULL_CYCLE_DWELL_STEPS
    if horizon != period * 4:
        raise ValueError("sequence router requires a 1000-step four-mode cycle")
    for event_seed in event_seeds:
        run_config = copy.deepcopy(config)
        run_config.stochastic_mode_fixed_id = -1
        env = make_env(run_config, seed_offset=event_seed)
        tasks = env.sample_tasks(4)
        env.build_rollout_fn(policy_graphdef)
        for controller_index, controller in enumerate(BEHAVIOR_CONTROLLERS):
            for episode, sequence in enumerate(protocol.FULL_CYCLE_SEQUENCES):
                env.configure_eval_mode_sequence(tasks, sequence, period)
                parts = []
                labels = []
                for chunk in range(4):
                    mode = int(env.current_task_id)
                    if mode != int(sequence[chunk]):
                        raise RuntimeError("full-cycle collection lost mode order")
                    key = jax.random.PRNGKey(
                        event_seed * 100_000
                        + controller_index * 10_000
                        + episode * 100
                        + chunk)
                    transitions, _ = env.rollout(
                        policy_states[controller],
                        period,
                        key,
                        continue_state=chunk > 0,
                    )
                    parts.append(tuple(np.asarray(value)
                                       for value in transitions))
                    labels.append(np.full((period,), mode, dtype=np.int32))
                joined = tuple(np.concatenate(
                    [part[index] for part in parts], axis=0)
                    for index in range(5))
                sequences.append(emission_train._as_sequence(
                    joined,
                    np.concatenate(labels),
                    {
                        "kind": "full_cycle",
                        "event_seed": event_seed,
                        "controller": controller,
                        "episode": episode,
                        "mode_sequence": list(sequence),
                    },
                ))
            print(
                f"  full-cycle data seed={event_seed} "
                f"controller={controller} sequences={len(sequences)}",
                flush=True,
            )
        if hasattr(env, "close"):
            env.close()
    return sequences


def prepare_sequences(emission_model, sequences):
    emission_fn = emission_train.build_emission_fn(emission_model)
    params = nnx.state(emission_model, nnx.Param)
    prepared = []
    for sequence in sequences:
        evidence = emission_train.sequence_emissions(
            emission_fn, params, sequence)
        labels = np.asarray(sequence["mode_id"], dtype=np.int32)
        switch_weight = np.ones_like(labels, dtype=np.float32)
        switch_points = np.flatnonzero(labels[1:] != labels[:-1]) + 1
        for point in switch_points:
            switch_weight[point:min(point + 64, len(labels))] = 3.0
        prepared.append({
            "evidence": evidence.astype(np.float32),
            "labels": labels,
            "weights": switch_weight,
            "identity": sequence["identity"],
        })
    return prepared


def sample_batch(sequences, batch_size, context_length, rng):
    selected = rng.integers(0, len(sequences), size=batch_size)
    evidence = []
    labels = []
    weights = []
    for index in selected:
        sequence = sequences[int(index)]
        maximum = len(sequence["labels"]) - context_length
        if maximum < 0:
            raise ValueError("sequence shorter than requested context")
        start = int(rng.integers(0, maximum + 1))
        stop = start + context_length
        evidence.append(sequence["evidence"][start:stop])
        labels.append(sequence["labels"][start:stop])
        weights.append(sequence["weights"][start:stop])
    return (
        jnp.asarray(np.stack(evidence)),
        jnp.asarray(np.stack(labels)),
        jnp.asarray(np.stack(weights)),
    )


def build_update(model, learning_rate: float, burnin: int):
    graphdef, initial_params, non_parameter_state = nnx.split(
        model, nnx.Param, ...)
    optimizer = optax.chain(
        optax.clip_by_global_norm(5.0),
        optax.adam(learning_rate),
    )
    optimizer_state = optimizer.init(initial_params)

    @jax.jit
    def update(params, opt_state, evidence, labels, weights):
        def loss_fn(candidate):
            current = nnx.merge(
                graphdef, candidate, non_parameter_state)

            def one(sequence):
                return current.sequence(sequence)[1]

            logits = jax.vmap(one)(evidence)
            losses = optax.softmax_cross_entropy_with_integer_labels(
                logits, labels)
            mask = (jnp.arange(losses.shape[1]) >= burnin)[None, :]
            effective = weights * mask
            loss = jnp.sum(losses * effective) / jnp.maximum(
                jnp.sum(effective), 1.0)
            accuracy = jnp.sum(
                (jnp.argmax(logits, axis=-1) == labels) * effective
            ) / jnp.maximum(jnp.sum(effective), 1.0)
            return loss, accuracy

        (loss, accuracy), gradients = jax.value_and_grad(
            loss_fn, has_aux=True)(params)
        updates, next_opt_state = optimizer.update(
            gradients, opt_state, params)
        next_params = optax.apply_updates(params, updates)
        return next_params, next_opt_state, loss, accuracy

    return update, optimizer_state


def sequence_posteriors(model, evidence):
    _, logits = model.sequence(jnp.asarray(evidence))
    return np.asarray(jax.nn.softmax(logits, axis=-1))


def route_sequence(posteriors, labels, table, decision_config):
    posteriors = np.asarray(posteriors, dtype=np.float64)
    before_action = np.concatenate([
        np.full((1, 4), 0.25, dtype=np.float64),
        posteriors[:-1],
    ])
    decisions = []
    for count, posterior in enumerate(before_action):
        selected, _ = utility.select_utility_controller(
            posterior, count, table, decision_config)
        decisions.append(selected)
    return utility.routing_metrics(
        np.asarray(decisions, dtype=np.int32),
        np.asarray(labels, dtype=np.int32),
        table["oracle_controller_map"],
        burnin=32,
    )


def validate(model, sequences, table, decision_config):
    by_kind = {"stationary": [], "full_cycle": []}
    physical = {"stationary": [], "full_cycle": []}
    for sequence in sequences:
        posteriors = sequence_posteriors(model, sequence["evidence"])
        labels = sequence["labels"]
        kind = str(sequence["identity"]["kind"])
        index = np.arange(len(labels)) >= 32
        physical[kind].append(float(np.mean(
            np.argmax(posteriors[index], axis=-1) == labels[index])))
        by_kind[kind].append(route_sequence(
            posteriors, labels, table, decision_config))

    def aggregate(kind):
        rows = by_kind[kind]
        return {
            name: float(np.mean([float(row[name]) for row in rows]))
            for name in (
                "coverage", "conditional_accuracy", "action_accuracy",
                "wrong_route_rate", "median_switch_delay")
        }

    stationary = aggregate("stationary")
    full_cycle = aggregate("full_cycle")
    gate = {
        "stationary_action_accuracy_at_least_90pct": (
            stationary["action_accuracy"] >= 0.90),
        "stationary_wrong_route_at_most_10pct": (
            stationary["wrong_route_rate"] <= 0.10),
        "full_cycle_action_accuracy_at_least_90pct": (
            full_cycle["action_accuracy"] >= 0.90),
        "full_cycle_wrong_route_at_most_10pct": (
            full_cycle["wrong_route_rate"] <= 0.10),
        "full_cycle_median_delay_below_75": (
            full_cycle["median_switch_delay"] < 75.0),
    }
    gate["passed"] = all(gate.values())
    return {
        "stationary": stationary,
        "full_cycle": full_cycle,
        "physical_mode_accuracy": {
            kind: float(np.mean(values))
            for kind, values in physical.items()
        },
        "gate": gate,
    }


def save_training_checkpoint(
    update_index,
    model,
    optimizer_state,
    rng,
    training_config,
    history,
):
    state_path = (
        protocol.MODEL_ROOT / f"train_state_{int(update_index):06d}.npz")
    combined = (nnx.state(model, nnx.Param), optimizer_state)
    metadata = protocol.emission.save_parameter_state(state_path, combined)
    payload = {
        "schema": TRAIN_STATE_SCHEMA,
        "status": "in_progress",
        "update_index": int(update_index),
        "parameter_path": state_path.name,
        "parameter_file": protocol.file_record(state_path),
        "parameter_leaves": metadata,
        "rng_state": rng.bit_generator.state,
        "training_config": training_config,
        "model_config": protocol.MODEL_CONFIG,
        "source_files": source_records(),
        "history": history,
    }
    protocol.write_json_atomic(protocol.TRAIN_STATE_META_PATH, payload)
    for old_path in protocol.MODEL_ROOT.glob("train_state_*.npz"):
        if old_path != state_path:
            old_path.unlink()


def load_training_checkpoint(model, optimizer_state, training_config):
    if not protocol.TRAIN_STATE_META_PATH.is_file():
        return None
    payload = json.loads(protocol.TRAIN_STATE_META_PATH.read_text())
    state_path = protocol.MODEL_ROOT / str(payload.get("parameter_path", ""))
    if (payload.get("schema") != TRAIN_STATE_SCHEMA
            or payload.get("status") != "in_progress"
            or payload.get("training_config") != training_config
            or payload.get("model_config") != protocol.MODEL_CONFIG):
        raise ValueError("stale or incompatible sequence-router train state")
    # No optimizer update has occurred at index zero, so rebuilding from the
    # fixed initialization seed is equivalent and permits a code-fix retry.
    if int(payload.get("update_index", -1)) == 0:
        print("Rebuilding pre-training sequence-router state", flush=True)
        return None
    if (payload.get("source_files") != source_records()
            or payload.get("parameter_file")
            != protocol.file_record(state_path)):
        raise ValueError("stale or incompatible sequence-router train state")
    restored = protocol.emission.load_parameter_state(
        state_path,
        (nnx.state(model, nnx.Param), optimizer_state),
        payload["parameter_leaves"],
    )
    return payload, restored


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--burnin", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.updates, args.batch_size, args.context_length,
           args.checkpoint_interval) <= 0:
        parser.error("integer training controls must be positive")
    if not 0 <= args.burnin < args.context_length:
        parser.error("burnin must be in [0, context_length)")
    return args


def main() -> None:
    args = parse_args()
    protocol.configure()
    if protocol.MANIFEST_PATH.is_file():
        protocol.load_manifest()
        print(f"Complete sequence router already exists: {protocol.MODEL_ROOT}")
        return
    if protocol.MODEL_ROOT.exists() and not args.resume:
        raise RuntimeError(
            f"partial sequence-router root requires --resume: "
            f"{protocol.MODEL_ROOT}")
    protocol.MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    table = utility.load_utility_table()
    config, agents, policy_states = specialist_audit._controller_policy_states(
        protocol.FAMILY)
    policy_graphdef = nnx.graphdef(agents["robust"].policy)
    emission_model, emission_manifest = load_frozen_emission_model(
        agents["robust"].obs_dim, agents["robust"].act_dim)
    model = make_sequence_model(seed=20260718)
    update_fn, optimizer_state = build_update(
        model, args.learning_rate, args.burnin)
    training_config = {
        "updates": args.updates,
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "burnin": args.burnin,
        "learning_rate": args.learning_rate,
        "checkpoint_interval": args.checkpoint_interval,
        "train_event_seeds": list(protocol.TRAIN_EVENT_SEEDS),
        "validation_event_seeds": list(protocol.VALIDATION_EVENT_SEEDS),
        "behavior_controllers": list(BEHAVIOR_CONTROLLERS),
        "full_cycle_dwell_steps": protocol.FULL_CYCLE_DWELL_STEPS,
        "full_cycle_sequences": [
            list(value) for value in protocol.FULL_CYCLE_SEQUENCES],
    }

    rng = np.random.default_rng(20260718)
    history = []
    start_update = 0
    restored = load_training_checkpoint(
        model, optimizer_state, training_config) if args.resume else None
    if restored is not None:
        payload, (params, optimizer_state) = restored
        nnx.update(model, params)
        start_update = int(payload["update_index"])
        history = list(payload.get("history") or [])
        rng.bit_generator.state = payload["rng_state"]
        print(f"Resumed sequence router at update {start_update}", flush=True)
    else:
        save_training_checkpoint(
            0, model, optimizer_state, rng, training_config, history)

    print("Collecting switch-matched sequence-router datasets", flush=True)
    train_raw = emission_train.collect_stationary_sequences(
        config, policy_graphdef, policy_states,
        protocol.TRAIN_EVENT_SEEDS, 1000)
    train_raw += collect_full_cycle_sequences(
        config, policy_graphdef, policy_states,
        protocol.TRAIN_EVENT_SEEDS)
    validation_raw = emission_train.collect_stationary_sequences(
        config, policy_graphdef, policy_states,
        protocol.VALIDATION_EVENT_SEEDS, 1000)
    validation_raw += collect_full_cycle_sequences(
        config, policy_graphdef, policy_states,
        protocol.VALIDATION_EVENT_SEEDS)
    train_sequences = prepare_sequences(emission_model, train_raw)
    validation_sequences = prepare_sequences(
        emission_model, validation_raw)

    for update_index in range(start_update, args.updates):
        batch = sample_batch(
            train_sequences, args.batch_size, args.context_length, rng)
        params = nnx.state(model, nnx.Param)
        params, optimizer_state, loss, accuracy = update_fn(
            params, optimizer_state, *batch)
        nnx.update(model, params)
        completed = update_index + 1
        if completed == 1 or completed % 50 == 0:
            row = {
                "update": completed,
                "loss": float(loss),
                "weighted_accuracy": float(accuracy),
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        if completed % args.checkpoint_interval == 0:
            save_training_checkpoint(
                completed, model, optimizer_state, rng,
                training_config, history)

    decision_config = protocol.emission.RouterConfig.from_dict(
        emission_manifest["router_config"])
    validation = validate(
        model, validation_sequences, table, decision_config)
    params = nnx.state(model, nnx.Param)
    metadata = protocol.emission.save_parameter_state(
        protocol.MODEL_PATH, params)
    manifest = {
        "schema": protocol.SCHEMA,
        "status": "complete",
        "family": protocol.FAMILY,
        "env": protocol.ENV,
        "training_seed": 0,
        "train_event_seeds": list(protocol.TRAIN_EVENT_SEEDS),
        "validation_event_seeds": list(protocol.VALIDATION_EVENT_SEEDS),
        "holdout_event_seeds": list(protocol.HOLDOUT_EVENT_SEEDS),
        "model_config": protocol.MODEL_CONFIG,
        "training_config": training_config,
        "training_history": history,
        "validation": validation,
        "validation_gate_pass": bool(validation["gate"]["passed"]),
        "decision_config": decision_config.to_dict(),
        "oracle_controller_map": table["oracle_controller_map"],
        "parameter_file": protocol.file_record(protocol.MODEL_PATH),
        "parameter_leaves": metadata,
        "emission_manifest_file": protocol.file_record(
            protocol.emission.MANIFEST_PATH),
        "emission_parameter_file": protocol.file_record(
            protocol.emission.MODEL_PATH),
        "utility_table_file": protocol.file_record(utility.TABLE_PATH),
        "source_files": source_records(),
        "data_policy": (
            "balanced stationary and 250-step four-mode cycles from robust "
            "and selected specialist behavior; frozen calibrated emission "
            "likelihoods only"),
    }
    protocol.write_json_atomic(protocol.MANIFEST_PATH, manifest)
    protocol.load_manifest()
    if protocol.TRAIN_STATE_META_PATH.exists():
        protocol.TRAIN_STATE_META_PATH.unlink()
    for path in protocol.MODEL_ROOT.glob("train_state_*.npz"):
        path.unlink()
    print(
        "SEQUENCE ROUTER COMPLETE: "
        f"gate={'PASS' if validation['gate']['passed'] else 'FAIL'} "
        f"output={protocol.MODEL_ROOT}",
        flush=True,
    )


if __name__ == "__main__":
    main()
