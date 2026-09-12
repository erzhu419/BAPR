"""Train a causal selector over frozen slow and fast sequence filters."""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_experiments.analysis import bapr_v3_sequence_router_gate as protocol
from jax_experiments.analysis import train_bapr_v3_sequence_router_dual as dual_train
from jax_experiments.analysis import train_bapr_v3_sequence_router_v2 as curriculum


UPDATES = 1500
BATCH_SIZE = 512
LEARNING_RATE = 3e-4


def source_records():
    paths = (
        Path(__file__).resolve(),
        Path(protocol.__file__).resolve(),
        Path(dual_train.__file__).resolve(),
        Path(curriculum.__file__).resolve(),
    )
    return {
        str(path.relative_to(protocol.ROOT)): protocol.file_record(path)
        for path in paths
    }


def load_dual_manifest():
    path = protocol.dual.MANIFEST_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (payload.get("schema") != protocol.dual.SCHEMA
            or payload.get("status") != "complete"
            or payload.get("family") != protocol.FAMILY
            or payload.get("env") != protocol.ENV
            or payload.get("slow_parameter_file")
            != protocol.file_record(protocol.dual.SLOW_MODEL_PATH)
            or payload.get("fast_parameter_file")
            != protocol.file_record(protocol.dual.FAST_MODEL_PATH)
            or payload.get("privileged_correctness_oracle", {}).get(
                "full_cycle", {}).get("action_accuracy", 0.0) < 0.90):
        raise ValueError("dual-router artifact has no valid selector headroom")
    return payload


def before_action_probabilities(logits):
    posteriors = np.asarray(jax.nn.softmax(logits, axis=-1), dtype=np.float64)
    uniform = np.full((len(posteriors), 1, 4), 0.25, dtype=np.float64)
    return np.concatenate([uniform, posteriors[:, :-1]], axis=1)


def gate_features(slow_logits, fast_logits, oracle_map):
    slow = before_action_probabilities(slow_logits)
    fast = before_action_probabilities(fast_logits)

    def summaries(probabilities):
        ordered = np.sort(probabilities, axis=-1)
        confidence = ordered[..., -1:]
        margin = ordered[..., -1:] - ordered[..., -2:-1]
        entropy = -np.sum(
            probabilities * np.log(np.clip(probabilities, 1e-8, 1.0)),
            axis=-1, keepdims=True)
        return confidence, margin, entropy

    slow_summary = summaries(slow)
    fast_summary = summaries(fast)
    difference = np.abs(slow - fast)
    symmetric_kl = 0.5 * np.sum(
        slow * (np.log(np.clip(slow, 1e-8, 1.0))
                - np.log(np.clip(fast, 1e-8, 1.0)))
        + fast * (np.log(np.clip(fast, 1e-8, 1.0))
                  - np.log(np.clip(slow, 1e-8, 1.0))),
        axis=-1, keepdims=True,
    )
    mapping = np.asarray(oracle_map, dtype=np.int32)
    slow_controller = mapping[np.argmax(slow, axis=-1)]
    fast_controller = mapping[np.argmax(fast, axis=-1)]
    disagreement = (slow_controller != fast_controller)[..., None].astype(
        np.float64)
    count = np.arange(slow.shape[1], dtype=np.float64)[None, :, None]
    count = np.broadcast_to(
        np.log1p(count) / np.log(float(slow.shape[1])),
        (*slow.shape[:2], 1),
    )
    features = np.concatenate([
        slow, fast,
        *slow_summary, *fast_summary,
        difference, symmetric_kl, disagreement, count,
    ], axis=-1)
    if features.shape[-1] != len(protocol.FEATURE_NAMES):
        raise RuntimeError("causal gate feature schema changed")
    return features.astype(np.float32), slow_controller, fast_controller


def exclusive_targets(slow_controller, fast_controller, labels, oracle_map):
    expected = np.asarray(oracle_map, dtype=np.int32)[labels]
    slow_correct = slow_controller == expected
    fast_correct = fast_controller == expected
    mask = slow_correct ^ fast_correct
    target = fast_correct.astype(np.float32)
    return mask, target


def initialize_params(seed, input_dim, hidden_dim):
    rng = np.random.default_rng(seed)
    return {
        "w1": jnp.asarray(rng.normal(
            0.0, input_dim ** -0.5, (input_dim, hidden_dim)),
            dtype=jnp.float32),
        "b1": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "w2": jnp.asarray(rng.normal(
            0.0, hidden_dim ** -0.5, (hidden_dim, 1)),
            dtype=jnp.float32),
        "b2": jnp.zeros((1,), dtype=jnp.float32),
    }


def gate_logits(params, features):
    hidden = jax.nn.silu(features @ params["w1"] + params["b1"])
    return (hidden @ params["w2"] + params["b2"])[..., 0]


def build_update(params):
    optimizer = optax.chain(
        optax.clip_by_global_norm(5.0),
        optax.adam(LEARNING_RATE),
    )
    optimizer_state = optimizer.init(params)

    @jax.jit
    def update(current, state, features, targets):
        def loss_fn(candidate):
            logits = gate_logits(candidate, features)
            loss = optax.sigmoid_binary_cross_entropy(
                logits, targets).mean()
            loss += 1e-5 * sum(
                jnp.sum(value * value) for value in candidate.values())
            accuracy = jnp.mean(
                (logits >= 0.0) == (targets >= 0.5))
            return loss, accuracy

        (loss, accuracy), gradients = jax.value_and_grad(
            loss_fn, has_aux=True)(current)
        updates, next_state = optimizer.update(gradients, state, current)
        return optax.apply_updates(current, updates), next_state, loss, accuracy

    return update, optimizer_state


def select_balanced_batch(features, targets, rng):
    negative = np.flatnonzero(targets < 0.5)
    positive = np.flatnonzero(targets >= 0.5)
    if min(len(negative), len(positive)) == 0:
        raise ValueError("causal gate has no exclusive-correct examples")
    half = BATCH_SIZE // 2
    indices = np.concatenate([
        rng.choice(negative, half, replace=len(negative) < half),
        rng.choice(positive, BATCH_SIZE - half,
                   replace=len(positive) < BATCH_SIZE - half),
    ])
    rng.shuffle(indices)
    return jnp.asarray(features[indices]), jnp.asarray(targets[indices])


def gate_metrics(
    params,
    features,
    slow_controller,
    fast_controller,
    labels,
    kind,
    oracle_map,
    threshold,
):
    probabilities = np.asarray(jax.nn.sigmoid(gate_logits(
        params, jnp.asarray(features))))
    choose_fast = (
        (probabilities >= float(threshold))
        & (slow_controller != fast_controller)
    )
    decisions = np.where(choose_fast, fast_controller, slow_controller)
    metrics = dual_train._aggregate_decisions(
        decisions, labels, kind, oracle_map)
    metrics["fast_selection_rate"] = {
        "stationary": float(np.mean(choose_fast[kind == 0])),
        "full_cycle": float(np.mean(choose_fast[kind == 1])),
    }
    return metrics


def candidate_key(metrics):
    checks = dual_train.gate(metrics)
    return (
        int(checks["passed"]),
        min(metrics["stationary"]["action_accuracy"],
            metrics["full_cycle"]["action_accuracy"]),
        metrics["full_cycle"]["action_accuracy"],
        metrics["stationary"]["action_accuracy"],
    )


def save_initial_state(dual_manifest):
    protocol.write_json_atomic(protocol.TRAIN_STATE_META_PATH, {
        "schema": protocol.TRAIN_STATE_SCHEMA,
        "status": "dataset_ready",
        "dataset_file": protocol.file_record(protocol.DATASET_PATH),
        "dual_manifest_file": protocol.file_record(protocol.dual.MANIFEST_PATH),
        "slow_parameter_file": protocol.file_record(
            protocol.dual.SLOW_MODEL_PATH),
        "fast_parameter_file": protocol.file_record(
            protocol.dual.FAST_MODEL_PATH),
        "feature_names": list(protocol.FEATURE_NAMES),
        "model_config": protocol.MODEL_CONFIG,
        "source_files": source_records(),
    })


def validate_initial_state():
    payload = json.loads(protocol.TRAIN_STATE_META_PATH.read_text())
    if (payload.get("schema") != protocol.TRAIN_STATE_SCHEMA
            or payload.get("status") != "dataset_ready"
            or payload.get("dataset_file")
            != protocol.file_record(protocol.DATASET_PATH)
            or payload.get("dual_manifest_file")
            != protocol.file_record(protocol.dual.MANIFEST_PATH)
            or payload.get("slow_parameter_file")
            != protocol.file_record(protocol.dual.SLOW_MODEL_PATH)
            or payload.get("fast_parameter_file")
            != protocol.file_record(protocol.dual.FAST_MODEL_PATH)
            or payload.get("feature_names") != list(protocol.FEATURE_NAMES)
            or payload.get("model_config") != protocol.MODEL_CONFIG
            or payload.get("source_files") != source_records()):
        raise ValueError("stale causal-gate dataset checkpoint")


def main() -> None:
    protocol.configure()
    if protocol.MANIFEST_PATH.is_file():
        print(f"Complete causal gate exists: {protocol.MANIFEST_PATH}")
        return
    dual_manifest = load_dual_manifest()
    protocol.MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    if protocol.TRAIN_STATE_META_PATH.is_file():
        validate_initial_state()
        dataset = curriculum._load_dataset(protocol.DATASET_PATH)
        print("Resumed causal-gate evidence dataset", flush=True)
    else:
        if protocol.DATASET_PATH.exists():
            raise RuntimeError("orphaned causal-gate evidence dataset")
        dataset = curriculum.collect_dataset(protocol.DATASET_PATH)
        save_initial_state(dual_manifest)

    slow_model, slow_params = dual_train.load_model(
        protocol.dual.SLOW_MODEL_PATH,
        dual_manifest["slow_parameter_leaves"])
    fast_model, fast_params = dual_train.load_model(
        protocol.dual.FAST_MODEL_PATH,
        dual_manifest["fast_parameter_leaves"])
    slow_fn = curriculum.build_validation_fn(slow_model)
    fast_fn = curriculum.build_validation_fn(fast_model)
    table = protocol.utility.load_utility_table()
    oracle_map = table["oracle_controller_map"]

    splits = {}
    for split in ("train", "validation"):
        evidence = jnp.asarray(dataset[f"{split}_evidence"])
        slow_logits = slow_fn(slow_params, evidence)
        fast_logits = fast_fn(fast_params, evidence)
        features, slow_controller, fast_controller = gate_features(
            slow_logits, fast_logits, oracle_map)
        mask, targets = exclusive_targets(
            slow_controller, fast_controller,
            dataset[f"{split}_labels"], oracle_map)
        splits[split] = {
            "features": features,
            "slow_controller": slow_controller,
            "fast_controller": fast_controller,
            "mask": mask,
            "targets": targets,
        }

    train_features = splits["train"]["features"]
    train_mask = splits["train"]["mask"]
    feature_mean = np.mean(train_features[train_mask], axis=0)
    feature_std = np.std(train_features[train_mask], axis=0)
    feature_std = np.maximum(feature_std, 1e-4)
    for split in splits.values():
        split["features"] = (
            (split["features"] - feature_mean) / feature_std
        ).astype(np.float32)

    flat_train_features = splits["train"]["features"][train_mask]
    flat_train_targets = splits["train"]["targets"][train_mask]
    candidates = []
    parameter_states = {}
    histories = {}
    for seed in protocol.TRAINING_SEEDS:
        params = initialize_params(seed, **protocol.MODEL_CONFIG)
        update, optimizer_state = build_update(params)
        rng = np.random.default_rng(20260718 + seed)
        history = []
        for update_index in range(UPDATES):
            batch = select_balanced_batch(
                flat_train_features, flat_train_targets, rng)
            params, optimizer_state, loss, accuracy = update(
                params, optimizer_state, *batch)
            completed = update_index + 1
            if completed % 250 == 0:
                history.append({
                    "update": completed,
                    "loss": float(loss),
                    "balanced_accuracy": float(accuracy),
                })
        parameter_states[seed] = params
        histories[str(seed)] = history
        validation = splits["validation"]
        for threshold in protocol.THRESHOLDS:
            metrics = gate_metrics(
                params, validation["features"],
                validation["slow_controller"],
                validation["fast_controller"],
                dataset["validation_labels"], dataset["validation_kind"],
                oracle_map, threshold)
            candidates.append({
                "seed": seed,
                "threshold": threshold,
                "metrics": metrics,
                "gate": dual_train.gate(metrics),
            })
    selected = max(candidates, key=lambda row: candidate_key(row["metrics"]))
    selected_params = {
        **parameter_states[selected["seed"]],
        "feature_mean": jnp.asarray(feature_mean),
        "feature_std": jnp.asarray(feature_std),
    }
    metadata = protocol.emission.save_parameter_state(
        protocol.MODEL_PATH, selected_params)
    train_counts = {
        "exclusive_rows": int(np.sum(train_mask)),
        "fast_correct_only": int(np.sum(
            splits["train"]["targets"][train_mask] >= 0.5)),
        "slow_correct_only": int(np.sum(
            splits["train"]["targets"][train_mask] < 0.5)),
    }
    manifest = {
        "schema": protocol.SCHEMA,
        "status": "complete",
        "family": protocol.FAMILY,
        "env": protocol.ENV,
        "training_seed": selected["seed"],
        "train_event_seeds": list(protocol.TRAIN_EVENT_SEEDS),
        "validation_event_seeds": list(protocol.VALIDATION_EVENT_SEEDS),
        "sealed_holdout_event_seeds": list(protocol.HOLDOUT_EVENT_SEEDS),
        "feature_names": list(protocol.FEATURE_NAMES),
        "model_config": protocol.MODEL_CONFIG,
        "training_seeds": list(protocol.TRAINING_SEEDS),
        "thresholds": list(protocol.THRESHOLDS),
        "updates": UPDATES,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "training_counts": train_counts,
        "training_histories": histories,
        "candidates": candidates,
        "selected": selected,
        "validation_gate_pass": bool(selected["gate"]["passed"]),
        "parameter_file": protocol.file_record(protocol.MODEL_PATH),
        "parameter_leaves": metadata,
        "dual_manifest_file": protocol.file_record(
            protocol.dual.MANIFEST_PATH),
        "slow_parameter_file": protocol.file_record(
            protocol.dual.SLOW_MODEL_PATH),
        "fast_parameter_file": protocol.file_record(
            protocol.dual.FAST_MODEL_PATH),
        "utility_table_file": protocol.file_record(protocol.utility.TABLE_PATH),
        "oracle_controller_map": oracle_map,
        "source_files": source_records(),
        "causality": (
            "gate inputs are previous-transition slow/fast posterior summaries "
            "and observed-step count only; no mode, switch time, or future data"),
    }
    protocol.write_json_atomic(protocol.MANIFEST_PATH, manifest)
    protocol.DATASET_PATH.unlink()
    protocol.TRAIN_STATE_META_PATH.unlink()
    print(
        "CAUSAL SEQUENCE GATE COMPLETE: "
        f"seed={selected['seed']} threshold={selected['threshold']} "
        f"gate={'PASS' if selected['gate']['passed'] else 'FAIL'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
