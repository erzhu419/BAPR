"""Train switch-centered causal filters over one frozen evidence dataset."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.analysis import bapr_v3_sequence_router_v2 as protocol
from jax_experiments.analysis import diagnose_bapr_v3_sequence_router as diagnostic
from jax_experiments.analysis import (
    run_bapr_v3_independent_specialist_audit as specialist_audit,
)
from jax_experiments.analysis import train_bapr_v3_sequence_router as v1_train


BATCH_SIZE = 16
BURNIN = 8
VALIDATION_INTERVAL = 100


def source_records() -> dict[str, dict[str, Any]]:
    paths = (
        Path(__file__).resolve(),
        Path(protocol.__file__).resolve(),
        Path(v1_train.__file__).resolve(),
        Path(diagnostic.__file__).resolve(),
        protocol.ROOT / "jax_experiments/networks/causal_mode_filter.py",
    )
    return {
        str(path.relative_to(protocol.ROOT)): protocol.file_record(path)
        for path in paths
    }


def _pack_sequences(sequences):
    return {
        "evidence": np.stack([
            np.asarray(row["evidence"], dtype=np.float32)
            for row in sequences]),
        "labels": np.stack([
            np.asarray(row["labels"], dtype=np.int8)
            for row in sequences]),
        "kind": np.asarray([
            int(row["identity"]["kind"] == "full_cycle")
            for row in sequences], dtype=np.int8),
    }


def _save_dataset(train, validation, path=None):
    path = protocol.DATASET_PATH if path is None else Path(path)
    arrays = {
        "train_evidence": train["evidence"],
        "train_labels": train["labels"],
        "train_kind": train["kind"],
        "validation_evidence": validation["evidence"],
        "validation_labels": validation["labels"],
        "validation_kind": validation["kind"],
    }
    temporary = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return arrays


def _load_dataset(path=None):
    path = protocol.DATASET_PATH if path is None else Path(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    expected = {
        "train_evidence", "train_labels", "train_kind",
        "validation_evidence", "validation_labels", "validation_kind",
    }
    if set(arrays) != expected:
        raise ValueError("incomplete switch-centered evidence dataset")
    for split in ("train", "validation"):
        evidence = arrays[f"{split}_evidence"]
        labels = arrays[f"{split}_labels"]
        kind = arrays[f"{split}_kind"]
        if (evidence.ndim != 3 or evidence.shape[-1] != 4
                or labels.shape != evidence.shape[:2]
                or kind.shape != (evidence.shape[0],)
                or evidence.shape[1] != 1000):
            raise ValueError("invalid switch-centered evidence shape")
    return arrays


def collect_dataset(path=None):
    config, agents, policy_states = specialist_audit._controller_policy_states(
        protocol.FAMILY)
    graphdef = nnx.graphdef(agents["robust"].policy)
    emission_model, _ = v1_train.load_frozen_emission_model(
        agents["robust"].obs_dim, agents["robust"].act_dim)

    def collect(seeds):
        raw = v1_train.emission_train.collect_stationary_sequences(
            config, graphdef, policy_states, seeds, 1000)
        raw += v1_train.collect_full_cycle_sequences(
            config, graphdef, policy_states, seeds)
        return _pack_sequences(v1_train.prepare_sequences(
            emission_model, raw))

    print("Collecting one frozen evidence dataset for all curricula", flush=True)
    return _save_dataset(
        collect(protocol.TRAIN_EVENT_SEEDS),
        collect(protocol.VALIDATION_EVENT_SEEDS),
        path,
    )


def switch_age(labels):
    labels = np.asarray(labels)
    output = np.full(labels.shape, -1, dtype=np.int32)
    for row in range(len(labels)):
        age = -1
        for step in range(1, labels.shape[1]):
            if labels[row, step] != labels[row, step - 1]:
                age = 0
            elif age >= 0:
                age += 1
            output[row, step] = age
    return output


def sample_curriculum_batch(
    evidence,
    labels,
    kind,
    ages,
    variant,
    rng,
    batch_size=BATCH_SIZE,
):
    context = int(variant["context_length"])
    full_indices = np.flatnonzero(kind == 1)
    batch_evidence = []
    batch_labels = []
    batch_weights = []
    for _ in range(batch_size):
        targeted = (
            len(full_indices) > 0
            and rng.random() < float(variant["switch_fraction"])
        )
        if targeted:
            sequence_index = int(rng.choice(full_indices))
            points = np.flatnonzero(
                labels[sequence_index, 1:] != labels[sequence_index, :-1]) + 1
            point = int(rng.choice(points))
            target_position = context // 3 + int(rng.integers(
                -max(1, context // 16), max(2, context // 16 + 1)))
            start = int(np.clip(
                point - target_position, 0, labels.shape[1] - context))
        else:
            sequence_index = int(rng.integers(0, len(evidence)))
            start = int(rng.integers(
                0, labels.shape[1] - context + 1))
        stop = start + context
        local_age = ages[sequence_index, start:stop]
        weights = np.ones((context,), dtype=np.float32)
        selected = (
            (local_age >= 0)
            & (local_age < int(variant["switch_span"]))
        )
        weights[selected] = float(variant["switch_weight"])
        batch_evidence.append(evidence[sequence_index, start:stop])
        batch_labels.append(labels[sequence_index, start:stop])
        batch_weights.append(weights)
    return (
        jnp.asarray(np.stack(batch_evidence)),
        jnp.asarray(np.stack(batch_labels), dtype=jnp.int32),
        jnp.asarray(np.stack(batch_weights)),
    )


def build_validation_fn(model):
    graphdef, _, non_parameter_state = nnx.split(model, nnx.Param, ...)

    @jax.jit
    def validate(params, evidence):
        current = nnx.merge(graphdef, params, non_parameter_state)
        return jax.vmap(lambda sequence: current.sequence(sequence)[1])(
            evidence)

    return validate


def no_fallback_metrics(logits, labels, kind, oracle_map):
    posteriors = np.asarray(jax.nn.softmax(logits, axis=-1))
    mapping = np.asarray(oracle_map, dtype=np.int32)
    rows = {"stationary": [], "full_cycle": []}
    for index in range(len(posteriors)):
        before = diagnostic.before_action(posteriors[index])
        decisions = mapping[np.argmax(before, axis=1)]
        key = "full_cycle" if int(kind[index]) else "stationary"
        rows[key].append(protocol.utility.routing_metrics(
            decisions, labels[index], oracle_map, burnin=32))
    return {
        key: diagnostic._aggregate_metrics(values)
        for key, values in rows.items()
    }


def validation_key(metrics):
    stationary = metrics["stationary"]
    full_cycle = metrics["full_cycle"]
    return (
        min(stationary["action_accuracy"], full_cycle["action_accuracy"]),
        full_cycle["action_accuracy"],
        stationary["action_accuracy"],
        -full_cycle["median_switch_delay"],
    )


def _variant_path(name):
    return protocol.MODEL_ROOT / f"variant_{name}.npz"


def save_state(completed, dataset_record):
    payload = {
        "schema": protocol.TRAIN_STATE_SCHEMA,
        "status": "in_progress",
        "completed_variants": completed,
        "dataset_file": dataset_record,
        "variants": protocol.VARIANTS,
        "model_config": protocol.MODEL_CONFIG,
        "source_files": source_records(),
    }
    protocol.write_json_atomic(protocol.TRAIN_STATE_META_PATH, payload)


def load_state():
    if not protocol.TRAIN_STATE_META_PATH.is_file():
        return {}
    payload = json.loads(protocol.TRAIN_STATE_META_PATH.read_text())
    if (payload.get("schema") != protocol.TRAIN_STATE_SCHEMA
            or payload.get("status") != "in_progress"
            or payload.get("variants") != protocol.VARIANTS
            or payload.get("model_config") != protocol.MODEL_CONFIG
            or payload.get("source_files") != source_records()
            or payload.get("dataset_file")
            != protocol.file_record(protocol.DATASET_PATH)):
        raise ValueError("stale switch-centered sequence training state")
    completed = payload.get("completed_variants") or {}
    for name, record in completed.items():
        if name not in protocol.VARIANTS:
            raise ValueError("unknown completed curriculum variant")
        if record.get("parameter_file") != protocol.file_record(
                _variant_path(name)):
            raise ValueError("completed curriculum parameters changed")
    return completed


def train_variant(name, variant, dataset, table):
    model = v1_train.make_sequence_model(seed=20260718)
    update, optimizer_state = v1_train.build_update(
        model, float(variant["learning_rate"]), BURNIN)
    validation_fn = build_validation_fn(model)
    rng = np.random.default_rng(20260718)
    ages = switch_age(dataset["train_labels"])
    best_params = None
    best_metrics = None
    best_update = 0
    history = []
    for update_index in range(int(variant["updates"])):
        batch = sample_curriculum_batch(
            dataset["train_evidence"], dataset["train_labels"],
            dataset["train_kind"], ages, variant, rng)
        params = nnx.state(model, nnx.Param)
        params, optimizer_state, loss, accuracy = update(
            params, optimizer_state, *batch)
        nnx.update(model, params)
        completed = update_index + 1
        if completed % VALIDATION_INTERVAL == 0:
            logits = validation_fn(
                params, jnp.asarray(dataset["validation_evidence"]))
            metrics = no_fallback_metrics(
                logits, dataset["validation_labels"],
                dataset["validation_kind"],
                table["oracle_controller_map"],
            )
            row = {
                "update": completed,
                "loss": float(loss),
                "batch_accuracy": float(accuracy),
                "validation": metrics,
            }
            history.append(row)
            if best_metrics is None or validation_key(metrics) > validation_key(
                    best_metrics):
                best_metrics = metrics
                best_params = jax.tree.map(lambda value: jnp.array(value), params)
                best_update = completed
            print(json.dumps({
                "variant": name,
                "update": completed,
                "loss": float(loss),
                "stationary": metrics["stationary"]["action_accuracy"],
                "full_cycle": metrics["full_cycle"]["action_accuracy"],
                "delay": metrics["full_cycle"]["median_switch_delay"],
            }, sort_keys=True), flush=True)
    if best_params is None or best_metrics is None:
        raise RuntimeError("curriculum variant produced no validation checkpoint")
    metadata = protocol.emission.save_parameter_state(
        _variant_path(name), best_params)
    return {
        "config": variant,
        "best_update": best_update,
        "best_validation": best_metrics,
        "history": history,
        "parameter_file": protocol.file_record(_variant_path(name)),
        "parameter_leaves": metadata,
    }


def _prepared_validation(dataset):
    return [
        {
            "evidence": dataset["validation_evidence"][index],
            "labels": dataset["validation_labels"][index],
            "identity": {
                "kind": "full_cycle" if int(
                    dataset["validation_kind"][index]) else "stationary",
            },
        }
        for index in range(len(dataset["validation_evidence"]))
    ]


def finalize(completed, dataset, table):
    best_name = max(
        completed,
        key=lambda name: validation_key(
            completed[name]["best_validation"]),
    )
    best_record = completed[best_name]
    model = v1_train.make_sequence_model(seed=20260718)
    template = nnx.state(model, nnx.Param)
    best_params = protocol.emission.load_parameter_state(
        _variant_path(best_name), template,
        best_record["parameter_leaves"])
    nnx.update(model, best_params)
    metadata = protocol.emission.save_parameter_state(
        protocol.MODEL_PATH, best_params)
    prepared = _prepared_validation(dataset)
    base_config = protocol.emission.RouterConfig.from_dict(
        protocol.emission.load_manifest()["router_config"])
    candidates = []
    posterior_by_temperature = {
        temperature: [
            diagnostic.posterior_trace(model, row["evidence"], temperature)
            for row in prepared
        ]
        for temperature in diagnostic.TEMPERATURES
    }
    for temperature in diagnostic.TEMPERATURES:
        for confidence in diagnostic.CONFIDENCE_THRESHOLDS:
            for margin in diagnostic.MARGIN_THRESHOLDS:
                for history in diagnostic.MIN_HISTORIES:
                    config = replace(
                        base_config,
                        confidence_threshold=confidence,
                        margin_threshold=margin,
                        min_history=history,
                    )
                    row = diagnostic.score_config(
                        prepared, posterior_by_temperature[temperature],
                        table, config)
                    row["config"] = {
                        "temperature": temperature,
                        "confidence_threshold": confidence,
                        "margin_threshold": margin,
                        "min_history": history,
                    }
                    candidates.append(row)
    decision = max(candidates, key=diagnostic.candidate_key)
    public_variant_results = {
        name: {
            key: value for key, value in record.items()
            if key not in ("parameter_file", "parameter_leaves")
        }
        for name, record in completed.items()
    }
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
        "variants": protocol.VARIANTS,
        "variant_results": public_variant_results,
        "selected_variant": best_name,
        "selected_validation": best_record["best_validation"],
        "selected_decision": decision,
        "validation_gate_pass": bool(decision["gate"]["passed"]),
        "parameter_file": protocol.file_record(protocol.MODEL_PATH),
        "parameter_leaves": metadata,
        "emission_manifest_file": protocol.file_record(
            protocol.emission.MANIFEST_PATH),
        "emission_parameter_file": protocol.file_record(
            protocol.emission.MODEL_PATH),
        "utility_table_file": protocol.file_record(protocol.utility.TABLE_PATH),
        "oracle_controller_map": table["oracle_controller_map"],
        "source_files": source_records(),
        "data_policy": (
            "one frozen evidence dataset; controlled uniform versus "
            "switch-centered causal curricula; internal validation checkpoint "
            "selection; no utility holdout access"),
    }
    protocol.write_json_atomic(protocol.MANIFEST_PATH, manifest)
    protocol.load_manifest()
    for path in protocol.MODEL_ROOT.glob("variant_*.npz"):
        path.unlink()
    protocol.DATASET_PATH.unlink()
    protocol.TRAIN_STATE_META_PATH.unlink()
    print(
        "SWITCH-CENTERED SEQUENCE ROUTER COMPLETE: "
        f"variant={best_name} "
        f"gate={'PASS' if decision['gate']['passed'] else 'FAIL'}",
        flush=True,
    )


def main() -> None:
    protocol.configure()
    if protocol.MANIFEST_PATH.is_file():
        protocol.load_manifest()
        print(f"Complete switch-centered router exists: {protocol.MODEL_ROOT}")
        return
    protocol.MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    if protocol.TRAIN_STATE_META_PATH.is_file():
        dataset = _load_dataset()
        completed = load_state()
        print(
            "Resumed switch-centered curriculum after variants: "
            + ",".join(completed), flush=True)
    else:
        if protocol.DATASET_PATH.exists():
            raise RuntimeError("orphaned curriculum evidence without train state")
        dataset = collect_dataset()
        completed = {}
        save_state(completed, protocol.file_record(protocol.DATASET_PATH))
    table = protocol.utility.load_utility_table()
    for name, variant in protocol.VARIANTS.items():
        if name in completed:
            continue
        completed[name] = train_variant(name, variant, dataset, table)
        save_state(completed, protocol.file_record(protocol.DATASET_PATH))
    finalize(completed, dataset, table)


if __name__ == "__main__":
    main()
