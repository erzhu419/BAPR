"""Measure two-timescale headroom before learning a causal switch gate."""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.analysis import bapr_v3_sequence_router_dual as protocol
from jax_experiments.analysis import diagnose_bapr_v3_sequence_router as diagnostic
from jax_experiments.analysis import train_bapr_v3_sequence_router as v1_train
from jax_experiments.analysis import train_bapr_v3_sequence_router_v2 as curriculum


VALIDATION_INTERVAL = 100
BATCH_SIZE = 16
BURNIN = 8


def source_records():
    paths = (
        Path(__file__).resolve(),
        Path(protocol.__file__).resolve(),
        Path(curriculum.__file__).resolve(),
        protocol.ROOT / "jax_experiments/networks/causal_mode_filter.py",
    )
    return {
        str(path.relative_to(protocol.ROOT)): protocol.file_record(path)
        for path in paths
    }


def load_model(path, metadata):
    model = v1_train.make_sequence_model(seed=20260718)
    template = nnx.state(model, nnx.Param)
    params = protocol.emission.load_parameter_state(
        path, template, metadata)
    nnx.update(model, params)
    return model, params


def action_predictions(logits, oracle_map):
    posteriors = np.asarray(jax.nn.softmax(logits, axis=-1))
    mapping = np.asarray(oracle_map, dtype=np.int32)
    return np.stack([
        mapping[np.argmax(diagnostic.before_action(posterior), axis=1)]
        for posterior in posteriors
    ])


def _aggregate_decisions(decisions, labels, kind, oracle_map):
    rows = {"stationary": [], "full_cycle": []}
    for index in range(len(decisions)):
        key = "full_cycle" if int(kind[index]) else "stationary"
        rows[key].append(protocol.utility.routing_metrics(
            decisions[index], labels[index], oracle_map, burnin=32))
    return {
        key: diagnostic._aggregate_metrics(values)
        for key, values in rows.items()
    }


def window_gate_metrics(
    slow_logits,
    fast_logits,
    labels,
    kind,
    oracle_map,
    window,
):
    slow = action_predictions(slow_logits, oracle_map)
    fast = action_predictions(fast_logits, oracle_map)
    ages = curriculum.switch_age(labels)
    use_fast = (kind[:, None] == 1) & (ages >= 0) & (ages < int(window))
    decisions = np.where(use_fast, fast, slow)
    return _aggregate_decisions(decisions, labels, kind, oracle_map)


def correctness_oracle_metrics(
    slow_logits,
    fast_logits,
    labels,
    kind,
    oracle_map,
):
    slow = action_predictions(slow_logits, oracle_map)
    fast = action_predictions(fast_logits, oracle_map)
    expected = np.asarray(oracle_map, dtype=np.int32)[labels]
    use_fast = (fast == expected) & (slow != expected)
    return _aggregate_decisions(
        np.where(use_fast, fast, slow), labels, kind, oracle_map)


def assembly_key(metrics):
    return (
        metrics["full_cycle"]["action_accuracy"],
        metrics["stationary"]["action_accuracy"],
        -metrics["full_cycle"]["median_switch_delay"],
    )


def gate(metrics):
    stationary = metrics["stationary"]
    full_cycle = metrics["full_cycle"]
    checks = {
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
    checks["passed"] = all(checks.values())
    return checks


def variant_path(name):
    return protocol.MODEL_ROOT / f"variant_{name}.npz"


def save_state(completed, dataset_record, slow_manifest_record, slow_model_record):
    protocol.write_json_atomic(protocol.TRAIN_STATE_META_PATH, {
        "schema": protocol.TRAIN_STATE_SCHEMA,
        "status": "in_progress",
        "completed_variants": completed,
        "dataset_file": dataset_record,
        "slow_manifest_file": slow_manifest_record,
        "slow_model_file": slow_model_record,
        "fast_variants": protocol.FAST_VARIANTS,
        "windows": list(protocol.WINDOWS),
        "source_files": source_records(),
    })


def load_state(slow_manifest_record, slow_model_record):
    if not protocol.TRAIN_STATE_META_PATH.is_file():
        return {}
    payload = json.loads(protocol.TRAIN_STATE_META_PATH.read_text())
    if (payload.get("schema") != protocol.TRAIN_STATE_SCHEMA
            or payload.get("status") != "in_progress"
            or payload.get("dataset_file")
            != protocol.file_record(protocol.DATASET_PATH)
            or payload.get("slow_manifest_file") != slow_manifest_record
            or payload.get("slow_model_file") != slow_model_record
            or payload.get("fast_variants") != protocol.FAST_VARIANTS
            or payload.get("windows") != list(protocol.WINDOWS)
            or payload.get("source_files") != source_records()):
        raise ValueError("stale dual-router oracle training state")
    completed = payload.get("completed_variants") or {}
    for name, record in completed.items():
        if (name not in protocol.FAST_VARIANTS
                or record.get("parameter_file")
                != protocol.file_record(variant_path(name))):
            raise ValueError("dual-router fast checkpoint changed")
    return completed


def train_fast_variant(
    name,
    variant,
    dataset,
    table,
    slow_logits,
):
    model = v1_train.make_sequence_model(seed=20260718)
    update, optimizer_state = v1_train.build_update(
        model, float(variant["learning_rate"]), BURNIN)
    validation_fn = curriculum.build_validation_fn(model)
    rng = np.random.default_rng(20260718)
    ages = curriculum.switch_age(dataset["train_labels"])
    best_params = None
    best_assembly = None
    best_standalone = None
    best_window = None
    best_update = 0
    history = []
    for update_index in range(int(variant["updates"])):
        batch = curriculum.sample_curriculum_batch(
            dataset["train_evidence"], dataset["train_labels"],
            dataset["train_kind"], ages, variant, rng,
            batch_size=BATCH_SIZE)
        params = nnx.state(model, nnx.Param)
        params, optimizer_state, loss, accuracy = update(
            params, optimizer_state, *batch)
        nnx.update(model, params)
        completed = update_index + 1
        if completed % VALIDATION_INTERVAL:
            continue
        fast_logits = validation_fn(
            params, jnp.asarray(dataset["validation_evidence"]))
        standalone = curriculum.no_fallback_metrics(
            fast_logits, dataset["validation_labels"],
            dataset["validation_kind"], table["oracle_controller_map"])
        windows = {
            str(window): window_gate_metrics(
                slow_logits, fast_logits, dataset["validation_labels"],
                dataset["validation_kind"], table["oracle_controller_map"],
                window)
            for window in protocol.WINDOWS
        }
        window, assembly = max(
            windows.items(), key=lambda item: assembly_key(item[1]))
        row = {
            "update": completed,
            "loss": float(loss),
            "batch_accuracy": float(accuracy),
            "standalone": standalone,
            "best_window": int(window),
            "assembly": assembly,
        }
        history.append(row)
        if best_assembly is None or assembly_key(assembly) > assembly_key(
                best_assembly):
            best_params = jax.tree.map(lambda value: jnp.array(value), params)
            best_assembly = assembly
            best_standalone = standalone
            best_window = int(window)
            best_update = completed
        print(json.dumps({
            "variant": name,
            "update": completed,
            "loss": float(loss),
            "window": int(window),
            "assembly_full": assembly["full_cycle"]["action_accuracy"],
            "fast_full": standalone["full_cycle"]["action_accuracy"],
            "delay": assembly["full_cycle"]["median_switch_delay"],
        }, sort_keys=True), flush=True)
    if best_params is None:
        raise RuntimeError("fast curriculum produced no checkpoint")
    metadata = protocol.emission.save_parameter_state(
        variant_path(name), best_params)
    return {
        "config": variant,
        "best_update": best_update,
        "best_window": best_window,
        "best_assembly": best_assembly,
        "best_standalone": best_standalone,
        "history": history,
        "parameter_file": protocol.file_record(variant_path(name)),
        "parameter_leaves": metadata,
    }


def finalize(completed, dataset, table, slow_manifest, slow_params, slow_logits):
    best_name = max(
        completed,
        key=lambda name: assembly_key(completed[name]["best_assembly"]),
    )
    best = completed[best_name]
    fast_model, fast_params = load_model(
        variant_path(best_name), best["parameter_leaves"])
    fast_validation = curriculum.build_validation_fn(fast_model)(
        fast_params, jnp.asarray(dataset["validation_evidence"]))
    correctness = correctness_oracle_metrics(
        slow_logits, fast_validation, dataset["validation_labels"],
        dataset["validation_kind"], table["oracle_controller_map"])
    slow_metadata = protocol.emission.save_parameter_state(
        protocol.SLOW_MODEL_PATH, slow_params)
    fast_metadata = protocol.emission.save_parameter_state(
        protocol.FAST_MODEL_PATH, fast_params)
    passed = gate(best["best_assembly"])
    public = {
        name: {key: value for key, value in record.items()
               if key not in ("parameter_file", "parameter_leaves")}
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
        "sealed_holdout_event_seeds": list(protocol.HOLDOUT_EVENT_SEEDS),
        "fast_variants": protocol.FAST_VARIANTS,
        "windows": list(protocol.WINDOWS),
        "variant_results": public,
        "selected_fast_variant": best_name,
        "selected_window": best["best_window"],
        "selected_assembly": best["best_assembly"],
        "selected_gate": passed,
        "privileged_correctness_oracle": correctness,
        "slow_source_manifest_file": protocol.file_record(
            protocol.slow.MANIFEST_PATH),
        "slow_source_parameter_file": protocol.file_record(
            protocol.slow.MODEL_PATH),
        "slow_parameter_file": protocol.file_record(protocol.SLOW_MODEL_PATH),
        "slow_parameter_leaves": slow_metadata,
        "fast_parameter_file": protocol.file_record(protocol.FAST_MODEL_PATH),
        "fast_parameter_leaves": fast_metadata,
        "emission_manifest_file": protocol.file_record(
            protocol.emission.MANIFEST_PATH),
        "utility_table_file": protocol.file_record(protocol.utility.TABLE_PATH),
        "oracle_controller_map": table["oracle_controller_map"],
        "source_files": source_records(),
        "interpretation": (
            "privileged true-switch-time fixed-window ladder only; no learned "
            "gate and no utility holdout access"),
    }
    protocol.write_json_atomic(protocol.MANIFEST_PATH, manifest)
    for path in protocol.MODEL_ROOT.glob("variant_*.npz"):
        path.unlink()
    protocol.TRAIN_STATE_META_PATH.unlink()
    if not passed["passed"]:
        protocol.DATASET_PATH.unlink()
    print(
        "DUAL SEQUENCE ORACLE COMPLETE: "
        f"fast={best_name} window={best['best_window']} "
        f"gate={'PASS' if passed['passed'] else 'FAIL'}",
        flush=True,
    )


def main() -> None:
    protocol.configure()
    if protocol.MANIFEST_PATH.is_file():
        print(f"Complete dual sequence oracle exists: {protocol.MANIFEST_PATH}")
        return
    slow_manifest = protocol.slow.load_manifest()
    slow_manifest_record = protocol.file_record(protocol.slow.MANIFEST_PATH)
    slow_model_record = protocol.file_record(protocol.slow.MODEL_PATH)
    protocol.MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    if protocol.TRAIN_STATE_META_PATH.is_file():
        dataset = curriculum._load_dataset(protocol.DATASET_PATH)
        completed = load_state(slow_manifest_record, slow_model_record)
        print("Resumed dual oracle after: " + ",".join(completed), flush=True)
    else:
        if protocol.DATASET_PATH.exists():
            raise RuntimeError("orphaned dual-router evidence without state")
        dataset = curriculum.collect_dataset(protocol.DATASET_PATH)
        completed = {}
        save_state(
            completed, protocol.file_record(protocol.DATASET_PATH),
            slow_manifest_record, slow_model_record)
    table = protocol.utility.load_utility_table()
    slow_model, slow_params = load_model(
        protocol.slow.MODEL_PATH, slow_manifest["parameter_leaves"])
    slow_logits = curriculum.build_validation_fn(slow_model)(
        slow_params, jnp.asarray(dataset["validation_evidence"]))
    for name, variant in protocol.FAST_VARIANTS.items():
        if name in completed:
            continue
        completed[name] = train_fast_variant(
            name, variant, dataset, table, slow_logits)
        save_state(
            completed, protocol.file_record(protocol.DATASET_PATH),
            slow_manifest_record, slow_model_record)
    finalize(completed, dataset, table, slow_manifest, slow_params, slow_logits)


if __name__ == "__main__":
    main()
