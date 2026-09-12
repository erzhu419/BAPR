"""Finalize a trained router with vectorized validation selection."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from flax import nnx

from jax_experiments.analysis import bapr_v3_learned_control_router as protocol
from jax_experiments.analysis import (
    run_bapr_v3_independent_specialist_audit as specialist_audit,
)
from jax_experiments.analysis import (
    train_bapr_v3_learned_control_router as trainer,
)


def vectorized_candidate_metrics(prepared, controller_map):
    """Evaluate all filter/fallback candidates with one time loop per trace."""
    configs = list(trainer.candidate_configs())
    count = len(configs)
    hazard = np.asarray([row.hazard_rate for row in configs])[:, None]
    evidence_scale = np.asarray(
        [row.evidence_scale for row in configs])[:, None]
    confidence_threshold = np.asarray(
        [row.confidence_threshold for row in configs])
    margin_threshold = np.asarray(
        [row.margin_threshold for row in configs])
    min_history = np.asarray([row.min_history for row in configs])
    hysteresis = np.asarray([row.hysteresis_margin for row in configs])
    controller_ids = np.asarray([0, 2, 3], dtype=np.int32)
    mapping = np.asarray(tuple(controller_map), dtype=np.int32)
    by_kind = {"stationary": [], "switching": []}
    physical_accuracy = []

    for sequence_index, (sequence, log_likelihoods) in enumerate(prepared):
        evidence = np.asarray(log_likelihoods, dtype=np.float64)
        true_modes = np.asarray(sequence["mode_id"], dtype=np.int32)
        steps = len(true_modes)
        posterior = np.full((count, 4), 0.25, dtype=np.float64)
        previous = np.full((count,), -1, dtype=np.int32)
        decisions = np.empty((count, steps), dtype=np.int32)
        physical_correct = np.zeros((count,), dtype=np.float64)

        for step in range(steps):
            controller_prob = np.stack([
                posterior[:, 0],
                posterior[:, 1] + posterior[:, 2],
                posterior[:, 3],
            ], axis=1)
            top_index = np.argmax(controller_prob, axis=1)
            top_probability = controller_prob[
                np.arange(count), top_index]
            valid_previous = previous >= 0
            previous_index = np.where(
                previous == 0, 0, np.where(previous == 2, 1, 2))
            previous_probability = controller_prob[
                np.arange(count), previous_index]
            keep_previous = (
                valid_previous
                & (previous_probability >= confidence_threshold)
                & (previous_probability + hysteresis >= top_probability)
            )
            top_index = np.where(
                keep_previous, previous_index, top_index)
            top_probability = controller_prob[
                np.arange(count), top_index]
            competitors = controller_prob.copy()
            competitors[np.arange(count), top_index] = -np.inf
            second_probability = np.max(competitors, axis=1)
            eligible = (
                (step >= min_history)
                & (top_probability >= confidence_threshold)
                & (top_probability - second_probability >= margin_threshold)
            )
            selected = np.where(
                eligible, controller_ids[top_index], -1).astype(np.int32)
            decisions[:, step] = selected
            previous = selected

            switch_probability = hazard / 3.0
            prior = (
                (1.0 - hazard) * posterior
                + switch_probability * (1.0 - posterior))
            centered = evidence[step] - np.max(evidence[step])
            # The deployed model fixes evidence_clip=6.0.
            centered = np.maximum(centered, -6.0)
            logits = (
                np.log(np.clip(prior, 1e-12, 1.0))
                + evidence_scale * centered[None, :])
            logits -= np.max(logits, axis=1, keepdims=True)
            posterior = np.exp(logits)
            posterior /= np.sum(posterior, axis=1, keepdims=True)
            if step >= 32:
                physical_correct += (
                    np.argmax(posterior, axis=1) == true_modes[step])

        expected = mapping[true_modes]
        valid = np.arange(steps) >= 32
        adaptive = decisions[:, valid] >= 0
        correct = decisions[:, valid] == expected[None, valid]
        adaptive_count = np.sum(adaptive, axis=1)
        coverage = np.mean(adaptive, axis=1)
        conditional_accuracy = np.sum(
            adaptive & correct, axis=1) / np.maximum(adaptive_count, 1)
        wrong_rate = np.mean(adaptive & ~correct, axis=1)
        effective_accuracy = np.mean(correct, axis=1)

        switch_points = np.flatnonzero(true_modes[1:] != true_modes[:-1]) + 1
        delay_rows = []
        for switch in switch_points:
            later = switch_points[switch_points > switch]
            end = int(later[0]) if len(later) else steps
            delay = np.full((count,), end - switch, dtype=np.float64)
            unresolved = np.ones((count,), dtype=bool)
            wanted = int(expected[switch])
            for start in range(switch, max(switch, end - 8 + 1)):
                stable = np.all(
                    decisions[:, start:start + 8] == wanted, axis=1)
                found = unresolved & stable
                delay[found] = start - switch
                unresolved[found] = False
                if not np.any(unresolved):
                    break
            delay_rows.append(delay)
        median_delay = (
            np.median(np.stack(delay_rows, axis=1), axis=1)
            if delay_rows else np.zeros((count,), dtype=np.float64))
        by_kind[sequence["identity"]["kind"]].append({
            "coverage": coverage,
            "conditional_accuracy": conditional_accuracy,
            "wrong_route_rate": wrong_rate,
            "effective_accuracy": effective_accuracy,
            "median_switch_delay": median_delay,
        })
        physical_accuracy.append(
            physical_correct / max(steps - 32, 1))
        print(
            f"  vector validation sequence={sequence_index + 1}/"
            f"{len(prepared)} kind={sequence['identity']['kind']}",
            flush=True,
        )

    def aggregate(kind, name):
        return np.mean(np.stack([
            row[name] for row in by_kind[kind]
        ], axis=0), axis=0)

    stationary = {
        name: aggregate("stationary", name)
        for name in (
            "coverage", "conditional_accuracy", "wrong_route_rate",
            "effective_accuracy")
    }
    switching = {
        name: aggregate("switching", name)
        for name in (
            "coverage", "conditional_accuracy", "wrong_route_rate",
            "effective_accuracy", "median_switch_delay")
    }
    physical = np.mean(np.stack(physical_accuracy, axis=0), axis=0)
    records = []
    for index, config in enumerate(configs):
        gate = (
            stationary["coverage"][index] >= 0.60
            and switching["coverage"][index] >= 0.60
            and stationary["conditional_accuracy"][index] >= 0.90
            and switching["conditional_accuracy"][index] >= 0.90
            and stationary["wrong_route_rate"][index] <= 0.05
            and switching["wrong_route_rate"][index] <= 0.08
            and switching["median_switch_delay"][index] <= 100.0
        )
        score = (
            0.4 * stationary["effective_accuracy"][index]
            + 0.6 * switching["effective_accuracy"][index]
            - 2.0 * (
                stationary["wrong_route_rate"][index]
                + switching["wrong_route_rate"][index])
            - 0.001 * switching["median_switch_delay"][index]
        )
        records.append({
            "router_config": config.to_dict(),
            "stationary": {
                name: float(values[index])
                for name, values in stationary.items()
            },
            "switching": {
                name: float(values[index])
                for name, values in switching.items()
            },
            "physical_mode_accuracy": float(physical[index]),
            "validation_gate_pass": bool(gate),
            "selection_score": float(score),
        })
    records.sort(key=lambda row: (
        not row["validation_gate_pass"],
        -row["selection_score"],
        row["router_config"]["min_history"],
    ))
    return records


def main() -> None:
    protocol.configure()
    if protocol.MANIFEST_PATH.is_file():
        protocol.load_manifest()
        print(f"Complete learned router already exists: {protocol.MODEL_ROOT}")
        return
    state_payload = json.loads(
        trainer.TRAIN_STATE_META_PATH.read_text(encoding="utf-8"))
    training_config = state_payload["training_config"]
    if int(state_payload.get("update_index", -1)) != int(
            training_config["updates"]):
        raise RuntimeError("router estimator training is not complete")

    mapping, mapping_payload = protocol.control.load_controller_map()
    bundles = protocol.control.specialist_protocol.validate_family_bundles(
        protocol.FAMILY)
    config, agents, policy_states = specialist_audit._controller_policy_states(
        protocol.FAMILY)
    graphdef = nnx.graphdef(agents["robust"].policy)
    model = trainer.make_model(
        agents["robust"].obs_dim, agents["robust"].act_dim,
        seed=20260718)
    _, optimizer_state = trainer.build_update(
        model, float(training_config["learning_rate"]))
    restored = trainer.load_training_checkpoint(
        model, optimizer_state, training_config)
    if restored is None:
        raise RuntimeError("missing complete estimator training state")
    state_payload, (params, _) = restored
    nnx.update(model, params)

    validation_stationary = trainer.collect_stationary_sequences(
        config, graphdef, policy_states,
        protocol.VALIDATION_EVENT_SEEDS,
        int(training_config["validation_steps"]),
    )
    validation_switching = trainer.collect_switching_sequences(
        config, graphdef, policy_states,
        protocol.VALIDATION_EVENT_SEEDS,
        episodes_per_controller=2,
    )
    emission_fn = trainer.build_emission_fn(model)
    params = nnx.state(model, nnx.Param)
    validation_sequences = validation_stationary + validation_switching
    prepared = []
    for index, sequence in enumerate(validation_sequences):
        prepared.append((
            sequence,
            trainer.sequence_emissions(emission_fn, params, sequence),
        ))
        print(
            f"  emissions sequence={index + 1}/{len(validation_sequences)}",
            flush=True,
        )
    candidates = vectorized_candidate_metrics(prepared, mapping)
    selected = candidates[0]

    parameter_metadata = protocol.save_parameter_state(
        protocol.MODEL_PATH, params)
    bundle_root = protocol.control.specialist_protocol.family_bundle_root(
        protocol.FAMILY)
    finalizer_path = Path(__file__).resolve()
    source_files = dict(state_payload["source_files"])
    source_files[str(finalizer_path.relative_to(protocol.ROOT))] = (
        protocol.file_record(finalizer_path))
    manifest = {
        "schema": protocol.MODEL_SCHEMA,
        "status": "complete",
        "family": protocol.FAMILY,
        "env": protocol.ENV,
        "training_seed": 0,
        "controller_map": list(mapping),
        "controller_map_schema": mapping_payload["schema"],
        "controller_map_file": protocol.file_record(
            protocol.control.MAPPING_PATH),
        "train_event_seeds": list(protocol.TRAIN_EVENT_SEEDS),
        "validation_event_seeds": list(protocol.VALIDATION_EVENT_SEEDS),
        "holdout_event_seeds": list(protocol.HOLDOUT_EVENT_SEEDS),
        "behavior_controllers": list(protocol.BEHAVIOR_CONTROLLERS),
        "data_policy": (
            "stochastic actions from robust and selected specialist policies; "
            "commanded action only, never executed_action"),
        "training_config": training_config,
        "model_config": trainer.MODEL_CONFIG,
        "router_config": selected["router_config"],
        "validation_gate_pass": selected["validation_gate_pass"],
        "validation": selected,
        "validation_top_candidates": candidates[:10],
        "training_history": state_payload.get("history") or [],
        "parameter_file": protocol.file_record(protocol.MODEL_PATH),
        "parameter_leaves": parameter_metadata,
        "bundle_manifest_sha256": {
            name: protocol.sha256_file(
                bundle_root / name
                / protocol.control.specialist_protocol.BUNDLE_MANIFEST)
            for name in bundles
        },
        "source_files": source_files,
        "finalization": "vectorized-candidate-selection-v1",
    }
    protocol.write_json_atomic(protocol.MANIFEST_PATH, manifest)
    protocol.load_manifest()
    if trainer.TRAIN_STATE_META_PATH.exists():
        trainer.TRAIN_STATE_META_PATH.unlink()
    for path in protocol.MODEL_ROOT.glob("train_state_*.npz"):
        path.unlink()
    print(
        "LEARNED CONTROL ROUTER FINALIZED: "
        f"gate={'PASS' if selected['validation_gate_pass'] else 'FAIL'} "
        f"config={selected['router_config']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
