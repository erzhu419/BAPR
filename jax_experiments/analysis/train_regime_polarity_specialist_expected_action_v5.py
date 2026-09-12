"""Fine-tune expected-action system ID on independent specialist trajectories."""
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
from flax import nnx

from jax_experiments.analysis import (
    regime_polarity_expected_action_system_id as parent_protocol,
)
from jax_experiments.analysis import (
    regime_polarity_expected_action_system_id_model as parent_model,
)
from jax_experiments.analysis import (
    regime_polarity_source_headroom_v1 as source,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_model_v5 as model_lib,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_system_id_v5 as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_sticky_router_v3 as sticky_protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_source_controller_v1 as source_runner,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_router_audit_v1 as source_audit,
)
from jax_experiments.analysis import (
    train_regime_polarity_expected_action_system_id as parent_trainer,
)
from jax_experiments.analysis import (
    train_regime_polarity_inverse_system_id as filter_trainer,
)
from jax_experiments.train import (
    _reset_eval_switch_schedule,
    _select_eval_switch_sequence,
    make_env,
)


MODEL_SEED = 20_260_732
TRAIN_STATE_SCHEMA = (
    "bapr.regime-polarity-specialist-expected-action-train-state.v5"
)

# Reuse the proven optimizer and metric implementations against the v5 paths
# and model loader in this standalone process.
parent_trainer.protocol = protocol
parent_trainer.model_lib = model_lib


def _source_records() -> dict:
    paths = (
        Path(__file__).resolve(),
        Path(protocol.__file__).resolve(),
        Path(model_lib.__file__).resolve(),
        Path(sticky_protocol.__file__).resolve(),
        protocol.ROOT / "jax_experiments/networks/executed_action_inverse.py",
        protocol.ROOT / "jax_experiments/envs/brax_env.py",
        protocol.ROOT / "jax_experiments/envs/stochastic_mode_env.py",
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


def _source_bundle_records() -> dict:
    records = {}
    for split, seeds in (
        ("train", protocol.TRAIN_SOURCE_SEEDS),
        ("validation", protocol.VALIDATION_SOURCE_SEEDS),
    ):
        for seed in seeds:
            for role in protocol.STATIONARY_ARMS:
                source_runner.validate_bundle(role, seed)
                path = source.bundle_manifest(role, seed)
                records[f"{split}/{role}/seed_{seed}"] = (
                    protocol.file_record(path))
    return records


def _sequence(rows: dict[str, list], identity: dict) -> dict:
    obs = np.asarray(rows["obs"], dtype=np.float32)
    act = np.asarray(rows["act"], dtype=np.float32)
    labels = np.asarray(rows["mode_id"], dtype=np.int32)
    gains = protocol.mode_gain_vectors(act.shape[-1])
    return {
        "obs": obs,
        "act": act,
        "rew": np.asarray(rows["rew"], dtype=np.float32),
        "next_obs": np.asarray(rows["next_obs"], dtype=np.float32),
        "done": np.asarray(rows["done"], dtype=np.float32),
        "target_action": np.clip(
            act * gains[labels], -1.0, 1.0).astype(np.float32),
        "mode_id": labels,
        "identity": identity,
    }


def _select_action(stack, arm: str, observation, mode: int, router):
    if arm == "dynamic_oracle":
        return stack["actions"][f"specialist_{mode}"](observation)
    if arm == "sticky_confirm3_v4":
        return router.select_action(observation).action
    return stack["actions"][arm](observation)


def _collect_one(
    stack,
    arm: str,
    event_seed: int,
    *,
    fixed_mode: int | None,
    episode: int | None,
    steps: int,
) -> dict:
    config = copy.deepcopy(stack["config"])
    config.stochastic_mode_fixed_id = (
        -1 if fixed_mode is None else int(fixed_mode))
    env = make_env(config, seed_offset=int(event_seed) - int(config.seed))
    tasks = env.sample_tasks(len(protocol.MODES))
    router = None
    if arm == "sticky_confirm3_v4":
        router = sticky_protocol.StickySpecialistOption(
            stack["actions"]["robust_sac"],
            tuple(
                stack["actions"][f"specialist_{mode}"]
                for mode in protocol.MODES
            ),
            stack["estimator_factory"](),
            switch_confirmation_steps=3,
        )
    if fixed_mode is None:
        sequence, _ = _select_eval_switch_sequence(env, tasks, int(episode))
        _reset_eval_switch_schedule(
            env, sequence, config, protocol.DWELL_STEPS)
    else:
        env.set_nonstationary_para(tasks)
        env.set_task(tasks[int(fixed_mode)])
    observation = env.reset()
    if router is not None:
        router.reset()
    rows = {key: [] for key in (
        "obs", "act", "rew", "next_obs", "done", "mode_id")}
    try:
        for _ in range(int(steps)):
            mode = int(env.task_id_for_next_step())
            action = np.asarray(
                _select_action(stack, arm, observation, mode, router),
                dtype=np.float32,
            )
            next_observation, reward, done, info = env.step(action)
            if int(info["mode_used"]) != mode:
                raise RuntimeError("specialist estimator collection misaligned")
            if router is not None:
                router.observe_transition(
                    observation, action, reward, next_observation)
            rows["obs"].append(np.asarray(observation, dtype=np.float32))
            rows["act"].append(action)
            rows["rew"].append(float(reward))
            rows["next_obs"].append(
                np.asarray(next_observation, dtype=np.float32))
            rows["done"].append(float(done))
            rows["mode_id"].append(mode)
            observation = next_observation
            if done:
                observation = env.reset()
    finally:
        if hasattr(env, "close"):
            env.close()
    return _sequence(rows, {
        "kind": "stationary" if fixed_mode is not None else "switching",
        "source_seed": int(stack["config"].seed),
        "arm": arm,
        "event_seed": int(event_seed),
        "mode": None if fixed_mode is None else int(fixed_mode),
        "episode": episode,
    })


def _collect(source_seeds, event_seeds, stationary_steps, switching_episodes):
    sequences = []
    for source_seed in source_seeds:
        stack = source_audit._load_stack(source_seed)
        for event_seed in event_seeds:
            for arm in protocol.STATIONARY_ARMS:
                for mode in protocol.MODES:
                    sequences.append(_collect_one(
                        stack,
                        arm,
                        event_seed,
                        fixed_mode=mode,
                        episode=None,
                        steps=stationary_steps,
                    ))
                    print(
                        "specialist estimator stationary "
                        f"seed={source_seed} arm={arm} "
                        f"event={event_seed} mode={mode}",
                        flush=True,
                    )
            for arm in protocol.SWITCHING_ARMS:
                for episode in range(int(switching_episodes)):
                    sequences.append(_collect_one(
                        stack,
                        arm,
                        event_seed,
                        fixed_mode=None,
                        episode=episode,
                        steps=protocol.MAX_EPISODE_STEPS,
                    ))
                print(
                    "specialist estimator switching "
                    f"seed={source_seed} arm={arm} event={event_seed}",
                    flush=True,
                )
    return sequences


def _write_scheduler_checkpoint(update_index: int) -> None:
    payload = {
        "iteration": max(int(update_index) - 1, 0),
        "next_iteration": int(update_index),
        "total_steps": int(update_index),
        "algo": "specialist_expected_action_system_id_v5",
    }
    protocol.CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=".train_state.", suffix=".pkl",
        dir=protocol.CHECKPOINT_ROOT)
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
        "schema": TRAIN_STATE_SCHEMA,
        "status": "in_progress",
        "update_index": int(update_index),
        "parameter_file": protocol.file_record(protocol.TRAIN_STATE_NPZ),
        "parameter_leaves": metadata,
        "rng_state": rng.bit_generator.state,
        "model_config": protocol.MODEL_CONFIG,
        "source_files": _source_records(),
        "parent_model": _parent_model_records(),
        "history": history,
    })
    _write_scheduler_checkpoint(update_index)


def _restore_checkpoint(model, optimizer_state):
    if not protocol.TRAIN_STATE_JSON.is_file():
        return None
    payload = protocol.read_json(protocol.TRAIN_STATE_JSON)
    if (
        payload.get("schema") != TRAIN_STATE_SCHEMA
        or payload.get("status") != "in_progress"
        or payload.get("model_config") != protocol.MODEL_CONFIG
        or payload.get("source_files") != _source_records()
        or payload.get("parent_model") != _parent_model_records()
        or payload.get("parameter_file")
        != protocol.file_record(protocol.TRAIN_STATE_NPZ)
    ):
        raise ValueError("stale specialist estimator train state")
    state = protocol.load_parameter_state(
        protocol.TRAIN_STATE_NPZ,
        (nnx.state(model, nnx.Param), optimizer_state),
        payload["parameter_leaves"],
    )
    return payload, state


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=protocol.TRAIN_UPDATES)
    parser.add_argument(
        "--stationary-steps", type=int, default=protocol.STATIONARY_STEPS)
    parser.add_argument(
        "--switching-episodes", type=int,
        default=protocol.SWITCHING_EPISODES)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if min(
        args.updates,
        args.stationary_steps,
        args.switching_episodes,
        args.checkpoint_interval,
    ) <= 0:
        parser.error("all training controls must be positive")
    return args


def smoke() -> None:
    stack = source_audit._load_stack(protocol.TRAIN_SOURCE_SEEDS[0])
    stationary = _collect_one(
        stack,
        "robust_sac",
        protocol.TRAIN_EVENT_SEEDS[0],
        fixed_mode=0,
        episode=None,
        steps=2,
    )
    switching = _collect_one(
        stack,
        "sticky_confirm3_v4",
        protocol.TRAIN_EVENT_SEEDS[0],
        fixed_mode=None,
        episode=0,
        steps=8,
    )
    dataset = parent_trainer._flatten([stationary, switching])
    model, _, _, _, _ = parent_model.load_model(17, 6)
    update_fn, optimizer_state = parent_trainer._build_update(model)
    batch = parent_trainer._sample_batch(
        dataset, np.random.default_rng(MODEL_SEED))
    params, _, metrics = update_fn(
        nnx.state(model, nnx.Param), optimizer_state, *batch)
    leaves = jax.tree.leaves(params)
    if not all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves):
        raise ValueError("specialist estimator smoke produced nonfinite params")
    if not all(np.isfinite(float(value)) for value in metrics):
        raise ValueError("specialist estimator smoke produced nonfinite metrics")
    print(
        "SPECIALIST EXPECTED-ACTION SMOKE COMPLETE: "
        f"rows={len(dataset['obs'])} loss={float(metrics[0]):.6f}",
        flush=True,
    )


def main():
    args = parse_args()
    if args.smoke:
        smoke()
        return
    if protocol.MODEL_MANIFEST.is_file():
        reference = source_audit._load_stack(protocol.TRAIN_SOURCE_SEEDS[0])
        controller = source_audit.source_audit._load_controller(
            "robust_sac", protocol.TRAIN_SOURCE_SEEDS[0])
        model_lib.load_model(
            controller["agent"].obs_dim, controller["agent"].act_dim)
        del reference
        print(
            "SPECIALIST EXPECTED-ACTION MODEL ALREADY COMPLETE: "
            f"{protocol.MODEL_ROOT}",
            flush=True,
        )
        return
    if protocol.MODEL_ROOT.exists() and not args.resume:
        raise RuntimeError(
            f"partial specialist estimator root requires --resume: "
            f"{protocol.MODEL_ROOT}")
    protocol.MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    parent, _, _, _, parent_manifest = parent_model.load_model(17, 6)
    model = parent
    update_fn, optimizer_state = parent_trainer._build_update(model)
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
            f"Resumed specialist estimator at update {start_update}",
            flush=True,
        )
    else:
        _checkpoint_payload(0, model, optimizer_state, rng, history)

    train_sequences = _collect(
        protocol.TRAIN_SOURCE_SEEDS,
        protocol.TRAIN_EVENT_SEEDS,
        args.stationary_steps,
        args.switching_episodes,
    )
    validation_sequences = _collect(
        protocol.VALIDATION_SOURCE_SEEDS,
        protocol.VALIDATION_EVENT_SEEDS,
        args.stationary_steps,
        args.switching_episodes,
    )
    train_dataset = parent_trainer._flatten(train_sequences)

    for update_index in range(start_update, args.updates):
        batch = parent_trainer._sample_batch(train_dataset, rng)
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
    train_predictions = parent_trainer._prediction_rows(
        prediction_fn, params, train_sequences)
    validation_predictions = parent_trainer._prediction_rows(
        prediction_fn, params, validation_sequences)
    residual_variance = parent_trainer._residual_variance(train_predictions)
    gains = jnp.asarray(
        protocol.mode_gain_vectors(6), dtype=jnp.float32)
    evidence_fn = model_lib.build_evidence_fn(model)
    prepared = [
        (
            sequence,
            filter_trainer._sequence_evidence(
                evidence_fn,
                params,
                sequence,
                gains,
                jnp.asarray(residual_variance, dtype=jnp.float32),
            )[0],
        )
        for sequence in validation_sequences
    ]
    selected, candidates = filter_trainer._select_filter(prepared)
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
        "train_source_seeds": list(protocol.TRAIN_SOURCE_SEEDS),
        "validation_source_seeds": list(protocol.VALIDATION_SOURCE_SEEDS),
        "audit_source_seeds": list(protocol.AUDIT_SOURCE_SEEDS),
        "train_event_seeds": list(protocol.TRAIN_EVENT_SEEDS),
        "validation_event_seeds": list(protocol.VALIDATION_EVENT_SEEDS),
        "audit_event_seeds": list(protocol.AUDIT_EVENT_SEEDS),
        "stationary_arms": list(protocol.STATIONARY_ARMS),
        "switching_arms": list(protocol.SWITCHING_ARMS),
        "training_data_policy": (
            "true mode is used only offline to construct the supervised "
            "executed-action target; deployed inference consumes observation, "
            "commanded action, and next observation only"
        ),
        "initialization": _parent_model_records(),
        "training_config": {
            "updates": int(args.updates),
            "batch_size_per_head": protocol.BATCH_SIZE,
            "stationary_steps": int(args.stationary_steps),
            "switching_episodes": int(args.switching_episodes),
        },
        "mode_gain_vectors": np.asarray(gains).tolist(),
        "residual_variance": residual_variance.tolist(),
        "training_inverse_metrics": parent_trainer._inverse_metrics(
            train_predictions),
        "validation_inverse_metrics": parent_trainer._inverse_metrics(
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
        "parent_model_manifest": parent_manifest,
    }
    protocol.write_json_atomic(protocol.MODEL_MANIFEST, manifest)
    model_lib.load_model(17, 6)
    if protocol.CHECKPOINT_ROOT.exists():
        shutil.rmtree(protocol.CHECKPOINT_ROOT)
    print(
        "SPECIALIST EXPECTED-ACTION MODEL COMPLETE: "
        f"validation_gate="
        f"{'PASS' if selected['validation_gate_pass'] else 'FAIL'} "
        f"filter={selected['filter_config']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
