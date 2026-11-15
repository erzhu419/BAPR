"""Train a causal probabilistic router for the frozen specialist bank."""
from __future__ import annotations

import argparse
import copy
import json
import os
from itertools import product
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from jax_experiments.analysis import bapr_v3_learned_control_router as protocol
from jax_experiments.analysis import (
    run_bapr_v3_independent_specialist_audit as specialist_audit,
)
from jax_experiments.networks.probabilistic_regime_context import (
    ProbabilisticRegimeContext,
)
from jax_experiments.train import (
    _reset_eval_switch_schedule,
    _select_eval_switch_sequence,
    make_env,
)


TRAIN_STATE_META_PATH = protocol.MODEL_ROOT / "train_state.json"
MODEL_CONFIG = {
    "hidden_dim": 128,
    "ensemble_size": 5,
    "variance_model": "mode_empirical",
    "variance_floor": 1e-4,
    "variance_ceiling": 0.5,
    "fixed_variance": 0.02,
    "reward_scale": 10.0,
    "delta_scale": 1.0,
    "mean_loss_weight": 1.0,
    "variance_ema": 0.05,
}


def source_records() -> dict[str, dict[str, Any]]:
    paths = (
        Path(__file__).resolve(),
        Path(protocol.__file__).resolve(),
        Path(ProbabilisticRegimeContext.__module__.replace(".", "/") + ".py"),
    )
    resolved = []
    for path in paths:
        if not path.is_absolute():
            path = protocol.ROOT / path
        resolved.append(path.resolve())
    return {
        str(path.relative_to(protocol.ROOT)): protocol.file_record(path)
        for path in resolved
    }


def make_model(obs_dim: int, act_dim: int, seed: int,
               router_config: protocol.RouterConfig | None = None):
    selected = router_config or protocol.RouterConfig(
        hazard_rate=0.002,
        evidence_scale=1.0,
        confidence_threshold=0.6,
        margin_threshold=0.05,
        min_history=16,
    )
    return ProbabilisticRegimeContext(
        obs_dim,
        act_dim,
        num_modes=4,
        hidden_dim=MODEL_CONFIG["hidden_dim"],
        ensemble_size=MODEL_CONFIG["ensemble_size"],
        mode="supervised",
        likelihood="probabilistic",
        reward_scale=MODEL_CONFIG["reward_scale"],
        delta_scale=MODEL_CONFIG["delta_scale"],
        min_history=selected.min_history,
        hazard_rate=selected.hazard_rate,
        evidence_scale=selected.evidence_scale,
        fixed_variance=MODEL_CONFIG["fixed_variance"],
        variance_model=MODEL_CONFIG["variance_model"],
        variance_floor=MODEL_CONFIG["variance_floor"],
        variance_ceiling=MODEL_CONFIG["variance_ceiling"],
        mean_loss_weight=MODEL_CONFIG["mean_loss_weight"],
        variance_loss_weight=0.0,
        variance_prior_weight=0.0,
        evidence_clip=6.0,
        surprise_threshold=1e9,
        posterior_decay=selected.posterior_decay,
        change_reset_threshold=selected.change_reset_threshold,
        change_reset_alpha=selected.change_reset_alpha,
        change_reset_mix=selected.change_reset_mix,
        change_cusum_threshold=selected.change_cusum_threshold,
        change_cusum_drift=selected.change_cusum_drift,
        rngs=nnx.Rngs(seed),
    )


def _as_sequence(transitions, mode_ids, identity: dict[str, Any]):
    obs, act, rew, next_obs, done = (
        np.asarray(value) for value in transitions)
    length = int(obs.shape[0])
    mode_ids = np.broadcast_to(
        np.asarray(mode_ids, dtype=np.int32), (length,)).copy()
    return {
        "obs": obs.astype(np.float32),
        "act": act.astype(np.float32),
        "rew": rew.astype(np.float32),
        "next_obs": next_obs.astype(np.float32),
        "done": done.astype(np.float32),
        "mode_id": mode_ids,
        "identity": identity,
    }


def collect_stationary_sequences(
    config,
    graphdef,
    policy_states: dict[str, Any],
    event_seeds: tuple[int, ...],
    steps: int,
) -> list[dict[str, Any]]:
    sequences = []
    # Compile one scan per fixed physics mode, then vary only policy params and
    # PRNG streams. Rebuilding this closure for every behavior/seed dominates
    # the actual rollout time and previously caused a long launch plateau.
    for mode in range(4):
        run_config = copy.deepcopy(config)
        run_config.stochastic_mode_fixed_id = mode
        env = make_env(run_config, seed_offset=100 * mode)
        tasks = env.sample_tasks(4)
        env.set_nonstationary_para(tasks)
        env.set_task(tasks[mode])
        env.build_rollout_fn(graphdef)
        for event_seed in event_seeds:
            for controller_index, controller in enumerate(
                    protocol.BEHAVIOR_CONTROLLERS):
                key = jax.random.PRNGKey(
                    event_seed * 10_000 + controller_index * 100 + mode)
                transitions, _ = env.rollout(
                    policy_states[controller], steps, key)
                sequences.append(_as_sequence(
                    transitions,
                    mode,
                    {
                        "kind": "stationary",
                        "event_seed": event_seed,
                        "controller": controller,
                        "mode": mode,
                    },
                ))
            print(
                f"  stationary data mode={mode} event_seed={event_seed} "
                f"sequences={len(sequences)}",
                flush=True,
            )
        if hasattr(env, "close"):
            env.close()
    return sequences


def collect_switching_sequences(
    config,
    graphdef,
    policy_states: dict[str, Any],
    event_seeds: tuple[int, ...],
    episodes_per_controller: int,
) -> list[dict[str, Any]]:
    sequences = []
    period = 500
    horizon = int(config.max_episode_steps)
    if horizon % period:
        raise ValueError("router switching collection requires a 500-step horizon")
    for event_seed in event_seeds:
        # One compiled switching scan serves every behavior controller for this
        # event stream; configure_eval_switching resets its causal mode clock.
        run_config = copy.deepcopy(config)
        run_config.stochastic_mode_fixed_id = -1
        env = make_env(run_config, seed_offset=event_seed)
        tasks = env.sample_tasks(4)
        env.build_rollout_fn(graphdef)
        for controller_index, controller in enumerate(
                protocol.BEHAVIOR_CONTROLLERS):
            for episode in range(episodes_per_controller):
                switch_tasks, _ = _select_eval_switch_sequence(
                    env, tasks, episode)
                _reset_eval_switch_schedule(
                    env, switch_tasks, run_config, period)
                parts = []
                labels = []
                for chunk in range(horizon // period):
                    mode = int(env.current_task_id)
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
                sequences.append(_as_sequence(
                    joined,
                    np.concatenate(labels),
                    {
                        "kind": "switching",
                        "event_seed": event_seed,
                        "controller": controller,
                        "episode": episode,
                    },
                ))
            print(
                f"  switching data event_seed={event_seed} "
                f"controller={controller} sequences={len(sequences)}",
                flush=True,
            )
        if hasattr(env, "close"):
            env.close()
    return sequences


def build_update(model, learning_rate: float):
    graphdef = nnx.graphdef(model)
    optimizer = optax.chain(
        optax.clip_by_global_norm(10.0),
        optax.adam(learning_rate),
    )
    optimizer_state = optimizer.init(nnx.state(model, nnx.Param))

    @jax.jit
    def update(params, opt_state, obs, act, rew, next_obs, target_modes):
        def loss_fn(candidate):
            current = nnx.merge(graphdef, candidate)
            flat = (
                obs.reshape((-1, obs.shape[-1])),
                act.reshape((-1, act.shape[-1])),
                rew.reshape((-1,)),
                next_obs.reshape((-1, next_obs.shape[-1])),
                target_modes.reshape((-1, target_modes.shape[-1])),
            )
            predictive, classifier, _, _, residual_sq = jax.vmap(
                current.supervised_statistics)(*flat)
            predictive_loss = jnp.mean(predictive)
            classifier_loss = jnp.mean(classifier)
            loss = predictive_loss + 2.0 * classifier_loss
            variance_targets, variance_counts = (
                current.empirical_variance_targets(
                    residual_sq, flat[-1]))
            return loss, (
                predictive_loss,
                classifier_loss,
                variance_targets,
                variance_counts,
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


def sample_chunks(sequences: list[dict[str, Any]], length: int,
                  rng: np.random.Generator):
    rows = []
    for sequence in sequences:
        maximum = int(sequence["obs"].shape[0]) - length
        if maximum < 0:
            raise ValueError("training sequence is shorter than context length")
        start = int(rng.integers(0, maximum + 1))
        rows.append(slice(start, start + length))
    values = []
    for key in ("obs", "act", "rew", "next_obs"):
        values.append(jnp.asarray(np.stack([
            sequence[key][row]
            for sequence, row in zip(sequences, rows)
        ])))
    mode_ids = np.stack([
        sequence["mode_id"][row]
        for sequence, row in zip(sequences, rows)
    ])
    targets = jax.nn.one_hot(jnp.asarray(mode_ids), 4)
    return (*values, targets)


def build_emission_fn(model):
    graphdef = nnx.graphdef(model)

    @jax.jit
    def emissions(params, obs, act, rew, next_obs):
        current = nnx.merge(graphdef, params)

        def one(o, a, r, no):
            _, statistics = current.transition_statistics(o, a, r, no)
            return statistics[0]

        return jax.vmap(one)(obs, act, rew, next_obs)

    return emissions


def sequence_emissions(emission_fn, params, sequence,
                       batch_size: int = 4096) -> np.ndarray:
    outputs = []
    size = int(sequence["obs"].shape[0])
    for start in range(0, size, batch_size):
        stop = min(size, start + batch_size)
        outputs.append(np.asarray(emission_fn(
            params,
            jnp.asarray(sequence["obs"][start:stop]),
            jnp.asarray(sequence["act"][start:stop]),
            jnp.asarray(sequence["rew"][start:stop]),
            jnp.asarray(sequence["next_obs"][start:stop]),
        )))
    return np.concatenate(outputs, axis=0)


def candidate_configs():
    for hazard, evidence, confidence, margin, history in product(
            (0.001, 0.002, 0.005),
            (0.25, 0.5, 1.0, 2.0),
            (0.50, 0.60, 0.70, 0.80),
            (0.02, 0.05, 0.10),
            (8, 16, 32)):
        yield protocol.RouterConfig(
            hazard_rate=hazard,
            evidence_scale=evidence,
            confidence_threshold=confidence,
            margin_threshold=margin,
            min_history=history,
        )


def evaluate_candidate(config, prepared, controller_map):
    by_kind: dict[str, list[dict[str, Any]]] = {
        "stationary": [], "switching": []}
    physical_correct = []
    for sequence, log_likelihoods in prepared:
        decisions, posteriors, _ = protocol.causal_route_trace(
            log_likelihoods, controller_map, config)
        metrics = protocol.routing_metrics(
            decisions, sequence["mode_id"], controller_map)
        by_kind[sequence["identity"]["kind"]].append(metrics)
        index = np.arange(len(posteriors)) >= 32
        physical_correct.append(float(np.mean(
            np.argmax(posteriors[index], axis=-1)
            == sequence["mode_id"][index])))

    def mean_metric(kind, name):
        values = [float(row[name]) for row in by_kind[kind]]
        return float(np.mean(values)) if values else 0.0

    stationary = {
        name: mean_metric("stationary", name)
        for name in (
            "coverage", "conditional_accuracy", "wrong_route_rate",
            "effective_accuracy")
    }
    switching = {
        name: mean_metric("switching", name)
        for name in (
            "coverage", "conditional_accuracy", "wrong_route_rate",
            "effective_accuracy", "median_switch_delay")
    }
    gate = (
        stationary["coverage"] >= 0.60
        and switching["coverage"] >= 0.60
        and stationary["conditional_accuracy"] >= 0.90
        and switching["conditional_accuracy"] >= 0.90
        and stationary["wrong_route_rate"] <= 0.05
        and switching["wrong_route_rate"] <= 0.08
        and switching["median_switch_delay"] <= 100.0
    )
    score = (
        0.4 * stationary["effective_accuracy"]
        + 0.6 * switching["effective_accuracy"]
        - 2.0 * (
            stationary["wrong_route_rate"]
            + switching["wrong_route_rate"])
        - 0.001 * switching["median_switch_delay"]
    )
    return {
        "router_config": config.to_dict(),
        "stationary": stationary,
        "switching": switching,
        "physical_mode_accuracy": float(np.mean(physical_correct)),
        "validation_gate_pass": bool(gate),
        "selection_score": float(score),
    }


def save_training_checkpoint(update_index, model, optimizer_state, rng,
                             training_config, history):
    combined = (nnx.state(model, nnx.Param), optimizer_state)
    state_path = (
        protocol.MODEL_ROOT / f"train_state_{int(update_index):06d}.npz")
    metadata = protocol.save_parameter_state(state_path, combined)
    payload = {
        "schema": "bapr.v3-control-router-train-state.v1",
        "status": "in_progress",
        "update_index": int(update_index),
        "parameter_path": state_path.name,
        "parameter_file": protocol.file_record(state_path),
        "parameter_leaves": metadata,
        "rng_state": rng.bit_generator.state,
        "training_config": training_config,
        "model_config": MODEL_CONFIG,
        "source_files": source_records(),
        "history": history,
    }
    protocol.write_json_atomic(TRAIN_STATE_META_PATH, payload)
    for old_path in protocol.MODEL_ROOT.glob("train_state_*.npz"):
        if old_path != state_path:
            old_path.unlink()


def load_training_checkpoint(model, optimizer_state, training_config):
    if not TRAIN_STATE_META_PATH.is_file():
        return None
    payload = json.loads(TRAIN_STATE_META_PATH.read_text(encoding="utf-8"))
    state_path = protocol.MODEL_ROOT / str(payload.get("parameter_path", ""))
    if (payload.get("schema") != "bapr.v3-control-router-train-state.v1"
            or payload.get("status") != "in_progress"
            or payload.get("training_config") != training_config
            or payload.get("model_config") != MODEL_CONFIG
            or payload.get("source_files") != source_records()
            or payload.get("parameter_file")
            != protocol.file_record(state_path)):
        raise ValueError("stale or incompatible learned-router train state")
    restored = protocol.load_parameter_state(
        state_path,
        (nnx.state(model, nnx.Param), optimizer_state),
        payload["parameter_leaves"],
    )
    return payload, restored


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=1500)
    parser.add_argument("--train-steps", type=int, default=4000)
    parser.add_argument("--validation-steps", type=int, default=2000)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.updates, args.train_steps, args.validation_steps,
           args.context_length, args.checkpoint_interval) <= 0:
        parser.error("all integer training controls must be positive")
    return args


def main() -> None:
    args = parse_args()
    protocol.configure()
    if protocol.MANIFEST_PATH.is_file():
        protocol.load_manifest()
        print(f"Complete learned router already exists: {protocol.MODEL_ROOT}")
        return
    if protocol.MODEL_ROOT.exists() and not args.resume:
        raise RuntimeError(
            f"partial router root requires --resume: {protocol.MODEL_ROOT}")
    protocol.MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    mapping, mapping_payload = protocol.control.load_controller_map()
    bundles = protocol.control.specialist_protocol.validate_family_bundles(
        protocol.FAMILY)
    config, agents, policy_states = specialist_audit._controller_policy_states(
        protocol.FAMILY)
    graphdef = nnx.graphdef(agents["robust"].policy)
    model = make_model(
        agents["robust"].obs_dim, agents["robust"].act_dim, seed=20260718)
    update_fn, optimizer_state = build_update(model, args.learning_rate)
    training_config = {
        "updates": args.updates,
        "train_steps": args.train_steps,
        "validation_steps": args.validation_steps,
        "context_length": args.context_length,
        "learning_rate": args.learning_rate,
        "checkpoint_interval": args.checkpoint_interval,
        "train_event_seeds": list(protocol.TRAIN_EVENT_SEEDS),
        "validation_event_seeds": list(protocol.VALIDATION_EVENT_SEEDS),
        "behavior_controllers": list(protocol.BEHAVIOR_CONTROLLERS),
    }

    rng = np.random.default_rng(20260718)
    history: list[dict[str, float]] = []
    start_update = 0
    restored = load_training_checkpoint(
        model, optimizer_state, training_config) if args.resume else None
    if restored is not None:
        payload, (params, optimizer_state) = restored
        nnx.update(model, params)
        start_update = int(payload["update_index"])
        history = list(payload.get("history") or [])
        rng.bit_generator.state = payload["rng_state"]
        print(f"Resumed router estimator at update {start_update}", flush=True)
    else:
        save_training_checkpoint(
            0, model, optimizer_state, rng, training_config, history)

    print("Collecting deterministic-seed router datasets", flush=True)
    train_sequences = collect_stationary_sequences(
        config, graphdef, policy_states,
        protocol.TRAIN_EVENT_SEEDS, args.train_steps)
    validation_stationary = collect_stationary_sequences(
        config, graphdef, policy_states,
        protocol.VALIDATION_EVENT_SEEDS, args.validation_steps)
    validation_switching = collect_switching_sequences(
        config, graphdef, policy_states,
        protocol.VALIDATION_EVENT_SEEDS, episodes_per_controller=2)

    for update_index in range(start_update, args.updates):
        batch = sample_chunks(
            train_sequences, args.context_length, rng)
        params = nnx.state(model, nnx.Param)
        params, optimizer_state, metrics = update_fn(
            params, optimizer_state, *batch)
        nnx.update(model, params)
        variance_targets = metrics[-2]
        variance_counts = metrics[-1]
        current = model.mode_variances()
        ema = MODEL_CONFIG["variance_ema"]
        updated = jnp.where(
            variance_counts[:, None] > 0,
            (1.0 - ema) * current + ema * variance_targets,
            current,
        )
        model.mode_logvar_raw.value = model.calibrated_raw_from_variance(
            updated)
        completed = update_index + 1
        if completed == 1 or completed % 50 == 0:
            row = {
                "update": completed,
                "loss": float(metrics[0]),
                "predictive_loss": float(metrics[1]),
                "classifier_loss": float(metrics[2]),
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        if completed % args.checkpoint_interval == 0:
            save_training_checkpoint(
                completed, model, optimizer_state, rng,
                training_config, history)

    emission_fn = build_emission_fn(model)
    params = nnx.state(model, nnx.Param)
    validation_sequences = validation_stationary + validation_switching
    prepared = [
        (sequence, sequence_emissions(emission_fn, params, sequence))
        for sequence in validation_sequences
    ]
    candidates = [
        evaluate_candidate(candidate, prepared, mapping)
        for candidate in candidate_configs()
    ]
    candidates.sort(key=lambda row: (
        not row["validation_gate_pass"],
        -row["selection_score"],
        row["router_config"]["min_history"],
    ))
    selected = candidates[0]

    parameter_metadata = protocol.save_parameter_state(
        protocol.MODEL_PATH, params)
    bundle_root = protocol.control.specialist_protocol.family_bundle_root(
        protocol.FAMILY)
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
        "model_config": MODEL_CONFIG,
        "router_config": selected["router_config"],
        "validation_gate_pass": selected["validation_gate_pass"],
        "validation": selected,
        "validation_top_candidates": candidates[:10],
        "training_history": history,
        "parameter_file": protocol.file_record(protocol.MODEL_PATH),
        "parameter_leaves": parameter_metadata,
        "bundle_manifest_sha256": {
            name: protocol.sha256_file(
                bundle_root / name
                / protocol.control.specialist_protocol.BUNDLE_MANIFEST)
            for name in bundles
        },
        "source_files": source_records(),
    }
    protocol.write_json_atomic(protocol.MANIFEST_PATH, manifest)
    protocol.load_manifest()
    if TRAIN_STATE_META_PATH.exists():
        TRAIN_STATE_META_PATH.unlink()
    for path in protocol.MODEL_ROOT.glob("train_state_*.npz"):
        path.unlink()
    print(
        "LEARNED CONTROL ROUTER COMPLETE: "
        f"gate={'PASS' if selected['validation_gate_pass'] else 'FAIL'} "
        f"config={selected['router_config']} output={protocol.MODEL_ROOT}",
        flush=True,
    )


if __name__ == "__main__":
    main()
