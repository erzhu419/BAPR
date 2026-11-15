"""Train the causal polarity estimator on frozen exploratory controllers."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from jax_experiments.analysis import final_task_sweep
from jax_experiments.analysis import regime_polarity_headroom as exploratory
from jax_experiments.analysis import regime_polarity_posterior as protocol
from jax_experiments.analysis import regime_polarity_posterior_model as model_lib
from jax_experiments.common.checkpoint import load_checkpoint
from jax_experiments.common.logging import Logger
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.train import (
    _reset_eval_switch_schedule,
    _select_eval_switch_sequence,
    make_algo,
    make_env,
)


MODEL_SEED = 20_260_728
CHECKPOINT_ROOT = protocol.MODEL_ROOT / "checkpoints"
TRAIN_STATE_JSON = CHECKPOINT_ROOT / "train_state.json"
TRAIN_STATE_NPZ = CHECKPOINT_ROOT / "train_state.npz"


def _source_records() -> dict[str, dict[str, Any]]:
    paths = (
        Path(__file__).resolve(),
        Path(protocol.__file__).resolve(),
        Path(model_lib.__file__).resolve(),
        protocol.ROOT
        / "jax_experiments/networks/probabilistic_regime_context.py",
    )
    return {
        str(path.relative_to(protocol.ROOT)): protocol.file_record(path)
        for path in paths
    }


def _validate_bundle(source, seed: int, role: str) -> dict[str, Any]:
    directory = source.bundle_dir(protocol.ENV, role, seed)
    payload = source.read_json(directory / "bundle_manifest.json")
    required = {
        "checkpoints/params.pkl",
        "checkpoints/train_state.pkl",
        "logs/protocol_signature.json",
    }
    if (payload.get("schema") != source.BUNDLE_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("identity")
            != source.identity(protocol.ENV, role, seed)):
        raise ValueError(f"invalid source bundle: {directory}")
    records = payload.get("files") or {}
    if set(records) != required:
        raise ValueError(f"incomplete source bundle: {directory}")
    for relative, expected in records.items():
        path = directory / relative
        if not path.is_file() or source.file_record(path) != expected:
            raise ValueError(f"source bundle changed: {path}")
    expected_checkpoint = {
        "iteration": source.FINAL_ITERATION,
        "next_iteration": source.MAX_ITERS,
        "total_steps": source.FINAL_TOTAL_STEPS,
        "update_count": source.FINAL_UPDATE_COUNT,
        "algo": "regime_sac",
    }
    if payload.get("checkpoint") != expected_checkpoint:
        raise ValueError(f"wrong source checkpoint budget: {directory}")
    return payload


def _load_controller(source, seed: int, role: str):
    _validate_bundle(source, seed, role)
    directory = source.bundle_dir(protocol.ENV, role, seed)
    config = final_task_sweep.load_config(directory)
    if (config.algo != "regime_sac"
            or config.env_type != "stochastic_mode"
            or config.stochastic_mode_family != protocol.FAMILY
            or config.regime_context_source != role):
        raise ValueError(f"wrong source controller config: {directory}")
    env = make_env(config, seed_offset=0)
    tasks = env.sample_tasks(config.task_num)
    agent = make_algo(config.algo, env.obs_dim, env.act_dim, config)
    agent.set_task_metadata(tasks)
    replay = ReplayBuffer(
        env.obs_dim, env.act_dim, capacity=1,
        belief_dim=agent.belief_dim)
    with tempfile.TemporaryDirectory() as log_dir:
        logger = Logger(log_dir)
        next_iteration, total_steps = load_checkpoint(
            str(directory / "checkpoints"),
            agent,
            replay,
            logger,
            config.algo,
            load_replay_buffer=False,
        )
    if (int(next_iteration) != source.MAX_ITERS
            or int(total_steps) != source.FINAL_TOTAL_STEPS):
        raise ValueError(f"stale source controller: {directory}")
    if hasattr(env, "close"):
        env.close()
    return config, agent, nnx.state(agent.policy, nnx.Param)


def _sequence(transitions, labels, identity: dict[str, Any]):
    obs, act, rew, next_obs, done = (
        np.asarray(value) for value in transitions)
    labels = np.asarray(labels, dtype=np.int32)
    if labels.shape != (len(obs),):
        raise ValueError("transition labels have the wrong length")
    return {
        "obs": obs.astype(np.float32),
        "act": act.astype(np.float32),
        "rew": rew.astype(np.float32),
        "next_obs": next_obs.astype(np.float32),
        "done": done.astype(np.float32),
        "mode_id": labels,
        "identity": identity,
    }


def _policy_context(role: str, mode: int) -> np.ndarray:
    if role == "robust":
        return np.zeros((len(protocol.MODES),), dtype=np.float32)
    return np.eye(len(protocol.MODES), dtype=np.float32)[int(mode)]


def _collect_stationary(
    source,
    controller_seeds: tuple[int, ...],
    event_seeds: tuple[int, ...],
    steps: int,
) -> list[dict[str, Any]]:
    sequences = []
    for controller_seed in controller_seeds:
        for role in protocol.ROLES:
            config, agent, policy_state = _load_controller(
                source, controller_seed, role)
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
                            _policy_context(role, mode)),
                    )
                    sequences.append(_sequence(
                        transitions,
                        np.full((steps,), mode, dtype=np.int32),
                        {
                            "kind": "stationary",
                            "controller_seed": controller_seed,
                            "role": role,
                            "event_seed": event_seed,
                            "mode": mode,
                        },
                    ))
                    if hasattr(env, "close"):
                        env.close()
                    print(
                        "posterior data stationary "
                        f"seed={controller_seed} role={role} "
                        f"event={event_seed} mode={mode}",
                        flush=True,
                    )
    return sequences


def _collect_switching(
    source,
    controller_seeds: tuple[int, ...],
    event_seeds: tuple[int, ...],
    episodes: int,
) -> list[dict[str, Any]]:
    sequences = []
    chunks = protocol.MAX_EPISODE_STEPS // protocol.DWELL_STEPS
    if chunks * protocol.DWELL_STEPS != protocol.MAX_EPISODE_STEPS:
        raise ValueError("switching horizon must divide into complete dwells")
    for controller_seed in controller_seeds:
        for role in protocol.ROLES:
            config, agent, policy_state = _load_controller(
                source, controller_seed, role)
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
                        env, switch_tasks, run_config,
                        protocol.DWELL_STEPS)
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
                                _policy_context(role, mode)),
                            continue_state=chunk > 0,
                        )
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
                            "controller_seed": controller_seed,
                            "role": role,
                            "event_seed": event_seed,
                            "episode": episode,
                        },
                    ))
                if hasattr(env, "close"):
                    env.close()
                print(
                    "posterior data switching "
                    f"seed={controller_seed} role={role} event={event_seed}",
                    flush=True,
                )
    return sequences


def _build_update(model):
    graphdef = nnx.graphdef(model)
    optimizer = optax.chain(
        optax.clip_by_global_norm(10.0),
        optax.adam(protocol.MODEL_CONFIG["learning_rate"]),
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
            # The conditional generative objective owns the mean model. The
            # classifier is diagnostic only, so labels cannot manufacture
            # likelihood separation by pushing non-target heads apart.
            loss = jnp.mean(predictive)
            variance_targets, variance_counts = (
                current.empirical_variance_targets(
                    residual_sq, flat[-1]))
            return loss, (
                jnp.mean(classifier),
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


def _sample_batch(
    sequences: list[dict[str, Any]],
    length: int,
    rng: np.random.Generator,
):
    slices = []
    for sequence in sequences:
        maximum = len(sequence["obs"]) - length
        if maximum < 0:
            raise ValueError("training sequence is shorter than context length")
        start = int(rng.integers(0, maximum + 1))
        slices.append(slice(start, start + length))
    values = [
        jnp.asarray(np.stack([
            sequence[key][row]
            for sequence, row in zip(sequences, slices)
        ]))
        for key in ("obs", "act", "rew", "next_obs")
    ]
    labels = np.stack([
        sequence["mode_id"][row]
        for sequence, row in zip(sequences, slices)
    ])
    return (*values, jax.nn.one_hot(jnp.asarray(labels), len(protocol.MODES)))


def _sequence_emissions(emission_fn, params, sequence):
    values = emission_fn(
        params,
        jnp.asarray(sequence["obs"]),
        jnp.asarray(sequence["act"]),
        jnp.asarray(sequence["rew"]),
        jnp.asarray(sequence["next_obs"]),
    )
    return tuple(np.asarray(value) for value in values)


def _aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_names = (
        "mode_accuracy",
        "mean_true_probability",
        "negative_log_likelihood",
        "brier_score",
        "expected_calibration_error",
        "mean_normalized_entropy",
    )
    delays = [
        int(delay)
        for row in rows
        for delay in row["switch_delays"]
    ]
    return {
        **{
            name: float(np.mean([float(row[name]) for row in rows]))
            for name in scalar_names
        },
        "switch_count": len(delays),
        "median_switch_delay": (
            float(np.median(delays)) if delays else 0.0),
        "p90_switch_delay": (
            float(np.percentile(delays, 90)) if delays else 0.0),
    }


def _select_filter(prepared):
    candidates = []
    for candidate in protocol.filter_candidates():
        by_kind = {"stationary": [], "switching": []}
        for sequence, log_likelihoods in prepared:
            before, _ = protocol.causal_posteriors(
                log_likelihoods, candidate)
            row = protocol.posterior_metrics(
                before, sequence["mode_id"])
            by_kind[sequence["identity"]["kind"]].append(row)
        stationary = _aggregate_metrics(by_kind["stationary"])
        switching = _aggregate_metrics(by_kind["switching"])
        gate = (
            stationary["mode_accuracy"] >= protocol.MIN_MODE_ACCURACY
            and switching["mode_accuracy"] >= protocol.MIN_MODE_ACCURACY
            and switching["median_switch_delay"]
            <= protocol.MAX_MEDIAN_SWITCH_DELAY
            and switching["p90_switch_delay"]
            <= protocol.MAX_P90_SWITCH_DELAY
            and switching["brier_score"] <= protocol.MAX_BRIER_SCORE
        )
        score = (
            0.35 * stationary["mode_accuracy"]
            + 0.65 * switching["mode_accuracy"]
            - 0.25 * switching["brier_score"]
            - 0.002 * switching["median_switch_delay"]
            - 0.001 * switching["p90_switch_delay"]
        )
        candidates.append({
            "filter_config": candidate.to_dict(),
            "stationary": stationary,
            "switching": switching,
            "validation_gate_pass": bool(gate),
            "selection_score": float(score),
        })
    candidates.sort(key=lambda row: (
        not row["validation_gate_pass"],
        -row["selection_score"],
        row["filter_config"]["hazard_rate"],
        row["filter_config"]["evidence_scale"],
    ))
    return candidates[0], candidates


def _checkpoint_payload(
    update_index: int,
    model,
    optimizer_state,
    rng: np.random.Generator,
    history: list[dict[str, float]],
) -> None:
    metadata = protocol.save_parameter_state(
        TRAIN_STATE_NPZ,
        (nnx.state(model, nnx.Param), optimizer_state),
    )
    protocol.write_json_atomic(TRAIN_STATE_JSON, {
        "schema": "bapr.regime-polarity-posterior-train-state.v1",
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
            != "bapr.regime-polarity-posterior-train-state.v1"
            or payload.get("status") != "in_progress"
            or payload.get("model_config") != protocol.MODEL_CONFIG
            or payload.get("source_files") != _source_records()
            or payload.get("parameter_file")
            != protocol.file_record(TRAIN_STATE_NPZ)):
        raise ValueError("stale or incompatible posterior train state")
    state = protocol.load_parameter_state(
        TRAIN_STATE_NPZ,
        (nnx.state(model, nnx.Param), optimizer_state),
        payload["parameter_leaves"],
    )
    return payload, state


def _source_bundle_records():
    records = {}
    for source, split, seeds in (
        (exploratory, "train", protocol.TRAIN_CONTROLLER_SEEDS),
        (exploratory, "validation", protocol.VALIDATION_CONTROLLER_SEEDS),
    ):
        for seed in seeds:
            for role in protocol.ROLES:
                path = source.bundle_manifest(protocol.ENV, role, seed)
                _validate_bundle(source, seed, role)
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


def main() -> None:
    args = parse_args()
    if protocol.MODEL_MANIFEST.is_file():
        model_lib.load_model(17, 6)
        print(f"POLARITY POSTERIOR ALREADY COMPLETE: {protocol.MODEL_ROOT}")
        return
    if protocol.MODEL_ROOT.exists() and not args.resume:
        raise RuntimeError(
            f"partial posterior root requires --resume: {protocol.MODEL_ROOT}")
    protocol.MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    reference_config, reference_agent, _ = _load_controller(
        exploratory, protocol.TRAIN_CONTROLLER_SEEDS[0], "robust")
    del reference_config
    model = model_lib.make_model(
        reference_agent.obs_dim, reference_agent.act_dim, MODEL_SEED)
    update_fn, optimizer_state = _build_update(model)
    rng = np.random.default_rng(MODEL_SEED)
    history: list[dict[str, float]] = []
    start_update = 0
    restored = _restore_checkpoint(
        model, optimizer_state) if args.resume else None
    if restored is not None:
        payload, (params, optimizer_state) = restored
        nnx.update(model, params)
        start_update = int(payload["update_index"])
        history = list(payload.get("history") or [])
        rng.bit_generator.state = payload["rng_state"]
        print(f"Resumed posterior at update {start_update}", flush=True)
    else:
        _checkpoint_payload(0, model, optimizer_state, rng, history)

    train_sequences = (
        _collect_stationary(
            exploratory,
            protocol.TRAIN_CONTROLLER_SEEDS,
            protocol.TRAIN_EVENT_SEEDS,
            args.stationary_steps,
        )
        + _collect_switching(
            exploratory,
            protocol.TRAIN_CONTROLLER_SEEDS,
            protocol.TRAIN_EVENT_SEEDS,
            args.switching_episodes,
        )
    )
    validation_sequences = (
        _collect_stationary(
            exploratory,
            protocol.VALIDATION_CONTROLLER_SEEDS,
            protocol.VALIDATION_EVENT_SEEDS,
            args.stationary_steps,
        )
        + _collect_switching(
            exploratory,
            protocol.VALIDATION_CONTROLLER_SEEDS,
            protocol.VALIDATION_EVENT_SEEDS,
            args.switching_episodes,
        )
    )

    for update_index in range(start_update, args.updates):
        batch = _sample_batch(
            train_sequences, protocol.CONTEXT_LENGTH, rng)
        params, optimizer_state, metrics = update_fn(
            nnx.state(model, nnx.Param), optimizer_state, *batch)
        nnx.update(model, params)
        current = model.mode_variances()
        targets = metrics[-2]
        counts = metrics[-1][:, None]
        ema = protocol.MODEL_CONFIG["variance_ema"]
        updated = jnp.where(
            counts > 0,
            (1.0 - ema) * current + ema * targets,
            current,
        )
        model.mode_logvar_raw.value = model.calibrated_raw_from_variance(
            updated)
        completed = update_index + 1
        if completed == 1 or completed % 50 == 0:
            row = {
                "update": completed,
                "predictive_loss": float(metrics[0]),
                "classifier_diagnostic": float(metrics[1]),
                "variance_min": float(jnp.min(updated)),
                "variance_max": float(jnp.max(updated)),
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        if completed % args.checkpoint_interval == 0:
            _checkpoint_payload(
                completed, model, optimizer_state, rng, history)

    emission_fn = model_lib.build_emission_fn(model)
    params = nnx.state(model, nnx.Param)
    prepared = [
        (
            sequence,
            _sequence_emissions(
                emission_fn, params, sequence)[0],
        )
        for sequence in validation_sequences
    ]
    selected, candidates = _select_filter(prepared)
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
            "frozen robust and true-context oracle controllers; commanded "
            "actions only; mode labels used only by the conditional "
            "generative training objective"),
        "training_config": {
            "updates": args.updates,
            "stationary_steps": args.stationary_steps,
            "switching_episodes": args.switching_episodes,
            "context_length": protocol.CONTEXT_LENGTH,
        },
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
        "POLARITY POSTERIOR COMPLETE: "
        f"validation_gate={'PASS' if selected['validation_gate_pass'] else 'FAIL'} "
        f"filter={selected['filter_config']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
