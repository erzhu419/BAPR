"""Train closed-loop, return-aware polarity policy-compression students."""
from __future__ import annotations

import argparse
import copy
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from jax_experiments.analysis import (
    regime_polarity_policy_distillation_control_model as model_lib,
)
from jax_experiments.analysis import (
    regime_polarity_policy_distillation_control_v2 as protocol,
)
from jax_experiments.analysis import (
    train_regime_polarity_policy_distillation as base_trainer,
)
from jax_experiments.train import (
    _reset_eval_switch_schedule,
    _select_eval_switch_sequence,
    make_env,
)


# Reuse the proven action-causal collector, but bind its runtime globals to the
# v2 protocol and the frozen combined-ten teacher used here.
base_trainer.protocol = protocol
base_trainer.model_lib = model_lib
DatasetCollector = base_trainer.DatasetCollector
collect_dataset = base_trainer.collect_dataset
_one_hot = base_trainer._one_hot
_zero_context = base_trainer._zero_context


def _with_unit_weights(dataset: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    output = {key: np.asarray(value) for key, value in dataset.items()}
    output["weight"] = np.ones((len(output["obs"]),), dtype=np.float32)
    return output


def _append(left, right):
    if set(left) != set(right):
        raise ValueError("policy-compression dataset columns changed")
    return {
        key: np.concatenate([left[key], right[key]], axis=0)
        for key in left
    }


def _jax_batch(dataset, indices=None):
    keys = ("obs", "context", "target_action", "weight")
    if indices is None:
        return {key: jnp.asarray(dataset[key]) for key in keys}
    return {key: jnp.asarray(dataset[key][indices]) for key in keys}


def _copy_tree(tree):
    return jax.tree.map(lambda value: jnp.array(value), tree)


def sample_weights(
    teacher_returns,
    student_returns,
    modes,
    target_actions,
    student_actions,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return normalized regret, switch-age, and disagreement weights."""
    teacher_mean = float(np.mean(np.asarray(teacher_returns, dtype=np.float64)))
    student_mean = float(np.mean(np.asarray(student_returns, dtype=np.float64)))
    denominator = max(abs(teacher_mean), 500.0)
    regret = float(np.clip(
        (teacher_mean - student_mean) / denominator, 0.0, 1.5))

    modes = np.asarray(modes, dtype=np.int32)
    ages = np.empty((len(modes),), dtype=np.float32)
    age = 0
    horizon = int(protocol.MAX_EPISODE_STEPS)
    for index, mode in enumerate(modes):
        if index % horizon == 0 or mode != modes[index - 1]:
            age = 0
        ages[index] = age
        age += 1
    switch_urgency = np.exp(
        -ages / float(protocol.MODEL_CONFIG["switch_weight_tau"]))
    disagreement = np.mean(np.square(
        np.asarray(student_actions, dtype=np.float32)
        - np.asarray(target_actions, dtype=np.float32)), axis=-1)
    disagreement = np.clip(
        disagreement
        / float(protocol.MODEL_CONFIG["action_disagreement_normalizer"]),
        0.0,
        2.0,
    )
    weights = (
        1.0
        + float(protocol.MODEL_CONFIG["return_regret_scale"]) * regret
        + float(protocol.MODEL_CONFIG["switch_weight_scale"]) * switch_urgency
        + float(protocol.MODEL_CONFIG["action_disagreement_scale"])
        * disagreement
    )
    weights = np.clip(
        weights, 1.0, float(protocol.MODEL_CONFIG["max_sample_weight"]))
    weights = np.asarray(weights / np.mean(weights), dtype=np.float32)
    if (weights.shape != (len(modes),)
            or not np.all(np.isfinite(weights))
            or not np.isclose(float(np.mean(weights)), 1.0, atol=1e-5)):
        raise ValueError("invalid return-aware policy-compression weights")
    return weights, {
        "teacher_return_mean": teacher_mean,
        "student_return_mean": student_mean,
        "normalized_regret": regret,
        "weight_min": float(np.min(weights)),
        "weight_mean": float(np.mean(weights)),
        "weight_max": float(np.max(weights)),
        "action_disagreement_mean": float(np.mean(disagreement)),
    }


def _switching_returns(
    teacher,
    estimator,
    event_seed: int,
    episodes: int,
    arm: str,
    student_action=None,
) -> dict[str, Any]:
    config = teacher.config
    run_config = copy.deepcopy(config)
    run_config.stochastic_mode_fixed_id = -1
    env = make_env(
        run_config,
        seed_offset=int(event_seed) - int(config.seed),
    )
    tasks = env.sample_tasks(len(protocol.MODES))
    returns = []
    terminations = []
    try:
        for episode in range(int(episodes)):
            sequence, _ = _select_eval_switch_sequence(env, tasks, episode)
            _reset_eval_switch_schedule(
                env, sequence, run_config, protocol.DWELL_STEPS)
            observation = env.reset()
            estimator_state = estimator.initial_state()
            episode_return = 0.0
            terminated = False
            for _ in range(protocol.MAX_EPISODE_STEPS):
                mode = int(env.task_id_for_next_step())
                posterior = estimator.probabilities(estimator_state)
                if arm == "teacher":
                    action = model_lib.reduced_action(
                        teacher, "oracle", observation, posterior)
                elif arm == "student" and student_action is not None:
                    action = np.asarray(
                        student_action(observation, posterior),
                        dtype=np.float32,
                    )
                else:
                    raise ValueError(f"unknown control-validation arm {arm!r}")
                next_observation, reward, done, info = env.step(action)
                if int(info["mode_used"]) != mode:
                    raise RuntimeError(
                        "control-validation mode was not action-causal")
                estimator_state, _, _, _ = estimator.step(
                    estimator_state,
                    observation,
                    action,
                    reward,
                    next_observation,
                )
                episode_return += float(reward)
                observation = next_observation
                if done:
                    terminated = True
                    observation = env.reset()
            returns.append(float(episode_return))
            terminations.append(float(terminated))
    finally:
        if hasattr(env, "close"):
            env.close()
    return {
        "returns": returns,
        "return_mean": float(np.mean(returns)),
        "terminated_rate": float(np.mean(terminations)),
    }


def control_validation(teacher, estimator, student_action) -> dict[str, Any]:
    events = []
    student_values = []
    teacher_values = []
    termination_values = []
    for event_seed in protocol.CONTROL_VALIDATION_EVENT_SEEDS:
        student = _switching_returns(
            teacher,
            estimator,
            event_seed,
            protocol.CONTROL_VALIDATION_SWITCHING_EPISODES,
            "student",
            student_action,
        )
        reference = _switching_returns(
            teacher,
            estimator,
            event_seed,
            protocol.CONTROL_VALIDATION_SWITCHING_EPISODES,
            "teacher",
        )
        student_values.extend(student["returns"])
        teacher_values.extend(reference["returns"])
        termination_values.append(student["terminated_rate"])
        events.append({
            "event_seed": int(event_seed),
            "student": student,
            "teacher": reference,
        })
    student_mean = float(np.mean(student_values))
    teacher_mean = float(np.mean(teacher_values))
    terminated_rate = float(np.mean(termination_values))
    score = student_mean - (
        float(protocol.MODEL_CONFIG["termination_penalty"])
        * terminated_rate)
    return {
        "events": events,
        "student_return_mean": student_mean,
        "teacher_return_mean": teacher_mean,
        "student_teacher_gap": student_mean - teacher_mean,
        "student_terminated_rate": terminated_rate,
        "selection_score": float(score),
    }


def collect_dagger_increment(
    variant: str,
    collector: DatasetCollector,
    teacher,
    estimator,
    student_action,
    student_batch_action,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    combined = None
    event_metrics = []
    stationary_rows = 0
    switching_rows = 0
    for event_seed in protocol.DAGGER_EVENT_SEEDS:
        stationary = _with_unit_weights(collector.stationary(
            event_seed,
            "student",
            protocol.TRAIN_STATIONARY_EPISODES,
            student_action,
        ))
        switching = _with_unit_weights(collector.switching(
            event_seed,
            "student",
            protocol.DAGGER_SWITCHING_EPISODES,
            student_action,
        ))
        stationary_rows += len(stationary["obs"])
        switching_rows += len(switching["obs"])
        metrics: dict[str, Any] = {"event_seed": int(event_seed)}
        if variant == "mode_heads_return":
            student_eval = _switching_returns(
                teacher,
                estimator,
                event_seed,
                protocol.DAGGER_SWITCHING_EPISODES,
                "student",
                student_action,
            )
            teacher_eval = _switching_returns(
                teacher,
                estimator,
                event_seed,
                protocol.DAGGER_SWITCHING_EPISODES,
                "teacher",
            )
            predicted = np.asarray(student_batch_action(
                switching["obs"], switching["context"]), dtype=np.float32)
            switching["weight"], weight_metrics = sample_weights(
                teacher_eval["returns"],
                student_eval["returns"],
                switching["mode"],
                switching["target_action"],
                predicted,
            )
            metrics.update(weight_metrics)
        else:
            metrics.update({
                "teacher_return_mean": None,
                "student_return_mean": None,
                "normalized_regret": 0.0,
                "weight_min": 1.0,
                "weight_mean": 1.0,
                "weight_max": 1.0,
                "action_disagreement_mean": None,
            })
        current = _append(stationary, switching)
        combined = current if combined is None else _append(combined, current)
        event_metrics.append(metrics)
        print(
            f"compression DAgger variant={variant} event={event_seed} "
            f"rows={len(current['obs'])} weight_max={metrics['weight_max']:.3f}",
            flush=True,
        )
    if combined is None:
        raise ValueError("no closed-loop DAgger rows collected")
    return combined, {
        "stationary_rows": int(stationary_rows),
        "switching_rows": int(switching_rows),
        "total_rows": int(len(combined["obs"])),
        "events": event_metrics,
    }


def build_optimizer(model):
    graphdef = nnx.graphdef(model)
    config = protocol.MODEL_CONFIG
    optimizer = optax.chain(
        optax.clip_by_global_norm(float(config["gradient_clip"])),
        optax.adamw(
            float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        ),
    )
    clip = float(config["pre_tanh_clip"])
    pre_weight = float(config["pre_tanh_loss_weight"])

    def loss_fn(params, batch):
        current = nnx.merge(graphdef, params)
        mean, _ = current(batch["obs"], batch["context"])
        prediction = jnp.tanh(mean)
        target = batch["target_action"]
        weights = batch["weight"]
        denominator = jnp.maximum(jnp.sum(weights), 1e-6)
        action_per_row = jnp.mean(jnp.square(prediction - target), axis=-1)
        target_pre_tanh = jnp.arctanh(jnp.clip(target, -clip, clip))
        pre_tanh_per_row = jnp.mean(
            jnp.square(mean - target_pre_tanh), axis=-1)
        action_loss = jnp.sum(weights * action_per_row) / denominator
        pre_tanh_loss = jnp.sum(weights * pre_tanh_per_row) / denominator
        loss = action_loss + pre_weight * pre_tanh_loss
        return loss, (action_loss, pre_tanh_loss)

    @jax.jit
    def update(params, opt_state, batch):
        (loss, metrics), gradients = jax.value_and_grad(
            loss_fn, has_aux=True)(params, batch)
        updates, next_opt_state = optimizer.update(gradients, opt_state, params)
        return (
            optax.apply_updates(params, updates),
            next_opt_state,
            (loss,) + metrics,
        )

    @jax.jit
    def evaluate(params, batch):
        return loss_fn(params, batch)

    return optimizer, update, evaluate


def train_phase(
    model,
    initial_params,
    dataset,
    validation,
    seed: int,
    updates_count: int,
    phase_index: int,
):
    optimizer, update, evaluate = build_optimizer(model)
    params = _copy_tree(initial_params)
    opt_state = optimizer.init(params)
    rng = np.random.default_rng(
        int(seed) * 100_000 + int(phase_index) * 1_000 + 20260801)
    batch_size = int(protocol.MODEL_CONFIG["batch_size"])
    interval = int(protocol.MODEL_CONFIG["validation_interval"])
    validation_batch = _jax_batch(validation)
    best_params = _copy_tree(params)
    best_loss = float("inf")
    best_step = 0
    final_metrics = None
    for step in range(1, int(updates_count) + 1):
        indices = rng.integers(0, len(dataset["obs"]), size=batch_size)
        params, opt_state, metrics = update(
            params, opt_state, _jax_batch(dataset, indices))
        if step % interval == 0 or step == int(updates_count):
            validation_loss, validation_metrics = evaluate(
                params, validation_batch)
            value = float(validation_loss)
            final_metrics = {
                "train_total_loss": float(metrics[0]),
                "train_action_loss": float(metrics[1]),
                "train_pre_tanh_loss": float(metrics[2]),
                "validation_total_loss": value,
                "validation_action_loss": float(validation_metrics[0]),
                "validation_pre_tanh_loss": float(validation_metrics[1]),
            }
            if value < best_loss:
                best_loss = value
                best_step = int(step)
                best_params = _copy_tree(params)
            print(
                f"compression phase={phase_index} step={step}/{updates_count} "
                f"val={value:.6f} best={best_loss:.6f}",
                flush=True,
            )
    if final_metrics is None or not np.all(np.isfinite(
            np.asarray(list(final_metrics.values()), dtype=np.float64))):
        raise ValueError("non-finite closed-loop compression metrics")
    return best_params, {
        "phase_index": int(phase_index),
        "updates": int(updates_count),
        "dataset_rows": int(len(dataset["obs"])),
        "sample_weight_min": float(np.min(dataset["weight"])),
        "sample_weight_mean": float(np.mean(dataset["weight"])),
        "sample_weight_max": float(np.max(dataset["weight"])),
        "best_validation_loss": float(best_loss),
        "best_step": int(best_step),
        "final_metrics": final_metrics,
    }


def _save_dataset(path: Path, dataset: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **dataset)
        handle.flush()
        os.fsync(handle.fileno())


def _load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        output = {key: np.asarray(archive[key]) for key in archive.files}
    expected = {"obs", "context", "target_action", "mode", "weight"}
    if set(output) != expected:
        raise ValueError("checkpoint dataset columns changed")
    size = len(output["obs"])
    if (size <= 0 or any(len(value) != size for value in output.values())
            or not all(np.all(np.isfinite(value)) for value in output.values())):
        raise ValueError("invalid checkpoint dataset")
    return output


def _save_phase_checkpoint(
    variant: str,
    student_seed: int,
    phase_index: int,
    model,
    current_params,
    selected_params,
    increment,
    phase_metrics,
    collection_metrics,
    selected_phase: int,
    selected_score: float,
) -> None:
    root = protocol.work_dir(variant, student_seed)
    destination = root / f"phase_{int(phase_index)}"
    if destination.exists():
        raise ValueError(f"refusing to overwrite phase checkpoint {destination}")
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp.", dir=root))
    try:
        current_leaves = protocol.save_parameter_state(
            temporary / "current_params.npz", current_params)
        selected_leaves = protocol.save_parameter_state(
            temporary / "selected_params.npz", selected_params)
        _save_dataset(temporary / "dataset_increment.npz", increment)
        manifest = {
            "schema": protocol.CHECKPOINT_SCHEMA,
            "status": "complete",
            "identity": protocol.model_identity(variant, student_seed),
            "phase_index": int(phase_index),
            "current_parameter_file": protocol.file_record(
                temporary / "current_params.npz"),
            "current_parameter_leaves": current_leaves,
            "selected_parameter_file": protocol.file_record(
                temporary / "selected_params.npz"),
            "selected_parameter_leaves": selected_leaves,
            "dataset_increment_file": protocol.file_record(
                temporary / "dataset_increment.npz"),
            "phase_metrics": phase_metrics,
            "collection_metrics": collection_metrics,
            "selected_phase": int(selected_phase),
            "selected_control_validation_score": float(selected_score),
            "source_bundles": model_lib.source_records(variant),
            "frozen_estimator_manifest": protocol.file_record(
                protocol.ensemble.final.MODEL_MANIFEST),
            "frozen_estimator_parameters": protocol.file_record(
                protocol.ensemble.final.MODEL_PATH),
        }
        protocol.write_json_atomic(
            temporary / "checkpoint_manifest.json", manifest)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _load_phase_checkpoints(variant: str, student_seed: int, model):
    root = protocol.work_dir(variant, student_seed)
    if not root.exists():
        return None
    numbered = sorted(
        int(path.name.split("_", 1)[1])
        for path in root.glob("phase_[0-9]*")
        if path.is_dir()
    )
    if not numbered:
        return None
    if numbered != list(range(numbered[-1] + 1)):
        raise ValueError(f"non-contiguous compression checkpoints in {root}")
    template = nnx.state(model, nnx.Param)
    dataset = None
    manifests = []
    for phase_index in numbered:
        directory = root / f"phase_{phase_index}"
        manifest = protocol.read_json(directory / "checkpoint_manifest.json")
        if (manifest.get("schema") != protocol.CHECKPOINT_SCHEMA
                or manifest.get("status") != "complete"
                or manifest.get("identity")
                != protocol.model_identity(variant, student_seed)
                or int(manifest.get("phase_index", -1)) != phase_index
                or manifest.get("source_bundles")
                != model_lib.source_records(variant)
                or manifest.get("frozen_estimator_manifest")
                != protocol.file_record(protocol.ensemble.final.MODEL_MANIFEST)
                or manifest.get("frozen_estimator_parameters")
                != protocol.file_record(protocol.ensemble.final.MODEL_PATH)
                or manifest.get("current_parameter_file")
                != protocol.file_record(directory / "current_params.npz")
                or manifest.get("selected_parameter_file")
                != protocol.file_record(directory / "selected_params.npz")
                or manifest.get("dataset_increment_file")
                != protocol.file_record(directory / "dataset_increment.npz")):
            raise ValueError(f"invalid phase checkpoint {directory}")
        increment = _load_dataset(directory / "dataset_increment.npz")
        dataset = increment if dataset is None else _append(dataset, increment)
        manifests.append(manifest)
    latest_dir = root / f"phase_{numbered[-1]}"
    latest = manifests[-1]
    current = protocol.load_parameter_state(
        latest_dir / "current_params.npz",
        template,
        latest["current_parameter_leaves"],
    )
    selected = protocol.load_parameter_state(
        latest_dir / "selected_params.npz",
        template,
        latest["selected_parameter_leaves"],
    )
    return {
        "latest_phase": numbered[-1],
        "dataset": dataset,
        "current_params": current,
        "selected_params": selected,
        "selected_phase": int(latest["selected_phase"]),
        "selected_score": float(
            latest["selected_control_validation_score"]),
        "phase_metrics": [manifest["phase_metrics"] for manifest in manifests],
        "collection_metrics": [
            manifest["collection_metrics"] for manifest in manifests],
    }


def validate_model(variant: str, student_seed: int) -> dict[str, Any]:
    teacher = model_lib.load_teacher(variant)
    _, _, manifest = model_lib.load_student(
        variant, student_seed, teacher.obs_dim, teacher.act_dim)
    phases = manifest.get("training_phases") or []
    selected_phase = int(manifest.get("selected_phase", -1))
    if (manifest.get("source_bundles") != model_lib.source_records(variant)
            or manifest.get("frozen_estimator_manifest")
            != protocol.file_record(protocol.ensemble.final.MODEL_MANIFEST)
            or manifest.get("frozen_estimator_parameters")
            != protocol.file_record(protocol.ensemble.final.MODEL_PATH)
            or len(phases) != 1 + protocol.DAGGER_ROUNDS
            or selected_phase not in range(len(phases))
            or not np.isfinite(float(
                manifest.get("selected_control_validation_score", np.nan)))):
        raise ValueError("closed-loop distilled student provenance changed")
    return manifest


def run(variant: str, student_seed: int, resume: bool = False) -> None:
    variant = protocol.require_variant(variant)
    student_seed = protocol.require_student_seed(student_seed)
    destination = protocol.model_dir(variant, student_seed)
    if (destination / "model_manifest.json").is_file():
        try:
            validate_model(variant, student_seed)
        except (KeyError, OSError, TypeError, ValueError):
            pass
        else:
            print(f"CONTROL STUDENT ALREADY COMPLETE: {destination}")
            return

    protocol.assert_split_integrity()
    protocol.ensemble.validate_frozen_estimator()
    teacher = model_lib.load_teacher(variant)
    estimator = model_lib.make_estimator(teacher.obs_dim, teacher.act_dim)
    collector = DatasetCollector(teacher, estimator)
    student = model_lib.make_student(
        variant, teacher.obs_dim, teacher.act_dim, student_seed)
    student_apply = model_lib.build_student_action(student)
    student_batch_apply = model_lib.build_student_batch_action(student)

    restored = _load_phase_checkpoints(variant, student_seed, student) \
        if resume else None
    if restored is None:
        if protocol.work_dir(variant, student_seed).exists() and not resume:
            shutil.rmtree(protocol.work_dir(variant, student_seed))
        initial, initial_counts = collect_dataset(
            collector,
            protocol.TRAIN_EVENT_SEEDS,
            ("oracle_teacher", "learned_teacher", "robust_member"),
            protocol.TRAIN_STATIONARY_EPISODES,
            protocol.TRAIN_SWITCHING_EPISODES,
        )
        dataset = _with_unit_weights(initial)
        current_params = nnx.state(student, nnx.Param)
        selected_params = None
        selected_phase = -1
        selected_score = float("-inf")
        phases = []
        collections = []
        next_phase = 0
        initial_collection = {
            "kind": "initial",
            "behavior_rows": initial_counts,
            "total_rows": int(len(dataset["obs"])),
        }
    else:
        dataset = restored["dataset"]
        current_params = restored["current_params"]
        selected_params = restored["selected_params"]
        selected_phase = restored["selected_phase"]
        selected_score = restored["selected_score"]
        phases = restored["phase_metrics"]
        collections = restored["collection_metrics"]
        next_phase = int(restored["latest_phase"]) + 1
        initial_collection = collections[0]
        print(
            f"CONTROL STUDENT RESUME variant={variant} seed={student_seed} "
            f"after_phase={next_phase - 1} rows={len(dataset['obs'])}",
            flush=True,
        )

    validation_raw, validation_counts = collect_dataset(
        collector,
        protocol.SUPERVISED_VALIDATION_EVENT_SEEDS,
        ("learned_teacher", "robust_member"),
        protocol.TRAIN_STATIONARY_EPISODES,
        protocol.TRAIN_SWITCHING_EPISODES,
    )
    validation = _with_unit_weights(validation_raw)

    for phase_index in range(next_phase, 1 + protocol.DAGGER_ROUNDS):
        if phase_index == 0:
            increment = dataset
            collection_metrics = initial_collection
            updates = int(protocol.MODEL_CONFIG["initial_updates"])
        else:
            frozen_params = current_params

            def student_action(observation, context, state=frozen_params):
                return np.asarray(student_apply(
                    state,
                    jnp.asarray(observation, dtype=jnp.float32),
                    jnp.asarray(context, dtype=jnp.float32),
                ), dtype=np.float32)

            def student_batch_action(observations, contexts, state=frozen_params):
                return np.asarray(student_batch_apply(
                    state,
                    jnp.asarray(observations, dtype=jnp.float32),
                    jnp.asarray(contexts, dtype=jnp.float32),
                ), dtype=np.float32)

            increment, collection_metrics = collect_dagger_increment(
                variant,
                collector,
                teacher,
                estimator,
                student_action,
                student_batch_action,
            )
            collection_metrics["kind"] = "dagger"
            collection_metrics["round_index"] = int(phase_index - 1)
            dataset = _append(dataset, increment)
            updates = int(protocol.MODEL_CONFIG["dagger_updates_per_round"])

        current_params, phase_metrics = train_phase(
            student,
            current_params,
            dataset,
            validation,
            student_seed,
            updates,
            phase_index,
        )

        def candidate_action(observation, context, state=current_params):
            return np.asarray(student_apply(
                state,
                jnp.asarray(observation, dtype=jnp.float32),
                jnp.asarray(context, dtype=jnp.float32),
            ), dtype=np.float32)

        control = control_validation(teacher, estimator, candidate_action)
        phase_metrics["control_validation"] = control
        score = float(control["selection_score"])
        if (selected_params is None or score > selected_score
                or (np.isclose(score, selected_score)
                    and phase_metrics["best_validation_loss"]
                    < phases[selected_phase]["best_validation_loss"])):
            selected_params = _copy_tree(current_params)
            selected_phase = int(phase_index)
            selected_score = score
        phases.append(phase_metrics)
        collections.append(collection_metrics)
        _save_phase_checkpoint(
            variant,
            student_seed,
            phase_index,
            student,
            current_params,
            selected_params,
            increment,
            phase_metrics,
            collection_metrics,
            selected_phase,
            selected_score,
        )
        print(
            f"compression phase={phase_index} control={score:.3f} "
            f"selected_phase={selected_phase} selected={selected_score:.3f}",
            flush=True,
        )

    if selected_params is None or len(phases) != 1 + protocol.DAGGER_ROUNDS:
        raise ValueError("closed-loop compression training did not finish all phases")
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp.", dir=destination.parent))
    try:
        parameter_leaves = protocol.save_parameter_state(
            temporary / "student_params.npz", selected_params)
        manifest = {
            "schema": protocol.MODEL_SCHEMA,
            "status": "complete",
            "identity": protocol.model_identity(variant, student_seed),
            "source_bundles": model_lib.source_records(variant),
            "frozen_estimator_manifest": protocol.file_record(
                protocol.ensemble.final.MODEL_MANIFEST),
            "frozen_estimator_parameters": protocol.file_record(
                protocol.ensemble.final.MODEL_PATH),
            "parameter_file": protocol.file_record(
                temporary / "student_params.npz"),
            "parameter_leaves": parameter_leaves,
            "selected_phase": int(selected_phase),
            "selected_control_validation_score": float(selected_score),
            "selected_control_validation": phases[selected_phase][
                "control_validation"],
            "dataset": {
                "final_rows": int(len(dataset["obs"])),
                "collections": collections,
                "validation_rows": int(len(validation["obs"])),
                "validation_behavior_rows": validation_counts,
            },
            "training_phases": phases,
        }
        protocol.write_json_atomic(
            temporary / "model_manifest.json", manifest)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_model(variant, student_seed)
    print(f"CONTROL STUDENT COMPLETE: {destination}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=protocol.VARIANTS, required=True)
    parser.add_argument(
        "--student-seed", type=int, choices=protocol.STUDENT_SEEDS,
        required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.variant, args.student_seed, args.resume)


if __name__ == "__main__":
    main()
