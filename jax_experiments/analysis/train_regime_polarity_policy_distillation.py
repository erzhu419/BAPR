"""Distill a frozen median policy ensemble with switch-matched DAgger."""
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
    regime_polarity_policy_distillation as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_policy_distillation_model as model_lib,
)
from jax_experiments.train import (
    _reset_eval_switch_schedule,
    _select_eval_switch_sequence,
    make_env,
)


def _empty_rows() -> dict[str, list[np.ndarray]]:
    return {key: [] for key in ("obs", "context", "target_action", "mode")}


def _finalize_rows(rows) -> dict[str, np.ndarray]:
    payload = {
        "obs": np.asarray(rows["obs"], dtype=np.float32),
        "context": np.asarray(rows["context"], dtype=np.float32),
        "target_action": np.asarray(rows["target_action"], dtype=np.float32),
        "mode": np.asarray(rows["mode"], dtype=np.int32),
    }
    size = len(payload["obs"])
    if size <= 0 or any(len(value) != size for value in payload.values()):
        raise ValueError("invalid policy-distillation dataset")
    if (payload["context"].shape[1:] != (len(protocol.MODES),)
            or not all(np.all(np.isfinite(value)) for value in payload.values())):
        raise ValueError("non-finite policy-distillation dataset")
    return payload


def _append(left, right):
    return {
        key: np.concatenate([left[key], right[key]], axis=0)
        for key in left
    }


def _one_hot(mode: int) -> np.ndarray:
    return np.eye(len(protocol.MODES), dtype=np.float32)[int(mode)]


def _zero_context() -> np.ndarray:
    return np.zeros((len(protocol.MODES),), dtype=np.float32)


class DatasetCollector:
    def __init__(self, teacher, estimator):
        self.teacher = teacher
        self.estimator = estimator

    def _action_and_target(
        self,
        behavior: str,
        observation,
        mode: int,
        posterior,
        member_index: int,
        student_action=None,
    ):
        if behavior == "oracle_teacher":
            context = _one_hot(mode)
            target = model_lib.reduced_action(
                self.teacher, "oracle", observation, context)
            return target, context, target
        context = np.asarray(posterior, dtype=np.float32)
        target = model_lib.reduced_action(
            self.teacher, "oracle", observation, context)
        if behavior == "learned_teacher":
            action = target
        elif behavior == "robust_member":
            action = model_lib.individual_action(
                self.teacher,
                "robust",
                member_index,
                observation,
                _zero_context(),
            )
        elif behavior == "student":
            if student_action is None:
                raise ValueError("student behavior requires a student action")
            action = np.asarray(
                student_action(observation, context), dtype=np.float32)
        else:
            raise ValueError(f"unknown collection behavior {behavior!r}")
        return action, context, target

    def _step_rows(
        self,
        env,
        behavior: str,
        horizon: int,
        member_index: int,
        rows,
        student_action=None,
        reset_estimator_on_done: bool = False,
    ):
        observation = env.reset()
        estimator_state = self.estimator.initial_state()
        for _ in range(int(horizon)):
            mode = int(env.task_id_for_next_step())
            posterior = self.estimator.probabilities(estimator_state)
            action, context, target = self._action_and_target(
                behavior,
                observation,
                mode,
                posterior,
                member_index,
                student_action,
            )
            rows["obs"].append(np.asarray(observation, dtype=np.float32))
            rows["context"].append(np.asarray(context, dtype=np.float32))
            rows["target_action"].append(
                np.asarray(target, dtype=np.float32))
            rows["mode"].append(mode)
            next_observation, reward, done, info = env.step(action)
            if int(info["mode_used"]) != mode:
                raise RuntimeError("distillation label was not action-causal")
            estimator_state, _, _, _ = self.estimator.step(
                estimator_state,
                observation,
                action,
                reward,
                next_observation,
            )
            observation = next_observation
            if done:
                observation = env.reset()
                if reset_estimator_on_done:
                    estimator_state = self.estimator.initial_state()
        return rows

    def stationary(
        self,
        event_seed: int,
        behavior: str,
        episodes: int,
        student_action=None,
    ):
        rows = _empty_rows()
        config = self.teacher.config
        member_index = int(event_seed) % len(self.teacher.controller_keys)
        for mode in protocol.MODES:
            run_config = copy.deepcopy(config)
            run_config.stochastic_mode_fixed_id = int(mode)
            env = make_env(
                run_config,
                seed_offset=int(event_seed) - int(config.seed),
            )
            tasks = env.sample_tasks(len(protocol.MODES))
            env.set_nonstationary_para(tasks)
            env.set_task(tasks[int(mode)])
            for _ in range(int(episodes)):
                self._step_rows(
                    env,
                    behavior,
                    protocol.MAX_EPISODE_STEPS,
                    member_index,
                    rows,
                    student_action,
                    reset_estimator_on_done=True,
                )
            if hasattr(env, "close"):
                env.close()
        return _finalize_rows(rows)

    def switching(
        self,
        event_seed: int,
        behavior: str,
        episodes: int,
        student_action=None,
    ):
        rows = _empty_rows()
        config = self.teacher.config
        run_config = copy.deepcopy(config)
        run_config.stochastic_mode_fixed_id = -1
        env = make_env(
            run_config,
            seed_offset=int(event_seed) - int(config.seed),
        )
        tasks = env.sample_tasks(len(protocol.MODES))
        for episode in range(int(episodes)):
            sequence, _ = _select_eval_switch_sequence(env, tasks, episode)
            _reset_eval_switch_schedule(
                env, sequence, run_config, protocol.DWELL_STEPS)
            member_index = (
                int(event_seed) + int(episode)
            ) % len(self.teacher.controller_keys)
            self._step_rows(
                env,
                behavior,
                protocol.MAX_EPISODE_STEPS,
                member_index,
                rows,
                student_action,
            )
        if hasattr(env, "close"):
            env.close()
        return _finalize_rows(rows)


def collect_dataset(
    collector: DatasetCollector,
    event_seeds,
    behaviors,
    stationary_episodes: int,
    switching_episodes: int,
    student_action=None,
):
    combined = None
    counts = {}
    for event_seed in event_seeds:
        for behavior in behaviors:
            pieces = []
            if int(stationary_episodes) > 0:
                pieces.append(collector.stationary(
                    int(event_seed), behavior, stationary_episodes,
                    student_action))
            if int(switching_episodes) > 0:
                pieces.append(collector.switching(
                    int(event_seed), behavior, switching_episodes,
                    student_action))
            current = pieces[0]
            for piece in pieces[1:]:
                current = _append(current, piece)
            combined = current if combined is None else _append(
                combined, current)
            counts[behavior] = counts.get(behavior, 0) + len(current["obs"])
            print(
                f"distillation data event={event_seed} "
                f"behavior={behavior} rows={len(current['obs'])}",
                flush=True,
            )
    if combined is None:
        raise ValueError("no distillation data collected")
    return combined, counts


def _jax_batch(dataset, indices=None):
    keys = ("obs", "context", "target_action")
    if indices is None:
        return {key: jnp.asarray(dataset[key]) for key in keys}
    return {key: jnp.asarray(dataset[key][indices]) for key in keys}


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
        action_loss = jnp.mean(jnp.square(prediction - target))
        target_pre_tanh = jnp.arctanh(jnp.clip(target, -clip, clip))
        pre_tanh_loss = jnp.mean(jnp.square(mean - target_pre_tanh))
        loss = action_loss + pre_weight * pre_tanh_loss
        return loss, (action_loss, pre_tanh_loss)

    @jax.jit
    def update(params, opt_state, batch):
        (loss, metrics), gradients = jax.value_and_grad(
            loss_fn, has_aux=True)(params, batch)
        updates, next_opt_state = optimizer.update(
            gradients, opt_state, params)
        next_params = optax.apply_updates(params, updates)
        return next_params, next_opt_state, (loss,) + metrics

    @jax.jit
    def evaluate(params, batch):
        return loss_fn(params, batch)

    return optimizer, update, evaluate


def _copy_tree(tree):
    return jax.tree.map(lambda value: jnp.array(value), tree)


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
            (validation_loss, validation_metrics) = evaluate(
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
                best_step = step
                best_params = _copy_tree(params)
            print(
                f"distillation phase={phase_index} step={step}/"
                f"{updates_count} val={value:.6f} best={best_loss:.6f}",
                flush=True,
            )
    if final_metrics is None or not np.all(np.isfinite(
            np.asarray(list(final_metrics.values()), dtype=np.float64))):
        raise ValueError("non-finite distillation optimization metrics")
    return best_params, {
        "phase_index": int(phase_index),
        "updates": int(updates_count),
        "dataset_rows": int(len(dataset["obs"])),
        "best_validation_loss": float(best_loss),
        "best_step": int(best_step),
        "final_metrics": final_metrics,
    }


def validate_model(group: str, student_seed: int) -> dict[str, Any]:
    teacher = model_lib.load_teacher(group)
    _, _, manifest = model_lib.load_student(
        group, student_seed, teacher.obs_dim, teacher.act_dim)
    if (manifest.get("source_bundles") != model_lib.source_records(group)
            or manifest.get("frozen_estimator_manifest")
            != protocol.file_record(protocol.ensemble.final.MODEL_MANIFEST)
            or manifest.get("frozen_estimator_parameters")
            != protocol.file_record(protocol.ensemble.final.MODEL_PATH)
            or len(manifest.get("training_phases") or [])
            != 1 + protocol.DAGGER_ROUNDS):
        raise ValueError("distilled student provenance changed")
    return manifest


def run(group: str, student_seed: int) -> None:
    group = protocol.require_teacher_group(group)
    student_seed = protocol.require_student_seed(student_seed)
    destination = protocol.model_dir(group, student_seed)
    if (destination / "model_manifest.json").is_file():
        try:
            validate_model(group, student_seed)
        except (KeyError, OSError, TypeError, ValueError):
            pass
        else:
            print(f"DISTILLED STUDENT ALREADY COMPLETE: {destination}")
            return

    protocol.ensemble.validate_frozen_estimator()
    teacher = model_lib.load_teacher(group)
    estimator = model_lib.make_estimator(teacher.obs_dim, teacher.act_dim)
    collector = DatasetCollector(teacher, estimator)
    dataset, initial_counts = collect_dataset(
        collector,
        protocol.TRAIN_EVENT_SEEDS,
        ("oracle_teacher", "learned_teacher", "robust_member"),
        protocol.TRAIN_STATIONARY_EPISODES,
        protocol.TRAIN_SWITCHING_EPISODES,
    )
    validation, validation_counts = collect_dataset(
        collector,
        protocol.VALIDATION_EVENT_SEEDS,
        ("learned_teacher", "robust_member"),
        protocol.TRAIN_STATIONARY_EPISODES,
        protocol.TRAIN_SWITCHING_EPISODES,
    )

    student = model_lib.make_student(
        teacher.obs_dim, teacher.act_dim, student_seed)
    params = nnx.state(student, nnx.Param)
    phases = []
    params, metrics = train_phase(
        student,
        params,
        dataset,
        validation,
        student_seed,
        int(protocol.MODEL_CONFIG["initial_updates"]),
        phase_index=0,
    )
    phases.append(metrics)

    student_apply = model_lib.build_student_action(student)
    dagger_counts = []
    for round_index in range(protocol.DAGGER_ROUNDS):
        current_params = params

        def student_action(observation, context, state=current_params):
            return np.asarray(student_apply(
                state,
                jnp.asarray(observation, dtype=jnp.float32),
                jnp.asarray(context, dtype=jnp.float32),
            ), dtype=np.float32)

        dagger, counts = collect_dataset(
            collector,
            protocol.DAGGER_EVENT_SEEDS,
            ("student",),
            protocol.TRAIN_STATIONARY_EPISODES,
            protocol.DAGGER_SWITCHING_EPISODES,
            student_action,
        )
        dagger_counts.append(counts)
        dataset = _append(dataset, dagger)
        params, metrics = train_phase(
            student,
            params,
            dataset,
            validation,
            student_seed,
            int(protocol.MODEL_CONFIG["dagger_updates_per_round"]),
            phase_index=round_index + 1,
        )
        phases.append(metrics)

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
            temporary / "student_params.npz", params)
        manifest = {
            "schema": protocol.MODEL_SCHEMA,
            "status": "complete",
            "identity": protocol.model_identity(group, student_seed),
            "source_bundles": model_lib.source_records(group),
            "frozen_estimator_manifest": protocol.file_record(
                protocol.ensemble.final.MODEL_MANIFEST),
            "frozen_estimator_parameters": protocol.file_record(
                protocol.ensemble.final.MODEL_PATH),
            "parameter_file": protocol.file_record(
                temporary / "student_params.npz"),
            "parameter_leaves": parameter_leaves,
            "dataset": {
                "final_rows": int(len(dataset["obs"])),
                "initial_behavior_rows": initial_counts,
                "dagger_behavior_rows": dagger_counts,
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
    validate_model(group, student_seed)
    print(f"DISTILLED STUDENT COMPLETE: {destination}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher-group", choices=protocol.TEACHER_GROUPS, required=True)
    parser.add_argument(
        "--student-seed", type=int, choices=protocol.STUDENT_SEEDS,
        required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.teacher_group, args.student_seed)


if __name__ == "__main__":
    main()
