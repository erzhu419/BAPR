"""Evaluate frozen mode-head contexts on one paired diagnostic event."""
from __future__ import annotations

import argparse
import copy
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from jax_experiments.analysis import (
    regime_polarity_policy_distillation_context_ablation_v1 as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_policy_distillation_control_model as model_lib,
)
from jax_experiments.analysis import (
    train_regime_polarity_policy_distillation_control_v2 as trainer,
)
from jax_experiments.train import (
    _reset_eval_switch_schedule,
    _select_eval_switch_sequence,
    make_env,
)


def _uses_estimator(arm: str) -> bool:
    return arm in ("student_learned", "teacher_learned_median")


def _student_context(arm: str, mode: int, posterior, event_seed: int):
    if arm == "student_learned":
        return posterior
    if arm == "student_oracle":
        return trainer._one_hot(mode)
    if arm == "student_uniform":
        return np.full((len(protocol.MODES),), 1.0 / len(protocol.MODES),
                       dtype=np.float32)
    if arm.startswith("student_fixed_"):
        return trainer._one_hot(int(arm.rsplit("_", 1)[1]))
    if arm == "student_cyclic":
        return trainer._one_hot((int(mode) + 1) % len(protocol.MODES))
    if arm == "student_shuffled":
        return trainer._one_hot(protocol.shuffled_mode_map(event_seed)[mode])
    raise ValueError(f"arm {arm!r} does not define a student context")


def _action(
    arm: str,
    teacher,
    student_action,
    student_params,
    observation,
    mode: int,
    posterior,
    event_seed: int,
) -> np.ndarray:
    if arm == protocol.FIXED_ROBUST_ARM:
        target = ("final", 719)
        try:
            index = teacher.controller_keys.index(target)
        except ValueError as exc:
            raise ValueError("frozen robust 719 controller is absent") from exc
        return model_lib.individual_action(
            teacher, "robust", index, observation, trainer._zero_context())
    if arm == "teacher_oracle_median":
        return model_lib.reduced_action(
            teacher, "oracle", observation, trainer._one_hot(mode))
    if arm == "teacher_learned_median":
        return model_lib.reduced_action(
            teacher, "oracle", observation, posterior)
    context = _student_context(arm, mode, posterior, event_seed)
    return np.asarray(
        student_action(
            student_params,
            jnp.asarray(observation, dtype=jnp.float32),
            jnp.asarray(context, dtype=jnp.float32),
        ),
        dtype=np.float32,
    )


def _stationary(
    config,
    teacher,
    estimator,
    student_action,
    student_params,
    arm: str,
    event_seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for mode in protocol.MODES:
        run_config = copy.deepcopy(config)
        run_config.stochastic_mode_fixed_id = int(mode)
        env = make_env(
            run_config, seed_offset=int(event_seed) - int(config.seed))
        tasks = env.sample_tasks(len(protocol.MODES))
        env.set_nonstationary_para(tasks)
        env.set_task(tasks[int(mode)])
        returns = []
        terminations = []
        posterior_rows = []
        labels = []
        for _ in range(protocol.AUDIT_EPISODES_PER_TASK):
            observation = env.reset()
            estimator_state = estimator.initial_state()
            episode_return = 0.0
            episode_terminated = False
            for _ in range(protocol.MAX_EPISODE_STEPS):
                posterior = estimator.probabilities(estimator_state)
                action = _action(
                    arm, teacher, student_action, student_params,
                    observation, int(mode), posterior, event_seed)
                next_observation, reward, done, info = env.step(action)
                if int(info["mode_used"]) != int(mode):
                    raise RuntimeError("stationary ablation mode changed")
                if _uses_estimator(arm):
                    posterior_rows.append(np.asarray(posterior).copy())
                    labels.append(int(mode))
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
                    episode_terminated = True
                    observation = env.reset()
                    if _uses_estimator(arm):
                        estimator_state = estimator.initial_state()
            returns.append(episode_return)
            terminations.append(float(episode_terminated))
        row = {
            "arm": arm,
            "mode": int(mode),
            "returns": [float(value) for value in returns],
            "return_mean": float(np.mean(returns)),
            "return_std": float(np.std(returns)),
            "terminated_rate": float(np.mean(terminations)),
        }
        if _uses_estimator(arm):
            row["posterior_metrics"] = (
                protocol.ensemble.final.posterior_metrics(
                    np.asarray(posterior_rows),
                    np.asarray(labels, dtype=np.int32),
                )
            )
        rows.append(row)
        if hasattr(env, "close"):
            env.close()
    return rows


def _switching(
    config,
    teacher,
    estimator,
    student_action,
    student_params,
    arm: str,
    event_seed: int,
) -> dict[str, Any]:
    run_config = copy.deepcopy(config)
    run_config.stochastic_mode_fixed_id = -1
    env = make_env(
        run_config, seed_offset=int(event_seed) - int(config.seed))
    tasks = env.sample_tasks(len(protocol.MODES))
    returns = []
    terminations = []
    posterior_rows = []
    labels = []
    mode_traces = []
    for episode in range(protocol.AUDIT_SWITCHING_EPISODES):
        sequence, _ = _select_eval_switch_sequence(env, tasks, episode)
        _reset_eval_switch_schedule(
            env, sequence, run_config, protocol.DWELL_STEPS)
        observation = env.reset()
        estimator_state = estimator.initial_state()
        episode_return = 0.0
        episode_terminated = False
        episode_modes = []
        for _ in range(protocol.MAX_EPISODE_STEPS):
            mode = int(env.task_id_for_next_step())
            episode_modes.append(mode)
            posterior = estimator.probabilities(estimator_state)
            action = _action(
                arm, teacher, student_action, student_params,
                observation, mode, posterior, event_seed)
            next_observation, reward, done, info = env.step(action)
            if int(info["mode_used"]) != mode:
                raise RuntimeError("switching ablation mode misaligned")
            if _uses_estimator(arm):
                posterior_rows.append(np.asarray(posterior).copy())
                labels.append(mode)
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
                episode_terminated = True
                observation = env.reset()
        returns.append(episode_return)
        terminations.append(float(episode_terminated))
        mode_traces.append(episode_modes)
    if hasattr(env, "close"):
        env.close()
    row = {
        "arm": arm,
        "returns": [float(value) for value in returns],
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "terminated_rate": float(np.mean(terminations)),
        "mode_traces": mode_traces,
    }
    if _uses_estimator(arm):
        row["posterior_metrics"] = (
            protocol.ensemble.final.posterior_metrics(
                np.asarray(posterior_rows),
                np.asarray(labels, dtype=np.int32),
            )
        )
    return row


def validate_audit(event_seed: int) -> dict[str, Any]:
    protocol.validate_frozen_candidate()
    event_seed = protocol.require_audit_event_seed(event_seed)
    destination = protocol.audit_dir(
        protocol.TEACHER_GROUP, protocol.STUDENT_SEED, event_seed)
    manifest = protocol.read_json(destination / "audit_manifest.json")
    identity = protocol.audit_identity(
        protocol.TEACHER_GROUP, protocol.STUDENT_SEED, event_seed)
    if (
        manifest.get("schema") != protocol.AUDIT_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("identity") != identity
        or manifest.get("student_manifest")
        != protocol.file_record(protocol.model_manifest(
            protocol.TEACHER_GROUP, protocol.STUDENT_SEED))
        or manifest.get("student_parameters")
        != protocol.file_record(protocol.model_path(
            protocol.TEACHER_GROUP, protocol.STUDENT_SEED))
        or manifest.get("source_bundles")
        != model_lib.source_records(protocol.TEACHER_GROUP)
        or manifest.get("frozen_estimator_manifest")
        != protocol.file_record(protocol.ensemble.final.MODEL_MANIFEST)
        or manifest.get("frozen_estimator_parameters")
        != protocol.file_record(protocol.ensemble.final.MODEL_PATH)
        or manifest.get("result_file")
        != protocol.file_record(destination / "results.json")
    ):
        raise ValueError(f"invalid context-ablation audit: {destination}")
    result = protocol.read_json(destination / "results.json")
    expected = set(protocol.ARM_LABELS)
    if (
        result.get("schema") != protocol.EVENT_SCHEMA
        or result.get("status") != "complete"
        or result.get("identity") != identity
        or {row.get("arm") for row in result.get("switching", [])}
        != expected
        or len(result.get("stationary", []))
        != len(expected) * len(protocol.MODES)
    ):
        raise ValueError("incomplete context-ablation result matrix")
    switching_traces = None
    for row in result["switching"]:
        values = np.asarray(row.get("returns"), dtype=np.float64)
        traces = row.get("mode_traces")
        if (
            values.shape != (protocol.AUDIT_SWITCHING_EPISODES,)
            or not np.all(np.isfinite(values))
            or len(traces) != protocol.AUDIT_SWITCHING_EPISODES
            or any(len(trace) != protocol.MAX_EPISODE_STEPS for trace in traces)
        ):
            raise ValueError("invalid switching context-ablation output")
        normalized = tuple(tuple(int(value) for value in trace) for trace in traces)
        if switching_traces is None:
            switching_traces = normalized
        elif normalized != switching_traces:
            raise ValueError("context-ablation arms used different mode streams")
    for row in result["stationary"]:
        values = np.asarray(row.get("returns"), dtype=np.float64)
        if (
            values.shape != (protocol.AUDIT_EPISODES_PER_TASK,)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("invalid stationary context-ablation output")
    return manifest


def run(event_seed: int) -> None:
    protocol.validate_frozen_candidate()
    event_seed = protocol.require_audit_event_seed(event_seed)
    destination = protocol.audit_dir(
        protocol.TEACHER_GROUP, protocol.STUDENT_SEED, event_seed)
    if (destination / "audit_manifest.json").is_file():
        try:
            validate_audit(event_seed)
        except (KeyError, OSError, TypeError, ValueError):
            pass
        else:
            print(f"CONTEXT ABLATION ALREADY COMPLETE: {destination}")
            return

    protocol.ensemble.validate_frozen_estimator()
    trainer.validate_model(protocol.TEACHER_GROUP, protocol.STUDENT_SEED)
    teacher = model_lib.load_teacher(protocol.TEACHER_GROUP)
    estimator = model_lib.make_estimator(teacher.obs_dim, teacher.act_dim)
    student, student_params, _ = model_lib.load_student(
        protocol.TEACHER_GROUP,
        protocol.STUDENT_SEED,
        teacher.obs_dim,
        teacher.act_dim,
    )
    student_action = model_lib.build_student_action(student)
    stationary = []
    switching = []
    for arm in protocol.ARM_LABELS:
        stationary.extend(_stationary(
            teacher.config,
            teacher,
            estimator,
            student_action,
            student_params,
            arm,
            event_seed,
        ))
        switching.append(_switching(
            teacher.config,
            teacher,
            estimator,
            student_action,
            student_params,
            arm,
            event_seed,
        ))
        print(f"context ablation event={event_seed} arm={arm} complete",
              flush=True)

    identity = protocol.audit_identity(
        protocol.TEACHER_GROUP, protocol.STUDENT_SEED, event_seed)
    result = {
        "schema": protocol.EVENT_SCHEMA,
        "status": "complete",
        "identity": identity,
        "stationary": stationary,
        "switching": switching,
    }
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp.", dir=destination.parent))
    try:
        protocol.write_json_atomic(temporary / "results.json", result)
        manifest = {
            "schema": protocol.AUDIT_SCHEMA,
            "status": "complete",
            "identity": identity,
            "source_bundles": model_lib.source_records(
                protocol.TEACHER_GROUP),
            "student_manifest": protocol.file_record(
                protocol.model_manifest(
                    protocol.TEACHER_GROUP, protocol.STUDENT_SEED)),
            "student_parameters": protocol.file_record(
                protocol.model_path(
                    protocol.TEACHER_GROUP, protocol.STUDENT_SEED)),
            "frozen_estimator_manifest": protocol.file_record(
                protocol.ensemble.final.MODEL_MANIFEST),
            "frozen_estimator_parameters": protocol.file_record(
                protocol.ensemble.final.MODEL_PATH),
            "result_file": protocol.file_record(temporary / "results.json"),
        }
        protocol.write_json_atomic(
            temporary / "audit_manifest.json", manifest)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_audit(event_seed)
    print(f"CONTEXT ABLATION COMPLETE: {destination}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event-seed",
        type=int,
        choices=protocol.AUDIT_EVENT_SEEDS,
        required=True,
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.event_seed)


if __name__ == "__main__":
    main()
