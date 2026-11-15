"""Strict audit of one distilled posterior-conditioned polarity policy."""
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
    regime_polarity_policy_distillation as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_policy_distillation_model as model_lib,
)
from jax_experiments.analysis import (
    train_regime_polarity_policy_distillation as trainer,
)
from jax_experiments.train import (
    _reset_eval_switch_schedule,
    _select_eval_switch_sequence,
    make_env,
)


def robust_label(source_group: str, seed: int) -> str:
    return f"robust_{source_group}_seed_{int(seed)}"


def arm_labels(group: str) -> tuple[str, ...]:
    return (
        *(robust_label(source_group, seed)
          for source_group, seed in protocol.controller_keys(group)),
        "teacher_oracle_median",
        "teacher_learned_median",
        "student_learned",
    )


def _action(
    arm: str,
    teacher,
    student_action,
    student_params,
    observation,
    mode: int,
    posterior,
) -> np.ndarray:
    if arm == "teacher_oracle_median":
        return model_lib.reduced_action(
            teacher, "oracle", observation, trainer._one_hot(mode))
    if arm == "teacher_learned_median":
        return model_lib.reduced_action(
            teacher, "oracle", observation, posterior)
    if arm == "student_learned":
        return np.asarray(student_action(
            student_params,
            jnp.asarray(observation, dtype=jnp.float32),
            jnp.asarray(posterior, dtype=jnp.float32),
        ), dtype=np.float32)
    for index, (source_group, seed) in enumerate(teacher.controller_keys):
        if arm == robust_label(source_group, seed):
            return model_lib.individual_action(
                teacher,
                "robust",
                index,
                observation,
                trainer._zero_context(),
            )
    raise ValueError(f"unknown distillation audit arm {arm!r}")


def _uses_estimator(arm: str) -> bool:
    return arm in ("teacher_learned_median", "student_learned")


def _stationary(
    group: str,
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
            run_config,
            seed_offset=int(event_seed) - int(config.seed),
        )
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
                    arm,
                    teacher,
                    student_action,
                    student_params,
                    observation,
                    int(mode),
                    posterior,
                )
                next_observation, reward, done, info = env.step(action)
                if int(info["mode_used"]) != int(mode):
                    raise RuntimeError("stationary distillation mode changed")
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
            row["posterior_metrics"] = protocol.ensemble.final.posterior_metrics(
                np.asarray(posterior_rows),
                np.asarray(labels, dtype=np.int32),
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
        run_config,
        seed_offset=int(event_seed) - int(config.seed),
    )
    tasks = env.sample_tasks(len(protocol.MODES))
    returns = []
    terminations = []
    posterior_rows = []
    labels = []
    for episode in range(protocol.AUDIT_SWITCHING_EPISODES):
        sequence, _ = _select_eval_switch_sequence(env, tasks, episode)
        _reset_eval_switch_schedule(
            env, sequence, run_config, protocol.DWELL_STEPS)
        observation = env.reset()
        estimator_state = estimator.initial_state()
        episode_return = 0.0
        episode_terminated = False
        for _ in range(protocol.MAX_EPISODE_STEPS):
            mode = int(env.task_id_for_next_step())
            posterior = estimator.probabilities(estimator_state)
            action = _action(
                arm,
                teacher,
                student_action,
                student_params,
                observation,
                mode,
                posterior,
            )
            next_observation, reward, done, info = env.step(action)
            if int(info["mode_used"]) != mode:
                raise RuntimeError("switching distillation label misaligned")
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
    if hasattr(env, "close"):
        env.close()
    row = {
        "arm": arm,
        "returns": [float(value) for value in returns],
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "terminated_rate": float(np.mean(terminations)),
    }
    if _uses_estimator(arm):
        row["posterior_metrics"] = protocol.ensemble.final.posterior_metrics(
            np.asarray(posterior_rows),
            np.asarray(labels, dtype=np.int32),
        )
    return row


def validate_audit(
    group: str,
    student_seed: int,
    event_seed: int,
) -> dict[str, Any]:
    group = protocol.require_teacher_group(group)
    student_seed = protocol.require_student_seed(student_seed)
    event_seed = protocol.require_audit_event_seed(event_seed)
    destination = protocol.audit_dir(group, student_seed, event_seed)
    manifest = protocol.read_json(destination / "audit_manifest.json")
    identity = protocol.audit_identity(group, student_seed, event_seed)
    if (manifest.get("schema") != protocol.AUDIT_SCHEMA
            or manifest.get("status") != "complete"
            or manifest.get("identity") != identity
            or manifest.get("source_bundles")
            != model_lib.source_records(group)
            or manifest.get("student_manifest")
            != protocol.file_record(protocol.model_manifest(
                group, student_seed))
            or manifest.get("student_parameters")
            != protocol.file_record(protocol.model_path(group, student_seed))
            or manifest.get("result_file")
            != protocol.file_record(destination / "results.json")):
        raise ValueError(f"invalid policy-distillation audit: {destination}")
    result = protocol.read_json(destination / "results.json")
    expected = set(arm_labels(group))
    if (result.get("schema") != protocol.EVENT_SCHEMA
            or result.get("status") != "complete"
            or result.get("identity") != identity
            or {row.get("arm") for row in result.get("switching", [])}
            != expected
            or len(result.get("stationary", []))
            != len(expected) * len(protocol.MODES)):
        raise ValueError("incomplete policy-distillation result matrix")
    for row in result["stationary"]:
        values = np.asarray(row.get("returns"), dtype=np.float64)
        if (values.shape != (protocol.AUDIT_EPISODES_PER_TASK,)
                or not np.all(np.isfinite(values))):
            raise ValueError("invalid stationary distillation returns")
    for row in result["switching"]:
        values = np.asarray(row.get("returns"), dtype=np.float64)
        if (values.shape != (protocol.AUDIT_SWITCHING_EPISODES,)
                or not np.all(np.isfinite(values))):
            raise ValueError("invalid switching distillation returns")
    return manifest


def run(group: str, student_seed: int, event_seed: int) -> None:
    group = protocol.require_teacher_group(group)
    student_seed = protocol.require_student_seed(student_seed)
    event_seed = protocol.require_audit_event_seed(event_seed)
    destination = protocol.audit_dir(group, student_seed, event_seed)
    if (destination / "audit_manifest.json").is_file():
        try:
            validate_audit(group, student_seed, event_seed)
        except (KeyError, OSError, TypeError, ValueError):
            pass
        else:
            print(f"DISTILLED STUDENT AUDIT ALREADY COMPLETE: {destination}")
            return

    protocol.ensemble.validate_frozen_estimator()
    trainer.validate_model(group, student_seed)
    teacher = model_lib.load_teacher(group)
    estimator = model_lib.make_estimator(teacher.obs_dim, teacher.act_dim)
    student, student_params, _ = model_lib.load_student(
        group, student_seed, teacher.obs_dim, teacher.act_dim)
    student_action = model_lib.build_student_action(student)
    stationary = []
    switching = []
    for arm in arm_labels(group):
        stationary.extend(_stationary(
            group,
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
        print(
            f"distillation audit group={group} student={student_seed} "
            f"event={event_seed} arm={arm} complete",
            flush=True,
        )

    identity = protocol.audit_identity(group, student_seed, event_seed)
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
            "source_bundles": model_lib.source_records(group),
            "frozen_estimator_manifest": protocol.file_record(
                protocol.ensemble.final.MODEL_MANIFEST),
            "frozen_estimator_parameters": protocol.file_record(
                protocol.ensemble.final.MODEL_PATH),
            "student_manifest": protocol.file_record(
                protocol.model_manifest(group, student_seed)),
            "student_parameters": protocol.file_record(
                protocol.model_path(group, student_seed)),
            "result_file": protocol.file_record(
                temporary / "results.json"),
        }
        protocol.write_json_atomic(
            temporary / "audit_manifest.json", manifest)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_audit(group, student_seed, event_seed)
    print(f"DISTILLED STUDENT AUDIT COMPLETE: {destination}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher-group", choices=protocol.TEACHER_GROUPS, required=True)
    parser.add_argument(
        "--student-seed", type=int, choices=protocol.STUDENT_SEEDS,
        required=True)
    parser.add_argument(
        "--event-seed", type=int, choices=protocol.AUDIT_EVENT_SEEDS,
        required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.teacher_group, args.student_seed, args.event_seed)


if __name__ == "__main__":
    main()
