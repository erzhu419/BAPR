"""Evaluate causal robust-fallback configurations on one development event."""
from __future__ import annotations

import argparse
import copy
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

import jax.numpy as jnp
import numpy as np

from jax_experiments.analysis import (
    regime_polarity_policy_distillation_control_model as model_lib,
)
from jax_experiments.analysis import (
    regime_polarity_policy_distillation_transient_fallback_v1 as protocol,
)
from jax_experiments.analysis import (
    train_regime_polarity_policy_distillation_control_v2 as trainer,
)
from jax_experiments.train import (
    _reset_eval_switch_schedule,
    _select_eval_switch_sequence,
    make_env,
)


def _adaptive_action(student_action, student_params, observation, context):
    return np.asarray(
        student_action(
            student_params,
            jnp.asarray(observation, dtype=jnp.float32),
            jnp.asarray(context, dtype=jnp.float32),
        ),
        dtype=np.float32,
    )


def _robust_action(teacher, observation):
    target = ("final", 719)
    try:
        index = teacher.controller_keys.index(target)
    except ValueError as exc:
        raise ValueError("frozen robust final seed 719 is absent") from exc
    return model_lib.individual_action(
        teacher,
        "robust",
        index,
        observation,
        trainer._zero_context(),
    )


def evaluate_switching_arm(
    config,
    teacher,
    estimator,
    student_action,
    student_params,
    arm: str,
    event_seed: int,
) -> dict[str, Any]:
    fallback_config = (
        protocol.require_config(arm)
        if arm in protocol.FALLBACK_CONFIG_BY_NAME else None
    )
    if arm not in protocol.BASELINE_ARMS and fallback_config is None:
        raise ValueError(f"unknown transient-fallback arm {arm!r}")

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
    fallback_actions = 0
    adaptive_wrong_actions = 0
    total_actions = 0
    trigger_counts = []
    try:
        for episode in range(protocol.SWITCHING_EPISODES):
            sequence, _ = _select_eval_switch_sequence(env, tasks, episode)
            _reset_eval_switch_schedule(
                env, sequence, run_config, protocol.DWELL_STEPS)
            observation = env.reset()
            estimator_state = estimator.initial_state()
            fallback_state = protocol.initial_fallback_state()
            episode_return = 0.0
            episode_terminated = False
            for _ in range(protocol.MAX_EPISODE_STEPS):
                mode = int(env.task_id_for_next_step())
                posterior = estimator.probabilities(estimator_state)
                if arm == protocol.ROBUST_ARM:
                    action = _robust_action(teacher, observation)
                    uses_estimator = False
                elif arm == protocol.ORACLE_ARM:
                    action = _adaptive_action(
                        student_action,
                        student_params,
                        observation,
                        trainer._one_hot(mode),
                    )
                    uses_estimator = False
                else:
                    uses_estimator = True
                    if fallback_config is not None and fallback_state.fallback:
                        action = _robust_action(teacher, observation)
                        fallback_actions += 1
                    else:
                        action = _adaptive_action(
                            student_action,
                            student_params,
                            observation,
                            posterior,
                        )
                        adaptive_wrong_actions += int(
                            int(np.argmax(posterior)) != mode)
                    posterior_rows.append(np.asarray(posterior).copy())
                    labels.append(mode)
                next_observation, reward, done, info = env.step(action)
                if int(info["mode_used"]) != mode:
                    raise RuntimeError("transient-fallback mode was not causal")
                if uses_estimator:
                    next_state, evidence, _, _ = estimator.step(
                        estimator_state,
                        observation,
                        action,
                        reward,
                        next_observation,
                    )
                    if fallback_config is not None:
                        fallback_state = protocol.update_fallback_state(
                            fallback_state,
                            posterior,
                            estimator.probabilities(next_state),
                            evidence,
                            fallback_config,
                        )
                    estimator_state = next_state
                episode_return += float(reward)
                observation = next_observation
                total_actions += 1
                if done:
                    episode_terminated = True
                    observation = env.reset()
            returns.append(float(episode_return))
            terminations.append(float(episode_terminated))
            if fallback_config is not None:
                trigger_counts.append(int(fallback_state.trigger_count))
    finally:
        if hasattr(env, "close"):
            env.close()

    row: dict[str, Any] = {
        "arm": arm,
        "returns": returns,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "terminated_rate": float(np.mean(terminations)),
    }
    if posterior_rows:
        row["posterior_metrics"] = protocol.frozen.development.ensemble.final.posterior_metrics(
            np.asarray(posterior_rows),
            np.asarray(labels, dtype=np.int32),
        )
    if fallback_config is not None:
        row["fallback_config"] = fallback_config.to_dict()
        row["fallback_action_fraction"] = float(
            fallback_actions / max(total_actions, 1))
        row["adaptive_wrong_action_fraction"] = float(
            adaptive_wrong_actions / max(total_actions, 1))
        row["trigger_count_mean"] = float(np.mean(trigger_counts))
    return row


def evaluate_event(event_seed: int, arms: Iterable[str]) -> list[dict[str, Any]]:
    protocol.validate_frozen_candidate()
    teacher = model_lib.load_teacher(protocol.TEACHER_GROUP)
    estimator = model_lib.make_estimator(teacher.obs_dim, teacher.act_dim)
    student, student_params, _ = model_lib.load_student(
        protocol.TEACHER_GROUP,
        protocol.STUDENT_SEED,
        teacher.obs_dim,
        teacher.act_dim,
    )
    student_action = model_lib.build_student_action(student)
    rows = []
    for arm in arms:
        rows.append(evaluate_switching_arm(
            teacher.config,
            teacher,
            estimator,
            student_action,
            student_params,
            str(arm),
            int(event_seed),
        ))
        print(
            f"transient fallback event={event_seed} arm={arm} complete",
            flush=True,
        )
    return rows


def _validate_result(
    result: dict[str, Any],
    identity: dict[str, Any],
    expected_arms: Iterable[str],
    schema: str,
) -> None:
    expected = tuple(expected_arms)
    rows = result.get("switching", [])
    if (result.get("schema") != schema
            or result.get("status") != "complete"
            or result.get("identity") != identity
            or tuple(row.get("arm") for row in rows) != expected):
        raise ValueError("incomplete transient-fallback result")
    for row in rows:
        values = np.asarray(row.get("returns"), dtype=np.float64)
        if (values.shape != (protocol.SWITCHING_EPISODES,)
                or not np.all(np.isfinite(values))):
            raise ValueError("invalid transient-fallback return vector")


def validate_screen(event_seed: int) -> dict[str, Any]:
    event_seed = protocol.require_screen_event_seed(event_seed)
    destination = protocol.screen_dir(event_seed)
    identity = protocol.common_identity(
        "transient_fallback_development_screen", event_seed)
    manifest = protocol.read_json(destination / "screen_manifest.json")
    if (manifest.get("schema") != protocol.SCREEN_SCHEMA
            or manifest.get("status") != "complete"
            or manifest.get("identity") != identity
            or manifest.get("source_bundles")
            != model_lib.source_records(protocol.TEACHER_GROUP)
            or manifest.get("student_manifest")
            != protocol.FROZEN_STUDENT_MANIFEST_RECORD
            or manifest.get("student_parameters")
            != protocol.FROZEN_STUDENT_PARAMETER_RECORD
            or manifest.get("result_file")
            != protocol.file_record(destination / "results.json")):
        raise ValueError(f"invalid transient-fallback screen {destination}")
    result = protocol.read_json(destination / "results.json")
    _validate_result(
        result, identity, protocol.SCREEN_ARMS, protocol.SCREEN_EVENT_SCHEMA)
    return manifest


def run(event_seed: int) -> None:
    event_seed = protocol.require_screen_event_seed(event_seed)
    destination = protocol.screen_dir(event_seed)
    if (destination / "screen_manifest.json").is_file():
        try:
            validate_screen(event_seed)
        except (KeyError, OSError, TypeError, ValueError):
            pass
        else:
            print(f"TRANSIENT FALLBACK SCREEN ALREADY COMPLETE: {destination}")
            return

    identity = protocol.common_identity(
        "transient_fallback_development_screen", event_seed)
    result = {
        "schema": protocol.SCREEN_EVENT_SCHEMA,
        "status": "complete",
        "identity": identity,
        "switching": evaluate_event(event_seed, protocol.SCREEN_ARMS),
    }
    if destination.exists() or destination.is_symlink():
        shutil.rmtree(destination) if destination.is_dir() else destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp.", dir=destination.parent))
    try:
        protocol.write_json_atomic(temporary / "results.json", result)
        manifest = {
            "schema": protocol.SCREEN_SCHEMA,
            "status": "complete",
            "identity": identity,
            "source_bundles": model_lib.source_records(protocol.TEACHER_GROUP),
            "frozen_estimator_manifest": protocol.file_record(
                protocol.frozen.development.ensemble.final.MODEL_MANIFEST),
            "frozen_estimator_parameters": protocol.file_record(
                protocol.frozen.development.ensemble.final.MODEL_PATH),
            "student_manifest": protocol.FROZEN_STUDENT_MANIFEST_RECORD,
            "student_parameters": protocol.FROZEN_STUDENT_PARAMETER_RECORD,
            "result_file": protocol.file_record(temporary / "results.json"),
        }
        protocol.write_json_atomic(temporary / "screen_manifest.json", manifest)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_screen(event_seed)
    print(f"TRANSIENT FALLBACK SCREEN COMPLETE: {destination}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event-seed", type=int, choices=protocol.SCREEN_EVENT_SEEDS,
        required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.event_seed)


if __name__ == "__main__":
    main()
