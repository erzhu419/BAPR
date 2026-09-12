"""Run one frozen student/event delayed-oracle causal-ceiling audit."""
from __future__ import annotations

import argparse
import copy
import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from jax_experiments.analysis import (
    regime_polarity_policy_distillation_control_model as model_lib,
)
from jax_experiments.analysis import (
    regime_polarity_policy_distillation_transient_causal_ceiling_v1 as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_policy_distillation_transient_fallback_v1 as fallback,
)
from jax_experiments.analysis import (
    run_regime_polarity_policy_distillation_transient_fallback_screen_v1 as evaluator,
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


def evaluate_arm(
    config,
    teacher,
    estimator,
    student_action,
    student_params,
    arm: str,
    event_seed: int,
) -> dict[str, Any]:
    if arm not in protocol.ARMS:
        raise ValueError(f"unknown causal-ceiling arm {arm!r}")
    delay = protocol.arm_delay(arm)
    fallback_config = (
        fallback.require_config(protocol.FALLBACK_ARM)
        if arm == protocol.FALLBACK_ARM else None
    )

    run_config = copy.deepcopy(config)
    run_config.stochastic_mode_fixed_id = -1
    env = make_env(
        run_config, seed_offset=int(event_seed) - int(config.seed))
    tasks = env.sample_tasks(len(fallback.MODES))
    returns = []
    terminations = []
    posterior_rows = []
    labels = []
    mode_trace = []
    fallback_actions = 0
    adaptive_wrong_actions = 0
    stale_actions = 0
    switch_count = 0
    total_actions = 0
    trigger_counts = []
    try:
        for episode in range(fallback.SWITCHING_EPISODES):
            sequence, _ = _select_eval_switch_sequence(env, tasks, episode)
            _reset_eval_switch_schedule(
                env, sequence, run_config, fallback.DWELL_STEPS)
            observation = env.reset()
            estimator_state = estimator.initial_state()
            fallback_state = fallback.initial_fallback_state()
            context_mode = None
            pending_mode = None
            delay_remaining = 0
            previous_mode = None
            episode_return = 0.0
            episode_terminated = False
            for _ in range(fallback.MAX_EPISODE_STEPS):
                mode = int(env.task_id_for_next_step())
                mode_trace.append(mode)
                if previous_mode is not None and mode != previous_mode:
                    switch_count += 1
                previous_mode = mode

                if delay is not None:
                    if context_mode is None:
                        context_mode = mode
                        pending_mode = mode
                    elif mode != pending_mode:
                        pending_mode = mode
                        delay_remaining = delay

                posterior = estimator.probabilities(estimator_state)
                uses_estimator = arm in (
                    protocol.LEARNED_ARM, protocol.FALLBACK_ARM)
                if arm == protocol.ROBUST_ARM:
                    action = evaluator._robust_action(teacher, observation)
                elif arm == protocol.ORACLE_ARM:
                    action = _adaptive_action(
                        student_action, student_params, observation,
                        trainer._one_hot(mode))
                elif delay is not None:
                    action = _adaptive_action(
                        student_action, student_params, observation,
                        trainer._one_hot(context_mode))
                    stale_actions += int(context_mode != mode)
                elif arm == protocol.LEARNED_ARM:
                    action = _adaptive_action(
                        student_action, student_params, observation, posterior)
                    adaptive_wrong_actions += int(
                        int(np.argmax(posterior)) != mode)
                elif fallback_state.fallback:
                    action = evaluator._robust_action(teacher, observation)
                    fallback_actions += 1
                else:
                    action = _adaptive_action(
                        student_action, student_params, observation, posterior)
                    adaptive_wrong_actions += int(
                        int(np.argmax(posterior)) != mode)

                if uses_estimator:
                    posterior_rows.append(np.asarray(posterior).copy())
                    labels.append(mode)
                next_observation, reward, done, info = env.step(action)
                if int(info["mode_used"]) != mode:
                    raise RuntimeError("causal-ceiling mode was not aligned")

                if uses_estimator:
                    next_state, evidence, _, _ = estimator.step(
                        estimator_state,
                        observation,
                        action,
                        reward,
                        next_observation,
                    )
                    if fallback_config is not None:
                        fallback_state = fallback.update_fallback_state(
                            fallback_state,
                            posterior,
                            estimator.probabilities(next_state),
                            evidence,
                            fallback_config,
                        )
                    estimator_state = next_state
                if delay is not None and pending_mode != context_mode:
                    delay_remaining -= 1
                    if delay_remaining <= 0:
                        context_mode = pending_mode
                        delay_remaining = 0

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

    trace_bytes = bytes(int(mode) for mode in mode_trace)
    row: dict[str, Any] = {
        "arm": arm,
        "returns": returns,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "terminated_rate": float(np.mean(terminations)),
        "mode_trace_sha256": hashlib.sha256(trace_bytes).hexdigest(),
        "switch_count": int(switch_count),
        "total_actions": int(total_actions),
    }
    if posterior_rows:
        row["posterior_metrics"] = (
            fallback.frozen.development.ensemble.final.posterior_metrics(
                np.asarray(posterior_rows),
                np.asarray(labels, dtype=np.int32),
            )
        )
        row["adaptive_wrong_action_fraction"] = float(
            adaptive_wrong_actions / max(total_actions, 1))
    if delay is not None:
        row["delay_steps"] = int(delay)
        row["stale_action_count"] = int(stale_actions)
        row["stale_action_fraction"] = float(
            stale_actions / max(total_actions, 1))
    if fallback_config is not None:
        row["fallback_config"] = fallback_config.to_dict()
        row["fallback_action_fraction"] = float(
            fallback_actions / max(total_actions, 1))
        row["trigger_count_mean"] = float(np.mean(trigger_counts))
    return row


def evaluate_event(student_seed: int, event_seed: int) -> list[dict[str, Any]]:
    protocol.validate_upstream(student_seed)
    teacher = model_lib.load_teacher(protocol.TEACHER_GROUP)
    estimator = model_lib.make_estimator(teacher.obs_dim, teacher.act_dim)
    student, student_params, _ = model_lib.load_student(
        protocol.TEACHER_GROUP,
        student_seed,
        teacher.obs_dim,
        teacher.act_dim,
    )
    student_action = model_lib.build_student_action(student)
    rows = []
    for arm in protocol.ARMS:
        rows.append(evaluate_arm(
            teacher.config,
            teacher,
            estimator,
            student_action,
            student_params,
            arm,
            event_seed,
        ))
        print(
            f"causal ceiling student={student_seed} event={event_seed} "
            f"arm={arm} complete",
            flush=True,
        )
    return rows


def _validate_result(result: dict[str, Any], identity: dict[str, Any]) -> None:
    rows = result.get("switching", [])
    if (
        result.get("schema") != protocol.EVENT_SCHEMA
        or result.get("status") != "complete"
        or result.get("identity") != identity
        or tuple(row.get("arm") for row in rows) != protocol.ARMS
    ):
        raise ValueError("incomplete causal-ceiling result")
    trace_hash = None
    stale_counts = []
    for row in rows:
        values = np.asarray(row.get("returns"), dtype=np.float64)
        current_hash = row.get("mode_trace_sha256")
        if (
            values.shape != (fallback.SWITCHING_EPISODES,)
            or not np.all(np.isfinite(values))
            or not isinstance(current_hash, str)
            or len(current_hash) != 64
        ):
            raise ValueError("invalid causal-ceiling return vector")
        if trace_hash is None:
            trace_hash = current_hash
        elif current_hash != trace_hash:
            raise ValueError("causal-ceiling arms used different mode streams")
        delay = protocol.arm_delay(row["arm"])
        if delay is not None:
            stale_counts.append((delay, int(row["stale_action_count"])))
    if [count for _, count in stale_counts] != sorted(
            count for _, count in stale_counts):
        raise ValueError("delayed-oracle stale actions are not monotone")


def validate_audit(student_seed: int, event_seed: int) -> dict[str, Any]:
    student_seed = protocol.require_student_seed(student_seed)
    event_seed = protocol.require_event_seed(event_seed)
    protocol.validate_upstream(student_seed)
    destination = protocol.audit_dir(student_seed, event_seed)
    identity = protocol.identity(student_seed, event_seed)
    manifest = protocol.read_json(destination / "audit_manifest.json")
    records = protocol.parent.student_records(student_seed)
    if (
        manifest.get("schema") != protocol.AUDIT_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("identity") != identity
        or manifest.get("source_bundles")
        != model_lib.source_records(protocol.TEACHER_GROUP)
        or manifest.get("student_manifest") != records["manifest"]
        or manifest.get("student_parameters") != records["parameters"]
        or manifest.get("frozen_cross_student_analysis")
        != protocol.FROZEN_CROSS_STUDENT_ANALYSIS_RECORD
        or manifest.get("result_file")
        != protocol.file_record(destination / "results.json")
    ):
        raise ValueError(f"invalid causal-ceiling audit {destination}")
    result = protocol.read_json(destination / "results.json")
    _validate_result(result, identity)
    return manifest


def run(student_seed: int, event_seed: int) -> None:
    student_seed = protocol.require_student_seed(student_seed)
    event_seed = protocol.require_event_seed(event_seed)
    destination = protocol.audit_dir(student_seed, event_seed)
    if (destination / "audit_manifest.json").is_file():
        try:
            validate_audit(student_seed, event_seed)
        except (KeyError, OSError, TypeError, ValueError):
            pass
        else:
            print(f"CAUSAL CEILING ALREADY COMPLETE: {destination}")
            return

    identity = protocol.identity(student_seed, event_seed)
    result = {
        "schema": protocol.EVENT_SCHEMA,
        "status": "complete",
        "identity": identity,
        "switching": evaluate_event(student_seed, event_seed),
    }
    if destination.exists() or destination.is_symlink():
        shutil.rmtree(destination) if destination.is_dir() else destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp.", dir=destination.parent))
    try:
        protocol.write_json_atomic(temporary / "results.json", result)
        records = protocol.parent.student_records(student_seed)
        manifest = {
            "schema": protocol.AUDIT_SCHEMA,
            "status": "complete",
            "identity": identity,
            "source_bundles": model_lib.source_records(protocol.TEACHER_GROUP),
            "student_manifest": records["manifest"],
            "student_parameters": records["parameters"],
            "frozen_estimator_manifest":
                protocol.parent.FROZEN_ESTIMATOR_MANIFEST_RECORD,
            "frozen_estimator_parameters":
                protocol.parent.FROZEN_ESTIMATOR_PARAMETER_RECORD,
            "frozen_cross_student_analysis":
                protocol.FROZEN_CROSS_STUDENT_ANALYSIS_RECORD,
            "result_file": protocol.file_record(temporary / "results.json"),
        }
        protocol.write_json_atomic(
            temporary / "audit_manifest.json", manifest)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_audit(student_seed, event_seed)
    print(f"CAUSAL CEILING COMPLETE: {destination}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--student-seed", type=int, choices=protocol.STUDENT_SEEDS,
        required=True)
    parser.add_argument(
        "--event-seed", type=int, choices=protocol.EVENT_SEEDS,
        required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.student_seed, args.event_seed)


if __name__ == "__main__":
    main()
