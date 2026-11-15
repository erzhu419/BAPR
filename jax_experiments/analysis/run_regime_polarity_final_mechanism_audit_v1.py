"""Evaluate one frozen final BAPR student on one mechanism event stream."""
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
    regime_polarity_final_mechanism_audit_v1 as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_policy_distillation_control_model as model_lib,
)
from jax_experiments.analysis import (
    run_regime_polarity_fallback_final_audit_v1 as final_audit,
)
from jax_experiments.analysis import (
    train_regime_polarity_posterior as controller_loader,
)
from jax_experiments.common.causal_fallback import (
    initial_fallback_state,
    update_fallback_state,
)
from jax_experiments.train import (
    _reset_eval_switch_schedule,
    _select_eval_switch_sequence,
    make_env,
)


def _one_hot(mode: int) -> np.ndarray:
    output = np.zeros((len(protocol.MODES),), dtype=np.float32)
    output[int(mode)] = 1.0
    return output


def _zero() -> np.ndarray:
    return np.zeros((len(protocol.MODES),), dtype=np.float32)


def _uniform() -> np.ndarray:
    return np.full(
        (len(protocol.MODES),), 1.0 / len(protocol.MODES),
        dtype=np.float32,
    )


def _load_stack(student_seed: int):
    frozen = protocol.frozen
    config, robust_agent, robust_params = controller_loader._load_controller(
        frozen.development.ensemble.final,
        frozen.ROBUST_SEED,
        "robust",
    )
    student, student_params, manifest = final_audit._load_student(
        student_seed, robust_agent.obs_dim, robust_agent.act_dim)
    return {
        "config": config,
        "robust_action": final_audit._robust_action_fn(
            robust_agent, robust_params),
        "student_action": model_lib.build_student_action(student),
        "student_params": student_params,
        "student_manifest": manifest,
        "estimator": model_lib.make_estimator(
            robust_agent.obs_dim, robust_agent.act_dim),
    }


def _adaptive_action(stack, observation, context) -> np.ndarray:
    return np.asarray(
        stack["student_action"](
            stack["student_params"],
            jnp.asarray(observation, dtype=jnp.float32),
            jnp.asarray(context, dtype=jnp.float32),
        ),
        dtype=np.float32,
    )


def _uses_estimator(arm: str) -> bool:
    return arm in (protocol.LEARNED_ARM, protocol.FALLBACK_ARM)


def _fixed_context(arm: str) -> int | None:
    if arm not in protocol.FIXED_ARMS:
        return None
    return int(arm.rsplit("_", 1)[1])


def _context_for_arm(
    arm: str,
    mode: int,
    posterior: np.ndarray,
    event_seed: int,
    delayed_mode: int | None,
) -> np.ndarray | None:
    if arm == protocol.ROBUST_ARM or arm == protocol.FALLBACK_ARM:
        return None
    if arm == protocol.LEARNED_ARM:
        return posterior
    if arm == protocol.TRUE_ARM:
        return _one_hot(mode)
    if arm == protocol.ZERO_ARM:
        return _zero()
    if arm == protocol.UNIFORM_ARM:
        return _uniform()
    fixed = _fixed_context(arm)
    if fixed is not None:
        return _one_hot(fixed)
    if arm == protocol.CYCLIC_ARM:
        return _one_hot((int(mode) + 1) % len(protocol.MODES))
    if arm == protocol.SHUFFLED_ARM:
        return _one_hot(protocol.shuffled_mode_map(event_seed)[int(mode)])
    if arm in protocol.DELAY_ARMS:
        if delayed_mode is None:
            raise ValueError("delayed arm did not initialize a context mode")
        return _one_hot(delayed_mode)
    raise ValueError(f"unsupported mechanism arm {arm!r}")


def _action(
    stack,
    arm: str,
    observation,
    mode: int,
    posterior: np.ndarray,
    event_seed: int,
    delayed_mode: int | None,
    fallback_state,
) -> tuple[np.ndarray, str, np.ndarray | None]:
    context = _context_for_arm(
        arm, mode, posterior, event_seed, delayed_mode)
    if arm == protocol.ROBUST_ARM:
        return stack["robust_action"](observation), "robust", None
    if arm == protocol.FALLBACK_ARM:
        if fallback_state.fallback:
            return stack["robust_action"](observation), "robust", None
        return (
            _adaptive_action(stack, observation, posterior),
            "adaptive",
            posterior,
        )
    return _adaptive_action(stack, observation, context), "adaptive", context


def _posterior_metrics(rows, labels) -> dict[str, float] | None:
    if not rows:
        return None
    return protocol.frozen.development.ensemble.final.posterior_metrics(
        np.asarray(rows), np.asarray(labels, dtype=np.int32))


def _stationary_arm(stack, arm: str, event_seed: int) -> list[dict[str, Any]]:
    rows = []
    for mode in protocol.MODES:
        run_config = copy.deepcopy(stack["config"])
        run_config.stochastic_mode_fixed_id = int(mode)
        env = make_env(
            run_config,
            seed_offset=int(event_seed) - int(run_config.seed),
        )
        tasks = env.sample_tasks(len(protocol.MODES))
        env.set_nonstationary_para(tasks)
        env.set_task(tasks[int(mode)])
        returns = []
        terminations = []
        fallback_actions = 0
        total_actions = 0
        posterior_rows = []
        posterior_labels = []
        try:
            for _ in range(protocol.STATIONARY_EPISODES):
                observation = env.reset()
                estimator_state = stack["estimator"].initial_state()
                fallback_state = initial_fallback_state()
                episode_return = 0.0
                terminated = False
                for _ in range(protocol.MAX_EPISODE_STEPS):
                    posterior = np.asarray(
                        stack["estimator"].probabilities(estimator_state),
                        dtype=np.float32,
                    )
                    action, source, _ = _action(
                        stack, arm, observation, int(mode), posterior,
                        event_seed, None, fallback_state)
                    next_observation, reward, done, info = env.step(action)
                    if int(info["mode_used"]) != int(mode):
                        raise RuntimeError("stationary mechanism mode changed")
                    if _uses_estimator(arm):
                        posterior_rows.append(posterior.copy())
                        posterior_labels.append(int(mode))
                        next_state, evidence, _, _ = stack["estimator"].step(
                            estimator_state,
                            observation,
                            action,
                            reward,
                            next_observation,
                        )
                        if arm == protocol.FALLBACK_ARM:
                            fallback_state = update_fallback_state(
                                fallback_state,
                                posterior,
                                stack["estimator"].probabilities(next_state),
                                evidence,
                                protocol.frozen.FALLBACK_CONFIG,
                            )
                        estimator_state = next_state
                    fallback_actions += int(source == "robust")
                    total_actions += 1
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
        row: dict[str, Any] = {
            "arm": arm,
            "mode": int(mode),
            "returns": returns,
            "return_mean": float(np.mean(returns)),
            "terminated_rate": float(np.mean(terminations)),
            "total_actions": int(total_actions),
        }
        metrics = _posterior_metrics(posterior_rows, posterior_labels)
        if metrics is not None:
            row["posterior_metrics"] = metrics
        if arm == protocol.FALLBACK_ARM:
            row["fallback_action_fraction"] = float(
                fallback_actions / max(total_actions, 1))
        rows.append(row)
    return rows


def _transient_bin(offset: int) -> str | None:
    for start, end in protocol.TRANSIENT_BINS:
        if start <= int(offset) < end:
            return f"{start}:{end}"
    return None


def _switching_arm(stack, arm: str, event_seed: int) -> dict[str, Any]:
    delay = protocol.delay_for_arm(arm)
    run_config = copy.deepcopy(stack["config"])
    run_config.stochastic_mode_fixed_id = -1
    env = make_env(
        run_config,
        seed_offset=int(event_seed) - int(run_config.seed),
    )
    tasks = env.sample_tasks(len(protocol.MODES))
    returns = []
    terminations = []
    trace = []
    posterior_rows = []
    posterior_labels = []
    fallback_actions = 0
    total_actions = 0
    context_correct = 0
    context_count = 0
    switch_count = 0
    mode_rewards = {
        int(mode): {"sum": 0.0, "count": 0} for mode in protocol.MODES}
    transient = {
        f"{start}:{end}": {"sum": 0.0, "count": 0}
        for start, end in protocol.TRANSIENT_BINS
    }
    try:
        for episode in range(protocol.SWITCHING_EPISODES):
            sequence, _ = _select_eval_switch_sequence(env, tasks, episode)
            _reset_eval_switch_schedule(
                env, sequence, run_config, protocol.DWELL_STEPS)
            observation = env.reset()
            estimator_state = stack["estimator"].initial_state()
            fallback_state = initial_fallback_state()
            previous_mode = None
            delayed_mode = None
            pending_mode = None
            delay_remaining = 0
            post_switch_offset = None
            episode_return = 0.0
            terminated = False
            for _ in range(protocol.MAX_EPISODE_STEPS):
                mode = int(env.task_id_for_next_step())
                trace.append(mode)
                if previous_mode is None:
                    delayed_mode = mode
                    pending_mode = mode
                elif mode != previous_mode:
                    switch_count += 1
                    post_switch_offset = 0
                    if delay is not None:
                        pending_mode = mode
                        delay_remaining = int(delay)
                previous_mode = mode

                posterior = np.asarray(
                    stack["estimator"].probabilities(estimator_state),
                    dtype=np.float32,
                )
                action, source, context = _action(
                    stack, arm, observation, mode, posterior, event_seed,
                    delayed_mode, fallback_state)
                if context is not None and float(np.sum(context)) > 1e-6:
                    context_correct += int(int(np.argmax(context)) == mode)
                    context_count += 1
                next_observation, reward, done, info = env.step(action)
                if int(info["mode_used"]) != mode:
                    raise RuntimeError("switching mechanism mode misaligned")

                if _uses_estimator(arm):
                    posterior_rows.append(posterior.copy())
                    posterior_labels.append(mode)
                    next_state, evidence, _, _ = stack["estimator"].step(
                        estimator_state,
                        observation,
                        action,
                        reward,
                        next_observation,
                    )
                    if arm == protocol.FALLBACK_ARM:
                        fallback_state = update_fallback_state(
                            fallback_state,
                            posterior,
                            stack["estimator"].probabilities(next_state),
                            evidence,
                            protocol.frozen.FALLBACK_CONFIG,
                        )
                    estimator_state = next_state

                if delay is not None and pending_mode != delayed_mode:
                    delay_remaining -= 1
                    if delay_remaining <= 0:
                        delayed_mode = pending_mode
                        delay_remaining = 0

                reward_value = float(reward)
                mode_rewards[mode]["sum"] += reward_value
                mode_rewards[mode]["count"] += 1
                if post_switch_offset is not None:
                    key = _transient_bin(post_switch_offset)
                    if key is not None:
                        transient[key]["sum"] += reward_value
                        transient[key]["count"] += 1
                    post_switch_offset += 1
                fallback_actions += int(source == "robust")
                total_actions += 1
                episode_return += reward_value
                observation = next_observation
                if done:
                    terminated = True
                    observation = env.reset()
            returns.append(float(episode_return))
            terminations.append(float(terminated))
    finally:
        if hasattr(env, "close"):
            env.close()

    row: dict[str, Any] = {
        "arm": arm,
        "returns": returns,
        "return_mean": float(np.mean(returns)),
        "terminated_rate": float(np.mean(terminations)),
        "mode_trace_sha256": hashlib.sha256(bytes(trace)).hexdigest(),
        "switch_count": int(switch_count),
        "total_actions": int(total_actions),
        "mode_reward_mean": {
            str(mode): float(value["sum"] / max(value["count"], 1))
            for mode, value in mode_rewards.items()
        },
        "post_switch_reward_mean": {
            key: float(value["sum"] / max(value["count"], 1))
            for key, value in transient.items()
        },
        "post_switch_sample_count": {
            key: int(value["count"]) for key, value in transient.items()
        },
    }
    if context_count:
        row["context_correct_fraction"] = float(
            context_correct / context_count)
    metrics = _posterior_metrics(posterior_rows, posterior_labels)
    if metrics is not None:
        row["posterior_metrics"] = metrics
    if arm == protocol.FALLBACK_ARM:
        row["fallback_action_fraction"] = float(
            fallback_actions / max(total_actions, 1))
    if delay is not None:
        row["delay_steps"] = int(delay)
    return row


def evaluate(student_seed: int, event_seed: int) -> dict[str, Any]:
    protocol.validate_registration()
    student_seed = protocol.require_student_seed(student_seed)
    event_seed = protocol.require_event_seed(event_seed)
    stack = _load_stack(student_seed)
    stationary = []
    for arm in protocol.STATIONARY_ARMS:
        stationary.extend(_stationary_arm(stack, arm, event_seed))
        print(
            f"mechanism stationary student={student_seed} "
            f"event={event_seed} arm={arm} complete",
            flush=True,
        )
    switching = []
    for arm in protocol.SWITCHING_ARMS:
        switching.append(_switching_arm(stack, arm, event_seed))
        print(
            f"mechanism switching student={student_seed} "
            f"event={event_seed} arm={arm} complete",
            flush=True,
        )
    return {
        "schema": protocol.EVENT_SCHEMA,
        "status": "complete",
        "identity": protocol.audit_identity(student_seed, event_seed),
        "registration": protocol.registration_record(),
        "stationary": stationary,
        "switching": switching,
    }


def validate_result(payload: dict[str, Any], student_seed: int,
                    event_seed: int) -> None:
    identity = protocol.audit_identity(student_seed, event_seed)
    if (
        payload.get("schema") != protocol.EVENT_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("identity") != identity
        or payload.get("registration") != protocol.registration_record()
    ):
        raise ValueError("invalid final mechanism result identity")
    stationary = payload.get("stationary") or []
    switching = payload.get("switching") or []
    if len(stationary) != len(protocol.STATIONARY_ARMS) * len(protocol.MODES):
        raise ValueError("incomplete stationary mechanism matrix")
    if [row.get("arm") for row in switching] != list(protocol.SWITCHING_ARMS):
        raise ValueError("incomplete switching mechanism matrix")
    trace_hash = None
    for row in switching:
        values = np.asarray(row.get("returns"), dtype=np.float64)
        current_hash = row.get("mode_trace_sha256")
        if (
            values.shape != (protocol.SWITCHING_EPISODES,)
            or not np.all(np.isfinite(values))
            or int(row.get("total_actions", -1))
            != protocol.SWITCHING_EPISODES * protocol.MAX_EPISODE_STEPS
            or not isinstance(current_hash, str)
            or len(current_hash) != 64
        ):
            raise ValueError("invalid switching mechanism output")
        if trace_hash is None:
            trace_hash = current_hash
        elif current_hash != trace_hash:
            raise ValueError("mechanism arms used different mode streams")
    for row in stationary:
        values = np.asarray(row.get("returns"), dtype=np.float64)
        if (
            row.get("arm") not in protocol.STATIONARY_ARMS
            or int(row.get("mode", -1)) not in protocol.MODES
            or values.shape != (protocol.STATIONARY_EPISODES,)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("invalid stationary mechanism output")


def validate_manifest(student_seed: int, event_seed: int) -> dict[str, Any]:
    destination = protocol.audit_dir(student_seed, event_seed)
    manifest = protocol.read_json(destination / "audit_manifest.json")
    result = protocol.read_json(destination / "results.json")
    validate_result(result, student_seed, event_seed)
    expected_identity = protocol.audit_identity(student_seed, event_seed)
    if (
        manifest.get("schema") != protocol.AUDIT_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("identity") != expected_identity
        or manifest.get("registration") != protocol.registration_record()
        or manifest.get("student_manifest") != protocol.file_record(
            protocol.frozen.model_manifest("mode_heads", student_seed))
        or manifest.get("student_parameters") != protocol.file_record(
            protocol.frozen.model_path("mode_heads", student_seed))
        or manifest.get("result_file") != protocol.file_record(
            destination / "results.json")
    ):
        raise ValueError(f"invalid final mechanism manifest: {destination}")
    return manifest


def run(student_seed: int, event_seed: int) -> None:
    protocol.validate_registration()
    student_seed = protocol.require_student_seed(student_seed)
    event_seed = protocol.require_event_seed(event_seed)
    destination = protocol.audit_dir(student_seed, event_seed)
    if (destination / "audit_manifest.json").is_file():
        try:
            validate_manifest(student_seed, event_seed)
        except (KeyError, OSError, TypeError, ValueError):
            pass
        else:
            print(f"FINAL MECHANISM ALREADY COMPLETE: {destination}")
            return
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp.", dir=destination.parent))
    try:
        payload = evaluate(student_seed, event_seed)
        protocol.write_json_atomic(temporary / "results.json", payload)
        manifest = {
            "schema": protocol.AUDIT_SCHEMA,
            "status": "complete",
            "identity": protocol.audit_identity(student_seed, event_seed),
            "registration": protocol.registration_record(),
            "student_manifest": protocol.file_record(
                protocol.frozen.model_manifest("mode_heads", student_seed)),
            "student_parameters": protocol.file_record(
                protocol.frozen.model_path("mode_heads", student_seed)),
            "frozen_estimator_manifest": protocol.file_record(
                protocol.frozen.ensemble.final.MODEL_MANIFEST),
            "frozen_estimator_parameters": protocol.file_record(
                protocol.frozen.ensemble.final.MODEL_PATH),
            "result_file": protocol.file_record(temporary / "results.json"),
        }
        protocol.write_json_atomic(
            temporary / "audit_manifest.json", manifest)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_manifest(student_seed, event_seed)
    print(
        f"FINAL MECHANISM COMPLETE: student={student_seed} "
        f"event={event_seed}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--student-seed", type=int, choices=protocol.STUDENT_SEEDS,
        required=True)
    parser.add_argument(
        "--event-seed", type=int, choices=protocol.EVENT_SEEDS,
        required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.student_seed, args.event_seed)


if __name__ == "__main__":
    main()
