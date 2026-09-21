"""Run one frozen-policy Ant oracle action-compensation audit."""
from __future__ import annotations

import argparse
import copy
import hashlib
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_ant_action_compensation_v29 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_action_compensation_structural_v28 as structural,
)
from jax_experiments.analysis import (
    run_regime_polarity_ant_full_state_audit_v22 as v22_audit,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_safe_utility_audit_v8 as utility_audit,
)
from jax_experiments.common.checkpoint import _patch_flax_variablestate_unpickle
from jax_experiments.train import make_env


def _mean(values) -> float:
    return float(np.mean([float(value) for value in values]))


def _load_stack(seed: int) -> dict[str, Any]:
    _patch_flax_variablestate_unpickle()
    v22_audit._bind()
    controllers = v22_audit.base._load_controllers(
        protocol.policy_parent.CONTROL_VARIANT, seed)
    return utility_audit._stack_from_controllers(controllers)


def _select_action(
    stack: dict[str, Any], arm: str, observation: np.ndarray,
    mode: int, reference_mode: int,
) -> tuple[np.ndarray, float | None]:
    actions = stack["actions"]
    if arm == protocol.ROBUST_ARM:
        return np.asarray(actions["robust_sac"](observation)), None
    if arm == protocol.DYNAMIC_BANK_ARM:
        return np.asarray(actions[f"specialist_{mode}"](observation)), None
    reference = np.asarray(
        actions[f"specialist_{reference_mode}"](observation))
    if arm == protocol.NO_COMPENSATION_ARM:
        return reference, None
    if arm != protocol.ORACLE_COMPENSATION_ARM:
        raise ValueError(f"unknown V29 arm {arm!r}")
    command = protocol.compensate_action(reference, reference_mode, mode)
    gains = np.asarray(
        protocol.mode_gain_vectors(int(command.shape[-1])), dtype=np.float32)
    ideal_signal = gains[reference_mode] * reference
    actual_signal = gains[mode] * command
    return command, float(np.mean(np.abs(actual_signal - ideal_signal)))


def _rollout_summary(
    returns: list[float], termination_counts: list[int],
    first_termination_steps: list[int], total_actions: int,
    signal_errors: list[float],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "returns": returns,
        "return_mean": _mean(returns),
        "termination_counts": termination_counts,
        "termination_count": int(sum(termination_counts)),
        "terminated_rate": _mean(count > 0 for count in termination_counts),
        "first_termination_steps": first_termination_steps,
        "mean_first_termination_step": _mean(first_termination_steps),
        "survival_fraction_before_first_termination": float(
            _mean(first_termination_steps) / protocol.MAX_EPISODE_STEPS),
        "total_actions": int(total_actions),
    }
    if signal_errors:
        row["mean_abs_execution_signal_error"] = _mean(signal_errors)
        row["max_abs_execution_signal_error"] = float(max(signal_errors))
    return row


def _stationary(
    stack: dict[str, Any], arm: str, reference_mode: int,
    event_seed: int, mode: int, episodes: int,
) -> dict[str, Any]:
    mode = protocol.require_mode(mode)
    config = copy.deepcopy(stack["config"])
    config.stochastic_mode_fixed_id = mode
    env = make_env(config, seed_offset=int(event_seed) - int(config.seed))
    tasks = env.sample_tasks(len(protocol.MODES))
    env.set_nonstationary_para(tasks)
    env.set_task(tasks[mode])
    returns: list[float] = []
    termination_counts: list[int] = []
    first_termination_steps: list[int] = []
    signal_errors: list[float] = []
    total_actions = 0
    try:
        for _ in range(int(episodes)):
            observation = env.reset()
            episode_return = 0.0
            terminations = 0
            first_termination = protocol.MAX_EPISODE_STEPS
            for step in range(protocol.MAX_EPISODE_STEPS):
                action, signal_error = _select_action(
                    stack, arm, observation, mode, reference_mode)
                next_observation, reward, done, info = env.step(action)
                if int(info["mode_used"]) != mode:
                    raise RuntimeError("V29 stationary mode changed")
                if signal_error is not None:
                    signal_errors.append(signal_error)
                total_actions += 1
                episode_return += float(reward)
                observation = next_observation
                if done:
                    terminations += 1
                    if terminations == 1:
                        first_termination = step + 1
                    observation = env.reset()
            returns.append(float(episode_return))
            termination_counts.append(int(terminations))
            first_termination_steps.append(int(first_termination))
    finally:
        if hasattr(env, "close"):
            env.close()
    return _rollout_summary(
        returns, termination_counts, first_termination_steps,
        total_actions, signal_errors)


def _switching(
    stack: dict[str, Any], arm: str, reference_mode: int, event_seed: int,
) -> dict[str, Any]:
    event_seed = protocol.require_switching_event_seed(event_seed)
    config = copy.deepcopy(stack["config"])
    config.stochastic_mode_fixed_id = -1
    env = make_env(config, seed_offset=event_seed - int(config.seed))
    tasks = env.sample_tasks(len(protocol.MODES))
    returns: list[float] = []
    termination_counts: list[int] = []
    first_termination_steps: list[int] = []
    signal_errors: list[float] = []
    trace: list[int] = []
    mode_counts = {str(mode): 0 for mode in protocol.MODES}
    episode_sequences = []
    total_actions = 0
    try:
        configure = getattr(env, "configure_eval_mode_sequence", None)
        if not callable(configure):
            raise RuntimeError("V29 requires explicit mode schedules")
        for episode in range(protocol.SWITCHING_EPISODES):
            sequence = protocol.switching_sequence(event_seed, episode)
            episode_sequences.append(list(sequence))
            configure(tasks, sequence, protocol.DWELL_STEPS)
            observation = env.reset()
            episode_return = 0.0
            terminations = 0
            first_termination = protocol.MAX_EPISODE_STEPS
            for step in range(protocol.MAX_EPISODE_STEPS):
                mode = int(env.task_id_for_next_step())
                trace.append(mode)
                mode_counts[str(mode)] += 1
                action, signal_error = _select_action(
                    stack, arm, observation, mode, reference_mode)
                next_observation, reward, done, info = env.step(action)
                if int(info["mode_used"]) != mode:
                    raise RuntimeError("V29 switching action used a wrong mode")
                if signal_error is not None:
                    signal_errors.append(signal_error)
                total_actions += 1
                episode_return += float(reward)
                observation = next_observation
                if done:
                    terminations += 1
                    if terminations == 1:
                        first_termination = step + 1
                    observation = env.reset()
            returns.append(float(episode_return))
            termination_counts.append(int(terminations))
            first_termination_steps.append(int(first_termination))
    finally:
        if hasattr(env, "close"):
            env.close()
    row = _rollout_summary(
        returns, termination_counts, first_termination_steps,
        total_actions, signal_errors)
    row.update({
        "mode_trace_sha256": hashlib.sha256(bytes(trace)).hexdigest(),
        "mode_counts": mode_counts,
        "base_schedule": list(protocol.SWITCHING_SCHEDULES[event_seed]),
        "episode_sequences": episode_sequences,
    })
    return row


def _reference_calibration(
    stack: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], int]:
    events: dict[str, Any] = {}
    for event_seed in protocol.CALIBRATION_EVENT_SEEDS:
        events[str(event_seed)] = {}
        for mode in protocol.MODES:
            events[str(event_seed)][str(mode)] = _stationary(
                stack, protocol.DYNAMIC_BANK_ARM, mode, event_seed, mode,
                protocol.CALIBRATION_EPISODES_PER_MODE)
    matrix = {}
    for mode in protocol.MODES:
        rows = [events[str(seed)][str(mode)]
                for seed in protocol.CALIBRATION_EVENT_SEEDS]
        matrix[str(mode)] = {
            "native_return_mean": _mean(row["return_mean"] for row in rows),
            "native_terminated_rate": _mean(
                row["terminated_rate"] for row in rows),
            "native_termination_count": int(sum(
                int(row["termination_count"]) for row in rows)),
            "event_returns": {
                str(event_seed): float(row["return_mean"])
                for event_seed, row in zip(
                    protocol.CALIBRATION_EVENT_SEEDS, rows)
            },
        }

    def score(mode: int) -> tuple[float, int]:
        row = matrix[str(mode)]
        utility = (
            float(row["native_return_mean"])
            - 1_000_000.0 * float(row["native_terminated_rate"])
        )
        return utility, -int(mode)

    selected = max(protocol.MODES, key=score)
    return events, matrix, int(selected)


def _identity(seed: int) -> dict[str, Any]:
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "benchmark_role": "ant_oracle_action_compensation_development_audit",
        "training_seed": protocol.require_training_seed(seed),
        "calibration_event_seeds": list(protocol.CALIBRATION_EVENT_SEEDS),
        "stationary_holdout_event_seeds": list(
            protocol.STATIONARY_HOLDOUT_EVENT_SEEDS),
        "switching_event_seeds": list(protocol.SWITCHING_EVENT_SEEDS),
        "switching_schedules": {
            str(key): list(value)
            for key, value in protocol.SWITCHING_SCHEDULES.items()
        },
        "arms": list(protocol.ARMS),
    }


def evaluate(seed: int) -> dict[str, Any]:
    seed = protocol.require_training_seed(seed)
    structural.validate_structural_audit()
    stack = _load_stack(seed)
    calibration_events, calibration_matrix, reference_mode = (
        _reference_calibration(stack))
    stationary: dict[str, Any] = {}
    for event_seed in protocol.STATIONARY_HOLDOUT_EVENT_SEEDS:
        stationary[str(event_seed)] = {
            arm: {
                str(mode): _stationary(
                    stack, arm, reference_mode, event_seed, mode,
                    protocol.STATIONARY_EPISODES_PER_MODE)
                for mode in protocol.MODES
            }
            for arm in protocol.ARMS
        }
        print(f"V29 stationary event={event_seed} complete", flush=True)
    switching: dict[str, Any] = {}
    equivalence: dict[str, Any] = {}
    event_hashes = set()
    for event_seed in protocol.SWITCHING_EVENT_SEEDS:
        rows = {
            arm: _switching(stack, arm, reference_mode, event_seed)
            for arm in protocol.ARMS
        }
        hashes = {str(row["mode_trace_sha256"]) for row in rows.values()}
        if len(hashes) != 1:
            raise RuntimeError("V29 switching arms used different mode streams")
        event_hashes.update(hashes)
        native = _stationary(
            stack, protocol.NO_COMPENSATION_ARM, reference_mode,
            event_seed, reference_mode, protocol.SWITCHING_EPISODES)
        oracle = rows[protocol.ORACLE_COMPENSATION_ARM]
        return_error = float(np.max(np.abs(
            np.asarray(oracle["returns"], dtype=np.float64)
            - np.asarray(native["returns"], dtype=np.float64))))
        equivalence[str(event_seed)] = {
            "native_reference_mode": int(reference_mode),
            "native_returns": native["returns"],
            "oracle_compensation_returns": oracle["returns"],
            "native_termination_counts": native["termination_counts"],
            "oracle_termination_counts": oracle["termination_counts"],
            "max_abs_return_error": return_error,
            "pass": bool(
                return_error <= protocol.EXACT_RETURN_ATOL
                and native["termination_counts"]
                == oracle["termination_counts"]
                and native["first_termination_steps"]
                == oracle["first_termination_steps"]
            ),
        }
        switching[str(event_seed)] = rows
        print(f"V29 switching event={event_seed} complete", flush=True)
    if len(event_hashes) != len(protocol.SWITCHING_EVENT_SEEDS):
        raise RuntimeError("V29 switching event streams are not distinct")
    return {
        "schema": protocol.AUDIT_SCHEMA,
        "status": "complete",
        "identity": _identity(seed),
        "registration": protocol.file_record(protocol.REGISTRATION_PATH),
        "structural_audit": protocol.file_record(
            structural.protocol.STRUCTURAL_MANIFEST),
        "frozen_inputs": protocol.frozen_input_records(seed),
        "calibration": {
            "reference_events": calibration_events,
            "reference_matrix": calibration_matrix,
            "selected_reference_mode": int(reference_mode),
        },
        "stationary_holdout": stationary,
        "switching_holdout": switching,
        "oracle_equivalence": equivalence,
    }


def _validate_rollout(row: dict[str, Any], episodes: int) -> None:
    returns = row.get("returns") or []
    counts = row.get("termination_counts") or []
    first = row.get("first_termination_steps") or []
    if (
        len(returns) != episodes
        or len(counts) != episodes
        or len(first) != episodes
        or not all(math.isfinite(float(value)) for value in returns)
        or any(int(value) < 0 for value in counts)
        or any(not 1 <= int(value) <= protocol.MAX_EPISODE_STEPS
               for value in first)
        or int(row.get("termination_count", -1)) != sum(map(int, counts))
        or int(row.get("total_actions", -1))
        != episodes * protocol.MAX_EPISODE_STEPS
    ):
        raise ValueError("invalid V29 rollout")


def validate_result(payload: dict[str, Any], seed: int) -> None:
    seed = protocol.require_training_seed(seed)
    if (
        payload.get("schema") != protocol.AUDIT_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("identity") != _identity(seed)
        or payload.get("registration")
        != protocol.file_record(protocol.REGISTRATION_PATH)
        or payload.get("structural_audit")
        != protocol.file_record(structural.protocol.STRUCTURAL_MANIFEST)
        or payload.get("frozen_inputs") != protocol.frozen_input_records(seed)
    ):
        raise ValueError("invalid V29 audit identity")
    selected = int(payload["calibration"].get("selected_reference_mode", -1))
    if selected not in protocol.MODES:
        raise ValueError("V29 reference selection is missing")
    stationary = payload.get("stationary_holdout") or {}
    if set(stationary) != {
        str(value) for value in protocol.STATIONARY_HOLDOUT_EVENT_SEEDS
    }:
        raise ValueError("incomplete V29 stationary holdout")
    for event in stationary.values():
        if set(event) != set(protocol.ARMS):
            raise ValueError("incomplete V29 stationary arms")
        for arm in protocol.ARMS:
            if set(event[arm]) != {str(mode) for mode in protocol.MODES}:
                raise ValueError("incomplete V29 stationary modes")
            for row in event[arm].values():
                _validate_rollout(row, protocol.STATIONARY_EPISODES_PER_MODE)
    switching = payload.get("switching_holdout") or {}
    equivalence = payload.get("oracle_equivalence") or {}
    expected_events = {str(value) for value in protocol.SWITCHING_EVENT_SEEDS}
    if set(switching) != expected_events or set(equivalence) != expected_events:
        raise ValueError("incomplete V29 switching holdout")
    expected_count = (
        protocol.SWITCHING_EPISODES * protocol.MAX_EPISODE_STEPS
        // len(protocol.MODES))
    event_hashes = set()
    for event_seed, event in switching.items():
        if set(event) != set(protocol.ARMS):
            raise ValueError("incomplete V29 switching arms")
        hashes = set()
        for row in event.values():
            _validate_rollout(row, protocol.SWITCHING_EPISODES)
            if (
                row.get("mode_counts") != {
                    str(mode): expected_count for mode in protocol.MODES}
                or row.get("base_schedule")
                != list(protocol.SWITCHING_SCHEDULES[int(event_seed)])
            ):
                raise ValueError("invalid V29 switching schedule")
            hashes.add(str(row.get("mode_trace_sha256") or ""))
        if len(hashes) != 1 or "" in hashes:
            raise ValueError("V29 switching streams differ across arms")
        event_hashes.update(hashes)
        oracle = event[protocol.ORACLE_COMPENSATION_ARM]
        if float(oracle.get("max_abs_execution_signal_error", math.inf)) > (
            protocol.EXACT_ACTION_ATOL
        ):
            raise ValueError("V29 oracle compensation is not exact")
        if equivalence[event_seed].get("pass") is not True:
            raise ValueError("V29 oracle trajectory equivalence failed")
    if len(event_hashes) != len(protocol.SWITCHING_EVENT_SEEDS):
        raise ValueError("V29 switching streams are not distinct")


def validate_audit(seed: int) -> dict[str, Any]:
    seed = protocol.require_training_seed(seed)
    manifest = protocol.read_json(protocol.audit_manifest(seed))
    payload = protocol.read_json(protocol.audit_result(seed))
    validate_result(payload, seed)
    if (
        manifest.get("schema") != protocol.AUDIT_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("identity") != _identity(seed)
        or manifest.get("registration")
        != protocol.file_record(protocol.REGISTRATION_PATH)
        or manifest.get("structural_audit")
        != protocol.file_record(structural.protocol.STRUCTURAL_MANIFEST)
        or manifest.get("frozen_inputs") != protocol.frozen_input_records(seed)
        or manifest.get("audit") != protocol.file_record(
            protocol.audit_result(seed))
    ):
        raise ValueError("invalid V29 audit manifest")
    return manifest


def run(seed: int) -> None:
    seed = protocol.require_training_seed(seed)
    protocol.validate_registration()
    structural.validate_structural_audit()
    destination = protocol.audit_dir(seed)
    if protocol.audit_manifest(seed).is_file():
        try:
            validate_audit(seed)
        except (KeyError, OSError, TypeError, ValueError):
            pass
        else:
            print(f"V29 AUDIT ALREADY COMPLETE: seed={seed}", flush=True)
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
        payload = evaluate(seed)
        validate_result(payload, seed)
        protocol.write_json_atomic(temporary / "audit.json", payload)
        protocol.write_json_atomic(
            temporary / "audit_manifest.json",
            {
                "schema": protocol.AUDIT_SCHEMA,
                "status": "complete",
                "identity": _identity(seed),
                "registration": protocol.file_record(
                    protocol.REGISTRATION_PATH),
                "structural_audit": protocol.file_record(
                    structural.protocol.STRUCTURAL_MANIFEST),
                "frozen_inputs": protocol.frozen_input_records(seed),
                "audit": protocol.file_record(temporary / "audit.json"),
            },
        )
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_audit(seed)
    print(f"V29 AUDIT COMPLETE: seed={seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for idempotent execution")
    run(args.seed)


if __name__ == "__main__":
    main()
