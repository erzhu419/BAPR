"""Run the frozen V28 single-policy polarity-compensation audit."""
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
    regime_polarity_action_compensation_v28 as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_system_id_v5 as v5_model,
)
from jax_experiments.analysis import (
    run_regime_polarity_action_compensation_structural_v28 as structural,
)
from jax_experiments.analysis import (
    run_regime_polarity_full_state_confirmation_audit_v21 as v21_audit,
)
from jax_experiments.analysis import (
    run_regime_polarity_v5_final_comparison_audit_v18 as base,
)
from jax_experiments.common.checkpoint import _patch_flax_variablestate_unpickle
from jax_experiments.train import make_env


def _mean(values) -> float:
    return float(np.mean([float(value) for value in values]))


def _bind() -> None:
    # V21 names this schema BASELINE_BUNDLE_SCHEMA while its inherited V18
    # validator reads BUNDLE_SCHEMA. Supply the identical frozen value at
    # runtime so validation remains active without editing the V21 source.
    if not hasattr(protocol.parent, "BUNDLE_SCHEMA"):
        protocol.parent.BUNDLE_SCHEMA = protocol.parent.BASELINE_BUNDLE_SCHEMA
    v21_audit._bind()
    base.protocol = protocol


def _load_stacks(seed: int):
    _bind()
    _patch_flax_variablestate_unpickle()
    frozen_controllers = v21_audit._load_frozen_controllers(seed)
    sac5_controllers = base._load_sac5_controllers(
        seed, frozen_controllers)
    bapr_stack, sac5_stack = base._action_stacks(
        frozen_controllers, sac5_controllers)
    bapr_stack["reference_agent"] = frozen_controllers["robust_sac"]["agent"]
    sac5_stack["reference_agent"] = sac5_controllers["robust_sac"]["agent"]
    return bapr_stack, sac5_stack


def _reference_calibration(
    stack: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], int]:
    events: dict[str, Any] = {}
    for event_seed in protocol.CALIBRATION_EVENT_SEEDS:
        events[str(event_seed)] = {}
        for mode in protocol.MODES:
            events[str(event_seed)][str(mode)] = base._stationary_direct(
                stack["config"], stack["actions"][f"specialist_{mode}"],
                event_seed, mode, protocol.CALIBRATION_EPISODES_PER_MODE)
    matrix = {}
    for mode in protocol.MODES:
        rows = [events[str(seed)][str(mode)]
                for seed in protocol.CALIBRATION_EVENT_SEEDS]
        matrix[str(mode)] = {
            "native_return_mean": _mean(row["return_mean"] for row in rows),
            "native_terminated_rate": _mean(
                row["terminated_rate"] for row in rows),
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


def _posterior_metrics(
    posterior_rows: list[np.ndarray], labels: list[int],
) -> dict[str, float]:
    posterior = np.asarray(posterior_rows, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int32)
    target = np.eye(len(protocol.MODES), dtype=np.float64)[truth]
    return {
        "mode_accuracy": float(np.mean(np.argmax(posterior, axis=1) == truth)),
        "brier_score": float(np.mean(np.sum(
            (posterior - target) ** 2, axis=1))),
    }


def _compensated_stationary(
    stack: dict[str, Any],
    reference_mode: int,
    route: str,
    event_seed: int,
    mode: int,
) -> dict[str, Any]:
    if route not in {"true_mode", "v5_posterior"}:
        raise ValueError(f"unsupported V28 compensation route {route!r}")
    config = copy.deepcopy(stack["config"])
    config.stochastic_mode_fixed_id = int(mode)
    env = make_env(config, seed_offset=int(event_seed) - int(config.seed))
    tasks = env.sample_tasks(len(protocol.MODES))
    env.set_nonstationary_para(tasks)
    env.set_task(tasks[int(mode)])
    action_fn = stack["actions"][f"specialist_{reference_mode}"]
    estimator = (
        v5_model.make_estimator(int(env.obs_dim), int(env.act_dim))
        if route == "v5_posterior" else None)
    gains = np.asarray(protocol.mode_gain_vectors(env.act_dim), dtype=np.float32)
    returns = []
    terminations = []
    posterior_rows: list[np.ndarray] = []
    labels: list[int] = []
    route_correct = 0
    total_actions = 0
    signal_error_sum = 0.0
    signal_error_max = 0.0
    try:
        for _ in range(protocol.STATIONARY_EPISODES_PER_MODE):
            observation = env.reset()
            estimator_state = (
                estimator.initial_state() if estimator is not None else None)
            episode_return = 0.0
            terminated = False
            for _ in range(protocol.MAX_EPISODE_STEPS):
                if estimator is None:
                    estimated_mode = int(mode)
                else:
                    posterior = np.asarray(
                        estimator.probabilities(estimator_state),
                        dtype=np.float64)
                    posterior_rows.append(posterior.copy())
                    labels.append(int(mode))
                    estimated_mode = int(np.argmax(posterior))
                reference_action = np.asarray(action_fn(observation))
                command = protocol.compensate_action(
                    reference_action, reference_mode, estimated_mode)
                ideal_signal = gains[reference_mode] * reference_action
                actual_signal = gains[int(mode)] * command
                signal_error = float(np.mean(np.abs(
                    actual_signal - ideal_signal)))
                signal_error_sum += signal_error
                signal_error_max = max(signal_error_max, signal_error)
                next_observation, reward, done, info = env.step(command)
                if int(info["mode_used"]) != int(mode):
                    raise RuntimeError("V28 stationary mode changed")
                if estimator is not None:
                    estimator_state, _, _, _ = estimator.step(
                        estimator_state, observation, command, reward,
                        next_observation)
                route_correct += int(estimated_mode == int(mode))
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
    result: dict[str, Any] = {
        "returns": returns,
        "return_mean": _mean(returns),
        "terminated_rate": _mean(terminations),
        "total_actions": int(total_actions),
        "routing_mode_accuracy": float(route_correct / max(total_actions, 1)),
        "mean_abs_execution_signal_error": float(
            signal_error_sum / max(total_actions, 1)),
        "max_abs_execution_signal_error": float(signal_error_max),
    }
    if posterior_rows:
        result["posterior_metrics"] = _posterior_metrics(
            posterior_rows, labels)
    return result


def _compensated_switching(
    stack: dict[str, Any],
    reference_mode: int,
    route: str,
    event_seed: int,
) -> dict[str, Any]:
    if route not in {"true_mode", "v5_posterior"}:
        raise ValueError(f"unsupported V28 compensation route {route!r}")
    event_seed = protocol.require_switching_event_seed(event_seed)
    config = copy.deepcopy(stack["config"])
    config.stochastic_mode_fixed_id = -1
    env = make_env(config, seed_offset=event_seed - int(config.seed))
    tasks = env.sample_tasks(len(protocol.MODES))
    action_fn = stack["actions"][f"specialist_{reference_mode}"]
    estimator = (
        v5_model.make_estimator(int(env.obs_dim), int(env.act_dim))
        if route == "v5_posterior" else None)
    gains = np.asarray(protocol.mode_gain_vectors(env.act_dim), dtype=np.float32)
    returns = []
    terminations = []
    trace: list[int] = []
    mode_counts = {str(mode): 0 for mode in protocol.MODES}
    episode_sequences = []
    posterior_rows: list[np.ndarray] = []
    labels: list[int] = []
    route_correct = 0
    total_actions = 0
    signal_error_sum = 0.0
    signal_error_max = 0.0
    switch_delays: list[int] = []
    window_totals = {size: 0 for size in (1, 2, 4, 8)}
    window_wrong = {size: 0 for size in (1, 2, 4, 8)}
    try:
        configure = getattr(env, "configure_eval_mode_sequence", None)
        if not callable(configure):
            raise RuntimeError("V28 requires explicit mode schedules")
        for episode in range(protocol.SWITCHING_EPISODES):
            sequence = protocol.switching_sequence(event_seed, episode)
            episode_sequences.append(list(sequence))
            configure(tasks, sequence, protocol.DWELL_STEPS)
            observation = env.reset()
            estimator_state = (
                estimator.initial_state() if estimator is not None else None)
            episode_return = 0.0
            episode_terminated = False
            previous_mode: int | None = None
            switch_offset: int | None = None
            switch_detected = False
            for _ in range(protocol.MAX_EPISODE_STEPS):
                mode = int(env.task_id_for_next_step())
                trace.append(mode)
                mode_counts[str(mode)] += 1
                if previous_mode is not None and mode != previous_mode:
                    if switch_offset is not None and not switch_detected:
                        switch_delays.append(int(switch_offset))
                    switch_offset = 0
                    switch_detected = False
                if estimator is None:
                    estimated_mode = mode
                else:
                    posterior = np.asarray(
                        estimator.probabilities(estimator_state),
                        dtype=np.float64)
                    posterior_rows.append(posterior.copy())
                    labels.append(mode)
                    estimated_mode = int(np.argmax(posterior))
                correct = estimated_mode == mode
                if switch_offset is not None:
                    for size in window_totals:
                        if switch_offset < size:
                            window_totals[size] += 1
                            window_wrong[size] += int(not correct)
                    if correct and not switch_detected:
                        switch_delays.append(int(switch_offset))
                        switch_detected = True
                reference_action = np.asarray(action_fn(observation))
                command = protocol.compensate_action(
                    reference_action, reference_mode, estimated_mode)
                ideal_signal = gains[reference_mode] * reference_action
                actual_signal = gains[mode] * command
                signal_error = float(np.mean(np.abs(
                    actual_signal - ideal_signal)))
                signal_error_sum += signal_error
                signal_error_max = max(signal_error_max, signal_error)
                next_observation, reward, done, info = env.step(command)
                if int(info["mode_used"]) != mode:
                    raise RuntimeError("V28 switching action used a wrong mode")
                if estimator is not None:
                    estimator_state, _, _, _ = estimator.step(
                        estimator_state, observation, command, reward,
                        next_observation)
                route_correct += int(correct)
                total_actions += 1
                episode_return += float(reward)
                observation = next_observation
                previous_mode = mode
                if switch_offset is not None:
                    switch_offset += 1
                if done:
                    episode_terminated = True
                    observation = env.reset()
            if switch_offset is not None and not switch_detected:
                switch_delays.append(int(switch_offset))
            returns.append(float(episode_return))
            terminations.append(float(episode_terminated))
    finally:
        if hasattr(env, "close"):
            env.close()
    result: dict[str, Any] = {
        "returns": returns,
        "return_mean": _mean(returns),
        "terminated_rate": _mean(terminations),
        "total_actions": int(total_actions),
        "mode_trace_sha256": hashlib.sha256(bytes(trace)).hexdigest(),
        "mode_counts": mode_counts,
        "base_schedule": list(protocol.SWITCHING_SCHEDULES[event_seed]),
        "episode_sequences": episode_sequences,
        "routing_mode_accuracy": float(route_correct / max(total_actions, 1)),
        "controller_mismatch_action_fraction": float(
            1.0 - route_correct / max(total_actions, 1)),
        "mean_abs_execution_signal_error": float(
            signal_error_sum / max(total_actions, 1)),
        "max_abs_execution_signal_error": float(signal_error_max),
        "switch_detection_delays": switch_delays,
        "switch_detection_delay_median": (
            float(np.median(switch_delays)) if switch_delays else None),
        "wrong_route_fraction_after_switch": {
            str(size): float(window_wrong[size] / max(window_totals[size], 1))
            for size in window_totals
        },
    }
    if posterior_rows:
        result["posterior_metrics"] = _posterior_metrics(
            posterior_rows, labels)
    return result


def _stationary_holdout(
    bapr_stack: dict[str, Any],
    sac5_stack: dict[str, Any],
    reference_mode: int,
    bapr_map: dict[str, Any],
    sac5_map: dict[str, str],
) -> dict[str, Any]:
    events = {}
    reference_action = bapr_stack["actions"][f"specialist_{reference_mode}"]
    for event_seed in protocol.STATIONARY_HOLDOUT_EVENT_SEEDS:
        arms = {arm: {} for arm in protocol.ARMS}
        for mode in protocol.MODES:
            key = str(mode)
            arms[protocol.ROBUST_ARM][key] = base._stationary_direct(
                bapr_stack["config"], bapr_stack["actions"]["robust_sac"],
                event_seed, mode, protocol.STATIONARY_EPISODES_PER_MODE)
            arms[protocol.NO_COMPENSATION_ARM][key] = base._stationary_direct(
                bapr_stack["config"], reference_action, event_seed, mode,
                protocol.STATIONARY_EPISODES_PER_MODE)
            arms[protocol.ORACLE_COMPENSATION_ARM][key] = (
                _compensated_stationary(
                    bapr_stack, reference_mode, "true_mode", event_seed, mode))
            arms[protocol.CAUSAL_COMPENSATION_ARM][key] = (
                _compensated_stationary(
                    bapr_stack, reference_mode, "v5_posterior",
                    event_seed, mode))
            arms[protocol.V21_BANK_ARM][key] = base._routed_stationary(
                bapr_stack, bapr_map, "v5_posterior", event_seed, mode)
            arms[protocol.SAC5_ARM][key] = base._routed_stationary(
                sac5_stack, sac5_map, "v5_posterior", event_seed, mode)
        events[str(event_seed)] = arms
        print(f"V28 stationary event={event_seed} complete", flush=True)
    return events


def _switching_holdout(
    bapr_stack: dict[str, Any],
    sac5_stack: dict[str, Any],
    reference_mode: int,
    bapr_map: dict[str, Any],
    sac5_map: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    events = {}
    equivalence = {}
    event_hashes = set()
    reference_controller = f"specialist_{reference_mode}"
    for event_seed in protocol.SWITCHING_EVENT_SEEDS:
        rows = {
            protocol.ROBUST_ARM: base._switching_bank(
                bapr_stack, bapr_map, "fixed", event_seed,
                fixed_controller="robust_sac"),
            protocol.NO_COMPENSATION_ARM: base._switching_bank(
                bapr_stack, bapr_map, "fixed", event_seed,
                fixed_controller=reference_controller),
            protocol.ORACLE_COMPENSATION_ARM: _compensated_switching(
                bapr_stack, reference_mode, "true_mode", event_seed),
            protocol.CAUSAL_COMPENSATION_ARM: _compensated_switching(
                bapr_stack, reference_mode, "v5_posterior", event_seed),
            protocol.V21_BANK_ARM: base._switching_bank(
                bapr_stack, bapr_map, "v5_posterior", event_seed),
            protocol.SAC5_ARM: base._switching_bank(
                sac5_stack, sac5_map, "v5_posterior", event_seed),
        }
        hashes = {str(row["mode_trace_sha256"]) for row in rows.values()}
        if len(hashes) != 1:
            raise RuntimeError("V28 switching arms used different mode streams")
        event_hashes.update(hashes)
        native = base._stationary_direct(
            bapr_stack["config"],
            bapr_stack["actions"][reference_controller],
            event_seed, reference_mode, protocol.SWITCHING_EPISODES)
        oracle_returns = np.asarray(
            rows[protocol.ORACLE_COMPENSATION_ARM]["returns"],
            dtype=np.float64)
        native_returns = np.asarray(native["returns"], dtype=np.float64)
        max_error = float(np.max(np.abs(oracle_returns - native_returns)))
        equivalence[str(event_seed)] = {
            "native_reference_mode": int(reference_mode),
            "native_returns": native["returns"],
            "oracle_compensation_returns": (
                rows[protocol.ORACLE_COMPENSATION_ARM]["returns"]),
            "max_abs_return_error": max_error,
            "pass": bool(max_error <= protocol.EXACT_RETURN_ATOL),
        }
        events[str(event_seed)] = rows
        print(f"V28 switching event={event_seed} complete", flush=True)
    if len(event_hashes) != len(protocol.SWITCHING_EVENT_SEEDS):
        raise RuntimeError("V28 switching streams are not distinct")
    return events, equivalence


def _identity(seed: int) -> dict[str, Any]:
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "benchmark_role": "frozen_v21_action_compensation_mechanism_audit",
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
    bapr_stack, sac5_stack = _load_stacks(seed)
    reference_events, reference_matrix, reference_mode = (
        _reference_calibration(bapr_stack))
    bapr_events, bapr_matrix, bapr_map = base._bapr_calibration(bapr_stack)
    sac5_events, sac5_matrix, sac5_static, sac5_map = (
        base._sac5_calibration(sac5_stack))
    switching, oracle_equivalence = _switching_holdout(
        bapr_stack, sac5_stack, reference_mode, bapr_map, sac5_map)
    return {
        "schema": protocol.AUDIT_SCHEMA,
        "status": "complete",
        "identity": _identity(seed),
        "registration": protocol.file_record(protocol.REGISTRATION_PATH),
        "structural_audit": protocol.file_record(
            protocol.STRUCTURAL_MANIFEST),
        "frozen_inputs": protocol.frozen_input_records(seed),
        "calibration": {
            "reference_events": reference_events,
            "reference_matrix": reference_matrix,
            "selected_reference_mode": int(reference_mode),
            "bapr_events": bapr_events,
            "bapr_matrix": bapr_matrix,
            "bapr_utility_map": bapr_map,
            "sac5_events": sac5_events,
            "sac5_matrix": sac5_matrix,
            "sac5_best_static": sac5_static,
            "sac5_mode_map": sac5_map,
        },
        "stationary_holdout": _stationary_holdout(
            bapr_stack, sac5_stack, reference_mode, bapr_map, sac5_map),
        "switching_holdout": switching,
        "oracle_equivalence": oracle_equivalence,
    }


def validate_result(payload: dict[str, Any], seed: int) -> None:
    seed = protocol.require_training_seed(seed)
    if (
        payload.get("schema") != protocol.AUDIT_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("identity") != _identity(seed)
        or payload.get("registration")
        != protocol.file_record(protocol.REGISTRATION_PATH)
        or payload.get("structural_audit")
        != protocol.file_record(protocol.STRUCTURAL_MANIFEST)
        or payload.get("frozen_inputs")
        != protocol.frozen_input_records(seed)
    ):
        raise ValueError("invalid V28 audit identity")
    calibration = payload.get("calibration") or {}
    if int(calibration.get("selected_reference_mode", -1)) not in protocol.MODES:
        raise ValueError("V28 reference selection is missing")
    stationary = payload.get("stationary_holdout") or {}
    if set(stationary) != {
        str(value) for value in protocol.STATIONARY_HOLDOUT_EVENT_SEEDS
    }:
        raise ValueError("incomplete V28 stationary holdout")
    for event in stationary.values():
        if set(event) != set(protocol.ARMS):
            raise ValueError("incomplete V28 stationary arms")
        for arm in protocol.ARMS:
            if set(event[arm]) != {str(mode) for mode in protocol.MODES}:
                raise ValueError("incomplete V28 stationary modes")
            for row in event[arm].values():
                returns = row.get("returns") or []
                if (
                    len(returns) != protocol.STATIONARY_EPISODES_PER_MODE
                    or not all(math.isfinite(float(value)) for value in returns)
                    or int(row.get("total_actions", -1))
                    != (protocol.STATIONARY_EPISODES_PER_MODE
                        * protocol.MAX_EPISODE_STEPS)
                ):
                    raise ValueError("invalid V28 stationary rollout")
    switching = payload.get("switching_holdout") or {}
    equivalence = payload.get("oracle_equivalence") or {}
    expected_events = {str(value) for value in protocol.SWITCHING_EVENT_SEEDS}
    if set(switching) != expected_events or set(equivalence) != expected_events:
        raise ValueError("incomplete V28 switching holdout")
    expected_count = (
        protocol.SWITCHING_EPISODES * protocol.MAX_EPISODE_STEPS
        // len(protocol.MODES))
    event_hashes = set()
    for event_seed, event in switching.items():
        if set(event) != set(protocol.ARMS):
            raise ValueError("incomplete V28 switching arms")
        hashes = set()
        for row in event.values():
            returns = row.get("returns") or []
            if (
                len(returns) != protocol.SWITCHING_EPISODES
                or not all(math.isfinite(float(value)) for value in returns)
                or int(row.get("total_actions", -1))
                != protocol.SWITCHING_EPISODES * protocol.MAX_EPISODE_STEPS
                or row.get("mode_counts") != {
                    str(mode): expected_count for mode in protocol.MODES}
                or row.get("base_schedule")
                != list(protocol.SWITCHING_SCHEDULES[int(event_seed)])
            ):
                raise ValueError("invalid V28 switching rollout")
            hashes.add(str(row.get("mode_trace_sha256") or ""))
        if len(hashes) != 1 or "" in hashes:
            raise ValueError("V28 switching streams differ across arms")
        event_hashes.update(hashes)
        oracle = event[protocol.ORACLE_COMPENSATION_ARM]
        if (
            oracle.get("routing_mode_accuracy") != 1.0
            or float(oracle.get("max_abs_execution_signal_error", math.inf))
            > protocol.EXACT_ACTION_ATOL
        ):
            raise ValueError("V28 oracle compensation is not exact")
        if (
            equivalence[event_seed].get("pass") is not True
            or float(equivalence[event_seed].get(
                "max_abs_return_error", math.inf))
            > protocol.EXACT_RETURN_ATOL
        ):
            raise ValueError("V28 oracle trajectory equivalence failed")
    if len(event_hashes) != len(protocol.SWITCHING_EVENT_SEEDS):
        raise ValueError("V28 switching event traces are not distinct")


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
        != protocol.file_record(protocol.STRUCTURAL_MANIFEST)
        or manifest.get("frozen_inputs")
        != protocol.frozen_input_records(seed)
        or manifest.get("audit")
        != protocol.file_record(protocol.audit_result(seed))
    ):
        raise ValueError("invalid V28 audit manifest")
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
            print(f"V28 AUDIT ALREADY COMPLETE: seed={seed}", flush=True)
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
                    protocol.STRUCTURAL_MANIFEST),
                "frozen_inputs": protocol.frozen_input_records(seed),
                "audit": protocol.file_record(temporary / "audit.json"),
            },
        )
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_audit(seed)
    print(f"V28 AUDIT COMPLETE: seed={seed}", flush=True)


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
