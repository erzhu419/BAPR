"""Run one frozen-policy Ant finite-horizon paired branch-risk audit."""
from __future__ import annotations

import argparse
import copy
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.analysis import (
    regime_polarity_ant_branch_risk_v30 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_ant_action_compensation_audit_v29 as v29_audit,
)
from jax_experiments.analysis import (
    run_regime_polarity_ant_full_state_audit_v22 as v22_audit,
)
from jax_experiments.common.checkpoint import _patch_flax_variablestate_unpickle
from jax_experiments.envs.brax_env import apply_action_disturbance
from jax_experiments.train import make_env


def _empty_counter() -> dict[str, float | int]:
    return {
        "n": 0,
        "candidate_terminated": 0,
        "fallback_terminated": 0,
        "rescued": 0,
        "harmed": 0,
        "both_terminated": 0,
        "neither_terminated": 0,
        "candidate_return_sum": 0.0,
        "fallback_return_sum": 0.0,
    }


def _add_pair(
    counter: dict[str, float | int], candidate: dict[str, Any],
    fallback: dict[str, Any],
) -> None:
    candidate_done = bool(candidate["terminated"])
    fallback_done = bool(fallback["terminated"])
    counter["n"] += 1
    counter["candidate_terminated"] += int(candidate_done)
    counter["fallback_terminated"] += int(fallback_done)
    counter["rescued"] += int(candidate_done and not fallback_done)
    counter["harmed"] += int(not candidate_done and fallback_done)
    counter["both_terminated"] += int(candidate_done and fallback_done)
    counter["neither_terminated"] += int(not candidate_done and not fallback_done)
    counter["candidate_return_sum"] += float(candidate["return"])
    counter["fallback_return_sum"] += float(fallback["return"])


def _add_candidate(counter: dict[str, float | int], row: dict[str, Any]) -> None:
    counter["n"] += 1
    counter["candidate_terminated"] += int(bool(row["terminated"]))
    counter["candidate_return_sum"] += float(row["return"])


def _finalize(counter: dict[str, float | int]) -> dict[str, Any]:
    n = int(counter["n"])
    if n <= 0:
        raise ValueError("cannot finalize an empty branch counter")
    candidate_terminated = int(counter["candidate_terminated"])
    fallback_terminated = int(counter["fallback_terminated"])
    candidate_survived = n - candidate_terminated
    return {
        **counter,
        "candidate_termination_risk": candidate_terminated / n,
        "fallback_termination_risk": fallback_terminated / n,
        "absolute_risk_reduction": (
            candidate_terminated - fallback_terminated) / n,
        "rescue_fraction_given_candidate_failure": (
            int(counter["rescued"]) / candidate_terminated
            if candidate_terminated else 0.0),
        "harm_fraction_given_candidate_survival": (
            int(counter["harmed"]) / candidate_survived
            if candidate_survived else 0.0),
        "candidate_return_mean": float(counter["candidate_return_sum"]) / n,
        "fallback_return_mean": float(counter["fallback_return_sum"]) / n,
        "fallback_minus_candidate_return": (
            float(counter["fallback_return_sum"])
            - float(counter["candidate_return_sum"])) / n,
    }


def _finalize_candidate(counter: dict[str, float | int]) -> dict[str, Any]:
    n = int(counter["n"])
    if n <= 0:
        raise ValueError("cannot finalize an empty candidate counter")
    terminated = int(counter["candidate_terminated"])
    return {
        "n": n,
        "terminated": terminated,
        "termination_risk": terminated / n,
        "return_mean": float(counter["candidate_return_sum"]) / n,
    }


def _load_controllers(
    seed: int, reference_mode: int,
) -> dict[str, dict[str, Any]]:
    _patch_flax_variablestate_unpickle()
    v22_audit._bind()
    v22_audit.base._bind()
    robust = v22_audit.base._load_source(seed)
    specialist = v22_audit.base.base._load_specialist(
        protocol.parent.policy_parent.CONTROL_VARIANT,
        seed,
        reference_mode,
        robust,
    )
    return {
        "robust_sac": robust,
        f"specialist_{reference_mode}": specialist,
    }


def _deterministic_action(controller: dict[str, Any]):
    graphdef = controller["policy_graphdef"]

    @jax.jit
    def action(policy_params, observation):
        policy = nnx.merge(graphdef, policy_params)
        obs = jnp.asarray(observation, dtype=jnp.float32)[None]
        return policy.deterministic(obs)[0]

    def wrapped(observation):
        return np.asarray(
            action(controller["policy_params"], observation),
            dtype=np.float32,
        )

    return wrapped


def _make_branch_rollout(env, controller: dict[str, Any]):
    graphdef = controller["policy_graphdef"]
    step_fn = env._step_fn

    @jax.jit
    def rollout(
        policy_params, initial_state, system, action_gain, action_noise_std,
        packet_loss_prob, burst_prob, burst_std, action_multiplier, step_keys,
    ):
        policy = nnx.merge(graphdef, policy_params)

        def body(carry, noise_key):
            state, alive = carry

            def advance(_):
                action = policy.deterministic(state.obs[None])[0]
                command = jnp.clip(action * action_multiplier, -1.0, 1.0)
                executed = apply_action_disturbance(
                    command, noise_key, action_gain, action_noise_std,
                    packet_loss_prob, burst_prob, burst_std)
                next_state = step_fn(system, state, executed)
                terminated = next_state.done > 0.5
                torso_z = next_state.pipeline_state.x.pos[0, 2]
                return (
                    (next_state, jnp.logical_not(terminated)),
                    (next_state.reward, terminated, torso_z),
                )

            def idle(_):
                torso_z = state.pipeline_state.x.pos[0, 2]
                return (
                    (state, jnp.asarray(False)),
                    (jnp.asarray(0.0, dtype=jnp.float32),
                     jnp.asarray(False), torso_z),
                )

            return jax.lax.cond(alive, advance, idle, operand=None)

        (_, _), outputs = jax.lax.scan(
            body, (initial_state, jnp.asarray(True)), step_keys)
        return outputs

    return rollout


def _branch_key(
    training_seed: int, event_seed: int, snapshot_step: int, replicate: int,
):
    key = jax.random.PRNGKey(protocol.BRANCH_KEY_BASE)
    for value in (training_seed, event_seed, snapshot_step, replicate):
        key = jax.random.fold_in(key, int(value))
    return key


def _trace_rows(outputs) -> dict[int, dict[str, Any]]:
    rewards, dones, torso_z = map(np.asarray, outputs)
    rows = {}
    for horizon in protocol.RISK_HORIZONS:
        prefix_done = np.flatnonzero(dones[:horizon])
        rows[horizon] = {
            "terminated": bool(prefix_done.size),
            "termination_step": (
                int(prefix_done[0]) + 1 if prefix_done.size else None),
            "return": float(np.sum(rewards[:horizon], dtype=np.float64)),
            "end_torso_z": float(torso_z[horizon - 1]),
        }
    return rows


def _source_snapshots(
    env, action, event_seed: int, reference_mode: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tasks = env.sample_tasks(len(protocol.MODES))
    env.set_nonstationary_para(tasks)
    env.set_task(tasks[reference_mode])
    observation = env.reset()
    states = [env._state]
    source_return = 0.0
    termination_step = None
    for step in range(1, protocol.SOURCE_MAX_STEPS + 1):
        observation, reward, done, _ = env.step(action(observation))
        source_return += float(reward)
        if done:
            termination_step = step
            break
        states.append(env._state)

    selected: dict[int, set[str]] = {}
    for step in protocol.FIXED_SNAPSHOT_STEPS:
        if step < len(states):
            selected.setdefault(step, set()).add("fixed")
    if termination_step is not None:
        for offset in protocol.PRETERMINATION_OFFSETS:
            step = termination_step - offset
            if 0 <= step < len(states):
                selected.setdefault(step, set()).add(f"preterm_{offset}")

    lower, upper = env.env._healthy_z_range
    snapshots = []
    for step, tags in sorted(selected.items()):
        state = states[step]
        torso_z = float(state.pipeline_state.x.pos[0, 2])
        health_margin = min(torso_z - float(lower), float(upper) - torso_z)
        snapshots.append({
            "event_seed": int(event_seed),
            "source_step": int(step),
            "tags": sorted(tags),
            "state": state,
            "torso_z": torso_z,
            "health_margin": float(health_margin),
        })
    return snapshots, {
        "event_seed": int(event_seed),
        "reference_mode": int(reference_mode),
        "physical_termination_step": termination_step,
        "truncated_at_source_horizon": termination_step is None,
        "return_before_termination_or_truncation": source_return,
        "snapshot_steps": [int(row["source_step"]) for row in snapshots],
        "snapshot_count": len(snapshots),
    }


def _mode_parameters(env, tasks) -> dict[int, tuple[Any, ...]]:
    rows = {}
    for mode in protocol.MODES:
        env.set_task(tasks[mode])
        rows[mode] = (
            env._current_sys,
            *env.action_disturbance_params(),
        )
    return rows


def _identity(seed: int) -> dict[str, Any]:
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "benchmark_role": "ant_finite_horizon_paired_branch_risk_development",
        "training_seed": protocol.require_training_seed(seed),
        "source_event_seeds": list(protocol.SOURCE_EVENT_SEEDS),
        "reference_mode": protocol.selected_reference_mode(seed),
        "modes": list(protocol.MODES),
        "risk_horizons": list(protocol.RISK_HORIZONS),
        "continuations_per_snapshot": protocol.CONTINUATIONS_PER_SNAPSHOT,
        "arms": list(protocol.ARMS),
    }


def evaluate(seed: int, *, smoke: bool = False) -> dict[str, Any]:
    seed = protocol.require_training_seed(seed)
    if not smoke:
        protocol.validate_registration()
    v29_audit.validate_audit(seed)
    reference_mode = protocol.selected_reference_mode(seed)
    controllers = _load_controllers(seed, reference_mode)
    robust = controllers["robust_sac"]
    candidate = controllers[f"specialist_{reference_mode}"]
    config = copy.deepcopy(robust["config"])
    config.stochastic_mode_fixed_id = reference_mode
    env = make_env(config, seed_offset=0)
    tasks = env.sample_tasks(len(protocol.MODES))
    parameters = _mode_parameters(env, tasks)
    candidate_action = _deterministic_action(candidate)
    candidate_rollout = _make_branch_rollout(env, candidate)
    fallback_rollout = _make_branch_rollout(env, robust)
    gains = np.asarray(
        protocol.parent.mode_gain_vectors(env.act_dim), dtype=np.float32)
    horizons = [protocol.MAX_RISK_HORIZON] if smoke else protocol.RISK_HORIZONS
    continuations = 1 if smoke else protocol.CONTINUATIONS_PER_SNAPSHOT
    events = protocol.SOURCE_EVENT_SEEDS[:1] if smoke else protocol.SOURCE_EVENT_SEEDS

    mode_counters = {
        str(mode): {str(h): _empty_counter() for h in horizons}
        for mode in protocol.MODES
    }
    overall_counters = {str(h): _empty_counter() for h in horizons}
    unique_candidate_counters = {
        str(h): _empty_counter() for h in horizons
    }
    source_rows = []
    snapshot_rows = []
    max_candidate_return_error = 0.0
    candidate_done_mismatches = 0
    invariance_comparisons = 0
    snapshot_count = 0

    try:
        for event_seed in events:
            source_config = copy.deepcopy(config)
            source_env = make_env(
                source_config, seed_offset=int(event_seed) - int(config.seed))
            try:
                snapshots, source_row = _source_snapshots(
                    source_env, candidate_action, event_seed, reference_mode)
            finally:
                source_env.close()
            if smoke:
                snapshots = snapshots[:1]
                source_row["snapshot_steps"] = [snapshots[0]["source_step"]]
                source_row["snapshot_count"] = len(snapshots)
            source_rows.append(source_row)

            for snapshot in snapshots:
                snapshot_count += 1
                per_snapshot = {
                    key: value for key, value in snapshot.items()
                    if key != "state"
                }
                per_snapshot["primary_horizon"] = {}
                for replicate in range(continuations):
                    key = _branch_key(
                        seed, event_seed, snapshot["source_step"], replicate)
                    step_keys = jax.random.split(
                        key, protocol.MAX_RISK_HORIZON)
                    candidate_by_mode = {}
                    for mode in protocol.MODES:
                        system, gain, noise_std, packet, burst, burst_std = (
                            parameters[mode])
                        multiplier = gains[mode] * gains[reference_mode]
                        candidate_outputs = candidate_rollout(
                            candidate["policy_params"], snapshot["state"],
                            system, gain, noise_std, packet, burst, burst_std,
                            jnp.asarray(multiplier), step_keys)
                        fallback_outputs = fallback_rollout(
                            robust["policy_params"], snapshot["state"],
                            system, gain, noise_std, packet, burst, burst_std,
                            jnp.ones((env.act_dim,), dtype=jnp.float32),
                            step_keys)
                        candidate_rows = _trace_rows(candidate_outputs)
                        fallback_rows = _trace_rows(fallback_outputs)
                        candidate_by_mode[mode] = (
                            np.asarray(candidate_outputs[0]),
                            np.asarray(candidate_outputs[1]),
                        )
                        for horizon in horizons:
                            candidate_row = candidate_rows[horizon]
                            fallback_row = fallback_rows[horizon]
                            _add_pair(
                                mode_counters[str(mode)][str(horizon)],
                                candidate_row, fallback_row)
                            _add_pair(
                                overall_counters[str(horizon)],
                                candidate_row, fallback_row)
                            if mode == protocol.MODES[0]:
                                _add_candidate(
                                    unique_candidate_counters[str(horizon)],
                                    candidate_row)
                            if horizon == protocol.MAX_RISK_HORIZON:
                                counter = per_snapshot["primary_horizon"].setdefault(
                                    str(mode), _empty_counter())
                                _add_pair(counter, candidate_row, fallback_row)

                    baseline_rewards, baseline_dones = candidate_by_mode[
                        protocol.MODES[0]]
                    for mode in protocol.MODES[1:]:
                        rewards, dones = candidate_by_mode[mode]
                        max_candidate_return_error = max(
                            max_candidate_return_error,
                            float(np.max(np.abs(
                                np.cumsum(rewards, dtype=np.float64)
                                - np.cumsum(baseline_rewards, dtype=np.float64)
                            ))),
                        )
                        candidate_done_mismatches += int(
                            not np.array_equal(dones, baseline_dones))
                        invariance_comparisons += 1
                per_snapshot["primary_horizon"] = {
                    mode: _finalize(counter)
                    for mode, counter in per_snapshot["primary_horizon"].items()
                }
                snapshot_rows.append(per_snapshot)
                print(
                    f"V30 seed={seed} event={event_seed} "
                    f"snapshot={snapshot['source_step']} complete",
                    flush=True,
                )
    finally:
        env.close()

    return {
        "schema": protocol.AUDIT_SCHEMA,
        "status": "smoke" if smoke else "complete",
        "identity": _identity(seed),
        "registration": (
            None if smoke else protocol.file_record(protocol.REGISTRATION_PATH)),
        "frozen_inputs": protocol.frozen_input_records(seed),
        "source_trajectories": source_rows,
        "snapshot_count": snapshot_count,
        "snapshot_primary_rows": snapshot_rows,
        "mode_horizons": {
            mode: {horizon: _finalize(counter)
                   for horizon, counter in rows.items()}
            for mode, rows in mode_counters.items()
        },
        "overall_horizons": {
            horizon: _finalize(counter)
            for horizon, counter in overall_counters.items()
        },
        "unique_candidate_horizons": {
            horizon: _finalize_candidate(counter)
            for horizon, counter in unique_candidate_counters.items()
        },
        "candidate_mode_invariance": {
            "comparisons": invariance_comparisons,
            "max_abs_cumulative_return_error": max_candidate_return_error,
            "termination_trace_mismatches": candidate_done_mismatches,
            "pass": bool(
                max_candidate_return_error
                <= protocol.EXACT_BRANCH_RETURN_ATOL
                and candidate_done_mismatches == 0),
        },
        "accounting": {
            "new_training_interactions": 0,
            "saved_simulator_states": 0,
            "saved_branch_trajectories": 0,
            "source_environment_steps": int(sum(
                row["physical_termination_step"]
                or protocol.SOURCE_MAX_STEPS for row in source_rows)),
            "paired_branch_rollouts": (
                snapshot_count * continuations * len(protocol.MODES)),
        },
    }


def validate_result(payload: dict[str, Any], seed: int) -> None:
    seed = protocol.require_training_seed(seed)
    if (
        payload.get("schema") != protocol.AUDIT_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("identity") != _identity(seed)
        or payload.get("registration")
        != protocol.file_record(protocol.REGISTRATION_PATH)
        or payload.get("frozen_inputs") != protocol.frozen_input_records(seed)
    ):
        raise ValueError("invalid V30 audit identity")
    snapshots = int(payload.get("snapshot_count", 0))
    if snapshots <= 0 or len(payload.get("snapshot_primary_rows") or []) != snapshots:
        raise ValueError("V30 snapshot inventory is incomplete")
    if len(payload.get("source_trajectories") or []) != len(
        protocol.SOURCE_EVENT_SEEDS
    ):
        raise ValueError("V30 source trajectories are incomplete")
    paired_n = snapshots * protocol.CONTINUATIONS_PER_SNAPSHOT
    for mode in protocol.MODES:
        rows = payload["mode_horizons"].get(str(mode)) or {}
        if set(rows) != {str(value) for value in protocol.RISK_HORIZONS}:
            raise ValueError("V30 mode horizons are incomplete")
        if any(int(row.get("n", -1)) != paired_n for row in rows.values()):
            raise ValueError("V30 mode branch count is invalid")
    overall_n = paired_n * len(protocol.MODES)
    if any(
        int(row.get("n", -1)) != overall_n
        for row in payload["overall_horizons"].values()
    ):
        raise ValueError("V30 overall branch count is invalid")
    if any(
        int(row.get("n", -1)) != paired_n
        for row in payload["unique_candidate_horizons"].values()
    ):
        raise ValueError("V30 unique candidate count is invalid")
    invariance = payload.get("candidate_mode_invariance") or {}
    expected_comparisons = paired_n * (len(protocol.MODES) - 1)
    if (
        int(invariance.get("comparisons", -1)) != expected_comparisons
        or invariance.get("pass") is not True
        or not math.isfinite(float(
            invariance.get("max_abs_cumulative_return_error", math.inf)))
    ):
        raise ValueError("V30 compensated candidate is not mode invariant")


def validate_audit(seed: int) -> dict[str, Any]:
    seed = protocol.require_training_seed(seed)
    payload = protocol.read_json(protocol.audit_result(seed))
    manifest = protocol.read_json(protocol.audit_manifest(seed))
    validate_result(payload, seed)
    if (
        manifest.get("schema") != protocol.AUDIT_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("identity") != _identity(seed)
        or manifest.get("registration")
        != protocol.file_record(protocol.REGISTRATION_PATH)
        or manifest.get("frozen_inputs") != protocol.frozen_input_records(seed)
        or manifest.get("audit") != protocol.file_record(
            protocol.audit_result(seed))
    ):
        raise ValueError("invalid V30 audit manifest")
    return manifest


def run(seed: int) -> None:
    seed = protocol.require_training_seed(seed)
    protocol.validate_registration()
    destination = protocol.audit_dir(seed)
    if protocol.audit_manifest(seed).is_file():
        try:
            validate_audit(seed)
        except (KeyError, OSError, TypeError, ValueError):
            pass
        else:
            print(f"V30 AUDIT ALREADY COMPLETE: seed={seed}", flush=True)
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
        protocol.write_json_atomic(temporary / "audit.json", payload)
        protocol.write_json_atomic(
            temporary / "audit_manifest.json",
            {
                "schema": protocol.AUDIT_SCHEMA,
                "status": "complete",
                "identity": _identity(seed),
                "registration": protocol.file_record(
                    protocol.REGISTRATION_PATH),
                "frozen_inputs": protocol.frozen_input_records(seed),
                "audit": protocol.file_record(temporary / "audit.json"),
            },
        )
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_audit(seed)
    print(f"V30 AUDIT COMPLETE: seed={seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        payload = evaluate(args.seed, smoke=True)
        print(
            "V30 SMOKE COMPLETE: "
            f"snapshots={payload['snapshot_count']} "
            f"invariance={payload['candidate_mode_invariance']['pass']}",
            flush=True,
        )
        return
    if not args.resume:
        raise SystemExit("--resume is required for idempotent execution")
    run(args.seed)


if __name__ == "__main__":
    main()
