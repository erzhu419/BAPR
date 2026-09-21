"""Verify the exact action and trajectory symmetry required by V28."""
from __future__ import annotations

import argparse
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jax_experiments.analysis import (
    regime_polarity_action_compensation_v28 as protocol,
)
from jax_experiments.configs.default import Config
from jax_experiments.envs.brax_env import apply_action_disturbance
from jax_experiments.train import make_env


STRUCTURAL_ENVS = ("HalfCheetah-v2", "Ant-v2")
STRUCTURAL_ACTION_DIMS = (3, 6, 8)
TRAJECTORY_STEPS = 12
STRUCTURAL_SEED = 202_609_12


def _identity() -> dict[str, Any]:
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "environments": list(STRUCTURAL_ENVS),
        "action_dimensions": list(STRUCTURAL_ACTION_DIMS),
        "trajectory_steps": TRAJECTORY_STEPS,
        "seed": STRUCTURAL_SEED,
    }


def algebra_audit() -> dict[str, Any]:
    rows = []
    maximum = 0.0
    rng = np.random.RandomState(STRUCTURAL_SEED)
    key = jax.random.PRNGKey(STRUCTURAL_SEED)
    for act_dim in STRUCTURAL_ACTION_DIMS:
        gains = np.asarray(protocol.mode_gain_vectors(act_dim), dtype=np.float32)
        actions = rng.uniform(-1.0, 1.0, size=(17, act_dim)).astype(np.float32)
        for reference_mode in protocol.MODES:
            for target_mode in protocol.MODES:
                key, event_key = jax.random.split(key)
                command = protocol.compensate_action(
                    actions, reference_mode, target_mode)
                reference_executed = apply_action_disturbance(
                    jnp.asarray(actions), event_key,
                    jnp.asarray(gains[reference_mode]), 0.02)
                target_executed = apply_action_disturbance(
                    jnp.asarray(command), event_key,
                    jnp.asarray(gains[target_mode]), 0.02)
                error = float(np.max(np.abs(
                    np.asarray(reference_executed)
                    - np.asarray(target_executed))))
                maximum = max(maximum, error)
                rows.append({
                    "action_dim": int(act_dim),
                    "reference_mode": int(reference_mode),
                    "target_mode": int(target_mode),
                    "max_abs_executed_action_error": error,
                })
    return {
        "rows": rows,
        "max_abs_executed_action_error": maximum,
        "pass": bool(maximum <= protocol.EXACT_ACTION_ATOL),
    }


def _config(env_name: str, mode: int) -> Config:
    config = Config()
    config.env_name = str(env_name)
    config.brax_backend = "spring"
    config.env_type = "stochastic_mode"
    config.stochastic_mode_family = protocol.FAMILY
    config.stochastic_mode_dwell_steps = protocol.DWELL_STEPS
    config.stochastic_mode_dwell_distribution = "fixed"
    config.stochastic_mode_fixed_id = int(mode)
    config.task_num = len(protocol.MODES)
    config.test_task_num = len(protocol.MODES)
    config.seed = STRUCTURAL_SEED
    config.max_episode_steps = protocol.MAX_EPISODE_STEPS
    return config


def _synthetic_action(observation: np.ndarray, act_dim: int) -> np.ndarray:
    obs = np.asarray(observation, dtype=np.float32)
    phase = np.linspace(-0.35, 0.35, act_dim, dtype=np.float32)
    return np.tanh(0.35 * obs[:act_dim] + phase).astype(np.float32)


def _trajectory_pair(
    env_name: str,
    reference_env,
    target_env,
    reference_tasks,
    target_tasks,
    reference_mode: int,
    target_mode: int,
) -> dict[str, Any]:
    maximum_observation = 0.0
    maximum_execution = 0.0
    maximum_reward = 0.0
    compared_steps = 0
    reference_env.set_nonstationary_para(reference_tasks)
    target_env.set_nonstationary_para(target_tasks)
    reference_env.set_task(reference_tasks[int(reference_mode)])
    target_env.set_task(target_tasks[int(target_mode)])
    pair_seed = STRUCTURAL_SEED + 100 * int(reference_mode) + int(target_mode)
    pair_key = jax.random.PRNGKey(pair_seed)
    reference_env.rng = pair_key
    target_env.rng = pair_key
    reference_obs = reference_env.reset()
    target_obs = target_env.reset()
    maximum_observation = float(np.max(np.abs(
        np.asarray(reference_obs) - np.asarray(target_obs))))
    for _ in range(TRAJECTORY_STEPS):
        action = _synthetic_action(reference_obs, reference_env.act_dim)
        target_command = protocol.compensate_action(
            action, reference_mode, target_mode)
        next_reference, reference_reward, reference_done, reference_info = (
            reference_env.step(action))
        next_target, target_reward, target_done, target_info = (
            target_env.step(target_command))
        maximum_observation = max(
            maximum_observation,
            float(np.max(np.abs(
                np.asarray(next_reference) - np.asarray(next_target)))),
        )
        maximum_execution = max(
            maximum_execution,
            float(np.max(np.abs(
                np.asarray(reference_info["executed_action"])
                - np.asarray(target_info["executed_action"])))),
        )
        maximum_reward = max(
            maximum_reward,
            abs(float(reference_reward) - float(target_reward)),
        )
        if bool(reference_done) != bool(target_done):
            raise RuntimeError("coupled trajectories disagree on termination")
        compared_steps += 1
        reference_obs = next_reference
        target_obs = next_target
        if reference_done:
            break
    passed = bool(
        compared_steps > 0
        and maximum_observation <= protocol.EXACT_TRAJECTORY_ATOL
        and maximum_execution <= protocol.EXACT_ACTION_ATOL
        and maximum_reward <= protocol.EXACT_TRAJECTORY_ATOL
    )
    return {
        "environment": env_name,
        "reference_mode": int(reference_mode),
        "target_mode": int(target_mode),
        "compared_steps": int(compared_steps),
        "max_abs_observation_error": maximum_observation,
        "max_abs_executed_action_error": maximum_execution,
        "max_abs_reward_error": maximum_reward,
        "pass": passed,
    }


def trajectory_audit() -> dict[str, Any]:
    rows = []
    for env_name in STRUCTURAL_ENVS:
        reference_env = make_env(_config(env_name, 0), seed_offset=0)
        target_env = make_env(_config(env_name, 0), seed_offset=0)
        try:
            reference_tasks = reference_env.sample_tasks(len(protocol.MODES))
            target_tasks = target_env.sample_tasks(len(protocol.MODES))
            for reference_mode in protocol.MODES:
                for target_mode in protocol.MODES:
                    rows.append(_trajectory_pair(
                        env_name,
                        reference_env,
                        target_env,
                        reference_tasks,
                        target_tasks,
                        reference_mode,
                        target_mode,
                    ))
        finally:
            if hasattr(reference_env, "close"):
                reference_env.close()
            if hasattr(target_env, "close"):
                target_env.close()
    return {
        "rows": rows,
        "max_abs_observation_error": max(
            row["max_abs_observation_error"] for row in rows),
        "max_abs_executed_action_error": max(
            row["max_abs_executed_action_error"] for row in rows),
        "max_abs_reward_error": max(
            row["max_abs_reward_error"] for row in rows),
        "pass": all(bool(row["pass"]) for row in rows),
    }


def evaluate() -> dict[str, Any]:
    algebra = algebra_audit()
    trajectory = trajectory_audit()
    return {
        "schema": protocol.STRUCTURAL_SCHEMA,
        "status": "complete",
        "identity": _identity(),
        "registration": protocol.file_record(protocol.REGISTRATION_PATH),
        "algebra": algebra,
        "trajectory": trajectory,
        "pass": bool(algebra["pass"] and trajectory["pass"]),
    }


def validate_result(payload: dict[str, Any]) -> None:
    if (
        payload.get("schema") != protocol.STRUCTURAL_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("identity") != _identity()
        or payload.get("registration")
        != protocol.file_record(protocol.REGISTRATION_PATH)
        or payload.get("pass") is not True
    ):
        raise ValueError("invalid or failed V28 structural audit")
    for section, tolerance in (
        ("algebra", protocol.EXACT_ACTION_ATOL),
        ("trajectory", protocol.EXACT_TRAJECTORY_ATOL),
    ):
        row = payload.get(section) or {}
        if row.get("pass") is not True:
            raise ValueError(f"V28 {section} symmetry failed")
        values = [
            float(value) for key, value in row.items()
            if key.startswith("max_abs_")
        ]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError(f"V28 {section} metrics are invalid")
        if max(values) > tolerance:
            raise ValueError(f"V28 {section} tolerance exceeded")


def validate_structural_audit() -> dict[str, Any]:
    manifest = protocol.read_json(protocol.STRUCTURAL_MANIFEST)
    payload = protocol.read_json(protocol.STRUCTURAL_RESULT)
    validate_result(payload)
    if (
        manifest.get("schema") != protocol.STRUCTURAL_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("identity") != _identity()
        or manifest.get("registration")
        != protocol.file_record(protocol.REGISTRATION_PATH)
        or manifest.get("result")
        != protocol.file_record(protocol.STRUCTURAL_RESULT)
    ):
        raise ValueError("invalid V28 structural manifest")
    return manifest


def run() -> None:
    protocol.validate_registration()
    if protocol.STRUCTURAL_MANIFEST.is_file():
        try:
            validate_structural_audit()
        except (KeyError, OSError, TypeError, ValueError):
            pass
        else:
            print("V28 STRUCTURAL AUDIT ALREADY COMPLETE", flush=True)
            return
    destination = protocol.STRUCTURAL_ROOT
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp.", dir=destination.parent))
    try:
        payload = evaluate()
        validate_result(payload)
        protocol.write_json_atomic(temporary / "structural_audit.json", payload)
        protocol.write_json_atomic(
            temporary / "structural_manifest.json",
            {
                "schema": protocol.STRUCTURAL_SCHEMA,
                "status": "complete",
                "identity": _identity(),
                "registration": protocol.file_record(
                    protocol.REGISTRATION_PATH),
                "result": protocol.file_record(
                    temporary / "structural_audit.json"),
            },
        )
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_structural_audit()
    print("V28 STRUCTURAL AUDIT COMPLETE: pass=true", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for idempotent execution")
    run()


if __name__ == "__main__":
    main()
