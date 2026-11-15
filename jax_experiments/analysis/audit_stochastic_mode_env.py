#!/usr/bin/env python3
"""Audit BAPR-v3 stochastic mode semantics without training a policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jax_experiments.envs.brax_env import apply_action_disturbance
from jax_experiments.envs.stochastic_mode_env import (
    MODE_FAMILIES,
    StochasticModeEnv,
)


def audit_family(env_name: str, family: str, samples: int, seed: int):
    env = StochasticModeEnv(
        env_name, family=family, dwell_steps=500,
        dwell_distribution="fixed", seed=seed, backend="spring")
    tasks = env.sample_tasks(env.num_modes)
    rows = []
    keys = jax.random.split(jax.random.PRNGKey(seed + 1), samples)
    zero_command = jnp.zeros((env.act_dim,), dtype=jnp.float32)
    probe_command = jnp.full((env.act_dim,), 0.5, dtype=jnp.float32)

    for task in tasks:
        env.set_task(task)
        (gain, noise_std, packet_loss_prob,
         burst_prob, burst_std) = env.action_disturbance_params()
        executed_zero = jax.vmap(
            lambda key: apply_action_disturbance(
                zero_command, key, gain, noise_std, packet_loss_prob,
                burst_prob, burst_std))(keys)
        executed_probe = jax.vmap(
            lambda key: apply_action_disturbance(
                probe_command, key, gain, noise_std, packet_loss_prob,
                burst_prob, burst_std))(keys)
        packet_loss_rate = jnp.mean(jnp.all(
            jnp.abs(executed_probe) < 1e-7, axis=1))
        zero_command_nonzero_rate = jnp.mean(jnp.any(
            jnp.abs(executed_zero) > 1e-7, axis=1))
        target_gain = np.asarray(task["action_gain"], dtype=np.float32)
        target_gain_json = (
            float(target_gain)
            if target_gain.ndim == 0
            else target_gain.tolist()
        )
        rows.append({
            "mode_id": int(task["mode_id"]),
            "gravity_scale": float(task["gravity_scale"]),
            "target_action_gain": target_gain_json,
            "target_action_noise_std": float(task["action_noise_std"]),
            "target_packet_loss_prob": float(
                task.get("packet_loss_prob", 0.0)),
            "target_burst_prob": float(task.get("burst_prob", 0.0)),
            "target_burst_std": float(task.get("burst_std", 0.0)),
            "empirical_action_mean_abs": float(jnp.abs(
                jnp.mean(executed_zero, axis=0)).mean()),
            "empirical_action_std": float(jnp.std(
                executed_zero, axis=0).mean()),
            "empirical_probe_mean": np.asarray(
                jnp.mean(executed_probe, axis=0)).tolist(),
            "empirical_packet_loss_rate": float(packet_loss_rate),
            "empirical_zero_command_nonzero_rate": float(
                zero_command_nonzero_rate),
            "physics_gravity_z": float(env._current_sys.gravity[-1]),
        })

    gravity = np.asarray([row["physics_gravity_z"] for row in rows])
    measured_std = np.asarray([
        row["empirical_action_std"] for row in rows])
    target_std = np.asarray([
        row["target_action_noise_std"] for row in rows])
    measured_packet = np.asarray([
        row["empirical_packet_loss_rate"] for row in rows])
    target_packet = np.asarray([
        row["target_packet_loss_prob"] for row in rows])
    measured_burst = np.asarray([
        row["empirical_zero_command_nonzero_rate"] for row in rows])
    target_burst = np.asarray([
        row["target_burst_prob"] for row in rows])
    structured_gains = [
        np.asarray(row["target_action_gain"], dtype=np.float32).reshape(-1)
        for row in rows
    ]
    structured_probe_ok = all(
        np.allclose(
            np.asarray(row["empirical_probe_mean"], dtype=np.float32),
            0.5 * gain,
            atol=0.012,
            rtol=0.04,
        )
        for row, gain in zip(rows, structured_gains)
    )
    structured_patterns = {
        tuple(np.round(gain, decimals=6)) for gain in structured_gains
    }
    checks = {
        "finite": bool(np.all(np.isfinite(gravity))
                       and np.all(np.isfinite(measured_std))
                       and np.all(np.isfinite(measured_packet))
                       and np.all(np.isfinite(measured_burst))),
        "gaussian_variance_matches": bool(
            family in ("packet_loss", "burst_torque")
            or np.allclose(
                measured_std, target_std, atol=0.012, rtol=0.12)),
        "packet_loss_rate_matches": bool(
            family != "packet_loss"
            or np.allclose(
                measured_packet, target_packet, atol=0.025, rtol=0.15)),
        "burst_event_rate_matches": bool(
            family != "burst_torque"
            or np.allclose(
                measured_burst, target_burst, atol=0.025, rtol=0.18)),
        "variance_only_keeps_mean_physics": bool(
            family != "variance_only"
            or (np.ptp(gravity) < 1e-7
                and len({row["target_action_gain"] for row in rows}) == 1)),
        "event_families_keep_robot_fixed": bool(
            family not in ("packet_loss", "burst_torque")
            or (np.ptp(gravity) < 1e-7
                and {row["target_action_gain"] for row in rows} == {1.0}
                and {row["target_action_noise_std"] for row in rows}
                == {0.0})),
        "structured_channel_is_equal_severity_and_distinct": bool(
            family != "structured_channel"
            or (np.ptp(gravity) < 1e-7
                and {row["target_action_noise_std"] for row in rows}
                == {0.04}
                and len(structured_patterns) == 4
                and all(
                    abs(
                        np.count_nonzero(np.isclose(gain, 0.45))
                        - np.count_nonzero(np.isclose(gain, 1.0))) <= 1
                    for gain in structured_gains)
                and structured_probe_ok)),
        "actuator_polarity_is_fixed_distinct_and_balanced": bool(
            family != "actuator_polarity"
            or (np.ptp(gravity) < 1e-7
                and {row["target_action_noise_std"] for row in rows}
                == {0.02}
                and len(structured_patterns) == 4
                and all(
                    np.all(np.isin(
                        np.round(gain, decimals=6), (-1.0, 1.0)))
                    and np.any(gain < 0.0)
                    and np.any(gain > 0.0)
                    and abs(
                        np.count_nonzero(gain < 0.0)
                        - np.count_nonzero(gain > 0.0)) <= 1
                    for gain in structured_gains)
                and structured_probe_ok)),
        "deterministic_family_has_zero_process_noise": bool(
            family != "deterministic_mean"
            or np.max(np.abs(measured_std)) < 1e-7),
    }
    return {"family": family, "rows": rows, "checks": checks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="HalfCheetah-v2")
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = {
        "env": args.env,
        "samples": args.samples,
        "families": [
            audit_family(args.env, family, args.samples, args.seed + index)
            for index, family in enumerate(MODE_FAMILIES)
        ],
    }
    result["passed"] = all(
        all(family["checks"].values()) for family in result["families"])
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
