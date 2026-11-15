"""Eval-only final checkpoint sweep over saved continuous train/test tasks.

This is for post-training audit runs. It reloads a completed checkpoint,
reconstructs the exact train/test task split from the run's
protocol_signature.json, and evaluates the final policy over many tasks instead
of the training loop's cheap first-task proxy for continuous task lists.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import Any

import jax
import numpy as np
from flax import nnx

from jax_experiments.algos.bapr_v2 import _gravity_latent
from jax_experiments.common.checkpoint import load_checkpoint
from jax_experiments.common.logging import Logger
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.configs.default import Config
from jax_experiments.train import (
    _eval_task_id_for_action,
    _eval_policy_state,
    _oracle_latent_for_eval,
    _reset_eval_switch_schedule,
    _select_eval_switch_sequence,
    make_algo,
    make_env,
)


def load_config(run_dir: Path) -> Config:
    signature_path = run_dir / "logs" / "protocol_signature.json"
    if not signature_path.exists():
        signature_path = run_dir / "checkpoints" / "protocol_signature.json"
    if not signature_path.exists():
        raise FileNotFoundError(f"missing protocol signature: {signature_path}")
    signature = json.loads(signature_path.read_text())
    saved = signature.get("config", {})

    config = Config()
    valid_fields = {field.name for field in fields(Config)}
    for key, value in saved.items():
        if key in valid_fields:
            setattr(config, key, value)
    config.save_root = str(run_dir.parent)
    config.run_name = run_dir.name
    config.resume = False
    return config


def task_stats(task: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in task.items():
        arr = np.asarray(value, dtype=np.float32)
        out[f"{key}_mean"] = float(np.mean(arr))
        out[f"{key}_min"] = float(np.min(arr))
        out[f"{key}_max"] = float(np.max(arr))
    return out


def protocol_gravity_latent(task: dict[str, Any], config: Config) -> float:
    """Physical task coordinate independent of a policy's latent scaling."""
    if isinstance(task, dict) and "mode_id" in task:
        return float(task["mode_id"])
    return float(_gravity_latent(
        task, 1, config.log_scale_limit,
        config.task_scale_distribution, "task_distribution")[0])


def task_policy_latent(agent, task: dict[str, Any], config: Config):
    """Return the task coordinate expected by the evaluated policy."""
    if not hasattr(agent, "latent_dim"):
        # Context-free SAC/RE-SAC and state-context ESCP do not expose a
        # BAPR policy latent.  Record the protocol's physical coordinate for
        # switching provenance without inventing a latent interface.
        return np.asarray(
            [protocol_gravity_latent(task, config)], dtype=np.float32)
    if isinstance(task, dict) and "mode_id" in task:
        mode_id = int(task["mode_id"])
        latent = np.zeros((int(agent.latent_dim),), dtype=np.float32)
        if not 0 <= mode_id < len(latent):
            raise ValueError(
                f"mode_id={mode_id} outside policy latent size {len(latent)}")
        latent[mode_id] = 1.0
        return latent
    return _gravity_latent(
        task, agent.latent_dim, config.log_scale_limit,
        config.task_scale_distribution,
        config.bapr_v2_latent_scale_mode)


def resolve_bapr_v2_eval_mode(agent, context_source: str,
                              advantage: str) -> tuple[int | None, bool | None]:
    if not getattr(agent, "uses_transition_context", False):
        return None, None
    source = (
        int(agent.rollout_context_source())
        if context_source == "checkpoint"
        else int({
            "robust": agent.CONTEXT_ROBUST,
            "oracle": agent.CONTEXT_ORACLE,
            "learned": agent.CONTEXT_LEARNED,
        }[context_source])
    )
    enabled = (
        bool(agent.advantage_gate_active())
        if advantage == "checkpoint"
        else advantage == "on"
    )
    return source, enabled


def require_checkpoint_iteration(
    checkpoint_next_iter: int,
    minimum_next_iter: int | None,
) -> None:
    if minimum_next_iter is None:
        return
    if int(checkpoint_next_iter) < int(minimum_next_iter):
        raise RuntimeError(
            "stale checkpoint: loaded next_iter="
            f"{int(checkpoint_next_iter)}, required >= "
            f"{int(minimum_next_iter)}")


def oracle_eval_mode_ids(agent, args: argparse.Namespace,
                         context_source: int | None):
    """Return dynamic/fixed oracle contexts requested for one checkpoint load."""
    if not bool(getattr(args, "oracle_context_ladder", False)):
        return (args.fixed_oracle_mode_id,)
    if args.fixed_oracle_mode_id is not None:
        raise ValueError(
            "--oracle-context-ladder cannot be combined with "
            "--fixed-oracle-mode-id")
    if (not getattr(agent, "uses_transition_context", False)
            or context_source != agent.CONTEXT_ORACLE):
        raise ValueError(
            "--oracle-context-ladder requires "
            "--bapr-v2-context-source oracle")
    return (None, *range(int(agent.latent_dim)))


def episode_returns(rewards: np.ndarray, dones: np.ndarray,
                    episodes: int, horizon: int):
    returns = []
    terminated = []
    steps = []
    for ep in range(episodes):
        start = ep * horizon
        end = start + horizon
        ep_r = 0.0
        term = False
        step_count = horizon
        for idx in range(start, end):
            ep_r += float(rewards[idx])
            if float(dones[idx]) > 0.5:
                term = True
                step_count = idx - start + 1
                break
        returns.append(ep_r)
        terminated.append(term)
        steps.append(step_count)
    return (
        np.asarray(returns, dtype=np.float64),
        np.asarray(terminated, dtype=np.float64),
        np.asarray(steps, dtype=np.float64),
    )


def evaluate_task_split(agent, env, config: Config, tasks, split: str,
                        episodes_per_task: int, max_tasks: int | None,
                        rng_seed: int, context_source: int | None = None,
                        advantage_enabled: bool | None = None,
                        fixed_oracle_task=None):
    policy_params, context_params, belief_vec = _eval_policy_state(agent, config)
    horizon = int(config.max_episode_steps)
    n_steps = episodes_per_task * horizon
    rows = []
    selected = list(tasks if max_tasks is None else tasks[:max_tasks])
    for task_idx, task in enumerate(selected):
        env.set_task(task)
        if getattr(agent, "uses_regime_context", False):
            reset_context = getattr(
                agent, "reset_eval_context_state", None)
            if callable(reset_context):
                reset_context()
            agent.set_eval_task(task)
            belief_vec = agent._build_belief_jax()
        rng_key = jax.random.fold_in(jax.random.PRNGKey(rng_seed), task_idx)
        if getattr(agent, "uses_transition_context", False):
            source = (
                agent.rollout_context_source()
                if context_source is None else context_source)
            gate_enabled = (
                agent.advantage_gate_active()
                if advantage_enabled is None else advantage_enabled)
            oracle_latent = _oracle_latent_for_eval(
                agent, source,
                fixed_oracle_task if fixed_oracle_task is not None else task)
            rewards, dones, _ = env.eval_rollout_adaptive(
                policy_params, context_params,
                agent.context_net.initial_state(), oracle_latent,
                n_steps, rng_key, episode_horizon=horizon,
                critic_params=nnx.state(agent.critic, nnx.Param),
                context_source=source,
                advantage_enabled=gate_enabled,
                advantage_margin=config.bapr_v2_advantage_margin,
                advantage_lcb_scale=config.bapr_v2_advantage_lcb_scale)
        else:
            rewards, dones = env.eval_rollout(
                policy_params,
                n_steps,
                rng_key,
                context_params=context_params,
                belief_vec=belief_vec,
                episode_horizon=horizon,
            )
        ep_returns, ep_terminated, ep_steps = episode_returns(
            rewards, dones, episodes_per_task, horizon)
        row = {
            "split": split,
            "task_index": task_idx,
            "episodes": int(episodes_per_task),
            "return_mean": float(np.mean(ep_returns)),
            "return_std": float(np.std(ep_returns)),
            "return_min": float(np.min(ep_returns)),
            "return_max": float(np.max(ep_returns)),
            "terminated_rate": float(np.mean(ep_terminated)),
            "steps_mean": float(np.mean(ep_steps)),
        }
        row.update(task_stats(task))
        rows.append(row)
    return rows


def evaluate_switching(agent, env, config: Config, tasks,
                       episodes: int, period_steps: int, rng_seed: int,
                       context_source: int | None = None,
                       advantage_enabled: bool | None = None,
                       fixed_oracle_task=None):
    del rng_seed  # Sequential env reset uses env's own deterministic PRNG.
    horizon = int(config.max_episode_steps)
    rows = []
    trace_rows = []
    snapshot = None
    if getattr(agent, "uses_transition_context", False):
        snapshot = agent.snapshot_adaptation()
        context_source = (
            agent.rollout_context_source()
            if context_source is None else context_source)
        advantage_enabled = (
            agent.advantage_gate_active()
            if advantage_enabled is None else advantage_enabled)
    for ep in range(episodes):
        switch_tasks, source_indices = _select_eval_switch_sequence(
            env, tasks, ep)
        _reset_eval_switch_schedule(env, switch_tasks, config, period_steps)
        if snapshot is not None:
            agent.reset_adaptation()
        obs = env.reset()
        if getattr(agent, "uses_regime_context", False):
            reset_context = getattr(
                agent, "reset_eval_context_state", None)
            if callable(reset_context):
                reset_context()
        ep_r = 0.0
        switches = 0
        termination_count = 0
        prev_task_id = int(getattr(env, "current_task_id", 0))
        done_step = horizon
        terminated = False
        for step in range(horizon):
            peek_task = getattr(env, "task_id_for_next_step", None)
            physics_action_task_id = (
                int(peek_task()) if callable(peek_task) else prev_task_id)
            if getattr(agent, "uses_regime_context", False):
                agent.set_oracle_task_id(physics_action_task_id)
            if (snapshot is not None
                    and context_source == agent.CONTEXT_ORACLE
                    and fixed_oracle_task is not None):
                action_task_id = -1
                agent.set_eval_task(fixed_oracle_task)
            else:
                action_task_id = (
                    physics_action_task_id
                    if snapshot is not None
                    and context_source == agent.CONTEXT_ORACLE
                    else _eval_task_id_for_action(
                        agent, env, prev_task_id,
                        context_source=context_source))
                if (snapshot is not None
                        and context_source == agent.CONTEXT_ORACLE
                        and 0 <= action_task_id < len(switch_tasks)):
                    agent.set_eval_task(switch_tasks[action_task_id])
            if (getattr(agent, "uses_regime_context", False)
                    and hasattr(agent, "eval_context_mode_id")):
                action_task_id = int(agent.eval_context_mode_id())
            context_before = (
                np.asarray(
                    agent.context_for_source(context_source),
                    dtype=np.float64)
                if snapshot is not None else None)
            gate_before = (
                float(context_before[-1])
                if context_before is not None else math.nan)
            true_latent_before = (
                task_policy_latent(
                    agent, switch_tasks[physics_action_task_id], config)
                if snapshot is not None
                and 0 <= physics_action_task_id < len(switch_tasks)
                else None)
            true_mode_before = (
                int(np.argmax(true_latent_before))
                if true_latent_before is not None
                and isinstance(switch_tasks[physics_action_task_id], dict)
                and "mode_id" in switch_tasks[physics_action_task_id]
                else -1)
            posterior_before = (
                context_before[:agent.latent_dim]
                if context_before is not None else None)
            estimated_mode_before = (
                int(np.argmax(posterior_before))
                if posterior_before is not None else -1)
            expected_mode_before = (
                float(np.dot(
                    posterior_before,
                    np.arange(agent.latent_dim, dtype=np.float64)))
                if posterior_before is not None else math.nan)
            pre_obs = obs
            action = (
                agent.select_action(
                    obs, deterministic=True,
                    context_source=context_source,
                    advantage_enabled=advantage_enabled)
                if snapshot is not None
                else agent.select_action(obs, deterministic=True))
            obs, reward, done, _ = env.step(action)
            if snapshot is not None:
                agent.observe_transition(
                    pre_obs, action, reward, obs, done)
            context_after = (
                np.asarray(
                    agent.context_for_source(context_source),
                    dtype=np.float64)
                if snapshot is not None else None)
            posterior_after = (
                context_after[:agent.latent_dim]
                if context_after is not None else None)
            estimated_mode_after = (
                int(np.argmax(posterior_after))
                if posterior_after is not None else -1)
            expected_mode_after = (
                float(np.dot(
                    posterior_after,
                    np.arange(agent.latent_dim, dtype=np.float64)))
                if posterior_after is not None else math.nan)
            ep_r += float(reward)
            task_id = int(getattr(env, "current_task_id", prev_task_id))
            switched = task_id != prev_task_id
            if switched:
                switches += 1
            trace_rows.append({
                "episode": ep,
                "step": step + 1,
                "physics_task_before": prev_task_id,
                "physics_source_task_before": (
                    source_indices[prev_task_id]
                    if 0 <= prev_task_id < len(source_indices) else -1),
                "physics_action_task_id": physics_action_task_id,
                "physics_action_source_task_id": (
                    source_indices[physics_action_task_id]
                    if 0 <= physics_action_task_id < len(source_indices)
                    else -1),
                "action_task_id": action_task_id,
                "action_source_task_id": (
                    source_indices[action_task_id]
                    if 0 <= action_task_id < len(source_indices) else -1),
                "eval_context_delay_remaining": int(
                    agent.eval_context_delay_remaining())
                    if getattr(agent, "uses_regime_context", False)
                    and hasattr(agent, "eval_context_delay_remaining")
                    else 0,
                "fixed_oracle_mode_id": (
                    int(fixed_oracle_task["mode_id"])
                    if isinstance(fixed_oracle_task, dict)
                    and "mode_id" in fixed_oracle_task else -1),
                "physics_task_after": task_id,
                "physics_source_task_after": (
                    source_indices[task_id]
                    if 0 <= task_id < len(source_indices) else -1),
                "switched": bool(switched),
                "reward": float(reward),
                "done": bool(done),
                "context_error": (
                    float(agent._last_context_error)
                    if snapshot is not None else math.nan
                ),
                "gate_before": gate_before,
                "gate_after": (
                    float(agent._last_context_gate)
                    if snapshot is not None else math.nan
                ),
                "residual_advantage": (
                    float(getattr(agent, "_last_advantage", math.nan))
                    if snapshot is not None else math.nan
                ),
                "residual_advantage_gate": (
                    float(getattr(
                        agent, "_last_advantage_gate", math.nan))
                    if snapshot is not None else math.nan
                ),
                "true_latent_before_0": (
                    float(true_latent_before[0])
                    if true_latent_before is not None else math.nan),
                "estimated_latent_before_0": (
                    float(context_before[0])
                    if context_before is not None else math.nan),
                "estimated_latent_after_0": (
                    float(context_after[0])
                    if context_after is not None else math.nan),
                "true_mode_before": true_mode_before,
                "estimated_mode_before": estimated_mode_before,
                "estimated_mode_after": estimated_mode_after,
                "expected_mode_before": expected_mode_before,
                "expected_mode_after": expected_mode_after,
                "mode_probability_before": (
                    float(posterior_before[true_mode_before])
                    if posterior_before is not None
                    and true_mode_before >= 0 else math.nan),
                "mode_probability_after": (
                    float(posterior_after[true_mode_before])
                    if posterior_after is not None
                    and true_mode_before >= 0 else math.nan),
                "mode_correct_before": (
                    float(estimated_mode_before == true_mode_before)
                    if true_mode_before >= 0 else math.nan),
                "mode_correct_after": (
                    float(estimated_mode_after == true_mode_before)
                    if true_mode_before >= 0 else math.nan),
                "expected_mode_abs_error_before": (
                    abs(expected_mode_before - true_mode_before)
                    if true_mode_before >= 0 else math.nan),
                "expected_mode_abs_error_after": (
                    abs(expected_mode_after - true_mode_before)
                    if true_mode_before >= 0 else math.nan),
                "latent_abs_error_before": (
                    float(abs(context_before[0] - true_latent_before[0]))
                    if context_before is not None
                    and true_latent_before is not None else math.nan),
                "latent_abs_error_after": (
                    float(abs(context_after[0] - true_latent_before[0]))
                    if context_after is not None
                    and true_latent_before is not None else math.nan),
            })
            prev_task_id = task_id
            if done:
                termination_count += 1
                if not terminated:
                    done_step = step + 1
                terminated = True
                # Keep the mode schedule and adaptation state alive across a
                # simulator reset so the fixed-horizon stream reaches switches.
                obs = env.reset()
        rows.append({
            "episode": ep,
            "return": float(ep_r),
            "metric_semantics": "fixed_horizon_stream_sum",
            "switch_sequence_source_indices": "|".join(
                str(index) for index in source_indices),
            "switch_sequence_latent_0": "|".join(
                str(float(task_policy_latent(agent, task, config)[0]))
                for task in switch_tasks),
            "switch_sequence_protocol_latent_0": "|".join(
                str(protocol_gravity_latent(task, config))
                for task in switch_tasks),
            "switch_latent_span": (
                float(np.ptp([
                    protocol_gravity_latent(task, config)
                    for task in switch_tasks
                ])) if switch_tasks else math.nan),
            "switch_count": int(switches),
            "terminated": bool(terminated),
            "termination_count": int(termination_count),
            "done_step": int(done_step),
            "done_before_first_switch": bool(terminated and done_step < period_steps),
        })
    if snapshot is not None:
        agent.restore_adaptation(snapshot)
    return rows, trace_rows


def mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return math.nan, math.nan
    return float(np.mean(arr)), float(np.std(arr))


def finite_correlation(xs, ys):
    pairs = [
        (float(x), float(y)) for x, y in zip(xs, ys)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 2:
        return math.nan
    x, y = np.asarray(pairs, dtype=np.float64).T
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def binary_auc(labels, scores):
    positives = [
        float(score) for label, score in zip(labels, scores)
        if bool(label) and math.isfinite(float(score))
    ]
    negatives = [
        float(score) for label, score in zip(labels, scores)
        if not bool(label) and math.isfinite(float(score))
    ]
    if not positives or not negatives:
        return math.nan
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += float(positive > negative)
            wins += 0.5 * float(positive == negative)
    return wins / (len(positives) * len(negatives))


def switch_detection_metrics(trace_rows, threshold: float,
                             max_delay_steps: int):
    auc = binary_auc(
        [row["switched"] for row in trace_rows],
        [row["context_error"] for row in trace_rows],
    )
    delays = []
    detected = 0
    by_episode = {}
    for row in trace_rows:
        by_episode.setdefault(int(row["episode"]), []).append(row)
    for episode_rows in by_episode.values():
        for switch_row in [row for row in episode_rows if row["switched"]]:
            switch_step = int(switch_row["step"])
            next_switch = min(
                [
                    int(row["step"]) for row in episode_rows
                    if row["switched"] and int(row["step"]) > switch_step
                ] or [switch_step + max_delay_steps + 1]
            )
            limit = min(switch_step + max_delay_steps, next_switch - 1)
            candidates = [
                row for row in episode_rows
                if switch_step <= int(row["step"]) <= limit
                and math.isfinite(float(row["context_error"]))
                and float(row["context_error"]) >= threshold
            ]
            if candidates:
                detected += 1
                delays.append(int(candidates[0]["step"]) - switch_step)
    total = sum(bool(row["switched"]) for row in trace_rows)
    return {
        "switch_auc": auc,
        "median_detection_delay": (
            float(np.median(delays)) if delays else math.nan
        ),
        "detection_rate": detected / total if total else math.nan,
        "detection_events": total,
        "detection_threshold": float(threshold),
        "detection_window_steps": int(max_delay_steps),
    }


def summarize(run_meta, task_rows, switching_rows, trace_rows,
              detection_threshold: float, detection_window_steps: int):
    detection = switch_detection_metrics(
        trace_rows, detection_threshold, detection_window_steps)
    rows = []
    for split in ("train", "test"):
        split_rows = [r for r in task_rows if r["split"] == split]
        task_means = [r["return_mean"] for r in split_rows]
        term_rates = [r["terminated_rate"] for r in split_rows]
        mean_return, std_tasks = mean_std(task_means)
        term_mean, _ = mean_std(term_rates)
        rows.append({
            **run_meta,
            "metric_group": "stationary",
            "split": split,
            "n_tasks": len(split_rows),
            "episodes_per_task": (
                split_rows[0]["episodes"] if split_rows else 0),
            "return_mean": mean_return,
            "return_std_across_tasks": std_tasks,
            "terminated_rate_mean": term_mean,
            "switching_episodes": "",
            "switch_return_mean": "",
            "switch_return_std": "",
            "switch_count_mean": "",
            "done_before_first_switch_rate": "",
            "done_step_mean": "",
            "termination_count_mean": "",
            "switch_auc": "",
            "median_detection_delay": "",
            "detection_rate": "",
            "detection_events": "",
            "detection_threshold": "",
            "detection_window_steps": "",
            "latent_mae_before": "",
            "latent_mae_after": "",
            "latent_correlation_before": "",
            "latent_correlation_after": "",
            "switch_action_alignment_rate": "",
        })

    switch_returns = [r["return"] for r in switching_rows]
    switch_counts = [r["switch_count"] for r in switching_rows]
    done_before = [float(r["done_before_first_switch"]) for r in switching_rows]
    done_steps = [r["done_step"] for r in switching_rows]
    termination_counts = [r["termination_count"] for r in switching_rows]
    latent_spans = [r["switch_latent_span"] for r in switching_rows]
    sw_mean, sw_std = mean_std(switch_returns)
    sw_count_mean, _ = mean_std(switch_counts)
    done_before_mean, _ = mean_std(done_before)
    done_step_mean, _ = mean_std(done_steps)
    termination_count_mean, _ = mean_std(termination_counts)
    latent_span_mean, _ = mean_std(latent_spans)
    latent_mae_before, _ = mean_std([
        r["latent_abs_error_before"] for r in trace_rows
        if math.isfinite(float(r["latent_abs_error_before"]))
    ])
    latent_mae_after, _ = mean_std([
        r["latent_abs_error_after"] for r in trace_rows
        if math.isfinite(float(r["latent_abs_error_after"]))
    ])
    true_latent = [r["true_latent_before_0"] for r in trace_rows]
    latent_corr_before = finite_correlation(
        true_latent, [r["estimated_latent_before_0"] for r in trace_rows])
    latent_corr_after = finite_correlation(
        true_latent, [r["estimated_latent_after_0"] for r in trace_rows])
    switch_alignment, _ = mean_std([
        float(r["action_task_id"] == r["physics_action_task_id"])
        for r in trace_rows
        if r["physics_action_task_id"] != r["physics_task_before"]
    ])
    mode_rows = [
        row for row in trace_rows
        if int(row.get("true_mode_before", -1)) >= 0
    ]
    mode_accuracy_before, _ = mean_std([
        row["mode_correct_before"] for row in mode_rows])
    mode_accuracy_after, _ = mean_std([
        row["mode_correct_after"] for row in mode_rows])
    expected_mode_mae_before, _ = mean_std([
        row["expected_mode_abs_error_before"] for row in mode_rows])
    expected_mode_mae_after, _ = mean_std([
        row["expected_mode_abs_error_after"] for row in mode_rows])
    expected_mode_correlation_before = finite_correlation(
        [row["true_mode_before"] for row in mode_rows],
        [row["expected_mode_before"] for row in mode_rows])
    expected_mode_correlation_after = finite_correlation(
        [row["true_mode_before"] for row in mode_rows],
        [row["expected_mode_after"] for row in mode_rows])
    rows.append({
        **run_meta,
        "metric_group": "switching",
        "split": "test_sequence",
        "n_tasks": "",
        "episodes_per_task": "",
        "return_mean": "",
        "return_std_across_tasks": "",
        "terminated_rate_mean": "",
        "switching_episodes": len(switching_rows),
        "switch_return_mean": sw_mean,
        "switch_return_std": sw_std,
        "switch_count_mean": sw_count_mean,
        "done_before_first_switch_rate": done_before_mean,
        "done_step_mean": done_step_mean,
        "termination_count_mean": termination_count_mean,
        "switch_latent_span_mean": latent_span_mean,
        "latent_mae_before": latent_mae_before,
        "latent_mae_after": latent_mae_after,
        "latent_correlation_before": latent_corr_before,
        "latent_correlation_after": latent_corr_after,
        "switch_action_alignment_rate": switch_alignment,
        "mode_accuracy_before": mode_accuracy_before,
        "mode_accuracy_after": mode_accuracy_after,
        "expected_mode_mae_before": expected_mode_mae_before,
        "expected_mode_mae_after": expected_mode_mae_after,
        "expected_mode_correlation_before": (
            expected_mode_correlation_before),
        "expected_mode_correlation_after": expected_mode_correlation_after,
        **detection,
    })
    return rows


def evaluate_run(run_dir: Path, args: argparse.Namespace):
    config = load_config(run_dir)
    env = make_env(config, seed_offset=0)
    train_tasks = env.sample_tasks(config.task_num)
    validation_tasks = env.sample_tasks(config.test_task_num)
    if args.heldout_task_stream == "reserved":
        reserved_count = int(getattr(config, "reserved_test_task_num", 0))
        if reserved_count <= 0:
            raise ValueError(
                "reserved held-out stream requested but checkpoint config "
                "has reserved_test_task_num=0")
        test_tasks = env.sample_tasks(reserved_count)
    else:
        test_tasks = validation_tasks

    agent = make_algo(config.algo, env.obs_dim, env.act_dim, config)
    if hasattr(agent, "set_task_metadata"):
        agent.set_task_metadata(train_tasks)
    replay_buffer = ReplayBuffer(
        env.obs_dim,
        env.act_dim,
        capacity=1,
        belief_dim=getattr(agent, "belief_dim", 0),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(tmpdir)
        start_iter, total_steps = load_checkpoint(
            str(run_dir / "checkpoints"),
            agent,
            replay_buffer,
            logger,
            config.algo,
            load_replay_buffer=False,
        )
    require_checkpoint_iteration(
        start_iter, args.min_checkpoint_next_iter)

    context_source, advantage_enabled = resolve_bapr_v2_eval_mode(
        agent, args.bapr_v2_context_source, args.bapr_v2_advantage)
    if (args.fixed_oracle_mode_id is not None
            and context_source != agent.CONTEXT_ORACLE):
        raise ValueError(
            "--fixed-oracle-mode-id requires "
            "--bapr-v2-context-source oracle")
    eval_mode_ids = oracle_eval_mode_ids(agent, args, context_source)
    policy_graphdef = nnx.graphdef(agent.policy)
    context_graphdef = None
    transition_context_graphdef = None
    rollout_critic_graphdef = None
    if getattr(agent, "uses_transition_context", False):
        transition_context_graphdef = nnx.graphdef(agent.context_net)
        rollout_critic_graphdef = nnx.graphdef(agent.critic)
    elif hasattr(agent, "context_net"):
        context_graphdef = nnx.graphdef(agent.context_net)

    all_task_rows = []
    all_switching_rows = []
    all_trace_rows = []
    all_summary_rows = []
    for eval_mode_id in eval_mode_ids:
        fixed_oracle_task = None
        if eval_mode_id is not None:
            matches = [
                task for task in test_tasks
                if isinstance(task, dict)
                and int(task.get("mode_id", -1)) == eval_mode_id
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"expected one task for fixed oracle mode "
                    f"{eval_mode_id}, found {len(matches)}")
            fixed_oracle_task = matches[0]

        # Recreate the eval environment so every context sees the same physical
        # task sequence and stochastic event stream.
        eval_env = make_env(config, seed_offset=args.eval_seed_offset)
        eval_env.build_rollout_fn(
            policy_graphdef, context_graphdef,
            transition_context_graphdef=transition_context_graphdef,
            critic_graphdef=rollout_critic_graphdef,
            direct_policy_context=getattr(
                agent, "uses_regime_context", False))
        max_tasks = args.max_tasks
        task_rows = []
        if not args.switching_only:
            if not args.stationary_test_only:
                task_rows.extend(evaluate_task_split(
                    agent, eval_env, config, train_tasks, "train",
                    args.episodes_per_task, max_tasks, args.rng_seed,
                    context_source, advantage_enabled, fixed_oracle_task))
            task_rows.extend(evaluate_task_split(
                agent, eval_env, config, test_tasks, "test",
                args.episodes_per_task, max_tasks, args.rng_seed + 10_000,
                context_source, advantage_enabled, fixed_oracle_task))
        switching_rows, trace_rows = evaluate_switching(
            agent, eval_env, config, test_tasks,
            args.switching_episodes, args.switching_period_steps,
            args.rng_seed + 20_000, context_source, advantage_enabled,
            fixed_oracle_task)

        run_meta = {
            "run_name": run_dir.name,
            "env": config.env_name.replace("-v2", ""),
            "algo": config.algo,
            "seed": int(config.seed),
            "checkpoint_next_iter": int(start_iter),
            "checkpoint_total_steps": int(total_steps),
            "eval_context_source": args.bapr_v2_context_source,
            "eval_oracle_mode_id": (
                "dynamic" if eval_mode_id is None else int(eval_mode_id)),
            "eval_advantage": args.bapr_v2_advantage,
            "heldout_task_stream": args.heldout_task_stream,
        }
        for row in task_rows:
            row.update(run_meta)
        for row in switching_rows:
            row.update(run_meta)
        for row in trace_rows:
            row.update(run_meta)
        summary_rows = summarize(
            run_meta, task_rows, switching_rows, trace_rows,
            float(config.bapr_v2_gate_error_threshold),
            args.detection_window_steps)
        all_task_rows.extend(task_rows)
        all_switching_rows.extend(switching_rows)
        all_trace_rows.extend(trace_rows)
        all_summary_rows.extend(summary_rows)
        if hasattr(eval_env, "close"):
            eval_env.close()

    if hasattr(env, "close"):
        env.close()
    return (
        all_task_rows, all_switching_rows,
        all_trace_rows, all_summary_rows)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
    else:
        fieldnames = []
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--resume", action="store_true",
        help="Scheduler staging marker; evaluation always loads the checkpoint.")
    parser.add_argument(
        "--resume-from", type=Path,
        help="Scheduler-injected checkpoint anchor; bundle loading is fixed.")
    parser.add_argument("--episodes-per-task", type=int, default=3)
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Limit train/test tasks per split for smoke tests.")
    parser.add_argument("--switching-episodes", type=int, default=5)
    parser.add_argument(
        "--switching-only", action="store_true",
        help="Skip the expensive stationary 40/40 task sweep.")
    parser.add_argument("--switching-period-steps", type=int, default=500)
    parser.add_argument(
        "--heldout-task-stream", choices=("validation", "reserved"),
        default="validation",
        help="Use stream 2 for model selection or untouched stream 3 for final reporting.")
    parser.add_argument("--detection-window-steps", type=int, default=50)
    parser.add_argument(
        "--bapr-v2-context-source",
        choices=("checkpoint", "robust", "oracle", "learned"),
        default="checkpoint",
        help="Override only the BAPR-v2 deployment context for mechanism audits.")
    parser.add_argument(
        "--bapr-v2-advantage", choices=("checkpoint", "on", "off"),
        default="checkpoint",
        help="Override only the BAPR-v2 critic-LCB action gate.")
    parser.add_argument(
        "--fixed-oracle-mode-id", type=int,
        help=("Hold privileged policy context at one mode while physics tasks "
              "still vary; requires explicit oracle context."))
    parser.add_argument(
        "--oracle-context-ladder", action="store_true",
        help=("Evaluate dynamic oracle and fixed modes 0..latent_dim-1 after "
              "one checkpoint load; requires explicit oracle context."))
    parser.add_argument("--rng-seed", type=int, default=20260707)
    parser.add_argument(
        "--eval-seed-offset", type=int, default=1000,
        help="Independent environment/event-stream seed offset.")
    parser.add_argument(
        "--stationary-test-only", action="store_true",
        help="Evaluate only the test task list before the switching stream.")
    parser.add_argument(
        "--min-checkpoint-next-iter", type=int,
        help="Fail before evaluation output if the loaded checkpoint is older.")
    args = parser.parse_args()

    all_task_rows = []
    all_switching_rows = []
    all_trace_rows = []
    all_summary_rows = []
    for idx, run_dir in enumerate(args.run_dir, start=1):
        print(f"[{idx}/{len(args.run_dir)}] {run_dir}", flush=True)
        task_rows, switching_rows, trace_rows, summary_rows = evaluate_run(
            run_dir, args)
        all_task_rows.extend(task_rows)
        all_switching_rows.extend(switching_rows)
        all_trace_rows.extend(trace_rows)
        all_summary_rows.extend(summary_rows)
        jax.clear_caches()

    write_csv(args.out_dir / "task_returns.csv", all_task_rows)
    write_csv(args.out_dir / "switching_returns.csv", all_switching_rows)
    write_csv(args.out_dir / "switching_trace.csv", all_trace_rows)
    write_csv(args.out_dir / "summary.csv", all_summary_rows)
    print(f"Wrote {args.out_dir}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
