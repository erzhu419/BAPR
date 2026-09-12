"""Main training entry point for JAX-based RL experiments.

Usage:
    conda run -n jax-rl python -m jax_experiments.train --algo resac --env Hopper-v2
    conda run -n jax-rl python -m jax_experiments.train --algo escp  --env Hopper-v2
    conda run -n jax-rl python -m jax_experiments.train --algo bapr  --env Hopper-v2
"""
import os
import sys
import argparse
import hashlib
import json
import time
import faulthandler
import platform
import numpy as np
from dataclasses import asdict

# Must set CUDA lib path before JAX import
NVIDIA_LIB = None
for p in sys.path:
    candidate = os.path.join(p, "nvidia")
    if os.path.isdir(candidate):
        NVIDIA_LIB = candidate
        break
if NVIDIA_LIB is None:
    # Fallback: find via site-packages
    import site
    for sp in site.getsitepackages():
        candidate = os.path.join(sp, "nvidia")
        if os.path.isdir(candidate):
            NVIDIA_LIB = candidate
            break
if NVIDIA_LIB is not None:
    lib_dirs = []
    for subdir in os.listdir(NVIDIA_LIB):
        lib_path = os.path.join(NVIDIA_LIB, subdir, "lib")
        if os.path.isdir(lib_path):
            lib_dirs.append(lib_path)
    if lib_dirs:
        os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import jax
import jax.numpy as jnp
from flax import nnx

from jax_experiments.configs.default import Config
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.common.logging import Logger
from jax_experiments.common.checkpoint import save_checkpoint, load_checkpoint, has_checkpoint
from jax_experiments.envs.brax_env import BraxNonstationaryEnv as NonstationaryEnv
from jax_experiments.envs.discrete_mode_env import DiscreteModePiecewiseEnv
from jax_experiments.envs.stochastic_mode_env import StochasticModeEnv


def make_env(config, seed_offset=0):
    """Build training or eval env based on config.env_type.

    env_type='continuous': original ESCP-style continuous task family
    env_type='discrete_mode': K=4 discrete semantic modes (BAPR sweet spot)
    env_type='stochastic_mode': persistent mode-conditioned transition noise
    """
    env_type = getattr(config, 'env_type', 'continuous')
    if env_type == 'discrete_mode':
        return DiscreteModePiecewiseEnv(
            env_name=config.env_name,
            mean_dwell_iters=config.discrete_mean_dwell_iters,
            steps_per_iter=config.samples_per_iter,
            dwell_distribution=config.discrete_dwell_distribution,
            reward_shaping=config.discrete_reward_shaping,
            seed=config.seed + seed_offset,
            backend=config.brax_backend,
            mode_variant=getattr(config, 'discrete_mode_variant', 'orig'))
    if env_type == 'stochastic_mode':
        fixed_mode_id = int(getattr(
            config, 'stochastic_mode_fixed_id', -1))
        return StochasticModeEnv(
            env_name=config.env_name,
            family=config.stochastic_mode_family,
            dwell_steps=config.stochastic_mode_dwell_steps,
            dwell_distribution=config.stochastic_mode_dwell_distribution,
            seed=config.seed + seed_offset,
            backend=config.brax_backend,
            fixed_mode_id=(
                None if fixed_mode_id < 0 else fixed_mode_id))
    return NonstationaryEnv(
        config.env_name, rand_params=config.varying_params,
        log_scale_limit=config.log_scale_limit,
        task_scale_distribution=getattr(
            config, 'task_scale_distribution', 'exp'),
        task_seed_salt=getattr(config, 'task_seed_salt', 0),
        seed=config.seed + seed_offset, backend=config.brax_backend)


def make_algo(algo_name: str, obs_dim: int, act_dim: int, config: Config):
    """Instantiate the chosen algorithm."""
    if algo_name == "resac":
        from jax_experiments.algos.resac import RESAC
        return RESAC(obs_dim, act_dim, config, seed=config.seed)
    elif algo_name == "escp":
        from jax_experiments.algos.escp import ESCP
        return ESCP(obs_dim, act_dim, config, seed=config.seed)
    elif algo_name == "bapr":
        from jax_experiments.algos.bapr import BAPR
        return BAPR(obs_dim, act_dim, config, seed=config.seed)
    elif algo_name == "bapr_v2":
        from jax_experiments.algos.bapr_v2 import BAPRv2
        return BAPRv2(obs_dim, act_dim, config, seed=config.seed)
    elif algo_name == "bapr_v3":
        from jax_experiments.algos.bapr_v3 import BAPRv3
        return BAPRv3(obs_dim, act_dim, config, seed=config.seed)
    elif algo_name == "bapr_regime":
        from jax_experiments.algos.bapr_regime import BAPRRegime
        return BAPRRegime(obs_dim, act_dim, config, seed=config.seed)
    elif algo_name == "bapr_v4":
        from jax_experiments.algos.bapr_v4 import BAPRv4
        return BAPRv4(obs_dim, act_dim, config, seed=config.seed)
    elif algo_name == "bapr_v5":
        from jax_experiments.algos.bapr_v5 import BAPRv5
        return BAPRv5(obs_dim, act_dim, config, seed=config.seed)
    elif algo_name == "bapr_v6":
        from jax_experiments.algos.bapr_v6 import BAPRv6
        return BAPRv6(obs_dim, act_dim, config, seed=config.seed)
    elif algo_name == "anchored_regime_sac":
        from jax_experiments.algos.anchored_regime_sac import (
            AnchoredRegimeSAC,
        )
        return AnchoredRegimeSAC(
            obs_dim, act_dim, config, seed=config.seed)
    elif algo_name == "frozen_anchored_regime_sac":
        from jax_experiments.algos.frozen_anchor_sac import (
            FrozenAnchoredRegimeSAC,
        )
        return FrozenAnchoredRegimeSAC(
            obs_dim, act_dim, config, seed=config.seed)
    elif algo_name == "frozen_mode_residual_sac":
        from jax_experiments.algos.frozen_anchor_sac import (
            FrozenModeResidualSAC,
        )
        return FrozenModeResidualSAC(
            obs_dim, act_dim, config, seed=config.seed)
    elif algo_name == "sac":
        from jax_experiments.algos.sac_base import SACBase
        return SACBase(obs_dim, act_dim, config, seed=config.seed)
    elif algo_name == "regime_sac":
        from jax_experiments.algos.regime_sac import RegimeSAC
        return RegimeSAC(obs_dim, act_dim, config, seed=config.seed)
    # --- Ablation variants ---
    elif algo_name == "bapr_no_bocd":
        from jax_experiments.algos.bapr_ablations import BAPRNoBocd
        return BAPRNoBocd(obs_dim, act_dim, config, seed=config.seed)
    elif algo_name == "bapr_no_rmdm":
        from jax_experiments.algos.bapr_ablations import BAPRNoRmdm
        return BAPRNoRmdm(obs_dim, act_dim, config, seed=config.seed)
    elif algo_name == "bapr_no_adapt_beta":
        from jax_experiments.algos.bapr_ablations import BAPRNoAdaptBeta
        return BAPRNoAdaptBeta(obs_dim, act_dim, config, seed=config.seed)
    elif algo_name == "bapr_fixed_decay":
        from jax_experiments.algos.bapr_ablations import BAPRFixedDecay
        return BAPRFixedDecay(obs_dim, act_dim, config, seed=config.seed)
    elif algo_name == "bad_bapr":
        from jax_experiments.algos.bad_bapr import BadBAPR
        return BadBAPR(obs_dim, act_dim, config, seed=config.seed)
    elif algo_name == "bapr_unsupervised":
        from jax_experiments.algos.bapr_unsupervised import BAPRUnsupervised
        return BAPRUnsupervised(obs_dim, act_dim, config, seed=config.seed)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def _eval_policy_state(agent, config: Config):
    from flax import nnx
    if getattr(config, 'use_ema_eval', False) and hasattr(agent, 'ema_policy'):
        policy_params = nnx.state(agent.ema_policy, nnx.Param)
    else:
        policy_params = nnx.state(agent.policy, nnx.Param)
    context_params = None
    if hasattr(agent, 'context_net'):
        context_params = nnx.state(agent.context_net, nnx.Param)

    belief_vec = None
    if (not getattr(agent, "uses_transition_context", False)
            and hasattr(agent, 'belief_dim') and agent.belief_dim > 0):
        if hasattr(agent, '_build_belief_jax'):
            belief_vec = agent._build_belief_jax()
        else:
            import jax.numpy as _jnp
            belief_vec = _jnp.asarray(agent.belief_tracker.belief,
                                      dtype=_jnp.float32)
    return policy_params, context_params, belief_vec


def _oracle_latent_for_eval(agent, context_source: int, task=None):
    """Expose privileged task metadata only to explicit oracle evaluation."""
    if int(context_source) == int(agent.CONTEXT_ORACLE):
        if task is not None:
            agent.set_eval_task(task)
        return agent.oracle_latent
    return jnp.zeros_like(agent.oracle_latent)


def _select_eval_tasks(tasks):
    if tasks is None or len(tasks) == 0:
        return [None]
    if len(tasks) <= 10:
        return tasks
    return [tasks[0]]


def _strict_horizon_returns(rewards, dones, n_episodes: int, horizon: int):
    ep_rewards = []
    for ep in range(n_episodes):
        start = ep * horizon
        end = start + horizon
        ep_r = 0.0
        for i in range(start, end):
            ep_r += float(rewards[i])
            if float(dones[i]) > 0.5:
                break
        ep_rewards.append(ep_r)
    return ep_rewards


def paired_safe_target(base_return, adaptive_return,
                       base_termination, adaptive_termination, config):
    """Task-level safe-adaptation target from paired deterministic rollouts."""
    scale = float(max(config.bapr_v2_paired_return_scale, 1e-6))
    denominator = abs(float(base_return)) + abs(float(adaptive_return)) + scale
    gain = (float(adaptive_return) - float(base_return)) / denominator
    risk_gap = float(adaptive_termination) - float(base_termination)
    gain_temp = float(max(config.bapr_v2_paired_gain_temperature, 1e-6))
    risk_temp = float(max(config.bapr_v2_paired_risk_temperature, 1e-6))
    gain_score = 1.0 / (1.0 + np.exp(-(
        gain - float(config.bapr_v2_paired_gain_margin)) / gain_temp))
    risk_score = 1.0 / (1.0 + np.exp((
        risk_gap - float(config.bapr_v2_paired_risk_tolerance)) / risk_temp))
    return float(gain_score * risk_score), float(gain), float(risk_gap)


def calibrate_paired_safe_targets(agent, env, config: Config, tasks):
    """Compare frozen robust and oracle-teacher policies on every train task."""
    episodes = int(config.bapr_v2_paired_calibration_episodes)
    if episodes <= 0 or not getattr(agent, "uses_transition_context", False):
        return None
    policy_params, context_params, _ = _eval_policy_state(agent, config)
    critic_params = nnx.state(agent.critic, nnx.Param)
    horizon = int(config.max_episode_steps)
    snapshot = agent.snapshot_adaptation()
    targets, gains, risk_gaps = [], [], []
    for task_idx, task in enumerate(tasks):
        env.set_task(task)
        agent.set_eval_task(task)
        key = jax.random.fold_in(jax.random.PRNGKey(87_000), task_idx)
        outputs = []
        for source in (agent.CONTEXT_ROBUST, agent.CONTEXT_ORACLE):
            rewards, dones, _ = env.eval_rollout_adaptive(
                policy_params, context_params,
                agent.context_net.initial_state(), agent.oracle_latent,
                episodes * horizon, key, episode_horizon=horizon,
                critic_params=critic_params, context_source=source,
                advantage_enabled=False,
                advantage_margin=config.bapr_v2_advantage_margin,
                advantage_lcb_scale=config.bapr_v2_advantage_lcb_scale)
            returns = _strict_horizon_returns(
                rewards, dones, episodes, horizon)
            terminations = [
                float(np.any(np.asarray(dones)[
                    ep * horizon:(ep + 1) * horizon] > 0.5))
                for ep in range(episodes)
            ]
            outputs.append((float(np.mean(returns)),
                            float(np.mean(terminations))))
        target, gain, risk_gap = paired_safe_target(
            outputs[0][0], outputs[1][0],
            outputs[0][1], outputs[1][1], config)
        targets.append(target)
        gains.append(gain)
        risk_gaps.append(risk_gap)
    agent.restore_adaptation(snapshot)
    agent.set_safe_task_targets(targets, gains, risk_gaps)
    print(
        "  Paired safety calibration: "
        f"target={np.mean(targets):.3f}+-{np.std(targets):.3f}, "
        f"gain={np.mean(gains):+.3f}, risk_gap={np.mean(risk_gaps):+.3f}")
    return np.asarray(targets), np.asarray(gains), np.asarray(risk_gaps)


def evaluate_stationary(agent, env, config: Config, tasks=None,
                        n_episodes: int = 10, context_source=None,
                        advantage_enabled=None, record_diagnostics=True):
    """Deterministic fixed-task eval with strict episode horizon.

    Behavior depends on `tasks`:
      - None or len(tasks)==1: single task (legacy behavior)
      - len(tasks) <= 10 (e.g. discrete_mode K modes): rotate through ALL
        tasks, run n_episodes per task, average across all; this measures
        cross-mode generalization (BAPR's design target).
      - len(tasks) > 10 (continuous, 40 tasks): use first task only
        (full sweep too expensive every eval).

    Uses EMA policy only when config.use_ema_eval is enabled. In the BAPR
    redesign runs, EMA was a diagnostic path and could mask online-policy
    regressions, so online evaluation is the default.
    """
    policy_params, context_params, belief_vec = _eval_policy_state(agent, config)
    rng_key = jax.random.PRNGKey(42)  # fixed key for reproducible eval
    horizon = int(config.max_episode_steps)
    n_steps = n_episodes * horizon
    eval_tasks = _select_eval_tasks(tasks)

    all_rewards = []
    adaptation_snapshot = None
    recurrent_snapshot = None
    if getattr(agent, "uses_transition_context", False):
        adaptation_snapshot = agent.snapshot_adaptation()
        selected_source = (
            agent.CONTEXT_LEARNED
            if context_source is None else int(context_source))
        selected_advantage = (
            agent.advantage_gate_active()
            if advantage_enabled is None else bool(advantage_enabled))
    elif getattr(agent, "uses_recurrent_context", False):
        recurrent_snapshot = agent.snapshot_recurrent_context()
    for task in eval_tasks:
        if task is not None:
            env.set_task(task)
            if getattr(agent, "uses_regime_context", False):
                agent.set_eval_task(task)
        if getattr(agent, "uses_transition_context", False):
            oracle_latent = _oracle_latent_for_eval(
                agent, selected_source, task)
            initial_adaptation = agent.context_net.initial_state()
            rew_np, done_np, diagnostics = env.eval_rollout_adaptive(
                policy_params, context_params, initial_adaptation,
                oracle_latent, n_steps, rng_key,
                episode_horizon=horizon,
                critic_params=nnx.state(agent.critic, nnx.Param),
                context_source=selected_source,
                advantage_enabled=selected_advantage,
                advantage_margin=config.bapr_v2_advantage_margin,
                advantage_lcb_scale=config.bapr_v2_advantage_lcb_scale)
            if record_diagnostics:
                agent._last_eval_context_gate = diagnostics["gate_mean"]
                agent._last_eval_context_error = diagnostics["error_mean"]
                agent._last_eval_advantage = diagnostics["advantage_mean"]
                agent._last_eval_advantage_gate = diagnostics[
                    "advantage_gate_mean"]
        elif getattr(agent, "uses_recurrent_context", False):
            rew_np, done_np = env.eval_rollout_recurrent(
                policy_params, context_params, n_steps, rng_key,
                episode_horizon=horizon)
        else:
            if getattr(agent, "uses_regime_context", False):
                belief_vec = agent._build_belief_jax()
            rew_np, done_np = env.eval_rollout(
                policy_params, n_steps, rng_key,
                context_params=context_params, belief_vec=belief_vec,
                episode_horizon=horizon)
        all_rewards.extend(
            _strict_horizon_returns(rew_np, done_np, n_episodes, horizon))

    if adaptation_snapshot is not None:
        agent.restore_adaptation(adaptation_snapshot)
    if recurrent_snapshot is not None:
        agent.restore_recurrent_context(recurrent_snapshot)

    return float(np.mean(all_rewards)), float(np.std(all_rewards))


def _reset_eval_switch_schedule(env, tasks, config: Config, period_steps: int):
    period_steps = max(1, int(period_steps))
    if tasks is None or len(tasks) == 0:
        return
    configure = getattr(env, "configure_eval_switching", None)
    if callable(configure):
        configure(tasks, period_steps)
        return
    if hasattr(env, "_mode_sys"):
        env.current_task_id = 0
        env._set_sys(env._mode_sys[0])
        env._step_counter = 0
        env.dwell_steps_mean = period_steps
        env.dwell_distribution = "fixed"
        env._next_switch_step = period_steps
        env._switch_history = [(0, 0)]
    else:
        env.set_nonstationary_para(
            tasks or [], changing_period=period_steps, changing_interval=1)


def _task_numeric_vector(task):
    """Flatten numeric task metadata for protocol-level distance checks."""
    if not isinstance(task, dict):
        return np.empty((0,), dtype=np.float64)
    parts = []
    for key in sorted(task):
        try:
            value = np.asarray(task[key], dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            continue
        if value.size:
            parts.append(value)
    return np.concatenate(parts) if parts else np.empty((0,), dtype=np.float64)


def _select_eval_switch_sequence(env, tasks, episode_index: int = 0):
    """Choose a deterministic, maximally separated task pair for eval.

    Continuous task lists are sampled in random order, so blindly evaluating
    task indices 0 -> 1 can produce an almost stationary stream.  Select the
    farthest standardized pair and alternate direction across episodes.  The
    discrete-mode environment owns pre-built mode systems and keeps its native
    schedule.
    """
    selected = list(tasks or [])
    source_indices = list(range(len(selected)))
    if len(selected) < 2 or hasattr(env, "_mode_sys"):
        return selected, source_indices

    vectors = [_task_numeric_vector(task) for task in selected]
    widths = {vector.size for vector in vectors}
    if len(widths) != 1 or not widths or next(iter(widths)) == 0:
        pair = (0, len(selected) - 1)
    else:
        matrix = np.stack(vectors, axis=0)
        scale = np.std(matrix, axis=0)
        normalized = (matrix - np.mean(matrix, axis=0)) / np.where(
            scale > 1e-8, scale, 1.0)
        distances = np.sum(
            np.square(normalized[:, None, :] - normalized[None, :, :]),
            axis=-1,
        )
        np.fill_diagonal(distances, -np.inf)
        pair = tuple(int(value) for value in np.unravel_index(
            int(np.argmax(distances)), distances.shape))
    if int(episode_index) % 2:
        pair = (pair[1], pair[0])
    return [selected[pair[0]], selected[pair[1]]], [pair[0], pair[1]]


def _eval_task_id_for_action(agent, env, current_task_id: int,
                             context_source=None) -> int:
    """Give a privileged oracle the task used by the upcoming physics step."""
    explicit_oracle = (
        context_source is not None
        and int(context_source) == int(getattr(agent, "CONTEXT_ORACLE", -1)))
    if (not explicit_oracle
            and getattr(agent, "context_mode", None) != "oracle"):
        return int(current_task_id)
    peek = getattr(env, "task_id_for_next_step", None)
    if not callable(peek):
        return int(current_task_id)
    return int(peek())


def evaluate_switching_online(agent, env, config: Config, tasks=None,
                              n_episodes: int = 1, period_steps: int = 500,
                              context_source=None, advantage_enabled=None):
    """Fixed-horizon deterministic streams where the environment switches.

    A physics termination resets only the simulator state. The mode clock and
    causal adaptation state continue, otherwise fragile environments can end
    before the first switch and the metric never tests adaptation.
    """
    horizon = int(config.max_episode_steps)
    ep_rewards = []
    switch_counts = []
    adaptation_snapshot = None
    recurrent_snapshot = None
    if getattr(agent, "uses_transition_context", False):
        adaptation_snapshot = agent.snapshot_adaptation()
        selected_source = (
            agent.CONTEXT_LEARNED
            if context_source is None else int(context_source))
        selected_advantage = (
            agent.advantage_gate_active()
            if advantage_enabled is None else bool(advantage_enabled))
    elif getattr(agent, "uses_recurrent_context", False):
        recurrent_snapshot = agent.snapshot_recurrent_context()

    for ep in range(n_episodes):
        switch_tasks, _ = _select_eval_switch_sequence(env, tasks, ep)
        _reset_eval_switch_schedule(env, switch_tasks, config, period_steps)
        if adaptation_snapshot is not None:
            agent.reset_adaptation()
        if recurrent_snapshot is not None:
            agent.reset_recurrent_context()
        obs = env.reset()
        ep_r = 0.0
        switches = 0
        prev_task_id = int(getattr(env, "current_task_id", 0))
        for _ in range(horizon):
            action_task_id = _eval_task_id_for_action(
                agent, env, prev_task_id,
                context_source=(
                    selected_source if adaptation_snapshot is not None
                    else None))
            if (adaptation_snapshot is not None
                    and selected_source == agent.CONTEXT_ORACLE
                    and switch_tasks
                    and 0 <= action_task_id < len(switch_tasks)):
                agent.set_eval_task(switch_tasks[action_task_id])
            elif getattr(agent, "uses_regime_context", False):
                agent.set_oracle_task_id(action_task_id)
            pre_obs = obs
            if adaptation_snapshot is not None:
                action = agent.select_action(
                    obs, deterministic=True,
                    context_source=selected_source,
                    advantage_enabled=selected_advantage)
            else:
                action = agent.select_action(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            if adaptation_snapshot is not None:
                agent.observe_transition(
                    pre_obs, action, reward, obs, done)
            if recurrent_snapshot is not None:
                agent.finish_recurrent_step(done)
            ep_r += float(reward)
            task_id = int(getattr(env, "current_task_id", prev_task_id))
            if task_id != prev_task_id:
                switches += 1
                prev_task_id = task_id
            if done:
                obs = env.reset()
        ep_rewards.append(ep_r)
        switch_counts.append(switches)

    if adaptation_snapshot is not None:
        agent.restore_adaptation(adaptation_snapshot)
    if recurrent_snapshot is not None:
        agent.restore_recurrent_context(recurrent_snapshot)

    return (
        float(np.mean(ep_rewards)),
        float(np.std(ep_rewards)),
        float(np.mean(switch_counts)),
    )


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _json_safe(value.item())
        return {
            "shape": list(value.shape),
            "min": float(np.min(value)),
            "max": float(np.max(value)),
            "mean": float(np.mean(value)),
        }
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _task_summary(tasks, max_tasks: int = 3):
    if tasks is None:
        return {"count": 0, "examples": []}
    examples = [_json_safe(task) for task in list(tasks)[:max_tasks]]
    return {"count": len(tasks), "examples": examples}


def write_protocol_signature(log_dir: str, config: Config, train_tasks,
                             test_tasks, reserved_test_tasks,
                             checkpoint_loaded: bool,
                             start_iteration: int, total_steps: int):
    config_dict = asdict(config)
    for key, value in vars(config).items():
        if key not in config_dict and not key.startswith("_"):
            config_dict[key] = value
    signature = {
        "argv": sys.argv,
        "host": platform.node(),
        "python": sys.executable,
        "config": _json_safe(config_dict),
        "checkpoint_loaded": bool(checkpoint_loaded),
        "start_iteration": int(start_iteration),
        "total_steps_at_start": int(total_steps),
        "eval": {
            "protocol": getattr(config, "eval_protocol", "stationary"),
            "stationary_id": "train_tasks representative/all-K",
            "stationary_ood": "test_tasks representative/all-K",
            "switching_online": (
                "enabled only when eval_protocol=full; deterministic policy, "
                "max-distance task pair with alternating direction, true env "
                "switching, no eval-time gradient updates; BAPR-v2 "
                "does perform causal latent-state updates from observed "
                "transitions"
            ),
            "deterministic": True,
            "horizon_enforced": True,
            "max_episode_steps": int(config.max_episode_steps),
            "eval_episodes": int(config.eval_episodes),
            "switching_eval_episodes": int(getattr(
                config, "eval_switching_episodes", 1)),
            "switching_period_steps": int(getattr(
                config, "eval_switching_period_steps", 500)),
        },
        "train_tasks": _task_summary(train_tasks),
        "test_tasks": _task_summary(test_tasks),
        "reserved_test_tasks": _task_summary(reserved_test_tasks),
    }
    path = os.path.join(log_dir, "protocol_signature.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(signature, fh, indent=2, sort_keys=True)


def _sha256_array_fields(fields):
    """Hash typed array fields without relying on pickle serialization."""
    combined = hashlib.sha256()
    field_hashes = {}
    for name in sorted(fields):
        value = np.ascontiguousarray(np.asarray(fields[name]))
        metadata = json.dumps({
            "name": str(name),
            "dtype": value.dtype.str,
            "shape": list(value.shape),
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload = value.view(np.uint8).tobytes()
        digest = hashlib.sha256(metadata + b"\0" + payload).hexdigest()
        field_hashes[str(name)] = digest
        combined.update(metadata)
        combined.update(b"\0")
        combined.update(payload)
    return combined.hexdigest(), field_hashes


def _write_json_atomic(path: str, payload, *, overwrite: bool = True) -> None:
    if not overwrite and os.path.exists(path):
        raise FileExistsError(
            f"refusing to overwrite immutable audit artifact: {path}")
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, path)


def build_resume_boundary_audit(agent, replay_buffer, config: Config,
                                iteration: int, total_steps: int):
    """Record the shared-restart state before the first resumed rollout."""
    expected_iteration = getattr(
        config, "resume_boundary_expected_iteration", None)
    expected_total_steps = getattr(
        config, "resume_boundary_expected_total_steps", None)
    expected_update_count = getattr(
        config, "resume_boundary_expected_update_count", None)
    if (expected_iteration is not None
            and int(iteration) != int(expected_iteration)):
        raise RuntimeError(
            "resume boundary iteration mismatch: "
            f"actual={iteration}, expected={expected_iteration}")
    if (expected_total_steps is not None
            and int(total_steps) != int(expected_total_steps)):
        raise RuntimeError(
            "resume boundary total_steps mismatch: "
            f"actual={total_steps}, expected={expected_total_steps}")
    if (expected_update_count is not None
            and int(agent.update_count) != int(expected_update_count)):
        raise RuntimeError(
            "resume boundary update_count mismatch: "
            f"actual={int(agent.update_count)}, "
            f"expected={expected_update_count}")

    stage_fn = getattr(agent, "training_stage", None)
    stage = stage_fn(iteration) if callable(stage_fn) else None
    if stage not in ("robust", "teacher"):
        raise RuntimeError(
            "resume boundary audit is restricted to robust/teacher fork "
            f"stages, got {stage!r}")
    source_fn = getattr(agent, "rollout_context_source", None)
    context_source = (
        int(source_fn(iteration)) if callable(source_fn) else None)
    update_flags_fn = getattr(agent, "controller_update_flags", None)
    update_flags = (
        list(map(bool, update_flags_fn(iteration)))
        if callable(update_flags_fn) else None)
    gate_fn = getattr(agent, "train_policy_gate", None)

    policy_equivalence = None
    policy = getattr(agent, "policy", None)
    task_latents = getattr(agent, "task_latents", None)
    robust_source = int(getattr(agent, "CONTEXT_ROBUST", -1))
    oracle_source = int(getattr(agent, "CONTEXT_ORACLE", -1))
    if context_source in (robust_source, oracle_source):
        if policy is None or not callable(policy) or task_latents is None:
            raise RuntimeError(
                "resume boundary audit requires a callable paired policy "
                "and non-empty task_latents")
        task_latents_np = np.asarray(task_latents)
        if (task_latents_np.ndim != 2
                or task_latents_np.shape[0] <= 0
                or task_latents_np.shape[1] <= 0
                or not np.all(np.isfinite(task_latents_np))):
            raise RuntimeError(
                "resume boundary audit received invalid task_latents")
        count = min(32, int(replay_buffer.size))
        if count <= 0:
            raise RuntimeError(
                "resume boundary audit requires a non-empty replay buffer")
        observations = jnp.asarray(replay_buffer.obs[:count])
        base_mean, base_log_std = policy(observations, None)
        base_mean_np = np.asarray(base_mean)
        base_log_std_np = np.asarray(base_log_std)
        finite = bool(
            np.all(np.isfinite(base_mean_np))
            and np.all(np.isfinite(base_log_std_np)))
        max_mean_abs = 0.0
        max_log_std_abs = 0.0
        tested_contexts = []
        if context_source == robust_source:
            contexts = [(
                "robust_zero",
                jnp.zeros(
                    (count, task_latents_np.shape[1] + 1),
                    dtype=observations.dtype),
            )]
        else:
            contexts = []
            for task_index, latent in enumerate(task_latents_np):
                latent_batch = jnp.broadcast_to(
                    jnp.asarray(latent, dtype=observations.dtype),
                    (count, int(latent.size)))
                contexts.append((
                    f"oracle_task_{task_index}",
                    jnp.concatenate([
                        latent_batch,
                        jnp.ones((count, 1), dtype=observations.dtype),
                    ], axis=-1),
                ))
        for label, context in contexts:
            candidate_mean, candidate_log_std = policy(observations, context)
            candidate_mean_np = np.asarray(candidate_mean)
            candidate_log_std_np = np.asarray(candidate_log_std)
            finite = finite and bool(
                np.all(np.isfinite(candidate_mean_np))
                and np.all(np.isfinite(candidate_log_std_np)))
            max_mean_abs = max(max_mean_abs, float(np.max(np.abs(
                candidate_mean_np - base_mean_np))))
            max_log_std_abs = max(max_log_std_abs, float(np.max(np.abs(
                candidate_log_std_np - base_log_std_np))))
            tested_contexts.append(label)
        tolerance = 1e-6
        passed = bool(
            finite
            and max_mean_abs <= tolerance
            and max_log_std_abs <= tolerance)
        policy_equivalence = {
            "observations": count,
            "task_latents": int(task_latents_np.shape[0]),
            "tested_contexts": tested_contexts,
            "tested_rollout_context_source": context_source,
            "finite": finite,
            "max_abs_mean_diff": max_mean_abs,
            "max_abs_log_std_diff": max_log_std_abs,
            "tolerance": tolerance,
            "pass": passed,
        }
        if not passed:
            raise RuntimeError(
                "resume boundary policy failed the pre-rollout canary: "
                f"source={context_source}, finite={finite}, "
                f"max_abs_mean_diff={max_mean_abs:.3e}, "
                f"max_abs_log_std_diff={max_log_std_abs:.3e}")
    else:
        raise RuntimeError(
            "resume boundary audit supports only robust/oracle rollout "
            f"sources, got {context_source}")

    return {
        "schema": "bapr.resume-boundary-audit.v1",
        "semantics": "shared-checkpoint/common-restart; not exact continuation",
        "run_name": str(config.run_name),
        "seed": int(config.seed),
        "iteration": int(iteration),
        "total_steps_before_rollout": int(total_steps),
        "replay_size_before_rollout": int(replay_buffer.size),
        "agent_update_count_before_rollout": int(agent.update_count),
        "training_stage": stage,
        "rollout_context_source": context_source,
        "controller_update_flags": update_flags,
        "train_policy_gate": (
            bool(gate_fn(iteration)) if callable(gate_fn) else None),
        "conditioned_warmstarted": bool(getattr(
            agent, "_conditioned_warmstarted", False)),
        "policy_equivalence": policy_equivalence,
    }


def finish_resume_boundary_audit(payload, recent_rollout, log_dir: str,
                                 config: Config):
    """Write a pre-gradient physical-rollout canary for paired forks."""
    if recent_rollout is None:
        raise RuntimeError(
            "resume boundary audit requires a non-random recent rollout")
    required = ("obs", "act", "rew", "next_obs", "done", "task_id")
    missing = [name for name in required if name not in recent_rollout]
    if missing:
        raise RuntimeError(
            "resume boundary rollout is missing fields: " + ", ".join(missing))
    arrays = {
        name: np.asarray(recent_rollout[name]) for name in required
    }
    lengths = {name: int(value.shape[0]) if value.ndim else -1
               for name, value in arrays.items()}
    expected_transitions = int(config.samples_per_iter)
    if (set(lengths.values()) != {expected_transitions}
            or expected_transitions <= 0):
        raise RuntimeError(
            "resume boundary rollout length mismatch: "
            f"actual={lengths}, expected={expected_transitions}")
    if (arrays["obs"].ndim != 2 or arrays["act"].ndim != 2
            or arrays["obs"].shape[1] <= 0 or arrays["act"].shape[1] <= 0
            or arrays["next_obs"].shape != arrays["obs"].shape
            or arrays["rew"].ndim not in (1, 2)
            or (arrays["rew"].ndim == 2 and arrays["rew"].shape[1] != 1)
            or arrays["done"].ndim not in (1, 2)
            or (arrays["done"].ndim == 2 and arrays["done"].shape[1] != 1)
            or arrays["task_id"].ndim != 1):
        raise RuntimeError(
            "resume boundary rollout has invalid field shapes: "
            + str({name: list(value.shape) for name, value in arrays.items()}))
    for name, value in arrays.items():
        if value.dtype.kind not in "biuf" or not np.all(np.isfinite(value)):
            raise RuntimeError(
                f"resume boundary rollout field {name} is non-numeric or "
                "contains non-finite values")
    task_ids = arrays["task_id"]
    if (not np.all(task_ids == np.asarray(task_ids, dtype=np.int64))
            or np.any(task_ids < 0)
            or np.any(task_ids >= int(config.task_num))):
        raise RuntimeError(
            "resume boundary rollout contains invalid task_id values")
    done = arrays["done"]
    if np.any(done < 0) or np.any(done > 1):
        raise RuntimeError(
            "resume boundary rollout contains invalid done values")
    digest, field_hashes = _sha256_array_fields(arrays)
    payload = dict(payload)
    payload["physical_rollout"] = {
        "fields": list(required),
        "sha256": digest,
        "field_sha256": field_hashes,
        "transitions": int(np.asarray(recent_rollout["rew"]).shape[0]),
        "finite": True,
        "validated_shapes": {
            name: list(value.shape) for name, value in arrays.items()
        },
        "excludes_context_by_design": True,
    }
    path = os.path.join(log_dir, "resume_boundary_audit.json")
    _write_json_atomic(path, payload, overwrite=False)
    print(
        "  Resume boundary audit: "
        f"stage={payload['training_stage']}, "
        f"physical_sha256={digest[:16]}..., path={path}")


def collect_samples(agent, env, replay_buffer, config, n_steps: int,
                     current_iter: int = 0):
    """Collect n_steps via GPU scan-fused rollout.

    Fuses policy + physics + auto-reset in ONE XLA call.
    Data stays on GPU: rollout returns JAX arrays -> push_batch_jax -> buffer(GPU).
    """
    is_random = replay_buffer.size < config.start_train_steps
    rng_key = jax.random.PRNGKey(config.seed + replay_buffer.size)

    if is_random:
        if hasattr(agent, '_last_ema_rollout_active'):
            agent._last_ema_rollout_active = False
        # Random exploration: sequential API (small overhead ok)
        obs = env.reset()
        episode_rewards = []
        episode_reward = 0.0
        episode_start = True
        for _ in range(n_steps):
            peek = getattr(env, "task_id_for_next_step", None)
            action_task_id = int(
                peek() if callable(peek) else env.current_task_id)
            if (getattr(agent, "uses_transition_context", False)
                    or getattr(agent, "uses_regime_context", False)):
                agent.set_oracle_task_id(action_task_id)
                context = agent.rollout_context(current_iter)
            else:
                context = None
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            if getattr(agent, "uses_transition_context", False):
                next_context = agent.observe_transition(
                    obs, action, reward, next_obs, done)
            elif getattr(agent, "uses_regime_context", False):
                next_context = agent.context_for_task_id(
                    int(getattr(env, "current_task_id", action_task_id)))
            else:
                next_context = None
            transition_task_id = int(info.get(
                "mode_used", action_task_id))
            replay_buffer.push(
                obs, action, reward, next_obs, done, transition_task_id,
                belief=context, next_belief=next_context,
                episode_start=episode_start)
            episode_reward += reward
            obs = next_obs
            episode_start = bool(done)
            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0.0
                obs = env.reset()
        return episode_rewards, None  # no recent_rollout in random phase
    else:
        # A fused scan uses one static System. Environments with an explicit
        # regime clock are therefore split into equal chunks below; simulator
        # state is preserved while the hidden System changes between chunks.
        use_ema_rollout = (
            getattr(config, 'use_ema_rollout', False)
            and hasattr(agent, 'ema_policy')
            and current_iter >= getattr(config, 'ema_rollout_start_iter', 0)
        )
        if getattr(config, 'ema_rollout_require_reg_latched', False):
            use_ema_rollout = (
                use_ema_rollout
                and bool(getattr(agent, '_reg_latched', False))
            )
        if hasattr(agent, '_last_ema_rollout_active'):
            agent._last_ema_rollout_active = bool(use_ema_rollout)
        if use_ema_rollout:
            policy_params = nnx.state(agent.ema_policy, nnx.Param)
        else:
            policy_params = nnx.state(agent.policy, nnx.Param)
        rollout_warmup = current_iter < config.context_warmup_iters

        if getattr(agent, "uses_transition_context", False):
            previous_task = getattr(
                agent, "_v2_last_rollout_task_id", None)
            context_params = nnx.state(agent.context_net, nnx.Param)
            critic_params = nnx.state(agent.critic, nnx.Param)
            switch_steps = int(config.bapr_v2_switch_rollout_steps)
            if switch_steps <= 0:
                switch_steps = int(getattr(env, "rollout_chunk_steps", 0))
            if switch_steps <= 0 or switch_steps >= n_steps:
                chunk_sizes = [n_steps]
            else:
                chunk_sizes = [switch_steps] * (n_steps // switch_steps)
                if n_steps % switch_steps:
                    chunk_sizes.append(n_steps % switch_steps)
            chunk_keys = jax.random.split(rng_key, len(chunk_sizes))
            transition_chunks = [[] for _ in range(10)]
            task_id_chunks = []
            rollout_task_ids = []
            final_adaptation = agent.adaptation_state
            for chunk_index, (chunk_size, chunk_key) in enumerate(
                    zip(chunk_sizes, chunk_keys)):
                rollout_task_id = int(env.current_task_id)
                rollout_task_ids.append(rollout_task_id)
                agent.set_oracle_task_id(rollout_task_id)
                chunk, _, final_adaptation = env.rollout_adaptive(
                    policy_params, context_params, final_adaptation,
                    agent.oracle_latent, chunk_size, chunk_key,
                    warmup=rollout_warmup,
                    critic_params=critic_params,
                    context_source=agent.rollout_context_source(current_iter),
                    advantage_enabled=agent.advantage_gate_active(current_iter),
                    advantage_margin=config.bapr_v2_advantage_margin,
                    advantage_lcb_scale=config.bapr_v2_advantage_lcb_scale,
                    continue_state=chunk_index > 0)
                for values, output in zip(transition_chunks, chunk):
                    values.append(output)
                task_id_chunks.append(jnp.full(
                    chunk_size, rollout_task_id, dtype=jnp.int32))
            transitions = tuple(
                jnp.concatenate(values, axis=0)
                for values in transition_chunks)
            (obs, act, rew, nobs, done, context,
             next_context, context_error, advantage,
             advantage_gate) = transitions
            task_ids = jnp.concatenate(task_id_chunks, axis=0)
            agent._v2_rollout_task_changed = bool(
                (previous_task is not None
                 and int(previous_task) != rollout_task_ids[0])
                or len(set(rollout_task_ids)) > 1)
            agent._v2_last_rollout_task_id = int(rollout_task_ids[-1])
            agent.adaptation_state = final_adaptation
            agent._last_context_error = float(jnp.mean(context_error[-64:]))
            agent._last_context_gate = float(jnp.mean(context[-64:, -1]))
            agent._last_advantage = float(jnp.mean(advantage[-64:]))
            agent._last_advantage_gate = float(
                jnp.mean(advantage_gate[-64:]))
            replay_buffer.push_batch_jax(
                obs, act, rew.reshape(-1, 1), nobs,
                done.reshape(-1, 1), task_ids,
                belief=context, next_belief=next_context)
            recent_rollout = {
                "obs": obs,
                "act": act,
                "rew": rew.reshape(-1, 1),
                "next_obs": nobs,
                "done": done.reshape(-1, 1),
                "task_id": task_ids,
                "context": context,
                "next_context": next_context,
                "context_error": context_error,
                "advantage": advantage,
                "advantage_gate": advantage_gate,
                "rollout_task_id": int(rollout_task_ids[-1]),
                "rollout_task_ids": np.asarray(
                    rollout_task_ids, dtype=np.int32),
            }
            rew_np = np.asarray(rew)
            done_np = np.asarray(done)
            ep_rewards = []
            ep_reward = 0.0
            for reward, terminal in zip(rew_np, done_np):
                ep_reward += float(reward)
                if float(terminal) > 0.5:
                    ep_rewards.append(ep_reward)
                    ep_reward = 0.0
            return ep_rewards, recent_rollout

        if getattr(agent, "uses_recurrent_context", False):
            context_params = nnx.state(agent.context_net, nnx.Param)
            switch_steps = int(getattr(env, "rollout_chunk_steps", 0))
            if switch_steps <= 0 or switch_steps >= n_steps:
                chunk_sizes = [n_steps]
            else:
                chunk_sizes = [switch_steps] * (n_steps // switch_steps)
                if n_steps % switch_steps:
                    chunk_sizes.append(n_steps % switch_steps)
            chunk_keys = jax.random.split(rng_key, len(chunk_sizes))
            transition_chunks = [[] for _ in range(5)]
            task_id_chunks = []
            rollout_task_ids = []
            recurrent_hidden = agent.context_net.initial_hidden((1,))
            previous_action = jnp.zeros(
                (agent.act_dim,), dtype=jnp.float32)
            for chunk_index, (chunk_size, chunk_key) in enumerate(
                    zip(chunk_sizes, chunk_keys)):
                rollout_task_id = int(env.current_task_id)
                rollout_task_ids.append(rollout_task_id)
                (chunk, _, recurrent_hidden, previous_action) = (
                    env.rollout_recurrent(
                        policy_params, context_params, recurrent_hidden,
                        previous_action, chunk_size, chunk_key,
                        warmup=rollout_warmup,
                        context_noise_sigma=float(getattr(
                            config, "escp_bottleneck_sigma", 0.0)),
                        continue_state=chunk_index > 0))
                for values, output in zip(transition_chunks, chunk):
                    values.append(output)
                task_id_chunks.append(jnp.full(
                    chunk_size, rollout_task_id, dtype=jnp.int32))
            obs, act, rew, nobs, done = tuple(
                jnp.concatenate(values, axis=0)
                for values in transition_chunks)
            task_ids = jnp.concatenate(task_id_chunks, axis=0)
            replay_buffer.push_batch_jax(
                obs, act, rew.reshape(-1, 1), nobs,
                done.reshape(-1, 1), task_ids)

            episode_rewards = []
            episode_reward = 0.0
            for reward, terminal in zip(np.asarray(rew), np.asarray(done)):
                episode_reward += float(reward)
                if float(terminal) > 0.5:
                    episode_rewards.append(episode_reward)
                    episode_reward = 0.0
            return episode_rewards, {
                "obs": obs,
                "act": act,
                "rew": rew.reshape(-1, 1),
                "next_obs": nobs,
                "done": done.reshape(-1, 1),
                "task_id": task_ids,
                "rollout_task_id": int(rollout_task_ids[-1]),
                "rollout_task_ids": np.asarray(
                    rollout_task_ids, dtype=np.int32),
            }

        context_params = None
        if hasattr(agent, 'context_net'):
            context_params = nnx.state(agent.context_net, nnx.Param)

        # v15+: pass BAPR's belief vector if the agent is belief-conditioned.
        # _build_belief_jax encapsulates the format choice (legacy ρ vs new
        # concat([ρ, μ])); fall back to raw legacy belief for compat.
        belief_vec = None
        if hasattr(agent, 'belief_dim') and agent.belief_dim > 0:
            if hasattr(agent, '_build_belief_jax'):
                belief_vec = agent._build_belief_jax()
            else:
                belief_vec = jnp.asarray(agent.belief_tracker.belief, dtype=jnp.float32)
        direct_regime_context = bool(
            getattr(agent, "uses_regime_context", False))

        # New stochastic-mode environments expose rollout_chunk_steps so every
        # baseline observes the same mode clock as BAPR-v3.  The simulator state
        # is carried across chunks; only the hidden mode/system changes.
        switch_steps = int(getattr(env, "rollout_chunk_steps", 0))
        if switch_steps <= 0 or switch_steps >= n_steps:
            chunk_sizes = [n_steps]
        else:
            chunk_sizes = [switch_steps] * (n_steps // switch_steps)
            if n_steps % switch_steps:
                chunk_sizes.append(n_steps % switch_steps)
        chunk_keys = jax.random.split(rng_key, len(chunk_sizes))
        transition_chunks = [[] for _ in range(5)]
        task_id_chunks = []
        rollout_task_ids = []
        context_chunks = []
        next_context_chunks = []
        for chunk_index, (chunk_size, chunk_key) in enumerate(
                zip(chunk_sizes, chunk_keys)):
            rollout_task_id = int(env.current_task_id)
            rollout_task_ids.append(rollout_task_id)
            chunk_belief = belief_vec
            if direct_regime_context:
                agent.set_oracle_task_id(rollout_task_id)
                chunk_belief = agent.rollout_context(current_iter)
                if rollout_warmup:
                    chunk_belief = jnp.zeros_like(chunk_belief)
            chunk, _ = env.rollout(
                policy_params, chunk_size, chunk_key,
                context_params=context_params, belief_vec=chunk_belief,
                warmup=rollout_warmup,
                continue_state=chunk_index > 0)
            for values, output in zip(transition_chunks, chunk):
                values.append(output)
            task_id_chunks.append(jnp.full(
                chunk_size, rollout_task_id, dtype=jnp.int32))
            if direct_regime_context:
                next_mode_id = int(env.current_task_id)
                agent.set_oracle_task_id(next_mode_id)
                next_chunk_belief = agent.rollout_context(current_iter)
                if rollout_warmup:
                    next_chunk_belief = jnp.zeros_like(next_chunk_belief)
                current_batch = jnp.broadcast_to(
                    chunk_belief[None, :],
                    (chunk_size, agent.belief_dim))
                next_batch = current_batch.at[-1].set(next_chunk_belief)
                context_chunks.append(current_batch)
                next_context_chunks.append(next_batch)
        obs, act, rew, nobs, done = tuple(
            jnp.concatenate(values, axis=0)
            for values in transition_chunks)
        task_ids = jnp.concatenate(task_id_chunks, axis=0)
        rew_np = np.asarray(rew)
        done_np = np.asarray(done)
        ep_rewards = []
        ep_reward = 0.0
        for reward, terminal in zip(rew_np, done_np):
            ep_reward += float(reward)
            if float(terminal) > 0.5:
                ep_rewards.append(ep_reward)
                ep_reward = 0.0

        # Zero-copy push: JAX arrays go directly to GPU-native replay buffer
        # GPT-5.5 advice #2: store the belief that produced this rollout.
        # If warmup is on, we also stored zero context — store zero belief
        # so the critic sees a consistent (zero ctx, zero belief) input.
        if direct_regime_context:
            stored_belief = jnp.concatenate(context_chunks, axis=0)
            stored_next_belief = jnp.concatenate(
                next_context_chunks, axis=0)
        elif rollout_warmup and belief_vec is not None:
            stored_belief = jnp.zeros_like(belief_vec)
            stored_next_belief = stored_belief
        else:
            stored_belief = belief_vec
            stored_next_belief = belief_vec
        replay_buffer.push_batch_jax(obs, act, rew.reshape(-1, 1),
                                     nobs, done.reshape(-1, 1), task_ids,
                                     belief=stored_belief,
                                     next_belief=stored_next_belief)

        # NOTE: BAPR's BOCD must detect regime changes from observable signals
        # (reward shifts, Q-std spikes, surprise). The oracle reset that previously
        # lived here (`if env.current_task_id != prev_task_id: agent.reset_episode()`)
        # was using PRIVILEGED hidden task IDs and suppressed the very first
        # post-switch surprise — exactly the signal BAPR needs to detect changes.
        # Removed per GPT-5.5 review (2026-04-26). BOCD now operates in zero-shot
        # mode: it must surface change-points from rewards/Q-std alone.

        # Return recent rollout dict so BAPR can compute surprise from THIS
        # rollout (not random replay batches that mix old regimes).
        recent_rollout = {
            "obs": obs,
            "act": act,
            "rew": rew.reshape(-1, 1),
            "next_obs": nobs,
            "done": done.reshape(-1, 1),
            "task_id": task_ids,
            "rollout_task_id": int(rollout_task_ids[-1]),
            "rollout_task_ids": np.asarray(
                rollout_task_ids, dtype=np.int32),
        }
        if direct_regime_context:
            recent_rollout["context"] = stored_belief
            recent_rollout["next_context"] = stored_next_belief
        return ep_rewards, recent_rollout


def train(config: Config):
    """Main training loop."""
    print(f"{'='*60}")
    print(f"  Algorithm: {config.algo.upper()}")
    print(f"  Environment: {config.env_name}")
    print(f"  Varying: {config.varying_params}")
    print(f"  Seed: {config.seed}")
    print(f"  Updates/iter: {config.updates_per_iter}  Samples/iter: {config.samples_per_iter}")
    print(f"  Ensemble size: {config.ensemble_size}  Hidden dim: {config.hidden_dim}")
    print(f"  Brax backend: {config.brax_backend}")
    print(f"  JAX devices: {jax.devices()}")
    print(f"{'='*60}")

    # ``action_space.sample()`` uses NumPy during the initial random replay
    # warm-up.  Seed that path explicitly so same-seed runs do not diverge
    # before the first JAX update merely because they are separate processes.
    np.random.seed(int(config.seed))

    # Create environment (continuous or discrete-mode based on config.env_type)
    env = make_env(config, seed_offset=0)
    obs_dim = env.obs_dim
    act_dim = env.act_dim

    # Sample tasks
    train_tasks = env.sample_tasks(config.task_num)
    test_tasks = env.sample_tasks(config.test_task_num)
    reserved_test_tasks = (
        env.sample_tasks(config.reserved_test_task_num)
        if int(config.reserved_test_task_num) > 0 else [])

    # Setup non-stationary switching
    env.set_nonstationary_para(train_tasks, config.changing_period, config.changing_interval)

    # Create agent
    agent = make_algo(config.algo, obs_dim, act_dim, config)
    if hasattr(agent, "set_task_metadata"):
        agent.set_task_metadata(train_tasks)

    # Build scan-fused rollout (compiles policy+physics into one XLA call)
    policy_graphdef = nnx.graphdef(agent.policy)
    context_graphdef = None
    transition_context_graphdef = None
    recurrent_context_graphdef = None
    recurrent_context_non_params = None
    rollout_critic_graphdef = None
    if getattr(agent, "uses_transition_context", False):
        transition_context_graphdef = nnx.graphdef(agent.context_net)
        rollout_critic_graphdef = nnx.graphdef(agent.critic)
    elif getattr(agent, "uses_recurrent_context", False):
        (recurrent_context_graphdef, _,
         recurrent_context_non_params) = nnx.split(
            agent.context_net, nnx.Param, ...)
    elif hasattr(agent, 'context_net'):
        context_graphdef = nnx.graphdef(agent.context_net)
    env.build_rollout_fn(
        policy_graphdef, context_graphdef,
        transition_context_graphdef=transition_context_graphdef,
        recurrent_context_graphdef=recurrent_context_graphdef,
        recurrent_context_non_params=recurrent_context_non_params,
        critic_graphdef=rollout_critic_graphdef,
        direct_policy_context=getattr(
            agent, "uses_regime_context", False))
    context_kind = (
        "transition" if transition_context_graphdef is not None
        else "recurrent" if recurrent_context_graphdef is not None
        else "state" if context_graphdef is not None
        else "direct" if getattr(agent, "uses_regime_context", False)
        else "no")
    print(f"  Built scan-fused rollout for {config.env_name} "
          f"(context={context_kind})")

    # Replay buffer
    # GPT-5.5 advice #2: replay buffer needs belief_dim so it can store
    # per-transition belief vectors (zero-width when belief_conditioning off).
    belief_dim_for_buffer = getattr(agent, 'belief_dim', 0)
    replay_buffer = ReplayBuffer(obs_dim, act_dim, capacity=config.replay_size,
                                  belief_dim=belief_dim_for_buffer)

    # Logging
    run_name = config.run_name or f"{config.algo}_{config.env_name}_{config.seed}"
    log_dir = os.path.join(config.save_root, run_name, "logs")
    model_dir = os.path.join(config.save_root, run_name, "models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    logger = Logger(log_dir)
    diag_path = os.path.join(log_dir, "python_runtime_diagnostics.log")
    try:
        diag_fh = open(diag_path, "a", buffering=1)
        faulthandler.enable(file=diag_fh, all_threads=True)
        print(
            f"[python_diag] ts={time.strftime('%Y-%m-%dT%H:%M:%S%z')} "
            f"pid={os.getpid()} ppid={os.getppid()} host={platform.node()} "
            f"python={sys.executable} argv={' '.join(sys.argv)}",
            file=diag_fh,
            flush=True)
        for key in (
            "CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS",
            "XLA_PYTHON_CLIENT_PREALLOCATE",
            "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_FLAGS",
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "TMPDIR"):
            print(f"[python_diag] env {key}={os.environ.get(key, '')}",
                  file=diag_fh, flush=True)
    except Exception as exc:
        print(f"[python_diag] failed to enable diagnostics: {exc}")

    # --- Checkpoint resume ---
    start_iteration = 0
    total_steps = 0
    checkpoint_loaded = False
    ckpt_dir = os.path.join(config.save_root, run_name, "checkpoints")
    if getattr(config, 'resume', False) and has_checkpoint(ckpt_dir):
        start_iteration, total_steps = load_checkpoint(
            ckpt_dir, agent, replay_buffer, logger, config.algo)
        checkpoint_loaded = True
        print(f"Resumed from checkpoint: iter={start_iteration}, steps={total_steps}")
    else:
        print(f"Logging to: {log_dir}")
        print(f"Starting training... (first {config.start_train_steps} steps are random exploration)")
    min_resume_iteration = int(getattr(config, "min_resume_iteration", -1))
    if min_resume_iteration >= 0 and (
            not checkpoint_loaded
            or start_iteration < min_resume_iteration):
        raise RuntimeError(
            "resume checkpoint is older than required: "
            f"loaded_iter={start_iteration if checkpoint_loaded else 'none'}, "
            f"required_iter>={min_resume_iteration}, ckpt_dir={ckpt_dir}")
    write_protocol_signature(
        log_dir, config, train_tasks, test_tasks, reserved_test_tasks,
        checkpoint_loaded,
        start_iteration, total_steps)
    if config.algo in (
            "bapr_v2", "bapr_v3", "bapr_regime", "bapr_v4",
            "bapr_v5", "bapr_v6", "anchored_regime_sac",
            "frozen_anchored_regime_sac",
            "frozen_mode_residual_sac"):
        print(
            f"{config.algo.upper()} config: "
            f"mode={config.bapr_v2_mode}, "
            f"policy_mode={config.bapr_v2_policy_mode}, "
            f"num_experts={config.bapr_v2_num_experts}, "
            f"latent_scale_mode={config.bapr_v2_latent_scale_mode}, "
            f"policy_context_source={config.bapr_v2_policy_context_source}, "
            f"training_schedule={config.bapr_v2_training_schedule}, "
            f"base_pretrain_iters={config.bapr_v2_base_pretrain_iters}, "
            f"teacher_iters={config.bapr_v2_teacher_iters}, "
            f"student_iters={config.bapr_v2_student_iters}, "
            f"warmstart_conditioned={config.bapr_v2_warmstart_conditioned}, "
            f"latent_dim={config.bapr_v2_latent_dim}, "
            f"history={config.bapr_v2_context_length}, "
            f"fallback={config.bapr_v2_use_fallback}, "
            f"residual_delta={config.bapr_v2_residual_delta}, "
            f"policy_gate_init={config.bapr_v2_policy_gate_init}, "
            f"action_deviation_weight={config.bapr_v2_action_deviation_weight}, "
            f"switch_rollout_steps={config.bapr_v2_switch_rollout_steps}, "
            f"paired_episodes={config.bapr_v2_paired_calibration_episodes}, "
            f"gate_supervision={config.bapr_v2_gate_supervision_weight}, "
            f"unsafe_deviation={config.bapr_v2_unsafe_deviation_weight}, "
            f"advantage_gate={config.bapr_v2_advantage_gate}, "
            f"advantage_margin={config.bapr_v2_advantage_margin}, "
            f"advantage_lcb_scale={config.bapr_v2_advantage_lcb_scale}, "
            f"train_advantage_constraint="
            f"{config.bapr_v2_train_advantage_constraint}, "
            f"train_advantage_lcb_scale="
            f"{config.bapr_v2_train_advantage_lcb_scale}, "
            f"train_advantage_margin="
            f"{config.bapr_v2_train_advantage_margin}, "
            f"train_update_filter={config.bapr_v2_train_update_filter}, "
            f"train_update_tolerance="
            f"{config.bapr_v2_train_update_tolerance}, "
            f"train_update_floor={config.bapr_v2_train_update_floor}, "
            f"actor_objective={config.bapr_v2_actor_objective}, "
            f"reg_weight={config.bapr_v2_reg_weight}, "
            f"likelihood={getattr(config, 'bapr_v3_likelihood', 'n/a')}, "
            f"variance_model={config.bapr_v3_variance_model}, "
            f"variance_bounds=({config.bapr_v3_variance_floor:g}, "
            f"{config.bapr_v3_variance_ceiling:g}), "
            f"variance_ema={config.bapr_v3_variance_ema:g}, "
            f"instant_classifier_weight="
            f"{config.bapr_v3_instant_classifier_weight:g}, "
            f"freeze_teacher="
            f"{config.bapr_v3_freeze_teacher_after_teacher}, "
            f"estimator_rollout_source="
            f"{config.bapr_v3_estimator_rollout_source}, "
            f"reset_context_on_resume="
            f"{config.bapr_v3_reset_context_on_resume}, "
            f"context_ladder={config.bapr_v3_eval_context_ladder}, "
            f"regime_inference_iters={config.bapr_regime_inference_iters}, "
            f"regime_adaptation_source="
            f"{config.bapr_regime_adaptation_source}, "
            f"regime_freeze_context="
            f"{config.bapr_regime_freeze_context_after_inference}, "
            f"regime_advantage_fallback="
            f"{config.bapr_regime_advantage_fallback}, "
            f"option_hold_steps={config.bapr_v4_option_hold_steps}, "
            f"option_confidence="
            f"{config.bapr_v4_option_confidence_threshold:g}, "
            f"option_margin={config.bapr_v4_option_margin_threshold:g}, "
            f"option_hysteresis="
            f"{config.bapr_v4_option_hysteresis_margin:g}, "
            f"v4_cusum=({config.bapr_v4_cusum_threshold:g}, "
            f"{config.bapr_v4_cusum_drift:g}), "
            f"v4_training_mix="
            f"{config.bapr_v4_training_robust_slots}/"
            f"{config.bapr_v4_training_source_period}")
    elif config.algo == "bapr":
        print(
            "BAPR config: "
            f"adaptation_mode={config.bapr_adaptation_mode}, "
            f"actor_objective={config.actor_objective}, "
            f"belief_conditioned={config.belief_conditioned}, "
            f"use_regime_belief={config.use_regime_belief}, "
            f"gate_warmup_iters={config.bapr_gate_warmup_iters}, "
            f"surprise_threshold={config.bapr_surprise_threshold}, "
            f"gate_max={config.bapr_gate_max}, "
            f"recent_frac_cap={config.bapr_recent_frac_cap}, "
            f"recent_frac_floor={config.bapr_recent_frac_floor}, "
            f"recent_true_floor={config.bapr_recent_true_floor}, "
            f"recent_floor_mode={config.bapr_recent_floor_mode}, "
            f"recent_floor_ratio_low={config.bapr_recent_floor_ratio_low}, "
            f"recent_floor_ratio_high={config.bapr_recent_floor_ratio_high}, "
            f"recent_floor_mid_frac={config.bapr_recent_floor_mid_frac}, "
            f"recent_floor_extreme_frac={config.bapr_recent_floor_extreme_frac}, "
            f"recent_disagreement_gate={config.bapr_recent_disagreement_gate}, "
            f"recent_open_if_reg_latched={config.bapr_recent_open_if_reg_latched}, "
            f"recent_qstd_threshold={config.bapr_recent_qstd_threshold}, "
            f"recent_qstd_ratio_threshold={config.bapr_recent_qstd_ratio_threshold}, "
            f"reg_disagreement_gate={config.bapr_reg_disagreement_gate}, "
            f"reg_warmup_iters={config.bapr_reg_warmup_iters}, "
            f"reg_max_iters={config.bapr_reg_max_iters}, "
            f"reg_latch={config.bapr_reg_latch}, "
            f"reg_require_both={config.bapr_reg_require_both}, "
            f"reg_qstd_threshold={config.bapr_reg_qstd_threshold}, "
            f"reg_qstd_ratio_threshold={config.bapr_reg_qstd_ratio_threshold}, "
            f"reg_emergency_gate={config.bapr_reg_emergency_gate}, "
            f"reg_emergency_scale={config.bapr_reg_emergency_scale}, "
            f"reg_emergency_qstd_threshold={config.bapr_reg_emergency_qstd_threshold}, "
            f"reg_emergency_qstd_ratio_threshold={config.bapr_reg_emergency_qstd_ratio_threshold}, "
            f"reg_latched_scale={config.bapr_reg_latched_scale}, "
            f"reg_perf_collapse_gate={config.bapr_reg_perf_collapse_gate}, "
            f"reg_perf_collapse_drop_frac={config.bapr_reg_perf_collapse_drop_frac}, "
            f"reg_perf_collapse_scale={config.bapr_reg_perf_collapse_scale}, "
            f"controller_mode={config.bapr_controller_mode}, "
            f"controller_drop_frac={config.bapr_controller_drop_frac}, "
            f"controller_soft_decay={config.bapr_controller_soft_decay}, "
            f"controller_soft_min_signal={config.bapr_controller_soft_min_signal}, "
            f"controller_low_qstd_ratio={config.bapr_controller_low_qstd_ratio}, "
            f"controller_latch={config.bapr_controller_latch}, "
            f"controller_release_drop_frac={config.bapr_controller_release_drop_frac}, "
            f"controller_max_active_iters={config.bapr_controller_max_active_iters}, "
            f"controller_min_active_iters={config.bapr_controller_min_active_iters}, "
            f"controller_exit_cooldown_iters={config.bapr_controller_exit_cooldown_iters}, "
            f"controller_exit_signal_threshold={config.bapr_controller_exit_signal_threshold}, "
            f"controller_exit_improve_frac={config.bapr_controller_exit_improve_frac}, "
            f"controller_exit_drawdown_frac={config.bapr_controller_exit_drawdown_frac}, "
            f"controller_latched_signal={config.bapr_controller_latched_signal}, "
            f"controller_reg_multiplier={config.bapr_controller_reg_multiplier}, "
            f"controller_reg_recover_iters={config.bapr_controller_reg_recover_iters}, "
            f"controller_recent_multiplier={config.bapr_controller_recent_multiplier}, "
            f"controller_actor_update_multiplier={config.bapr_controller_actor_update_multiplier}, "
            f"controller_actor_recover_iters={config.bapr_controller_actor_recover_iters}, "
            f"actor_lcb_perf_gate={config.bapr_actor_lcb_perf_gate}, "
            f"actor_lcb_perf_drop_frac={config.bapr_actor_lcb_perf_drop_frac}, "
            f"actor_lcb_perf_warmup_iters={config.bapr_actor_lcb_perf_warmup_iters}, "
            f"residual_delta={config.bapr_residual_delta}, "
            f"residual_gate_scale={config.bapr_residual_gate_scale}, "
            f"residual_adv_margin={config.bapr_residual_adv_margin}, "
            f"residual_adv_temp={config.bapr_residual_adv_temp}, "
            f"residual_qstd_scale={config.bapr_residual_qstd_scale}, "
            f"residual_behavior_weight={config.bapr_residual_behavior_weight}, "
            f"residual_action_penalty={config.bapr_residual_action_penalty}, "
            f"weight_reg={config.weight_reg}, "
            f"beta_ood={config.beta_ood}, "
            f"use_ema_eval={config.use_ema_eval}, "
            f"use_ema_rollout={config.use_ema_rollout}, "
            f"ema_rollout_start_iter={config.ema_rollout_start_iter}, "
            f"ema_rollout_require_reg_latched={config.ema_rollout_require_reg_latched}"
        )

    # Separate eval env (offset seed by 1000 for different mode-switch RNG)
    eval_env = make_env(config, seed_offset=1000)
    eval_env.build_rollout_fn(
        policy_graphdef, context_graphdef,
        transition_context_graphdef=transition_context_graphdef,
        recurrent_context_graphdef=recurrent_context_graphdef,
        recurrent_context_non_params=recurrent_context_non_params,
        critic_graphdef=rollout_critic_graphdef,
        direct_policy_context=getattr(
            agent, "uses_regime_context", False))

    initial_random_steps = int(
        getattr(config, "initial_random_steps", 0))
    if initial_random_steps and start_iteration == 0:
        if int(config.start_train_steps) != initial_random_steps:
            raise ValueError(
                "initial_random_steps requires an equal start_train_steps "
                "threshold")
        if replay_buffer.size > initial_random_steps:
            raise ValueError(
                "pre-iteration replay exceeds the registered random warmup: "
                f"size={replay_buffer.size}, target={initial_random_steps}")
        remaining = initial_random_steps - int(replay_buffer.size)
        if remaining:
            print(
                "Collecting checkpointed pre-iteration random warmup: "
                f"remaining={remaining}/{initial_random_steps}",
                flush=True)
            warmup_rewards, _ = collect_samples(
                agent, env, replay_buffer, config, remaining,
                current_iter=-1)
            total_steps += remaining
            if warmup_rewards:
                logger.log(
                    "initial_random_reward_mean",
                    float(np.mean(warmup_rewards)))
            logger.save()
            save_checkpoint(
                ckpt_dir, agent, replay_buffer, logger,
                -1, total_steps, config.algo)
        if (int(replay_buffer.size) != initial_random_steps
                or int(total_steps) < initial_random_steps):
            raise RuntimeError(
                "pre-iteration random warmup did not reach its boundary: "
                f"replay={replay_buffer.size}, steps={total_steps}, "
                f"target={initial_random_steps}")
        print(
            "Pre-iteration random warmup complete: "
            f"replay={replay_buffer.size}, total_steps={total_steps}",
            flush=True)

    total_steps = total_steps  # from checkpoint or 0
    start_time = time.time()
    collect_time_total = 0.0
    train_time_total = 0.0
    eval_time_total = 0.0

    for iteration in range(start_iteration, config.max_iters):
        iter_start = time.time()
        if hasattr(agent, "set_training_iteration"):
            agent.set_training_iteration(iteration)
        resume_boundary_audit = None
        if (bool(getattr(config, "resume_boundary_audit", False))
                and checkpoint_loaded
                and iteration == start_iteration):
            resume_boundary_audit = build_resume_boundary_audit(
                agent, replay_buffer, config, iteration, total_steps)
        if (getattr(agent, "training_schedule", None)
                == "constrained_deploy"
                and agent.training_stage(iteration) in (
                    "student", "deployment")
                and not agent._safe_targets_calibrated
                and int(config.bapr_v2_paired_calibration_episodes) > 0):
            calibrate_paired_safe_targets(
                agent, eval_env, config, train_tasks)
        if (hasattr(agent, "consume_replay_reset_request")
                and agent.consume_replay_reset_request()):
            replay_buffer.clear()
            print(
                "  Cleared replay at learned-latent deployment boundary; "
                "old oracle/stale contexts will not be sampled.")

        # --- Collect samples ---
        collect_start = time.time()
        prev_task_id = int(env.current_task_id)
        ep_rewards, recent_rollout = collect_samples(
            agent, env, replay_buffer, config, config.samples_per_iter,
            current_iter=iteration)
        if resume_boundary_audit is not None:
            finish_resume_boundary_audit(
                resume_boundary_audit, recent_rollout, log_dir, config)
        oracle_switched = int(env.current_task_id != prev_task_id)
        collect_time = time.time() - collect_start
        collect_time_total += collect_time
        total_steps += config.samples_per_iter

        if len(ep_rewards) > 0:
            logger.log("train_reward_mean", float(np.mean(ep_rewards)))
            logger.log("train_reward_std", float(np.std(ep_rewards)))

        # --- Training updates (fused via lax.scan) ---
        train_start = time.time()
        if replay_buffer.size >= config.start_train_steps:
            # GPU-native sampling: rng_key ensures fully on-device indexing
            sample_key = jax.random.PRNGKey(config.seed + iteration)
            _algo = config.algo
            _bapr_v2 = _algo in (
                "bapr_v2", "bapr_v3", "bapr_regime", "bapr_v4",
                "bapr_v5", "bapr_v6", "anchored_regime_sac",
                "frozen_anchored_regime_sac",
                "frozen_mode_residual_sac")
            _bapr_like = _algo in ("bapr", "bapr_no_rmdm", "bad_bapr",
                                    "bapr_unsupervised")
            _escp_like = _algo in ("escp", "bapr_no_bocd", "bapr_no_adapt_beta",
                                    "bapr_fixed_decay")
            _resac_like = _algo == "resac"

            # Change 3: belief-aware replay sampling for BAPR. When BOCD detects
            # a regime shift (high λ_w), sample more from recent transitions.
            if (_algo == "escp"
                    and getattr(agent, "uses_recurrent_context", False)):
                stacked = replay_buffer.sample_stacked_sequences(
                    config.updates_per_iter, config.batch_size,
                    int(getattr(config, "escp_history_length", 16)),
                    rng_key=sample_key)
            elif _bapr_like and hasattr(
                    replay_buffer, 'sample_stacked_mixed'):
                if hasattr(agent, "recent_replay_fraction"):
                    recent_frac = float(agent.recent_replay_fraction())
                else:
                    prev_lam = float(getattr(agent, "_current_weighted_lambda", 0.0))
                    recent_cap = float(getattr(config, "bapr_recent_frac_cap", 0.8))
                    recent_frac = min(recent_cap, max(0.0, prev_lam))
                stacked = replay_buffer.sample_stacked_mixed(
                    config.updates_per_iter, config.batch_size,
                    rng_key=sample_key, recent_frac=recent_frac,
                    recent_window=getattr(config, "recent_replay_window", 50_000))
            else:
                stacked = replay_buffer.sample_stacked(
                    config.updates_per_iter, config.batch_size, rng_key=sample_key)

            if _bapr_v2:
                metrics = agent.multi_update(
                    stacked, current_iter=iteration,
                    recent_rollout=recent_rollout)
            elif _bapr_like:
                metrics = agent.multi_update(
                    stacked, current_iter=iteration,
                    recent_rewards=ep_rewards if ep_rewards else None,
                    recent_rollout=recent_rollout)
                metrics["recent_replay_frac"] = recent_frac
            elif _escp_like:
                metrics = agent.multi_update(
                    stacked, current_iter=iteration)
            elif _resac_like:
                metrics = agent.multi_update(
                    stacked, current_iter=iteration)
            else:
                metrics = agent.multi_update(stacked)

            for k, v in metrics.items():
                # Log scalars (int/float/bool/np scalar) and arrays (np.ndarray)
                if isinstance(v, (int, float, bool, np.integer, np.floating,
                                  np.ndarray)):
                    logger.log(k, v)
            if hasattr(agent, '_last_ema_rollout_active'):
                logger.log("ema_rollout_active", agent._last_ema_rollout_active)
        train_time = time.time() - train_start
        train_time_total += train_time

        # --- Evaluation (expensive, only every log_interval) ---
        eval_mean = None
        eval_start = time.time()
        if iteration % config.log_interval == 0:
            eval_mean, eval_std = evaluate_stationary(
                agent, eval_env, config, test_tasks,
                n_episodes=config.eval_episodes)
            logger.log("eval_stationary_ood", eval_mean)
            logger.log("eval_stationary_ood_std", eval_std)
            logger.log("eval_reward", eval_mean)
            logger.log("eval_reward_std", eval_std)
            if getattr(config, "eval_protocol", "stationary") == "full":
                id_mean, id_std = evaluate_stationary(
                    agent, eval_env, config, train_tasks,
                    n_episodes=config.eval_episodes)
                logger.log("eval_stationary_id", id_mean)
                logger.log("eval_stationary_id_std", id_std)
                sw_mean, sw_std, sw_count = evaluate_switching_online(
                    agent, eval_env, config, test_tasks,
                    n_episodes=getattr(config, "eval_switching_episodes", 1),
                    period_steps=getattr(
                        config, "eval_switching_period_steps", 500))
                logger.log("eval_switching_online", sw_mean)
                logger.log("eval_switching_online_std", sw_std)
                logger.log("eval_switching_online_switches", sw_count)
                if (config.algo in (
                        "bapr_v3", "bapr_regime", "bapr_v4", "bapr_v5",
                        "bapr_v6", "anchored_regime_sac",
                        "frozen_anchored_regime_sac",
                        "frozen_mode_residual_sac")
                        and bool(config.bapr_v3_eval_context_ladder)):
                    for label, source in (
                            ("robust", agent.CONTEXT_ROBUST),
                            ("oracle", agent.CONTEXT_ORACLE)):
                        ladder_mean, ladder_std = evaluate_stationary(
                            agent, eval_env, config, test_tasks,
                            n_episodes=config.eval_episodes,
                            context_source=source,
                            advantage_enabled=False,
                            record_diagnostics=False)
                        logger.log(
                            f"eval_stationary_{label}", ladder_mean)
                        logger.log(
                            f"eval_stationary_{label}_std", ladder_std)
                        (ladder_sw_mean, ladder_sw_std,
                         ladder_sw_count) = evaluate_switching_online(
                            agent, eval_env, config, test_tasks,
                            n_episodes=getattr(
                                config, "eval_switching_episodes", 1),
                            period_steps=getattr(
                                config, "eval_switching_period_steps", 500),
                            context_source=source,
                            advantage_enabled=False)
                        logger.log(
                            f"eval_switching_{label}", ladder_sw_mean)
                        logger.log(
                            f"eval_switching_{label}_std", ladder_sw_std)
                        logger.log(
                            f"eval_switching_{label}_switches",
                            ladder_sw_count)
            if hasattr(agent, "_last_eval_context_gate"):
                logger.log(
                    "v2_eval_context_gate",
                    float(agent._last_eval_context_gate))
                logger.log(
                    "v2_eval_context_error",
                    float(agent._last_eval_context_error))
                logger.log(
                    "v2_eval_advantage",
                    float(getattr(agent, "_last_eval_advantage", 0.0)))
                logger.log(
                    "v2_eval_advantage_gate",
                    float(getattr(
                        agent, "_last_eval_advantage_gate", 0.0)))
            # Feed eval to BAPR for performance gating
            if hasattr(agent, 'report_eval'):
                agent.report_eval(eval_mean)
            # 2-stage feasibility filter (peak_R based):
            #   stage 1 (early_kill_iter / early_kill_reward): must show signal
            #   stage 2 (stage2_kill_iter / stage2_kill_reward): must show real learning
            #   pass both → train to max_iters
            if not hasattr(config, '_peak_eval_reward'):
                config._peak_eval_reward = -float('inf')
            if eval_mean > config._peak_eval_reward:
                config._peak_eval_reward = float(eval_mean)
            for stage_name, kit, krew in [
                ("STAGE 1",
                 getattr(config, 'early_kill_iter', None),
                 getattr(config, 'early_kill_reward', None)),
                ("STAGE 2",
                 getattr(config, 'stage2_kill_iter', None),
                 getattr(config, 'stage2_kill_reward', None)),
            ]:
                if (kit is not None and krew is not None
                        and iteration >= kit
                        and config._peak_eval_reward < krew):
                    print(f"[EARLY KILL {stage_name}] iter={iteration} "
                          f"peak_eval_reward={config._peak_eval_reward:.1f} "
                          f"(current={eval_mean:.1f}) < threshold {krew} "
                          f"at iter {kit}; exiting with code 99.")
                    logger.save()
                    sys.exit(99)
        eval_time = time.time() - eval_start
        eval_time_total += eval_time

        logger.log("total_steps", total_steps)
        logger.log("iteration", iteration)
        logger.log("mode_id", env.current_task_id)
        logger.log("oracle_switch", oracle_switched)
        logger.log("iter_time", time.time() - iter_start)
        logger.log("collect_time", collect_time)
        logger.log("train_time", train_time)

        # --- Always print status ---
        iter_time = time.time() - iter_start
        q_std_str = ""
        if "q_std_mean" in (metrics if replay_buffer.size >= config.start_train_steps else {}):
            q_std_str = f" | Q-std: {metrics.get('q_std_mean', 0):.2f}"
        eval_str = f" | Eval: {eval_mean:.1f}" if eval_mean is not None else ""
        extra = f"TaskID: {env.current_task_id}{q_std_str}{eval_str}"
        if hasattr(agent, '_current_weighted_lambda'):
            extra += f" | λ_w: {agent._current_weighted_lambda:.3f}"
        if hasattr(agent, 'context_net'):
            warmup = iteration < config.context_warmup_iters
            extra += f" | {'[WARMUP]' if warmup else '[ACTIVE]'}"
        if getattr(config, "oracle_reset_on_switch", False) and oracle_switched:
            extra += " | oracle-reset"
        extra += f" | {iter_time:.1f}s/iter"
        logger.print_status(iteration, extra)

        if getattr(config, "oracle_reset_on_switch", False) and oracle_switched:
            if hasattr(agent, "reset_episode"):
                agent.reset_episode()

        # --- Save ---
        if iteration % config.save_interval == 0:
            logger.save()
            # Save checkpoint for resume
            save_checkpoint(ckpt_dir, agent, replay_buffer, logger,
                            iteration, total_steps, config.algo)

    # Final save
    logger.save()
    save_checkpoint(ckpt_dir, agent, replay_buffer, logger,
                    config.max_iters - 1, total_steps, config.algo)
    elapsed = time.time() - start_time
    print(f"\nTraining complete! Total time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"  Collect time: {collect_time_total:.0f}s ({collect_time_total/elapsed*100:.1f}%)")
    print(f"  Train time:   {train_time_total:.0f}s ({train_time_total/elapsed*100:.1f}%)")
    print(f"  Eval time:    {eval_time_total:.0f}s ({eval_time_total/elapsed*100:.1f}%)")
    print(f"Results saved to: {log_dir}")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="JAX RL Training")
    parser.add_argument("--algo", type=str, default="resac",
                        choices=["resac", "escp", "bapr", "bapr_v2",
                                 "bapr_v3", "bapr_regime", "bapr_v4",
                                 "bapr_v5", "bapr_v6",
                                 "anchored_regime_sac",
                                 "frozen_anchored_regime_sac",
                                 "frozen_mode_residual_sac",
                                 "sac", "regime_sac",
                                 "bapr_no_bocd", "bapr_no_rmdm",
                                 "bapr_no_adapt_beta", "bapr_fixed_decay",
                                 "bad_bapr", "bapr_unsupervised"])
    parser.add_argument("--env", type=str, default="Hopper-v2")
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--max_iters", type=int, default=2000)
    parser.add_argument("--varying_params", nargs="+", default=["gravity"])
    parser.add_argument("--task_num", type=int, default=40)
    parser.add_argument("--test_task_num", type=int, default=40)
    parser.add_argument("--reserved_test_task_num", type=int, default=0)
    parser.add_argument("--task_seed_salt", type=int, default=0)
    parser.add_argument("--save_root", type=str, default="jax_experiments/results")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--ep_dim", type=int, default=2)
    parser.add_argument("--ensemble_size", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override the actor/critic/temperature learning rate.")
    parser.add_argument(
        "--clip_norm", type=float, default=None,
        help="Override gradient clipping norm; non-positive disables clipping.")
    parser.add_argument("--samples_per_iter", type=int, default=None)
    parser.add_argument("--updates_per_iter", type=int, default=None)
    parser.add_argument("--start_train_steps", type=int, default=None)
    parser.add_argument(
        "--initial_random_steps", type=int, default=None,
        help="Collect and checkpoint this many random steps before iteration 0.")
    parser.add_argument("--context_warmup_iters", type=int, default=None)
    parser.add_argument("--rbf_radius", type=float, default=None,
                        help="Override RMDM RBF bandwidth for context loss.")
    parser.add_argument("--consistency_loss_weight", type=float, default=None,
                        help="Override RMDM within-task consistency weight.")
    parser.add_argument("--diversity_loss_weight", type=float, default=None,
                        help="Override RMDM cross-task diversity weight.")
    parser.add_argument("--rmdm_max_tasks", type=int, default=None,
                        help="Static cap for unique task ids in RMDM loss.")
    parser.add_argument("--escp_target_mode", type=str, default=None,
                        choices=["independent", "twin_min"])
    parser.add_argument("--escp_actor_mode", type=str, default=None,
                        choices=["lcb", "twin_min"])
    parser.add_argument("--escp_context_min_steps", type=int, default=None)
    parser.add_argument("--escp_context_min_tasks", type=int, default=None)
    parser.add_argument("--escp_alpha_max", type=float, default=None)
    parser.add_argument("--escp_context_mode", type=str, default=None,
                        choices=["state_mlp", "recurrent"])
    parser.add_argument("--escp_history_length", type=int, default=None)
    parser.add_argument("--escp_policy_lr", type=float, default=None)
    parser.add_argument("--escp_critic_lr", type=float, default=None)
    parser.add_argument("--escp_context_lr", type=float, default=None)
    parser.add_argument("--escp_alpha_lr", type=float, default=None)
    parser.add_argument("--escp_target_entropy_ratio", type=float,
                        default=None)
    parser.add_argument("--escp_bottleneck_sigma", type=float, default=None)
    parser.add_argument("--escp_prototype_tau", type=float, default=None)
    parser.add_argument("--no_escp_finite_guard", action="store_true")
    parser.add_argument("--bapr_v2_mode", type=str, default=None,
                        choices=["robust", "oracle", "supervised", "hybrid"])
    parser.add_argument("--bapr_v2_latent_dim", type=int, default=None)
    parser.add_argument("--bapr_v2_latent_scale_mode", type=str, default=None,
                        choices=["legacy_exp", "task_distribution"])
    parser.add_argument("--bapr_v2_policy_context_source", type=str,
                        default=None, choices=["stored", "oracle_task"])
    parser.add_argument("--bapr_v2_training_schedule", type=str,
                        default=None, choices=[
                            "joint", "teacher_student",
                            "constrained_deploy"])
    parser.add_argument("--bapr_v2_base_pretrain_iters", type=int,
                        default=None)
    parser.add_argument("--bapr_v2_teacher_iters", type=int, default=None)
    parser.add_argument("--bapr_v2_student_iters", type=int, default=None)
    parser.add_argument(
        "--bapr_v2_warmstart_conditioned", action="store_true")
    parser.add_argument("--bapr_v2_context_hidden_dim", type=int, default=None)
    parser.add_argument("--bapr_v2_context_length", type=int, default=None)
    parser.add_argument("--bapr_v2_context_chunks", type=int, default=None)
    parser.add_argument("--bapr_v2_context_burnin", type=int, default=None)
    parser.add_argument("--bapr_v2_context_lr", type=float, default=None)
    parser.add_argument("--bapr_v2_predictive_weight", type=float, default=None)
    parser.add_argument("--bapr_v2_supervised_weight", type=float, default=None)
    parser.add_argument("--bapr_v2_hybrid_supervised_weight", type=float,
                        default=None)
    parser.add_argument("--bapr_v2_temporal_weight", type=float, default=None)
    parser.add_argument("--bapr_v2_reward_scale", type=float, default=None)
    parser.add_argument("--bapr_v2_delta_scale", type=float, default=None)
    parser.add_argument("--bapr_v2_min_history", type=int, default=None)
    parser.add_argument("--bapr_v2_gate_error_threshold", type=float,
                        default=None)
    parser.add_argument("--bapr_v2_gate_error_scale", type=float, default=None)
    parser.add_argument("--bapr_v2_error_ema_alpha", type=float, default=None)
    parser.add_argument("--bapr_v2_reset_temperature", type=float, default=None)
    parser.add_argument("--no_bapr_v2_fallback", action="store_true")
    parser.add_argument("--bapr_v2_policy_mode", type=str, default=None,
                        choices=["residual", "direct", "gated_direct",
                                 "expert", "categorical_expert"])
    parser.add_argument("--bapr_v2_num_experts", type=int, default=None)
    parser.add_argument("--bapr_v2_residual_delta", type=float, default=None)
    parser.add_argument("--bapr_v2_policy_gate_init", type=float, default=None)
    parser.add_argument("--bapr_v2_action_deviation_weight", type=float,
                        default=None)
    parser.add_argument("--bapr_v2_switch_rollout_steps", type=int,
                        default=None)
    parser.add_argument(
        "--bapr_v2_freeze_gate_in_teacher", action="store_true")
    parser.add_argument("--bapr_v2_paired_calibration_episodes", type=int,
                        default=None)
    parser.add_argument("--bapr_v2_paired_gain_margin", type=float,
                        default=None)
    parser.add_argument("--bapr_v2_paired_gain_temperature", type=float,
                        default=None)
    parser.add_argument("--bapr_v2_paired_risk_tolerance", type=float,
                        default=None)
    parser.add_argument("--bapr_v2_paired_risk_temperature", type=float,
                        default=None)
    parser.add_argument("--bapr_v2_paired_return_scale", type=float,
                        default=None)
    parser.add_argument("--bapr_v2_gate_supervision_weight", type=float,
                        default=None)
    parser.add_argument("--bapr_v2_unsafe_deviation_weight", type=float,
                        default=None)
    parser.add_argument("--bapr_v2_context_dropout", type=float, default=None)
    parser.add_argument("--bapr_v2_base_aux_weight", type=float, default=None)
    parser.add_argument("--bapr_v2_advantage_gate", action="store_true")
    parser.add_argument("--bapr_v2_advantage_margin", type=float, default=None)
    parser.add_argument("--bapr_v2_advantage_lcb_scale", type=float,
                        default=None)
    parser.add_argument(
        "--bapr_v2_train_advantage_constraint", action="store_true")
    parser.add_argument("--bapr_v2_train_advantage_lcb_scale", type=float,
                        default=None)
    parser.add_argument("--bapr_v2_train_advantage_margin", type=float,
                        default=None)
    parser.add_argument("--bapr_v2_train_advantage_temperature", type=float,
                        default=None)
    parser.add_argument("--bapr_v2_train_advantage_weight", type=float,
                        default=None)
    parser.add_argument("--bapr_v2_train_update_filter", action="store_true")
    parser.add_argument("--bapr_v2_train_update_tolerance", type=float,
                        default=None)
    parser.add_argument("--bapr_v2_train_update_floor", type=float,
                        default=None)
    parser.add_argument("--bapr_v2_actor_objective", type=str, default=None,
                        choices=["mean", "lcb"])
    parser.add_argument(
        "--bapr_v2_critic_target_mode", type=str, default=None,
        choices=["independent", "min"])
    parser.add_argument("--bapr_v2_freeze_alpha", action="store_true")
    parser.add_argument("--bapr_v2_beta_ood", type=float, default=None)
    parser.add_argument("--bapr_v2_reg_weight", type=float, default=None)
    parser.add_argument("--bapr_v2_reg_norm_ref", type=float, default=None)
    parser.add_argument("--bapr_v3_likelihood", type=str, default=None,
                        choices=["point", "probabilistic"])
    parser.add_argument("--bapr_v3_context_ensemble_size", type=int,
                        default=None)
    parser.add_argument("--bapr_v3_hazard_rate", type=float, default=None)
    parser.add_argument("--bapr_v3_evidence_scale", type=float, default=None)
    parser.add_argument("--bapr_v3_fixed_variance", type=float, default=None)
    parser.add_argument("--bapr_v3_logvar_min", type=float, default=None)
    parser.add_argument("--bapr_v3_logvar_max", type=float, default=None)
    parser.add_argument("--bapr_v3_variance_model", type=str, default=None,
                        choices=["legacy_state", "mode_calibrated",
                                 "mode_empirical",
                                 "mode_shared_empirical",
                                 "inverse_empirical"])
    parser.add_argument("--bapr_v3_variance_floor", type=float, default=None)
    parser.add_argument("--bapr_v3_variance_ceiling", type=float,
                        default=None)
    parser.add_argument("--bapr_v3_variance_ema", type=float, default=None)
    parser.add_argument("--bapr_v3_mean_loss_weight", type=float,
                        default=None)
    parser.add_argument("--bapr_v3_variance_loss_weight", type=float,
                        default=None)
    parser.add_argument("--bapr_v3_variance_prior_weight", type=float,
                        default=None)
    parser.add_argument("--bapr_v3_instant_classifier_weight", type=float,
                        default=None)
    parser.add_argument("--bapr_v3_evidence_clip", type=float, default=None)
    parser.add_argument("--bapr_v3_surprise_threshold", type=float,
                        default=None)
    parser.add_argument("--bapr_v3_surprise_scale", type=float, default=None)
    parser.add_argument(
        "--bapr_v3_freeze_teacher_after_teacher", action="store_true",
        help="Freeze the oracle-trained residual policy and critic in the "
             "deployment stage while the causal context keeps learning.")
    parser.add_argument(
        "--bapr_v3_estimator_rollout_source", type=str, default=None,
        choices=["learned", "robust"],
        help="Policy context used to collect post-teacher estimator data.")
    parser.add_argument(
        "--bapr_v3_eval_context_ladder", action="store_true",
        help="Log same-checkpoint robust/oracle/learned V3 evaluations.")
    parser.add_argument(
        "--bapr_v3_reset_context_on_resume", action="store_true",
        help="Keep the resumed policy/critic/replay state but reinitialize "
             "the BAPR-v3 context model and its optimizer.")
    parser.add_argument(
        "--bapr_v4_training_source_period", type=int, default=None,
        help="Iterations per robust/oracle rollout-source cycle.")
    parser.add_argument(
        "--bapr_v4_training_robust_slots", type=int, default=None,
        help="Robust rollout iterations at the start of each source cycle.")
    parser.add_argument("--bapr_regime_inference_iters", type=int,
                        default=None)
    parser.add_argument(
        "--bapr_regime_adaptation_source", type=str, default=None,
        choices=["learned", "oracle"])
    parser.add_argument(
        "--bapr_regime_update_context_during_adaptation",
        action="store_true",
        help="Continue context updates after posterior-conditioned residual "
             "training starts instead of freezing the calibrated estimator.")
    parser.add_argument(
        "--no_bapr_regime_clear_replay", action="store_true",
        help="Keep pre-adaptation replay entries. The default clears them to "
             "avoid stale zero-context transitions.")
    parser.add_argument(
        "--no_bapr_regime_zero_residual_init", action="store_true",
        help="Do not zero the residual output at the adaptation boundary.")
    parser.add_argument(
        "--no_bapr_regime_advantage_fallback", action="store_true",
        help="Disable conservative Q-advantage fallback to the robust actor.")
    parser.add_argument("--beta", type=float, default=None,
                        help="Override LCB coefficient beta")
    parser.add_argument("--actor_objective", type=str, default=None,
                        choices=["lcb", "gated_lcb", "reg_gated_lcb",
                                 "qstd_gated_lcb", "mean", "ucb",
                                 "conservative_residual",
                                 "residual_advantage", "v80_residual"],
                        help="BAPR actor objective: lcb (legacy), gated_lcb "
                             "(surprise-gated conservative risk), mean "
                             "(SAC-like ensemble mean), reg_gated_lcb "
                             "(LCB only while RE-SAC regularizer is active), "
                             "qstd_gated_lcb (LCB only under actor q_std gate), "
                             "ucb (optimistic), or conservative_residual "
                             "(v80 bounded adaptive residual from EMA base).")
    parser.add_argument("--weight_reg", type=float, default=None,
                        help="Override RE-SAC/BAPR critic weight regularizer.")
    parser.add_argument("--beta_ood", type=float, default=None,
                        help="Override RE-SAC/BAPR OOD critic regularizer.")
    parser.add_argument("--beta_bc", type=float, default=None,
                        help="Override RE-SAC behavior-cloning regularizer.")
    parser.add_argument(
        "--critic_actor_ratio", type=int, default=None,
        help="RE-SAC critic updates per actor update.")
    parser.add_argument("--resac_independent_ratio", type=float, default=None,
                        help="Blend independent and shared-min critic targets.")
    parser.add_argument("--resac_anchor_lambda", type=float, default=None)
    parser.add_argument("--resac_adaptive_beta", action="store_true")
    parser.add_argument("--resac_beta_start", type=float, default=None)
    parser.add_argument("--resac_beta_end", type=float, default=None)
    parser.add_argument("--resac_beta_warmup", type=float, default=None)
    parser.add_argument("--resac_critic_actor_ratio", type=int, default=None)
    parser.add_argument("--resac_beta_bc", type=float, default=None)
    parser.add_argument("--resac_clip_norm", type=float, default=None)
    parser.add_argument("--ema_tau", type=float, default=None,
                        help="EMA policy update coefficient.")
    parser.add_argument("--use_ema_eval", action="store_true",
                        help="Evaluate BAPR with EMA policy instead of online policy.")
    parser.add_argument("--use_ema_rollout", action="store_true",
                        help="Collect training rollouts with EMA policy instead of online policy.")
    parser.add_argument("--ema_rollout_start_iter", type=int, default=None,
                        help="Start EMA-policy rollout at/after this training iteration.")
    parser.add_argument("--ema_rollout_require_reg_latched", action="store_true",
                        help="Use EMA rollout only after BAPR's reg latch is active.")
    parser.add_argument("--bapr_adaptation_mode", type=str, default=None,
                        choices=["gate", "legacy"],
                        help="BAPR redesign mode. gate: bounded surprise gate "
                             "for recent replay/optional gated_lcb, no default "
                             "belief-conditioned Q. legacy: old BOCD adaptive "
                             "beta path.")
    parser.add_argument("--bapr_surprise_threshold", type=float, default=None,
                        help="BAPR gate: surprise threshold before adaptation.")
    parser.add_argument("--bapr_gate_warmup_iters", type=int, default=None,
                        help="BAPR gate: iterations before recent replay gate can open.")
    parser.add_argument("--bapr_gate_gain", type=float, default=None,
                        help="BAPR gate: multiplier on surprise above threshold.")
    parser.add_argument("--bapr_gate_ema_alpha", type=float, default=None,
                        help="BAPR gate: EMA smoothing coefficient.")
    parser.add_argument("--bapr_gate_max", type=float, default=None,
                        help="BAPR gate: maximum adaptation gate value.")
    parser.add_argument("--bapr_recent_frac_cap", type=float, default=None,
                        help="BAPR gate: maximum fraction of recent replay.")
    parser.add_argument("--bapr_recent_frac_floor", type=float, default=None,
                        help="BAPR gate: minimum recent replay fraction while the adaptation gate is active.")
    parser.add_argument("--bapr_recent_true_floor", action="store_true",
                        help="Apply bapr_recent_frac_floor even when the surprise gate is zero.")
    parser.add_argument("--bapr_recent_floor_mode", type=str, default=None,
                        choices=("always", "low_disagreement", "reg_latched", "ratio_schedule"),
                        help="How to apply bapr_recent_frac_floor under the disagreement gate.")
    parser.add_argument("--bapr_recent_floor_ratio_low", type=float, default=None,
                        help="ratio_schedule: q_std / |q_mean| below this uses bapr_recent_frac_floor.")
    parser.add_argument("--bapr_recent_floor_ratio_high", type=float, default=None,
                        help="ratio_schedule: q_std / |q_mean| above this uses the extreme floor.")
    parser.add_argument("--bapr_recent_floor_mid_frac", type=float, default=None,
                        help="ratio_schedule: recent replay floor for moderate critic disagreement.")
    parser.add_argument("--bapr_recent_floor_extreme_frac", type=float, default=None,
                        help="ratio_schedule: recent replay floor for extreme critic disagreement.")
    parser.add_argument("--bapr_recent_disagreement_gate", action="store_true",
                        help="Enable critic-disagreement safety gate for recent replay.")
    parser.add_argument("--bapr_recent_open_if_reg_latched", action="store_true",
                        help="Bypass the recent replay disagreement gate after the RE-SAC regularizer latch fires.")
    parser.add_argument("--bapr_recent_qstd_threshold", type=float, default=None,
                        help="Allow recent replay only below this q_std.")
    parser.add_argument("--bapr_recent_qstd_ratio_threshold", type=float, default=None,
                        help="Allow recent replay only below q_std / |q_mean|.")
    parser.add_argument("--bapr_reg_disagreement_gate", action="store_true",
                        help="Enable critic-disagreement safety gate for RE-SAC regularizers.")
    parser.add_argument("--bapr_reg_warmup_iters", type=int, default=None,
                        help="Keep RE-SAC regularizers disabled for this many iterations.")
    parser.add_argument("--bapr_reg_max_iters", type=int, default=None,
                        help="Disable RE-SAC regularizers at/after this iteration; <=0 means no cap.")
    parser.add_argument("--bapr_reg_latch", action="store_true",
                        help="Once high disagreement is detected before reg_max_iters, keep RE-SAC regularizers enabled.")
    parser.add_argument("--bapr_reg_require_both", action="store_true",
                        help="Enable RE-SAC regularizers only when both q_std and q_std/|q_mean| are high.")
    parser.add_argument("--bapr_reg_qstd_threshold", type=float, default=None,
                        help="Enable RE-SAC regularizers above this q_std.")
    parser.add_argument("--bapr_reg_qstd_ratio_threshold", type=float, default=None,
                        help="Enable RE-SAC regularizers above q_std / |q_mean|.")
    parser.add_argument("--bapr_reg_emergency_gate", action="store_true",
                        help="Re-enable RE-SAC regularizers after reg_max_iters if critic disagreement explodes.")
    parser.add_argument("--bapr_reg_emergency_scale", type=float, default=None,
                        help="Scale for emergency RE-SAC regularizers after reg_max_iters.")
    parser.add_argument("--bapr_reg_emergency_qstd_threshold", type=float, default=None,
                        help="Emergency RE-SAC regularizer threshold on q_std.")
    parser.add_argument("--bapr_reg_emergency_qstd_ratio_threshold", type=float, default=None,
                        help="Emergency RE-SAC regularizer threshold on q_std / |q_mean|.")
    parser.add_argument("--bapr_reg_latched_scale", type=float, default=None,
                        help="Scale RE-SAC regularizers after the permanent disagreement latch fires.")
    parser.add_argument("--bapr_reg_perf_collapse_gate", action="store_true",
                        help="Temporarily reduce latched RE-SAC regularization after severe reward drawdown.")
    parser.add_argument("--bapr_reg_perf_collapse_drop_frac", type=float, default=None,
                        help="Reward drawdown fraction required to reduce latched RE-SAC regularization.")
    parser.add_argument("--bapr_reg_perf_collapse_scale", type=float, default=None,
                        help="Multiplier for RE-SAC regularization while the collapse guard is active.")
    parser.add_argument("--bapr_reg_perf_collapse_warmup_iters", type=int, default=None,
                        help="Warmup before the RE-SAC regularization collapse guard can activate.")
    parser.add_argument("--bapr_controller_mode", type=str, default=None,
                        choices=["off", "recovery", "soft", "soft_recovery"],
                        help="Second-layer BAPR algorithm controller mode.")
    parser.add_argument("--bapr_controller_warmup_iters", type=int, default=None,
                        help="Warmup before the algorithm controller can activate.")
    parser.add_argument("--bapr_controller_min_peak", type=float, default=None,
                        help="Minimum smoothed reward peak before controller activation.")
    parser.add_argument("--bapr_controller_drop_frac", type=float, default=None,
                        help="Reward drawdown fraction required by the algorithm controller.")
    parser.add_argument("--bapr_controller_drop_ramp", type=float, default=None,
                        help="Drawdown ramp width for continuous controller signal.")
    parser.add_argument("--bapr_controller_soft_decay", type=float, default=None,
                        help="Soft controller signal decay per iteration.")
    parser.add_argument("--bapr_controller_soft_min_signal", type=float, default=None,
                        help="Minimum soft controller signal before multipliers are applied.")
    parser.add_argument("--bapr_controller_low_qstd_ratio", type=float, default=None,
                        help="Controller low-disagreement threshold on q_std / |q_mean|.")
    parser.add_argument("--bapr_controller_low_qstd_threshold", type=float, default=None,
                        help="Optional absolute q_std threshold for low-disagreement controller activation.")
    parser.add_argument("--bapr_controller_latch", action="store_true",
                        help="Keep the algorithm controller active until reward drawdown recovers.")
    parser.add_argument("--bapr_controller_release_drop_frac", type=float, default=None,
                        help="Release a latched algorithm controller below this reward drawdown.")
    parser.add_argument("--bapr_controller_max_active_iters", type=int, default=None,
                        help="Force-release the recovery controller after this many active iterations; 0 disables.")
    parser.add_argument("--bapr_controller_min_active_iters", type=int, default=None,
                        help="Minimum active iterations before signal/improvement/drawdown exits can fire.")
    parser.add_argument("--bapr_controller_exit_cooldown_iters", type=int, default=None,
                        help="After a controller exit, suppress re-entry for this many iterations.")
    parser.add_argument("--bapr_controller_exit_signal_threshold", type=float, default=None,
                        help="Exit after min-active iters once soft controller signal drops below this threshold.")
    parser.add_argument("--bapr_controller_exit_improve_frac", type=float, default=None,
                        help="Exit after min-active iters once smoothed reward improves by this peak-normalized fraction.")
    parser.add_argument("--bapr_controller_exit_drawdown_frac", type=float, default=None,
                        help="Exit after min-active iters once drawdown is at or below this fraction; negative disables.")
    parser.add_argument("--bapr_controller_latched_signal", type=float, default=None,
                        help="Minimum controller signal while the controller is latched.")
    parser.add_argument("--bapr_controller_reg_multiplier", type=float, default=None,
                        help="Target RE-SAC regularizer multiplier at full controller activation.")
    parser.add_argument("--bapr_controller_reg_recover_iters", type=int, default=None,
                        help="Linearly restore controller regularization multiplier to 1 over this many active iterations.")
    parser.add_argument("--bapr_controller_recent_multiplier", type=float, default=None,
                        help="Target recent replay multiplier at full controller activation.")
    parser.add_argument("--bapr_controller_recent_add_frac", type=float, default=None,
                        help="Extra recent replay fraction at full controller activation.")
    parser.add_argument("--bapr_controller_lcb_multiplier", type=float, default=None,
                        help="Target actor LCB multiplier at full controller activation.")
    parser.add_argument("--bapr_controller_actor_update_multiplier", type=float, default=None,
                        help="Target actor update multiplier at full controller activation.")
    parser.add_argument("--bapr_controller_actor_recover_iters", type=int, default=None,
                        help="Linearly restore actor update multiplier to 1 over this many active iterations.")
    parser.add_argument("--bapr_actor_lcb_qstd_gate", action="store_true",
                        help="Enable actor-only LCB when q_std/q_ratio crosses thresholds.")
    parser.add_argument("--bapr_actor_lcb_qstd_threshold", type=float, default=None,
                        help="Actor-only LCB gate q_std threshold.")
    parser.add_argument("--bapr_actor_lcb_qstd_ratio_threshold", type=float, default=None,
                        help="Actor-only LCB gate q_std / |q_mean| threshold.")
    parser.add_argument("--bapr_actor_lcb_require_both", action="store_true",
                        help="Actor-only LCB gate requires both q_std and ratio thresholds.")
    parser.add_argument("--bapr_actor_lcb_scale", type=float, default=None,
                        help="Actor-only LCB scale when the gate is active.")
    parser.add_argument("--bapr_actor_lcb_perf_gate", action="store_true",
                        help="Require a train/eval reward drawdown before actor-only LCB can activate.")
    parser.add_argument("--bapr_actor_lcb_perf_warmup_iters", type=int, default=None,
                        help="Actor LCB performance gate warmup before drawdown checks are allowed.")
    parser.add_argument("--bapr_actor_lcb_perf_drop_frac", type=float, default=None,
                        help="Fractional reward drawdown required before actor-only LCB can activate.")
    parser.add_argument("--bapr_actor_lcb_perf_min_peak", type=float, default=None,
                        help="Minimum smoothed reward peak before actor LCB performance gate can activate.")
    parser.add_argument("--bapr_actor_lcb_perf_ema_alpha", type=float, default=None,
                        help="EMA alpha for train/eval reward drawdown tracked by actor LCB gate.")
    parser.add_argument("--bapr_residual_delta", type=float, default=None,
                        help="v80 max absolute residual action from the zero-context EMA base.")
    parser.add_argument("--bapr_residual_gate_scale", type=float, default=None,
                        help="v80 multiplier for the conservative residual gate.")
    parser.add_argument("--bapr_residual_adv_margin", type=float, default=None,
                        help="v80 conservative advantage margin before residual gate opens.")
    parser.add_argument("--bapr_residual_adv_temp", type=float, default=None,
                        help="v80 sigmoid temperature for residual conservative advantage.")
    parser.add_argument("--bapr_residual_qstd_scale", type=float, default=None,
                        help="v80 positive Q-std penalty in conservative advantage.")
    parser.add_argument("--bapr_residual_behavior_weight", type=float, default=None,
                        help="v80 behavior regularization toward base action when gate is low.")
    parser.add_argument("--bapr_residual_action_penalty", type=float, default=None,
                        help="v80 penalty on the gated residual action magnitude.")
    parser.add_argument("--backend", type=str, default="spring",
                        choices=["spring", "generalized"],
                        help="Brax physics backend: spring (fast) or generalized (accurate)")
    parser.add_argument("--log_scale_limit", type=float, default=None,
                        help="Override log_scale_limit for gravity sampling range")
    parser.add_argument("--task_scale_distribution", type=str, default=None,
                        choices=["exp", "pow1p5"],
                        help="Continuous task scale distribution: exp uses "
                             "e**u; pow1p5 uses ESCP-style 1.5**u.")
    parser.add_argument("--penalty_scale", type=float, default=None,
                        help="Override BAPR penalty_scale")
    parser.add_argument("--bapr_grad_clip_norm", type=float, default=None,
                        help="Override BAPR global gradient clipping norm; "
                             "use 0 to disable")
    parser.add_argument("--hazard_rate", type=float, default=None,
                        help="Override BOCD hazard_rate")
    parser.add_argument("--regime_hazard_rate", type=float, default=None,
                        help="Override joint-regime BOCD hazard_rate")
    parser.add_argument("--max_run_length", type=int, default=None,
                        help="Override BOCD max run length H")
    parser.add_argument("--num_regimes", type=int, default=None,
                        help="Override joint belief regime-cluster count K")
    parser.add_argument("--surprise_weights", nargs=3, type=float, default=None,
                        metavar=("W_REWARD", "W_Q", "W_REG"),
                        help="Override normalized surprise weights")
    parser.add_argument("--oracle_reset_on_switch", action="store_true",
                        help="Oracle upper-bound ablation: reset BOCD at true switches")
    parser.add_argument("--changing_period", type=int, default=None,
                        help="Override task changing_period (env steps per task)")
    parser.add_argument("--changing_interval", type=int, default=None,
                        help="Override how often the task schedule is checked")
    parser.add_argument("--log_interval", type=int, default=None,
                        help="Eval every N iterations (default: 5)")
    parser.add_argument("--eval_episodes", type=int, default=None,
                        help="Number of eval episodes (default: 5)")
    parser.add_argument("--max_episode_steps", type=int, default=None,
                        help="Strict train/eval episode horizon.")
    parser.add_argument("--eval_protocol", type=str, default=None,
                        choices=["stationary", "full"],
                        help="stationary: strict fixed-task eval only; full: "
                             "also log stationary ID and switching-online eval.")
    parser.add_argument("--eval_switching_episodes", type=int, default=None,
                        help="Number of switching-online eval episodes.")
    parser.add_argument("--eval_switching_period_steps", type=int, default=None,
                        help="Force an eval environment switch every N steps "
                             "during switching-online eval.")
    parser.add_argument("--save_interval", type=int, default=None,
                        help="Save checkpoint every N iterations (default: 50)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint if available")
    parser.add_argument(
        "--min_resume_iteration", type=int, default=None,
        help="Abort before training unless resume starts at or after this iter.")
    parser.add_argument(
        "--resume_boundary_audit", action="store_true",
        help="At the first resumed iteration, assert an oracle warm-start "
             "action canary and write a pre-gradient physical-rollout hash.")
    parser.add_argument(
        "--resume_boundary_expected_iteration", type=int, default=None,
        help="With --resume_boundary_audit, require this exact resume iter.")
    parser.add_argument(
        "--resume_boundary_expected_total_steps", type=int, default=None,
        help="With --resume_boundary_audit, require this exact step count.")
    parser.add_argument(
        "--resume_boundary_expected_update_count", type=int, default=None,
        help="With --resume_boundary_audit, require this exact update count.")
    parser.add_argument("--env_type", type=str, default=None,
                        choices=["continuous", "discrete_mode",
                                 "stochastic_mode"],
                        help="continuous, discrete_mode, or persistent "
                             "stochastic_mode transition kernels")
    parser.add_argument("--mean_dwell_iters", type=int, default=None,
                        help="discrete_mode: avg iters per mode dwell (default 60)")
    parser.add_argument("--use_regime_belief", action="store_true",
                        help="BAPR Minimum redesign: use joint regime belief "
                             "b(h, z) — Q sees [s, a, e, ρ, μ] (24-dim belief).")
    parser.add_argument("--no_belief_conditioning", action="store_true",
                        help="BAPR ablation: disable belief vector inputs to "
                             "actor/critic unless --use_regime_belief is set.")
    parser.add_argument("--critic_target_mode", type=str, default=None,
                        choices=["min", "independent"],
                        help='BAPR critic target operator: "min" (legacy LCB, '
                             'collapses ensemble) or "independent" (each Q_i '
                             'learns own target, ESCP-like). GPT-5.5 advice #4.')
    parser.add_argument("--no_per_trans_belief", action="store_true",
                        help='Ablation toggle for GPT-5.5 advice #2: when set, '
                             'critic broadcasts current belief instead of using '
                             'per-transition stored belief.')
    parser.add_argument("--mode_variant", type=str, default=None,
                        help="discrete_mode mode set selector. 'orig' (default) "
                             "uses MODE_DEFINITIONS; e.g. 'g_mild', 'd_mild', "
                             "'m_mild', 'mixed_mild' use milder MODE_VARIANTS "
                             "presets. See discrete_mode_env.MODE_VARIANTS.")
    parser.add_argument("--stochastic_mode_family", type=str, default=None,
                        choices=["deterministic_mean", "variance_only",
                                 "mean_variance", "packet_loss",
                                 "burst_torque", "structured_channel",
                                 "actuator_polarity"])
    parser.add_argument("--stochastic_mode_dwell_steps", type=int,
                        default=None)
    parser.add_argument("--stochastic_mode_dwell_distribution", type=str,
                        default=None, choices=["fixed", "exponential"])
    parser.add_argument("--regime_context_source", type=str, default=None,
                        choices=["robust", "oracle"],
                        help="Condition regime_sac on zero or true mode.")
    parser.add_argument(
        "--stochastic_mode_fixed_id", type=int, default=None,
        choices=range(4),
        help="Lock stochastic-mode training to one persistent mode. Strict "
             "evaluation can explicitly re-enable switching.")
    parser.add_argument("--early_kill_iter", type=int, default=None,
                        help="Sweep feasibility filter: if eval_reward stays "
                             "below --early_kill_reward through this iter, "
                             "exit with code 99 (config too hard).")
    parser.add_argument("--early_kill_reward", type=float, default=None,
                        help="Reward threshold paired with --early_kill_iter.")
    parser.add_argument("--stage2_kill_iter", type=int, default=None,
                        help="Stage 2 kill checkpoint (e.g. 1000). At this iter "
                             "or later, kill if peak_R < --stage2_kill_reward.")
    parser.add_argument("--stage2_kill_reward", type=float, default=None,
                        help="Stage 2 reward threshold (e.g. 2000). Pairs with "
                             "--stage2_kill_iter; intent is 'must demonstrate "
                             "real learning by this iter'.")

    args = parser.parse_args()

    if args.resume_boundary_audit:
        missing_boundary_guards = [
            name for name, value in (
                ("--resume_boundary_expected_iteration",
                 args.resume_boundary_expected_iteration),
                ("--resume_boundary_expected_total_steps",
                 args.resume_boundary_expected_total_steps),
                ("--resume_boundary_expected_update_count",
                 args.resume_boundary_expected_update_count),
            ) if value is None
        ]
        if missing_boundary_guards:
            parser.error(
                "--resume_boundary_audit requires exact guards: "
                + ", ".join(missing_boundary_guards))

    config = Config()
    config.algo = args.algo
    config.env_name = args.env
    config.seed = args.seed
    config.max_iters = args.max_iters
    config.varying_params = args.varying_params
    config.task_num = args.task_num
    config.test_task_num = args.test_task_num
    config.reserved_test_task_num = args.reserved_test_task_num
    config.task_seed_salt = args.task_seed_salt
    config.save_root = args.save_root
    config.run_name = args.run_name
    config.ep_dim = args.ep_dim
    config.ensemble_size = args.ensemble_size
    config.hidden_dim = args.hidden_dim
    if args.lr is not None:
        if not np.isfinite(args.lr) or args.lr <= 0.0:
            parser.error("--lr must be finite and positive")
        config.lr = args.lr
    if args.clip_norm is not None:
        if not np.isfinite(args.clip_norm):
            parser.error("--clip_norm must be finite")
        config.clip_norm = args.clip_norm
    # Only override Config defaults when explicitly provided
    if args.samples_per_iter is not None:
        config.samples_per_iter = args.samples_per_iter
    if args.updates_per_iter is not None:
        config.updates_per_iter = args.updates_per_iter
    if args.start_train_steps is not None:
        config.start_train_steps = args.start_train_steps
    if args.initial_random_steps is not None:
        if args.initial_random_steps < 0:
            parser.error("--initial_random_steps must be non-negative")
        config.initial_random_steps = args.initial_random_steps
    if args.context_warmup_iters is not None:
        config.context_warmup_iters = args.context_warmup_iters
    if args.rbf_radius is not None:
        config.rbf_radius = args.rbf_radius
    if args.consistency_loss_weight is not None:
        config.consistency_loss_weight = args.consistency_loss_weight
    if args.diversity_loss_weight is not None:
        config.diversity_loss_weight = args.diversity_loss_weight
    if args.rmdm_max_tasks is not None:
        config.rmdm_max_tasks = args.rmdm_max_tasks
    for field_name in (
            "escp_target_mode", "escp_actor_mode",
            "escp_context_min_steps", "escp_context_min_tasks",
            "escp_alpha_max", "escp_context_mode",
            "escp_history_length", "escp_policy_lr", "escp_critic_lr",
            "escp_context_lr", "escp_alpha_lr",
            "escp_target_entropy_ratio", "escp_bottleneck_sigma",
            "escp_prototype_tau"):
        value = getattr(args, field_name)
        if value is not None:
            setattr(config, field_name, value)
    if int(config.escp_history_length) <= 0:
        parser.error("--escp_history_length must be positive")
    if float(config.escp_target_entropy_ratio) <= 0.0:
        parser.error("--escp_target_entropy_ratio must be positive")
    if float(config.escp_bottleneck_sigma) < 0.0:
        parser.error("--escp_bottleneck_sigma must be non-negative")
    if not 0.0 <= float(config.escp_prototype_tau) < 1.0:
        parser.error("--escp_prototype_tau must be in [0, 1)")
    if args.no_escp_finite_guard:
        config.escp_finite_guard = False
    for field_name in (
            "bapr_v2_mode", "bapr_v2_latent_dim",
            "bapr_v2_latent_scale_mode",
            "bapr_v2_policy_context_source",
            "bapr_v2_training_schedule", "bapr_v2_base_pretrain_iters",
            "bapr_v2_teacher_iters", "bapr_v2_student_iters",
            "bapr_v2_context_hidden_dim", "bapr_v2_context_length",
            "bapr_v2_context_chunks", "bapr_v2_context_burnin",
            "bapr_v2_context_lr", "bapr_v2_predictive_weight",
            "bapr_v2_supervised_weight",
            "bapr_v2_hybrid_supervised_weight", "bapr_v2_temporal_weight",
            "bapr_v2_reward_scale", "bapr_v2_delta_scale",
            "bapr_v2_min_history", "bapr_v2_gate_error_threshold",
            "bapr_v2_gate_error_scale", "bapr_v2_error_ema_alpha",
            "bapr_v2_reset_temperature", "bapr_v2_policy_mode",
            "bapr_v2_num_experts", "bapr_v2_residual_delta",
            "bapr_v2_policy_gate_init",
            "bapr_v2_action_deviation_weight",
            "bapr_v2_switch_rollout_steps",
            "bapr_v2_paired_calibration_episodes",
            "bapr_v2_paired_gain_margin",
            "bapr_v2_paired_gain_temperature",
            "bapr_v2_paired_risk_tolerance",
            "bapr_v2_paired_risk_temperature",
            "bapr_v2_paired_return_scale",
            "bapr_v2_gate_supervision_weight",
            "bapr_v2_unsafe_deviation_weight",
            "bapr_v2_context_dropout", "bapr_v2_base_aux_weight",
            "bapr_v2_advantage_margin", "bapr_v2_advantage_lcb_scale",
            "bapr_v2_train_advantage_lcb_scale",
            "bapr_v2_train_advantage_margin",
            "bapr_v2_train_advantage_temperature",
            "bapr_v2_train_advantage_weight",
            "bapr_v2_train_update_tolerance",
            "bapr_v2_train_update_floor",
            "bapr_v2_actor_objective", "bapr_v2_critic_target_mode",
            "bapr_v2_beta_ood",
            "bapr_v2_reg_weight", "bapr_v2_reg_norm_ref",
            "bapr_v3_likelihood", "bapr_v3_context_ensemble_size",
            "bapr_v3_hazard_rate", "bapr_v3_evidence_scale",
            "bapr_v3_fixed_variance", "bapr_v3_logvar_min",
            "bapr_v3_logvar_max", "bapr_v3_variance_model",
            "bapr_v3_variance_floor", "bapr_v3_variance_ceiling",
            "bapr_v3_variance_ema",
            "bapr_v3_mean_loss_weight", "bapr_v3_variance_loss_weight",
            "bapr_v3_variance_prior_weight",
            "bapr_v3_instant_classifier_weight",
            "bapr_v3_evidence_clip",
            "bapr_v3_surprise_threshold",
            "bapr_v3_surprise_scale",
            "bapr_v3_estimator_rollout_source",
            "bapr_v4_training_source_period",
            "bapr_v4_training_robust_slots",
            "bapr_regime_inference_iters",
            "bapr_regime_adaptation_source"):
        value = getattr(args, field_name)
        if value is not None:
            setattr(config, field_name, value)
    if args.no_bapr_v2_fallback:
        config.bapr_v2_use_fallback = False
    if args.bapr_v2_freeze_gate_in_teacher:
        config.bapr_v2_freeze_gate_in_teacher = True
    if args.bapr_v2_advantage_gate:
        config.bapr_v2_advantage_gate = True
    if args.bapr_v2_train_advantage_constraint:
        config.bapr_v2_train_advantage_constraint = True
    if args.bapr_v2_train_update_filter:
        config.bapr_v2_train_update_filter = True
    if args.bapr_v2_freeze_alpha:
        config.bapr_v2_freeze_alpha = True
    if args.bapr_v2_warmstart_conditioned:
        config.bapr_v2_warmstart_conditioned = True
    if args.bapr_v3_freeze_teacher_after_teacher:
        config.bapr_v3_freeze_teacher_after_teacher = True
    if args.bapr_v3_eval_context_ladder:
        config.bapr_v3_eval_context_ladder = True
    if args.bapr_v3_reset_context_on_resume:
        config.bapr_v3_reset_context_on_resume = True
    if args.bapr_regime_update_context_during_adaptation:
        config.bapr_regime_freeze_context_after_inference = False
    if args.no_bapr_regime_clear_replay:
        config.bapr_regime_clear_replay_on_adaptation = False
    if args.no_bapr_regime_zero_residual_init:
        config.bapr_regime_zero_residual_init = False
    if args.no_bapr_regime_advantage_fallback:
        config.bapr_regime_advantage_fallback = False
    if args.beta is not None:
        config.beta = args.beta
    if args.actor_objective is not None:
        config.actor_objective = args.actor_objective
    if args.weight_reg is not None:
        config.weight_reg = args.weight_reg
    if args.beta_ood is not None:
        config.beta_ood = args.beta_ood
    if args.beta_bc is not None:
        config.beta_bc = args.beta_bc
    if args.critic_actor_ratio is not None:
        if args.critic_actor_ratio <= 0:
            parser.error("--critic_actor_ratio must be positive")
        config.critic_actor_ratio = args.critic_actor_ratio
    if args.resac_independent_ratio is not None:
        if not 0.0 <= args.resac_independent_ratio <= 1.0:
            parser.error("--resac_independent_ratio must be in [0, 1]")
        config.resac_independent_ratio = args.resac_independent_ratio
    if args.resac_anchor_lambda is not None:
        if args.resac_anchor_lambda < 0.0:
            parser.error("--resac_anchor_lambda must be non-negative")
        config.resac_anchor_lambda = args.resac_anchor_lambda
    if args.resac_adaptive_beta:
        config.resac_adaptive_beta = True
    for field_name in (
            "resac_beta_start", "resac_beta_end", "resac_beta_warmup"):
        value = getattr(args, field_name)
        if value is not None:
            setattr(config, field_name, value)
    if args.resac_critic_actor_ratio is not None:
        if args.resac_critic_actor_ratio <= 0:
            parser.error("--resac_critic_actor_ratio must be positive")
        config.resac_critic_actor_ratio = args.resac_critic_actor_ratio
    if args.resac_beta_bc is not None:
        if args.resac_beta_bc < 0.0:
            parser.error("--resac_beta_bc must be non-negative")
        config.resac_beta_bc = args.resac_beta_bc
    if args.resac_clip_norm is not None:
        if args.resac_clip_norm < 0.0:
            parser.error("--resac_clip_norm must be non-negative")
        config.resac_clip_norm = args.resac_clip_norm
    if args.ema_tau is not None:
        if not 0.0 <= args.ema_tau <= 1.0:
            parser.error("--ema_tau must be in [0, 1]")
        config.ema_tau = args.ema_tau
    if args.use_ema_eval:
        config.use_ema_eval = True
    if args.use_ema_rollout:
        config.use_ema_rollout = True
    if args.ema_rollout_start_iter is not None:
        config.ema_rollout_start_iter = args.ema_rollout_start_iter
    if args.ema_rollout_require_reg_latched:
        config.ema_rollout_require_reg_latched = True
    if args.bapr_adaptation_mode is not None:
        config.bapr_adaptation_mode = args.bapr_adaptation_mode
    if args.bapr_surprise_threshold is not None:
        config.bapr_surprise_threshold = args.bapr_surprise_threshold
    if args.bapr_gate_warmup_iters is not None:
        config.bapr_gate_warmup_iters = args.bapr_gate_warmup_iters
    if args.bapr_gate_gain is not None:
        config.bapr_gate_gain = args.bapr_gate_gain
    if args.bapr_gate_ema_alpha is not None:
        config.bapr_gate_ema_alpha = args.bapr_gate_ema_alpha
    if args.bapr_gate_max is not None:
        config.bapr_gate_max = args.bapr_gate_max
    if args.bapr_recent_frac_cap is not None:
        config.bapr_recent_frac_cap = args.bapr_recent_frac_cap
    if args.bapr_recent_frac_floor is not None:
        config.bapr_recent_frac_floor = args.bapr_recent_frac_floor
    if args.bapr_recent_true_floor:
        config.bapr_recent_true_floor = True
    if args.bapr_recent_floor_mode is not None:
        config.bapr_recent_floor_mode = args.bapr_recent_floor_mode
    if args.bapr_recent_floor_ratio_low is not None:
        config.bapr_recent_floor_ratio_low = args.bapr_recent_floor_ratio_low
    if args.bapr_recent_floor_ratio_high is not None:
        config.bapr_recent_floor_ratio_high = args.bapr_recent_floor_ratio_high
    if args.bapr_recent_floor_mid_frac is not None:
        config.bapr_recent_floor_mid_frac = args.bapr_recent_floor_mid_frac
    if args.bapr_recent_floor_extreme_frac is not None:
        config.bapr_recent_floor_extreme_frac = args.bapr_recent_floor_extreme_frac
    if args.bapr_recent_disagreement_gate:
        config.bapr_recent_disagreement_gate = True
    if args.bapr_recent_open_if_reg_latched:
        config.bapr_recent_open_if_reg_latched = True
    if args.bapr_recent_qstd_threshold is not None:
        config.bapr_recent_qstd_threshold = args.bapr_recent_qstd_threshold
    if args.bapr_recent_qstd_ratio_threshold is not None:
        config.bapr_recent_qstd_ratio_threshold = args.bapr_recent_qstd_ratio_threshold
    if args.bapr_reg_disagreement_gate:
        config.bapr_reg_disagreement_gate = True
    if args.bapr_reg_warmup_iters is not None:
        config.bapr_reg_warmup_iters = args.bapr_reg_warmup_iters
    if args.bapr_reg_max_iters is not None:
        config.bapr_reg_max_iters = args.bapr_reg_max_iters
    if args.bapr_reg_latch:
        config.bapr_reg_latch = True
    if args.bapr_reg_require_both:
        config.bapr_reg_require_both = True
    if args.bapr_reg_qstd_threshold is not None:
        config.bapr_reg_qstd_threshold = args.bapr_reg_qstd_threshold
    if args.bapr_reg_qstd_ratio_threshold is not None:
        config.bapr_reg_qstd_ratio_threshold = args.bapr_reg_qstd_ratio_threshold
    if args.bapr_reg_emergency_gate:
        config.bapr_reg_emergency_gate = True
    if args.bapr_reg_emergency_scale is not None:
        config.bapr_reg_emergency_scale = args.bapr_reg_emergency_scale
    if args.bapr_reg_emergency_qstd_threshold is not None:
        config.bapr_reg_emergency_qstd_threshold = args.bapr_reg_emergency_qstd_threshold
    if args.bapr_reg_emergency_qstd_ratio_threshold is not None:
        config.bapr_reg_emergency_qstd_ratio_threshold = args.bapr_reg_emergency_qstd_ratio_threshold
    if args.bapr_reg_latched_scale is not None:
        config.bapr_reg_latched_scale = args.bapr_reg_latched_scale
    if args.bapr_reg_perf_collapse_gate:
        config.bapr_reg_perf_collapse_gate = True
    if args.bapr_reg_perf_collapse_drop_frac is not None:
        config.bapr_reg_perf_collapse_drop_frac = args.bapr_reg_perf_collapse_drop_frac
    if args.bapr_reg_perf_collapse_scale is not None:
        config.bapr_reg_perf_collapse_scale = args.bapr_reg_perf_collapse_scale
    if args.bapr_reg_perf_collapse_warmup_iters is not None:
        config.bapr_reg_perf_collapse_warmup_iters = args.bapr_reg_perf_collapse_warmup_iters
    if args.bapr_controller_mode is not None:
        config.bapr_controller_mode = args.bapr_controller_mode
    if args.bapr_controller_warmup_iters is not None:
        config.bapr_controller_warmup_iters = args.bapr_controller_warmup_iters
    if args.bapr_controller_min_peak is not None:
        config.bapr_controller_min_peak = args.bapr_controller_min_peak
    if args.bapr_controller_drop_frac is not None:
        config.bapr_controller_drop_frac = args.bapr_controller_drop_frac
    if args.bapr_controller_drop_ramp is not None:
        config.bapr_controller_drop_ramp = args.bapr_controller_drop_ramp
    if args.bapr_controller_soft_decay is not None:
        config.bapr_controller_soft_decay = args.bapr_controller_soft_decay
    if args.bapr_controller_soft_min_signal is not None:
        config.bapr_controller_soft_min_signal = args.bapr_controller_soft_min_signal
    if args.bapr_controller_low_qstd_ratio is not None:
        config.bapr_controller_low_qstd_ratio = args.bapr_controller_low_qstd_ratio
    if args.bapr_controller_low_qstd_threshold is not None:
        config.bapr_controller_low_qstd_threshold = args.bapr_controller_low_qstd_threshold
    if args.bapr_controller_latch:
        config.bapr_controller_latch = True
    if args.bapr_controller_release_drop_frac is not None:
        config.bapr_controller_release_drop_frac = args.bapr_controller_release_drop_frac
    if args.bapr_controller_max_active_iters is not None:
        config.bapr_controller_max_active_iters = args.bapr_controller_max_active_iters
    if args.bapr_controller_min_active_iters is not None:
        config.bapr_controller_min_active_iters = args.bapr_controller_min_active_iters
    if args.bapr_controller_exit_cooldown_iters is not None:
        config.bapr_controller_exit_cooldown_iters = (
            args.bapr_controller_exit_cooldown_iters)
    if args.bapr_controller_exit_signal_threshold is not None:
        config.bapr_controller_exit_signal_threshold = (
            args.bapr_controller_exit_signal_threshold)
    if args.bapr_controller_exit_improve_frac is not None:
        config.bapr_controller_exit_improve_frac = (
            args.bapr_controller_exit_improve_frac)
    if args.bapr_controller_exit_drawdown_frac is not None:
        config.bapr_controller_exit_drawdown_frac = (
            args.bapr_controller_exit_drawdown_frac)
    if args.bapr_controller_latched_signal is not None:
        config.bapr_controller_latched_signal = args.bapr_controller_latched_signal
    if args.bapr_controller_reg_multiplier is not None:
        config.bapr_controller_reg_multiplier = args.bapr_controller_reg_multiplier
    if args.bapr_controller_reg_recover_iters is not None:
        config.bapr_controller_reg_recover_iters = args.bapr_controller_reg_recover_iters
    if args.bapr_controller_recent_multiplier is not None:
        config.bapr_controller_recent_multiplier = args.bapr_controller_recent_multiplier
    if args.bapr_controller_recent_add_frac is not None:
        config.bapr_controller_recent_add_frac = args.bapr_controller_recent_add_frac
    if args.bapr_controller_lcb_multiplier is not None:
        config.bapr_controller_lcb_multiplier = args.bapr_controller_lcb_multiplier
    if args.bapr_controller_actor_update_multiplier is not None:
        config.bapr_controller_actor_update_multiplier = (
            args.bapr_controller_actor_update_multiplier)
    if args.bapr_controller_actor_recover_iters is not None:
        config.bapr_controller_actor_recover_iters = (
            args.bapr_controller_actor_recover_iters)
    if args.bapr_actor_lcb_qstd_gate:
        config.bapr_actor_lcb_qstd_gate = True
    if args.bapr_actor_lcb_qstd_threshold is not None:
        config.bapr_actor_lcb_qstd_threshold = args.bapr_actor_lcb_qstd_threshold
    if args.bapr_actor_lcb_qstd_ratio_threshold is not None:
        config.bapr_actor_lcb_qstd_ratio_threshold = args.bapr_actor_lcb_qstd_ratio_threshold
    if args.bapr_actor_lcb_require_both:
        config.bapr_actor_lcb_require_both = True
    if args.bapr_actor_lcb_scale is not None:
        config.bapr_actor_lcb_scale = args.bapr_actor_lcb_scale
    if args.bapr_actor_lcb_perf_gate:
        config.bapr_actor_lcb_perf_gate = True
    if args.bapr_actor_lcb_perf_warmup_iters is not None:
        config.bapr_actor_lcb_perf_warmup_iters = args.bapr_actor_lcb_perf_warmup_iters
    if args.bapr_actor_lcb_perf_drop_frac is not None:
        config.bapr_actor_lcb_perf_drop_frac = args.bapr_actor_lcb_perf_drop_frac
    if args.bapr_actor_lcb_perf_min_peak is not None:
        config.bapr_actor_lcb_perf_min_peak = args.bapr_actor_lcb_perf_min_peak
    if args.bapr_actor_lcb_perf_ema_alpha is not None:
        config.bapr_actor_lcb_perf_ema_alpha = args.bapr_actor_lcb_perf_ema_alpha
    if args.bapr_residual_delta is not None:
        config.bapr_residual_delta = args.bapr_residual_delta
    if args.bapr_residual_gate_scale is not None:
        config.bapr_residual_gate_scale = args.bapr_residual_gate_scale
    if args.bapr_residual_adv_margin is not None:
        config.bapr_residual_adv_margin = args.bapr_residual_adv_margin
    if args.bapr_residual_adv_temp is not None:
        config.bapr_residual_adv_temp = args.bapr_residual_adv_temp
    if args.bapr_residual_qstd_scale is not None:
        config.bapr_residual_qstd_scale = args.bapr_residual_qstd_scale
    if args.bapr_residual_behavior_weight is not None:
        config.bapr_residual_behavior_weight = args.bapr_residual_behavior_weight
    if args.bapr_residual_action_penalty is not None:
        config.bapr_residual_action_penalty = args.bapr_residual_action_penalty
    config.brax_backend = args.backend
    if args.log_scale_limit is not None:
        config.log_scale_limit = args.log_scale_limit
    if args.task_scale_distribution is not None:
        config.task_scale_distribution = args.task_scale_distribution
    if args.penalty_scale is not None:
        config.penalty_scale = args.penalty_scale
    if args.bapr_grad_clip_norm is not None:
        config.bapr_grad_clip_norm = args.bapr_grad_clip_norm
    if args.hazard_rate is not None:
        config.hazard_rate = args.hazard_rate
    if args.regime_hazard_rate is not None:
        config.regime_hazard_rate = args.regime_hazard_rate
    if args.max_run_length is not None:
        config.max_run_length = args.max_run_length
    if args.num_regimes is not None:
        config.num_regimes = args.num_regimes
    if args.surprise_weights is not None:
        (config.surprise_reward_weight,
         config.surprise_q_weight,
         config.surprise_reg_weight) = args.surprise_weights
    if args.log_interval is not None:
        config.log_interval = args.log_interval
    if args.eval_episodes is not None:
        config.eval_episodes = args.eval_episodes
    if args.max_episode_steps is not None:
        config.max_episode_steps = args.max_episode_steps
    if args.eval_protocol is not None:
        config.eval_protocol = args.eval_protocol
    if args.eval_switching_episodes is not None:
        config.eval_switching_episodes = args.eval_switching_episodes
    if args.eval_switching_period_steps is not None:
        config.eval_switching_period_steps = args.eval_switching_period_steps
    if args.env_type is not None:
        config.env_type = args.env_type
    if args.mean_dwell_iters is not None:
        config.discrete_mean_dwell_iters = args.mean_dwell_iters
    if args.mode_variant is not None:
        config.discrete_mode_variant = args.mode_variant
    if args.stochastic_mode_family is not None:
        config.stochastic_mode_family = args.stochastic_mode_family
    if args.stochastic_mode_dwell_steps is not None:
        config.stochastic_mode_dwell_steps = (
            args.stochastic_mode_dwell_steps)
    if args.stochastic_mode_dwell_distribution is not None:
        config.stochastic_mode_dwell_distribution = (
            args.stochastic_mode_dwell_distribution)
    if args.regime_context_source is not None:
        config.regime_context_source = args.regime_context_source
    if args.stochastic_mode_fixed_id is not None:
        config.stochastic_mode_fixed_id = args.stochastic_mode_fixed_id
    if args.early_kill_iter is not None:
        config.early_kill_iter = args.early_kill_iter
    if args.early_kill_reward is not None:
        config.early_kill_reward = args.early_kill_reward
    if args.stage2_kill_iter is not None:
        config.stage2_kill_iter = args.stage2_kill_iter
    if args.stage2_kill_reward is not None:
        config.stage2_kill_reward = args.stage2_kill_reward
    if args.changing_period is not None:
        config.changing_period = args.changing_period
    if args.changing_interval is not None:
        config.changing_interval = args.changing_interval
    if args.save_interval is not None:
        config.save_interval = args.save_interval
    if args.use_regime_belief:
        config.use_regime_belief = True
    if args.no_belief_conditioning:
        config.belief_conditioned = False
    if args.critic_target_mode is not None:
        config.critic_target_mode = args.critic_target_mode
    if args.no_per_trans_belief:
        config.use_per_transition_belief = False
    if args.oracle_reset_on_switch:
        config.oracle_reset_on_switch = True
    config.resume = args.resume
    config.resume_boundary_audit = args.resume_boundary_audit
    config.resume_boundary_expected_iteration = (
        args.resume_boundary_expected_iteration)
    config.resume_boundary_expected_total_steps = (
        args.resume_boundary_expected_total_steps)
    config.resume_boundary_expected_update_count = (
        args.resume_boundary_expected_update_count)
    if args.min_resume_iteration is not None:
        config.min_resume_iteration = args.min_resume_iteration

    # Redesigned BAPR defaults: keep the no-BOCD/context backbone and use the
    # ensemble mean actor. Static/gated LCB variants were too conservative on
    # HalfCheetah/Ant; stability comes from EMA evaluation, not Q-std penalties.
    if config.algo == "bapr" and config.bapr_adaptation_mode == "gate":
        if args.actor_objective is None:
            config.actor_objective = "mean"
        if not args.use_regime_belief and not args.no_belief_conditioning:
            config.belief_conditioned = False

    train(config)


if __name__ == "__main__":
    main()
