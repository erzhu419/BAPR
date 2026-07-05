"""Main training entry point for JAX-based RL experiments.

Usage:
    conda run -n jax-rl python -m jax_experiments.train --algo resac --env Hopper-v2
    conda run -n jax-rl python -m jax_experiments.train --algo escp  --env Hopper-v2
    conda run -n jax-rl python -m jax_experiments.train --algo bapr  --env Hopper-v2
"""
import os
import sys
import argparse
import time
import faulthandler
import platform
import numpy as np

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


def make_env(config, seed_offset=0):
    """Build training or eval env based on config.env_type.

    env_type='continuous': original ESCP-style continuous task family
    env_type='discrete_mode': K=4 discrete semantic modes (BAPR sweet spot)
    """
    if getattr(config, 'env_type', 'continuous') == 'discrete_mode':
        return DiscreteModePiecewiseEnv(
            env_name=config.env_name,
            mean_dwell_iters=config.discrete_mean_dwell_iters,
            steps_per_iter=config.samples_per_iter,
            dwell_distribution=config.discrete_dwell_distribution,
            reward_shaping=config.discrete_reward_shaping,
            seed=config.seed + seed_offset,
            backend=config.brax_backend,
            mode_variant=getattr(config, 'discrete_mode_variant', 'orig'))
    return NonstationaryEnv(
        config.env_name, rand_params=config.varying_params,
        log_scale_limit=config.log_scale_limit,
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
    elif algo_name == "sac":
        from jax_experiments.algos.sac_base import SACBase
        return SACBase(obs_dim, act_dim, config, seed=config.seed)
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


def evaluate(agent, env, config: Config, tasks=None, n_episodes: int = 10):
    """Fast GPU-scan eval: deterministic policy, single JIT call for all episodes.

    Behavior depends on `tasks`:
      - None or len(tasks)==1: single task (legacy behavior)
      - len(tasks) <= 10 (e.g. discrete_mode K modes): rotate through ALL
        tasks, run n_episodes per task, average across all → measures
        cross-mode generalization (BAPR's design target).
      - len(tasks) > 10 (continuous, 40 tasks): use first task only
        (full sweep too expensive every eval).

    Uses EMA policy only when config.use_ema_eval is enabled. In the BAPR
    redesign runs, EMA was a diagnostic path and could mask online-policy
    regressions, so online evaluation is the default.
    """
    from flax import nnx
    if getattr(config, 'use_ema_eval', False) and hasattr(agent, 'ema_policy'):
        policy_params = nnx.state(agent.ema_policy, nnx.Param)
    else:
        policy_params = nnx.state(agent.policy, nnx.Param)
    context_params = None
    if hasattr(agent, 'context_net'):
        context_params = nnx.state(agent.context_net, nnx.Param)

    rng_key = jax.random.PRNGKey(42)  # fixed key for reproducible eval
    n_steps = n_episodes * config.max_episode_steps

    # Decide which tasks to eval over
    if tasks is None or len(tasks) == 0:
        eval_tasks = [None]
    elif len(tasks) <= 10:
        eval_tasks = tasks  # rotate through all (discrete_mode case)
    else:
        eval_tasks = [tasks[0]]  # representative only (continuous case)

    belief_vec = None
    if hasattr(agent, 'belief_dim') and agent.belief_dim > 0:
        # Use the agent's own _build_belief_jax if available — it knows whether
        # to emit ρ(h) (legacy) or concat([ρ, μ]) (Minimum redesign).
        if hasattr(agent, '_build_belief_jax'):
            belief_vec = agent._build_belief_jax()
        else:
            import jax.numpy as _jnp
            belief_vec = _jnp.asarray(agent.belief_tracker.belief, dtype=_jnp.float32)

    all_rewards = []
    for task in eval_tasks:
        if task is not None:
            env.set_task(task)
        rew_np, done_np = env.eval_rollout(
            policy_params, n_steps, rng_key,
            context_params=context_params, belief_vec=belief_vec)
        ep_rewards, ep_r, completed = [], 0.0, 0
        for i in range(n_steps):
            ep_r += rew_np[i]
            if done_np[i] > 0.5:
                ep_rewards.append(ep_r)
                ep_r = 0.0
                completed += 1
                if completed >= n_episodes:
                    break
        if not ep_rewards:
            ep_rewards = [ep_r]
        all_rewards.extend(ep_rewards)

    return float(np.mean(all_rewards)), float(np.std(all_rewards))


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
        for _ in range(n_steps):
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            replay_buffer.push(obs, action, reward, next_obs, done, env.current_task_id)
            episode_reward += reward
            obs = next_obs
            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0.0
                obs = env.reset()
        return episode_rewards, None  # no recent_rollout in random phase
    else:
        # Scan-fused GPU rollout: sys is fixed for the entire 4000-step scan
        # (lax.scan requires static structure). Task switching happens at iter boundary.
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

        # Task ID used for the entire scan rollout (env switches at iter boundaries
        # only, so all transitions in this scan share the same task)
        rollout_task_id = env.current_task_id
        # GPT-5.5 advice #3: warmup-consistent rollout — when training-side
        # would zero out context for the first context_warmup_iters, also
        # zero it on the rollout side so the data we collect was generated
        # by the same policy class the critic is being trained against.
        rollout_warmup = current_iter < config.context_warmup_iters
        (obs, act, rew, nobs, done), ep_rewards = env.rollout(
            policy_params, n_steps, rng_key,
            context_params=context_params, belief_vec=belief_vec,
            warmup=rollout_warmup)

        # Zero-copy push: JAX arrays go directly to GPU-native replay buffer
        task_ids = jnp.full(n_steps, rollout_task_id, dtype=jnp.int32)
        # GPT-5.5 advice #2: store the belief that produced this rollout.
        # If warmup is on, we also stored zero context — store zero belief
        # so the critic sees a consistent (zero ctx, zero belief) input.
        if rollout_warmup and belief_vec is not None:
            stored_belief = jnp.zeros_like(belief_vec)
        else:
            stored_belief = belief_vec
        replay_buffer.push_batch_jax(obs, act, rew.reshape(-1, 1),
                                     nobs, done.reshape(-1, 1), task_ids,
                                     belief=stored_belief)

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
            "rollout_task_id": int(rollout_task_id),
        }
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

    # Create environment (continuous or discrete-mode based on config.env_type)
    env = make_env(config, seed_offset=0)
    obs_dim = env.obs_dim
    act_dim = env.act_dim

    # Sample tasks
    train_tasks = env.sample_tasks(config.task_num)
    test_tasks = env.sample_tasks(config.test_task_num)

    # Setup non-stationary switching
    env.set_nonstationary_para(train_tasks, config.changing_period, config.changing_interval)

    # Create agent
    agent = make_algo(config.algo, obs_dim, act_dim, config)

    # Build scan-fused rollout (compiles policy+physics into one XLA call)
    policy_graphdef = nnx.graphdef(agent.policy)
    context_graphdef = None
    if hasattr(agent, 'context_net'):
        context_graphdef = nnx.graphdef(agent.context_net)
    env.build_rollout_fn(policy_graphdef, context_graphdef)
    print(f"  Built scan-fused rollout for {config.env_name} (context={'yes' if context_graphdef else 'no'})")

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
    ckpt_dir = os.path.join(config.save_root, run_name, "checkpoints")
    if getattr(config, 'resume', False) and has_checkpoint(ckpt_dir):
        start_iteration, total_steps = load_checkpoint(
            ckpt_dir, agent, replay_buffer, logger, config.algo)
        print(f"Resumed from checkpoint: iter={start_iteration}, steps={total_steps}")
    else:
        print(f"Logging to: {log_dir}")
        print(f"Starting training... (first {config.start_train_steps} steps are random exploration)")
    if config.algo == "bapr":
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
    eval_env.build_rollout_fn(policy_graphdef, context_graphdef)  # needed for _rollout_scan_det

    total_steps = total_steps  # from checkpoint or 0
    start_time = time.time()
    collect_time_total = 0.0
    train_time_total = 0.0
    eval_time_total = 0.0

    for iteration in range(start_iteration, config.max_iters):
        iter_start = time.time()

        # --- Collect samples ---
        collect_start = time.time()
        prev_task_id = int(env.current_task_id)
        ep_rewards, recent_rollout = collect_samples(
            agent, env, replay_buffer, config, config.samples_per_iter,
            current_iter=iteration)
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
            _bapr_like = _algo in ("bapr", "bapr_no_rmdm", "bad_bapr",
                                    "bapr_unsupervised")
            _escp_like = _algo in ("escp", "bapr_no_bocd", "bapr_no_adapt_beta",
                                    "bapr_fixed_decay")

            # Change 3: belief-aware replay sampling for BAPR. When BOCD detects
            # a regime shift (high λ_w), sample more from recent transitions.
            if _bapr_like and hasattr(replay_buffer, 'sample_stacked_mixed'):
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

            if _bapr_like:
                metrics = agent.multi_update(
                    stacked, current_iter=iteration,
                    recent_rewards=ep_rewards if ep_rewards else None,
                    recent_rollout=recent_rollout)
                metrics["recent_replay_frac"] = recent_frac
            elif _escp_like:
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
            eval_mean, eval_std = evaluate(agent, eval_env, config, test_tasks,
                                           n_episodes=config.eval_episodes)
            logger.log("eval_reward", eval_mean)
            logger.log("eval_reward_std", eval_std)
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
                        choices=["resac", "escp", "bapr", "sac",
                                 "bapr_no_bocd", "bapr_no_rmdm",
                                 "bapr_no_adapt_beta", "bapr_fixed_decay",
                                 "bad_bapr", "bapr_unsupervised"])
    parser.add_argument("--env", type=str, default="Hopper-v2")
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--max_iters", type=int, default=2000)
    parser.add_argument("--varying_params", nargs="+", default=["gravity"])
    parser.add_argument("--task_num", type=int, default=40)
    parser.add_argument("--test_task_num", type=int, default=40)
    parser.add_argument("--save_root", type=str, default="jax_experiments/results")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--ep_dim", type=int, default=2)
    parser.add_argument("--ensemble_size", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--samples_per_iter", type=int, default=None)
    parser.add_argument("--updates_per_iter", type=int, default=None)
    parser.add_argument("--context_warmup_iters", type=int, default=None)
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
    parser.add_argument("--log_interval", type=int, default=None,
                        help="Eval every N iterations (default: 5)")
    parser.add_argument("--eval_episodes", type=int, default=None,
                        help="Number of eval episodes (default: 5)")
    parser.add_argument("--save_interval", type=int, default=None,
                        help="Save checkpoint every N iterations (default: 50)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint if available")
    parser.add_argument("--env_type", type=str, default=None,
                        choices=["continuous", "discrete_mode"],
                        help="continuous (ESCP-style 40 tasks) or discrete_mode "
                             "(K=4 semantic modes, exp dwell — BAPR sweet spot)")
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

    config = Config()
    config.algo = args.algo
    config.env_name = args.env
    config.seed = args.seed
    config.max_iters = args.max_iters
    config.varying_params = args.varying_params
    config.task_num = args.task_num
    config.test_task_num = args.test_task_num
    config.save_root = args.save_root
    config.run_name = args.run_name
    config.ep_dim = args.ep_dim
    config.ensemble_size = args.ensemble_size
    config.hidden_dim = args.hidden_dim
    # Only override Config defaults when explicitly provided
    if args.samples_per_iter is not None:
        config.samples_per_iter = args.samples_per_iter
    if args.updates_per_iter is not None:
        config.updates_per_iter = args.updates_per_iter
    if args.context_warmup_iters is not None:
        config.context_warmup_iters = args.context_warmup_iters
    if args.beta is not None:
        config.beta = args.beta
    if args.actor_objective is not None:
        config.actor_objective = args.actor_objective
    if args.weight_reg is not None:
        config.weight_reg = args.weight_reg
    if args.beta_ood is not None:
        config.beta_ood = args.beta_ood
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
    if args.env_type is not None:
        config.env_type = args.env_type
    if args.mean_dwell_iters is not None:
        config.discrete_mean_dwell_iters = args.mean_dwell_iters
    if args.mode_variant is not None:
        config.discrete_mode_variant = args.mode_variant
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
