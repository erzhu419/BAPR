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
            backend=config.brax_backend)
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

    Uses EMA policy if available (BAPR) for more stable evaluation.
    """
    from flax import nnx
    if hasattr(agent, 'ema_policy'):
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


def collect_samples(agent, env, replay_buffer, config, n_steps: int):
    """Collect n_steps via GPU scan-fused rollout.

    Fuses policy + physics + auto-reset in ONE XLA call.
    Data stays on GPU: rollout returns JAX arrays -> push_batch_jax -> buffer(GPU).
    """
    is_random = replay_buffer.size < config.start_train_steps
    rng_key = jax.random.PRNGKey(config.seed + replay_buffer.size)

    if is_random:
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
        (obs, act, rew, nobs, done), ep_rewards = env.rollout(
            policy_params, n_steps, rng_key,
            context_params=context_params, belief_vec=belief_vec)

        # Zero-copy push: JAX arrays go directly to GPU-native replay buffer
        task_ids = jnp.full(n_steps, rollout_task_id, dtype=jnp.int32)
        replay_buffer.push_batch_jax(obs, act, rew.reshape(-1, 1),
                                     nobs, done.reshape(-1, 1), task_ids)

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
    replay_buffer = ReplayBuffer(obs_dim, act_dim, capacity=config.replay_size)

    # Logging
    run_name = config.run_name or f"{config.algo}_{config.env_name}_{config.seed}"
    log_dir = os.path.join(config.save_root, run_name, "logs")
    model_dir = os.path.join(config.save_root, run_name, "models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    logger = Logger(log_dir)

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
        ep_rewards, recent_rollout = collect_samples(
            agent, env, replay_buffer, config, config.samples_per_iter)
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
                prev_lam = float(getattr(agent, "_current_weighted_lambda", 0.0))
                recent_frac = min(0.8, max(0.0, prev_lam))
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
        eval_time = time.time() - eval_start
        eval_time_total += eval_time

        logger.log("total_steps", total_steps)
        logger.log("iteration", iteration)
        logger.log("mode_id", env.current_task_id)
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
        extra += f" | {iter_time:.1f}s/iter"
        logger.print_status(iteration, extra)

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
    parser.add_argument("--backend", type=str, default="spring",
                        choices=["spring", "generalized"],
                        help="Brax physics backend: spring (fast) or generalized (accurate)")
    parser.add_argument("--log_scale_limit", type=float, default=None,
                        help="Override log_scale_limit for gravity sampling range")
    parser.add_argument("--penalty_scale", type=float, default=None,
                        help="Override BAPR penalty_scale")
    parser.add_argument("--hazard_rate", type=float, default=None,
                        help="Override BOCD hazard_rate")
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
    config.brax_backend = args.backend
    if args.log_scale_limit is not None:
        config.log_scale_limit = args.log_scale_limit
    if args.penalty_scale is not None:
        config.penalty_scale = args.penalty_scale
    if args.hazard_rate is not None:
        config.hazard_rate = args.hazard_rate
    if args.log_interval is not None:
        config.log_interval = args.log_interval
    if args.eval_episodes is not None:
        config.eval_episodes = args.eval_episodes
    if args.env_type is not None:
        config.env_type = args.env_type
    if args.mean_dwell_iters is not None:
        config.discrete_mean_dwell_iters = args.mean_dwell_iters
    if args.changing_period is not None:
        config.changing_period = args.changing_period
    if args.save_interval is not None:
        config.save_interval = args.save_interval
    if args.use_regime_belief:
        config.use_regime_belief = True
    config.resume = args.resume

    train(config)


if __name__ == "__main__":
    main()
