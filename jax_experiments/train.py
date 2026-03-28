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
from jax_experiments.envs.brax_env import BraxNonstationaryEnv as NonstationaryEnv


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
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def evaluate(agent, env, config: Config, tasks=None, n_episodes: int = 10):
    """Evaluate using sequential API (few episodes, overhead acceptable)."""
    rewards = []
    for ep in range(n_episodes):
        if tasks is not None:
            env.set_task(tasks[ep % len(tasks)])
        obs = env.reset()
        episode_reward = 0.0
        for _ in range(config.max_episode_steps):
            action = agent.select_action(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                break
        rewards.append(episode_reward)
    return float(np.mean(rewards)), float(np.std(rewards))


def collect_samples(agent, env, replay_buffer, config, n_steps: int):
    """Collect n_steps via GPU scan-fused rollout.

    Fuses policy + physics + auto-reset in ONE XLA call.
    Single bulk CPU transfer at the end.
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
        return episode_rewards
    else:
        # Scan-fused GPU rollout: sys is fixed for the entire 4000-step scan
        # (lax.scan requires static structure). Task switching happens at iter boundary.
        policy_params = nnx.state(agent.policy, nnx.Param)
        context_params = None
        if hasattr(agent, 'context_net'):
            context_params = nnx.state(agent.context_net, nnx.Param)

        prev_task_id = env.current_task_id  # record before rollout
        (obs, act, rew, nobs, done), ep_rewards = env.rollout(
            policy_params, n_steps, rng_key, context_params=context_params)

        # Bulk push to replay buffer (all steps share current task_id)
        task_ids = np.full(n_steps, env.current_task_id, dtype=np.int32)
        replay_buffer.push_batch(obs, act, rew.reshape(-1, 1),
                                 nobs, done.reshape(-1, 1), task_ids)

        # Reset belief only when TASK switches (not on every episode end)
        # BOCD handles within-task episode resets naturally via its own dynamics
        if hasattr(agent, 'reset_episode') and env.current_task_id != prev_task_id:
            agent.reset_episode()

        return ep_rewards


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

    # Create environment
    env = NonstationaryEnv(config.env_name, rand_params=config.varying_params,
                           log_scale_limit=config.log_scale_limit, seed=config.seed,
                           backend=config.brax_backend)
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

    print(f"Logging to: {log_dir}")
    print(f"Starting training... (first {config.start_train_steps} steps are random exploration)")

    # Separate eval env to avoid polluting training env's state
    eval_env = NonstationaryEnv(config.env_name, rand_params=config.varying_params,
                                log_scale_limit=config.log_scale_limit, seed=config.seed + 1000,
                                backend=config.brax_backend)

    total_steps = 0
    start_time = time.time()

    for iteration in range(config.max_iters):
        iter_start = time.time()

        # --- Collect samples ---
        ep_rewards = collect_samples(agent, env, replay_buffer, config, config.samples_per_iter)
        total_steps += config.samples_per_iter

        if len(ep_rewards) > 0:
            logger.log("train_reward_mean", float(np.mean(ep_rewards)))
            logger.log("train_reward_std", float(np.std(ep_rewards)))

        # --- Training updates (fused via lax.scan) ---
        if replay_buffer.size >= config.start_train_steps:
            stacked = replay_buffer.sample_stacked(
                config.updates_per_iter, config.batch_size)
            if config.algo == "bapr":
                metrics = agent.multi_update(
                    stacked, current_iter=iteration,
                    recent_rewards=ep_rewards if ep_rewards else None)
            elif config.algo == "escp":
                metrics = agent.multi_update(
                    stacked, current_iter=iteration)
            else:
                metrics = agent.multi_update(stacked)

            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    logger.log(k, v)

        # --- Evaluation (expensive, only every log_interval) ---
        eval_mean = None
        if iteration % config.log_interval == 0:
            eval_mean, eval_std = evaluate(agent, eval_env, config, test_tasks,
                                           n_episodes=config.eval_episodes)
            logger.log("eval_reward", eval_mean)
            logger.log("eval_reward_std", eval_std)
        logger.log("total_steps", total_steps)
        logger.log("iteration", iteration)
        logger.log("mode_id", env.current_task_id)

        # --- Always print status ---
        iter_time = time.time() - iter_start
        q_std_str = ""
        if "q_std_mean" in (metrics if replay_buffer.size >= config.start_train_steps else {}):
            q_std_str = f" | Q-std: {metrics.get('q_std_mean', 0):.2f}"
        eval_str = f" | Eval: {eval_mean:.1f}" if eval_mean is not None else ""
        extra = f"TaskID: {env.current_task_id}{q_std_str}{eval_str}"
        if config.algo == "bapr":
            extra += f" | λ_w: {agent._current_weighted_lambda:.3f}"
        if config.algo in ("escp", "bapr"):
            warmup = iteration < config.context_warmup_iters
            extra += f" | {'[WARMUP]' if warmup else '[ACTIVE]'}"
        extra += f" | {iter_time:.1f}s/iter"
        logger.print_status(iteration, extra)

        # --- Save ---
        if iteration % config.save_interval == 0:
            logger.save()

    # Final save
    logger.save()
    elapsed = time.time() - start_time
    print(f"\nTraining complete! Total time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"Results saved to: {log_dir}")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="JAX RL Training")
    parser.add_argument("--algo", type=str, default="resac",
                        choices=["resac", "escp", "bapr"])
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

    train(config)


if __name__ == "__main__":
    main()
