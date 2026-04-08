"""ID/OOD evaluation: load trained checkpoints and evaluate on ID vs OOD tasks.

Evaluates trained agents on:
  - In-Distribution (ID): tasks sampled from the same range as training
  - Out-of-Distribution (OOD): tasks sampled from a wider range (ood_scale × training)

Usage:
    python -m jax_experiments.analysis.eval_id_ood \
        --results_root jax_experiments/results \
        --envs HalfCheetah-v2 Hopper-v2 Walker2d-v2 Ant-v2 \
        --algos bapr escp resac \
        --output paper/fig_id_ood.pdf
"""
import argparse
import os
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure CUDA libs available
NVIDIA_LIB = None
for p in sys.path:
    candidate = os.path.join(p, "nvidia")
    if os.path.isdir(candidate):
        NVIDIA_LIB = candidate
        break
if NVIDIA_LIB is None:
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

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
from flax import nnx

from jax_experiments.configs.default import Config
from jax_experiments.common.checkpoint import load_checkpoint, has_checkpoint
from jax_experiments.envs.brax_env import BraxNonstationaryEnv as NonstationaryEnv


ALGO_COLORS = {
    "bapr": "#2196F3", "escp": "#FF9800", "resac": "#4CAF50",
    "sac": "#9E9E9E", "bapr_no_bocd": "#E91E63",
    "bapr_fixed_decay": "#FF5722",
}

ALGO_LABELS = {
    "bapr": "BAPR", "escp": "ESCP", "resac": "RE-SAC", "sac": "SAC",
    "bapr_no_bocd": "w/o BOCD", "bapr_fixed_decay": "FixedDecay",
}

# Environment configs (must match training)
ENV_PARAMS = {
    "HalfCheetah-v2": (["gravity"], 0.75),
    "Hopper-v2": (["gravity"], 0.75),
    "Walker2d-v2": (["dof_damping", "body_mass"], 1.5),
    "Ant-v2": (["gravity"], 0.75),
}


def eval_on_tasks(agent, env, tasks, config, n_episodes=5):
    """Evaluate agent on a specific set of tasks."""
    policy_params = nnx.state(agent.policy, nnx.Param)
    context_params = None
    if hasattr(agent, 'context_net'):
        context_params = nnx.state(agent.context_net, nnx.Param)

    all_rewards = []
    for task in tasks:
        env.set_task(task)
        rng_key = jax.random.PRNGKey(42)
        n_steps = n_episodes * config.max_episode_steps

        rew_np, done_np = env.eval_rollout(
            policy_params, n_steps, rng_key, context_params=context_params)

        ep_r, completed = 0.0, 0
        for i in range(n_steps):
            ep_r += rew_np[i]
            if done_np[i] > 0.5:
                all_rewards.append(ep_r)
                ep_r = 0.0
                completed += 1
                if completed >= n_episodes:
                    break

    return np.array(all_rewards) if all_rewards else np.array([0.0])


def load_agent_from_checkpoint(ckpt_dir, algo, obs_dim, act_dim, config):
    """Construct agent and load checkpoint weights."""
    from jax_experiments.train import make_algo
    from jax_experiments.common.replay_buffer import ReplayBuffer
    from jax_experiments.common.logging import Logger
    import tempfile

    agent = make_algo(algo, obs_dim, act_dim, config)
    # Dummy replay buffer / logger (not used, but required by load_checkpoint)
    rb = ReplayBuffer(obs_dim, act_dim, capacity=1000)
    tmp_log = tempfile.mkdtemp()
    logger = Logger(tmp_log)

    load_checkpoint(ckpt_dir, agent, rb, logger, algo)
    return agent


def find_checkpoints(results_root, algo, env):
    """Find all checkpoint directories for algo/env."""
    # Pattern: algo_env_seed_tag/checkpoints or algo_env_seed/checkpoints
    patterns = [
        os.path.join(results_root, f"{algo}_{env}_*/checkpoints"),
        os.path.join(results_root, f"{algo}_{env}/checkpoints"),
    ]
    ckpt_dirs = []
    for pat in patterns:
        for d in sorted(glob.glob(pat)):
            if has_checkpoint(d):
                ckpt_dirs.append(d)
    return ckpt_dirs


def evaluate_id_ood(results_root, env_name, algos, ood_scale=1.5,
                    n_id_tasks=20, n_ood_tasks=20, n_episodes=3):
    """Run ID/OOD evaluation for all algos on one environment."""
    varying_params, log_scale = ENV_PARAMS.get(env_name, (["gravity"], 0.75))

    config = Config()
    config.env_name = env_name
    config.varying_params = varying_params
    config.log_scale_limit = log_scale
    config.brax_backend = "spring"

    # Create eval env
    env = NonstationaryEnv(env_name, rand_params=varying_params,
                           log_scale_limit=log_scale, seed=9999,
                           backend="spring")
    obs_dim = env.obs_dim
    act_dim = env.act_dim

    # Sample ID and OOD tasks
    id_tasks = env.sample_tasks(n_id_tasks)

    orig_limit = env.log_scale_limit
    env.log_scale_limit = orig_limit * ood_scale
    ood_tasks = env.sample_tasks(n_ood_tasks)
    env.log_scale_limit = orig_limit

    results = {}
    for algo in algos:
        ckpt_dirs = find_checkpoints(results_root, algo, env_name)
        if not ckpt_dirs:
            print(f"  No checkpoint for {algo}/{env_name}, skipping")
            continue

        id_rewards_all = []
        ood_rewards_all = []

        for ckpt_dir in ckpt_dirs:
            print(f"  Loading {ckpt_dir} ...")
            config.algo = algo
            agent = load_agent_from_checkpoint(ckpt_dir, algo, obs_dim, act_dim, config)

            # Build rollout fn for eval
            policy_graphdef = nnx.graphdef(agent.policy)
            context_graphdef = None
            if hasattr(agent, 'context_net'):
                context_graphdef = nnx.graphdef(agent.context_net)
            env.build_rollout_fn(policy_graphdef, context_graphdef)

            id_rews = eval_on_tasks(agent, env, id_tasks, config, n_episodes)
            ood_rews = eval_on_tasks(agent, env, ood_tasks, config, n_episodes)
            id_rewards_all.append(float(np.mean(id_rews)))
            ood_rewards_all.append(float(np.mean(ood_rews)))

        results[algo] = {
            "id_mean": float(np.mean(id_rewards_all)),
            "id_std": float(np.std(id_rewards_all)),
            "ood_mean": float(np.mean(ood_rewards_all)),
            "ood_std": float(np.std(ood_rewards_all)),
            "n_seeds": len(ckpt_dirs),
        }
        print(f"  {algo}: ID={results[algo]['id_mean']:.0f}±{results[algo]['id_std']:.0f}  "
              f"OOD={results[algo]['ood_mean']:.0f}±{results[algo]['ood_std']:.0f}")

    env.close()
    return results


def plot_id_ood_comparison(all_results, envs, algos, output=None, figsize=(12, 5)):
    """Plot ID vs OOD performance comparison (grouped bar chart)."""
    n_envs = len(envs)
    fig, axes = plt.subplots(1, n_envs, figsize=figsize)
    if n_envs == 1:
        axes = [axes]

    for ax, env in zip(axes, envs):
        if env not in all_results:
            continue
        results = all_results[env]

        algo_names, id_means, ood_means, id_stds, ood_stds = [], [], [], [], []
        for algo in algos:
            if algo not in results:
                continue
            r = results[algo]
            algo_names.append(ALGO_LABELS.get(algo, algo))
            id_means.append(r["id_mean"])
            ood_means.append(r["ood_mean"])
            id_stds.append(r["id_std"])
            ood_stds.append(r["ood_std"])

        if not algo_names:
            continue

        x = np.arange(len(algo_names))
        width = 0.35
        ax.bar(x - width/2, id_means, width, yerr=id_stds,
               label='ID', color='#4CAF50', alpha=0.7, capsize=3, edgecolor='black')
        ax.bar(x + width/2, ood_means, width, yerr=ood_stds,
               label='OOD', color='#F44336', alpha=0.7, capsize=3, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(algo_names, rotation=30, fontsize=8)
        ax.set_title(env.replace('-v2', ''), fontsize=12)
        if ax == axes[0]:
            ax.set_ylabel('Mean Episode Reward', fontsize=10)
            ax.legend(fontsize=9)

        # Annotate degradation %
        for i in range(len(algo_names)):
            if abs(id_means[i]) > 1e-6:
                deg = (id_means[i] - ood_means[i]) / abs(id_means[i]) * 100
                y_pos = max(id_means[i], ood_means[i]) + max(id_stds[i], ood_stds[i])
                color = 'red' if deg > 20 else 'orange' if deg > 10 else 'green'
                ax.text(x[i], y_pos * 1.02, f"{deg:+.0f}%",
                        ha='center', fontsize=7, color=color, fontweight='bold')

    plt.tight_layout()
    if output:
        os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches='tight')
        fig.savefig(output.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
        print(f"Saved ID/OOD plot to: {output}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="ID/OOD Generalization Evaluation")
    parser.add_argument("--results_root", type=str, default="jax_experiments/results")
    parser.add_argument("--envs", nargs="+",
                        default=["HalfCheetah-v2", "Hopper-v2", "Walker2d-v2", "Ant-v2"])
    parser.add_argument("--algos", nargs="+", default=["bapr", "escp", "resac"])
    parser.add_argument("--ood_scale", type=float, default=1.5,
                        help="OOD range = ood_scale × training range")
    parser.add_argument("--n_tasks", type=int, default=20)
    parser.add_argument("--n_episodes", type=int, default=3)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    all_results = {}
    for env in args.envs:
        print(f"\n{'='*60}")
        print(f"  Evaluating ID/OOD on {env}")
        print(f"{'='*60}")
        all_results[env] = evaluate_id_ood(
            args.results_root, env, args.algos,
            ood_scale=args.ood_scale,
            n_id_tasks=args.n_tasks,
            n_ood_tasks=args.n_tasks,
            n_episodes=args.n_episodes)

    if args.output:
        plot_id_ood_comparison(all_results, args.envs, args.algos, args.output)

    # Print summary table
    print(f"\n{'='*80}")
    print("ID/OOD Summary (mean reward)")
    print(f"{'='*80}")
    header = f"{'Algo':<20}" + "".join(f"{'ID':>8} {'OOD':>8} {'Δ%':>6}  " for _ in args.envs)
    print("               " + "  ".join(f"{e.replace('-v2',''):^22}" for e in args.envs))
    print(header)
    for algo in args.algos:
        row = f"{ALGO_LABELS.get(algo, algo):<20}"
        for env in args.envs:
            r = all_results.get(env, {}).get(algo)
            if r:
                deg = (r['id_mean'] - r['ood_mean']) / max(abs(r['id_mean']), 1) * 100
                row += f"{r['id_mean']:8.0f} {r['ood_mean']:8.0f} {deg:+5.0f}%  "
            else:
                row += f"{'---':>8} {'---':>8} {'---':>6}  "
        print(row)


if __name__ == "__main__":
    main()
