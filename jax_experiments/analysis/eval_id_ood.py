"""ID/OOD evaluation framework.

Evaluates trained agents on:
  - In-Distribution (ID): tasks sampled from the same range as training
  - Out-of-Distribution (OOD): tasks sampled from a wider range (e.g., 133% or 150%)

Adds --eval_mode (id|ood|both) to train.py and provides standalone eval script.

Usage:
    python -m jax_experiments.analysis.eval_id_ood \
        --results_root jax_experiments/results \
        --env HalfCheetah-v2 \
        --algos bapr escp resac \
        --output paper/fig_id_ood.pdf
"""
import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt


ALGO_COLORS = {
    "bapr": "#2196F3", "escp": "#FF9800", "resac": "#4CAF50",
    "sac": "#9E9E9E", "bapr_no_bocd": "#E91E63",
    "bapr_fixed_decay": "#FF5722",
}

ALGO_LABELS = {
    "bapr": "BAPR", "escp": "ESCP", "resac": "RE-SAC", "sac": "SAC",
    "bapr_no_bocd": "w/o BOCD", "bapr_fixed_decay": "FixedDecay",
}


def eval_agent_on_tasks(agent, env, tasks, config, n_episodes=5):
    """Evaluate agent on a specific set of tasks (ID or OOD)."""
    from flax import nnx
    import jax

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

        ep_rewards, ep_r, completed = [], 0.0, 0
        for i in range(n_steps):
            ep_r += rew_np[i]
            if done_np[i] > 0.5:
                ep_rewards.append(ep_r)
                ep_r = 0.0
                completed += 1
                if completed >= n_episodes:
                    break
        if ep_rewards:
            all_rewards.extend(ep_rewards)

    return np.array(all_rewards) if all_rewards else np.array([0.0])


def generate_ood_tasks(env, n_tasks, ood_scale=1.5, seed=999):
    """Generate OOD tasks with wider parameter range.

    Temporarily widens log_scale_limit, samples tasks, then restores.
    """
    original_limit = env.log_scale_limit

    # OOD tasks use ood_scale × the training range
    env.log_scale_limit = original_limit * ood_scale
    tasks = env.sample_tasks(n_tasks)
    env.log_scale_limit = original_limit

    return tasks


def plot_id_ood_comparison(results, envs, algos, output=None, figsize=(12, 5)):
    """Plot ID vs OOD performance comparison."""
    n_envs = len(envs)
    fig, axes = plt.subplots(1, n_envs, figsize=figsize)
    if n_envs == 1:
        axes = [axes]

    for ax, env in zip(axes, envs):
        if env not in results:
            continue

        algo_names = []
        id_means = []
        ood_means = []
        id_stds = []
        ood_stds = []

        for algo in algos:
            if algo not in results[env]:
                continue
            r = results[env][algo]
            algo_names.append(ALGO_LABELS.get(algo, algo))
            id_means.append(r.get("id_mean", 0))
            ood_means.append(r.get("ood_mean", 0))
            id_stds.append(r.get("id_std", 0))
            ood_stds.append(r.get("ood_std", 0))

        if not algo_names:
            continue

        x = np.arange(len(algo_names))
        width = 0.35

        bars1 = ax.bar(x - width/2, id_means, width, yerr=id_stds,
                       label='ID', color='#4CAF50', alpha=0.7,
                       capsize=3, edgecolor='black')
        bars2 = ax.bar(x + width/2, ood_means, width, yerr=ood_stds,
                       label='OOD', color='#F44336', alpha=0.7,
                       capsize=3, edgecolor='black')

        ax.set_xticks(x)
        ax.set_xticklabels(algo_names, rotation=30, fontsize=8)
        ax.set_title(env.replace('-v2', ''), fontsize=12)
        if ax == axes[0]:
            ax.set_ylabel('Mean Episode Reward', fontsize=10)
            ax.legend(fontsize=9)

        # Annotate degradation percentage
        for i in range(len(algo_names)):
            if id_means[i] > 0:
                degradation = (id_means[i] - ood_means[i]) / abs(id_means[i]) * 100
                y_pos = max(id_means[i], ood_means[i]) + max(id_stds[i], ood_stds[i])
                color = 'red' if degradation > 20 else 'orange' if degradation > 10 else 'green'
                ax.text(x[i], y_pos * 1.02, f"↓{degradation:.0f}%",
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=str, default="jax_experiments/results")
    parser.add_argument("--envs", nargs="+", default=["HalfCheetah-v2"])
    parser.add_argument("--algos", nargs="+", default=["bapr", "escp", "resac"])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # For now, display placeholder — real eval requires loading saved models
    print("ID/OOD evaluation requires saved model checkpoints.")
    print("To enable: add model saving to train.py (save agent params at --save_interval)")
    print("Then this script will load checkpoints and eval on ID/OOD task sets.")


if __name__ == "__main__":
    main()
