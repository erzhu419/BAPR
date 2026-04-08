"""Adaptation speed / regime recovery time analysis.

For each mode switch, measures how many iterations it takes for the agent's
performance to recover to within X% of its pre-switch level.

Usage:
    python -m jax_experiments.analysis.adaptation_speed \
        --results_root jax_experiments/results \
        --env HalfCheetah-v2 --output paper/fig_adaptation_speed.pdf
"""
import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt


def load_metric(log_dir, name):
    path = os.path.join(log_dir, f"{name}.npy")
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)
    return None


def compute_recovery_times(eval_reward, mode_ids, log_interval=5,
                           recovery_threshold=0.9, window=10):
    """Compute recovery time after each mode switch.

    Args:
        eval_reward: evaluation rewards (logged every log_interval iters)
        mode_ids: per-iteration mode IDs
        log_interval: eval frequency
        recovery_threshold: fraction of pre-switch reward to declare recovery
        window: smoothing window for reward

    Returns:
        list of (switch_iter, recovery_steps) pairs
    """
    n_iters = len(mode_ids)
    switches = np.where(np.diff(mode_ids) != 0)[0] + 1

    # Map iteration -> eval index
    n_eval = len(eval_reward)
    eval_iters = np.arange(0, n_eval * log_interval, log_interval)[:n_eval]

    # Smooth eval reward
    if len(eval_reward) > window:
        smoothed = np.convolve(eval_reward, np.ones(window)/window, mode='same')
    else:
        smoothed = eval_reward

    recovery_data = []

    for sw in switches:
        # Pre-switch performance: mean of last `window` evals before switch
        pre_eval_idx = np.searchsorted(eval_iters, sw) - 1
        if pre_eval_idx < 1:
            continue

        pre_start = max(0, pre_eval_idx - window)
        pre_perf = np.mean(smoothed[pre_start:pre_eval_idx])

        if pre_perf <= 0:
            continue  # skip if negative baseline

        target = recovery_threshold * pre_perf

        # Post-switch: find first eval point exceeding target
        post_eval_idx = np.searchsorted(eval_iters, sw)
        recovered = False
        for j in range(post_eval_idx, n_eval):
            if smoothed[j] >= target:
                recovery_iters = eval_iters[j] - sw
                recovery_data.append((int(sw), int(recovery_iters)))
                recovered = True
                break

        if not recovered:
            # Never recovered
            remaining = (n_iters - sw)
            recovery_data.append((int(sw), int(remaining)))

    return recovery_data


def find_runs(results_root, env, algo):
    """Find all result directories matching algo/env.

    Uses exact algo prefix + numeric seed to avoid e.g. 'bapr'
    matching 'bapr_no_bocd'.
    """
    # algo_env_<digit>*/logs — matches algo_env_0, algo_env_8_ablation, etc.
    pattern = os.path.join(results_root, f"{algo}_{env}_[0-9]*/logs")
    exact = os.path.join(results_root, f"{algo}_{env}", "logs")
    results = sorted(set(glob.glob(pattern) + glob.glob(exact)))
    return results


def plot_adaptation_speed(results_root, env, algos=None, output=None,
                          log_interval=5, figsize=(12, 5)):
    """Compare adaptation speed across algorithms."""
    if algos is None:
        algos = ["bapr", "escp", "resac", "sac",
                 "bapr_no_bocd", "bapr_fixed_decay"]

    algo_colors = {
        "bapr": "#2196F3", "escp": "#FF9800", "resac": "#4CAF50",
        "sac": "#9E9E9E", "bapr_no_bocd": "#E91E63",
        "bapr_no_rmdm": "#9C27B0", "bapr_no_adapt_beta": "#795548",
        "bapr_fixed_decay": "#FF5722", "bad_bapr": "#F44336",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    all_recovery_data = {}

    for algo in algos:
        log_dirs = find_runs(results_root, env, algo)
        if not log_dirs:
            continue

        for log_dir in log_dirs:
            eval_rew = load_metric(log_dir, "eval_reward")
            mode_ids = load_metric(log_dir, "mode_id")

            if eval_rew is None or mode_ids is None:
                continue

            recoveries = compute_recovery_times(eval_rew, mode_ids, log_interval)
            if recoveries:
                if algo not in all_recovery_data:
                    all_recovery_data[algo] = []
                all_recovery_data[algo].extend([r[1] for r in recoveries])

    # Plot 1: Box plot of recovery times
    if all_recovery_data:
        algo_names = list(all_recovery_data.keys())
        recovery_vals = [all_recovery_data[a] for a in algo_names]
        colors = [algo_colors.get(a, '#666666') for a in algo_names]

        bp = ax1.boxplot(recovery_vals, labels=algo_names, patch_artist=True,
                        showmeans=True, meanprops=dict(marker='D', markerfacecolor='white'))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax1.set_ylabel('Recovery Iterations', fontsize=11)
        ax1.set_title(f'Adaptation Speed — {env}', fontsize=12)
        ax1.tick_params(axis='x', rotation=30)

        # Plot 2: Mean recovery time bar chart with error bars
        means = [np.mean(v) for v in recovery_vals]
        stds = [np.std(v) / max(np.sqrt(len(v)), 1) for v in recovery_vals]

        bars = ax2.bar(algo_names, means, yerr=stds, capsize=4,
                      color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Mean Recovery Iterations', fontsize=11)
        ax2.set_title(f'Mean Adaptation Speed — {env}', fontsize=12)
        ax2.tick_params(axis='x', rotation=30)

    plt.tight_layout()

    if output:
        os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches='tight')
        fig.savefig(output.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
        print(f"Saved adaptation speed plot to: {output}")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=str, default="jax_experiments/results")
    parser.add_argument("--env", type=str, default="HalfCheetah-v2")
    parser.add_argument("--algos", nargs="+", default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    plot_adaptation_speed(args.results_root, args.env, args.algos, args.output)


if __name__ == "__main__":
    main()
