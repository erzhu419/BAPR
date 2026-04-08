"""Multi-seed aggregation: plot mean ± 95% CI across seeds with bootstrap.

Usage:
    python -m jax_experiments.analysis.multiseed_plot \
        --results_root jax_experiments/results \
        --envs HalfCheetah-v2 Hopper-v2 Walker2d-v2 Ant-v2 \
        --algos bapr escp resac sac \
        --output paper/fig_multiseed_curves.pdf
"""
import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


ALGO_COLORS = {
    "bapr": "#2196F3", "escp": "#FF9800", "resac": "#4CAF50",
    "sac": "#9E9E9E", "bapr_no_bocd": "#E91E63",
    "bapr_no_rmdm": "#9C27B0", "bapr_no_adapt_beta": "#795548",
    "bapr_fixed_decay": "#FF5722", "bad_bapr": "#F44336",
}

ALGO_LABELS = {
    "bapr": "BAPR (Ours)", "escp": "ESCP", "resac": "RE-SAC",
    "sac": "SAC", "bapr_no_bocd": "BAPR w/o BOCD",
    "bapr_no_rmdm": "BAPR w/o RMDM", "bapr_no_adapt_beta": "BAPR w/o Adapt-β",
    "bapr_fixed_decay": "BAPR-FixedDecay", "bad_bapr": "Bad-BAPR",
}


def bootstrap_ci(data, n_bootstrap=1000, ci=0.95, axis=0):
    """Compute bootstrap confidence interval along axis."""
    n = data.shape[axis]
    if n <= 1:
        return data.mean(axis=axis), data.mean(axis=axis)

    rng = np.random.RandomState(42)
    boot_means = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        boot_means.append(np.take(data, idx, axis=axis).mean(axis=axis))
    boot_means = np.array(boot_means)

    alpha = (1 - ci) / 2
    lower = np.percentile(boot_means, 100 * alpha, axis=0)
    upper = np.percentile(boot_means, 100 * (1 - alpha), axis=0)
    return lower, upper


def find_seed_runs(results_root, algo, env):
    """Find all seed directories for a given algo/env combo.

    Uses exact algo prefix to avoid e.g. 'bapr' matching 'bapr_no_bocd'.
    Pattern: algo_env_seed/logs or algo_env/logs.
    """
    # algo_env_<seed>/logs — seed is numeric
    seed_pattern = os.path.join(results_root, f"{algo}_{env}_[0-9]*/logs")
    # algo_env/logs (no seed suffix)
    exact_pattern = os.path.join(results_root, f"{algo}_{env}", "logs")
    dirs = glob.glob(seed_pattern) + glob.glob(exact_pattern)
    return sorted(set(dirs))


def load_aligned_metric(log_dirs, metric_name, max_len=None):
    """Load metric from multiple seed dirs and align to shortest length."""
    arrays = []
    for ld in log_dirs:
        path = os.path.join(ld, f"{metric_name}.npy")
        if os.path.exists(path):
            arr = np.load(path).astype(float)
            if len(arr) > 0:
                arrays.append(arr)

    if not arrays:
        return None

    min_len = min(len(a) for a in arrays)
    if max_len:
        min_len = min(min_len, max_len)
    return np.stack([a[:min_len] for a in arrays])  # (n_seeds, T)


def plot_multiseed(results_root, envs, algos, output=None,
                   metric="eval_reward", figsize=None, smoothing=5):
    """Plot multi-seed comparison with bootstrap CI."""
    n_envs = len(envs)
    if figsize is None:
        figsize = (4.5 * n_envs, 4)

    fig, axes = plt.subplots(1, n_envs, figsize=figsize, sharey=False)
    if n_envs == 1:
        axes = [axes]

    for ax, env in zip(axes, envs):
        for algo in algos:
            log_dirs = find_seed_runs(results_root, algo, env)
            if not log_dirs:
                continue

            data = load_aligned_metric(log_dirs, metric)
            if data is None:
                continue

            n_seeds, T = data.shape
            x = np.arange(T)

            # Smooth
            if smoothing > 1 and T > smoothing:
                kernel = np.ones(smoothing) / smoothing
                smoothed = np.apply_along_axis(
                    lambda m: np.convolve(m, kernel, mode='same'), 1, data)
            else:
                smoothed = data

            mean = smoothed.mean(axis=0)
            color = ALGO_COLORS.get(algo, '#666666')
            label = ALGO_LABELS.get(algo, algo)

            if n_seeds >= 3:
                lower, upper = bootstrap_ci(smoothed, ci=0.95)
                ax.fill_between(x, lower, upper, alpha=0.15, color=color)
                label += f" ({n_seeds} seeds)"
            elif n_seeds == 2:
                std = smoothed.std(axis=0)
                ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)
                label += f" ({n_seeds} seeds)"

            ax.plot(x, mean, color=color, linewidth=1.5, label=label)

        env_short = env.replace('-v2', '').replace('-v4', '')
        ax.set_title(env_short, fontsize=12, fontweight='bold')
        ax.set_xlabel('Eval Point', fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel('Evaluation Reward', fontsize=10)
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output:
        os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches='tight')
        fig.savefig(output.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
        print(f"Saved multi-seed plot to: {output}")
    else:
        plt.show()

    plt.close(fig)


def print_final_table(results_root, envs, algos, metric="eval_reward"):
    """Print LaTeX-formatted results table with mean ± std."""
    print("\n" + "=" * 80)
    print("Final Performance Table (last 20% of training)")
    print("=" * 80)

    header = "Algorithm " + " ".join(f"& {e.replace('-v2',''):<16}" for e in envs) + r" \\"
    print(header)
    print(r"\midrule")

    for algo in algos:
        row = f"{ALGO_LABELS.get(algo, algo):<20}"
        for env in envs:
            log_dirs = find_seed_runs(results_root, algo, env)
            data = load_aligned_metric(log_dirs, metric) if log_dirs else None

            if data is not None:
                n_seeds, T = data.shape
                n_last = max(1, T // 5)
                final_vals = data[:, -n_last:].mean(axis=1)
                mean = final_vals.mean()
                std = final_vals.std()
                row += f" & {mean:.0f} ± {std:.0f}"
            else:
                row += " & ---"

        row += r" \\"
        print(row)

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=str, default="jax_experiments/results")
    parser.add_argument("--envs", nargs="+",
                        default=["HalfCheetah-v2", "Hopper-v2", "Walker2d-v2", "Ant-v2"])
    parser.add_argument("--algos", nargs="+",
                        default=["bapr", "escp", "resac"])
    parser.add_argument("--metric", type=str, default="eval_reward")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--table", action="store_true", help="Print results table")
    args = parser.parse_args()

    if args.table:
        print_final_table(args.results_root, args.envs, args.algos, args.metric)
    else:
        plot_multiseed(args.results_root, args.envs, args.algos, args.output,
                      args.metric)


if __name__ == "__main__":
    main()
