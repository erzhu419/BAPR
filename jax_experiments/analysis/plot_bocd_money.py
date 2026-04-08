"""BOCD Money Plot: the 'soul figure' of the BAPR paper.

Generates a time-aligned 3-panel visualization:
  Top:    True environment mode (color-coded background bands)
  Middle: BOCD run-length posterior ρ(h) heatmap
  Bottom: Adaptive conservatism β_eff (or λ_w) over training iterations

Usage:
    python -m jax_experiments.analysis.plot_bocd_money \
        --log_dir jax_experiments/results/bapr_HalfCheetah-v2_8/logs \
        --output paper/fig_bocd_money_plot.pdf
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle


def load_metric(log_dir, name, default=None):
    path = os.path.join(log_dir, f"{name}.npy")
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)
    return default


def plot_bocd_money(log_dir, output_path=None, figsize=(14, 8)):
    """Generate the BOCD dynamics visualization (3-panel money plot)."""

    # Load metrics
    mode_ids = load_metric(log_dir, "mode_id")
    iterations = load_metric(log_dir, "iteration")
    weighted_lambda = load_metric(log_dir, "weighted_lambda")
    belief_entropy = load_metric(log_dir, "belief_entropy")
    effective_window = load_metric(log_dir, "effective_window")
    eval_reward = load_metric(log_dir, "eval_reward")
    bocd_belief = load_metric(log_dir, "bocd_belief")  # may be array of arrays
    surprise_r = load_metric(log_dir, "surprise_r")
    surprise_q = load_metric(log_dir, "surprise_q")

    if mode_ids is None:
        raise FileNotFoundError(f"No mode_id.npy in {log_dir}")

    n_iters = len(mode_ids)
    iters = np.arange(n_iters)

    # Detect mode switches
    mode_switches = np.where(np.diff(mode_ids) != 0)[0] + 1

    # Unique modes and color map
    unique_modes = np.unique(mode_ids)
    n_modes = len(unique_modes)
    cmap = plt.cm.get_cmap('tab20', max(n_modes, 20))
    mode_colors = {m: cmap(i % 20) for i, m in enumerate(unique_modes)}

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True,
                             gridspec_kw={'height_ratios': [1.5, 2.5, 1.5]})

    # ---- Panel 1: Mode timeline + eval reward ----
    ax1 = axes[0]
    # Draw mode background bands
    starts = [0] + list(mode_switches)
    ends = list(mode_switches) + [n_iters]
    for s, e in zip(starts, ends):
        m = mode_ids[s]
        color = mode_colors[m]
        ax1.axvspan(s, e, alpha=0.3, color=color, linewidth=0)

    # Overlay eval reward
    if eval_reward is not None and len(eval_reward) > 0:
        # eval_reward is logged every log_interval iterations
        n_eval = len(eval_reward)
        if iterations is not None and len(iterations) >= n_iters:
            log_interval = max(1, n_iters // n_eval) if n_eval > 0 else 5
        else:
            log_interval = 5
        eval_iters = np.arange(0, n_eval * log_interval, log_interval)[:n_eval]
        ax1.plot(eval_iters, eval_reward, color='black', linewidth=1.5,
                 label='Eval Reward', zorder=5)
        ax1.set_ylabel('Eval Reward', fontsize=10)
        ax1.legend(loc='upper left', fontsize=8)

    # Mark mode switches with vertical lines
    for sw in mode_switches:
        ax1.axvline(sw, color='red', linewidth=0.5, alpha=0.6, linestyle='--')

    ax1.set_title('Environment Mode Timeline + Evaluation Reward', fontsize=12)

    # ---- Panel 2: BOCD belief heatmap ----
    ax2 = axes[1]
    if bocd_belief is not None and len(bocd_belief) > 0:
        # bocd_belief is an array of belief vectors
        try:
            belief_matrix = np.stack(bocd_belief)  # (T, H)
            if belief_matrix.ndim == 2:
                im = ax2.imshow(belief_matrix.T, aspect='auto', origin='lower',
                               extent=[0, len(belief_matrix), 0, belief_matrix.shape[1]],
                               cmap='hot', interpolation='nearest',
                               norm=mcolors.PowerNorm(gamma=0.5))
                fig.colorbar(im, ax=ax2, label='ρ(h)', shrink=0.8)
                ax2.set_ylabel('Run-length h', fontsize=10)
        except (ValueError, TypeError):
            ax2.text(0.5, 0.5, 'Belief data format error',
                    transform=ax2.transAxes, ha='center')
    elif effective_window is not None:
        # Fallback: plot effective window as a line
        trim = min(len(effective_window), n_iters)
        ax2.plot(np.arange(trim), effective_window[:trim],
                color='darkorange', linewidth=1.0)
        ax2.set_ylabel('Effective Window E[h]', fontsize=10)
        ax2.fill_between(np.arange(trim), 0, effective_window[:trim],
                        alpha=0.3, color='darkorange')
    else:
        ax2.text(0.5, 0.5, 'No belief data available',
                transform=ax2.transAxes, ha='center')

    for sw in mode_switches:
        ax2.axvline(sw, color='red', linewidth=0.5, alpha=0.6, linestyle='--')
    ax2.set_title('BOCD Run-Length Posterior ρ(h)', fontsize=12)

    # ---- Panel 3: Adaptive conservatism λ_w ----
    ax3 = axes[2]
    if weighted_lambda is not None:
        trim = min(len(weighted_lambda), n_iters)
        ax3.plot(np.arange(trim), weighted_lambda[:trim],
                color='steelblue', linewidth=1.0, label='λ_w')
        ax3.fill_between(np.arange(trim), 0, weighted_lambda[:trim],
                        alpha=0.2, color='steelblue')
        ax3.set_ylabel('λ_w (penalty weight)', fontsize=10)
        ax3.legend(loc='upper right', fontsize=8)

    # Highlight performance gating activations (if logged)
    perf_gate = load_metric(log_dir, "perf_gate_active")
    if perf_gate is not None:
        trim_pg = min(len(perf_gate), n_iters)
        gate_iters = np.where(np.array(perf_gate[:trim_pg], dtype=bool))[0]
        for gi in gate_iters:
            ax3.axvspan(gi - 0.5, gi + 0.5, alpha=0.3, color='red', linewidth=0)
        if len(gate_iters) > 0:
            ax3.plot([], [], color='red', alpha=0.3, linewidth=6, label='Perf Gate')
            ax3.legend(loc='upper right', fontsize=8)

    if belief_entropy is not None:
        ax3b = ax3.twinx()
        trim = min(len(belief_entropy), n_iters)
        ax3b.plot(np.arange(trim), belief_entropy[:trim],
                 color='green', linewidth=0.8, alpha=0.6, label='Belief Entropy')
        ax3b.set_ylabel('Belief Entropy', fontsize=9, color='green')
        ax3b.legend(loc='upper left', fontsize=8)

    for sw in mode_switches:
        ax3.axvline(sw, color='red', linewidth=0.5, alpha=0.6, linestyle='--')
    ax3.set_xlabel('Training Iteration', fontsize=10)
    ax3.set_title('Adaptive Conservatism (Bayesian Amnesia)', fontsize=12)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved BOCD money plot to: {output_path}")
        # Also save PNG
        png_path = output_path.replace('.pdf', '.png')
        fig.savefig(png_path, dpi=200, bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)
    return fig


def main():
    parser = argparse.ArgumentParser(description="BOCD Money Plot Generator")
    parser.add_argument("--log_dir", type=str, required=True,
                        help="Path to logs/ directory of a BAPR run")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PDF path (default: show)")
    parser.add_argument("--figsize_w", type=float, default=14)
    parser.add_argument("--figsize_h", type=float, default=8)
    args = parser.parse_args()

    plot_bocd_money(args.log_dir, args.output,
                   figsize=(args.figsize_w, args.figsize_h))


if __name__ == "__main__":
    main()
