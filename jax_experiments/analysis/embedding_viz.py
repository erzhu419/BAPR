"""t-SNE visualization of context embeddings and ensemble diversity analysis.

Usage:
    python -m jax_experiments.analysis.embedding_viz \
        --log_dir jax_experiments/results/bapr_HalfCheetah-v2_8/logs \
        --output paper/fig_embedding_tsne.pdf
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def load_metric(log_dir, name):
    path = os.path.join(log_dir, f"{name}.npy")
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)
    return None


def plot_embedding_evolution(log_dir, output=None, figsize=(14, 5)):
    """Plot context embedding evolution and ensemble diversity over training."""
    context_emb = load_metric(log_dir, "context_emb")
    mode_ids = load_metric(log_dir, "mode_id")
    q_std = load_metric(log_dir, "q_std_mean")
    eval_reward = load_metric(log_dir, "eval_reward")

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel 1: Context embedding dimensions over time, colored by mode
    if context_emb is not None and mode_ids is not None:
        try:
            emb_matrix = np.stack(context_emb)
            if emb_matrix.ndim == 2:
                n_iters, emb_dim = emb_matrix.shape
                trim = min(n_iters, len(mode_ids))

                unique_modes = np.unique(mode_ids[:trim])
                cmap = plt.cm.get_cmap('tab10', len(unique_modes))

                for d in range(min(emb_dim, 4)):
                    axes[0].plot(range(trim), emb_matrix[:trim, d],
                               alpha=0.6, linewidth=0.8, label=f'dim {d}')

                # Color background by mode
                mode_switches = np.where(np.diff(mode_ids[:trim]) != 0)[0] + 1
                starts = [0] + list(mode_switches)
                ends = list(mode_switches) + [trim]
                for s, e in zip(starts, ends):
                    m = mode_ids[s]
                    idx = np.searchsorted(unique_modes, m)
                    axes[0].axvspan(s, e, alpha=0.1, color=cmap(idx))

                axes[0].set_xlabel('Iteration')
                axes[0].set_ylabel('Embedding Value')
                axes[0].set_title('Context Embedding Evolution')
                axes[0].legend(fontsize=7)
        except (ValueError, TypeError):
            axes[0].text(0.5, 0.5, 'Embedding data error',
                        transform=axes[0].transAxes, ha='center')
    else:
        axes[0].text(0.5, 0.5, 'No embedding data',
                    transform=axes[0].transAxes, ha='center')

    # Panel 2: Q-std (ensemble diversity) over time
    if q_std is not None:
        axes[1].plot(q_std, color='steelblue', linewidth=0.8)
        axes[1].set_xlabel('Training Step')
        axes[1].set_ylabel('Mean Q-ensemble Std')
        axes[1].set_title('Ensemble Diversity (Q-std)')

        if mode_ids is not None:
            trim = min(len(q_std), len(mode_ids))
            mode_switches = np.where(np.diff(mode_ids[:trim]) != 0)[0] + 1
            for sw in mode_switches:
                if sw < len(q_std):
                    axes[1].axvline(sw, color='red', linewidth=0.5, alpha=0.4)
    else:
        axes[1].text(0.5, 0.5, 'No Q-std data',
                    transform=axes[1].transAxes, ha='center')

    # Panel 3: t-SNE of embeddings colored by mode (if enough data)
    if context_emb is not None and mode_ids is not None:
        try:
            emb_matrix = np.stack(context_emb)
            if emb_matrix.ndim == 2 and emb_matrix.shape[0] > 50:
                from sklearn.manifold import TSNE

                # Sample subset for speed
                n_sample = min(2000, emb_matrix.shape[0])
                indices = np.linspace(0, emb_matrix.shape[0]-1, n_sample, dtype=int)
                emb_sub = emb_matrix[indices]
                modes_sub = mode_ids[indices] if len(mode_ids) >= len(indices) else mode_ids[:n_sample]

                if emb_sub.shape[1] >= 2:
                    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
                    emb_2d = tsne.fit_transform(emb_sub)

                    unique_modes = np.unique(modes_sub)
                    cmap = plt.cm.get_cmap('tab20', len(unique_modes))
                    for i, m in enumerate(unique_modes[:20]):  # cap at 20 modes
                        mask = modes_sub == m
                        axes[2].scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                                       c=[cmap(i)], s=5, alpha=0.5, label=f'Mode {m}')

                    axes[2].set_xlabel('t-SNE 1')
                    axes[2].set_ylabel('t-SNE 2')
                    axes[2].set_title('t-SNE of Context Embeddings')
                    # Only show legend if few modes
                    if len(unique_modes) <= 10:
                        axes[2].legend(fontsize=6, markerscale=2)
                else:
                    axes[2].text(0.5, 0.5, 'Embedding dim < 2',
                                transform=axes[2].transAxes, ha='center')
        except ImportError:
            axes[2].text(0.5, 0.5, 'sklearn not installed\nfor t-SNE',
                        transform=axes[2].transAxes, ha='center')
        except Exception as e:
            axes[2].text(0.5, 0.5, f't-SNE error:\n{str(e)[:50]}',
                        transform=axes[2].transAxes, ha='center')
    else:
        axes[2].text(0.5, 0.5, 'No embedding data\nfor t-SNE',
                    transform=axes[2].transAxes, ha='center')

    plt.tight_layout()

    if output:
        os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches='tight')
        fig.savefig(output.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
        print(f"Saved embedding visualization to: {output}")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    plot_embedding_evolution(args.log_dir, args.output)


if __name__ == "__main__":
    main()
