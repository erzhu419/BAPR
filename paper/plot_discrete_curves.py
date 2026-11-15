"""Generate Fig. 2 (training curves on the discrete-regime benchmark).

Reads logs from `jax_experiments/archive_paper/` and produces a 2-panel
figure: (a) HalfCheetah N=5, (b) Ant N=3, BAPR-regime vs ESCP, with
mean + 95% bootstrap CI band across seeds.

Output: paper/fig_discrete_curves.pdf
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ARCHIVE = Path(__file__).parent.parent / "jax_experiments" / "archive_paper"

HC_BAPR = [
    ARCHIVE / "day4_gpu1" / "bapr_regime_HC_0",          # seed 0
    ARCHIVE / "day4_gpu1" / "bapr_regime_HC_tune_p05",   # seed 0 tune (winner) — same seed=0
    # Note: bapr_regime_HC_0 was the original p_scale=2.0 (killed); tune_p05 is seed=0 final.
    # Day 4 used 'bapr_regime_HC_tune_p05' for seed 0, 'bapr_regime_HC_tune_p05_s1' for seed 1.
]
# Correct list: seed 0 = day4_gpu1/bapr_regime_HC_tune_p05; seed 1 = ..._s1; seeds 2/3/4 in day6
HC_BAPR_SEEDS = [
    (0, ARCHIVE / "day4_gpu1" / "bapr_regime_HC_tune_p05"),
    (1, ARCHIVE / "day4_gpu1" / "bapr_regime_HC_tune_p05_s1"),
    (2, ARCHIVE / "day6_gpu1" / "bapr_regime_HC_p05_s2"),
    (3, ARCHIVE / "day6_gpu1" / "bapr_regime_HC_p05_s3"),
    (4, ARCHIVE / "day6_gpu1" / "bapr_regime_HC_p05_s4"),
]
HC_ESCP_SEEDS = [
    (0, ARCHIVE / "day4_gpu2" / "escp_HC_0"),
    (1, ARCHIVE / "day4_gpu2" / "escp_HC_1"),
    (2, ARCHIVE / "day6_gpu1" / "escp_HC_s2"),
    (3, ARCHIVE / "day6_gpu1" / "escp_HC_s3"),
    (4, ARCHIVE / "day6_gpu1" / "escp_HC_s4"),
]
ANT_BAPR_SEEDS = [
    (0, ARCHIVE / "day5_gpu2" / "bapr_regime_Ant_p05_s0"),
    (1, ARCHIVE / "day6_gpu2" / "bapr_regime_Ant_p05_s1"),
    (2, ARCHIVE / "day6_gpu2" / "bapr_regime_Ant_p05_s2"),
]
ANT_ESCP_SEEDS = [
    (0, ARCHIVE / "day5_gpu2" / "escp_Ant_s0"),
    (1, ARCHIVE / "day6_gpu2" / "escp_Ant_s1"),
    (2, ARCHIVE / "day6_gpu2" / "escp_Ant_s2"),
]


def load_run(run_dir: Path, log_interval: int = 5):
    """Return (eval_iters, evals) arrays.

    iteration.npy is per-train-iter (length 1500), eval_reward.npy is
    per-eval (length 300, since log_interval=5). Reconstruct the iter
    axis for evals as 0, 5, 10, ..., 5*(N-1).
    """
    ev = np.load(run_dir / "logs" / "eval_reward.npy")
    eval_iters = np.arange(len(ev)) * log_interval
    return eval_iters, ev


def aggregate(seed_runs):
    """Stack runs onto a common iter grid; return (iters, mean, lo, hi).
    Uses 95% bootstrap CI from per-iter cross-seed values."""
    raw = [load_run(d) for _, d in seed_runs]
    common_iter = raw[0][0]
    matrix = []
    for it, ev in raw:
        # Truncate / interpolate to common_iter length
        m = min(len(common_iter), len(ev))
        matrix.append(ev[:m])
    common_iter = common_iter[:m]
    matrix = np.stack(matrix)  # (n_seeds, n_iters)

    mean = matrix.mean(axis=0)
    # 95% CI via bootstrap over seeds at each iter
    rng = np.random.default_rng(0)
    n_boot = 1000
    n_seeds = matrix.shape[0]
    boot_means = np.empty((n_boot, matrix.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, n_seeds, size=n_seeds)
        boot_means[b] = matrix[idx].mean(axis=0)
    lo = np.percentile(boot_means, 2.5, axis=0)
    hi = np.percentile(boot_means, 97.5, axis=0)
    return common_iter, mean, lo, hi


def smooth(y, k=10):
    """Simple moving average for visual clarity."""
    if k <= 1:
        return y
    pad = k // 2
    y_pad = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    kernel = np.ones(k) / k
    return np.convolve(y_pad, kernel, mode="valid")[: len(y)]


def thousands(x, _):
    if x >= 1000:
        v = x / 1000
        return f"{v:.1f}k" if v != int(v) else f"{int(v)}k"
    return f"{int(x)}"


fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

for ax, (env, bapr, escp) in zip(
    axes,
    [
        ("HalfCheetah, $K{=}4$ modes, $N{=}5$ seeds", HC_BAPR_SEEDS, HC_ESCP_SEEDS),
        ("Ant, $K{=}4$ modes, $N{=}3$ seeds",         ANT_BAPR_SEEDS, ANT_ESCP_SEEDS),
    ],
):
    for label, seeds, color in [
        ("ESCP",                        escp, "#1f77b4"),
        ("BAPR (joint $b(h,z)$, ours)", bapr, "#d62728"),
    ]:
        it, mean, lo, hi = aggregate(seeds)
        m = smooth(mean, k=15)
        l = smooth(lo,   k=15)
        h = smooth(hi,   k=15)
        ax.plot(it, m, color=color, lw=1.6, label=label)
        ax.fill_between(it, l, h, color=color, alpha=0.18)

    ax.set_title(env, fontsize=10)
    ax.set_xlabel("training iteration")
    ax.set_ylabel("evaluation return (avg over $K$ test modes)")
    ax.xaxis.set_major_formatter(FuncFormatter(thousands))
    ax.yaxis.set_major_formatter(FuncFormatter(thousands))
    ax.set_xticks([0, 300, 600, 900, 1200, 1500])
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

plt.tight_layout()
out_path = Path(__file__).parent / "fig_discrete_curves.pdf"
plt.savefig(out_path, bbox_inches="tight")
print(f"saved {out_path}")
