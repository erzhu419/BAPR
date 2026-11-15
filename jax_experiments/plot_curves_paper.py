#!/usr/bin/env python3
import re, os, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

ALGOS  = ["resac", "escp", "bapr"]
ALL_ENVS = ["Hopper-v2", "Walker2d-v2", "HalfCheetah-v2", "Ant-v2"]
COLORS = {"resac": "#1f77b4", "escp": "#ff7f0e", "bapr": "#2ca02c"}
LABELS = {"resac": "RE-SAC", "escp": "ESCP", "bapr": "BAPR"}

BG   = "white"
CARD = "white"
GRID = "#E0E0E0"

def parse_log(path):
    iters, rewards, eval_iters, eval_rewards = [], [], [], []
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            if not line.startswith("Iter"):
                continue
            m_iter = re.search(r"Iter\s+(\d+)", line)
            m_rew  = re.search(r"Reward:\s+([\d.]+)", line)
            m_eval = re.search(r"Eval:\s+([\d.]+)", line)
            if m_iter and m_rew:
                iters.append(int(m_iter.group(1)))
                rewards.append(float(m_rew.group(1)))
            if m_iter and m_eval:
                eval_iters.append(int(m_iter.group(1)))
                eval_rewards.append(float(m_eval.group(1)))
    if not iters:
        return None
    return {"iters": np.array(iters), "rewards": np.array(rewards), "eval_iters": np.array(eval_iters), "eval_rewards": np.array(eval_rewards)}

def smooth(x, w=20):
    if len(x) <= w:
        return x, np.arange(len(x))
    kern = np.ones(w) / w
    return np.convolve(x, kern, 'valid'), np.arange(w - 1, len(x))

def find_log(log_dir, algo, env, seed=8):
    import glob, os
    env_short = env.split('-')[0]
    patterns = [os.path.join(log_dir, f"{algo}_{env_short}*.log"), os.path.join(log_dir, f"{algo}*{env_short}*.log"), os.path.join(log_dir, f"{algo}_{env}_{seed}*.log")]
    matches = []
    for pat in patterns: matches.extend(glob.glob(pat))
    if matches: return max(set(matches), key=os.path.getmtime)
    return os.path.join(log_dir, f"{algo}_{env}_{seed}.log")

def plot(log_dir, envs, out_path, seed=8, max_iters=2000):
    n_envs = len(envs)
    fig = plt.figure(figsize=(5.5 * n_envs, 8.5), facecolor=BG)

    gs = gridspec.GridSpec(2, n_envs, figure=fig, top=0.92, bottom=0.09, left=max(0.07, 0.04 + 0.03 / n_envs), right=0.98, hspace=0.35, wspace=0.25)
    ROW_YLABEL = ["Training Reward (smoothed)", "Eval Reward"]
    
    all_data = {}
    for env in envs:
        for algo in ALGOS:
            path = find_log(log_dir, algo, env, seed)
            all_data[(algo, env)] = parse_log(path)

    for col, env in enumerate(envs):
        for row in range(2):
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor(CARD)
            ax.tick_params(colors='black', labelsize=10)
            for sp in ax.spines.values():
                sp.set_color('#888')
                sp.set_linewidth(1.0)
            ax.grid(True, color=GRID, linewidth=0.8, linestyle='--')
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

            plotted = False
            for algo in ALGOS:
                d = all_data[(algo, env)]
                if d is None: continue
                c, lbl = COLORS[algo], LABELS[algo]
                
                if row == 0 and len(d['iters']) > 5:
                    sw = min(30, max(5, len(d['rewards']) // 8))
                    sm, idx = smooth(d['rewards'], sw)
                    xs = d['iters'][idx]
                    ax.plot(xs, sm, color=c, lw=2.5, label=lbl, alpha=0.9, zorder=3)
                    ax.fill_between(xs, sm * 0.88, sm * 1.12, color=c, alpha=0.15, zorder=2)
                    plotted = True
                elif row == 1 and len(d['eval_iters']) >= 2:
                    ax.plot(d['eval_iters'], d['eval_rewards'], color=c, lw=2.0, marker='o', ms=5, label=lbl, alpha=0.9, zorder=3, markeredgecolor='white', markeredgewidth=0.5)
                    plotted = True

            if row == 0:
                env_short = env.replace('-v2', '').replace('-v4', '').replace('-v5', '')
                ax.set_title(env_short, color='black', fontsize=16, fontweight='bold', pad=12)
            if col == 0:
                ax.set_ylabel(ROW_YLABEL[row], color='black', fontsize=12, fontweight='bold')
            if row == 1:
                ax.set_xlabel("Iteration", color='black', fontsize=12, fontweight='bold')
                
            if plotted:
                ax.legend(fontsize=10, loc='best', framealpha=0.9, edgecolor='#CCC', labelcolor='black')
            else:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes, color='black')

    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=BG)
    print(f"Saved -> {out_path}")

if __name__ == "__main__":
    os.makedirs("jax_experiments/results", exist_ok=True)
    plot("jax_experiments/logs", ALL_ENVS, "jax_experiments/results/training_curves_paper.png", seed=8)
