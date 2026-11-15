"""Money plot for §5.7 Q3/Q4: BOCD detection + adaptive conservatism dynamics.

Generates a 4-panel figure for one representative run showing:
  1. True mode (shaded bg) + eval reward (line)
  2. BOCD belief entropy (smaller = more confident in current mode)
  3. Alpha (entropy temp) — proxy for adaptive conservatism activity
  4. Effective belief window (run-length-equivalent)

Also computes per-switch detection delay table (B).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Choose representative run (Ant winner config)
ROOT = "/home/erzhu419/mine_code/BAPR/jax_experiments"
RUN = f"{ROOT}/results_no2_full/bapr_no2_Ant_s2/logs"
OUT_FIG = "/home/erzhu419/mine_code/BAPR/paper/fig_bocd_dynamics.pdf"
OUT_TABLE = "/home/erzhu419/mine_code/BAPR/paper/detection_delay_summary.txt"


def load(name):
    return np.load(f"{RUN}/{name}.npy")


def main():
    eval_R = load("eval_reward")
    iters = load("iteration")
    mode_id = load("mode_id")
    belief_H = load("belief_entropy")
    alpha = load("alpha")
    eff_win = load("effective_window")

    # Per-iter signals (mode_id, belief_H, alpha, eff_win, iters)
    n = min(len(iters), len(mode_id), len(belief_H), len(alpha), len(eff_win))
    x = iters[:n]
    mode_id = mode_id[:n]
    belief_H = belief_H[:n]
    alpha = alpha[:n]
    eff_win = eff_win[:n]
    # eval_R is logged every log_interval=5 iters → independent x axis
    log_interval = max(1, n // max(len(eval_R), 1))
    x_eval = np.arange(len(eval_R)) * log_interval

    # ===== Money plot =====
    fig, axes = plt.subplots(4, 1, figsize=(7.0, 6.0), sharex=True,
                             gridspec_kw={"hspace": 0.18})
    K = int(mode_id.max()) + 1
    cmap = plt.get_cmap("Set2")

    # Panel 1: mode shading + eval reward
    ax = axes[0]
    switches = np.where(np.diff(mode_id) != 0)[0]
    bounds = np.concatenate([[0], switches + 1, [n]])
    for i in range(len(bounds) - 1):
        m = int(mode_id[bounds[i]])
        ax.axvspan(x[bounds[i]], x[min(bounds[i + 1], n - 1)], color=cmap(m / K),
                   alpha=0.25, lw=0)
    ax.plot(x_eval, eval_R, color="black", lw=1.0)
    ax.set_ylabel("Eval R", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title("BAPR on Ant-v2 discrete-mode (seed 2): "
                 "true mode (background) + return + BOCD signals",
                 fontsize=9)

    # Panel 2: belief entropy (low = confident)
    ax = axes[1]
    for i in range(len(bounds) - 1):
        m = int(mode_id[bounds[i]])
        ax.axvspan(x[bounds[i]], x[min(bounds[i + 1], n - 1)], color=cmap(m / K),
                   alpha=0.15, lw=0)
    ax.plot(x, belief_H, color="C3", lw=0.9, label=r"$H[\rho_t]$")
    ax.set_ylabel("Belief H", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(loc="upper right", fontsize=7)

    # Panel 3: alpha (entropy temperature) — proxy for conservatism modulation
    ax = axes[2]
    for i in range(len(bounds) - 1):
        m = int(mode_id[bounds[i]])
        ax.axvspan(x[bounds[i]], x[min(bounds[i + 1], n - 1)], color=cmap(m / K),
                   alpha=0.15, lw=0)
    ax.plot(x, alpha, color="C0", lw=0.9, label=r"$\alpha$ (entropy temp)")
    ax.set_ylabel(r"$\alpha$", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(loc="upper right", fontsize=7)

    # Panel 4: effective belief window
    ax = axes[3]
    for i in range(len(bounds) - 1):
        m = int(mode_id[bounds[i]])
        ax.axvspan(x[bounds[i]], x[min(bounds[i + 1], n - 1)], color=cmap(m / K),
                   alpha=0.15, lw=0)
    ax.plot(x, eff_win, color="C2", lw=0.9, label="effective window")
    ax.set_ylabel("Eff. window", fontsize=9)
    ax.set_xlabel("Iteration", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(loc="upper right", fontsize=7)

    plt.tight_layout()
    plt.savefig(OUT_FIG, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT_FIG}")

    # ===== Detection delay table =====
    # For each switch event in mode_id, find when belief_H exceeds baseline+threshold.
    # Baseline = median belief_H in last 30 iters before switch.
    delays = []
    h_base = belief_H[: max(switches[0], 30)].std() if len(switches) > 0 else 0.05
    for sw in switches:
        if sw < 5 or sw + 50 >= n:
            continue
        local_baseline = belief_H[max(0, sw - 30): sw].mean()
        thr = local_baseline + 1.5 * (belief_H[max(0, sw - 30): sw].std() + 0.01)
        # Find first iter post-switch where belief_H > thr
        post = belief_H[sw: min(sw + 50, n)]
        hits = np.where(post > thr)[0]
        if len(hits):
            delays.append(int(hits[0]))
    delays = np.asarray(delays, dtype=float)
    if len(delays) == 0:
        print("No detection events captured.")
        return

    summary = (
        f"Detection delay summary (run: bapr_no2_Ant_s2, {len(switches)} mode switches)\n"
        f"  Detected events: {len(delays)} / {len(switches)}\n"
        f"  Median detection delay: {int(np.median(delays))} iters\n"
        f"  Mean delay:             {delays.mean():.1f} iters\n"
        f"  90th-percentile delay:  {int(np.quantile(delays, 0.9))} iters\n"
        f"  Detection rate:         {len(delays) / len(switches):.0%}\n"
    )
    print(summary)
    with open(OUT_TABLE, "w") as f:
        f.write(summary)
    print(f"Saved {OUT_TABLE}")


if __name__ == "__main__":
    main()
