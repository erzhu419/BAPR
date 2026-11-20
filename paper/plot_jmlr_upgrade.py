#!/usr/bin/env python3
"""Generate JMLR upgrade figures for generalized reparameterization (v50) and MuJoCo v5 multi-morphology suite (v51)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper"
V50_JSON = ROOT / "reproducibility/jmlr_brax_mechanism_v50/analysis.json"
V51_JSON = ROOT / "reproducibility/jmlr_mujoco_v51/analysis.json"

COLORS = {
    "ink": "#232530",
    "muted": "#666A78",
    "grid": "#E7E8EE",
    "bapr": "#C95B72",
    "factored": "#A63D57",
    "oracle": "#2E7D5B",
    "sac": "#5E6472",
    "escp": "#6D79A8",
    "rma": "#D08C3F",
    "belief_sac": "#4D8FAC",
    "warn": "#B84A39",
    "uncomp": "#9A9CA6",
}


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.0,
        "axes.titlesize": 7.8,
        "axes.titleweight": "bold",
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "axes.edgecolor": COLORS["ink"],
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "png", "svg"):
        fig.savefig(PAPER_DIR / f"{stem}.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.16, y: float = 1.07) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9.0, fontweight="bold", color=COLORS["ink"], va="bottom", ha="left")


def mean_and_ci(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    n = len(arr)
    m = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    hw = float(student_t.ppf(0.975, n - 1)) * sd / math.sqrt(n) if n > 1 else 0.0
    return m, hw


def plot_v50_mechanism(v50: dict) -> None:
    fig = plt.figure(figsize=(6.5, 3.35))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.28, 1.26, 1.12), wspace=0.46)

    # Panel a: Tempering & Filter Ablations (n=10)
    ax_a = fig.add_subplot(grid[0, 0])
    p1 = v50["part1_primary_ablations"]
    arms_a = [
        ("instantaneous", "No filter", COLORS["warn"]),
        ("untempered_lockin_h0_d100", "$h=0,\\,\\delta=1.0$", COLORS["warn"]),
        ("tempered_h0_d098", "$h=0,\\,\\delta=0.98$", COLORS["belief_sac"]),
        ("sticky_hmm_d100", "HMM ($\\delta=1.0$)", COLORS["escp"]),
        ("single_head_d098", "1-head ($K=1$)", COLORS["sac"]),
        ("bapr_d098", "BAPR ($\\delta=0.98$)", COLORS["bapr"]),
    ]
    y_pos = np.arange(len(arms_a))[::-1]
    for y, (key, _, col) in zip(y_pos, arms_a):
        vals = p1[key]["switching_seed_means"]
        m, hw = mean_and_ci(vals)
        ax_a.errorbar(m, y, xerr=hw, fmt="D", markersize=3.8, color=COLORS["ink"],
                      markerfacecolor=col, markeredgewidth=0.6, capsize=2.2, linewidth=0.9, zorder=4)
        ax_a.scatter(vals, np.full(len(vals), y) + np.linspace(-0.10, 0.10, len(vals)),
                     s=9, color=col, alpha=0.55, zorder=2)
    ax_a.set_yticks(y_pos, [lbl for _, lbl, _ in arms_a])
    ax_a.set_xlabel("Switching return ($n=10$)")
    ax_a.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    ax_a.set_title("Filter & tempering", loc="left", fontsize=7.3)
    add_panel_label(ax_a, "a", x=-0.36)

    # Panel b: M=64 Polarity Hypercube (O(d) Factored vs O(2^d) Joint)
    ax_b = fig.add_subplot(grid[0, 1])
    p2_all = v50["part2_hypercube_64"]["all_15_episodes"]
    p2_1j = v50["part2_hypercube_64"]["single_joint_flip_episodes"]
    arms_b = [
        ("uncompensated_64", "Uncomp.", p2_all, COLORS["uncomp"]),
        ("joint_64", "Joint $O(2^d)$", p2_all, COLORS["escp"]),
        ("factored_64", "Factored $O(d)$", p2_all, COLORS["bapr"]),
        ("factored_64", "1-joint $O(d)$", p2_1j, COLORS["factored"]),
        ("oracle_64", "True mode", p2_all, COLORS["oracle"]),
    ]
    x_pos = np.arange(len(arms_b))
    for x, (key, _, src, col) in enumerate(arms_b):
        vals = src[key]["switching_seed_means"]
        m, hw = mean_and_ci(vals)
        ax_b.scatter(x + np.linspace(-0.08, 0.08, len(vals)), vals, s=10, color=col, alpha=0.65, zorder=2)
        ax_b.errorbar(x, m, yerr=hw, fmt="D", markersize=3.8, color=COLORS["ink"],
                      markerfacecolor=col, markeredgewidth=0.6, capsize=2.2, linewidth=0.9, zorder=4)
    ax_b.set_xticks(x_pos, [lbl for _, lbl, _, _ in arms_b], rotation=30, ha="right", fontsize=5.9)
    ax_b.set_xlim(-0.5, len(arms_b) - 0.5)
    ax_b.set_ylabel("Switching return")
    ax_b.grid(axis="y", color=COLORS["grid"], linewidth=0.6)
    ax_b.set_title("$M=2^6=64$ hypercube", loc="left", fontsize=7.3)
    add_panel_label(ax_b, "b", x=-0.22)

    # Panel c: Lossy 0.5x Gain Projection & Two-Term Gap Decomposition
    ax_c = fig.add_subplot(grid[0, 2])
    p3 = v50["part3_lossy_projection"]["arms"]
    arms_c = [
        ("uncompensated", "Uncomp.", COLORS["uncomp"]),
        ("causal_projected_bapr", "Causal BAPR", COLORS["bapr"]),
        ("projected_oracle", "Proj. oracle", COLORS["belief_sac"]),
        ("native_reference", "Unatten. ref.", COLORS["oracle"]),
    ]
    x_c = np.arange(len(arms_c))
    for x, (key, _, col) in enumerate(arms_c):
        vals = p3[key]["switching_seed_means"]
        m, hw = mean_and_ci(vals)
        ax_c.scatter(x + np.linspace(-0.08, 0.08, len(vals)), vals, s=10, color=col, alpha=0.65, zorder=2)
        ax_c.errorbar(x, m, yerr=hw, fmt="D", markersize=3.8, color=COLORS["ink"],
                      markerfacecolor=col, markeredgewidth=0.6, capsize=2.2, linewidth=0.9, zorder=4)
    ax_c.set_xticks(x_c, [lbl for _, lbl, _ in arms_c], rotation=30, ha="right", fontsize=5.9)
    ax_c.set_xlim(-0.5, len(arms_c) - 0.5)
    ax_c.grid(axis="y", color=COLORS["grid"], linewidth=0.6)
    ax_c.set_title("Lossy $0.5\\times$ projection", loc="left", fontsize=7.3)
    add_panel_label(ax_c, "c", x=-0.16)

    fig.subplots_adjust(left=0.14, right=0.99, top=0.87, bottom=0.24)
    save_figure(fig, "fig_jmlr_mechanism_generalization")


def plot_v51_mujoco(v51: dict) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(6.3, 2.95), gridspec_kw={"wspace": 0.38})
    env_order = [
        ("halfcheetah", "HalfCheetah-v5 ($d=6$)", "a"),
        ("ant", "Ant-v5 ($d=8$)", "b"),
        ("walker2d", "Walker2d-v5 ($d=6$)", "c"),
        ("hopper", "Hopper-v5 ($d=3$)", "d"),
    ]
    arm_order = [
        ("sac", "SAC", COLORS["sac"]),
        ("escp256", "ESCP-256", COLORS["escp"]),
        ("rma", "RMA", COLORS["rma"]),
        ("belief_sac", "Belief-SAC", COLORS["belief_sac"]),
        ("bapr_selfcal", "BAPR", COLORS["bapr"]),
        ("oracle", "Oracle", COLORS["oracle"]),
    ]

    for ax_idx, (ax, (env_key, title, panel_lbl)) in enumerate(zip(axes, env_order)):
        prim = v51["environments"][env_key]["primary_4mode"]
        mat = np.asarray([prim[k]["switching_seed_means"] for k, _, _ in arm_order])  # [arms, seeds]
        for s_i in range(mat.shape[1]):
            ax.plot(range(len(arm_order)), mat[:, s_i], color="#C8CAD4", linewidth=0.5, alpha=0.5, zorder=1)
        offsets = np.linspace(-0.08, 0.08, mat.shape[1])
        for a_i, (k, _, col) in enumerate(arm_order):
            vals = prim[k]["switching_seed_means"]
            m, hw = mean_and_ci(vals)
            ax.scatter(a_i + offsets, vals, s=10, color=col, edgecolor="white", linewidth=0.3, alpha=0.9, zorder=3)
            ax.errorbar(a_i, m, yerr=hw, fmt="D", markersize=3.6, color=COLORS["ink"],
                        markerfacecolor=col, markeredgewidth=0.6, capsize=2.0, linewidth=0.9, zorder=5)
        ax.set_xticks(range(len(arm_order)), [lbl for _, lbl, _ in arm_order], rotation=35, ha="right", fontsize=6.0)
        ax.set_xlim(-0.45, len(arm_order) - 0.55)
        if ax_idx == 0:
            ax.set_ylabel("Switching return ($n=5$)")
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6)
        ax.set_title(title, loc="left", fontsize=7.0)
        add_panel_label(ax, panel_lbl, x=-0.20, y=1.06)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.25)
    save_figure(fig, "fig_jmlr_mujoco_multimorphology")


def main() -> None:
    configure_style()
    if V50_JSON.is_file():
        plot_v50_mechanism(json.loads(V50_JSON.read_text()))
        print("Generated fig_jmlr_mechanism_generalization.{pdf,png,svg}")
    if V51_JSON.is_file():
        plot_v51_mujoco(json.loads(V51_JSON.read_text()))
        print("Generated fig_jmlr_mujoco_multimorphology.{pdf,png,svg}")


if __name__ == "__main__":
    main()
