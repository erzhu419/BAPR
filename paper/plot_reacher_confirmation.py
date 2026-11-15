#!/usr/bin/env python3
"""Plot the independent Reacher confirmation from released compact audits."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

from plot_v32_figures import COLORS, add_panel_label, save_figure


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent / "reproducibility/reacher_confirmation"
PAIRS = (
    ("causal_95103", "conditioned", "Filtered\n(primary)"),
    ("no_filter", "conditioned_no_filter", "Instantaneous\n(secondary)"),
)


def interval(values):
    values = np.asarray(values, np.float64)
    mean = float(np.mean(values))
    radius = float(t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1)
                   / np.sqrt(len(values)))
    return mean, radius


def validate_results(aggregate, audits):
    seeds = aggregate["policy_seeds"]
    if [row["seed"] for row in audits] != seeds:
        raise ValueError("released Reacher audit seeds differ")
    values = {route: np.asarray([np.mean([e["return"] for e in row["events"][route]])
                                for row in audits])
              for route in aggregate["means"]}
    for route, returns in values.items():
        cached = [row["means"][route] for row in audits]
        if not np.allclose(returns, cached, atol=1e-9, rtol=0):
            raise ValueError(f"released episode mean differs for {route}")
        if not np.allclose(returns, aggregate["seed_means"][route], atol=1e-9, rtol=0):
            raise ValueError(f"released seed returns differ for {route}")
        if not np.isclose(np.mean(returns), aggregate["means"][route], atol=1e-9, rtol=0):
            raise ValueError(f"released aggregate mean differs for {route}")
    for route, recorded in aggregate["stationary_means"].items():
        mean = np.mean([e["return"] for row in audits for e in row["stationary"][route]])
        if not np.isclose(mean, recorded, atol=1e-9, rtol=0):
            raise ValueError(f"released stationary mean differs for {route}")
    for name, recorded in aggregate["comparisons"].items():
        left, right = name.split("-minus-")
        mean, radius = interval(values[left] - values[right])
        if not np.allclose([mean, mean - radius, mean + radius],
                           [recorded["mean"], *recorded["ci95"]], atol=1e-9, rtol=0):
            raise ValueError(f"released paired interval differs for {left}")
        if (recorded["n_policy_seeds"], recorded["wins"]) != (
                len(seeds), int(np.sum(values[left] > values[right]))):
            raise ValueError(f"released paired sample counts differ for {left}")
    return seeds, values


def load_results():
    aggregate = json.loads((PACKAGE / "analysis.json").read_text())
    audits = [json.loads((PACKAGE / "audits" / f"seed_{seed}" / "audit.json").read_text())
              for seed in aggregate["policy_seeds"]]
    return validate_results(aggregate, audits)


def main():
    seeds, values = load_results()
    source = HERE / "source_data"
    source.mkdir(exist_ok=True)
    with (source / "reacher_confirmation_seed_returns.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["policy_seed", *values])
        writer.writerows([seed, *[returns[index] for returns in values.values()]]
                         for index, seed in enumerate(seeds))

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.0, 2.65),
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    arms = ("robust", "conditioned", "causal_95103", "oracle")
    labels = ("Robust\nSAC", "Belief\nSAC", "BAPR", "True\nmode")
    colors = (COLORS["sac"], COLORS["escp"], COLORS["bapr"], COLORS["oracle"])
    offsets = np.linspace(-0.11, 0.11, len(seeds))
    for index, offset in enumerate(offsets):
        ax_a.plot([1 + offset, 2 + offset],
                  [values["conditioned"][index], values["causal_95103"][index]],
                  color=COLORS["muted"], alpha=0.45, linewidth=0.6, zorder=1)
    for position, (arm, color) in enumerate(zip(arms, colors, strict=True)):
        mean, radius = interval(values[arm])
        ax_a.scatter(position + offsets, values[arm], s=17, color=color, alpha=0.8,
                     edgecolors="white", linewidths=0.3, zorder=2)
        ax_a.errorbar(position + 0.21, mean, yerr=radius, fmt="D", markersize=4,
                      color=color, capsize=2, linewidth=1, zorder=3)
    ax_a.set_xticks(range(len(arms)), labels)
    ax_a.set_xlim(-0.4, 3.45)
    ax_a.set_ylabel("Switching return")
    ax_a.grid(axis="y", color=COLORS["grid"], linewidth=0.6)
    ax_a.set_title("Same-information control", loc="left", pad=12)
    add_panel_label(ax_a, "a", x=-0.18, y=1.10)

    for position, (left, right, unused) in enumerate(PAIRS):
        difference = values[left] - values[right]
        mean, radius = interval(difference)
        color = COLORS["bapr"] if position == 0 else COLORS["oracle"]
        ax_b.scatter(difference, position + offsets, s=17, color=color, alpha=0.8,
                     edgecolors="white", linewidths=0.3, zorder=2)
        ax_b.errorbar(mean, position + 0.22, xerr=radius, fmt="D", markersize=4,
                      color=color, capsize=2, linewidth=1, zorder=3)
    ax_b.axvline(0, color=COLORS["ink"], linewidth=0.8)
    ax_b.set_xlim(-0.5, 12)
    ax_b.set_ylim(1.55, -0.45)
    ax_b.set_yticks(range(len(PAIRS)), [row[2] for row in PAIRS])
    ax_b.set_xlabel("Compensation minus belief SAC")
    ax_b.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    ax_b.set_title("Paired return differences", loc="left", pad=12)
    add_panel_label(ax_b, "b", x=-0.40, y=1.10)
    fig.subplots_adjust(left=0.085, right=0.98, top=0.80, bottom=0.22, wspace=0.65)
    save_figure(fig, "fig_reacher_confirmation")
    for left, right, unused in PAIRS:
        mean, radius = interval(values[left] - values[right])
        print(f"{left} minus {right}: {mean:.3f} [{mean-radius:.3f},{mean+radius:.3f}]")


if __name__ == "__main__":
    main()
