#!/usr/bin/env python3
"""Generate the V32 manuscript figures from compact audit JSON files."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# Keep text editable in SVG/PDF exports.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["svg.hashsalt"] = "bapr-v32"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7.0
plt.rcParams["axes.titlesize"] = 8.0
plt.rcParams["axes.labelsize"] = 7.0
plt.rcParams["xtick.labelsize"] = 6.5
plt.rcParams["ytick.labelsize"] = 6.5
plt.rcParams["axes.linewidth"] = 0.75
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False
plt.rcParams["legend.fontsize"] = 6.2


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
COMPACT = ROOT / "reproducibility/v32"
AUDIT_ROOT = COMPACT / "audits"
ANALYSIS_JSON = COMPACT / "analysis.json"
SOURCE_DIR = HERE / "source_data"

T_CRITICAL_95_DF9 = 2.2621571627409915

ARMS = (
    "robust_sac_5p6m",
    "robust_sac_8p4m",
    "escp_recurrent_8p4m",
    "resac_b0_8p4m",
    "canonical_no_compensation",
    "canonical_true_mode_compensation",
    "canonical_v5_map_compensation",
)

ARM_LABELS = {
    "robust_sac_5p6m": "SAC (5.6M)",
    "robust_sac_8p4m": "SAC",
    "escp_recurrent_8p4m": "ESCP",
    "resac_b0_8p4m": "RE-SAC-0",
    "canonical_no_compensation": "No compensation",
    "canonical_true_mode_compensation": "True mode",
    "canonical_v5_map_compensation": "BAPR",
}

COLORS = {
    "sac": "#5F5F70",
    "escp": "#7D89B8",
    "resac": "#A8B5D8",
    "bapr": "#C65A73",
    "oracle": "#2A7F62",
    "no_comp": "#C9C9CF",
    "ink": "#292934",
    "muted": "#777784",
    "grid": "#E5E5EA",
    "fail": "#B64342",
    "pass": "#2E7D4A",
    "soft_blue": "#E7ECF8",
    "soft_rose": "#F5E3E8",
    "soft_green": "#E4F1E9",
}


def load_payloads():
    aggregate = json.loads(ANALYSIS_JSON.read_text())
    seeds = [int(seed) for seed in aggregate["training_seeds"]]
    audits = {
        seed: json.loads((AUDIT_ROOT / f"seed_{seed}" / "audit.json").read_text())
        for seed in seeds
    }
    return aggregate, seeds, audits


def switching_seed_mean(audit: dict, arm: str) -> float:
    return float(
        np.mean(
            [np.mean(event[arm]["returns"]) for event in audit["switching_holdout"].values()]
        )
    )


def stationary_seed_mean(audit: dict, arm: str) -> float:
    values = []
    for event in audit["stationary_holdout"].values():
        values.extend(np.mean(mode["returns"]) for mode in event[arm].values())
    return float(np.mean(values))


def validate_aggregate(
    aggregate: dict, seeds: list[int], audits: dict[int, dict]
) -> None:
    for seed, audit in audits.items():
        if audit["identity"]["training_seed"] != seed:
            raise ValueError(f"released audit seed differs for {seed}")
        cells = [row for event in audit["switching_holdout"].values() for row in event.values()]
        cells.extend(row for event in audit["stationary_holdout"].values()
                     for modes in event.values() for row in modes.values())
        for row in cells:
            if not np.isclose(np.mean(row["returns"]), row["return_mean"], atol=1e-9, rtol=0):
                raise ValueError(f"released episode mean differs for {seed}")
    for arm in ARMS:
        switching = np.mean(
            [switching_seed_mean(audits[seed], arm) for seed in seeds]
        )
        stationary = np.mean(
            [stationary_seed_mean(audits[seed], arm) for seed in seeds]
        )
        if not np.isclose(
            switching, aggregate["switching_return_means"][arm], atol=1e-9, rtol=0
        ):
            raise ValueError(f"switching aggregate mismatch for {arm}")
        if not np.isclose(
            stationary, aggregate["stationary_return_means"][arm], atol=1e-9, rtol=0
        ):
            raise ValueError(f"stationary aggregate mismatch for {arm}")
    for name, row in aggregate["comparisons"].items():
        differences = np.asarray([switching_seed_mean(audits[seed], row["left"])
                                  - switching_seed_mean(audits[seed], row["right"])
                                  for seed in seeds])
        mean = float(np.mean(differences))
        radius = T_CRITICAL_95_DF9 * np.std(differences, ddof=1) / np.sqrt(len(seeds))
        expected = [mean, mean - radius, mean + radius]
        if not np.allclose(expected, [row["paired_mean"], row["ci95_low"], row["ci95_high"]],
                           atol=1e-9, rtol=0):
            raise ValueError(f"paired interval differs for {name}")
        event_wins = sum(np.mean(event[row["left"]]["returns"])
                         > np.mean(event[row["right"]]["returns"])
                         for audit in audits.values() for event in audit["switching_holdout"].values())
        seed_wins = int(np.sum(differences > 0))
        if (seed_wins, event_wins) != (row["seed_wins"], row["event_wins"]):
            raise ValueError(f"paired win counts differ for {name}")
        passed = mean > 0 and mean - radius > 0 and seed_wins >= 8 and event_wins >= 24
        if passed != row["gate_pass"]:
            raise ValueError(f"frozen comparison gate differs for {name}")


def write_source_data(aggregate: dict, seeds: list[int], audits: dict[int, dict]) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    with (SOURCE_DIR / "fig_v32_returns.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("seed", "event_seed", "evaluation", "mode", "arm", "return_mean"),
        )
        writer.writeheader()
        for seed in seeds:
            audit = audits[seed]
            for event_seed, event in audit["switching_holdout"].items():
                for arm in ARMS:
                    writer.writerow(
                        {
                            "seed": seed,
                            "event_seed": event_seed,
                            "evaluation": "switching",
                            "mode": "",
                            "arm": arm,
                            "return_mean": event[arm]["return_mean"],
                        }
                    )
            for event_seed, event in audit["stationary_holdout"].items():
                for arm in ARMS:
                    for mode, row in event[arm].items():
                        writer.writerow(
                            {
                                "seed": seed,
                                "event_seed": event_seed,
                                "evaluation": "stationary",
                                "mode": mode,
                                "arm": arm,
                                "return_mean": row["return_mean"],
                            }
                        )

    with (SOURCE_DIR / "fig_v32_diagnostics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "seed",
                "event_seed",
                "mode_accuracy",
                "median_detection_delay_steps",
                "mean_abs_execution_signal_error",
            ),
        )
        writer.writeheader()
        for seed in seeds:
            for event_seed, event in audits[seed]["switching_holdout"].items():
                row = event["canonical_v5_map_compensation"]
                writer.writerow(
                    {
                        "seed": seed,
                        "event_seed": event_seed,
                        "mode_accuracy": row["routing_mode_accuracy"],
                        "median_detection_delay_steps": row[
                            "switch_detection_delay_median"
                        ],
                        "mean_abs_execution_signal_error": row[
                            "mean_abs_execution_signal_error"
                        ],
                    }
                )

    with (SOURCE_DIR / "fig_v32_decomposition.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "seed",
                "sac_switching_return",
                "oracle_switching_return",
                "causal_switching_return",
                "oracle_headroom",
                "inference_loss",
                "causal_net_advantage",
            ),
        )
        writer.writeheader()
        for seed in seeds:
            sac = switching_seed_mean(audits[seed], "robust_sac_8p4m")
            oracle = switching_seed_mean(
                audits[seed], "canonical_true_mode_compensation"
            )
            causal = switching_seed_mean(
                audits[seed], "canonical_v5_map_compensation"
            )
            writer.writerow(
                {
                    "seed": seed,
                    "sac_switching_return": sac,
                    "oracle_switching_return": oracle,
                    "causal_switching_return": causal,
                    "oracle_headroom": oracle - sac,
                    "inference_loss": oracle - causal,
                    "causal_net_advantage": causal - sac,
                }
            )

    with (SOURCE_DIR / "fig_v32_paired_comparisons.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "comparison",
                "paired_mean",
                "ci95_low",
                "ci95_high",
                "seed_wins",
                "seed_total",
                "event_wins",
                "event_total",
                "gate_pass",
            ),
        )
        writer.writeheader()
        for name, row in aggregate["comparisons"].items():
            writer.writerow(
                {
                    "comparison": name,
                    "paired_mean": row["paired_mean"],
                    "ci95_low": row["ci95_low"],
                    "ci95_high": row["ci95_high"],
                    "seed_wins": row["seed_wins"],
                    "seed_total": 10,
                    "event_wins": row["event_wins"],
                    "event_total": 30,
                    "gate_pass": row["gate_pass"],
                }
            )


def add_panel_label(ax, label: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=COLORS["ink"],
    )


def save_figure(fig, stem: str) -> None:
    fig.savefig(HERE / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(HERE / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(HERE / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_box(ax, xy, width, height, text, facecolor, edgecolor="#686875"):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.85,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=6.4,
        color=COLORS["ink"],
        linespacing=1.15,
    )
    return box


def arrow(ax, start, end, *, connectionstyle="arc3", text=None, text_xy=None):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.9,
            color=COLORS["ink"],
            connectionstyle=connectionstyle,
            shrinkA=2,
            shrinkB=2,
        )
    )
    if text is not None and text_xy is not None:
        ax.text(
            text_xy[0],
            text_xy[1],
            text,
            ha="center",
            va="center",
            fontsize=5.8,
            color=COLORS["muted"],
        )


def make_method_figure() -> None:
    fig = plt.figure(figsize=(6.0, 3.1))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.05, 1.95), wspace=0.26)

    ax_a = fig.add_subplot(grid[0, 0])
    polarity = np.asarray(
        [
            [-1, -1, -1, 1, 1, 1],
            [1, 1, 1, -1, -1, -1],
            [-1, 1, -1, 1, -1, 1],
            [1, -1, 1, -1, 1, -1],
        ]
    )
    cmap = ListedColormap(["#D9909F", "#8394C8"])
    ax_a.imshow(polarity, cmap=cmap, vmin=-1, vmax=1, aspect="equal")
    for row in range(4):
        for col in range(6):
            ax_a.text(
                col,
                row,
                "$-$" if polarity[row, col] < 0 else "$+$",
                ha="center",
                va="center",
                fontsize=7.0,
                color="white",
                fontweight="bold",
            )
    ax_a.set_xticks(range(6), [f"$u_{i}$" for i in range(1, 7)])
    ax_a.set_yticks(range(4), [f"mode {i}" for i in range(4)])
    ax_a.tick_params(length=0, pad=2)
    ax_a.set_xticks(np.arange(-0.5, 6, 1), minor=True)
    ax_a.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax_a.grid(which="minor", color="white", linewidth=1.2)
    ax_a.tick_params(which="minor", bottom=False, left=False)
    for spine in ax_a.spines.values():
        spine.set_visible(False)
    ax_a.set_title("Persistent actuator coordinates", loc="left", pad=7)
    ax_a.text(
        0.5,
        -0.25,
        "One transform per 250-step block;\nsimulator state is preserved at switches",
        transform=ax_a.transAxes,
        ha="center",
        va="top",
        fontsize=6.1,
        color=COLORS["muted"],
    )
    add_panel_label(ax_a, "a", x=-0.18, y=1.05)

    ax_b = fig.add_subplot(grid[0, 1])
    # Leave symmetric data-space margins so the outer box strokes are not
    # clipped at the axes boundary after tight PDF export.
    ax_b.set_xlim(-0.025, 1.025)
    ax_b.set_ylim(0, 1)
    ax_b.axis("off")
    ax_b.set_title("Causal action reparameterization", loc="left", pad=7)
    add_panel_label(ax_b, "b", x=-0.05, y=1.05)

    draw_box(
        ax_b,
        (0.00, 0.60),
        0.25,
        0.22,
        "Frozen policy\n$\pi_r(s_t)$",
        COLORS["soft_blue"],
    )
    draw_box(
        ax_b,
        (0.33, 0.60),
        0.31,
        0.22,
        "Mode\ncompensation\n$a_t=G_{\hat z_t}^{-1}G_r\pi_r(s_t)$",
        COLORS["soft_rose"],
        edgecolor=COLORS["bapr"],
    )
    draw_box(
        ax_b,
        (0.69, 0.60),
        0.31,
        0.22,
        "Actuator regime\n" r"$u_t=\mathrm{clip}(G_{z_t}a_t+\varepsilon_t)$",
        "#EFEFF2",
    )
    draw_box(
        ax_b,
        (0.68, 0.15),
        0.32,
        0.21,
        "Completed\ntransition\n$(s_t,a_t,r_t,s_{t+1})$",
        "#F3F3F5",
    )
    draw_box(
        ax_b,
        (0.35, 0.15),
        0.29,
        0.21,
        "Inverse\nensemble\n$\{f_j(s_t,s_{t+1})\}$",
        COLORS["soft_blue"],
    )
    draw_box(
        ax_b,
        (0.00, 0.15),
        0.28,
        0.21,
        "Tempered\nmode belief\n$b_{t+1}(z)$",
        COLORS["soft_green"],
        edgecolor=COLORS["oracle"],
    )

    arrow(ax_b, (0.25, 0.71), (0.33, 0.71))
    arrow(ax_b, (0.64, 0.71), (0.69, 0.71), text="command", text_xy=(0.665, 0.86))
    arrow(ax_b, (0.845, 0.60), (0.845, 0.36), text="observe", text_xy=(0.95, 0.48))
    arrow(ax_b, (0.68, 0.255), (0.64, 0.255))
    arrow(ax_b, (0.35, 0.255), (0.28, 0.255), text="evidence", text_xy=(0.315, 0.42))
    arrow(
        ax_b,
        (0.14, 0.36),
        (0.48, 0.60),
        connectionstyle="arc3,rad=-0.12",
        text="use at $t+1$",
        text_xy=(0.10, 0.52),
    )
    ax_b.text(
        0.5,
        0.04,
        "$a_t$ uses only $b_t$ and $s_t$; the true mode and $s_{t+1}$ are unavailable until after execution.",
        ha="center",
        va="bottom",
        fontsize=6.1,
        color=COLORS["muted"],
    )

    save_figure(fig, "fig_v32_method")


def make_results_figure(aggregate: dict, seeds: list[int], audits: dict[int, dict]) -> None:
    fig = plt.figure(figsize=(6.0, 3.35))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.55, 1.15, 1.30), wspace=0.48)

    methods = (
        "robust_sac_8p4m",
        "escp_recurrent_8p4m",
        "resac_b0_8p4m",
        "canonical_v5_map_compensation",
    )
    method_colors = (COLORS["sac"], COLORS["escp"], COLORS["resac"], COLORS["bapr"])
    values = np.asarray(
        [[switching_seed_mean(audits[seed], arm) for arm in methods] for seed in seeds]
    )

    ax_a = fig.add_subplot(grid[0, 0])
    for row in values:
        ax_a.plot(range(len(methods)), row, color="#BFC0C8", linewidth=0.55, alpha=0.48, zorder=1)
    offsets = np.linspace(-0.055, 0.055, len(seeds))
    for index, (arm, color) in enumerate(zip(methods, method_colors)):
        ax_a.scatter(
            index + offsets,
            values[:, index],
            s=13,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.92,
            zorder=3,
        )
        mean = float(np.mean(values[:, index]))
        half_width = T_CRITICAL_95_DF9 * float(np.std(values[:, index], ddof=1)) / np.sqrt(len(seeds))
        ax_a.errorbar(
            index,
            mean,
            yerr=half_width,
            fmt="D",
            markersize=4.0,
            color=COLORS["ink"],
            markerfacecolor=color,
            markeredgewidth=0.7,
            capsize=2.5,
            linewidth=1.0,
            zorder=5,
        )
    ax_a.set_xticks(range(len(methods)), [ARM_LABELS[arm] for arm in methods], rotation=24, ha="right")
    ax_a.set_ylabel("Switching return")
    ax_a.set_ylim(0, 4850)
    ax_a.grid(axis="y", color=COLORS["grid"], linewidth=0.6)
    ax_a.set_title("Ten new policy seeds", loc="left")
    ax_a.text(
        0.02,
        0.96,
        "diamonds: mean $\pm$ 95% CI",
        transform=ax_a.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
        color=COLORS["muted"],
    )
    add_panel_label(ax_a, "a")

    ax_b = fig.add_subplot(grid[0, 1])
    comparison_keys = (
        "causal_vs_equal_budget_sac",
        "causal_vs_equal_budget_escp",
        "causal_vs_equal_budget_resac",
    )
    comparison_labels = ("vs SAC", "vs ESCP", "vs RE-SAC-0")
    y = np.arange(3)[::-1]
    for ypos, key in zip(y, comparison_keys):
        row = aggregate["comparisons"][key]
        ax_b.plot(
            [row["ci95_low"], row["ci95_high"]],
            [ypos, ypos],
            color=COLORS["bapr"],
            linewidth=2.0,
            solid_capstyle="round",
        )
        ax_b.scatter(row["paired_mean"], ypos, s=29, marker="D", color=COLORS["bapr"], edgecolor="white", linewidth=0.5, zorder=3)
        ax_b.text(
            row["ci95_high"] + 55,
            ypos,
            f"{row['paired_mean']:+.0f}",
            va="center",
            ha="left",
            fontsize=6.0,
            color=COLORS["ink"],
        )
    ax_b.axvline(0, color=COLORS["ink"], linewidth=0.8)
    ax_b.set_yticks(y, comparison_labels)
    ax_b.set_xlabel("BAPR $-$ comparator")
    ax_b.set_xlim(-120, 2180)
    ax_b.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    ax_b.set_title("Paired effects", loc="left")
    add_panel_label(ax_b, "b")

    ax_c = fig.add_subplot(grid[0, 2])
    ypos = np.arange(3)[::-1]
    for row_y, key in zip(ypos, comparison_keys):
        row = aggregate["comparisons"][key]
        seed_ratio = row["seed_wins"] / 8.0
        event_ratio = row["event_wins"] / 24.0
        seed_color = COLORS["pass"] if seed_ratio >= 1 else COLORS["fail"]
        event_color = COLORS["pass"] if event_ratio >= 1 else COLORS["fail"]
        ax_c.plot([seed_ratio, event_ratio], [row_y + 0.10, row_y - 0.10], color="#C9CAD1", linewidth=0.7, zorder=1)
        ax_c.scatter(seed_ratio, row_y + 0.10, marker="o", s=27, color=seed_color, edgecolor="white", linewidth=0.4, zorder=3)
        ax_c.scatter(event_ratio, row_y - 0.10, marker="s", s=25, color=event_color, edgecolor="white", linewidth=0.4, zorder=3)
        ax_c.text(seed_ratio, row_y + 0.24, f"{row['seed_wins']}/8", ha="center", va="bottom", fontsize=5.5, color=seed_color)
        ax_c.text(event_ratio, row_y - 0.24, f"{row['event_wins']}/24", ha="center", va="top", fontsize=5.5, color=event_color)
    ax_c.axvline(1.0, color=COLORS["ink"], linewidth=0.85, linestyle="--")
    ax_c.set_yticks(ypos, ("SAC", "ESCP", "RE-SAC-0"))
    ax_c.set_xlim(0.82, 1.18)
    ax_c.set_ylim(-0.55, 2.55)
    ax_c.set_xlabel("Achieved / required wins")
    ax_c.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    ax_c.set_title("Preregistered consistency", loc="left")
    add_panel_label(ax_c, "c")

    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.24)
    save_figure(fig, "fig_v32_results")


def make_diagnostics_figure(aggregate: dict, seeds: list[int], audits: dict[int, dict]) -> None:
    fig = plt.figure(figsize=(6.0, 3.35))
    outer = fig.add_gridspec(1, 3, width_ratios=(1.45, 1.25, 0.90), wspace=0.48)

    mechanism_arms = (
        "canonical_no_compensation",
        "canonical_v5_map_compensation",
        "canonical_true_mode_compensation",
    )
    mechanism_colors = (COLORS["no_comp"], COLORS["bapr"], COLORS["oracle"])
    mechanism_values = np.asarray(
        [
            [switching_seed_mean(audits[seed], arm) for arm in mechanism_arms]
            for seed in seeds
        ]
    )

    ax_a = fig.add_subplot(outer[0, 0])
    for row in mechanism_values:
        ax_a.plot(range(3), row, color="#BFC0C8", linewidth=0.65, alpha=0.55, zorder=1)
    offsets = np.linspace(-0.05, 0.05, len(seeds))
    for index, color in enumerate(mechanism_colors):
        ax_a.scatter(index + offsets, mechanism_values[:, index], s=14, color=color, edgecolor="white", linewidth=0.35, zorder=3)
        ax_a.scatter(index, np.mean(mechanism_values[:, index]), s=34, marker="D", color=color, edgecolor=COLORS["ink"], linewidth=0.6, zorder=4)
    ax_a.set_xticks(range(3), ("No comp.", "Causal BAPR", "True mode"), rotation=22, ha="right")
    ax_a.set_ylabel("Switching return")
    ax_a.set_ylim(-300, 5000)
    ax_a.grid(axis="y", color=COLORS["grid"], linewidth=0.6)
    ax_a.set_title("Mechanism ladder", loc="left")
    add_panel_label(ax_a, "a")

    ax_b = fig.add_subplot(outer[0, 1])
    sac = np.asarray(
        [switching_seed_mean(audits[seed], "robust_sac_8p4m") for seed in seeds]
    )
    oracle = np.asarray(
        [
            switching_seed_mean(audits[seed], "canonical_true_mode_compensation")
            for seed in seeds
        ]
    )
    causal = np.asarray(
        [
            switching_seed_mean(audits[seed], "canonical_v5_map_compensation")
            for seed in seeds
        ]
    )
    headroom = oracle - sac
    net_advantage = causal - sac
    y = np.arange(len(seeds))[::-1]
    for ypos, oracle_delta, causal_delta in zip(y, headroom, net_advantage):
        ax_b.plot(
            [causal_delta, oracle_delta],
            [ypos, ypos],
            color="#BFC0C8",
            linewidth=1.0,
            zorder=1,
        )
    ax_b.scatter(
        headroom,
        y,
        s=24,
        marker="o",
        color=COLORS["oracle"],
        edgecolor="white",
        linewidth=0.45,
        label="true-mode headroom",
        zorder=3,
    )
    ax_b.scatter(
        net_advantage,
        y,
        s=25,
        marker="D",
        color=COLORS["bapr"],
        edgecolor="white",
        linewidth=0.45,
        label="causal net advantage",
        zorder=4,
    )
    ax_b.axvline(0, color=COLORS["ink"], linewidth=0.8)
    all_deltas = np.concatenate((headroom, net_advantage))
    x_pad = max(150.0, 0.08 * float(np.ptp(all_deltas)))
    ax_b.set_xlim(float(np.min(all_deltas) - x_pad), float(np.max(all_deltas) + x_pad))
    ax_b.set_yticks(y, [str(seed)[-3:] for seed in seeds])
    ax_b.set_xlabel("Return difference from SAC")
    ax_b.set_ylabel("Policy seed suffix")
    ax_b.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    ax_b.set_title("Headroom and inference loss", loc="left")
    ax_b.legend(
        loc="lower right",
        borderaxespad=0.15,
        frameon=False,
        fontsize=5.1,
        handletextpad=0.25,
        labelspacing=0.25,
    )
    add_panel_label(ax_b, "b")

    nested = outer[0, 2].subgridspec(2, 1, hspace=0.48)
    accuracies = []
    delays = []
    for seed in seeds:
        for event in audits[seed]["switching_holdout"].values():
            row = event["canonical_v5_map_compensation"]
            accuracies.append(100.0 * row["routing_mode_accuracy"])
            delays.append(float(row["switch_detection_delay_median"]))
    x_offsets = np.linspace(-0.16, 0.16, len(accuracies))

    ax_c1 = fig.add_subplot(nested[0, 0])
    ax_c1.scatter(x_offsets, accuracies, s=12, color=COLORS["bapr"], alpha=0.82, edgecolor="white", linewidth=0.3)
    ax_c1.plot([-0.19, 0.19], [np.mean(accuracies)] * 2, color=COLORS["ink"], linewidth=1.2)
    ax_c1.set_xlim(-0.24, 0.24)
    ax_c1.set_ylim(95.5, 100.0)
    ax_c1.set_xticks([])
    ax_c1.set_ylabel("Accuracy (%)")
    ax_c1.grid(axis="y", color=COLORS["grid"], linewidth=0.6)
    ax_c1.set_title("Inference across 30 events", loc="left", pad=4)
    ax_c1.text(0.97, 0.94, f"mean {np.mean(accuracies):.2f}%", transform=ax_c1.transAxes, ha="right", va="top", fontsize=5.7, color=COLORS["muted"])
    add_panel_label(ax_c1, "c", x=-0.28, y=1.08)

    ax_c2 = fig.add_subplot(nested[1, 0])
    ax_c2.scatter(x_offsets, delays, s=12, color=COLORS["oracle"], alpha=0.82, edgecolor="white", linewidth=0.3)
    ax_c2.plot([-0.19, 0.19], [np.median(delays)] * 2, color=COLORS["ink"], linewidth=1.2)
    ax_c2.set_xlim(-0.24, 0.24)
    ax_c2.set_ylim(0, 5)
    ax_c2.set_xticks([])
    ax_c2.set_ylabel("Delay (steps)")
    ax_c2.grid(axis="y", color=COLORS["grid"], linewidth=0.6)
    ax_c2.text(0.97, 0.08, f"median {np.median(delays):.0f}", transform=ax_c2.transAxes, ha="right", va="bottom", fontsize=5.7, color=COLORS["muted"])

    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.23)
    save_figure(fig, "fig_v32_diagnostics")


def main() -> None:
    aggregate, seeds, audits = load_payloads()
    validate_aggregate(aggregate, seeds, audits)
    write_source_data(aggregate, seeds, audits)
    make_method_figure()
    make_results_figure(aggregate, seeds, audits)
    make_diagnostics_figure(aggregate, seeds, audits)
    print("Generated V32 method, results, and diagnostics figures.")


if __name__ == "__main__":
    main()
