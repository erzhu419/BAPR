"""Diagnose mild final-sweep task performance by gravity scale.

This script reads eval-only final sweep outputs from the primary mild final
evaluation root plus the patched finalfix root, then writes:

- a Markdown diagnostic report,
- per gravity-bin metrics,
- BAPR per-task failure cases,
- switching episode diagnostics,
- an optional PNG return-vs-gravity plot when matplotlib is available.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


ENVS = ["Ant", "HalfCheetah", "Hopper", "Walker2d"]
ALGOS = ["sac", "escp", "resac", "bapr"]
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOTS = [
    ROOT / "jax_experiments" / "results_evalfix_mild_finaleval_finalfix",
    ROOT / "jax_experiments" / "results_evalfix_mild_finaleval",
]
DEFAULT_OUT_PREFIX = ROOT / "reports" / "mild_final_sweep_task_diagnostics_2026-07-10"


def fnum(value: str | float | int | None, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    return float(value)


def run_name(algo: str, env: str) -> str:
    return f"mildfix_escp_mild_gravity_{algo}_{env}_s0"


def find_run_file(roots: list[Path], algo: str, env: str, filename: str) -> Path:
    name = run_name(algo, env)
    for root in roots:
        path = root / name / filename
        if path.exists() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(f"missing {filename} for {name} in {roots}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def gravity_scale(row: dict[str, str]) -> float:
    # Gravity is [0, 0, -9.81 * scale]. final_task_sweep stores min/mean/max
    # over the vector, so abs(gravity_min) is the z magnitude.
    return abs(fnum(row.get("gravity_min"))) / 9.81


def quantile_bins(values: list[float]) -> tuple[float, float]:
    vals = sorted(values)
    if not vals:
        return math.nan, math.nan
    lo_idx = max(0, min(len(vals) - 1, int(len(vals) / 3)))
    hi_idx = max(0, min(len(vals) - 1, int(2 * len(vals) / 3)))
    return vals[lo_idx], vals[hi_idx]


def scale_bin(scale: float, q1: float, q2: float) -> str:
    if scale <= q1:
        return "low"
    if scale <= q2:
        return "mid"
    return "high"


def collect(roots: list[Path]):
    task_rows: list[dict] = []
    switch_rows: list[dict] = []
    sources: dict[str, str] = {}
    for env in ENVS:
        for algo in ALGOS:
            name = run_name(algo, env)
            task_path = find_run_file(roots, algo, env, "task_returns.csv")
            switch_path = find_run_file(roots, algo, env, "switching_returns.csv")
            sources[name] = str(task_path.parent)
            for row in read_csv(task_path):
                out = dict(row)
                out["algo"] = algo
                out["env"] = env
                out["run_name"] = name
                out["gravity_scale"] = gravity_scale(row)
                out["return_mean"] = fnum(row["return_mean"])
                out["terminated_rate"] = fnum(row["terminated_rate"])
                out["steps_mean"] = fnum(row["steps_mean"])
                out["task_index"] = int(row["task_index"])
                task_rows.append(out)
            for row in read_csv(switch_path):
                out = dict(row)
                out["algo"] = algo
                out["env"] = env
                out["run_name"] = name
                out["return"] = fnum(row["return"])
                out["switch_count"] = fnum(row["switch_count"])
                out["terminated"] = str(row["terminated"]).lower() == "true"
                out["done_before_first_switch"] = (
                    str(row["done_before_first_switch"]).lower() == "true"
                )
                out["done_step"] = fnum(row["done_step"])
                switch_rows.append(out)
    return task_rows, switch_rows, sources


def add_bins(task_rows: list[dict]) -> None:
    by_env_split: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in task_rows:
        by_env_split[(row["env"], row["split"])].append(row["gravity_scale"])
    cuts = {key: quantile_bins(vals) for key, vals in by_env_split.items()}
    for row in task_rows:
        q1, q2 = cuts[(row["env"], row["split"])]
        row["scale_bin"] = scale_bin(row["gravity_scale"], q1, q2)


def avg(vals) -> float:
    vals = [float(v) for v in vals if not math.isnan(float(v))]
    return mean(vals) if vals else math.nan


def fmt(x: float, digits: int = 1) -> str:
    if math.isnan(x):
        return "nan"
    return f"{x:.{digits}f}"


def bin_metrics(task_rows: list[dict]) -> list[dict]:
    out = []
    test_rows = [r for r in task_rows if r["split"] == "test"]
    for env in ENVS:
        for b in ["low", "mid", "high"]:
            bin_rows = [r for r in test_rows if r["env"] == env and r["scale_bin"] == b]
            if not bin_rows:
                continue
            for algo in ALGOS:
                rows = [r for r in bin_rows if r["algo"] == algo]
                out.append({
                    "env": env,
                    "scale_bin": b,
                    "algo": algo,
                    "n_tasks": len(rows),
                    "gravity_min": min(r["gravity_scale"] for r in rows),
                    "gravity_max": max(r["gravity_scale"] for r in rows),
                    "return_mean": avg(r["return_mean"] for r in rows),
                    "terminated_rate": avg(r["terminated_rate"] for r in rows),
                    "steps_mean": avg(r["steps_mean"] for r in rows),
                })
    return out


def bapr_failures(task_rows: list[dict]) -> list[dict]:
    test_rows = [r for r in task_rows if r["split"] == "test"]
    by_key: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in test_rows:
        by_key[(row["env"], row["task_index"])].append(row)
    out = []
    for (env, task_idx), rows in sorted(by_key.items()):
        if len(rows) != len(ALGOS):
            continue
        bapr = next(r for r in rows if r["algo"] == "bapr")
        best = max(rows, key=lambda r: r["return_mean"])
        non_bapr_best = max(
            [r for r in rows if r["algo"] != "bapr"],
            key=lambda r: r["return_mean"],
        )
        sorted_rows = sorted(rows, key=lambda r: r["return_mean"], reverse=True)
        rank = 1 + [r["algo"] for r in sorted_rows].index("bapr")
        out.append({
            "env": env,
            "task_index": task_idx,
            "scale_bin": bapr["scale_bin"],
            "gravity_scale": bapr["gravity_scale"],
            "bapr_return": bapr["return_mean"],
            "bapr_terminated_rate": bapr["terminated_rate"],
            "best_algo": best["algo"],
            "best_return": best["return_mean"],
            "best_terminated_rate": best["terminated_rate"],
            "best_non_bapr_algo": non_bapr_best["algo"],
            "best_non_bapr_return": non_bapr_best["return_mean"],
            "gap_to_best": bapr["return_mean"] - best["return_mean"],
            "gap_to_best_non_bapr": bapr["return_mean"] - non_bapr_best["return_mean"],
            "bapr_rank": rank,
        })
    return sorted(out, key=lambda r: r["gap_to_best"])


def switch_metrics(switch_rows: list[dict]) -> list[dict]:
    out = []
    for env in ENVS:
        for algo in ALGOS:
            rows = [r for r in switch_rows if r["env"] == env and r["algo"] == algo]
            out.append({
                "env": env,
                "algo": algo,
                "episodes": len(rows),
                "return_mean": avg(r["return"] for r in rows),
                "return_min": min(r["return"] for r in rows),
                "return_max": max(r["return"] for r in rows),
                "terminated_rate": avg(1.0 if r["terminated"] else 0.0 for r in rows),
                "done_before_first_switch_rate": avg(
                    1.0 if r["done_before_first_switch"] else 0.0 for r in rows
                ),
                "switch_count_mean": avg(r["switch_count"] for r in rows),
                "done_step_mean": avg(r["done_step"] for r in rows),
            })
    return out


def best_by_env(task_rows: list[dict]) -> list[dict]:
    test_rows = [r for r in task_rows if r["split"] == "test"]
    out = []
    for env in ENVS:
        env_rows = [r for r in test_rows if r["env"] == env]
        for algo in ALGOS:
            rows = [r for r in env_rows if r["algo"] == algo]
            out.append({
                "env": env,
                "algo": algo,
                "return_mean": avg(r["return_mean"] for r in rows),
                "terminated_rate": avg(r["terminated_rate"] for r in rows),
                "steps_mean": avg(r["steps_mean"] for r in rows),
            })
    return out


def make_plot(task_rows: list[dict], out_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    test_rows = [r for r in task_rows if r["split"] == "test"]
    colors = {
        "sac": "#4C78A8",
        "escp": "#F58518",
        "resac": "#54A24B",
        "bapr": "#B279A2",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False)
    for ax, env in zip(axes.ravel(), ENVS):
        for algo in ALGOS:
            rows = sorted(
                [r for r in test_rows if r["env"] == env and r["algo"] == algo],
                key=lambda r: r["gravity_scale"],
            )
            ax.plot(
                [r["gravity_scale"] for r in rows],
                [r["return_mean"] for r in rows],
                marker="o",
                markersize=2.8,
                linewidth=1.2,
                label=algo.upper() if algo != "bapr" else "BAPR",
                color=colors[algo],
                alpha=0.9,
            )
        ax.set_title(env)
        ax.set_xlabel("gravity scale")
        ax.set_ylabel("test return")
        ax.grid(True, alpha=0.25)
    axes[0][0].legend(ncol=4, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def markdown_report(
    path: Path,
    roots: list[Path],
    overall: list[dict],
    bins: list[dict],
    failures: list[dict],
    switches: list[dict],
    plot_path: Path | None,
) -> None:
    lines: list[str] = []
    lines.append("# Mild final-sweep per-task diagnostics, 2026-07-10")
    lines.append("")
    lines.append("Input roots, in priority order:")
    lines.append("")
    for root in roots:
        lines.append(f"- `{root}`")
    lines.append("")
    lines.append("Gravity scale is computed as `abs(gravity_min) / 9.81` from")
    lines.append("`task_returns.csv`; bins are per-environment test-task tertiles.")
    lines.append("")
    if plot_path is not None:
        lines.append(f"Plot: `{plot_path}`")
        lines.append("")

    lines.append("## Overall stationary test")
    lines.append("")
    lines.append("| env | algo | return | terminated | steps |")
    lines.append("|---|---|---:|---:|---:|")
    for env in ENVS:
        env_rows = sorted(
            [r for r in overall if r["env"] == env],
            key=lambda r: r["return_mean"],
            reverse=True,
        )
        for r in env_rows:
            lines.append(
                f"| {env} | {r['algo']} | {fmt(r['return_mean'])} | "
                f"{fmt(r['terminated_rate'], 2)} | {fmt(r['steps_mean'], 0)} |"
            )
    lines.append("")

    lines.append("## Gravity-bin winners")
    lines.append("")
    lines.append("| env | bin | gravity range | winner | BAPR return | best return | BAPR gap | BAPR term |")
    lines.append("|---|---|---:|---|---:|---:|---:|---:|")
    for env in ENVS:
        for b in ["low", "mid", "high"]:
            rows = [r for r in bins if r["env"] == env and r["scale_bin"] == b]
            if not rows:
                continue
            best = max(rows, key=lambda r: r["return_mean"])
            bapr = next(r for r in rows if r["algo"] == "bapr")
            g0 = min(r["gravity_min"] for r in rows)
            g1 = max(r["gravity_max"] for r in rows)
            lines.append(
                f"| {env} | {b} | {fmt(g0, 2)}-{fmt(g1, 2)} | {best['algo']} | "
                f"{fmt(bapr['return_mean'])} | {fmt(best['return_mean'])} | "
                f"{fmt(bapr['return_mean'] - best['return_mean'])} | "
                f"{fmt(bapr['terminated_rate'], 2)} |"
            )
    lines.append("")

    lines.append("## Worst BAPR task gaps")
    lines.append("")
    lines.append("| env | task | g scale | bin | BAPR | BAPR term | best | best return | gap |")
    lines.append("|---|---:|---:|---|---:|---:|---|---:|---:|")
    for r in failures[:16]:
        lines.append(
            f"| {r['env']} | {r['task_index']} | {fmt(r['gravity_scale'], 2)} | "
            f"{r['scale_bin']} | {fmt(r['bapr_return'])} | "
            f"{fmt(r['bapr_terminated_rate'], 2)} | {r['best_algo']} | "
            f"{fmt(r['best_return'])} | {fmt(r['gap_to_best'])} |"
        )
    lines.append("")

    lines.append("## Switching diagnostics")
    lines.append("")
    lines.append("| env | algo | return | terminated | done before first switch | switches | done step |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for env in ENVS:
        env_rows = sorted(
            [r for r in switches if r["env"] == env],
            key=lambda r: r["return_mean"],
            reverse=True,
        )
        for r in env_rows:
            lines.append(
                f"| {env} | {r['algo']} | {fmt(r['return_mean'])} | "
                f"{fmt(r['terminated_rate'], 2)} | "
                f"{fmt(r['done_before_first_switch_rate'], 2)} | "
                f"{fmt(r['switch_count_mean'], 1)} | {fmt(r['done_step_mean'], 0)} |"
            )
    lines.append("")

    lines.append("## Readout")
    lines.append("")
    lines.append("- BAPR's only clear overall stationary-test win is Hopper. The bin view")
    lines.append("  is more nuanced: it is close to ESCP on low gravity, wins the middle")
    lines.append("  gravity tertile, and is close to RE-SAC on high gravity. Its switching")
    lines.append("  return is low because it survives switches and then loses return after")
    lines.append("  the switch. SAC/ESCP")
    lines.append("  Hopper switching numbers are not strong adaptation evidence because")
    lines.append("  they terminate before the first switch in every switching episode.")
    lines.append("- Ant is not mainly a BAPR early-termination problem: ESCP is much")
    lines.append("  stronger despite high termination, while BAPR survives longer but earns")
    lines.append("  low return. That points to policy/objective quality rather than only")
    lines.append("  online mode inference.")
    lines.append("- HalfCheetah favors SAC across the full gravity range; BAPR is stable")
    lines.append("  but lower-return. This suggests a robust single policy may already be")
    lines.append("  sufficient under the mild protocol.")
    lines.append("- Walker2d is close between RE-SAC and BAPR on stationary test, but")
    lines.append("  RE-SAC dominates switching. BAPR's adaptation mechanism is not yet")
    lines.append("  providing the expected switch recovery.")
    lines.append("")
    lines.append("Implication: before expanding seeds, the next experiment should be an")
    lines.append("oracle/fallback diagnostic, not another BAPR gate tweak. We need to test")
    lines.append("whether task-aware conditioning has headroom at all, and whether BAPR's")
    lines.append("failure is estimator/gating or the adapted policy itself.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="*", type=Path, default=DEFAULT_ROOTS)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    args = parser.parse_args()

    roots = list(args.roots)
    task_rows, switch_rows, _ = collect(roots)
    add_bins(task_rows)

    bins = bin_metrics(task_rows)
    failures = bapr_failures(task_rows)
    switches = switch_metrics(switch_rows)
    overall = best_by_env(task_rows)

    prefix = args.out_prefix
    write_csv(prefix.with_name(prefix.name + "_bin_metrics.csv"), bins)
    write_csv(prefix.with_name(prefix.name + "_bapr_failures.csv"), failures)
    write_csv(prefix.with_name(prefix.name + "_switching.csv"), switches)
    write_csv(prefix.with_name(prefix.name + "_overall.csv"), overall)

    plot_path = prefix.with_suffix(".png")
    plotted = make_plot(task_rows, plot_path)
    markdown_report(
        prefix.with_suffix(".md"),
        roots,
        overall,
        bins,
        failures,
        switches,
        plot_path if plotted else None,
    )
    print(f"wrote {prefix.with_suffix('.md')}")
    if plotted:
        print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
