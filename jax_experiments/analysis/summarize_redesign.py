"""Summarize BAPR redesign sweeps by tag and environment.

Example:
    python3 -m jax_experiments.analysis.summarize_redesign \
      --results_root jax_experiments/results_paper \
      --prefix redesign_gate_v17
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_run_name(name: str):
    """Parse names like tag_bapr_HalfCheetah_dw60_s0."""
    marker = "_bapr_"
    if marker not in name or "_dw" not in name or "_s" not in name:
        return None
    tag, rest = name.rsplit(marker, 1)
    env, tail = rest.rsplit("_dw", 1)
    dwell_s, seed_s = tail.rsplit("_s", 1)
    try:
        return tag, env, int(dwell_s), int(seed_s)
    except ValueError:
        return None


def summarize(results_root: Path, prefix: str, tail_points: int):
    groups = defaultdict(list)

    for run_dir in sorted(results_root.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith(prefix):
            continue
        parsed = parse_run_name(run_dir.name)
        if parsed is None:
            continue
        tag, env, dwell, seed = parsed
        eval_path = run_dir / "logs" / "eval_reward.npy"
        if not eval_path.exists():
            continue
        values = np.load(eval_path)
        if values.size == 0:
            continue
        n_tail = min(tail_points, values.size)
        groups[(tag, env, dwell)].append(
            {
                "seed": seed,
                "last": float(values[-n_tail:].mean()),
                "best": float(values.max()),
                "n_eval": int(values.size),
                "done": (run_dir / "done.ok").exists(),
            }
        )

    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=Path,
                        default=Path("jax_experiments/results_paper"))
    parser.add_argument("--prefix", default="redesign_gate_v")
    parser.add_argument("--tail_points", type=int, default=20)
    args = parser.parse_args()

    groups = summarize(args.results_root, args.prefix, args.tail_points)
    if not groups:
        print("No matching runs found.")
        return

    header = (
        "tag,env,dwell,n,seeds,complete,last_mean,last_std,best_mean,best_std,"
        "eval_points_min,eval_points_max"
    )
    print(header)
    for (tag, env, dwell), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda row: row["seed"])
        lasts = np.asarray([row["last"] for row in rows], dtype=np.float64)
        bests = np.asarray([row["best"] for row in rows], dtype=np.float64)
        eval_counts = [row["n_eval"] for row in rows]
        seeds = " ".join(str(row["seed"]) for row in rows)
        complete = sum(1 for row in rows if row["done"])
        print(
            f"{tag},{env},{dwell},{len(rows)},\"{seeds}\",{complete},"
            f"{lasts.mean():.3f},{lasts.std():.3f},"
            f"{bests.mean():.3f},{bests.std():.3f},"
            f"{min(eval_counts)},{max(eval_counts)}"
        )


if __name__ == "__main__":
    main()
