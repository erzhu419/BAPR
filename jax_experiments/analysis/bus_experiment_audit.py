"""Aggregate legacy bus nonstationary experiments.

This is separate from the JAX MuJoCo audit because the bus experiments use the
older PyTorch scripts and log per-episode arrays under save_root/logs/<run>.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def load_metric(log_dir: Path, name: str) -> Optional[np.ndarray]:
    path = log_dir / f"{name}.npy"
    if not path.exists():
        return None
    try:
        return np.load(path, allow_pickle=True)
    except Exception:
        return None


def parse_run(run_name: str) -> Dict[str, object]:
    parts = run_name.split("_")
    algo = parts[0] if parts else ""
    seed = None
    if parts and parts[-1].startswith("s") and parts[-1][1:].isdigit():
        seed = int(parts[-1][1:])
    return {"run": run_name, "algo": algo, "seed": seed}


def summarize(log_dir: Path, last_episodes: int) -> Dict[str, object]:
    rewards = load_metric(log_dir, "rewards")
    if rewards is None:
        raise ValueError(f"missing rewards.npy in {log_dir}")
    rewards = np.asarray(rewards, dtype=float)
    tail = rewards[-min(last_episodes, len(rewards)):] if len(rewards) else rewards
    row = parse_run(log_dir.name)
    row.update({
        "episodes": len(rewards),
        "mean_last_episodes": f"{float(np.nanmean(tail)):.3f}" if len(tail) else "",
        "std_last_episodes": f"{float(np.nanstd(tail)):.3f}" if len(tail) else "",
        "best_reward": f"{float(np.nanmax(rewards)):.3f}" if len(rewards) else "",
    })
    for metric in [
        "q_stds_episode",
        "mode_switches_episode",
        "mode_id_episode",
        "weighted_lambdas_episode",
        "belief_entropies_episode",
        "effective_windows_episode",
    ]:
        arr = load_metric(log_dir, metric)
        if arr is None or len(arr) == 0:
            continue
        arr = np.asarray(arr, dtype=float)
        mtail = arr[-min(last_episodes, len(arr)):]
        row[f"{metric}_last_mean"] = f"{float(np.nanmean(mtail)):.3f}"
        row[f"{metric}_final"] = f"{float(arr[-1]):.3f}"
    return row


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: List[Dict[str, object]]) -> None:
    grouped: Dict[str, List[float]] = {}
    for row in rows:
        try:
            val = float(row["mean_last_episodes"])
        except Exception:
            continue
        grouped.setdefault(str(row.get("algo", "")), []).append(val)
    lines = ["# Bus Experiment Audit", ""]
    lines.append("| algo | n | mean | std | min | max |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for algo, vals in sorted(grouped.items()):
        arr = np.asarray(vals, dtype=float)
        lines.append(
            f"| {algo} | {len(arr)} | {np.mean(arr):.1f} | {np.std(arr):.1f} | "
            f"{np.min(arr):.1f} | {np.max(arr):.1f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_root", default="bus_experiments_paper_v2")
    parser.add_argument("--out_dir", default="paper/audit_tables")
    parser.add_argument("--last_episodes", type=int, default=50)
    args = parser.parse_args()

    logs_root = Path(args.save_root) / "logs"
    rows = []
    for log_dir in sorted(logs_root.glob("*")):
        if not log_dir.is_dir():
            continue
        try:
            rows.append(summarize(log_dir, args.last_episodes))
        except Exception as exc:
            print(f"[skip] {log_dir}: {exc}")
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "bus_run_summary.csv", rows)
    write_markdown(out_dir / "bus_aggregate_summary.md", rows)
    print(f"wrote bus audit for {len(rows)} runs to {out_dir}")


if __name__ == "__main__":
    main()
