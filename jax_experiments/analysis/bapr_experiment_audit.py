"""Aggregate BAPR paper experiments into CSV/Markdown audit tables.

The script is intentionally log-only: it does not load checkpoints and can run
on CPU servers. It covers the paper gaps that reviewers are likely to inspect:
seed-matched final performance, tail/worst-regime behavior, BOCD detection
quality, post-switch recovery, and runtime overhead.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


ALGOS = [
    "bapr_unsupervised",
    "bapr_no_adapt_beta",
    "bapr_fixed_decay",
    "bapr_no_bocd",
    "bapr_no_rmdm",
    "bad_bapr",
    "bapr",
    "escp",
    "resac",
    "sac",
]

ENV_SHORT_TO_FULL = {
    "HalfCheetah": "HalfCheetah-v2",
    "Hopper": "Hopper-v2",
    "Walker2d": "Walker2d-v2",
    "Ant": "Ant-v2",
}


def load_metric(log_dir: Path, name: str) -> Optional[np.ndarray]:
    path = log_dir / f"{name}.npy"
    if not path.exists():
        return None
    try:
        return np.load(path, allow_pickle=True)
    except Exception:
        return None


def parse_run_name(run_name: str) -> Dict[str, object]:
    meta: Dict[str, object] = {
        "tag": "",
        "algo": "",
        "env": "",
        "seed": None,
        "dwell": None,
    }
    if "_dw" not in run_name or "_s" not in run_name:
        return meta

    seed_part = run_name.rsplit("_s", 1)[-1]
    if seed_part.isdigit():
        meta["seed"] = int(seed_part)

    before_seed = run_name.rsplit("_s", 1)[0]
    before_dwell, dwell_part = before_seed.rsplit("_dw", 1)
    if dwell_part.isdigit():
        meta["dwell"] = int(dwell_part)

    for env_short, env_full in ENV_SHORT_TO_FULL.items():
        needle = f"_{env_short}"
        if before_dwell.endswith(needle):
            meta["env"] = env_full
            left = before_dwell[: -len(needle)]
            for algo in ALGOS:
                suffix = f"_{algo}"
                if left.endswith(suffix):
                    meta["algo"] = algo
                    meta["tag"] = left[: -len(suffix)].strip("_")
                    return meta
    return meta


def iter_log_dirs(results_roots: Iterable[Path]) -> Iterable[Path]:
    for root in results_roots:
        if not root.exists():
            continue
        for log_dir in sorted(root.glob("*/logs")):
            if (log_dir / "eval_reward.npy").exists():
                yield log_dir


def infer_log_interval(iteration: Optional[np.ndarray], eval_reward: np.ndarray) -> int:
    if iteration is None or len(iteration) == 0 or len(eval_reward) == 0:
        return 5
    return max(1, int(round(len(iteration) / len(eval_reward))))


def tail_slice(values: np.ndarray, log_interval: int, last_iters: int) -> np.ndarray:
    if len(values) == 0:
        return values
    n = max(1, int(math.ceil(last_iters / max(log_interval, 1))))
    return values[-min(n, len(values)) :]


def align_per_iter(values: Optional[np.ndarray], n_iters: int) -> Optional[np.ndarray]:
    if values is None or n_iters <= 0:
        return None
    arr = np.asarray(values, dtype=float)
    if len(arr) == n_iters:
        return arr
    if len(arr) > n_iters:
        return arr[-n_iters:]
    pad = np.full(n_iters - len(arr), np.nan)
    return np.concatenate([pad, arr])


def mode_at_eval(mode_ids: Optional[np.ndarray], n_eval: int, log_interval: int) -> Optional[np.ndarray]:
    if mode_ids is None or len(mode_ids) == 0 or n_eval == 0:
        return None
    idx = np.arange(n_eval) * log_interval
    idx = np.minimum(idx, len(mode_ids) - 1)
    return np.asarray(mode_ids)[idx].astype(int)


def recovery_times(eval_reward: np.ndarray, mode_ids: Optional[np.ndarray],
                   log_interval: int, window: int = 10,
                   threshold: float = 0.9) -> List[int]:
    if mode_ids is None or len(mode_ids) < 2 or len(eval_reward) == 0:
        return []
    switches = np.where(np.diff(mode_ids) != 0)[0] + 1
    eval_iters = np.arange(len(eval_reward)) * log_interval
    if len(eval_reward) > window:
        kernel = np.ones(window) / window
        smoothed = np.convolve(eval_reward, kernel, mode="same")
    else:
        smoothed = eval_reward

    out: List[int] = []
    for sw in switches:
        pre_idx = np.searchsorted(eval_iters, sw) - 1
        if pre_idx < 1:
            continue
        pre_start = max(0, pre_idx - window)
        pre_perf = float(np.nanmean(smoothed[pre_start:pre_idx]))
        if pre_perf <= 0 or not np.isfinite(pre_perf):
            continue
        target = threshold * pre_perf
        post_idx = np.searchsorted(eval_iters, sw)
        found = None
        for j in range(post_idx, len(smoothed)):
            if smoothed[j] >= target:
                found = int(eval_iters[j] - sw)
                break
        out.append(found if found is not None else int(len(mode_ids) - sw))
    return out


def robust_zscore(signal: np.ndarray) -> np.ndarray:
    valid = signal[np.isfinite(signal)]
    if len(valid) == 0:
        return np.full_like(signal, np.nan, dtype=float)
    med = float(np.nanmedian(valid))
    mad = float(np.nanmedian(np.abs(valid - med)))
    scale = 1.4826 * mad if mad > 1e-9 else float(np.nanstd(valid))
    if scale <= 1e-9:
        scale = 1.0
    return (signal - med) / scale


def detection_metrics(log_dir: Path, mode_ids: Optional[np.ndarray],
                      window: int = 50, warmup: int = 50) -> Dict[str, object]:
    empty = {
        "switches": 0,
        "detections": 0,
        "precision": "",
        "recall": "",
        "median_delay": "",
        "mean_delay": "",
        "p90_delay": "",
        "false_positives": "",
        "false_pos_per_1k": "",
    }
    if mode_ids is None or len(mode_ids) < 2:
        return empty

    entropy = align_per_iter(load_metric(log_dir, "belief_entropy"), len(mode_ids))
    lamb = align_per_iter(load_metric(log_dir, "weighted_lambda"), len(mode_ids))
    eff = align_per_iter(load_metric(log_dir, "effective_window"), len(mode_ids))
    signals = []
    for arr in (entropy, lamb, eff):
        if arr is not None:
            signals.append(robust_zscore(arr))
    if not signals:
        return empty

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        signal = np.nanmax(np.vstack(signals), axis=0)
    signal[: min(warmup, len(signal))] = np.nan
    finite = signal[np.isfinite(signal)]
    if len(finite) == 0:
        return empty
    threshold = max(2.0, float(np.nanpercentile(finite, 85)))
    candidates = np.where(signal >= threshold)[0]
    if len(candidates) > 1:
        dedup = [int(candidates[0])]
        for c in candidates[1:]:
            if int(c) - dedup[-1] > max(3, window // 5):
                dedup.append(int(c))
        candidates = np.array(dedup, dtype=int)

    switches = np.where(np.diff(mode_ids) != 0)[0] + 1
    switches = switches[switches >= warmup]
    delays: List[int] = []
    matched_candidates = set()
    for sw in switches:
        hits = candidates[(candidates >= sw) & (candidates <= sw + window)]
        if len(hits) == 0:
            continue
        hit = int(hits[0])
        delays.append(hit - int(sw))
        matched_candidates.add(hit)

    false_pos = 0
    for cand in candidates:
        if len(switches) == 0 or np.min(np.abs(switches - cand)) > window:
            false_pos += 1

    precision = len(matched_candidates) / len(candidates) if len(candidates) else 0.0
    recall = len(delays) / len(switches) if len(switches) else 0.0
    per_1k = false_pos / max(len(mode_ids), 1) * 1000.0
    return {
        "switches": int(len(switches)),
        "detections": int(len(delays)),
        "precision": f"{precision:.3f}",
        "recall": f"{recall:.3f}",
        "median_delay": f"{np.median(delays):.1f}" if delays else "",
        "mean_delay": f"{np.mean(delays):.1f}" if delays else "",
        "p90_delay": f"{np.percentile(delays, 90):.1f}" if delays else "",
        "false_positives": int(false_pos),
        "false_pos_per_1k": f"{per_1k:.2f}",
    }


def summarize_run(log_dir: Path, last_iters: int) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, object]]:
    run_dir = log_dir.parent
    meta = parse_run_name(run_dir.name)
    eval_reward = load_metric(log_dir, "eval_reward")
    iteration = load_metric(log_dir, "iteration")
    mode_ids = load_metric(log_dir, "mode_id")
    if eval_reward is None:
        raise ValueError(f"missing eval_reward: {log_dir}")
    eval_reward = np.asarray(eval_reward, dtype=float)
    iteration = np.asarray(iteration, dtype=int) if iteration is not None else None
    mode_ids = np.asarray(mode_ids, dtype=int) if mode_ids is not None else None
    log_interval = infer_log_interval(iteration, eval_reward)
    tail = tail_slice(eval_reward, log_interval, last_iters)
    final_mean = float(np.nanmean(tail)) if len(tail) else float("nan")
    final_std = float(np.nanstd(tail)) if len(tail) else float("nan")

    runtime = load_metric(log_dir, "iter_time")
    runtime_hours = float(np.nansum(runtime) / 3600.0) if runtime is not None else float("nan")
    train_time = load_metric(log_dir, "train_time")
    eval_time = load_metric(log_dir, "eval_reward_std")

    base = {
        "run": run_dir.name,
        "tag": meta["tag"],
        "algo": meta["algo"],
        "env": meta["env"],
        "seed": meta["seed"],
        "dwell": meta["dwell"],
        "n_eval": len(eval_reward),
        "last_iter": int(iteration[-1]) if iteration is not None and len(iteration) else "",
        "complete_1500": bool(iteration is not None and len(iteration) and iteration[-1] >= 1499),
        "mean_last_iters": f"{final_mean:.3f}",
        "std_last_iters": f"{final_std:.3f}",
        "best_eval": f"{float(np.nanmax(eval_reward)):.3f}",
        "runtime_hours": f"{runtime_hours:.3f}" if np.isfinite(runtime_hours) else "",
        "mean_iter_seconds": f"{float(np.nanmean(runtime)):.3f}" if runtime is not None else "",
        "mean_train_seconds": f"{float(np.nanmean(train_time)):.3f}" if train_time is not None else "",
        "has_eval_std": bool(eval_time is not None),
    }

    tail_row = dict(base)
    eval_modes = mode_at_eval(mode_ids, len(eval_reward), log_interval)
    if eval_modes is not None and len(tail):
        tail_modes = eval_modes[-len(tail):]
        per_mode = []
        for mode in sorted(set(tail_modes.tolist())):
            vals = tail[tail_modes == mode]
            if len(vals):
                per_mode.append((int(mode), float(np.nanmean(vals))))
        if per_mode:
            means = np.array([x[1] for x in per_mode], dtype=float)
            worst_mode, worst_mean = min(per_mode, key=lambda x: x[1])
            tail_row.update({
                "mode_means": ";".join(f"{m}:{v:.1f}" for m, v in per_mode),
                "worst_mode": worst_mode,
                "worst_mode_mean": f"{worst_mean:.3f}",
                "mean_worst_gap": f"{float(np.nanmean(means) - worst_mean):.3f}",
            })
    if len(tail):
        q = max(1, int(math.ceil(len(tail) * 0.25)))
        tail_row["worst_quartile_mean"] = f"{float(np.nanmean(np.sort(tail)[:q])):.3f}"
    recoveries = recovery_times(eval_reward, mode_ids, log_interval)
    tail_row["recovery_n"] = len(recoveries)
    tail_row["recovery_median_iters"] = f"{np.median(recoveries):.1f}" if recoveries else ""
    tail_row["recovery_mean_iters"] = f"{np.mean(recoveries):.1f}" if recoveries else ""

    detect = dict(base)
    detect.update(detection_metrics(log_dir, mode_ids))
    return base, tail_row, detect


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


def aggregate_markdown(rows: List[Dict[str, object]], out_path: Path) -> None:
    grouped: Dict[Tuple[str, str, str], List[float]] = {}
    for row in rows:
        key = (str(row.get("tag", "")), str(row.get("env", "")), str(row.get("algo", "")))
        try:
            val = float(row["mean_last_iters"])
        except Exception:
            continue
        grouped.setdefault(key, []).append(val)

    lines = ["# BAPR Experiment Audit", ""]
    lines.append("| tag | env | algo | n | mean | std | min | max |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for (tag, env, algo), vals in sorted(grouped.items()):
        arr = np.asarray(vals, dtype=float)
        lines.append(
            f"| {tag} | {env} | {algo} | {len(arr)} | "
            f"{np.mean(arr):.1f} | {np.std(arr):.1f} | "
            f"{np.min(arr):.1f} | {np.max(arr):.1f} |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_roots", nargs="+",
                        default=["jax_experiments/results_paper"])
    parser.add_argument("--out_dir", default="paper/audit_tables")
    parser.add_argument("--last_iters", type=int, default=200)
    parser.add_argument("--tag_prefix", default="",
                        help="Optional filter for run tag prefix")
    args = parser.parse_args()

    roots = [Path(p) for p in args.results_roots]
    out_dir = Path(args.out_dir)
    run_rows: List[Dict[str, object]] = []
    tail_rows: List[Dict[str, object]] = []
    detect_rows: List[Dict[str, object]] = []

    for log_dir in iter_log_dirs(roots):
        meta = parse_run_name(log_dir.parent.name)
        if args.tag_prefix and not str(meta.get("tag", "")).startswith(args.tag_prefix):
            continue
        try:
            run, tail, detect = summarize_run(log_dir, args.last_iters)
        except Exception as exc:
            print(f"[skip] {log_dir}: {exc}")
            continue
        run_rows.append(run)
        tail_rows.append(tail)
        detect_rows.append(detect)

    write_csv(out_dir / "run_summary.csv", run_rows)
    write_csv(out_dir / "tail_regime_summary.csv", tail_rows)
    write_csv(out_dir / "bocd_detection_summary.csv", detect_rows)
    aggregate_markdown(run_rows, out_dir / "aggregate_summary.md")
    print(f"wrote {out_dir} ({len(run_rows)} runs)")


if __name__ == "__main__":
    main()
