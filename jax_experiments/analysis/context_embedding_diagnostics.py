"""Diagnose whether ESCP/BAPR context embeddings encode train tasks.

This is a post-training analysis.  It samples observations from each run's
saved replay buffer, forwards them through the saved context network, and
checks whether the embedding separates task ids and gravity scales.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path
from typing import Any

# Keep this analysis off the GPUs by default.  It must be set before importing
# JAX through final_task_sweep/train/checkpoint.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from jax_experiments.analysis.final_task_sweep import load_config
from jax_experiments.common.checkpoint import load_checkpoint
from jax_experiments.common.logging import Logger
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.train import make_algo, make_env


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = ROOT / "jax_experiments" / "results_evalfix_mild"
DEFAULT_OUT_PREFIX = ROOT / "reports" / "mild_context_embedding_diagnostics_2026-07-10"
ENVS = ["Ant", "HalfCheetah", "Hopper", "Walker2d"]
ALGOS = ["escp", "bapr"]


def run_name(prefix: str, algo: str, env: str) -> str:
    return f"{prefix}_{algo}_{env}_s0"


def gravity_scale(task: dict[str, Any]) -> float:
    gravity = np.asarray(task["gravity"], dtype=np.float64)
    return float(abs(np.min(gravity)) / 9.81)


def quantile_edges(values: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(values, 1.0 / 3.0)), float(np.quantile(values, 2.0 / 3.0))


def gravity_bins(scales: np.ndarray) -> np.ndarray:
    q1, q2 = quantile_edges(scales)
    bins = np.zeros(scales.shape, dtype=np.int32)
    bins[scales > q1] = 1
    bins[scales > q2] = 2
    return bins


def load_replay_sample(
    replay_path: Path,
    max_per_task: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    replay = np.load(replay_path)
    size = int(replay["size"])
    obs = np.asarray(replay["obs"][:size], dtype=np.float32)
    task_id = np.asarray(replay["task_id"][:size], dtype=np.int32)

    indices: list[np.ndarray] = []
    counts: dict[int, int] = {}
    for tid in sorted(int(x) for x in np.unique(task_id)):
        tid_idx = np.flatnonzero(task_id == tid)
        counts[tid] = int(len(tid_idx))
        if len(tid_idx) > max_per_task:
            tid_idx = rng.choice(tid_idx, size=max_per_task, replace=False)
        indices.append(tid_idx)
    if not indices:
        raise RuntimeError(f"empty replay buffer: {replay_path}")
    idx = np.concatenate(indices)
    rng.shuffle(idx)
    return obs[idx], task_id[idx], counts


def context_embeddings(agent, obs: np.ndarray, batch_size: int) -> np.ndarray:
    chunks = []
    for start in range(0, len(obs), batch_size):
        batch = jnp.asarray(obs[start:start + batch_size])
        chunks.append(np.asarray(agent.context_net(batch), dtype=np.float64))
    return np.concatenate(chunks, axis=0)


def centroid_stats(emb: np.ndarray, task_id: np.ndarray, n_tasks: int):
    centroids = np.full((n_tasks, emb.shape[1]), np.nan, dtype=np.float64)
    within_vars = []
    counts = np.zeros(n_tasks, dtype=np.int64)
    for tid in range(n_tasks):
        mask = task_id == tid
        counts[tid] = int(mask.sum())
        if counts[tid] == 0:
            continue
        vals = emb[mask]
        centroids[tid] = vals.mean(axis=0)
        within_vars.append(float(np.mean(np.sum((vals - centroids[tid]) ** 2, axis=1))))

    present = counts > 0
    centered_centroids = centroids[present] - np.nanmean(centroids[present], axis=0)
    between = float(np.mean(np.sum(centered_centroids ** 2, axis=1))) if present.any() else math.nan
    within = float(np.mean(within_vars)) if within_vars else math.nan
    sep_ratio = between / (within + 1e-12) if not math.isnan(within) else math.nan
    return centroids, counts, between, within, sep_ratio


def nearest_centroid_accuracy(
    emb: np.ndarray,
    labels: np.ndarray,
    n_tasks: int,
    gravity_bin_by_task: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    train_parts = []
    test_parts = []
    for tid in range(n_tasks):
        idx = np.flatnonzero(labels == tid)
        if len(idx) < 4:
            continue
        rng.shuffle(idx)
        cut = max(1, len(idx) // 2)
        train_parts.append(idx[:cut])
        test_parts.append(idx[cut:])
    if not train_parts or not test_parts:
        return math.nan, math.nan, 0

    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    centroids = np.full((n_tasks, emb.shape[1]), np.nan, dtype=np.float64)
    for tid in range(n_tasks):
        idx = train_idx[labels[train_idx] == tid]
        if len(idx):
            centroids[tid] = emb[idx].mean(axis=0)
    valid = ~np.isnan(centroids).any(axis=1)
    valid_ids = np.flatnonzero(valid)
    centroids_valid = centroids[valid]

    diff = emb[test_idx, None, :] - centroids_valid[None, :, :]
    pred_task = valid_ids[np.argmin(np.sum(diff * diff, axis=-1), axis=1)]
    true_task = labels[test_idx]
    task_acc = float(np.mean(pred_task == true_task))
    bin_acc = float(np.mean(gravity_bin_by_task[pred_task] == gravity_bin_by_task[true_task]))
    return task_acc, bin_acc, int(len(test_idx))


def linear_r2(features: np.ndarray, target: np.ndarray) -> float:
    valid = ~np.isnan(features).any(axis=1) & np.isfinite(target)
    x = features[valid]
    y = target[valid]
    if len(y) <= x.shape[1] + 1:
        return math.nan
    design = np.concatenate([np.ones((len(x), 1)), x], axis=1)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 1e-12:
        return math.nan
    return 1.0 - ss_res / ss_tot


def evaluate_run(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(run_dir)
    env = make_env(config, seed_offset=0)
    train_tasks = env.sample_tasks(config.task_num)
    scales = np.asarray([gravity_scale(t) for t in train_tasks], dtype=np.float64)
    scale_bins = gravity_bins(scales)

    agent = make_algo(config.algo, env.obs_dim, env.act_dim, config)
    replay_buffer = ReplayBuffer(
        env.obs_dim,
        env.act_dim,
        capacity=1,
        belief_dim=getattr(agent, "belief_dim", 0),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(tmpdir)
        start_iter, total_steps = load_checkpoint(
            str(run_dir / "checkpoints"),
            agent,
            replay_buffer,
            logger,
            config.algo,
            load_replay_buffer=False,
        )

    rng = np.random.default_rng(args.seed + hash(run_dir.name) % 1_000_000)
    obs, task_id, replay_counts = load_replay_sample(
        run_dir / "checkpoints" / "replay_buffer.npz",
        args.max_per_task,
        rng,
    )
    emb = context_embeddings(agent, obs, args.batch_size)
    centroids, counts, between, within, sep_ratio = centroid_stats(
        emb, task_id, int(config.task_num))
    task_acc, bin_acc, n_eval = nearest_centroid_accuracy(
        emb, task_id, int(config.task_num), scale_bins, rng)
    r2_log_scale = linear_r2(centroids, np.log(scales))

    centroid_norms = np.linalg.norm(centroids[~np.isnan(centroids).any(axis=1)], axis=1)
    total_std = np.std(emb, axis=0)
    centroid_std = np.nanstd(centroids, axis=0)
    collapsed = (
        float(np.max(total_std)) < 1e-3 or
        float(np.max(centroid_std)) < 1e-3 or
        sep_ratio < 1e-3
    )
    return {
        "run_name": run_dir.name,
        "env": config.env_name.replace("-v2", ""),
        "algo": config.algo,
        "seed": int(config.seed),
        "checkpoint_next_iter": int(start_iter),
        "checkpoint_total_steps": int(total_steps),
        "task_num": int(config.task_num),
        "tasks_with_replay": int(np.sum(counts > 0)),
        "sampled_transitions": int(len(obs)),
        "heldout_transitions": int(n_eval),
        "min_replay_per_task": int(min(replay_counts.values())),
        "median_replay_per_task": float(np.median(list(replay_counts.values()))),
        "max_replay_per_task": int(max(replay_counts.values())),
        "embedding_dim": int(emb.shape[1]),
        "embedding_between_trace": between,
        "embedding_within_trace": within,
        "embedding_separation_ratio": sep_ratio,
        "embedding_total_std_max": float(np.max(total_std)),
        "centroid_std_max": float(np.max(centroid_std)),
        "context_collapsed": bool(collapsed),
        "task_id_nearest_centroid_acc": task_acc,
        "gravity_bin_nearest_centroid_acc": bin_acc,
        "centroid_to_log_gravity_r2": r2_log_scale,
        "centroid_norm_mean": float(np.mean(centroid_norms)),
        "centroid_norm_std": float(np.std(centroid_norms)),
        "gravity_scale_min": float(np.min(scales)),
        "gravity_scale_max": float(np.max(scales)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "nan"
    try:
        v = float(value)
    except Exception:
        return str(value)
    if math.isnan(v):
        return "nan"
    return f"{v:.{digits}f}"


def make_report(path: Path, rows: list[dict[str, Any]], run_root: Path) -> None:
    lines = [
        "# Mild Context Embedding Diagnostics",
        "",
        f"Run root: `{run_root}`",
        "",
        "This checks whether the saved ESCP/BAPR context network separates replay observations by train task and gravity scale.",
        "",
        "## Summary Table",
        "",
        "| env | algo | collapsed | task acc | gravity-bin acc | log-gravity R2 | sep ratio | emb std max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda r: (r["env"], r["algo"])):
        r2 = "collapsed" if row["context_collapsed"] else fmt(row["centroid_to_log_gravity_r2"])
        lines.append(
            f"| {row['env']} | {row['algo']} | "
            f"{row['context_collapsed']} | "
            f"{fmt(row['task_id_nearest_centroid_acc'])} | "
            f"{fmt(row['gravity_bin_nearest_centroid_acc'])} | "
            f"{r2} | "
            f"{fmt(row['embedding_separation_ratio'])} | "
            f"{fmt(row['embedding_total_std_max'])} |"
        )

    lines.extend([
        "",
        "## Readout",
        "",
    ])
    for env in ENVS:
        env_rows = [r for r in rows if r["env"] == env]
        if not env_rows:
            continue
        by_algo = {r["algo"]: r for r in env_rows}
        escp = by_algo.get("escp")
        bapr = by_algo.get("bapr")
        if not escp or not bapr:
            continue
        lines.append(
            f"- {env}: ESCP task acc {fmt(escp['task_id_nearest_centroid_acc'])}, "
            f"BAPR task acc {fmt(bapr['task_id_nearest_centroid_acc'])}; "
            f"collapsed flags ESCP={escp['context_collapsed']}, "
            f"BAPR={bapr['context_collapsed']}."
        )

    lines.extend([
        "",
        "Interpretation rule: if `collapsed=True`, log-gravity R2 is numerically unreliable because the embedding has near-zero variance. High task accuracy with low gravity R2 would mean the context distinguishes arbitrary task identities but is not ordered by the physical parameter. Low task accuracy means the online latent itself is too noisy. If BAPR is weak despite ESCP/BAPR having separable context, the failure is downstream in policy/objective/gating rather than representation.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-prefix", type=str, default="mildfix_escp_mild_gravity")
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--envs", nargs="+", default=ENVS)
    parser.add_argument("--algos", nargs="+", default=ALGOS)
    parser.add_argument("--max-per-task", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()

    rows = []
    for env in args.envs:
        for algo in args.algos:
            run_dir = args.run_root / run_name(args.run_prefix, algo, env)
            if not run_dir.exists():
                print(f"skip missing run: {run_dir}")
                continue
            print(f"diagnosing {run_dir.name}")
            rows.append(evaluate_run(run_dir, args))

    csv_path = args.out_prefix.with_suffix(".csv")
    md_path = args.out_prefix.with_suffix(".md")
    write_csv(csv_path, rows)
    make_report(md_path, rows, args.run_root)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
