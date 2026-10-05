#!/usr/bin/env python3
"""Batch-submit per-seed runtime-matched jobs, gated on smoke and compact inputs."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from scripts import submit_bapr_v3_budget_matched_fork as common
from jax_experiments.analysis import regularized_comparison_protocol_v39 as p

PREFIX = "BAPR/regularized-comparison/v39"


def spec(label, mode, output, gpu, *, arm=None, seed=None, inputs=(), waits=()):
    args = [mode]
    if arm is not None:
        args += ["--arm", arm]
    if seed is not None:
        args += ["--seed", str(seed)]
    prefix = ("JAX_PLATFORMS=cuda " if gpu else "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' ")
    units = p.budget(arm)["next_iteration"] if mode == "train" else 1
    command = (prefix + f"SCHEDULEURM_ETA_TOTAL_UNITS={units} "
               "XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.20 "
               "OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 "
               + ("XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' " if gpu else
                  "XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4' ")
               + f"PYTHONPATH={shlex.quote(str(ROOT))} "
               + shlex.quote(p.GPU_PYTHON if gpu else p.CPU_PYTHON)
               + " -u -m jax_experiments.analysis.run_regularized_comparison_v39 "
               + shlex.join(args) + " && echo DONE")
    row = dict(description=f"BAPR matched-runtime {label}", cmd=command, cwd=str(ROOT),
               project="BAPR", signature=f"{PREFIX}/{label}", priority="normal",
               vram=1500 if gpu else 0, ram_mb=6144 if gpu else 8192,
               cpu=2 if gpu else 4, allowed_nodes=None if gpu else list(p.CPU_NODES),
               result_dir=str(output.parent), local_result_dir=str(output.parent),
               wait_for_files=[str(p.PREREG), *map(str, waits)],
               stage_input_paths=[str(ROOT / "jax_experiments/analysis"), *map(str, inputs)],
               reroute_on_node_down=True,
               stage_excludes=["paper/", "jax_experiments/results_regularized_comparison_v39/",
                               "jax_experiments/eval_bundles_regularized_comparison_v39/"])
    if mode == "train":
        row.update(ckpt_dir=str(p.run_dir(arm, seed) / "checkpoints"), ckpt_glob="train_state.pkl",
                   resume_flag="", resume_managed_by_cmd=True, allow_initial_resume_scan_error=False)
    if not gpu:
        row.update(allow_cpu_training=True,
                   cpu_training_justification="Strict frozen-policy evaluation or JSON aggregation; no training updates.")
    if mode == "analyze":
        row.update(cpu=1, ram_mb=2048)
    return row, output


def candidates():
    rows = [spec("smoke", "smoke", p.SMOKE, True)]
    for seed in p.SEEDS:
        for arm in p.TRAIN_ARMS:
            source = p.bundle_dir("source", seed)
            rows.append(spec(f"train-{arm}-{seed}", "train", p.bundle_dir(arm, seed) / "complete.json", True,
                             seed=seed, arm=arm, waits=[p.SMOKE, *([source / "complete.json"] if arm == "reference" else [])],
                             inputs=[source] if arm == "reference" else []))
        actors = [p.bundle_dir(arm, seed) for arm in p.TRAIN_ARMS]
        rows.append(spec(f"audit-{seed}", "audit", p.audit_path(seed), False, seed=seed,
                         inputs=[p.ESTIMATOR, *actors], waits=[p.SMOKE, *[actor / "complete.json" for actor in actors]]))
    paths = [p.audit_path(seed) for seed in p.SEEDS]
    rows.append(spec("aggregate", "analyze", p.ANALYSIS, False,
                     inputs=[path.parent for path in paths], waits=paths))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    known = json.loads(common.QUEUE.read_text())["tasks"]
    rows = [row for row, output in candidates() if not output.is_file() and not any(
        task.get("signature") == row["signature"] and task.get("status") in common.ACTIVE_STATUSES for task in known)]
    if args.smoke_only:
        rows = [row for row in rows if row["signature"].endswith("/smoke")]
    if args.dry_run:
        print(json.dumps({"total": len(rows), "gpu": sum(row["vram"] > 0 for row in rows),
                          "tasks": [{key: row[key] for key in ("signature", "allowed_nodes", "wait_for_files")} for row in rows]}, indent=2))
        return
    if not p.PREREG.is_file():
        raise FileNotFoundError("comparison preregistration is required")
    if not rows:
        print("No new runtime-matched comparison tasks")
        return
    response = subprocess.run([sys.executable, str(common.SCHEDULER), "submit-jsonl", "--stdin", "--trusted", "--json",
                               "--intent-label", "bapr-regularized-v39-submit", "--intent-ttl", "900"],
                              input="".join(json.dumps(row) + "\n" for row in rows), text=True,
                              capture_output=True, check=True, env=common.scheduler_env())
    ids = [str(row["id"]) for row in json.loads(response.stdout)["submitted"]]
    if len(ids) != len(rows):
        raise RuntimeError("incomplete comparison batch submission")
    print("submitted:", ",".join(ids), flush=True)
    if args.dispatch:
        subprocess.run([sys.executable, str(common.SCHEDULER), "dispatch", "--bulk-window",
                        "--intent-label", "bapr-regularized-v39-dispatch", "--intent-ttl", "900",
                        *[item for task in ids for item in ("--task-id", task)]],
                       check=True, env=common.scheduler_env())


if __name__ == "__main__":
    main()
