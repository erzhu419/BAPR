#!/usr/bin/env python3
"""Batch-submit the three frozen mechanism-gap experiments using scheduler."""
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
import submit_bapr_v3_budget_matched_fork as common
from jax_experiments.analysis import mechanism_gap_protocol_v34 as p

PREFIX = "BAPR/mechanism-gap/v34"
GPU_NODES = ["jtl110gpu", "jtl110gpu2", "jtl311linux", "node007"]
CPU_NODES = [f"node00{index}" for index in range(1, 8)]
SMOKE_READY = p.BUNDLE_ROOT / "smoke/ready.json"
CPU_EVAL_PYTHON = "/home/zhengliang01/scheduleurm_work/conda_envs/bapr-v34-eval-0438/bin/python"


def command(mode, args, gpu, units=2100):
    prefix = (
        f"SCHEDULEURM_ETA_TOTAL_UNITS={units} "
        "JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 "
        "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
        if gpu else
        "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' "
        "XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4' "
    )
    return (prefix + "OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 "
            + f"PYTHONPATH={shlex.quote(str(ROOT))} "
            + f"{shlex.quote(str(common.JAX_PYTHON))} -u -m "
            + "jax_experiments.analysis.run_mechanism_gap_v34 "
            + shlex.join([mode, *args, "--resume"]) + " && echo DONE")


def spec(label, mode, args, output, gpu, dependencies=(), inputs=(), checkpoint=None, units=2100):
    result = {
        "description": label, "cmd": command(mode, args, gpu, units),
        "cwd": str(ROOT), "signature": f"{PREFIX}/{label}", "project": "BAPR",
        "vram": 1500 if gpu else 0, "ram_mb": 6144 if gpu else 8192,
        "cpu": 2 if gpu else 4, "priority": "normal",
        "allowed_nodes": GPU_NODES if gpu else CPU_NODES,
        "result_dir": str(output.parent), "local_result_dir": str(output.parent),
        "wait_for_files": [str(p.PREREG), *map(str, dependencies)],
        "stage_input_paths": list(map(str, inputs)), "stage_excludes": [
            "paper/", "jax_experiments/results_mechanism_gap_v34/",
            "jax_experiments/eval_bundles_mechanism_gap_v34/"],
        "reroute_on_node_down": True, "allow_initial_resume_scan_error": True,
    }
    if checkpoint is not None:
        result.update(ckpt_dir=str(checkpoint), ckpt_glob="train_state.pkl", resume_flag="", resume_managed_by_cmd=True)
    if not gpu:
        result.update(allow_cpu_training=True, cpu_training_justification="Frozen-policy evaluation or small JSON aggregation; no policy updates.")
    if mode == "analyze":
        result.update(cpu=1, ram_mb=2048)
    return result, output


def candidates():
    rows = [spec("smoke", "smoke", [], SMOKE_READY, True,
                 inputs=[p.RELEASE / "estimator", p.RELEASE / "reference_policies" / f"seed_{p.PROVIDER_SEED}"], units=1)]
    # Each node generates/caches the prescribed dataset locally. Within each
    # estimator pair, both models see exactly the same rows and minibatches.
    for seed in p.MODEL_SEEDS:
        directory = p.model_dir(seed)
        fitting = spec(f"identification-fit-{seed}", "fit", ["--seed", str(seed)], directory / "complete.json", True,
                       dependencies=[SMOKE_READY], inputs=[p.RELEASE / "reference_policies" / f"seed_{p.PROVIDER_SEED}"],
                       checkpoint=p.run_dir("identification", "paired", seed) / "checkpoints", units=p.MODEL_UPDATES)
        rows.append(fitting)
    for kind, seeds, arms, maximum in (
        ("hc", p.POLICY_SEEDS, ("reference", "conditioned"), p.HC_ITERS),
        ("reacher", p.REACHER_SEEDS, ("reference", "robust"), p.REACHER_ITERS),
    ):
        for seed in seeds:
            bundles = []
            for arm in arms:
                directory = p.bundle_dir(kind, arm, seed)
                bundles.append(directory)
                rows.append(spec(f"{kind}-train-{arm}-{seed}", "train", ["--kind", kind, "--arm", arm, "--seed", str(seed)],
                                 directory / "bundle.json", True, dependencies=[SMOKE_READY],
                                 inputs=[p.RELEASE / "estimator"] if arm == "conditioned" else [],
                                 checkpoint=p.run_dir(kind, arm, seed) / "checkpoints", units=maximum))
            inputs = [*bundles, *([p.RELEASE / "estimator"] if kind == "hc" else [])]
            rows.append(spec(f"{kind}-audit-{seed}", "audit", ["--kind", kind, "--seed", str(seed)], p.audit_path(kind, seed), False,
                             dependencies=[directory / "bundle.json" for directory in bundles], inputs=inputs))
    models = [p.model_dir(seed) for seed in p.MODEL_SEEDS]
    for seed in p.ABLATION_SEEDS:
        rows.append(spec(f"identification-audit-{seed}", "ablate", ["--seed", str(seed)], p.audit_path("identification", seed), False,
                         dependencies=[directory / "complete.json" for directory in models],
                         inputs=[*models, p.RELEASE / "estimator", p.RELEASE / "reference_policies" / f"seed_{seed}"]))
    for kind, seeds in (("identification", p.ABLATION_SEEDS), ("hc", p.POLICY_SEEDS), ("reacher", p.REACHER_SEEDS)):
        audits = [p.audit_path(kind, seed) for seed in seeds]
        rows.append(spec(f"{kind}-aggregate", "analyze", ["--kind", kind], p.analysis_path(kind), False,
                         dependencies=audits, inputs=[path.parent for path in audits]))
    return rows


def reacher_cpu_candidates():
    rows = []
    for seed in p.REACHER_SEEDS:
        bundles = []
        for arm in ("reference", "robust"):
            bundle = p.CPU_BUNDLE_ROOT / "reacher" / arm / f"seed_{seed}"
            bundles.append(bundle)
            rows.append(spec(
                f"reacher-cpu-train-{arm}-{seed}", "train",
                ["--kind", "reacher", "--arm", arm, "--seed", str(seed), "--sampling-device", "cpu"],
                bundle / "bundle.json", True, dependencies=[p.CPU_ADMISSION], inputs=[p.CPU_ADMISSION.parent],
                checkpoint=p.CPU_RUN_ROOT / "reacher" / arm / f"seed_{seed}" / "checkpoints",
                units=p.REACHER_ITERS))
        audit = p.CPU_RUN_ROOT / "audits/reacher" / f"seed_{seed}" / "audit.json"
        rows.append(spec(
            f"reacher-cpu-audit-{seed}", "audit",
            ["--kind", "reacher", "--seed", str(seed), "--sampling-device", "cpu"], audit, False,
            dependencies=[bundle / "bundle.json" for bundle in bundles], inputs=bundles))
    audits = [p.CPU_RUN_ROOT / "audits/reacher" / f"seed_{seed}" / "audit.json" for seed in p.REACHER_SEEDS]
    rows.append(spec(
        "reacher-cpu-aggregate", "analyze", ["--kind", "reacher", "--sampling-device", "cpu"],
        p.CPU_RUN_ROOT / "analysis/reacher/analysis.json", False,
        dependencies=audits, inputs=[audit.parent for audit in audits]))
    for row, output in rows:
        row["stage_excludes"] += [
            "jax_experiments/results_mechanism_gap_v34_reacher_cpu/",
            "jax_experiments/eval_bundles_mechanism_gap_v34_reacher_cpu/"]
        if row["vram"]:
            # node007's current JAX/Brax versions differ from this frozen group.
            row["allowed_nodes"] = ["jtl110gpu", "jtl110gpu2", "jtl311linux"]
            row["cmd"] = row["cmd"].replace("JAX_PLATFORMS=cuda ", "JAX_PLATFORMS=cuda,cpu ")
        else:
            row["cmd"] = row["cmd"].replace(str(common.JAX_PYTHON), CPU_EVAL_PYTHON)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--reacher-cpu", action="store_true", help="Submit the separately admitted, matched CPU-sampling group")
    args = parser.parse_args()
    known = common.scheduler_tasks()
    rows = reacher_cpu_candidates() if args.reacher_cpu else candidates()
    selected = [row for row, output in rows if not output.is_file() and not any(
        task.get("signature") == row["signature"] and task.get("status") in common.ACTIVE_STATUSES for task in known)]
    if args.dry_run:
        print(json.dumps({"total": len(selected), "gpu": sum(row["vram"] > 0 for row in selected),
                          "tasks": [{key: row[key] for key in ("signature", "allowed_nodes", "wait_for_files", "result_dir")} for row in selected]}, indent=2))
        return
    if args.reacher_cpu and not json.loads(p.CPU_ADMISSION.read_text())["passed"]:
        raise ValueError("CPU sampler admission must pass before submitting the replacement group")
    if not selected:
        print("No new mechanism-gap tasks")
        return
    response = subprocess.run([sys.executable, str(common.SCHEDULER), "submit-jsonl", "--stdin", "--trusted", "--json",
                               "--intent-label", "bapr-mechanism-gap-v34-submit", "--intent-ttl", "900"],
                              input="".join(json.dumps(row) + "\n" for row in selected), text=True,
                              capture_output=True, check=True, env=common.scheduler_env())
    payload = json.loads(response.stdout)
    ids = [str(row["id"]) for row in payload["submitted"]]
    if len(ids) != len(selected):
        raise RuntimeError(f"incomplete batch submission: {ids}")
    print("submitted:", ",".join(ids))
    if args.dispatch:
        subprocess.run([sys.executable, str(common.SCHEDULER), "dispatch", "--bulk-window",
                        "--intent-label", "bapr-mechanism-gap-v34-dispatch", "--intent-ttl", "900",
                        *[item for task_id in ids for item in ("--task-id", task_id)]], check=True, env=common.scheduler_env())


if __name__ == "__main__":
    main()
