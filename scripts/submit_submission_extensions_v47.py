"""Batch-submit three frozen publication extensions, with artifact dependencies."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from scripts import submit_bapr_v3_budget_matched_fork as scheduler
from jax_experiments.analysis import submission_extensions_protocol_v47 as p
from jax_experiments.analysis import mechanism_gap_protocol_v34 as mechanism
from jax_experiments.analysis import reacher_causal_protocol_v35 as calibration
from jax_experiments.analysis import reacher_information_confirmation_protocol_v38 as confirmation
from jax_experiments.analysis import regularized_comparison_protocol_v39 as runtime_comparison

PREFIX = "BAPR/submission-extensions/v47"


def spec(suite, label, mode, output, gpu, *, arm=None, seed=None, model=None,
         dependencies=(), inputs=(), checkpoint=None, units=1, node=None):
    source = suite == "source"
    python = (p.SOURCE_PYTHON if source else p.GPU_PYTHON) if gpu else p.CPU_PYTHON
    platforms = "cpu" if source or not gpu else "cuda,cpu"
    env = (f"JAX_PLATFORMS={platforms} XLA_PYTHON_CLIENT_PREALLOCATE=false "
           "OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 "
           f"SCHEDULEURM_ETA_TOTAL_UNITS={units} "
           "XLA_FLAGS='--xla_cpu_multi_thread_eigen=false --xla_gpu_enable_triton_gemm=false' ")
    if source and gpu:
        env += "LD_LIBRARY_PATH='' "
    if not gpu:
        env += "CUDA_VISIBLE_DEVICES='' "
    args = [mode]
    if source and mode == "setup":
        module, python = "scripts.prepare_resac_source_runtime_v47", p.GPU_PYTHON
        args = ["--node", node]
    else:
        module = "jax_experiments.analysis.run_resac_source_comparison_v47" if source else "jax_experiments.analysis.run_submission_extensions_v47"
        if not source:
            args += ["--suite", suite]
        for name, value in (("arm", arm), ("seed", seed), ("model", model)):
            if value is not None:
                args += ["--" + name, str(value)]
    row = {"project": "BAPR", "signature": f"{PREFIX}/{suite}/{label}",
        "description": f"BAPR {suite} extension {label}", "cwd": str(ROOT),
        "cmd": env + "PYTHONPATH=" + shlex.quote(str(ROOT)) + " " + shlex.quote(python)
               + " -u -m " + module + " " + shlex.join(args) + " && echo DONE",
        "allowed_nodes": [node] if node else list(p.SOURCE_GPU_NODES if source and gpu
                                                 else p.GPU_NODES if gpu else p.CPU_NODES),
        "cpu": 2 if gpu else 4, "ram_mb": 6144 if gpu else 8192,
        "vram": 1500 if gpu else 0, "priority": "normal", "reroute_on_node_down": True,
        "wait_for_files": [str(p.PREREG), *map(str, dependencies)],
        "stage_input_paths": [*map(str, inputs), *([str(p.SOURCE)] if source else [])],
        "stage_excludes": ["paper/", "jax_experiments/results_*/", "jax_experiments/eval_bundles_*/",
                           "reproducibility/lean/.lake/", "__pycache__/"],
        "result_dir": str(output.parent), "local_result_dir": str(output.parent)}
    if checkpoint is not None:
        row.update(ckpt_dir=str(checkpoint), ckpt_glob="train_state.pt" if source else "train_state.pkl",
                   resume_flag="", resume_managed_by_cmd=True, allow_initial_resume_scan_error=False)
    if not gpu:
        row.update(allow_cpu_training=True,
                   cpu_training_justification="Frozen-policy evaluation or compact JSON aggregation; no learner updates.")
    if mode == "analyze":
        row.update(cpu=1, ram_mb=2048)
    return row, output


def candidates():
    admission = mechanism.CPU_ADMISSION
    models = [calibration.model_dir(seed) for seed in p.REACHER_MODELS]
    rows = [spec("estimator", "smoke", "smoke", p.smoke("estimator"), True,
                 dependencies=[admission, *[path / "complete.json" for path in models]],
                 inputs=[admission.parent, *models])]
    for seed in p.REACHER_SEEDS:
        actors = [confirmation.screen_bundle(arm, seed) for arm in ("reference", "robust", "conditioned")]
        for model in p.REACHER_MODELS[1:]:
            directory = p.actor("estimator", "conditioned", seed, model)
            actors.append(directory)
            rows.append(spec("estimator", f"train-{model}-{seed}", "train", directory / "bundle.json", True,
                arm="conditioned", seed=seed, model=model, dependencies=[p.smoke("estimator"), admission],
                inputs=[admission.parent, calibration.model_dir(model)],
                checkpoint=p.RUN_ROOT / "estimator" / f"model_{model}/reacher/conditioned/seed_{seed}/checkpoints", units=400))
        rows.append(spec("estimator", f"audit-{seed}", "audit", p.audit_path("estimator", seed), False,
            seed=seed, dependencies=[p.smoke("estimator"), *[path / "bundle.json" for path in actors]], inputs=[*models, *actors]))
    hc_models = [p.hc_model_dir(seed) for seed in p.HC_MODELS]
    for seed in p.HC_SEEDS:
        actor = mechanism.RELEASE / "reference_policies" / f"seed_{seed}"
        rows.append(spec("hc-estimator", f"audit-{seed}", "audit", p.audit_path("hc-estimator", seed), False,
                         seed=seed, dependencies=[*[path / "complete.json" for path in hc_models]], inputs=[*hc_models, actor]))
    rows.append(spec("permutation", "smoke", "smoke", p.smoke("permutation"), True, inputs=[admission.parent]))
    provider = p.actor("permutation", "reference", p.PROVIDER_SEED)
    rows.append(spec("permutation", "provider", "train", provider / "bundle.json", True,
        arm="reference", seed=p.PROVIDER_SEED, dependencies=[p.smoke("permutation"), admission],
        inputs=[admission.parent], checkpoint=p.RUN_ROOT / f"permutation/reacher/reference/seed_{p.PROVIDER_SEED}/checkpoints", units=400))
    permutation_models = [p.model_dir(seed) for seed in p.PERMUTATION_MODELS]
    for model, directory in zip(p.PERMUTATION_MODELS, permutation_models, strict=True):
        rows.append(spec("permutation", f"fit-{model}", "fit", directory / "complete.json", True, model=model,
            dependencies=[provider / "bundle.json"], inputs=[provider],
            checkpoint=p.RUN_ROOT / f"permutation/estimators/seed_{model}/checkpoints", units=4500))
    for seed in p.PERMUTATION_SEEDS:
        actors = []
        for arm in ("reference", "robust", "conditioned"):
            actor = p.actor("permutation", arm, seed)
            actors.append(actor)
            conditioned = arm == "conditioned"
            rows.append(spec("permutation", f"train-{arm}-{seed}", "train", actor / "bundle.json", True,
                arm=arm, seed=seed, model=p.PERMUTATION_MODELS[0] if conditioned else None,
                dependencies=[p.smoke("permutation"), admission,
                              *([permutation_models[0] / "complete.json"] if conditioned else [])],
                inputs=[admission.parent, *([permutation_models[0]] if conditioned else [])],
                checkpoint=p.RUN_ROOT / f"permutation/reacher/{arm}/seed_{seed}/checkpoints", units=400))
        rows.append(spec("permutation", f"audit-{seed}", "audit", p.audit_path("permutation", seed), False,
            seed=seed, dependencies=[*[path / "bundle.json" for path in actors],
                                    *[path / "complete.json" for path in permutation_models]],
            inputs=[*actors, *permutation_models]))
    runtime_ready = []
    for node in ("jtl110gpu", "jtl110gpu2"):
        ready = p.BUNDLE_ROOT / "source/runtime" / node / "ready.json"
        runtime_ready.append(ready)
        rows.append(spec("source", f"runtime-{node}", "setup", ready, True, node=node,
                         inputs=[ROOT / "scripts", ROOT / "reproducibility/resac_source_comparison"]))
    rows.append(spec("source", "smoke", "smoke", p.smoke("source"), True,
                     dependencies=[path for path in runtime_ready if path.parent.name in p.SOURCE_GPU_NODES]))
    for seed in p.SOURCE_SEEDS:
        actors = []
        for arm in p.SOURCE_ARMS:
            actor = p.BUNDLE_ROOT / "source" / arm / f"seed_{seed}"
            actors.append(actor)
            rows.append(spec("source", f"train-{arm}-{seed}", "train", actor / "complete.json", True,
                arm=arm, seed=seed, dependencies=[p.smoke("source")],
                checkpoint=p.RUN_ROOT / f"source/{arm}/seed_{seed}/checkpoints", units=2100))
        reference = runtime_comparison.bundle_dir("reference", seed)
        rows.append(spec("source", f"audit-{seed}", "audit", p.audit_path("source", seed), False,
            seed=seed, dependencies=[*[path / "complete.json" for path in actors], reference / "complete.json"],
            inputs=[*actors, reference, mechanism.RELEASE / "estimator"]))
    for suite, seeds in (("estimator", p.REACHER_SEEDS), ("hc-estimator", p.HC_SEEDS),
                         ("permutation", p.PERMUTATION_SEEDS), ("source", p.SOURCE_SEEDS)):
        audits = [p.audit_path(suite, seed) for seed in seeds]
        rows.append(spec(suite, "aggregate", "analyze", p.analysis_path(suite), False,
                         dependencies=audits, inputs=[path.parent for path in audits]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()
    known = scheduler.scheduler_tasks() if not args.dry_run else []
    rows = [row for row, output in candidates() if not output.exists() and not any(
        task.get("signature") == row["signature"] and task.get("status") in scheduler.ACTIVE_STATUSES | {"done"} for task in known)]
    if args.dry_run:
        print(json.dumps({"total": len(rows), "gpu": sum(row["vram"] > 0 for row in rows),
                          "suites": dict(Counter(row["signature"].split("/")[3] for row in rows))}, indent=2))
        return
    if not p.PREREG.exists():
        raise FileNotFoundError(p.PREREG)
    if not rows:
        print("No new submission extension tasks")
        return
    result = subprocess.run([sys.executable, str(scheduler.SCHEDULER), "submit-jsonl", "--stdin", "--trusted", "--json",
        "--intent-label", "bapr-submission-v47-submit", "--intent-ttl", "900"],
        input="".join(json.dumps(row) + "\n" for row in rows), text=True, capture_output=True, check=True, env=scheduler.scheduler_env())
    ids = [str(task["id"]) for task in json.loads(result.stdout)["submitted"]]
    if len(ids) != len(rows):
        raise RuntimeError("incomplete extension batch submission")
    print("submitted:", ",".join(ids), flush=True)
    if args.dispatch:
        subprocess.run([sys.executable, str(scheduler.SCHEDULER), "dispatch", "--bulk-window",
            "--intent-label", "bapr-submission-v47-dispatch", "--intent-ttl", "900",
            *[item for task in ids for item in ("--task-id", task)]], check=True, env=scheduler.scheduler_env())


if __name__ == "__main__":
    main()
