"""Batch-submit the frozen 2x2 diagnostic, with smoke and compact-result gates."""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from scripts import submit_bapr_v3_budget_matched_fork as common
from jax_experiments.analysis import resac_update_density_protocol_v42 as p

PREFIX = "BAPR/resac-update-density/v42"


def spec(label, mode, output, gpu, *, arm=None, seed=None, waits=(), inputs=(),
         protocol=p, module="jax_experiments.analysis.run_resac_update_density_v42", prefix=PREFIX):
    p = protocol
    args = [mode, *(["--arm", arm] if arm else []), *(["--seed", str(seed)] if seed else [])]
    environment = "JAX_PLATFORMS=cuda " if gpu else "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' "
    flags = "--xla_gpu_enable_triton_gemm=false" if gpu else "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4"
    command = (environment + f"SCHEDULEURM_ETA_TOTAL_UNITS={250 if mode == 'train' else 1} "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.20 "
        "OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 "
        + f"XLA_FLAGS={shlex.quote(flags)} PYTHONPATH={shlex.quote(str(ROOT))} "
        + shlex.quote(p.GPU_PYTHON if gpu else p.CPU_PYTHON)
        + " -u -m " + module + " " + shlex.join(args) + " && echo DONE")
    high = mode == "smoke" or bool(arm and p.settings(arm)["updates_per_iter"] == 4000)
    row = dict(description=f"BAPR RE-SAC {p.PROTOCOL} {label}", project="BAPR", cwd=str(ROOT),
        cmd=command, signature=f"{prefix}/{label}", priority="normal", cpu=2 if gpu else 4,
        ram_mb=8192, vram=(1800 if high else 1500) if gpu else 0,
        allowed_nodes=list(p.GPU_NODES if gpu else p.CPU_NODES),
        wait_for_files=[str(p.PREREG), *map(str, waits)],
        stage_input_paths=[str(ROOT / "jax_experiments/analysis"), *map(str, inputs)],
        result_dir=str(output.parent), local_result_dir=str(output.parent), reroute_on_node_down=True,
        stage_excludes=["paper/", str(p.RUN_ROOT.relative_to(ROOT)) + "/",
                        str(p.BUNDLE_ROOT.relative_to(ROOT)) + "/"])
    if mode == "train":
        row.update(ckpt_dir=str(p.run_dir(arm, seed) / "checkpoints"), ckpt_glob="train_state.pkl",
                   resume_flag="", resume_managed_by_cmd=True, allow_initial_resume_scan_error=False)
    if not gpu:
        row.update(allow_cpu_training=True,
                   cpu_training_justification="Frozen actor evaluation or compact JSON aggregation; no training updates.")
    if mode == "analyze":
        row.update(cpu=1, ram_mb=2048)
    return row, output


def candidates(*, protocol=p, module="jax_experiments.analysis.run_resac_update_density_v42", prefix=PREFIX):
    p = protocol
    def make_spec(*args, **kwargs):
        return spec(*args, **kwargs, protocol=p, module=module, prefix=prefix)
    rows = [make_spec("smoke", "smoke", p.SMOKE, True)]
    for seed in p.SEEDS:
        actors = [p.bundle_dir(arm, seed) for arm in p.ARMS]
        for arm, actor in zip(p.ARMS, actors):
            rows.append(make_spec(f"train-{arm}-{seed}", "train", actor / "complete.json", True,
                             arm=arm, seed=seed, waits=[p.SMOKE]))
        rows.append(make_spec(f"audit-{seed}", "audit", p.audit_path(seed), False, seed=seed,
                         waits=[p.SMOKE, *[actor / "complete.json" for actor in actors]], inputs=actors))
    audits = [p.audit_path(seed) for seed in p.SEEDS]
    rows.append(make_spec("aggregate", "analyze", p.ANALYSIS, False, waits=audits,
                     inputs=[path.parent for path in audits]))
    return rows


def main(*, protocol=p, module="jax_experiments.analysis.run_resac_update_density_v42", prefix=PREFIX):
    p = protocol
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()
    known = common.scheduler_tasks()
    rows = [row for row, output in candidates(protocol=p, module=module, prefix=prefix) if not output.exists() and not any(
        task.get("signature") == row["signature"] and task.get("status") in common.ACTIVE_STATUSES
        for task in known)]
    if args.dry_run:
        print(json.dumps({"total": len(rows), "gpu": sum(row["vram"] > 0 for row in rows)}, indent=2))
        return
    if not p.PREREG.exists():
        raise FileNotFoundError("update-density preregistration is required")
    if not rows:
        print("No new update-density tasks")
        return
    response = subprocess.run([sys.executable, str(common.SCHEDULER), "submit-jsonl", "--stdin", "--trusted", "--json",
        "--intent-label", prefix.replace("/", "-") + "-submit", "--intent-ttl", "900"],
        input="".join(json.dumps(row) + "\n" for row in rows), text=True,
        capture_output=True, check=True, env=common.scheduler_env())
    ids = [str(task["id"]) for task in json.loads(response.stdout)["submitted"]]
    if len(ids) != len(rows):
        raise RuntimeError("incomplete density/norm batch submission")
    print("submitted:", ",".join(ids), flush=True)
    if args.dispatch:
        subprocess.run([sys.executable, str(common.SCHEDULER), "dispatch", "--bulk-window",
            "--intent-label", prefix.replace("/", "-") + "-dispatch", "--intent-ttl", "900",
            *[item for task in ids for item in ("--task-id", task)]], check=True, env=common.scheduler_env())


if __name__ == "__main__":
    main()
