"""Submit a native Torch smoke, then six independent calibration tasks."""
import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from scripts import submit_bapr_v3_budget_matched_fork as scheduler
from jax_experiments.analysis import resac_torch_native_protocol_v43 as p

PREFIX = "BAPR/resac-torch-native/v43"


def prepare_source():
    if (p.SOURCE / "halfcheetah.yaml").exists():
        return
    mine = ROOT.parent
    p.SOURCE.mkdir(parents=True, exist_ok=True)
    shutil.copytree(mine / "dsac/rlkit", p.SOURCE / "rlkit",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("resac.py", "resac_networks.py", "resac_trainer.py", "gym_patch.py"):
        shutil.copy2(mine / "RE-SAC/experiment" / name, p.SOURCE / name)
    shutil.copy2(mine / "RE-SAC/experiment/configs/resac/halfcheetah.yaml", p.SOURCE / "halfcheetah.yaml")


def spec(label, mode, output, arm=None, seed=None, protocol=p,
         module="jax_experiments.analysis.run_resac_torch_native_v43", prefix=PREFIX):
    gpu = mode != "analyze"
    python = protocol.PYTHON if gpu else "/home/erzhu419/.conda/envs/resac-jax/bin/python"
    args = [mode, *(["--arm", arm] if arm else []), *(["--seed", str(seed)] if seed else [])]
    env = ("LD_LIBRARY_PATH=/home/erzhu419/.mujoco/mujoco210/bin:/usr/lib/nvidia "
           "CPATH=/home/erzhu419/.venvs/resac-torch-native-v43/usr/include "
           "LIBRARY_PATH=/home/erzhu419/.venvs/resac-torch-native-v43/usr/lib/x86_64-linux-gnu "
           "MUJOCO_PY_MUJOCO_PATH=/home/erzhu419/.mujoco/mujoco210 "
           "OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 "
           f"SCHEDULEURM_ETA_TOTAL_UNITS={1000 if mode == 'train' else 1} ") if gpu else "CUDA_VISIBLE_DEVICES='' "
    row = {"description": f"BAPR native Torch stationary calibration {label}", "project": "BAPR",
        "cwd": str(ROOT), "signature": prefix + "/" + label,
        "cmd": env + "PYTHONPATH=" + shlex.quote(str(ROOT)) + " " + shlex.quote(python)
               + " -u -m " + shlex.quote(module) + " " + shlex.join(args) + " && echo DONE",
        "allowed_nodes": list(protocol.GPU_NODES if gpu else ("node001", "node002", "node003", "node004", "node005", "node006", "node007")),
        "cpu": 4 if gpu else 1, "ram_mb": 6144 if gpu else 2048, "vram": 1500 if gpu else 0,
        "wait_for_files": [str(protocol.PREREG)], "priority": "normal", "reroute_on_node_down": True,
        "stage_input_paths": [str(protocol.SOURCE), str(ROOT / "jax_experiments/analysis")],
        "result_dir": str(output.parent), "local_result_dir": str(output.parent),
        "stage_excludes": ["paper/", str(protocol.RUN_ROOT.relative_to(ROOT)) + "/",
                           str(protocol.COMPACT.relative_to(ROOT)) + "/"]}
    if mode == "train":
        row.update(wait_for_files=[str(protocol.PREREG), str(protocol.SMOKE)],
            ckpt_dir=str(protocol.directory(arm, seed) / "checkpoints"), ckpt_glob="train_state.pt",
            resume_flag="", resume_managed_by_cmd=True, allow_initial_resume_scan_error=False)
    if mode == "analyze":
        inputs = [protocol.result(arm, seed) for seed in protocol.SEEDS for arm in protocol.ARMS]
        row.update(wait_for_files=[str(protocol.PREREG), *map(str, inputs)],
            stage_input_paths=[*row["stage_input_paths"], *[str(path.parent) for path in inputs]],
            allow_cpu_training=True, cpu_training_justification="Compact native-calibration JSON aggregation only; no policy updates.")
    return row, output


def candidates(smoke_only=False):
    rows = [spec("smoke", "smoke", p.SMOKE)]
    if not smoke_only:
        for seed in p.SEEDS:
            for arm in p.ARMS:
                rows.append(spec(f"train-{arm}-{seed}", "train", p.result(arm, seed), arm, seed))
        rows.append(spec("aggregate", "analyze", p.COMPACT / "analysis/analysis.json"))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()
    prepare_source()
    known = scheduler.scheduler_tasks()
    rows = [row for row, output in candidates(args.smoke_only) if not output.exists() and not any(
        task.get("signature") == row["signature"] and task.get("status") in scheduler.ACTIVE_STATUSES for task in known)]
    if args.dry_run:
        print(json.dumps({"total": len(rows), "gpu": sum(row["vram"] > 0 for row in rows)}, indent=2))
        return
    if not p.PREREG.exists():
        raise FileNotFoundError(p.PREREG)
    if not args.smoke_only and not p.SMOKE.exists():
        raise RuntimeError("run native smoke before submitting production calibration")
    if not rows:
        print("No new native Torch tasks")
        return
    result = subprocess.run([sys.executable, str(scheduler.SCHEDULER), "submit-jsonl", "--stdin", "--trusted", "--json",
        "--intent-label", "bapr-native-v43-submit", "--intent-ttl", "900"],
        input="".join(json.dumps(row) + "\n" for row in rows), text=True, capture_output=True, check=True, env=scheduler.scheduler_env())
    ids = [str(task["id"]) for task in json.loads(result.stdout)["submitted"]]
    print("submitted:", ",".join(ids), flush=True)
    if args.dispatch:
        subprocess.run([sys.executable, str(scheduler.SCHEDULER), "dispatch", "--bulk-window",
            "--intent-label", "bapr-native-v43-dispatch", "--intent-ttl", "900",
            *[item for task in ids for item in ("--task-id", task)]], check=True, env=scheduler.scheduler_env())


if __name__ == "__main__":
    main()
