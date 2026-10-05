#!/usr/bin/env python3
"""Submit three short trained-policy timing checks, not new scientific runs."""
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from scripts import submit_bapr_v3_budget_matched_fork as common
from jax_experiments.analysis import regularized_comparison_protocol_v39 as p


def candidates():
    for arm in ("sac", "escp", "resac_full"):
        output = ROOT / "jax_experiments/results_runtime_v39" / arm / "timing.json"
        command = (
            "JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false "
            "XLA_PYTHON_CLIENT_MEM_FRACTION=0.20 OMP_NUM_THREADS=2 "
            "OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 "
            "XLA_FLAGS='--xla_gpu_enable_triton_gemm=false' "
            f"PYTHONPATH={shlex.quote(str(ROOT))} {shlex.quote(p.GPU_PYTHON)} -u -m "
            "jax_experiments.analysis.benchmark_v39_sampling "
            f"--arm {arm} --output {shlex.quote(str(output))} && echo DONE")
        yield {"project": "BAPR", "description": f"BAPR sampler timing {arm}; zero training updates",
               "signature": f"BAPR/runtime-v39/collection/{arm}", "cmd": command, "cwd": str(ROOT),
               "cpu": 2, "ram_mb": 6144, "vram": 1500, "priority": "normal",
               "allowed_nodes": ["jtl110gpu", "jtl110gpu2"],
               "wait_for_files": [str(p.bundle_dir(arm, p.SEEDS[0]) / "complete.json")],
               "stage_input_paths": [str(ROOT / "jax_experiments/analysis"),
                                     str(ROOT / "jax_experiments/envs"), str(p.bundle_dir(arm, p.SEEDS[0]))],
               "result_dir": str(output.parent), "local_result_dir": str(output.parent),
               "reroute_on_node_down": True}, output


def main():
    queue = json.loads(common.QUEUE.read_text())["tasks"]
    rows = [row for row, output in candidates() if not output.exists() and not any(
        task.get("signature") == row["signature"] and task.get("status") in common.ACTIVE_STATUSES
        for task in queue)]
    if not rows:
        print("No new sampler benchmarks")
        return
    response = subprocess.run([sys.executable, str(common.SCHEDULER), "submit-jsonl", "--stdin", "--trusted", "--json"],
                              input="".join(json.dumps(row) + "\n" for row in rows), text=True,
                              capture_output=True, check=True, env=common.scheduler_env())
    ids = [str(row["id"]) for row in json.loads(response.stdout)["submitted"]]
    print("submitted:", ",".join(ids), flush=True)
    subprocess.run([sys.executable, str(common.SCHEDULER), "dispatch", "--bulk-window",
                    "--intent-label", "bapr-runtime-v39", "--intent-ttl", "900",
                    *[item for task in ids for item in ("--task-id", task)]],
                   check=True, env=common.scheduler_env())


if __name__ == "__main__":
    main()
