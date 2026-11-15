#!/usr/bin/env python3
"""Submit the frozen Reacher causal-estimator DAG with scheduler."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import submit_mechanism_gap_v34 as legacy
from jax_experiments.analysis import reacher_causal_protocol_v35 as p

PREFIX = "BAPR/reacher-causal/v35"
GPU_NODES = ["jtl110gpu", "jtl110gpu2", "jtl311linux"]


def spec(label, mode, output, gpu, *, seed=None, dependencies=(), inputs=(), checkpoint=None, units=1):
    row, unused = legacy.spec(label, mode, [], output, gpu, dependencies, inputs, checkpoint, units)
    prefix = row["cmd"].split(" -m ")[0]
    if gpu:
        prefix = prefix.replace("JAX_PLATFORMS=cuda ", "JAX_PLATFORMS=cuda,cpu ")
    else:
        prefix = prefix.replace(str(legacy.common.JAX_PYTHON), legacy.CPU_EVAL_PYTHON)
    args = [mode, *(["--seed", str(seed)] if seed is not None else [])]
    row.update(cmd=prefix + " -m jax_experiments.analysis.run_reacher_causal_v35 " + shlex.join(args) + " && echo DONE",
               signature=f"{PREFIX}/{label}", allowed_nodes=GPU_NODES if gpu else legacy.CPU_NODES,
               wait_for_files=[str(p.PREREG), *map(str, dependencies)])
    row["stage_excludes"] += ["reproducibility/",
                              "jax_experiments/results_mechanism_gap_v34_reacher_cpu/",
                              "jax_experiments/eval_bundles_mechanism_gap_v34_reacher_cpu/",
                              "jax_experiments/results_reacher_causal_v35/",
                              "jax_experiments/eval_bundles_reacher_causal_v35/"]
    if mode in ("collect", "fit"):
        # Calibration arrays stay on this node; all fits consume identical rows.
        row.update(allowed_nodes=[p.DATA_NODE], require_node=p.DATA_NODE, reroute_on_node_down=False)
    if mode == "collect":
        row["vram"] = 512
    return row, output


def candidates():
    rows = [spec("smoke", "smoke", p.SMOKE, True)]
    rows.append(spec("provider", "provider", p.PROVIDER_BUNDLE / "bundle.json", True,
                     dependencies=[p.SMOKE, p.HEADROOM, legacy.p.CPU_ADMISSION],
                     inputs=[legacy.p.CPU_ADMISSION.parent],
                     checkpoint=p.PROVIDER_RUN_ROOT / "reacher/reference" / f"seed_{p.PROVIDER_SEED}" / "checkpoints",
                     units=legacy.p.REACHER_ITERS))
    rows.append(spec("collect", "collect", p.DATA_MANIFEST, True,
                     dependencies=[p.SMOKE, p.PROVIDER_BUNDLE / "bundle.json"], inputs=[p.PROVIDER_BUNDLE]))
    for seed in p.MODEL_SEEDS:
        rows.append(spec(f"fit-{seed}", "fit", p.model_dir(seed) / "complete.json", True, seed=seed,
                         dependencies=[p.SMOKE, p.DATA_MANIFEST], inputs=[p.DATA_MANIFEST.parent],
                         checkpoint=p.checkpoint_dir(seed), units=p.MODEL_UPDATES))
    models = [p.model_dir(seed) for seed in p.MODEL_SEEDS]
    for seed in p.POLICY_SEEDS:
        actors = [p.screen_bundle(arm, seed) for arm in ("reference", "robust")]
        rows.append(spec(f"audit-{seed}", "audit", p.audit_path(seed), False, seed=seed,
                         dependencies=[*[model / "complete.json" for model in models],
                                       *[actor / "bundle.json" for actor in actors]], inputs=[*models, *actors]))
    audits = [p.audit_path(seed) for seed in p.POLICY_SEEDS]
    rows.append(spec("aggregate", "analyze", p.ANALYSIS, False,
                     dependencies=audits, inputs=[path.parent for path in audits]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()
    if not json.loads(p.HEADROOM.read_text())["headroom_screen"]["passes"]:
        raise ValueError("Reacher oracle headroom must pass before estimator training")
    known = legacy.common.scheduler_tasks()
    rows = [row for row, output in candidates() if not output.is_file() and not any(
        task.get("signature") == row["signature"] and task.get("status") in legacy.common.ACTIVE_STATUSES for task in known)]
    if args.dry_run:
        print(json.dumps({"total": len(rows), "gpu": sum(row["vram"] > 0 for row in rows),
                          "tasks": [{key: row[key] for key in ("signature", "allowed_nodes", "wait_for_files")} for row in rows]}, indent=2))
        return
    if not rows:
        print("No new Reacher causal tasks")
        return
    response = subprocess.run([sys.executable, str(legacy.common.SCHEDULER), "submit-jsonl", "--stdin", "--trusted", "--json",
                               "--intent-label", "bapr-reacher-causal-v35-submit", "--intent-ttl", "900"],
                              input="".join(json.dumps(row) + "\n" for row in rows), text=True,
                              capture_output=True, check=True, env=legacy.common.scheduler_env())
    ids = [str(row["id"]) for row in json.loads(response.stdout)["submitted"]]
    if len(ids) != len(rows):
        raise RuntimeError("incomplete Reacher batch submission")
    print("submitted:", ",".join(ids), flush=True)
    if args.dispatch:
        subprocess.run([sys.executable, str(legacy.common.SCHEDULER), "dispatch", "--bulk-window",
                        "--intent-label", "bapr-reacher-causal-v35-dispatch", "--intent-ttl", "900",
                        *[item for task_id in ids for item in ("--task-id", task_id)]],
                       check=True, env=legacy.common.scheduler_env())


if __name__ == "__main__":
    main()
