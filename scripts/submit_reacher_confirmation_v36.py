#!/usr/bin/env python3
"""Submit ten fresh controllers and their frozen-estimator confirmation DAG."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import submit_reacher_causal_v35 as legacy
from jax_experiments.analysis import reacher_confirmation_protocol_v36 as p

PREFIX = "BAPR/reacher-confirmation/v36"


def prepare_metadata():
    if not p.FROZEN_METADATA.is_file():
        metadata = {
            "model": json.loads((p.source.model_dir(p.PRIMARY_MODEL) / "complete.json").read_text()),
            "provider": json.loads((p.source.PROVIDER_BUNDLE / "bundle.json").read_text()),
        }
        p.FROZEN_METADATA.parent.mkdir(parents=True, exist_ok=True)
        p.FROZEN_METADATA.write_text(json.dumps(metadata, indent=2) + "\n")


def spec(label, mode, output, gpu, *, seed=None, arm=None, dependencies=(), inputs=(), checkpoint=None, units=1):
    row, unused = legacy.spec(label, mode, output, gpu, dependencies=dependencies,
                              inputs=[p.FROZEN_METADATA.parent, *inputs], checkpoint=checkpoint, units=units)
    prefix = row["cmd"].split(" -m ")[0]
    args = [mode, *(["--seed", str(seed)] if seed is not None else []),
            *(["--arm", arm] if arm is not None else [])]
    row.update(cmd=prefix + " -m jax_experiments.analysis.run_reacher_confirmation_v36 " + shlex.join(args) + " && echo DONE",
               signature=f"{PREFIX}/{label}",
               wait_for_files=[str(p.PREREG), str(p.FROZEN_METADATA), *map(str, dependencies)])
    row["stage_excludes"] += ["jax_experiments/results_reacher_confirmation_v36/",
                              "jax_experiments/eval_bundles_reacher_confirmation_v36/"]
    return row, output


def candidates():
    admission = legacy.legacy.p.CPU_ADMISSION
    model = p.source.model_dir(p.PRIMARY_MODEL)
    rows = [spec("smoke", "smoke", p.SMOKE, True,
                 dependencies=[admission, p.source.ANALYSIS, model / "complete.json"],
                 inputs=[admission.parent, model])]
    for seed in p.POLICY_SEEDS:
        actors = [p.screen_bundle(arm, seed) for arm in ("reference", "robust")]
        for arm, actor in zip(("reference", "robust"), actors, strict=True):
            rows.append(spec(f"train-{arm}-{seed}", "train", actor / "bundle.json", True,
                             seed=seed, arm=arm, dependencies=[p.SMOKE, admission],
                             inputs=[admission.parent], checkpoint=p.checkpoint_dir(arm, seed),
                             units=legacy.legacy.p.REACHER_ITERS))
        rows.append(spec(f"audit-{seed}", "audit", p.audit_path(seed), False, seed=seed,
                         dependencies=[p.SMOKE, *[actor / "bundle.json" for actor in actors]],
                         inputs=[model, *actors]))
    audits = [p.audit_path(seed) for seed in p.POLICY_SEEDS]
    rows.append(spec("aggregate", "analyze", p.ANALYSIS, False,
                     dependencies=audits, inputs=[path.parent for path in audits]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()
    if not json.loads(p.source.ANALYSIS.read_text())["causal_screen"]["passes"]:
        raise ValueError("V35 causal screen must pass before independent confirmation")
    common = legacy.legacy.common
    known = common.scheduler_tasks()
    rows = [row for row, output in candidates() if not output.is_file() and not any(
        task.get("signature") == row["signature"] and task.get("status") in common.ACTIVE_STATUSES for task in known)]
    if args.dry_run:
        print(json.dumps({"total": len(rows), "gpu": sum(row["vram"] > 0 for row in rows),
                          "tasks": [{key: row[key] for key in ("signature", "allowed_nodes", "wait_for_files")} for row in rows]}, indent=2))
        return
    if not rows:
        print("No new Reacher confirmation tasks")
        return
    prepare_metadata()
    response = subprocess.run([sys.executable, str(common.SCHEDULER), "submit-jsonl", "--stdin", "--trusted", "--json",
                               "--intent-label", "bapr-reacher-confirmation-v36-submit", "--intent-ttl", "900"],
                              input="".join(json.dumps(row) + "\n" for row in rows), text=True,
                              capture_output=True, check=True, env=common.scheduler_env())
    ids = [str(row["id"]) for row in json.loads(response.stdout)["submitted"]]
    if len(ids) != len(rows):
        raise RuntimeError("incomplete Reacher confirmation submission")
    print("submitted:", ",".join(ids), flush=True)
    if args.dispatch:
        subprocess.run([sys.executable, str(common.SCHEDULER), "dispatch", "--bulk-window",
                        "--intent-label", "bapr-reacher-confirmation-v36-dispatch", "--intent-ttl", "900",
                        *[item for task_id in ids for item in ("--task-id", task_id)]],
                       check=True, env=common.scheduler_env())


if __name__ == "__main__":
    main()
