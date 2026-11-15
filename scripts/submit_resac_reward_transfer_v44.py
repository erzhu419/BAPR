"""Submit the native reward-transfer smoke and fresh-seed factorial calibration."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from scripts import submit_resac_torch_native_v43 as native
from jax_experiments.analysis import resac_reward_transfer_protocol_v44 as p

PREFIX = "BAPR/resac-reward-transfer/v44"
MODULE = "jax_experiments.analysis.run_resac_reward_transfer_v44"


def candidates(smoke_only=False):
    def spec(label, mode, output, arm=None, seed=None):
        row, output = native.spec(label, mode, output, arm, seed, protocol=p, module=MODULE, prefix=PREFIX)
        row["description"] = f"BAPR native reward-transfer {label}"
        row["vram"] = 600 if mode != "analyze" else 0
        return row, output
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
    native.prepare_source()
    known = native.scheduler.scheduler_tasks()
    rows = [row for row, output in candidates(args.smoke_only) if not output.exists() and not any(
        task.get("signature") == row["signature"] and task.get("status") in native.scheduler.ACTIVE_STATUSES for task in known)]
    if args.dry_run:
        print(json.dumps({"total": len(rows), "gpu": sum(row["vram"] > 0 for row in rows)}, indent=2))
        return
    if not p.PREREG.exists():
        raise FileNotFoundError(p.PREREG)
    if not args.smoke_only and not p.SMOKE.exists():
        raise RuntimeError("run reward-transform/resume smoke before production submission")
    if not rows:
        print("No new reward-transfer tasks")
        return
    result = subprocess.run([sys.executable, str(native.scheduler.SCHEDULER), "submit-jsonl", "--stdin", "--trusted", "--json",
        "--intent-label", "bapr-reward-v44-submit", "--intent-ttl", "900"],
        input="".join(json.dumps(row) + "\n" for row in rows), text=True, capture_output=True, check=True,
        env=native.scheduler.scheduler_env())
    ids = [str(task["id"]) for task in json.loads(result.stdout)["submitted"]]
    print("submitted:", ",".join(ids), flush=True)
    if args.dispatch:
        subprocess.run([sys.executable, str(native.scheduler.SCHEDULER), "dispatch", "--bulk-window",
            "--intent-label", "bapr-reward-v44-dispatch", "--intent-ttl", "900",
            *[item for task in ids for item in ("--task-id", task)]], check=True, env=native.scheduler.scheduler_env())


if __name__ == "__main__":
    main()
