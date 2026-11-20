"""Submit the JMLR Brax mechanism extension (v50) and MuJoCo v5 multi-morphology suite (v51-samestep)."""
from __future__ import annotations

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
import submit_bapr_v3_budget_matched_fork as scheduler

PREREG = ROOT / "reports/jmlr_upgrade_v50_v51_protocol_2026-11-16.md"
GPU_NODES = ("jtl110gpu", "jtl110gpu2", "node007")
CPU_AUDIT_NODES = ("node001", "node002", "node003", "node004", "node005", "node006", "jtl110gpu", "jtl110gpu2", "node007")

# Remote shell expands $HOME on each worker node without hardcoding usernames in git
BRAX_GPU_PYTHON_EXPR = "$HOME/.venvs/resac-jax-gpu1-0438/bin/python"
MUJOCO_GPU_ENV_EXPR = "$HOME/scheduleurm_work/conda_envs/csbapr-gpu-py310"

V50_SEEDS = (88003, 88021, 88039, 88057, 88079, 88103, 88121, 88139, 88157, 88179)
V50_OUT = ROOT / "reproducibility/jmlr_brax_mechanism_v50"

V51_ENVS = ("halfcheetah", "ant", "walker2d", "hopper")
V51_SEEDS = (51003, 51021, 51039, 51057, 51079)
V51_INDEP_ARMS = ("bapr", "bapr_noburst", "sac", "escp256", "rma")
V51_RUN_ROOT = ROOT / "jax_experiments/results_jmlr_mujoco_v51_samestep"
V51_BUNDLE_ROOT = ROOT / "jax_experiments/eval_bundles_jmlr_mujoco_v51_samestep"
V51_RELEASE = ROOT / "reproducibility/jmlr_mujoco_v51"
V51_CHECKPOINT_NAME = "training_checkpoint.npz"

STAGE_EXCLUDES = [
    "paper/",
    "gemini_bundle/",
    "jax_experiments/results_*/",
    "jax_experiments/eval_bundles_v*/",
    "jax_experiments/eval_bundles_jmlr_mujoco_v51*/",
    "jax_experiments/eval_bundles_mechanism*/",
    "jax_experiments/eval_bundles_reacher*/",
    "jax_experiments/eval_bundles_submission*/",
    "reproducibility/lean/.lake/",
    "__pycache__/",
]


def make_v50_spec(label: str, mode: str, output: Path, *, seed: int | None = None,
                  deps: list[Path] = (), inputs: list[Path] = (), gpu: bool = True) -> tuple[dict, Path]:
    platforms = "cuda,cpu" if gpu else "cpu"
    env = (
        f"JAX_PLATFORMS={platforms} XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "XLA_FLAGS='--xla_cpu_multi_thread_eigen=false --xla_gpu_enable_triton_gemm=false' "
    )
    if not gpu:
        env += "CUDA_VISIBLE_DEVICES='' "
    args = [mode] + (["--seed", str(seed)] if seed is not None else [])
    cmd = (
        env + f"PYTHONPATH={shlex.quote(str(ROOT))} "
        f'"{BRAX_GPU_PYTHON_EXPR}" -u -m jax_experiments.analysis.run_jmlr_brax_mechanism_v50 '
        + shlex.join(args) + " && echo DONE"
    )
    row = {
        "project": "BAPR",
        "signature": f"BAPR/jmlr-v50/{label}",
        "description": f"BAPR JMLR v50 Brax mechanism {label}",
        "cwd": str(ROOT),
        "cmd": cmd,
        "allowed_nodes": list(GPU_NODES),
        "cpu": 1,
        "ram_mb": 4096 if gpu else 2048,
        "vram": 950 if gpu else 0,
        "priority": "high",
        "reroute_on_node_down": True,
        "wait_for_files": [str(PREREG), *map(str, deps)],
        "stage_input_paths": [str(p) for p in inputs],
        "stage_excludes": STAGE_EXCLUDES,
        "result_dir": str(output.parent),
        "local_result_dir": str(output.parent),
        "allow_no_ckpt": True,
    }
    if not gpu:
        row.update(
            allow_cpu_training=True,
            cpu_training_justification="JSON reduction over completed per-seed audits; no model updates.",
        )
    return row, output


def make_v51_spec(label: str, mode: str, output: Path, *, env_key: str = "halfcheetah",
                  arm: str = "bapr", seed: int = 51003, deps: list[Path] = (),
                  inputs: list[Path] = (), ckpt: Path | None = None, gpu: bool = True) -> tuple[dict, Path]:
    env_prefix = (
        f'SITE_DIR="{MUJOCO_GPU_ENV_EXPR}/lib/python3.10/site-packages" && '
        'NVLIBS=$(printf "%s:" "$SITE_DIR"/nvidia/*/lib) && '
        'LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}" '
        'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 '
    )
    if not gpu:
        env_prefix += "CUDA_VISIBLE_DEVICES='' "
    args = [mode, "--env", env_key, "--arm", arm, "--seed", str(seed)]
    cmd = (
        env_prefix + f"PYTHONPATH={shlex.quote(str(ROOT))} "
        f'"{MUJOCO_GPU_ENV_EXPR}/bin/python" -u -m jax_experiments.analysis.run_jmlr_mujoco_suite_v51 '
        + shlex.join(args) + " && echo DONE"
    )
    row = {
        "project": "BAPR",
        "signature": f"BAPR/jmlr-v51-samestep/{label}",
        "description": f"BAPR JMLR v51-samestep MuJoCo {label}",
        "cwd": str(ROOT),
        "cmd": cmd,
        "allowed_nodes": list(GPU_NODES if gpu else CPU_AUDIT_NODES),
        "cpu": 1 if gpu else 2,
        "ram_mb": 4096 if gpu else 3072,
        "vram": 900 if gpu else 0,
        "priority": "high",
        "reroute_on_node_down": True,
        "wait_for_files": [str(PREREG), *map(str, deps)],
        "stage_input_paths": [str(p) for p in inputs],
        "stage_excludes": STAGE_EXCLUDES,
        "result_dir": str(output.parent),
        "local_result_dir": str(output.parent),
    }
    if ckpt is not None:
        row.update(
            ckpt_dir=str(ckpt),
            ckpt_glob=V51_CHECKPOINT_NAME,
            resume_flag="",
            resume_managed_by_cmd=True,
            allow_initial_resume_scan_error=True,
        )
    else:
        row["allow_no_ckpt"] = True
    if not gpu:
        row.update(
            allow_cpu_training=True,
            cpu_training_justification="Deterministic rollout audit or JSON aggregation; no gradient updates.",
        )
    return row, output


def build_candidates() -> list[tuple[dict, Path]]:
    rows: list[tuple[dict, Path]] = []

    # --- Suite v50: 10 frozen Brax seeds + 1 aggregate ---
    v32_est = ROOT / "reproducibility/v32/estimator"
    v50_seed_outs = []
    for s in V50_SEEDS:
        s_out = V50_OUT / "seeds" / f"seed_{s}.json"
        v50_seed_outs.append(s_out)
        ref_pol = ROOT / f"reproducibility/v32/reference_policies/seed_{s}"
        rows.append(make_v50_spec(f"seed-{s}", "seed", s_out, seed=s, inputs=[v32_est, ref_pol], gpu=True))
    rows.append(make_v50_spec(
        "aggregate", "analyze", V50_OUT / "analysis.json",
        deps=v50_seed_outs, inputs=[V50_OUT / "seeds"], gpu=False,
    ))

    # --- Suite v51-samestep: 4 MuJoCo v5 envs x 5 seeds x (6 train arms + 1 audit) + 1 aggregate ---
    all_audit_outs = []
    for env_key in V51_ENVS:
        for s in V51_SEEDS:
            bapr_bundle = V51_BUNDLE_ROOT / env_key / "bapr" / f"seed_{s}"
            arm_bundles = []
            for arm in V51_INDEP_ARMS:
                b_dir = V51_BUNDLE_ROOT / env_key / arm / f"seed_{s}"
                arm_bundles.append(b_dir)
                ckpt = V51_RUN_ROOT / env_key / arm / f"seed_{s}" / "checkpoints"
                rows.append(make_v51_spec(
                    f"{env_key}-train-{arm}-{s}", "train", b_dir / "bundle.json",
                    env_key=env_key, arm=arm, seed=s, ckpt=ckpt, gpu=True,
                ))
            # belief_sac waits on same-seed bapr/inverse.pt!
            bsac_dir = V51_BUNDLE_ROOT / env_key / "belief_sac" / f"seed_{s}"
            arm_bundles.append(bsac_dir)
            bsac_ckpt = V51_RUN_ROOT / env_key / "belief_sac" / f"seed_{s}" / "checkpoints"
            rows.append(make_v51_spec(
                f"{env_key}-train-belief_sac-{s}", "train", bsac_dir / "bundle.json",
                env_key=env_key, arm="belief_sac", seed=s,
                deps=[bapr_bundle / "bundle.json"], inputs=[bapr_bundle], ckpt=bsac_ckpt, gpu=True,
            ))
            # Per-seed audit waits on all 6 trained arm bundles (pure rollout eval, zero gradient updates)
            aud_out = V51_RELEASE / "audits" / env_key / f"seed_{s}.json"
            all_audit_outs.append(aud_out)
            rows.append(make_v51_spec(
                f"{env_key}-audit-{s}", "audit", aud_out,
                env_key=env_key, seed=s,
                deps=[b / "bundle.json" for b in arm_bundles],
                inputs=arm_bundles, gpu=False,
            ))

    rows.append(make_v51_spec(
        "aggregate", "analyze", V51_RELEASE / "analysis.json",
        deps=all_audit_outs,
        inputs=[V51_RELEASE / "audits" / e for e in V51_ENVS],
        gpu=False,
    ))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

    known = scheduler.scheduler_tasks() if not args.dry_run else []
    rows = [
        row for row, out in build_candidates()
        if not out.exists() and not any(
            t.get("signature") == row["signature"] and t.get("status") in scheduler.ACTIVE_STATUSES | {"done"}
            for t in known
        )
    ]
    if args.dry_run:
        print(json.dumps({
            "total_tasks": len(rows),
            "gpu_tasks": sum(r["vram"] > 0 for r in rows),
            "suites": dict(Counter(r["signature"].split("/")[1] for r in rows)),
        }, indent=2))
        return

    if not PREREG.is_file():
        raise FileNotFoundError(PREREG)
    if not rows:
        print("All v50/v51 outputs already present or active.")
        return

    res = subprocess.run(
        [
            sys.executable, str(scheduler.SCHEDULER),
            "submit-jsonl", "--stdin", "--trusted", "--json",
            "--intent-label", "bapr-jmlr-v51-samestep-submit", "--intent-ttl", "900",
        ],
        input="".join(json.dumps(r) + "\n" for r in rows),
        text=True, capture_output=True, check=True, env=scheduler.scheduler_env(),
    )
    ids = [str(t["id"]) for t in json.loads(res.stdout)["submitted"]]
    print(f"Submitted {len(ids)} tasks: {ids[0]} .. {ids[-1]}", flush=True)
    if args.dispatch:
        subprocess.run(
            [
                sys.executable, str(scheduler.SCHEDULER),
                "dispatch", "--bulk-window",
                "--intent-label", "bapr-jmlr-v51-samestep-dispatch", "--intent-ttl", "900",
                *[x for tid in ids for x in ("--task-id", tid)],
            ],
            check=True, env=scheduler.scheduler_env(),
        )


if __name__ == "__main__":
    main()
