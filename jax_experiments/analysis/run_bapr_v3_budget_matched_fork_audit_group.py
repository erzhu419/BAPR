#!/usr/bin/env python3
"""Run one causally paired BAPR-v3 fixed-context audit group.

One invocation evaluates all six controller conditions sequentially with the
same interpreter, code snapshot, host/GPU, and event stream.  Results are
published as one atomic ``event_seed_*`` bundle only after every evaluator and
the strict shared-fork producer-provenance check succeeds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from jax_experiments.analysis import (
    analyze_bapr_v3_budget_matched_fork_audit as audit,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = (
    ROOT / "jax_experiments" / "results_bapr_v3_budget_matched_fork_audit_v2")
EXPECTED_ROWS = {
    "summary.csv": 3,
    "task_returns.csv": 4,
    "switching_returns.csv": 5,
    "switching_trace.csv": 5000,
}
EXPECTED_NEXT_ITER = 1400
EXPECTED_TOTAL_STEPS = 5_600_000


@dataclass(frozen=True)
class Controller:
    directory: str
    branch: str
    context_source: str
    fixed_mode: int | None = None


CONTROLLERS = (
    Controller("robust", "robust_long", "robust"),
    Controller("oracle", "oracle_direct", "oracle"),
    *(Controller(f"fixed_mode_{mode}", "oracle_direct", "oracle", mode)
      for mode in range(4)),
)


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def source_tree_fingerprint(execution_root: Path = ROOT) -> dict[str, object]:
    """Hash the evaluator's Python source while excluding generated trees."""
    package_root = execution_root / "jax_experiments"
    files: list[tuple[Path, str]] = []
    for path in package_root.rglob("*.py"):
        relative = path.relative_to(package_root)
        if any(
            part == "__pycache__"
            or part.startswith("eval_bundles")
            or part.startswith("results_")
            for part in relative.parts
        ):
            continue
        files.append((path.resolve(), f"jax_experiments/{relative.as_posix()}"))
    files.extend(
        (path.resolve(), path.relative_to(ROOT).as_posix())
        for path in sorted(
            (ROOT / "scripts").glob("*bapr_v3_budget_matched_fork_audit*.py"))
    )
    files = list(dict.fromkeys(files))
    digest = hashlib.sha256()
    entries = []
    for path, relative in sorted(files, key=lambda item: item[1]):
        value = file_sha256(path)
        entries.append({"path": relative, "sha256": value})
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return {
        "algorithm": "sha256(path\\0sha256\\n)",
        "file_count": len(entries),
        "sha256": digest.hexdigest(),
        "files": entries,
    }


def runtime_fingerprint() -> dict[str, object]:
    """Probe the exact child interpreter and its selected JAX device."""
    probe = r'''
import json
import importlib.metadata
import os
import platform
import sys
import jax

def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

payload = {
    "host": platform.node(),
    "python_executable": sys.executable,
    "python_realpath": os.path.realpath(sys.executable),
    "python_version": platform.python_version(),
    "platform": platform.platform(),
    "packages": {
        name: version(name)
        for name in ("jax", "jaxlib", "flax", "optax", "numpy", "brax")
    },
    "jax": {
        "backend": jax.default_backend(),
        "devices": [
            {"platform": device.platform, "kind": device.device_kind,
             "id": int(device.id)}
            for device in jax.devices()
        ],
    },
}
print(json.dumps(payload, sort_keys=True))
'''
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=ROOT, text=True,
        capture_output=True, check=True, timeout=180)
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"runtime probe did not return JSON: {result.stdout!r}") from exc
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    try:
        nvidia = subprocess.run([
            "nvidia-smi", "--query-gpu=index,uuid,name,driver_version",
            "--format=csv,noheader,nounits",
        ], text=True, capture_output=True, check=True, timeout=30)
        rows = [line.strip() for line in nvidia.stdout.splitlines()
                if line.strip()]
        error = ""
    except Exception as exc:
        rows = []
        error = f"{type(exc).__name__}: {exc}"
    tokens = {token.strip() for token in visible.split(",") if token.strip()}
    selected = []
    for row in rows:
        columns = [item.strip() for item in row.split(",", 3)]
        if not tokens or any(
            token == columns[0] or (len(columns) > 1 and token == columns[1])
            for token in tokens
        ):
            selected.append(row)
    payload["gpu"] = {
        "cuda_visible_devices": visible,
        "selected_nvidia_smi_rows": selected,
        "nvidia_smi_error": error,
    }
    payload["sha256"] = canonical_sha256(payload)
    return payload


def evaluator_values(
    pair_dir: Path,
    output_dir: Path,
    controller: Controller,
    event_seed: int,
) -> list[str]:
    run_dir = pair_dir / controller.branch
    values = [
        "--run-dir", str(run_dir),
        "--out-dir", str(output_dir),
        "--episodes-per-task", "5",
        "--switching-episodes", "5",
        "--switching-period-steps", "500",
        "--heldout-task-stream", "validation",
        "--detection-window-steps", "50",
        "--bapr-v2-context-source", controller.context_source,
        "--bapr-v2-advantage", "off",
        "--rng-seed", str(20260715 + event_seed),
        "--eval-seed-offset", str(event_seed),
        "--max-tasks", "4",
        "--stationary-test-only",
        "--min-checkpoint-next-iter", str(EXPECTED_NEXT_ITER),
        "--resume-from", str(run_dir / "checkpoints" / "train_state.pkl"),
    ]
    if controller.fixed_mode is not None:
        values += ["--fixed-oracle-mode-id", str(controller.fixed_mode)]
    return values


def validate_basic_output(path: Path, controller: Controller) -> None:
    expected_oracle_mode = (
        "dynamic" if controller.fixed_mode is None
        else str(controller.fixed_mode))
    for filename, expected_count in EXPECTED_ROWS.items():
        csv_path = path / filename
        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != expected_count:
            raise RuntimeError(
                f"{csv_path} has {len(rows)} rows, expected {expected_count}")
        for row in rows:
            if row.get("run_name") != controller.branch:
                raise RuntimeError(
                    f"{csv_path} has wrong run_name={row.get('run_name')!r}")
            if row.get("eval_context_source") != controller.context_source:
                raise RuntimeError(
                    f"{csv_path} has wrong eval_context_source")
            if row.get("eval_oracle_mode_id") != expected_oracle_mode:
                raise RuntimeError(
                    f"{csv_path} has wrong eval_oracle_mode_id")
            try:
                next_iter = float(row["checkpoint_next_iter"])
                total_steps = float(row["checkpoint_total_steps"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"{csv_path} has invalid checkpoint identity") from exc
            if (not math.isfinite(next_iter) or not next_iter.is_integer()
                    or int(next_iter) != EXPECTED_NEXT_ITER):
                raise RuntimeError(f"{csv_path} has wrong checkpoint iteration")
            if (not math.isfinite(total_steps) or not total_steps.is_integer()
                    or int(total_steps) != EXPECTED_TOTAL_STEPS):
                raise RuntimeError(f"{csv_path} has wrong checkpoint steps")


def output_hashes(directory: Path) -> dict[str, str]:
    return {
        filename: file_sha256(directory / filename)
        for filename in EXPECTED_ROWS
    }


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-dir", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--family", choices=audit.FAMILIES, required=True)
    parser.add_argument("--env", choices=audit.FULL_ENVS, required=True)
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--event-seed", type=int, required=True)
    parser.add_argument(
        "--resume", action="store_true",
        help=("Declarative scheduleurm checkpoint-staging marker; the runner "
              "still validates the complete pair and never resumes training."))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.resume:
        raise SystemExit(
            "--resume is required so scheduleurm stages the producer pair root")
    if args.training_seed != 0:
        raise SystemExit("the preregistered v2 audit requires training seed 0")
    if args.event_seed not in audit.DEFAULT_EVENT_SEEDS:
        raise SystemExit(
            f"event seed must be one of {audit.DEFAULT_EVENT_SEEDS}")

    pair_dir = args.pair_dir.resolve()
    expected_pair = audit.pair_directory(
        pair_dir.parent, args.family, args.env, args.training_seed)
    if pair_dir != expected_pair.resolve():
        raise SystemExit(
            f"pair directory identity mismatch: {pair_dir} != {expected_pair}")
    provenance = audit.validate_pair_provenance(
        pair_dir, args.family, args.env, args.training_seed)
    orchestrator_hashes = audit.validate_live_audit_modules(
        provenance["source"])

    env_short = args.env.removesuffix("-v2")
    final_group = (
        args.results_root.resolve() / args.family / env_short
        / f"event_seed_{args.event_seed}")
    if final_group.exists() or final_group.is_symlink():
        try:
            audit.validate_group_bundle(
                final_group, pair_dir, args.family, args.env,
                args.training_seed, args.event_seed)
        except (OSError, ValueError):
            if final_group.is_dir() and not final_group.is_symlink():
                shutil.rmtree(final_group)
            else:
                final_group.unlink()
        else:
            print(f"Complete valid group already exists: {final_group}")
            return

    final_group.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{final_group.name}.tmp.", dir=final_group.parent))
    try:
        runtime = runtime_fingerprint()
        execution_root = temporary / "execution_source"
        audit.extract_pair_source_snapshot(pair_dir, provenance, execution_root)
        source = source_tree_fingerprint(execution_root)
        audit.validate_evaluator_runtime(provenance["runtime"], runtime)
        audit.validate_evaluator_source(provenance["source"], source)
        pair_manifest_path = pair_dir / "provenance" / "pair_manifest.json"
        pair_manifest_sha256 = file_sha256(pair_manifest_path)
        controller_records: dict[str, object] = {}
        common_env = os.environ.copy()
        common_env["PYTHONPATH"] = str(execution_root)
        for index, controller in enumerate(CONTROLLERS, start=1):
            output_dir = temporary / controller.directory
            values = evaluator_values(
                pair_dir, output_dir, controller, args.event_seed)
            command = [
                sys.executable, "-u", "-m",
                "jax_experiments.analysis.final_task_sweep", *values,
            ]
            print(
                f"[{index}/{len(CONTROLLERS)}] {controller.directory}",
                flush=True)
            subprocess.run(
                command, cwd=execution_root, env=common_env, check=True)
            validate_basic_output(output_dir, controller)
            controller_records[controller.directory] = {
                "branch": controller.branch,
                "context_source": controller.context_source,
                "fixed_mode": controller.fixed_mode,
                "command": command,
                "runtime_sha256": runtime["sha256"],
                "source_tree_sha256": source["sha256"],
                "execution_source": "validated_producer_archive",
                "output_sha256": output_hashes(output_dir),
            }

        provenance_after = audit.validate_pair_provenance(
            pair_dir, args.family, args.env, args.training_seed)
        if (file_sha256(pair_manifest_path) != pair_manifest_sha256
                or provenance_after != provenance):
            raise RuntimeError("producer pair manifest changed during evaluation")
        source_after = source_tree_fingerprint(execution_root)
        if source_after["sha256"] != source["sha256"]:
            raise RuntimeError("evaluator source tree changed during the group")

        shutil.rmtree(execution_root)
        group_manifest = {
            "schema": "bapr.v3-budget-matched-fork-audit-group.v2",
            "status": "complete",
            "family": args.family,
            "env": args.env,
            "training_seed": args.training_seed,
            "event_seed": args.event_seed,
            "rng_seed": 20260715 + args.event_seed,
            "eval_seed_offset": args.event_seed,
            "pair_name": pair_dir.name,
            "pair_manifest_schema": provenance["schema"],
            "pair_manifest_sha256": pair_manifest_sha256,
            "orchestrator_source_sha256": orchestrator_hashes,
            "execution_source": {
                "mode": "validated_producer_archive",
                "archive_path": provenance["source"]["archive"]["path"],
                "archive_sha256": provenance["source"]["archive"]["sha256"],
                "source_tree_sha256": source["sha256"],
            },
            "runtime": runtime,
            "source": source,
            "controllers": controller_records,
        }
        write_json_atomic(
            temporary / "provenance" / "group_manifest.json",
            group_manifest)
        os.replace(temporary, final_group)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"Published paired audit group: {final_group}", flush=True)


if __name__ == "__main__":
    main()
