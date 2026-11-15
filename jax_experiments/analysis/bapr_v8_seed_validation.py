"""Frozen independent-training-seed protocol for the final BAPR controller."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get(
    "BAPR_WORKSPACE_ROOT", Path(__file__).resolve().parents[2])).resolve()
ENV = "HalfCheetah-v2"
ENV_SHORT = "HalfCheetah"
FAMILY = "structured_channel"
TRAINING_SEEDS = tuple(range(5))
MODES = tuple(range(4))
CALIBRATION_EVENT_SEEDS = (2100, 2200)
EVALUATION_EVENT_SEEDS = (11100, 11200, 11300, 11400, 11500)

MAX_ITERS = 1400
FINAL_ITERATION = MAX_ITERS - 1
FINAL_TOTAL_STEPS = 5_600_000
FINAL_UPDATE_COUNT = 349_500
SAMPLES_PER_ITER = 4000
UPDATES_PER_ITER = 250

ROLES = ("sac", "escp", "resac", "specialist")
CONTROLLERS = ("sac", "escp", "resac", "bapr", "oracle")
DECISION_VARIANT = "cs4d025c80h8"
ROBUST_CONTROLLER = 4
FALLBACK_CONTROLLER = -1

RUN_ROOT = (
    ROOT / "jax_experiments" / "results_bapr_v8_seed_validation_runs_v1")
BUNDLE_ROOT = (
    ROOT / "jax_experiments" / "eval_bundles_bapr_v8_seed_validation_v1")
CALIBRATION_ROOT = (
    ROOT / "jax_experiments" / "results_bapr_v8_seed_validation_calibration_v1")
AUDIT_ROOT = (
    ROOT / "jax_experiments" / "results_bapr_v8_seed_validation_audit_v1")
ANALYSIS_ROOT = (
    ROOT / "jax_experiments" / "results_bapr_v8_seed_validation_analysis_v1")

ROUTER_ROOT = (
    ROOT / "jax_experiments" / "analysis" / "protocol_snapshots"
    / "v5_context_bootstrap")
ROUTER_MANIFEST = ROUTER_ROOT / "router_manifest.json"
ROUTER_PARAMS = ROUTER_ROOT / "router_params.npz"

BUNDLE_SCHEMA = "bapr.v8-seed-validation-bundle.v1"
CALIBRATION_SCHEMA = "bapr.v8-seed-validation-calibration.v1"
AUDIT_SCHEMA = "bapr.v8-seed-validation-audit.v1"
ANALYSIS_SCHEMA = "bapr.v8-seed-validation-analysis.v1"


def require_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in TRAINING_SEEDS:
        raise ValueError(f"training seed must be in {TRAINING_SEEDS}: {seed}")
    return seed


def require_role(role: str, mode: int | None = None) -> tuple[str, int | None]:
    role = str(role)
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}")
    if role == "specialist":
        if mode is None or int(mode) not in MODES:
            raise ValueError(f"specialist mode must be in {MODES}")
        return role, int(mode)
    if mode is not None:
        raise ValueError(f"role {role} does not accept a mode")
    return role, None


def role_name(role: str, mode: int | None = None) -> str:
    role, mode = require_role(role, mode)
    return f"specialist_mode_{mode}" if role == "specialist" else role


def run_dir(seed: int, role: str, mode: int | None = None) -> Path:
    return RUN_ROOT / f"seed_{require_seed(seed)}" / role_name(role, mode)


def bundle_dir(seed: int, role: str, mode: int | None = None) -> Path:
    return BUNDLE_ROOT / f"seed_{require_seed(seed)}" / role_name(role, mode)


def bundle_manifest(seed: int, role: str, mode: int | None = None) -> Path:
    return bundle_dir(seed, role, mode) / "bundle_manifest.json"


def bundle_required_paths(
    seed: int, role: str, mode: int | None = None,
) -> tuple[Path, ...]:
    directory = bundle_dir(seed, role, mode)
    return (
        directory / "bundle_manifest.json",
        directory / "checkpoints" / "params.pkl",
        directory / "checkpoints" / "train_state.pkl",
        directory / "logs" / "protocol_signature.json",
    )


def calibration_path(seed: int) -> Path:
    return CALIBRATION_ROOT / f"seed_{require_seed(seed)}" / "utility_table.json"


def audit_path(seed: int, event_seed: int) -> Path:
    if int(event_seed) not in EVALUATION_EVENT_SEEDS:
        raise ValueError(f"unregistered evaluation event seed {event_seed}")
    return (
        AUDIT_ROOT / f"seed_{require_seed(seed)}"
        / f"event_seed_{int(event_seed)}" / "group.json")


def analysis_path() -> Path:
    return ANALYSIS_ROOT / "summary.json"


def report_path() -> Path:
    return ANALYSIS_ROOT / "report.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "size": path.stat().st_size}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def checkpoint_record(directory: Path) -> dict[str, Any]:
    checkpoint = directory / "checkpoints"
    with (checkpoint / "train_state.pkl").open("rb") as handle:
        state = pickle.load(handle)
    with (checkpoint / "params.pkl").open("rb") as handle:
        params = pickle.load(handle)
    iteration = int(state["iteration"])
    return {
        "iteration": iteration,
        "next_iteration": iteration + 1,
        "total_steps": int(state["total_steps"]),
        "update_count": int(params["update_count"]),
        "algo": str(state["algo"]),
    }


def expected_algo(role: str) -> str:
    return "sac" if role in ("sac", "specialist") else role


def expected_lr(role: str) -> float:
    # Preserve the proven positive RE-SAC sign while matching the original
    # implementation's optimizer scale instead of the unstable JAX default.
    return 1e-5 if role == "resac" else 3e-4


def bundle_paths_for_seed(seed: int) -> list[Path]:
    return [
        bundle_manifest(seed, "sac"),
        bundle_manifest(seed, "escp"),
        bundle_manifest(seed, "resac"),
        *(bundle_manifest(seed, "specialist", mode) for mode in MODES),
    ]


def bundle_dependencies_for_seed(seed: int) -> list[Path]:
    paths = [
        *bundle_required_paths(seed, "sac"),
        *bundle_required_paths(seed, "escp"),
        *bundle_required_paths(seed, "resac"),
    ]
    for mode in MODES:
        paths.extend(bundle_required_paths(seed, "specialist", mode))
    return paths
