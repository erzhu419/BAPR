"""Frozen protocol for the shared-regime BAPR headroom screen."""
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
FAMILY = "mean_variance"
TRAINING_SEED = 0
ROLES = ("sac", "escp", "resac", "regime_robust", "regime_oracle")

MAX_ITERS = 1400
FINAL_ITERATION = MAX_ITERS - 1
FINAL_TOTAL_STEPS = 5_600_000
FINAL_UPDATE_COUNT = 350_000
SAMPLES_PER_ITER = 4000
UPDATES_PER_ITER = 250

RUN_ROOT = ROOT / "jax_experiments" / "results_bapr_regime_screen_v1"
BUNDLE_ROOT = ROOT / "jax_experiments" / "eval_bundles_bapr_regime_screen_v1"
BUNDLE_SCHEMA = "bapr.shared-regime-screen-bundle.v1"
AUDIT_ROOT = ROOT / "jax_experiments" / "results_bapr_regime_screen_audit_v1"
AUDIT_SCHEMA = "bapr.shared-regime-screen-audit.v1"
AUDIT_EVENT_SEEDS = (32100, 32200, 32300, 32400, 32500)
AUDIT_ROLES = ROLES


def require_role(role: str) -> str:
    role = str(role)
    if role not in ROLES:
        raise ValueError(f"unknown shared-regime role {role!r}")
    return role


def expected_algo(role: str) -> str:
    role = require_role(role)
    return "bapr_regime" if role.startswith("regime_") else role


def expected_lr(role: str) -> float:
    return 1e-5 if require_role(role) == "resac" else 3e-4


def run_dir(role: str) -> Path:
    return RUN_ROOT / require_role(role)


def bundle_dir(role: str) -> Path:
    return BUNDLE_ROOT / require_role(role)


def bundle_manifest(role: str) -> Path:
    return bundle_dir(role) / "bundle_manifest.json"


def require_audit_event_seed(event_seed: int) -> int:
    event_seed = int(event_seed)
    if event_seed not in AUDIT_EVENT_SEEDS:
        raise ValueError(f"unknown shared-regime audit event seed {event_seed}")
    return event_seed


def audit_dir(role: str, event_seed: int) -> Path:
    return (
        AUDIT_ROOT / require_role(role)
        / f"event_seed_{require_audit_event_seed(event_seed)}")


def audit_manifest(role: str, event_seed: int) -> Path:
    return audit_dir(role, event_seed) / "audit_manifest.json"


def analysis_json() -> Path:
    return AUDIT_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return AUDIT_ROOT / "analysis.md"


def bundle_required_paths(role: str) -> tuple[Path, ...]:
    directory = bundle_dir(role)
    return (
        directory / "bundle_manifest.json",
        directory / "checkpoints" / "params.pkl",
        directory / "checkpoints" / "train_state.pkl",
        directory / "logs" / "protocol_signature.json",
    )


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
