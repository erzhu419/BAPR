"""Frozen protocol for the bus robust-source and policy-bank headroom screen."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from mode_profiles import MODE_PROFILES


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "bus-policy-bank-headroom-v1-development"
TRAINING_SEEDS = (86_003, 86_021, 86_039)
MODES = tuple(MODE_PROFILES)

SOURCE_EPISODES = 500
SPECIALIST_EPISODES = 200
CHECKPOINT_INTERVAL = 10
CALIBRATION_EVENT_SEEDS = (188_001, 188_017)
STATIONARY_HOLDOUT_EVENT_SEEDS = (188_101, 188_117, 188_133)
SWITCHING_EVENT_SEEDS = (188_201, 188_217, 188_233)

MIN_STATIONARY_MODE_GAIN = 0.05
MIN_STATIONARY_MODE_WINS = 4
MIN_SWITCHING_GAIN = 0.10
REQUIRED_SWITCHING_EVENT_WINS = len(SWITCHING_EVENT_SEEDS)

RUN_ROOT = ROOT / "bus_policy_bank_results_v1"
SOURCE_RUN_ROOT = RUN_ROOT / "robust_sources"
SPECIALIST_RUN_ROOT = RUN_ROOT / "specialists"
BUNDLE_ROOT = ROOT / "bus_policy_bank_bundles_v1"
SOURCE_BUNDLE_ROOT = BUNDLE_ROOT / "robust_sources"
SPECIALIST_BUNDLE_ROOT = BUNDLE_ROOT / "specialists"
AUDIT_ROOT = ROOT / "bus_policy_bank_audits_v1"
ANALYSIS_ROOT = ROOT / "bus_policy_bank_analysis_v1"
REGISTRATION_ROOT = ROOT / "bus_experiments/deployments/policy_bank_headroom_v1"
REGISTRATION_PATH = REGISTRATION_ROOT / "registration.json"
REPORT = ROOT / "reports/bus_policy_bank_headroom_v1_2026-09-10.md"

REGISTRATION_SCHEMA = "bapr.bus-policy-bank-registration.v1"
AUDIT_SCHEMA = "bapr.bus-policy-bank-headroom-audit.v1"
ANALYSIS_SCHEMA = "bapr.bus-policy-bank-headroom-analysis.v1"


def require_seed(seed: int) -> int:
    value = int(seed)
    if value not in TRAINING_SEEDS:
        raise ValueError(f"unknown bus development seed: {value}")
    return value


def require_mode(mode: str) -> str:
    value = str(mode)
    if value not in MODES:
        raise ValueError(f"unknown bus mode: {value}")
    return value


def source_run_dir(seed: int) -> Path:
    return SOURCE_RUN_ROOT / f"seed_{require_seed(seed)}"


def source_bundle_dir(seed: int) -> Path:
    return SOURCE_BUNDLE_ROOT / f"seed_{require_seed(seed)}"


def source_manifest(seed: int) -> Path:
    return source_bundle_dir(seed) / "bundle_manifest.json"


def source_controller(seed: int) -> Path:
    return source_bundle_dir(seed) / "controller.pt"


def source_required_paths(seed: int) -> tuple[Path, ...]:
    directory = source_bundle_dir(seed)
    return (
        directory / "bundle_manifest.json",
        directory / "controller.pt",
        directory / "train_summary.json",
    )


def specialist_run_dir(seed: int, mode: str) -> Path:
    return (
        SPECIALIST_RUN_ROOT / f"seed_{require_seed(seed)}"
        / require_mode(mode)
    )


def specialist_bundle_dir(seed: int, mode: str) -> Path:
    return (
        SPECIALIST_BUNDLE_ROOT / f"seed_{require_seed(seed)}"
        / require_mode(mode)
    )


def specialist_manifest(seed: int, mode: str) -> Path:
    return specialist_bundle_dir(seed, mode) / "bundle_manifest.json"


def specialist_controller(seed: int, mode: str) -> Path:
    return specialist_bundle_dir(seed, mode) / "controller.pt"


def specialist_required_paths(seed: int, mode: str) -> tuple[Path, ...]:
    directory = specialist_bundle_dir(seed, mode)
    return (
        directory / "bundle_manifest.json",
        directory / "controller.pt",
        directory / "train_summary.json",
    )


def audit_dir(seed: int) -> Path:
    return AUDIT_ROOT / f"seed_{require_seed(seed)}"


def audit_result(seed: int) -> Path:
    return audit_dir(seed) / "audit.json"


def audit_manifest(seed: int) -> Path:
    return audit_dir(seed) / "audit_manifest.json"


def analysis_json() -> Path:
    return ANALYSIS_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return ANALYSIS_ROOT / "analysis.md"


def file_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": path.resolve().relative_to(ROOT).as_posix(),
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    temporary.replace(path)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def registration_source_paths() -> tuple[Path, ...]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "bus_experiments/frozen_policy_bank_core.py",
        ROOT / "bus_experiments/run_policy_bank_train_v1.py",
        ROOT / "bus_experiments/run_policy_bank_audit_v1.py",
        ROOT / "bus_experiments/analyze_policy_bank_headroom_v1.py",
        ROOT / "scripts/submit_bus_policy_bank_headroom_v1.py",
        ROOT / "env/sim.py",
        ROOT / "env/bus.py",
        ROOT / "env/route.py",
        ROOT / "env/station.py",
        ROOT / "env/passenger.py",
        ROOT / "mode_profiles.py",
        ROOT / "env/config.json",
        ROOT / "env/data/passenger_OD.xlsx",
        ROOT / "env/data/stop_news.xlsx",
        ROOT / "env/data/route_news.xlsx",
        ROOT / "env/data/time_table.xlsx",
    )
    return tuple(path.resolve() for path in paths)


def registration_payload() -> dict[str, Any]:
    paths = registration_source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"bus registration sources missing: {missing}")
    return {
        "schema": REGISTRATION_SCHEMA,
        "status": "registered",
        "created_before_training": True,
        "identity": {
            "protocol_version": PROTOCOL_VERSION,
            "training_seeds": list(TRAINING_SEEDS),
            "modes": list(MODES),
            "calibration_event_seeds": list(CALIBRATION_EVENT_SEEDS),
            "stationary_holdout_event_seeds": list(
                STATIONARY_HOLDOUT_EVENT_SEEDS),
            "switching_event_seeds": list(SWITCHING_EVENT_SEEDS),
        },
        "training": {
            "robust_source_episodes": SOURCE_EPISODES,
            "specialist_additional_episodes": SPECIALIST_EPISODES,
            "robust_source_environment": (
                "persistent MODE_PROFILES with random initial mode and "
                "1800-7200 second switches; within-mode route and passenger "
                "stochasticity remains active"
            ),
            "specialist_initialization": (
                "copy policy, critic, target critic, and alpha from the "
                "matched robust source; reset optimizers and replay"
            ),
            "regularization_sign": "+ weight_reg * reg_norm",
            "weight_reg": 0.01,
        },
        "gate": {
            "all_three_policy_seeds_pass": True,
            "minimum_stationary_mode_wins_per_seed": (
                MIN_STATIONARY_MODE_WINS),
            "minimum_stationary_mode_gain": MIN_STATIONARY_MODE_GAIN,
            "minimum_switching_gain": MIN_SWITCHING_GAIN,
            "all_switching_events_win": True,
            "dynamic_oracle_must_beat_every_fixed_controller": True,
        },
        "scope": {
            "estimator_training": False,
            "next_action": (
                "train the frozen causal mode estimator only if this true-mode "
                "policy-bank upper bound passes"
            ),
        },
        "sync_policy": (
            "sync compact controller bundles and JSON only; keep replay and "
            "training checkpoints on remote nodes"
        ),
        "source_records": {
            path.relative_to(ROOT).as_posix(): file_record(path)
            for path in paths
        },
    }


def create_registration() -> dict[str, Any]:
    payload = registration_payload()
    if REGISTRATION_PATH.is_file():
        if read_json(REGISTRATION_PATH) != payload:
            raise ValueError("existing bus policy-bank registration changed")
    else:
        write_json_atomic(REGISTRATION_PATH, payload)
    return validate_registration()


def validate_registration() -> dict[str, Any]:
    if not REGISTRATION_PATH.is_file():
        raise FileNotFoundError(f"missing bus registration: {REGISTRATION_PATH}")
    payload = read_json(REGISTRATION_PATH)
    if payload != registration_payload():
        raise ValueError("bus registration or frozen source closure changed")
    return payload


def assert_protocol_integrity() -> None:
    splits = (
        set(CALIBRATION_EVENT_SEEDS),
        set(STATIONARY_HOLDOUT_EVENT_SEEDS),
        set(SWITCHING_EVENT_SEEDS),
    )
    if any(
        left & right
        for index, left in enumerate(splits)
        for right in splits[index + 1:]
    ):
        raise ValueError("bus evaluation seed splits overlap")
    if len(MODES) != 5 or len(set(MODES)) != len(MODES):
        raise ValueError("bus protocol requires five unique persistent modes")


assert_protocol_integrity()
