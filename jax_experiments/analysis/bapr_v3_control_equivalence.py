"""Protocol constants and calibration for control-equivalent specialists."""
from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    bapr_v3_independent_specialists as specialist_protocol,
)
from jax_experiments.analysis import (
    run_bapr_v3_budget_matched_fork as fork_protocol,
)
from jax_experiments.analysis import (
    run_bapr_v3_independent_specialist_audit as specialist_audit,
)


ROOT = specialist_protocol.ROOT
FAMILY = "structured_channel"
ENV = "HalfCheetah-v2"
CALIBRATION_EVENT_SEEDS = (1100, 1200, 1300, 1400, 1500)
HOLDOUT_EVENT_SEEDS = (2100, 2200, 2300, 2400, 2500)
CALIBRATION_RESULTS = (
    ROOT / "jax_experiments"
    / "results_bapr_v3_structured_channel_independent_specialist_audit_v1"
)
MAPPING_ROOT = (
    ROOT / "jax_experiments"
    / "results_bapr_v3_structured_channel_control_equivalence_mapping_v1"
)
MAPPING_PATH = MAPPING_ROOT / "controller_map.json"
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_bapr_v3_structured_channel_control_equivalence_audit_v1"
)
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_bapr_v3_structured_channel_control_equivalence_analysis_v1"
)
MAPPING_SCHEMA = "bapr.v3-control-equivalence-map.v1"
GROUP_SCHEMA = "bapr.v3-control-equivalence-audit-group.v1"


def configure() -> None:
    specialist_protocol.configure_structured_channel_headroom(ENV)


def calibration_group_path(event_seed: int) -> Path:
    return (
        CALIBRATION_RESULTS / FAMILY / specialist_protocol.env_short()
        / f"event_seed_{event_seed}" / "group.json"
    )


def holdout_group_path(event_seed: int) -> Path:
    return (
        AUDIT_ROOT / FAMILY / specialist_protocol.env_short()
        / f"event_seed_{event_seed}" / "group.json"
    )


def file_record(path: Path) -> dict[str, Any]:
    return {
        "sha256": fork_protocol.sha256_file(path),
        "size": path.stat().st_size,
    }


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def select_controller_map() -> dict[str, Any]:
    configure()
    groups = {}
    records = {}
    bundle_hashes = None
    for event_seed in CALIBRATION_EVENT_SEEDS:
        path = calibration_group_path(event_seed)
        payload = json.loads(path.read_text(encoding="utf-8"))
        specialist_audit.validate_group(payload)
        if (payload.get("family") != FAMILY
                or payload.get("env") != ENV
                or payload.get("event_seed") != event_seed):
            raise ValueError(f"calibration provenance mismatch: {path}")
        if bundle_hashes is None:
            bundle_hashes = payload.get("bundle_manifest_sha256")
        elif payload.get("bundle_manifest_sha256") != bundle_hashes:
            raise ValueError("calibration groups use different policy bundles")
        groups[event_seed] = payload
        records[str(event_seed)] = file_record(path)

    mapping = []
    rows = {}
    for physics_mode in specialist_protocol.MODES:
        means = {}
        per_seed_winners = []
        per_seed = {}
        for event_seed, group in groups.items():
            values = {
                specialist_mode: float(group["stationary"][
                    f"fixed_mode_{specialist_mode}"][str(physics_mode)][
                        "mean"])
                for specialist_mode in specialist_protocol.MODES
            }
            per_seed[str(event_seed)] = {
                str(mode): value for mode, value in values.items()
            }
            per_seed_winners.append(max(values, key=values.get))
        for specialist_mode in specialist_protocol.MODES:
            means[specialist_mode] = statistics.mean(
                per_seed[str(seed)][str(specialist_mode)]
                for seed in CALIBRATION_EVENT_SEEDS
            )
        ordered = sorted(
            specialist_protocol.MODES,
            key=lambda mode: (-means[mode], mode),
        )
        winner = int(ordered[0])
        mapping.append(winner)
        rows[str(physics_mode)] = {
            "mean_returns": {
                str(mode): means[mode] for mode in specialist_protocol.MODES
            },
            "selected_specialist": winner,
            "margin_to_second": means[ordered[0]] - means[ordered[1]],
            "per_seed_winners": per_seed_winners,
            "per_seed_returns": per_seed,
        }
    payload = {
        "schema": MAPPING_SCHEMA,
        "status": "complete",
        "family": FAMILY,
        "env": ENV,
        "training_seed": specialist_protocol.SEED,
        "calibration_event_seeds": list(CALIBRATION_EVENT_SEEDS),
        "holdout_event_seeds": list(HOLDOUT_EVENT_SEEDS),
        "controller_map": mapping,
        "distinct_specialists": sorted(set(mapping)),
        "selection_rule": (
            "highest mean stationary return over calibration event streams; "
            "ties choose the smallest specialist id"
        ),
        "calibration_groups": records,
        "bundle_manifest_sha256": bundle_hashes,
        "rows": rows,
    }
    write_json_atomic(MAPPING_PATH, payload)
    return payload


def load_controller_map() -> tuple[tuple[int, ...], dict[str, Any]]:
    payload = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    mapping = tuple(int(value) for value in payload.get("controller_map", []))
    if (payload.get("schema") != MAPPING_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("family") != FAMILY
            or payload.get("env") != ENV
            or len(mapping) != len(specialist_protocol.MODES)
            or any(value not in specialist_protocol.MODES for value in mapping)):
        raise ValueError(f"invalid controller map: {MAPPING_PATH}")
    for seed, expected in payload.get("calibration_groups", {}).items():
        if file_record(calibration_group_path(int(seed))) != expected:
            raise ValueError("calibration group changed after map selection")
    return mapping, payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select", action="store_true")
    args = parser.parse_args()
    if not args.select:
        parser.error("--select is required")
    payload = select_controller_map()
    print(
        "CONTROL-EQUIVALENCE MAP COMPLETE: "
        f"map={payload['controller_map']} output={MAPPING_PATH}",
        flush=True,
    )


if __name__ == "__main__":
    main()
