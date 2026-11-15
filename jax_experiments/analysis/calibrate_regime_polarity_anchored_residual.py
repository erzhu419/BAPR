"""Calibrate safe residual modes on held-out event streams."""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from jax_experiments.analysis import (
    regime_polarity_anchored_eval as common,
)
from jax_experiments.analysis import (
    regime_polarity_anchored_residual as protocol,
)


def _identity(seed: int) -> dict[str, Any]:
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "env": protocol.ENV,
        "family": protocol.FAMILY,
        "training_seed": protocol.require_training_seed(seed),
        "event_seeds": list(protocol.CALIBRATION_EVENT_SEEDS),
        "arms": list(common.CALIBRATION_ARMS),
        "gain_margin": protocol.CALIBRATION_GAIN_MARGIN,
        "max_termination_gap": protocol.MAX_TERMINATION_GAP,
    }


def _mode_decisions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decisions = []
    for mode in protocol.MODES:
        stats = {}
        for arm in common.CALIBRATION_ARMS:
            selected = [
                row for row in rows
                if row["arm"] == arm and int(row["mode"]) == int(mode)
            ]
            stats[arm] = {
                "return_mean": float(np.mean([
                    value
                    for row in selected
                    for value in row["returns"]
                ])),
                "terminated_rate": float(np.mean([
                    row["terminated_rate"] for row in selected
                ])),
            }
        reference_return = max(
            stats["robust_continue"]["return_mean"],
            stats["anchored_base"]["return_mean"],
        )
        oracle_return = stats["oracle_residual"]["return_mean"]
        relative_gain = (
            oracle_return - reference_return
        ) / max(abs(reference_return), 1.0)
        reference_termination = min(
            stats["robust_continue"]["terminated_rate"],
            stats["anchored_base"]["terminated_rate"],
        )
        termination_gap = (
            stats["oracle_residual"]["terminated_rate"]
            - reference_termination
        )
        enabled = (
            relative_gain >= protocol.CALIBRATION_GAIN_MARGIN
            and termination_gap <= protocol.MAX_TERMINATION_GAP
        )
        decisions.append({
            "mode": int(mode),
            "enabled": bool(enabled),
            "relative_gain_over_best_robust": float(relative_gain),
            "termination_gap_over_safer_robust": float(termination_gap),
            "arms": stats,
        })
    return decisions


def validate(seed: int) -> dict[str, Any]:
    seed = protocol.require_training_seed(seed)
    destination = protocol.calibration_dir(seed)
    payload = protocol.read_json(
        destination / "calibration_manifest.json")
    if (payload.get("schema") != protocol.CALIBRATION_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("identity") != _identity(seed)
            or len(payload.get("mode_decisions") or [])
            != len(protocol.MODES)
            or len(payload.get("mode_mask") or [])
            != len(protocol.MODES)):
        raise ValueError(f"invalid anchored calibration: {destination}")
    expected = {"calibration.json"}
    records = payload.get("files") or {}
    if set(records) != expected:
        raise ValueError(f"incomplete anchored calibration: {destination}")
    for relative, record in records.items():
        path = destination / relative
        if not path.is_file() or protocol.file_record(path) != record:
            raise ValueError(f"anchored calibration changed: {path}")
    result = protocol.read_json(destination / "calibration.json")
    if (result.get("mode_mask") != payload["mode_mask"]
            or result.get("mode_decisions") != payload["mode_decisions"]):
        raise ValueError("anchored calibration manifest disagrees with result")
    return payload


def run(seed: int) -> None:
    seed = protocol.require_training_seed(seed)
    destination = protocol.calibration_dir(seed)
    if (destination / "calibration_manifest.json").is_file():
        validate(seed)
        print(f"ANCHORED CALIBRATION ALREADY COMPLETE: {destination}")
        return
    robust = common.load_controller("robust_continue", seed)
    anchored = common.load_controller("anchored", seed)
    if robust[0].seed != anchored[0].seed:
        raise ValueError("paired anchored controllers have different seeds")
    controllers = {
        "robust_continue": robust,
        "anchored": anchored,
    }
    action_fns = {
        role: common.policy_action_fn(value[1])
        for role, value in controllers.items()
    }
    rows = []
    disabled_mask = np.zeros((len(protocol.MODES),), dtype=bool)
    for event_seed in protocol.CALIBRATION_EVENT_SEEDS:
        for arm in common.CALIBRATION_ARMS:
            role = common.arm_source(arm)
            config, _, policy_state = controllers[role]
            stationary, _ = common.strict_stationary(
                config,
                arm,
                policy_state,
                action_fns[role],
                event_seed,
                mode_mask=disabled_mask,
            )
            for row in stationary:
                row["event_seed"] = int(event_seed)
            rows.extend(stationary)
    decisions = _mode_decisions(rows)
    mode_mask = [bool(row["enabled"]) for row in decisions]
    result = {
        "schema": protocol.CALIBRATION_SCHEMA,
        "status": "complete",
        "identity": _identity(seed),
        "mode_decisions": decisions,
        "mode_mask": mode_mask,
        "stationary": rows,
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp.", dir=destination.parent))
    try:
        protocol.write_json_atomic(
            temporary / "calibration.json", result)
        files = {
            "calibration.json": protocol.file_record(
                temporary / "calibration.json"),
        }
        payload = {
            "schema": protocol.CALIBRATION_SCHEMA,
            "status": "complete",
            "identity": _identity(seed),
            "mode_decisions": decisions,
            "mode_mask": mode_mask,
            "controller_bundles": {
                role: protocol.file_record(
                    protocol.branch_manifest(role, seed))
                for role in protocol.BRANCH_ROLES
            },
            "files": files,
        }
        protocol.write_json_atomic(
            temporary / "calibration_manifest.json", payload)
        if destination.exists() or destination.is_symlink():
            if destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate(seed)
    print(
        f"ANCHORED CALIBRATION COMPLETE seed={seed} mask={mode_mask}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for file-gated execution")
    run(args.seed)


if __name__ == "__main__":
    main()
