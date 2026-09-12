"""Calibrate a robust-inclusive controller map for one adapter bank."""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
from flax import nnx

from jax_experiments.analysis import final_task_sweep
from jax_experiments.analysis import regime_adapter_fork as protocol
from jax_experiments.analysis.regime_adapter_policy_bank import (
    load_policy_bank,
)
from jax_experiments.train import make_env


OUTPUT_FILES = ("stationary_returns.csv", "utility_map.json")
CONTROLLERS = (-1,) + protocol.MODES
CONTROLLER_LABELS = {
    -1: "frozen_base",
    **{mode: f"adapter_{mode}" for mode in protocol.MODES},
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    final_task_sweep.write_csv(path, rows)


def _validate_rows(path: Path, seed: int, delta: float) -> None:
    rows = _read_rows(path)
    expected = (
        len(protocol.CALIBRATION_EVENT_SEEDS)
        * len(CONTROLLERS) * len(protocol.MODES))
    if len(rows) != expected:
        raise ValueError(f"calibration has {len(rows)} rows, expected {expected}")
    identities = {
        (int(float(row["event_seed"])), row["controller"],
         int(float(row["mode_id_mean"])))
        for row in rows
    }
    expected_identities = {
        (event_seed, CONTROLLER_LABELS[controller], mode)
        for event_seed in protocol.CALIBRATION_EVENT_SEEDS
        for controller in CONTROLLERS for mode in protocol.MODES
    }
    if identities != expected_identities:
        raise ValueError("calibration rows do not cover the full paired matrix")
    for row in rows:
        if (int(float(row["training_seed"])) != seed
                or float(row["residual_delta"]) != delta
                or row["split"] != "calibration"):
            raise ValueError("calibration row identity changed")


def validate_calibration(seed: int, delta: float) -> dict:
    seed = protocol.require_seed(seed)
    delta = protocol.require_delta(delta)
    directory = protocol.calibration_dir(seed, delta)
    payload = protocol.read_json(directory / protocol.CALIBRATION_MANIFEST_NAME)
    expected_identity = {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "env": protocol.ENV,
        "training_seed": seed,
        "residual_delta": delta,
    }
    if (payload.get("schema") != protocol.CALIBRATION_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("identity") != expected_identity
            or set(payload.get("files") or {}) != set(OUTPUT_FILES)):
        raise ValueError(f"invalid regime-adapter calibration: {directory}")
    for relative, expected in payload["files"].items():
        path = directory / relative
        if not path.is_file() or protocol.file_record(path) != expected:
            raise ValueError(f"calibration file changed: {path}")
    _validate_rows(directory / "stationary_returns.csv", seed, delta)
    utility = protocol.read_json(directory / "utility_map.json")
    controller_map = utility.get("controller_map")
    if (not isinstance(controller_map, list)
            or len(controller_map) != len(protocol.MODES)
            or any(int(value) not in CONTROLLERS for value in controller_map)):
        raise ValueError("calibration utility map is invalid")
    return payload


def _select_utility_map(rows: list[dict]) -> dict:
    result = []
    table = []
    for physics_mode in protocol.MODES:
        candidates = []
        base_rows = [
            row for row in rows
            if int(row["physics_mode"]) == physics_mode
            and int(row["controller_id"]) == -1
        ]
        base_termination = float(np.mean([
            row["terminated_rate"] for row in base_rows]))
        for controller in CONTROLLERS:
            selected = [
                row for row in rows
                if int(row["physics_mode"]) == physics_mode
                and int(row["controller_id"]) == controller
            ]
            mean_return = float(np.mean([
                row["return_mean"] for row in selected]))
            termination = float(np.mean([
                row["terminated_rate"] for row in selected]))
            eligible = bool(
                controller == -1
                or termination <= base_termination + 0.02)
            candidates.append({
                "physics_mode": physics_mode,
                "controller_id": controller,
                "controller": CONTROLLER_LABELS[controller],
                "return_mean": mean_return,
                "terminated_rate": termination,
                "eligible": eligible,
            })
        eligible = [item for item in candidates if item["eligible"]]
        winner = max(
            eligible, key=lambda item: (
                item["return_mean"], -item["terminated_rate"],
                -abs(item["controller_id"] - physics_mode)))
        result.append(int(winner["controller_id"]))
        for item in candidates:
            item["selected"] = bool(item is winner)
            table.append(item)
    return {
        "controller_map": result,
        "controller_code": {
            "-1": "frozen_base",
            **{str(mode): f"adapter_{mode}" for mode in protocol.MODES},
        },
        "selection_rule": (
            "For each physics mode, maximize paired calibration return among "
            "the frozen base and four adapters, excluding an adapter when its "
            "termination rate exceeds the base by more than 0.02."),
        "table": table,
    }


def run(seed: int, delta: float) -> None:
    seed = protocol.require_seed(seed)
    delta = protocol.require_delta(delta)
    destination = protocol.calibration_dir(seed, delta)
    if protocol.calibration_manifest(seed, delta).is_file():
        validate_calibration(seed, delta)
        print(f"REGIME ADAPTER CALIBRATION ALREADY COMPLETE: {destination}")
        return
    bank = load_policy_bank(seed, delta)
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp.", dir=destination.parent))
    try:
        config = deepcopy(bank.config)
        config.stochastic_mode_fixed_id = -1
        rows = []
        for event_seed in protocol.CALIBRATION_EVENT_SEEDS:
            eval_env = make_env(config, seed_offset=event_seed - seed)
            eval_env.build_rollout_fn(
                nnx.graphdef(bank.agent.policy), direct_policy_context=True)
            try:
                for controller in CONTROLLERS:
                    bank.agent.set_controller_map(
                        (controller,) * len(protocol.MODES))
                    current = final_task_sweep.evaluate_task_split(
                        bank.agent, eval_env, config, bank.tasks,
                        "calibration", protocol.EPISODES_PER_TASK,
                        len(protocol.MODES), 20_290_722 + event_seed)
                    for row in current:
                        physics_mode = int(row["mode_id_mean"])
                        row.update({
                            "event_seed": event_seed,
                            "training_seed": seed,
                            "residual_delta": delta,
                            "controller_id": controller,
                            "controller": CONTROLLER_LABELS[controller],
                            "physics_mode": physics_mode,
                        })
                    rows.extend(current)
            finally:
                if hasattr(eval_env, "close"):
                    eval_env.close()
        _write_csv(temporary / "stationary_returns.csv", rows)
        utility = _select_utility_map(rows)
        utility.update({
            "schema": "bapr.regime-adapter-utility-map.v1",
            "training_seed": seed,
            "residual_delta": delta,
            "calibration_event_seeds": list(
                protocol.CALIBRATION_EVENT_SEEDS),
        })
        protocol.write_json_atomic(temporary / "utility_map.json", utility)
        _validate_rows(
            temporary / "stationary_returns.csv", seed, delta)
        payload = {
            "schema": protocol.CALIBRATION_SCHEMA,
            "status": "complete",
            "identity": {
                "protocol_version": protocol.PROTOCOL_VERSION,
                "env": protocol.ENV,
                "training_seed": seed,
                "residual_delta": delta,
            },
            "controller_map": utility["controller_map"],
            "files": {
                relative: protocol.file_record(temporary / relative)
                for relative in OUTPUT_FILES
            },
        }
        protocol.write_json_atomic(
            temporary / protocol.CALIBRATION_MANIFEST_NAME, payload)
        os.replace(temporary, destination)
    finally:
        bank.close()
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_calibration(seed, delta)
    print(f"REGIME ADAPTER CALIBRATION COMPLETE: {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=protocol.TRAINING_SEEDS,
                        required=True)
    parser.add_argument("--delta", type=float,
                        choices=protocol.RESIDUAL_DELTAS, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for scheduler staging")
    run(args.seed, args.delta)


if __name__ == "__main__":
    main()
