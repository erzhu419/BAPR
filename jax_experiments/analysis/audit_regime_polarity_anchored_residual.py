"""Strict stationary and switching audit for one anchored policy seed."""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from jax_experiments.analysis import (
    calibrate_regime_polarity_anchored_residual as calibration,
)
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
        "event_seeds": list(protocol.AUDIT_EVENT_SEEDS),
        "arms": list(common.AUDIT_ARMS),
        "confidence_threshold": protocol.CONFIDENCE_THRESHOLD,
        "benchmark_role": "fresh_seed_anchored_residual_development",
        "online_inputs": [
            "observation",
            "commanded_action",
            "next_observation",
        ],
        "online_forbidden": [
            "mode_id",
            "action_gain",
            "executed_action",
            "switch_clock",
        ],
    }


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _event_result(
    seed: int,
    event_seed: int,
    controllers,
    action_fns,
    estimator,
    mode_mask: np.ndarray,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    stationary = []
    stationary_metrics = []
    switching = []
    traces = {}
    for arm in common.AUDIT_ARMS:
        role = common.arm_source(arm)
        config, _, policy_state = controllers[role]
        arm_stationary, arm_metrics = common.strict_stationary(
            config,
            arm,
            policy_state,
            action_fns[role],
            event_seed,
            mode_mask=mode_mask,
            estimator=estimator if arm in common.LEARNED_ARMS else None,
        )
        arm_switching, trace = common.strict_switching(
            config,
            arm,
            policy_state,
            action_fns[role],
            estimator,
            event_seed,
            mode_mask=mode_mask,
        )
        stationary.extend(arm_stationary)
        stationary_metrics.extend(arm_metrics)
        switching.append(arm_switching)
        if arm in common.LEARNED_ARMS:
            traces[arm] = trace
    return {
        "schema": protocol.EVENT_SCHEMA,
        "status": "complete",
        "training_seed": int(seed),
        "event_seed": int(event_seed),
        "env": protocol.ENV,
        "family": protocol.FAMILY,
        "dwell_steps": protocol.DWELL_STEPS,
        "max_episode_steps": protocol.MAX_EPISODE_STEPS,
        "episodes_per_stationary_mode": protocol.EPISODES_PER_TASK,
        "switching_episodes": protocol.SWITCHING_EPISODES,
        "arms": list(common.AUDIT_ARMS),
        "mode_mask": [bool(value) for value in mode_mask],
        "confidence_threshold": protocol.CONFIDENCE_THRESHOLD,
        "frozen_model_manifest": protocol.FROZEN_MODEL_MANIFEST_RECORD,
        "frozen_model_parameters": protocol.FROZEN_MODEL_PARAMETER_RECORD,
        "stationary": stationary,
        "stationary_posterior_metrics": stationary_metrics,
        "switching": switching,
    }, traces


def validate(seed: int) -> dict[str, Any]:
    seed = protocol.require_training_seed(seed)
    protocol.validate_frozen_estimator()
    calibration_payload = calibration.validate(seed)
    destination = protocol.audit_dir(seed)
    payload = protocol.read_json(destination / "audit_manifest.json")
    if (payload.get("schema") != protocol.AUDIT_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("identity") != _identity(seed)
            or payload.get("mode_mask")
            != calibration_payload["mode_mask"]
            or payload.get("frozen_model_manifest")
            != protocol.FROZEN_MODEL_MANIFEST_RECORD
            or payload.get("frozen_model_parameters")
            != protocol.FROZEN_MODEL_PARAMETER_RECORD):
        raise ValueError(f"invalid anchored audit: {destination}")
    expected = {
        f"event_seed_{event_seed}/results.json"
        for event_seed in protocol.AUDIT_EVENT_SEEDS
    } | {
        f"event_seed_{event_seed}/{arm}_trace.npz"
        for event_seed in protocol.AUDIT_EVENT_SEEDS
        for arm in common.LEARNED_ARMS
    }
    records = payload.get("files") or {}
    if set(records) != expected:
        raise ValueError(f"incomplete anchored audit: {destination}")
    for relative, record in records.items():
        path = destination / relative
        if not path.is_file() or protocol.file_record(path) != record:
            raise ValueError(f"anchored audit changed: {path}")
    for event_seed in protocol.AUDIT_EVENT_SEEDS:
        result = protocol.read_json(
            destination / f"event_seed_{event_seed}/results.json")
        if (result.get("schema") != protocol.EVENT_SCHEMA
                or result.get("status") != "complete"
                or result.get("training_seed") != seed
                or result.get("event_seed") != event_seed
                or result.get("mode_mask") != payload["mode_mask"]
                or {row["arm"] for row in result["switching"]}
                != set(common.AUDIT_ARMS)
                or len(result["stationary"])
                != len(common.AUDIT_ARMS) * len(protocol.MODES)):
            raise ValueError("invalid anchored event result")
        for arm in common.LEARNED_ARMS:
            with np.load(
                    destination
                    / f"event_seed_{event_seed}/{arm}_trace.npz",
                    allow_pickle=False) as trace:
                expected_rows = (
                    protocol.SWITCHING_EPISODES
                    * protocol.MAX_EPISODE_STEPS)
                if (trace["posterior_before"].shape
                        != (expected_rows, len(protocol.MODES))
                        or trace["context"].shape
                        != (expected_rows, len(protocol.MODES) + 1)
                        or not np.allclose(
                            np.sum(trace["posterior_before"], axis=-1),
                            1.0,
                            atol=1e-5)):
                    raise ValueError("invalid anchored posterior trace")
    return payload


def run(seed: int) -> None:
    seed = protocol.require_training_seed(seed)
    destination = protocol.audit_dir(seed)
    if (destination / "audit_manifest.json").is_file():
        validate(seed)
        print(f"ANCHORED AUDIT ALREADY COMPLETE: {destination}")
        return
    calibration_payload = calibration.validate(seed)
    mode_mask = np.asarray(
        calibration_payload["mode_mask"], dtype=bool)
    protocol.validate_frozen_estimator()
    controllers = {
        role: common.load_controller(role, seed)
        for role in protocol.BRANCH_ROLES
    }
    if controllers["robust_continue"][0].seed != controllers["anchored"][0].seed:
        raise ValueError("paired anchored controllers have different seeds")
    action_fns = {
        role: common.policy_action_fn(value[1])
        for role, value in controllers.items()
    }
    estimator, _ = common.load_estimator()

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp.", dir=destination.parent))
    try:
        records = {}
        for event_seed in protocol.AUDIT_EVENT_SEEDS:
            result, traces = _event_result(
                seed,
                event_seed,
                controllers,
                action_fns,
                estimator,
                mode_mask,
            )
            event_dir = temporary / f"event_seed_{event_seed}"
            protocol.write_json_atomic(event_dir / "results.json", result)
            relative = f"event_seed_{event_seed}/results.json"
            records[relative] = protocol.file_record(temporary / relative)
            for arm, trace in traces.items():
                relative = f"event_seed_{event_seed}/{arm}_trace.npz"
                _write_npz(temporary / relative, trace)
                records[relative] = protocol.file_record(
                    temporary / relative)
            print(
                f"anchored audit seed={seed} event={event_seed} complete",
                flush=True,
            )
        payload = {
            "schema": protocol.AUDIT_SCHEMA,
            "status": "complete",
            "identity": _identity(seed),
            "mode_mask": [bool(value) for value in mode_mask],
            "calibration_manifest": protocol.file_record(
                protocol.calibration_manifest(seed)),
            "controller_bundles": {
                role: protocol.file_record(
                    protocol.branch_manifest(role, seed))
                for role in protocol.BRANCH_ROLES
            },
            "frozen_model_manifest": protocol.FROZEN_MODEL_MANIFEST_RECORD,
            "frozen_model_parameters": protocol.FROZEN_MODEL_PARAMETER_RECORD,
            "files": records,
        }
        protocol.write_json_atomic(
            temporary / "audit_manifest.json", payload)
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
        f"ANCHORED AUDIT COMPLETE seed={seed} "
        f"mask={mode_mask.astype(int).tolist()}",
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
