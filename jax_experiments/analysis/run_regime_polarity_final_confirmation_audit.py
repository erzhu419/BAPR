"""Audit the frozen v4 estimator on one newly trained policy seed."""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from flax import nnx

from jax_experiments.analysis import (
    regime_polarity_expected_action_system_id_model as model_lib,
)
from jax_experiments.analysis import (
    regime_polarity_final_confirmation as protocol,
)
from jax_experiments.analysis import run_regime_polarity_posterior_audit as base
from jax_experiments.analysis import (
    run_regime_polarity_inverse_system_id_audit as inverse_audit,
)
from jax_experiments.analysis import train_regime_polarity_posterior as common


ARMS = base.ARMS
LEARNED_ARMS = base.LEARNED_ARMS
EVENT_SCHEMA = "bapr.regime-polarity-final-confirmation-event.v5"
InverseSystemIDEstimator = inverse_audit.InverseSystemIDEstimator


def _identity(seed: int) -> dict[str, Any]:
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "env": protocol.ENV,
        "family": protocol.FAMILY,
        "training_seed": int(seed),
        "event_seeds": list(protocol.TEST_EVENT_SEEDS),
        "arms": list(ARMS),
        "benchmark_role": "independent_frozen_bapr_final_confirmation",
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
        "frozen_model_manifest": protocol.FROZEN_MODEL_MANIFEST_RECORD,
        "frozen_model_parameters": protocol.FROZEN_MODEL_PARAMETER_RECORD,
    }


def _event_result(
    seed,
    event_seed,
    robust,
    oracle,
    estimator,
):
    robust_config, robust_agent, robust_state = robust
    oracle_config, oracle_agent, oracle_state = oracle
    if robust_config.seed != oracle_config.seed:
        raise ValueError("paired controllers have different training seeds")
    tasks = [{"mode_id": mode} for mode in protocol.MODES]
    action_fns = {
        "robust": base._policy_action_fn(robust_agent),
        "oracle": base._policy_action_fn(oracle_agent),
    }
    stationary = []
    stationary_metrics = []
    switching = []
    traces = {}
    for arm in ARMS:
        source = "robust" if arm == "robust" else "oracle"
        agent = robust_agent if source == "robust" else oracle_agent
        state = robust_state if source == "robust" else oracle_state
        config = robust_config if source == "robust" else oracle_config
        action_fn = action_fns[source]
        arm_stationary, arm_metrics = base._strict_stationary(
            config,
            tasks,
            arm,
            state,
            action_fn,
            estimator,
            event_seed,
        )
        arm_switching, trace = base._strict_switching(
            config,
            tasks,
            arm,
            state,
            action_fn,
            estimator,
            event_seed,
        )
        stationary.extend(arm_stationary)
        stationary_metrics.extend(arm_metrics)
        switching.append(arm_switching)
        if arm in LEARNED_ARMS:
            traces[arm] = trace
    return {
        "schema": EVENT_SCHEMA,
        "status": "complete",
        "training_seed": int(seed),
        "event_seed": int(event_seed),
        "env": protocol.ENV,
        "family": protocol.FAMILY,
        "dwell_steps": protocol.DWELL_STEPS,
        "max_episode_steps": protocol.MAX_EPISODE_STEPS,
        "episodes_per_stationary_mode": protocol.EPISODES_PER_TASK,
        "switching_episodes": protocol.SWITCHING_EPISODES,
        "filter_config": estimator.filter_config.to_dict(),
        "model_manifest": protocol.FROZEN_MODEL_MANIFEST_RECORD,
        "model_parameters": protocol.FROZEN_MODEL_PARAMETER_RECORD,
        "online_inputs": _identity(seed)["online_inputs"],
        "stationary": stationary,
        "stationary_posterior_metrics": stationary_metrics,
        "switching": switching,
    }, traces


def validate_audit(seed: int) -> dict[str, Any]:
    seed = protocol.require_training_seed(seed)
    protocol.validate_frozen_estimator()
    destination = protocol.audit_dir(seed)
    payload = protocol.read_json(destination / "audit_manifest.json")
    if (payload.get("schema") != protocol.AUDIT_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("identity") != _identity(seed)
            or payload.get("model_manifest")
            != protocol.FROZEN_MODEL_MANIFEST_RECORD
            or payload.get("model_parameters")
            != protocol.FROZEN_MODEL_PARAMETER_RECORD):
        raise ValueError(f"invalid final-confirmation audit: {destination}")
    expected = {
        f"event_seed_{event_seed}/results.json"
        for event_seed in protocol.TEST_EVENT_SEEDS
    } | {
        f"event_seed_{event_seed}/{arm}_trace.npz"
        for event_seed in protocol.TEST_EVENT_SEEDS
        for arm in LEARNED_ARMS
    }
    records = payload.get("files") or {}
    if set(records) != expected:
        raise ValueError(f"incomplete final-confirmation audit: {destination}")
    for relative, record in records.items():
        path = destination / relative
        if not path.is_file() or protocol.file_record(path) != record:
            raise ValueError(f"final-confirmation file changed: {path}")
    for event_seed in protocol.TEST_EVENT_SEEDS:
        result = protocol.read_json(
            destination / f"event_seed_{event_seed}/results.json")
        if (result.get("schema") != EVENT_SCHEMA
                or result.get("status") != "complete"
                or result.get("training_seed") != seed
                or result.get("event_seed") != event_seed
                or result.get("model_manifest")
                != protocol.FROZEN_MODEL_MANIFEST_RECORD
                or result.get("model_parameters")
                != protocol.FROZEN_MODEL_PARAMETER_RECORD
                or {row["arm"] for row in result["switching"]}
                != set(ARMS)
                or len(result["stationary"])
                != len(ARMS) * len(protocol.MODES)):
            raise ValueError("invalid final-confirmation event result")
        for arm in LEARNED_ARMS:
            with np.load(
                    destination
                    / f"event_seed_{event_seed}/{arm}_trace.npz",
                    allow_pickle=False) as trace:
                expected_shape = (
                    protocol.SWITCHING_EPISODES
                    * protocol.MAX_EPISODE_STEPS,
                    len(protocol.MODES),
                )
                if trace["posterior_before"].shape != expected_shape:
                    raise ValueError(
                        "final-confirmation posterior trace has wrong horizon")
                if not np.allclose(
                        np.sum(trace["posterior_before"], axis=-1),
                        1.0,
                        atol=1e-5):
                    raise ValueError(
                        "final-confirmation posterior is not normalized")
    return payload


def run(seed: int):
    seed = protocol.require_training_seed(seed)
    destination = protocol.audit_dir(seed)
    if (destination / "audit_manifest.json").is_file():
        try:
            validate_audit(seed)
        except (KeyError, OSError, TypeError, ValueError):
            pass
        else:
            print(
                "FINAL CONFIRMATION AUDIT ALREADY COMPLETE: "
                f"{destination}",
                flush=True,
            )
            return

    protocol.validate_frozen_estimator()
    robust = common._load_controller(protocol, seed, "robust")
    oracle = common._load_controller(protocol, seed, "oracle")
    model, filter_config, gains, variance, _ = model_lib.load_model(
        robust[1].obs_dim, robust[1].act_dim)
    estimator = InverseSystemIDEstimator(
        model_lib.one_step_evidence(model),
        nnx.state(model, nnx.Param),
        gains,
        variance,
        filter_config,
    )

    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp.", dir=destination.parent))
    try:
        records = {}
        for event_seed in protocol.TEST_EVENT_SEEDS:
            result, traces = _event_result(
                seed,
                event_seed,
                robust,
                oracle,
                estimator,
            )
            event_dir = temporary / f"event_seed_{event_seed}"
            protocol.write_json_atomic(event_dir / "results.json", result)
            relative = f"event_seed_{event_seed}/results.json"
            records[relative] = protocol.file_record(temporary / relative)
            for arm, trace in traces.items():
                relative = f"event_seed_{event_seed}/{arm}_trace.npz"
                base._write_npz(temporary / relative, trace)
                records[relative] = protocol.file_record(
                    temporary / relative)
            print(
                "final-confirmation audit "
                f"seed={seed} event={event_seed} complete",
                flush=True,
            )
        payload = {
            "schema": protocol.AUDIT_SCHEMA,
            "status": "complete",
            "identity": _identity(seed),
            "model_manifest": protocol.FROZEN_MODEL_MANIFEST_RECORD,
            "model_parameters": protocol.FROZEN_MODEL_PARAMETER_RECORD,
            "controller_bundles": {
                role: protocol.file_record(
                    protocol.bundle_manifest(protocol.ENV, role, seed))
                for role in protocol.ROLES
            },
            "files": records,
        }
        protocol.write_json_atomic(
            temporary / "audit_manifest.json", payload)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_audit(seed)
    print(
        f"FINAL CONFIRMATION AUDIT COMPLETE: {destination}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    run(parse_args().seed)


if __name__ == "__main__":
    main()
