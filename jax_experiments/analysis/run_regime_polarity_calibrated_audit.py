"""Audit one unseen policy seed with the frozen calibrated posterior."""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.analysis import regime_polarity_confirmation as confirmation
from jax_experiments.analysis import regime_polarity_evidence_calibration as protocol
from jax_experiments.analysis import regime_polarity_posterior as v1
from jax_experiments.analysis import regime_polarity_posterior_model as model_lib
from jax_experiments.analysis import run_regime_polarity_posterior_audit as base
from jax_experiments.analysis import train_regime_polarity_posterior as trainer


ARMS = base.ARMS
LEARNED_ARMS = base.LEARNED_ARMS
EVENT_SCHEMA = "bapr.regime-polarity-calibrated-event.v2"


class CalibratedEstimator:
    """Combine frozen transition emissions with the frozen affine posterior."""

    def __init__(
        self,
        emission_fn,
        model_params,
        calibrator,
    ):
        self.emission_fn = emission_fn
        self.model_params = model_params
        self.calibrator = calibrator

    def initial_state(self):
        return self.calibrator.initial_state()

    def probabilities(self, state):
        return self.calibrator.probabilities(state)

    def step(self, state, obs, action, reward, next_obs):
        log_likelihood, aleatoric, epistemic = self.emission_fn(
            self.model_params,
            jnp.asarray(obs),
            jnp.asarray(action),
            jnp.asarray(reward),
            jnp.asarray(next_obs),
        )
        evidence = np.asarray(log_likelihood, dtype=np.float64)
        next_state = self.calibrator.update_from_evidence(
            state, evidence)
        return (
            next_state,
            evidence,
            np.asarray(aleatoric, dtype=np.float64),
            np.asarray(epistemic, dtype=np.float64),
        )


def _identity(seed: int) -> dict[str, Any]:
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "env": protocol.ENV,
        "family": protocol.FAMILY,
        "training_seed": int(seed),
        "event_seeds": list(protocol.TEST_EVENT_SEEDS),
        "arms": list(ARMS),
        "benchmark_role": "unseen_frozen_calibrated_posterior_screen",
    }


def _event_result(
    seed: int,
    event_seed: int,
    robust,
    oracle,
    estimator,
    calibrator_manifest,
    forward_manifest,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
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
        "episodes_per_stationary_mode": confirmation.EPISODES_PER_TASK,
        "switching_episodes": confirmation.SWITCHING_EPISODES,
        "calibrator_config": calibrator_manifest["calibrator_config"],
        "calibrator_parameter_file": (
            calibrator_manifest["parameter_file"]),
        "forward_model_parameter_file": (
            forward_manifest["parameter_file"]),
        "stationary": stationary,
        "stationary_posterior_metrics": stationary_metrics,
        "switching": switching,
    }, traces


def validate_audit(seed: int) -> dict[str, Any]:
    seed = int(seed)
    destination = protocol.audit_dir(seed)
    payload = protocol.read_json(destination / "audit_manifest.json")
    if (payload.get("schema") != protocol.AUDIT_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("identity") != _identity(seed)
            or payload.get("calibrator_manifest")
            != protocol.file_record(protocol.MODEL_MANIFEST)
            or payload.get("calibrator_parameters")
            != protocol.file_record(protocol.MODEL_PATH)
            or payload.get("forward_model_manifest")
            != protocol.file_record(v1.MODEL_MANIFEST)
            or payload.get("forward_model_parameters")
            != protocol.file_record(v1.MODEL_PATH)):
        raise ValueError(f"invalid calibrated audit: {destination}")
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
        raise ValueError(f"incomplete calibrated audit: {destination}")
    for relative, record in records.items():
        path = destination / relative
        if not path.is_file() or protocol.file_record(path) != record:
            raise ValueError(f"calibrated audit file changed: {path}")
    for event_seed in protocol.TEST_EVENT_SEEDS:
        result = protocol.read_json(
            destination / f"event_seed_{event_seed}/results.json")
        if (result.get("schema") != EVENT_SCHEMA
                or result.get("status") != "complete"
                or result.get("training_seed") != seed
                or result.get("event_seed") != event_seed
                or {row["arm"] for row in result["switching"]}
                != set(ARMS)
                or len(result["stationary"])
                != len(ARMS) * len(protocol.MODES)):
            raise ValueError("invalid calibrated event result")
        for arm in LEARNED_ARMS:
            with np.load(
                    destination
                    / f"event_seed_{event_seed}/{arm}_trace.npz",
                    allow_pickle=False) as trace:
                expected_shape = (
                    confirmation.SWITCHING_EPISODES
                    * protocol.MAX_EPISODE_STEPS,
                    len(protocol.MODES),
                )
                if trace["posterior_before"].shape != expected_shape:
                    raise ValueError(
                        "calibrated posterior trace has wrong horizon")
                if not np.allclose(
                        np.sum(trace["posterior_before"], axis=-1),
                        1.0,
                        atol=1e-5):
                    raise ValueError(
                        "calibrated posterior is not normalized")
    return payload


def run(seed: int) -> None:
    seed = int(seed)
    if seed not in protocol.TEST_CONTROLLER_SEEDS:
        raise ValueError(f"unknown calibrated audit seed {seed}")
    destination = protocol.audit_dir(seed)
    if (destination / "audit_manifest.json").is_file():
        try:
            validate_audit(seed)
        except (KeyError, OSError, TypeError, ValueError):
            pass
        else:
            print(
                "POLARITY CALIBRATED AUDIT ALREADY COMPLETE: "
                f"{destination}",
                flush=True,
            )
            return

    robust = trainer._load_controller(confirmation, seed, "robust")
    oracle = trainer._load_controller(confirmation, seed, "oracle")
    forward_model, _, forward_manifest = model_lib.load_model(
        robust[1].obs_dim, robust[1].act_dim)
    calibrator, calibrator_manifest = protocol.load_calibrator()
    estimator = CalibratedEstimator(
        model_lib.one_step_emission(forward_model),
        nnx.state(forward_model, nnx.Param),
        calibrator,
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
                calibrator_manifest,
                forward_manifest,
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
                "calibrated audit "
                f"seed={seed} event={event_seed} complete",
                flush=True,
            )
        payload = {
            "schema": protocol.AUDIT_SCHEMA,
            "status": "complete",
            "identity": _identity(seed),
            "calibrator_manifest": protocol.file_record(
                protocol.MODEL_MANIFEST),
            "calibrator_parameters": protocol.file_record(
                protocol.MODEL_PATH),
            "forward_model_manifest": protocol.file_record(
                v1.MODEL_MANIFEST),
            "forward_model_parameters": protocol.file_record(v1.MODEL_PATH),
            "controller_bundles": {
                role: protocol.file_record(
                    confirmation.bundle_manifest(
                        protocol.ENV, role, seed))
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
        f"POLARITY CALIBRATED AUDIT COMPLETE: {destination}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        choices=protocol.TEST_CONTROLLER_SEEDS,
        type=int,
        required=True,
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for scheduler input staging")
    run(args.seed)


if __name__ == "__main__":
    main()
