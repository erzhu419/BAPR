"""Audit one unseen policy seed with inverse system identification."""
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
from jax_experiments.analysis import (
    regime_polarity_inverse_system_id as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_inverse_system_id_model as model_lib,
)
from jax_experiments.analysis import run_regime_polarity_posterior_audit as base
from jax_experiments.analysis import train_regime_polarity_posterior as common


ARMS = base.ARMS
LEARNED_ARMS = base.LEARNED_ARMS
EVENT_SCHEMA = "bapr.regime-polarity-inverse-system-id-event.v3"


class InverseSystemIDEstimator:
    """Sticky posterior over known transforms from inferred executed action."""

    def __init__(
        self,
        evidence_fn,
        model_params,
        gain_vectors,
        residual_variance,
        filter_config,
    ):
        self.evidence_fn = evidence_fn
        self.model_params = model_params
        self.gain_vectors = gain_vectors
        self.residual_variance = residual_variance
        self.filter_config = filter_config

    def initial_state(self):
        return np.full(
            (len(protocol.MODES),),
            1.0 / len(protocol.MODES),
            dtype=np.float64,
        )

    @staticmethod
    def probabilities(state):
        return np.asarray(state, dtype=np.float64)

    def step(self, state, obs, action, reward, next_obs):
        del reward
        log_likelihood, aleatoric, epistemic = self.evidence_fn(
            self.model_params,
            jnp.asarray(obs),
            jnp.asarray(next_obs),
            jnp.asarray(action),
            self.gain_vectors,
            self.residual_variance,
        )
        evidence = np.asarray(log_likelihood, dtype=np.float64)
        next_state = protocol.posterior_update(
            state, evidence, self.filter_config)
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
        "benchmark_role": "unseen_inverse_system_id_screen",
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


def _event_result(
    seed,
    event_seed,
    robust,
    oracle,
    estimator,
    model_manifest,
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
        "episodes_per_stationary_mode": confirmation.EPISODES_PER_TASK,
        "switching_episodes": confirmation.SWITCHING_EPISODES,
        "filter_config": estimator.filter_config.to_dict(),
        "model_parameter_file": model_manifest["parameter_file"],
        "online_inputs": _identity(seed)["online_inputs"],
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
            or payload.get("model_manifest")
            != protocol.file_record(protocol.MODEL_MANIFEST)
            or payload.get("model_parameters")
            != protocol.file_record(protocol.MODEL_PATH)):
        raise ValueError(f"invalid inverse-system-ID audit: {destination}")
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
        raise ValueError(f"incomplete inverse-system-ID audit: {destination}")
    for relative, record in records.items():
        path = destination / relative
        if not path.is_file() or protocol.file_record(path) != record:
            raise ValueError(f"inverse-system-ID audit file changed: {path}")
    for event_seed in protocol.TEST_EVENT_SEEDS:
        result = protocol.read_json(
            destination / f"event_seed_{event_seed}/results.json")
        if (result.get("schema") != EVENT_SCHEMA
                or result.get("status") != "complete"
                or result.get("training_seed") != seed
                or result.get("event_seed") != event_seed
                or result.get("online_inputs")
                != _identity(seed)["online_inputs"]
                or {row["arm"] for row in result["switching"]}
                != set(ARMS)
                or len(result["stationary"])
                != len(ARMS) * len(protocol.MODES)):
            raise ValueError("invalid inverse-system-ID event result")
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
                        "inverse-system-ID posterior trace has wrong horizon")
                if not np.allclose(
                        np.sum(trace["posterior_before"], axis=-1),
                        1.0,
                        atol=1e-5):
                    raise ValueError(
                        "inverse-system-ID posterior is not normalized")
    return payload


def run(seed: int):
    seed = int(seed)
    if seed not in protocol.TEST_CONTROLLER_SEEDS:
        raise ValueError(f"unknown inverse-system-ID audit seed {seed}")
    destination = protocol.audit_dir(seed)
    if (destination / "audit_manifest.json").is_file():
        try:
            validate_audit(seed)
        except (KeyError, OSError, TypeError, ValueError):
            pass
        else:
            print(
                "INVERSE SYSTEM ID AUDIT ALREADY COMPLETE: "
                f"{destination}",
                flush=True,
            )
            return

    robust = common._load_controller(confirmation, seed, "robust")
    oracle = common._load_controller(confirmation, seed, "oracle")
    model, filter_config, gains, variance, model_manifest = (
        model_lib.load_model(robust[1].obs_dim, robust[1].act_dim))
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
                model_manifest,
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
                "inverse-system-ID audit "
                f"seed={seed} event={event_seed} complete",
                flush=True,
            )
        payload = {
            "schema": protocol.AUDIT_SCHEMA,
            "status": "complete",
            "identity": _identity(seed),
            "model_manifest": protocol.file_record(
                protocol.MODEL_MANIFEST),
            "model_parameters": protocol.file_record(
                protocol.MODEL_PATH),
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
        f"INVERSE SYSTEM ID AUDIT COMPLETE: {destination}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    run(parse_args().seed)


if __name__ == "__main__":
    main()
