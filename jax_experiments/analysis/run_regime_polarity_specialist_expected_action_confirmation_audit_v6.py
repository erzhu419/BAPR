"""Audit frozen v5 routing on fresh independently trained policy banks."""
from __future__ import annotations

import argparse
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.analysis import final_task_sweep
from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_confirmation_v6 as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_model_v5 as model_lib,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_router_diagnostic_v2 as diagnostic_audit,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_sticky_router_audit_v3 as sticky_audit,
)
from jax_experiments.analysis.run_regime_polarity_confirmation_source_controller_v6 import (
    validate_bundle,
)
from jax_experiments.common.checkpoint import load_checkpoint
from jax_experiments.common.logging import Logger
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.train import make_algo, make_env


def _identity(seed: int, event_seed: int) -> dict[str, Any]:
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "benchmark_role": "fresh_policy_bank_confirmation",
        "env": protocol.ENV,
        "family": protocol.FAMILY,
        "training_seed": protocol.require_training_seed(seed),
        "event_seed": protocol.require_event_seed(event_seed),
        "arms": list(protocol.ARMS),
        "primary_arm": protocol.PRIMARY_ARM,
        "frozen_estimator_protocol": "v5",
        "online_inputs": [
            "observation",
            "commanded_action",
            "next_observation",
        ],
    }


def _archived_source_records(seed: int) -> dict[str, Any]:
    """Recover frozen controller identities after final bundles are pruned."""
    seed = protocol.require_training_seed(seed)
    manifest = protocol.read_json(protocol.audit_manifest(seed))
    records = manifest.get("source_bundles") or {}
    if set(records) != set(protocol.ROLES):
        raise ValueError("invalid archived v6 source-bundle records")
    for record in records.values():
        if (
            set(record) != {"sha256", "size"}
            or not str(record["sha256"])
            or int(record["size"]) <= 0
        ):
            raise ValueError("invalid archived v6 source-bundle record")
    for event_seed in protocol.EVENT_SEEDS:
        event = protocol.read_json(protocol.event_result(seed, event_seed))
        if event.get("source_bundles") != records:
            raise ValueError("v6 events disagree on source-bundle identity")
    return records


def _expected_source_records(seed: int) -> dict[str, Any]:
    archived = _archived_source_records(seed)
    live_count = 0
    for role in protocol.ROLES:
        path = protocol.bundle_manifest(role, seed)
        if not path.is_file():
            continue
        live_count += 1
        if protocol.file_record(path) != archived[role]:
            raise ValueError("live v6 bundle disagrees with archived identity")
    if live_count == len(protocol.ROLES):
        return protocol.source_records(seed)
    return archived


def _load_controller(role: str, seed: int) -> dict[str, Any]:
    validate_bundle(role, seed)
    directory = protocol.bundle_dir(role, seed)
    config = final_task_sweep.load_config(directory)
    config.stochastic_mode_fixed_id = -1
    env = make_env(config, seed_offset=0)
    tasks = env.sample_tasks(len(protocol.MODES))
    agent = make_algo(config.algo, env.obs_dim, env.act_dim, config)
    if hasattr(agent, "set_task_metadata"):
        agent.set_task_metadata(tasks)
    replay = ReplayBuffer(
        env.obs_dim,
        env.act_dim,
        capacity=1,
        belief_dim=getattr(agent, "belief_dim", 0),
    )
    with tempfile.TemporaryDirectory() as temporary:
        logger = Logger(temporary)
        next_iteration, total_steps = load_checkpoint(
            str(directory / "checkpoints"),
            agent,
            replay,
            logger,
            config.algo,
            load_replay_buffer=False,
        )
    if (
        next_iteration != protocol.MAX_ITERS
        or total_steps != protocol.FINAL_TOTAL_STEPS
        or int(agent.update_count) != protocol.FINAL_UPDATE_COUNT
    ):
        raise ValueError(f"stale confirmation controller: {directory}")
    if hasattr(env, "close"):
        env.close()
    return {
        "config": config,
        "agent": agent,
        "policy_graphdef": nnx.graphdef(agent.policy),
        "policy_params": nnx.state(agent.policy, nnx.Param),
        "context_graphdef": None,
        "context_params": None,
    }


def _action_fn(controller):
    graphdef = controller["policy_graphdef"]

    @jax.jit
    def action(policy_params, observation):
        policy = nnx.merge(graphdef, policy_params)
        obs = jnp.asarray(observation, dtype=jnp.float32)[None]
        return policy.deterministic(obs)[0]

    return action


def _load_stack(seed: int) -> dict[str, Any]:
    controllers = {
        role: _load_controller(role, seed)
        for role in protocol.ROLES
    }
    actions = {}
    for role, controller in controllers.items():
        action_fn = _action_fn(controller)

        def action(observation, *, _fn=action_fn, _source=controller):
            return np.asarray(
                _fn(_source["policy_params"], observation),
                dtype=np.float32,
            )

        actions[role] = action
    reference = controllers["robust_sac"]
    return {
        "config": reference["config"],
        "actions": actions,
        "estimator_factory": lambda: model_lib.make_estimator(
            int(reference["agent"].obs_dim),
            int(reference["agent"].act_dim),
        ),
    }


def evaluate(seed: int, event_seed: int) -> dict[str, Any]:
    seed = protocol.require_training_seed(seed)
    event_seed = protocol.require_event_seed(event_seed)
    stack = _load_stack(seed)
    switching = {}
    for arm in protocol.ARMS:
        if arm == protocol.PRIMARY_ARM:
            switching[arm] = sticky_audit._sticky_arm(
                stack, arm, event_seed)
        else:
            switching[arm] = diagnostic_audit._switching_arm(
                stack, arm, event_seed)
    return {
        "schema": protocol.EVENT_SCHEMA,
        "status": "complete",
        "identity": _identity(seed, event_seed),
        "source_bundles": protocol.source_records(seed),
        "estimator": protocol.estimator_records(),
        "switching": switching,
    }


def validate_event(payload: dict, seed: int, event_seed: int) -> None:
    if (
        payload.get("schema") != protocol.EVENT_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("identity") != _identity(seed, event_seed)
        or payload.get("source_bundles") != _expected_source_records(seed)
        or payload.get("estimator") != protocol.estimator_records()
        or set(payload.get("switching") or {}) != set(protocol.ARMS)
    ):
        raise ValueError("invalid fresh policy-bank audit event")
    traces = set()
    for arm in protocol.ARMS:
        row = payload["switching"][arm]
        values = row.get("returns") or []
        if (
            len(values) != protocol.SWITCHING_EPISODES
            or not all(math.isfinite(float(value)) for value in values)
            or int(row.get("total_actions", -1))
            != protocol.SWITCHING_EPISODES * protocol.MAX_EPISODE_STEPS
        ):
            raise ValueError("invalid fresh policy-bank audit result")
        traces.add(str(row.get("mode_trace_sha256") or ""))
    if len(traces) != 1 or "" in traces:
        raise ValueError("confirmation arms used different switching streams")
    if payload["switching"]["dynamic_oracle"][
        "adaptive_mode_accuracy"] != 1.0:
        raise ValueError("dynamic oracle used a wrong specialist")


def validate_audit(seed: int) -> dict[str, Any]:
    seed = protocol.require_training_seed(seed)
    manifest = protocol.read_json(protocol.audit_manifest(seed))
    expected_files = {
        str(event_seed): protocol.file_record(
            protocol.event_result(seed, event_seed))
        for event_seed in protocol.EVENT_SEEDS
    }
    if (
        manifest.get("schema") != protocol.AUDIT_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("identity")
        != {
            "protocol_version": protocol.PROTOCOL_VERSION,
            "training_seed": seed,
            "event_seeds": list(protocol.EVENT_SEEDS),
        }
        or manifest.get("source_bundles") != _expected_source_records(seed)
        or manifest.get("estimator") != protocol.estimator_records()
        or manifest.get("event_files") != expected_files
    ):
        raise ValueError("invalid fresh policy-bank audit manifest")
    for event_seed in protocol.EVENT_SEEDS:
        validate_event(
            protocol.read_json(protocol.event_result(seed, event_seed)),
            seed,
            event_seed,
        )
    return manifest


def run(seed: int) -> None:
    seed = protocol.require_training_seed(seed)
    destination = protocol.audit_dir(seed)
    if protocol.audit_manifest(seed).is_file():
        try:
            validate_audit(seed)
        except (KeyError, OSError, TypeError, ValueError):
            pass
        else:
            print(f"FRESH POLICY-BANK AUDIT ALREADY COMPLETE: {destination}")
            return
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
        for event_seed in protocol.EVENT_SEEDS:
            result = temporary / f"event_seed_{event_seed}" / "results.json"
            protocol.write_json_atomic(result, evaluate(seed, event_seed))
            records[str(event_seed)] = protocol.file_record(result)
        protocol.write_json_atomic(
            temporary / "audit_manifest.json",
            {
                "schema": protocol.AUDIT_SCHEMA,
                "status": "complete",
                "identity": {
                    "protocol_version": protocol.PROTOCOL_VERSION,
                    "training_seed": seed,
                    "event_seeds": list(protocol.EVENT_SEEDS),
                },
                "source_bundles": protocol.source_records(seed),
                "estimator": protocol.estimator_records(),
                "event_files": records,
            },
        )
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_audit(seed)
    print(f"FRESH POLICY-BANK AUDIT COMPLETE: seed={seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.seed)


if __name__ == "__main__":
    main()
