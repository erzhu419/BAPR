"""Cross-evaluate v6 robust and specialist controllers on fixed modes."""
from __future__ import annotations

import argparse
import copy
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import jax
import numpy as np

from jax_experiments.analysis import final_task_sweep
from jax_experiments.analysis import (
    regime_polarity_specialist_capacity_diagnostic_v7 as protocol,
)
from jax_experiments.analysis.run_regime_polarity_specialist_expected_action_confirmation_audit_v6 import (
    _load_controller,
)
from jax_experiments.train import make_env


def _identity(seed: int, event_seed: int) -> dict[str, Any]:
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "benchmark_role": "stationary_controller_capacity_diagnostic",
        "env": protocol.ENV,
        "family": protocol.FAMILY,
        "training_seed": protocol.require_training_seed(seed),
        "event_seed": protocol.require_event_seed(event_seed),
        "roles": list(protocol.ROLES),
        "modes": list(protocol.MODES),
        "episodes_per_task": protocol.EPISODES_PER_TASK,
        "episode_horizon": protocol.MAX_EPISODE_STEPS,
    }


def _record(rewards, dones) -> dict[str, Any]:
    returns, terminated, steps = final_task_sweep.episode_returns(
        np.asarray(rewards),
        np.asarray(dones),
        protocol.EPISODES_PER_TASK,
        protocol.MAX_EPISODE_STEPS,
    )
    return {
        "returns": [float(value) for value in returns],
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "terminated": [bool(value) for value in terminated],
        "terminated_rate": float(np.mean(terminated)),
        "steps": [int(value) for value in steps],
        "total_actions": (
            protocol.EPISODES_PER_TASK * protocol.MAX_EPISODE_STEPS
        ),
    }


def _stationary(controller: dict[str, Any], seed: int, event_seed: int) -> dict:
    rows = {}
    for mode in protocol.MODES:
        config = copy.deepcopy(controller["config"])
        config.stochastic_mode_fixed_id = int(mode)
        env = make_env(config, seed_offset=event_seed - seed)
        tasks = env.sample_tasks(len(protocol.MODES))
        env.set_nonstationary_para(tasks)
        env.set_task(tasks[mode])
        env.build_rollout_fn(
            controller["policy_graphdef"], controller["context_graphdef"])
        key = jax.random.PRNGKey(event_seed * 100 + mode)
        rewards, dones = env.eval_rollout(
            controller["policy_params"],
            protocol.EPISODES_PER_TASK * protocol.MAX_EPISODE_STEPS,
            key,
            context_params=controller["context_params"],
            episode_horizon=protocol.MAX_EPISODE_STEPS,
        )
        rows[str(mode)] = _record(rewards, dones)
        if hasattr(env, "close"):
            env.close()
    return rows


def evaluate(seed: int, event_seed: int) -> dict[str, Any]:
    seed = protocol.require_training_seed(seed)
    event_seed = protocol.require_event_seed(event_seed)
    source_records = protocol.live_source_records(seed)
    stationary = {
        role: _stationary(_load_controller(role, seed), seed, event_seed)
        for role in protocol.ROLES
    }
    stationary["dynamic_oracle"] = {
        str(mode): stationary[f"specialist_{mode}"][str(mode)]
        for mode in protocol.MODES
    }
    return {
        "schema": protocol.EVENT_SCHEMA,
        "status": "complete",
        "identity": _identity(seed, event_seed),
        "source_bundles": source_records,
        "stationary": stationary,
    }


def validate_event(payload: dict, seed: int, event_seed: int) -> None:
    if (
        payload.get("schema") != protocol.EVENT_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("identity") != _identity(seed, event_seed)
        or payload.get("source_bundles") != protocol.source_records(seed)
        or set(payload.get("stationary") or {})
        != {*protocol.ROLES, "dynamic_oracle"}
    ):
        raise ValueError("invalid stationary capacity event")
    for role in (*protocol.ROLES, "dynamic_oracle"):
        rows = payload["stationary"][role]
        if set(rows) != {str(mode) for mode in protocol.MODES}:
            raise ValueError("stationary capacity event has missing modes")
        for row in rows.values():
            values = row.get("returns") or []
            if (
                len(values) != protocol.EPISODES_PER_TASK
                or not all(math.isfinite(float(value)) for value in values)
                or len(row.get("terminated") or [])
                != protocol.EPISODES_PER_TASK
                or len(row.get("steps") or [])
                != protocol.EPISODES_PER_TASK
                or int(row.get("total_actions", -1))
                != protocol.EPISODES_PER_TASK * protocol.MAX_EPISODE_STEPS
            ):
                raise ValueError("invalid stationary capacity rollout")
    for mode in protocol.MODES:
        if (
            payload["stationary"]["dynamic_oracle"][str(mode)]
            != payload["stationary"][f"specialist_{mode}"][str(mode)]
        ):
            raise ValueError("dynamic oracle is not the diagonal specialist")


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
        or manifest.get("source_bundles") != protocol.source_records(seed)
        or manifest.get("event_files") != expected_files
    ):
        raise ValueError("invalid stationary capacity audit manifest")
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
            print(f"STATIONARY CAPACITY AUDIT ALREADY COMPLETE: {destination}")
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
        source_records = protocol.live_source_records(seed)
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
                "source_bundles": source_records,
                "event_files": records,
            },
        )
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_audit(seed)
    print(f"STATIONARY CAPACITY AUDIT COMPLETE: seed={seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.seed)


if __name__ == "__main__":
    main()
