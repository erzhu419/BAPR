"""Audit frozen confirm3 routing with specialist-trajectory system ID v5."""
from __future__ import annotations

import argparse
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_model_v5 as model_lib,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_expected_action_system_id_v5 as protocol,
)
from jax_experiments.analysis import (
    regime_polarity_specialist_router_v1 as source_protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_router_audit_v1 as source_audit,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_router_diagnostic_v2 as diagnostic_audit,
)
from jax_experiments.analysis import (
    run_regime_polarity_specialist_sticky_router_audit_v3 as sticky_audit,
)


def _identity(seed: int, event_seed: int) -> dict[str, Any]:
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "benchmark_role": "specialist_trajectory_estimator_screen",
        "env": protocol.ENV,
        "family": protocol.FAMILY,
        "source_seed": protocol.require_source_seed(seed),
        "event_seed": protocol.require_event_seed(event_seed),
        "arms": list(protocol.ARMS),
        "primary_arm": protocol.PRIMARY_ARM,
        "online_inputs": [
            "observation",
            "commanded_action",
            "next_observation",
        ],
    }


def _load_stack(seed: int):
    stack = source_audit._load_stack(seed)
    controller = source_audit.source_audit._load_controller(
        "robust_sac", seed)
    obs_dim = int(controller["agent"].obs_dim)
    act_dim = int(controller["agent"].act_dim)
    stack["estimator_factory"] = lambda: model_lib.make_estimator(
        obs_dim, act_dim)
    return stack


def _archived_source_records(seed: int) -> dict[str, Any]:
    """Recover immutable source identities after historical bundles are pruned."""
    seed = protocol.require_source_seed(seed)
    first_event = protocol.read_json(
        protocol.event_result(seed, protocol.AUDIT_EVENT_SEEDS[0]))
    records = first_event.get("source_bundles") or {}
    if set(records) != set(source_audit.source.ROLES):
        raise ValueError("invalid archived source-bundle records")
    for record in records.values():
        if (
            set(record) != {"sha256", "size"}
            or not str(record["sha256"])
            or int(record["size"]) <= 0
        ):
            raise ValueError("invalid archived source-bundle record")

    manifest = protocol.read_json(protocol.MODEL_MANIFEST)
    frozen = manifest.get("source_bundles") or {}
    split = "train" if seed in protocol.TRAIN_SOURCE_SEEDS else "validation"
    for role in protocol.STATIONARY_ARMS:
        key = f"{split}/{role}/seed_{seed}"
        if records.get(role) != frozen.get(key):
            raise ValueError("archived source identity disagrees with v5 model")
    return records


def _expected_source_records(seed: int) -> dict[str, Any]:
    try:
        return source_protocol.source_records(seed)
    except FileNotFoundError:
        return _archived_source_records(seed)


def evaluate(seed: int, event_seed: int) -> dict[str, Any]:
    seed = protocol.require_source_seed(seed)
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
        "source_bundles": source_protocol.source_records(seed),
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
        raise ValueError("invalid specialist estimator audit event")
    traces = set()
    for arm in protocol.ARMS:
        row = payload["switching"][arm]
        values = row.get("returns") or []
        if (
            len(values) != sticky_audit.protocol.SWITCHING_EPISODES
            or not all(math.isfinite(float(value)) for value in values)
            or int(row.get("total_actions", -1))
            != (
                sticky_audit.protocol.SWITCHING_EPISODES
                * sticky_audit.protocol.MAX_EPISODE_STEPS
            )
        ):
            raise ValueError("invalid specialist estimator audit result")
        traces.add(str(row.get("mode_trace_sha256") or ""))
    if len(traces) != 1 or "" in traces:
        raise ValueError("specialist estimator arms used different mode streams")
    if payload["switching"]["dynamic_oracle"][
        "adaptive_mode_accuracy"] != 1.0:
        raise ValueError("dynamic oracle used a wrong specialist")


def validate_audit(seed: int) -> dict[str, Any]:
    seed = protocol.require_source_seed(seed)
    manifest = protocol.read_json(protocol.audit_manifest(seed))
    expected_files = {
        str(event_seed): protocol.file_record(
            protocol.event_result(seed, event_seed))
        for event_seed in protocol.AUDIT_EVENT_SEEDS
    }
    if (
        manifest.get("schema") != protocol.AUDIT_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("identity")
        != {
            "protocol_version": protocol.PROTOCOL_VERSION,
            "source_seed": seed,
            "event_seeds": list(protocol.AUDIT_EVENT_SEEDS),
        }
        or manifest.get("event_files") != expected_files
    ):
        raise ValueError("invalid specialist estimator audit")
    for event_seed in protocol.AUDIT_EVENT_SEEDS:
        validate_event(
            protocol.read_json(protocol.event_result(seed, event_seed)),
            seed,
            event_seed,
        )
    return manifest


def run(seed: int) -> None:
    seed = protocol.require_source_seed(seed)
    destination = protocol.audit_dir(seed)
    if protocol.audit_manifest(seed).is_file():
        try:
            validate_audit(seed)
        except (KeyError, OSError, TypeError, ValueError):
            pass
        else:
            print(f"SPECIALIST ESTIMATOR AUDIT ALREADY COMPLETE: {destination}")
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
        for event_seed in protocol.AUDIT_EVENT_SEEDS:
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
                    "source_seed": seed,
                    "event_seeds": list(protocol.AUDIT_EVENT_SEEDS),
                },
                "event_files": records,
            },
        )
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    validate_audit(seed)
    print(f"SPECIALIST ESTIMATOR AUDIT COMPLETE: seed={seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.seed)


if __name__ == "__main__":
    main()
