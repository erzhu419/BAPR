"""Run one fresh-seed CUSUM confirmation group."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

from flax import nnx

from jax_experiments.analysis import (
    bapr_v3_learned_control_router as estimator,
)
from jax_experiments.analysis import (
    bapr_v3_structured_channel_confirmation as protocol,
)
from jax_experiments.analysis import (
    run_bapr_v3_independent_specialist_audit as specialist_audit,
)
from jax_experiments.analysis import (
    run_bapr_v3_learned_control_router_audit as estimator_audit,
)
from jax_experiments.analysis import (
    run_bapr_v3_utility_aware_router_audit as base_audit,
)


def source_records():
    paths = (
        Path(__file__).resolve(),
        Path(protocol.__file__).resolve(),
        Path(base_audit.__file__).resolve(),
    )
    return {
        str(path.relative_to(protocol.ROOT)): protocol.file_record(path)
        for path in paths
    }


def validate_group(payload) -> None:
    event_seed = int(payload.get("event_seed", -1))
    if (payload.get("schema") != protocol.GROUP_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("family") != protocol.FAMILY
            or payload.get("env") != protocol.ENV
            or payload.get("decision_variant") != protocol.DECISION_VARIANT
            or event_seed not in protocol.EVENT_SEEDS
            or payload.get("primary_endpoints") != {
                "stationary_noninferiority_margin":
                    protocol.STATIONARY_NONINFERIORITY_MARGIN,
                "termination_rate_margin": protocol.TERMINATION_RATE_MARGIN,
                "min_oracle_recovery": protocol.MIN_ORACLE_RECOVERY,
                "min_full_cycle_wins": protocol.MIN_FULL_CYCLE_WINS,
                "min_committed_route_accuracy":
                    protocol.MIN_COMMITTED_ROUTE_ACCURACY,
            }
            or payload.get("source_files") != source_records()
            or set(payload.get("stationary") or {})
            != set(base_audit.CONTROLLERS)
            or set(payload.get("switching") or {})
            != {"slow_pair", "full_cycle"}):
        raise ValueError("invalid CUSUM confirmation group identity")

    for controller in base_audit.CONTROLLERS:
        if set(payload["stationary"][controller]) != {"0", "1", "2", "3"}:
            raise ValueError("incomplete confirmation stationary modes")
        for record in payload["stationary"][controller].values():
            episodes = record.get("episodes") or []
            if (len(episodes) != base_audit.EPISODES_PER_TASK
                    or not all(math.isfinite(float(row["return"]))
                               for row in episodes)):
                raise ValueError("invalid confirmation stationary episodes")
        for kind in ("slow_pair", "full_cycle"):
            episodes = payload["switching"][kind][controller].get(
                "episodes") or []
            if (len(episodes) != base_audit.SWITCHING_EPISODES
                    or not all(math.isfinite(float(row["return"]))
                               and int(row.get("termination_count", -1)) >= 0
                               and int(row.get("first_done_step", -1)) > 0
                               for row in episodes)):
                raise ValueError("invalid confirmation switching episodes")
            if kind == "full_cycle":
                for episode in episodes:
                    if any(int(episode["physics_mode_counts"][str(mode)]) <= 0
                           for mode in range(4)):
                        raise ValueError("confirmation did not visit every mode")


def primary_endpoints():
    return {
        "stationary_noninferiority_margin":
            protocol.STATIONARY_NONINFERIORITY_MARGIN,
        "termination_rate_margin": protocol.TERMINATION_RATE_MARGIN,
        "min_oracle_recovery": protocol.MIN_ORACLE_RECOVERY,
        "min_full_cycle_wins": protocol.MIN_FULL_CYCLE_WINS,
        "min_committed_route_accuracy":
            protocol.MIN_COMMITTED_ROUTE_ACCURACY,
    }


def run(event_seed: int) -> Path:
    protocol.configure()
    result_path = protocol.group_path(event_seed)
    output = result_path.parent
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        validate_group(payload)
        print(f"Complete CUSUM confirmation exists: {output}")
        return result_path

    table = protocol.utility.load_utility_table()
    estimator_manifest = estimator.load_manifest()
    oracle_map = tuple(int(value) for value in table["oracle_controller_map"])
    bundles = (
        protocol.utility.control.specialist_protocol.validate_family_bundles(
            protocol.FAMILY))
    bundle_root = (
        protocol.utility.control.specialist_protocol.family_bundle_root(
            protocol.FAMILY))
    bundle_hashes = {
        name: estimator.sha256_file(
            bundle_root / name
            / protocol.utility.control.specialist_protocol.BUNDLE_MANIFEST)
        for name in bundles
    }
    if bundle_hashes != table["bundle_manifest_sha256"]:
        raise RuntimeError("controller bank changed after utility freeze")

    source_before = (
        protocol.utility.control.fork_protocol.current_source_manifest())
    config, agents, policy_states = specialist_audit._controller_policy_states(
        protocol.FAMILY)
    action_fn = specialist_audit._base_action_fn(
        nnx.graphdef(agents["robust"].policy))
    base_router_config = estimator.RouterConfig.from_dict(
        estimator_manifest["router_config"])
    router_config = protocol.utility.decision_config(
        base_router_config, protocol.DECISION_VARIANT)
    model, router_config, observe = estimator_audit.load_router(
        estimator_manifest, agents["robust"].obs_dim,
        agents["robust"].act_dim,
        router_config_override=router_config)
    stationary = base_audit.evaluate_stationary(
        config, policy_states, action_fn, model, observe, table,
        router_config, oracle_map, event_seed)
    switching = {
        kind: base_audit.evaluate_switching_kind(
            config, policy_states, action_fn, model, observe, table,
            router_config, oracle_map, event_seed, kind)
        for kind in ("slow_pair", "full_cycle")
    }
    source_after = (
        protocol.utility.control.fork_protocol.current_source_manifest())
    if source_after["sha256"] != source_before["sha256"]:
        raise RuntimeError("source changed during CUSUM confirmation")

    payload = {
        "schema": protocol.GROUP_SCHEMA,
        "status": "complete",
        "family": protocol.FAMILY,
        "env": protocol.ENV,
        "training_seed": 0,
        "event_seed": int(event_seed),
        "event_seed_role": "fresh untouched CUSUM return confirmation",
        "decision_variant": protocol.DECISION_VARIANT,
        "primary_endpoints": primary_endpoints(),
        "oracle_controller_map": list(oracle_map),
        "utility_table_file": protocol.file_record(protocol.utility.TABLE_PATH),
        "estimator_manifest_file": protocol.file_record(
            estimator.MANIFEST_PATH),
        "estimator_parameter_file": protocol.file_record(
            estimator.MODEL_PATH),
        "bundle_manifest_sha256": bundle_hashes,
        "source_snapshot_sha256": source_before["sha256"],
        "source_files": source_records(),
        "router_config": router_config.to_dict(),
        "episodes_per_task": base_audit.EPISODES_PER_TASK,
        "switching_episodes": base_audit.SWITCHING_EPISODES,
        "slow_dwell_steps": base_audit.SLOW_DWELL_STEPS,
        "full_cycle_dwell_steps": base_audit.FULL_CYCLE_DWELL_STEPS,
        "stationary": stationary,
        "switching": switching,
    }
    validate_group(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.tmp.", dir=output.parent))
    try:
        protocol.write_json_atomic(temporary / "group.json", payload)
        if output.exists():
            shutil.rmtree(output)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"CUSUM CONFIRMATION COMPLETE: {output}", flush=True)
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.event_seed)


if __name__ == "__main__":
    main()
