"""Audit a calibration-frozen many-to-one physics-mode controller map."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from flax import nnx

from jax_experiments.analysis import bapr_v3_control_equivalence as protocol
from jax_experiments.analysis import (
    run_bapr_v3_independent_specialist_audit as specialist_audit,
)
from jax_experiments.train import make_env


CONTROLLER = "dynamic_control_oracle"
CONTROLLERS = (*specialist_audit.CONTROLLERS, CONTROLLER)


def evaluate_mapped_switching(
    config, graphdef, policy_states: dict[str, Any], event_seed: int,
    controller_map: tuple[int, ...],
) -> dict[str, Any]:
    template = make_env(config, seed_offset=event_seed)
    tasks = template.sample_tasks(4)
    if hasattr(template, "close"):
        template.close()
    action_fn = specialist_audit._base_action_fn(graphdef)
    return specialist_audit.evaluate_switching_controller(
        config,
        tasks,
        policy_states,
        action_fn,
        CONTROLLER,
        event_seed,
        controller_map=controller_map,
    )


def validate_group(payload: dict[str, Any]) -> None:
    if (payload.get("schema") != protocol.GROUP_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("family") != protocol.FAMILY
            or payload.get("env") != protocol.ENV
            or int(payload.get("event_seed", -1))
            not in protocol.HOLDOUT_EVENT_SEEDS):
        raise ValueError("invalid control-equivalence audit identity")
    mapping = tuple(int(value) for value in payload.get("controller_map", []))
    if len(mapping) != 4:
        raise ValueError("control-equivalence map has wrong cardinality")
    if set(payload.get("stationary") or {}) != set(CONTROLLERS):
        raise ValueError("stationary controller set is incomplete")
    if set(payload.get("switching") or {}) != set(CONTROLLERS):
        raise ValueError("switching controller set is incomplete")
    for controller in CONTROLLERS:
        stationary = payload["stationary"][controller]
        if set(stationary) != {"0", "1", "2", "3"}:
            raise ValueError(f"stationary rows incomplete for {controller}")
        for record in stationary.values():
            returns = record.get("returns") or []
            if (len(returns) != specialist_audit.EPISODES_PER_TASK
                    or not all(math.isfinite(float(value))
                               for value in returns)):
                raise ValueError("stationary return record is invalid")
        episodes = payload["switching"][controller].get("episodes") or []
        if (len(episodes) != specialist_audit.SWITCHING_EPISODES
                or not all(math.isfinite(float(row["return"]))
                           for row in episodes)):
            raise ValueError("switching return record is invalid")
    for physics_mode, specialist_mode in enumerate(mapping):
        if (payload["stationary"][CONTROLLER][str(physics_mode)]
                != payload["stationary"][
                    f"fixed_mode_{specialist_mode}"][str(physics_mode)]):
            raise ValueError("mapped stationary row is not exact")
    for row in payload["switching"][CONTROLLER]["trace_summary"]:
        if row.get("selection_alignment") != 1.0:
            raise ValueError("mapped controller did not follow frozen map")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event-seed",
        type=int,
        choices=protocol.HOLDOUT_EVENT_SEEDS,
        required=True,
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol.configure()
    output = args.out_dir.resolve()
    result_path = output / "group.json"
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        validate_group(payload)
        print(f"Complete valid control-equivalence audit exists: {output}")
        return

    controller_map, mapping_payload = protocol.load_controller_map()
    bundles = protocol.specialist_protocol.validate_family_bundles(
        protocol.FAMILY)
    source_before = protocol.fork_protocol.current_source_manifest()
    config, agents, policy_states = specialist_audit._controller_policy_states(
        protocol.FAMILY)
    graphdef = nnx.graphdef(agents["robust"].policy)
    stationary = specialist_audit.evaluate_stationary(
        config, graphdef, policy_states, args.event_seed)
    stationary[CONTROLLER] = {
        str(physics_mode): stationary[
            f"fixed_mode_{controller_map[physics_mode]}"][str(physics_mode)]
        for physics_mode in protocol.specialist_protocol.MODES
    }
    switching = specialist_audit.evaluate_switching(
        config, graphdef, policy_states, args.event_seed)
    switching[CONTROLLER] = evaluate_mapped_switching(
        config, graphdef, policy_states, args.event_seed, controller_map)
    source_after = protocol.fork_protocol.current_source_manifest()
    if source_after["sha256"] != source_before["sha256"]:
        raise RuntimeError("source changed during control-equivalence audit")

    payload = {
        "schema": protocol.GROUP_SCHEMA,
        "status": "complete",
        "family": protocol.FAMILY,
        "env": protocol.ENV,
        "training_seed": protocol.specialist_protocol.SEED,
        "event_seed": int(args.event_seed),
        "event_seed_role": (
            "untouched paired holdout stream selected after calibration map"
        ),
        "controller_map": list(controller_map),
        "controller_map_file": protocol.file_record(protocol.MAPPING_PATH),
        "calibration_event_seeds": mapping_payload[
            "calibration_event_seeds"],
        "source_snapshot_sha256": source_before["sha256"],
        "runner_sha256": protocol.fork_protocol.sha256_file(
            Path(__file__).resolve()),
        "bundle_manifest_sha256": {
            name: protocol.fork_protocol.sha256_file(
                protocol.specialist_protocol.family_bundle_root(
                    protocol.FAMILY) / name
                / protocol.specialist_protocol.BUNDLE_MANIFEST
            )
            for name in bundles
        },
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
    print(f"CONTROL-EQUIVALENCE AUDIT COMPLETE: {output}", flush=True)


if __name__ == "__main__":
    main()
