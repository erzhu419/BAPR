"""Train a fresh robust SAC source and publish compact controller state."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from flax import nnx

from jax_experiments.analysis import (
    regime_polarity_specialist_stability_v19 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_robust_source_v12 as base,
)
from jax_experiments.common.checkpoint import _to_numpy_tree


def _bind() -> None:
    base.protocol = protocol


def training_command(seed: int) -> list[str]:
    _bind()
    return base.training_command(seed)


def expected_config(seed: int) -> dict[str, Any]:
    _bind()
    return base.expected_config(seed)


def validate_signature(seed: int) -> dict[str, Any]:
    _bind()
    return base.validate_signature(seed)


def validate_bundle(seed: int) -> dict[str, Any]:
    seed = protocol.require_training_seed(seed)
    directory = protocol.source_bundle(seed)
    payload = protocol.read_json(protocol.source_manifest(seed))
    expected_files = {
        "policy/" + protocol.POLICY_NAME,
        "controller/" + protocol.CONTROLLER_STATE_NAME,
        "logs/protocol_signature.json",
    }
    if (
        payload.get("schema") != protocol.SOURCE_BUNDLE_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("identity") != protocol.source_identity(seed)
        or payload.get("checkpoint")
        != protocol.expected_source_checkpoint()
        or set(payload.get("files") or {}) != expected_files
    ):
        raise ValueError(f"invalid v19 robust source bundle: {directory}")
    for relative, record in payload["files"].items():
        path = directory / relative
        if not path.is_file() or protocol.file_record(path) != record:
            raise ValueError(f"changed v19 robust source file: {path}")
    with (
        directory / "controller" / protocol.CONTROLLER_STATE_NAME
    ).open("rb") as handle:
        state = pickle.load(handle)
    if (
        set(state) != {
            "critic", "target_critic", "log_alpha", "update_count"
        }
        or int(state["update_count"]) != protocol.SOURCE_UPDATE_COUNT
    ):
        raise ValueError("invalid v19 compact source controller state")
    return payload


def publish_bundle(seed: int) -> dict[str, Any]:
    seed = protocol.require_training_seed(seed)
    destination = protocol.source_bundle(seed)
    if protocol.source_manifest(seed).is_file():
        return validate_bundle(seed)
    _bind()
    run_dir = protocol.source_run_dir(seed)
    config, env, agent, next_iteration, total_steps = base._load_final_agent(seed)
    try:
        checkpoint = protocol.checkpoint_record(run_dir)
        if (
            checkpoint != protocol.expected_source_checkpoint()
            or next_iteration != protocol.SOURCE_NEXT_ITERATION
            or total_steps != protocol.SOURCE_TOTAL_STEPS
            or int(agent.update_count) != protocol.SOURCE_UPDATE_COUNT
            or int(config.stochastic_mode_fixed_id) != -1
        ):
            raise ValueError(f"incomplete v19 robust source: {run_dir}")
        signature = run_dir / "logs" / "protocol_signature.json"
        validate_signature(seed)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp.", dir=destination.parent))
        try:
            policy_path = temporary / "policy" / protocol.POLICY_NAME
            policy_path.parent.mkdir(parents=True, exist_ok=True)
            with policy_path.open("wb") as handle:
                pickle.dump(
                    _to_numpy_tree(nnx.state(agent.policy, nnx.Param)),
                    handle,
                )
            controller_path = (
                temporary / "controller" / protocol.CONTROLLER_STATE_NAME
            )
            controller_path.parent.mkdir(parents=True, exist_ok=True)
            with controller_path.open("wb") as handle:
                pickle.dump(
                    {
                        "critic": _to_numpy_tree(
                            nnx.state(agent.critic, nnx.Param)),
                        "target_critic": _to_numpy_tree(
                            nnx.state(agent.target_critic, nnx.Param)),
                        "log_alpha": _to_numpy_tree(agent.log_alpha),
                        "update_count": int(agent.update_count),
                    },
                    handle,
                )
            signature_target = temporary / "logs" / signature.name
            signature_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(signature, signature_target)
            files = {
                path.relative_to(temporary).as_posix():
                    protocol.file_record(path)
                for path in (policy_path, controller_path, signature_target)
            }
            protocol.write_json_atomic(
                temporary / "bundle_manifest.json",
                {
                    "schema": protocol.SOURCE_BUNDLE_SCHEMA,
                    "status": "complete",
                    "identity": protocol.source_identity(seed),
                    "checkpoint": checkpoint,
                    "files": files,
                    "contents": (
                        "policy, critic, target critic, alpha, and update "
                        "count; no optimizer or replay"
                    ),
                },
            )
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    finally:
        if hasattr(env, "close"):
            env.close()
    return validate_bundle(seed)


def run(seed: int) -> None:
    seed = protocol.require_training_seed(seed)
    protocol.validate_registration()
    if protocol.source_manifest(seed).is_file():
        validate_bundle(seed)
        print(f"V19 ROBUST SOURCE ALREADY COMPLETE: seed={seed}", flush=True)
        return
    command = training_command(seed)
    print("V19 ROBUST SOURCE TRAIN:", " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(protocol.ROOT)
    subprocess.run(command, cwd=protocol.ROOT, env=environment, check=True)
    payload = publish_bundle(seed)
    replay = protocol.source_run_dir(seed) / "checkpoints/replay_buffer.npz"
    if replay.is_file():
        replay.unlink()
    print(
        "V19 ROBUST SOURCE COMPLETE: "
        + json.dumps(payload["identity"], sort_keys=True),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for checkpoint-safe execution")
    run(args.seed)


if __name__ == "__main__":
    main()
