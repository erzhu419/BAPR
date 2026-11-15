"""Fine-tune one fixed-mode SAC specialist from a matched robust actor."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Any

from flax import nnx

from jax_experiments.analysis import final_task_sweep
from jax_experiments.analysis import (
    regime_polarity_robust_warmstart_confirmation_v12 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_robust_source_v12 as source_runner,
)
from jax_experiments.analysis import (
    run_regime_polarity_robust_warmstart_specialist_v11 as base,
)
from jax_experiments.common.checkpoint import (
    _patch_flax_variablestate_unpickle,
    _restore_tree_like,
    save_checkpoint,
)
from jax_experiments.common.logging import Logger
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.train import make_algo, make_env


BOOTSTRAP_SCHEMA = "bapr.robust-warmstart-specialist-bootstrap.v12"


def _bind() -> None:
    base.protocol = protocol


def _load_source(seed: int):
    seed = protocol.require_training_seed(seed)
    source_runner.validate_bundle(seed)
    directory = protocol.source_bundle(seed)
    config = final_task_sweep.load_config(directory)
    config.stochastic_mode_fixed_id = -1
    env = make_env(config, seed_offset=0)
    try:
        agent = make_algo("sac", env.obs_dim, env.act_dim, config)
        template = nnx.state(agent.policy, nnx.Param)
        _patch_flax_variablestate_unpickle()
        with (directory / "policy" / protocol.POLICY_NAME).open("rb") as handle:
            raw = pickle.load(handle)
        params = _restore_tree_like(
            template,
            raw,
            "v12 robust source policy parameters",
            allow_fallback=False,
        )
        nnx.update(agent.policy, params)
        agent.update_count = protocol.SOURCE_UPDATE_COUNT
        return config, agent
    finally:
        if hasattr(env, "close"):
            env.close()


def _bootstrap(
    seed: int,
    mode: int,
    run_dir: Path | None = None,
    require_registration: bool = True,
) -> dict[str, Any]:
    variant = "actor_only"
    seed = protocol.require_training_seed(seed)
    mode = protocol.require_mode(mode)
    run_dir = run_dir or protocol.run_dir(variant, seed, mode)
    manifest_path = run_dir / "checkpoints" / protocol.BOOTSTRAP_NAME
    expected_identity = protocol.identity(variant, seed, mode)
    if manifest_path.is_file():
        payload = protocol.read_json(manifest_path)
        if (
            payload.get("schema") != BOOTSTRAP_SCHEMA
            or payload.get("identity") != expected_identity
            or payload.get("actor_equivalence", {}).get("pass") is not True
        ):
            raise ValueError(f"invalid existing v12 bootstrap: {manifest_path}")
        return payload
    if run_dir.exists():
        raise RuntimeError(
            f"partial v12 branch exists without bootstrap: {run_dir}")

    _bind()
    source_config, source_agent = _load_source(seed)
    config = base._target_config(source_config, run_dir, mode)
    env = make_env(config, seed_offset=0)
    try:
        agent = make_algo("sac", env.obs_dim, env.act_dim, config)
        nnx.update(agent.policy, nnx.state(source_agent.policy, nnx.Param))
        agent.update_count = protocol.SOURCE_UPDATE_COUNT
        equivalence = base._actor_equivalence(source_agent, agent)
        if not equivalence["pass"]:
            raise RuntimeError(
                f"v12 robust actor warm-start mismatch: {equivalence}")

        run_dir.mkdir(parents=True, exist_ok=False)
        replay = ReplayBuffer(
            env.obs_dim,
            env.act_dim,
            capacity=config.replay_size,
            belief_dim=getattr(agent, "belief_dim", 0),
        )
        logger = Logger(str(run_dir / "logs"))
        save_checkpoint(
            str(run_dir / "checkpoints"),
            agent,
            replay,
            logger,
            iteration=protocol.SOURCE_ITERATION,
            total_steps=protocol.SOURCE_TOTAL_STEPS,
            algo="sac",
        )
        payload = {
            "schema": BOOTSTRAP_SCHEMA,
            "status": "complete",
            "identity": expected_identity,
            "registration": (
                protocol.file_record(protocol.REGISTRATION_PATH)
                if require_registration else {"smoke_only": True}
            ),
            "source_bundle_manifest": protocol.file_record(
                protocol.source_manifest(seed)),
            "source_checkpoint": protocol.expected_source_checkpoint(),
            "fork_checkpoint": protocol.expected_source_checkpoint(),
            "actor_equivalence": equivalence,
            "replay_reset": True,
            "optimizer_reset": True,
            "controller_initialization": ["actor"],
        }
        protocol.write_json_atomic(manifest_path, payload)
        return payload
    except Exception:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    finally:
        if hasattr(env, "close"):
            env.close()


def validate_bundle(seed: int, mode: int) -> dict[str, Any]:
    _bind()
    return base.validate_bundle("actor_only", seed, mode)


def training_command(seed: int, mode: int) -> list[str]:
    _bind()
    return base.training_command("actor_only", seed, mode)


def publish_bundle(seed: int, mode: int) -> dict[str, Any]:
    _bind()
    return base.publish_bundle("actor_only", seed, mode)


def run(seed: int, mode: int) -> None:
    seed = protocol.require_training_seed(seed)
    mode = protocol.require_mode(mode)
    protocol.validate_registration()
    if protocol.bundle_manifest("actor_only", seed, mode).is_file():
        validate_bundle(seed, mode)
        print(
            f"V12 WARMSTART SPECIALIST ALREADY COMPLETE: seed={seed} mode={mode}",
            flush=True,
        )
        return
    _bootstrap(seed, mode)
    command = training_command(seed, mode)
    print("V12 WARMSTART SPECIALIST TRAIN:", " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(protocol.ROOT)
    subprocess.run(
        command,
        cwd=protocol.ROOT,
        env=environment,
        check=True,
    )
    payload = publish_bundle(seed, mode)
    print(
        "V12 WARMSTART SPECIALIST COMPLETE: "
        + json.dumps(payload["identity"], sort_keys=True),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--mode", choices=protocol.MODES, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for checkpoint-safe execution")
    run(args.seed, args.mode)


if __name__ == "__main__":
    main()
