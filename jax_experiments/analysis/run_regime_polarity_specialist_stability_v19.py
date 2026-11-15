"""Train one v19 specialist initialization arm on a fixed mode."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from jax_experiments.analysis import final_task_sweep
from jax_experiments.analysis import (
    regime_polarity_specialist_stability_v19 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_robust_source_v19 as source_runner,
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


def _bind() -> None:
    base.protocol = protocol


def _restore_params(module, raw, name: str) -> None:
    template = nnx.state(module, nnx.Param)
    restored = _restore_tree_like(
        template, raw, name, allow_fallback=False)
    nnx.update(module, restored)


def _load_source(seed: int):
    seed = protocol.require_training_seed(seed)
    source_runner.validate_bundle(seed)
    directory = protocol.source_bundle(seed)
    config = final_task_sweep.load_config(directory)
    config.stochastic_mode_fixed_id = -1
    env = make_env(config, seed_offset=0)
    try:
        agent = make_algo("sac", env.obs_dim, env.act_dim, config)
        _patch_flax_variablestate_unpickle()
        with (directory / "policy" / protocol.POLICY_NAME).open("rb") as handle:
            policy_state = pickle.load(handle)
        with (
            directory / "controller" / protocol.CONTROLLER_STATE_NAME
        ).open("rb") as handle:
            controller_state = pickle.load(handle)
        _restore_params(agent.policy, policy_state, "v19 source policy")
        _restore_params(
            agent.critic, controller_state["critic"], "v19 source critic")
        _restore_params(
            agent.target_critic,
            controller_state["target_critic"],
            "v19 source target critic",
        )
        agent.log_alpha = jnp.asarray(controller_state["log_alpha"])
        agent.update_count = int(controller_state["update_count"])
        if agent.update_count != protocol.SOURCE_UPDATE_COUNT:
            raise ValueError("v19 source update count mismatch")
        return config, agent
    finally:
        if hasattr(env, "close"):
            env.close()


def _critic_equivalence(source_agent, target_agent) -> dict[str, Any]:
    observations = jax.random.normal(
        jax.random.PRNGKey(319_101), (64, source_agent.obs_dim))
    actions = jax.random.uniform(
        jax.random.PRNGKey(319_102),
        (64, source_agent.act_dim),
        minval=-1.0,
        maxval=1.0,
    )
    source_q = source_agent.critic(observations, actions)
    target_q = target_agent.critic(observations, actions)
    source_target_q = source_agent.target_critic(observations, actions)
    target_target_q = target_agent.target_critic(observations, actions)
    critic_error = float(jnp.max(jnp.abs(source_q - target_q)))
    target_error = float(jnp.max(
        jnp.abs(source_target_q - target_target_q)))
    alpha_error = float(jnp.abs(
        source_agent.log_alpha - target_agent.log_alpha))
    return {
        "pass": bool(
            critic_error <= 1e-7
            and target_error <= 1e-7
            and alpha_error <= 1e-7
        ),
        "critic_max_abs_error": critic_error,
        "target_critic_max_abs_error": target_error,
        "log_alpha_abs_error": alpha_error,
        "atol": 1e-7,
    }


def _bootstrap(
    variant: str,
    seed: int,
    mode: int,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    variant = protocol.require_variant(variant)
    seed = protocol.require_training_seed(seed)
    mode = protocol.require_mode(mode)
    run_dir = run_dir or protocol.run_dir(variant, seed, mode)
    manifest_path = run_dir / "checkpoints" / protocol.BOOTSTRAP_NAME
    expected_identity = protocol.identity(variant, seed, mode)
    if manifest_path.is_file():
        payload = protocol.read_json(manifest_path)
        if (
            payload.get("schema") != protocol.BOOTSTRAP_SCHEMA
            or payload.get("identity") != expected_identity
            or payload.get("actor_equivalence", {}).get("pass") is not True
        ):
            raise ValueError(f"invalid existing v19 bootstrap: {manifest_path}")
        if (
            variant == "full_state"
            and payload.get("controller_equivalence", {}).get("pass")
            is not True
        ):
            raise ValueError("invalid v19 full-state bootstrap")
        return payload
    if run_dir.exists():
        raise RuntimeError(
            f"partial v19 branch exists without bootstrap: {run_dir}")

    _bind()
    source_config, source_agent = _load_source(seed)
    config = base._target_config(source_config, run_dir, mode)
    env = make_env(config, seed_offset=0)
    try:
        agent = make_algo("sac", env.obs_dim, env.act_dim, config)
        nnx.update(agent.policy, nnx.state(source_agent.policy, nnx.Param))
        controller_equivalence = None
        if variant == "full_state":
            nnx.update(
                agent.critic, nnx.state(source_agent.critic, nnx.Param))
            nnx.update(
                agent.target_critic,
                nnx.state(source_agent.target_critic, nnx.Param),
            )
            agent.log_alpha = jnp.asarray(source_agent.log_alpha)
            base._reset_optimizers(agent)
            controller_equivalence = _critic_equivalence(
                source_agent, agent)
            if not controller_equivalence["pass"]:
                raise RuntimeError(
                    "v19 full-state warm-start mismatch: "
                    f"{controller_equivalence}")
        agent.update_count = protocol.SOURCE_UPDATE_COUNT
        actor_equivalence = base._actor_equivalence(source_agent, agent)
        if not actor_equivalence["pass"]:
            raise RuntimeError(
                f"v19 robust actor warm-start mismatch: {actor_equivalence}")

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
            "schema": protocol.BOOTSTRAP_SCHEMA,
            "status": "complete",
            "identity": expected_identity,
            "registration": protocol.file_record(protocol.REGISTRATION_PATH),
            "source_bundle_manifest": protocol.file_record(
                protocol.source_manifest(seed)),
            "source_checkpoint": protocol.expected_source_checkpoint(),
            "fork_checkpoint": protocol.expected_source_checkpoint(),
            "actor_equivalence": actor_equivalence,
            "controller_equivalence": controller_equivalence,
            "replay_reset": True,
            "optimizer_reset": True,
            "controller_initialization": (
                ["actor", "critic", "target_critic", "alpha"]
                if variant == "full_state" else ["actor"]
            ),
            "actor_update_after": (
                protocol.ACTOR_UPDATE_AFTER
                if variant == "critic_warmup" else None
            ),
        }
        protocol.write_json_atomic(manifest_path, payload)
        return payload
    except Exception:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    finally:
        if hasattr(env, "close"):
            env.close()


def training_command(variant: str, seed: int, mode: int) -> list[str]:
    variant = protocol.require_variant(variant)
    _bind()
    command = base.training_command(variant, seed, mode)
    if variant == "critic_warmup":
        module_index = command.index("-m") + 1
        command[module_index] = (
            "jax_experiments.analysis."
            "train_regime_polarity_sac_actor_delay_v19"
        )
    return command


def expected_config(variant: str, seed: int, mode: int) -> dict[str, Any]:
    variant = protocol.require_variant(variant)
    values = source_runner.expected_config(seed)
    values.update({
        "max_iters": protocol.FINAL_NEXT_ITERATION,
        "stochastic_mode_fixed_id": protocol.require_mode(mode),
        "start_train_steps": 0,
    })
    if variant == "critic_warmup":
        values["sac_actor_update_after"] = protocol.ACTOR_UPDATE_AFTER
    return values


def validate_signature(
    variant: str, seed: int, mode: int,
) -> dict[str, Any]:
    variant = protocol.require_variant(variant)
    seed = protocol.require_training_seed(seed)
    mode = protocol.require_mode(mode)
    path = protocol.run_dir(variant, seed, mode) / "logs/protocol_signature.json"
    signature = protocol.read_json(path)
    config = signature.get("config") or {}
    mismatches = {
        key: {"actual": config.get(key), "expected": expected}
        for key, expected in expected_config(variant, seed, mode).items()
        if config.get(key) != expected
    }
    if (
        variant != "critic_warmup"
        and config.get("sac_actor_update_after") is not None
    ):
        mismatches["sac_actor_update_after"] = {
            "actual": config.get("sac_actor_update_after"),
            "expected": None,
        }
    expected_start = {
        "checkpoint_loaded": True,
        "start_iteration": protocol.SOURCE_NEXT_ITERATION,
        "total_steps_at_start": protocol.SOURCE_TOTAL_STEPS,
    }
    for key, expected in expected_start.items():
        if signature.get(key) != expected:
            mismatches[key] = {
                "actual": signature.get(key), "expected": expected}
    if mismatches:
        raise ValueError(f"v19 specialist config mismatch: {mismatches}")
    return signature


def validate_bundle(
    variant: str, seed: int, mode: int,
) -> dict[str, Any]:
    _bind()
    payload = base.validate_bundle(variant, seed, mode)
    validate_signature(variant, seed, mode)
    return payload


def publish_bundle(
    variant: str, seed: int, mode: int,
) -> dict[str, Any]:
    _bind()
    payload = base.publish_bundle(variant, seed, mode)
    validate_signature(variant, seed, mode)
    return payload


def run(variant: str, seed: int, mode: int) -> None:
    variant = protocol.require_variant(variant)
    seed = protocol.require_training_seed(seed)
    mode = protocol.require_mode(mode)
    protocol.validate_registration()
    if protocol.bundle_manifest(variant, seed, mode).is_file():
        validate_bundle(variant, seed, mode)
        print(
            "V19 SPECIALIST ALREADY COMPLETE: "
            f"variant={variant} seed={seed} mode={mode}",
            flush=True,
        )
        return
    _bootstrap(variant, seed, mode)
    command = training_command(variant, seed, mode)
    print("V19 SPECIALIST TRAIN:", " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(protocol.ROOT)
    if variant == "critic_warmup":
        environment["BAPR_SAC_ACTOR_UPDATE_AFTER"] = str(
            protocol.ACTOR_UPDATE_AFTER)
    subprocess.run(
        command,
        cwd=protocol.ROOT,
        env=environment,
        check=True,
    )
    payload = publish_bundle(variant, seed, mode)
    replay = protocol.run_dir(
        variant, seed, mode) / "checkpoints/replay_buffer.npz"
    if replay.is_file():
        replay.unlink()
    print(
        "V19 SPECIALIST COMPLETE: "
        + json.dumps(payload["identity"], sort_keys=True),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=protocol.VARIANTS, required=True)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--mode", choices=protocol.MODES, type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for checkpoint-safe execution")
    run(args.variant, args.seed, args.mode)


if __name__ == "__main__":
    main()
