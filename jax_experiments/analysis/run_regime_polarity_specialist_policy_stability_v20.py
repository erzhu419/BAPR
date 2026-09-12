"""Train one v20 full-state specialist stability arm."""
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

import jax
import jax.numpy as jnp
from flax import nnx

from jax_experiments.analysis import final_task_sweep
from jax_experiments.analysis import (
    regime_polarity_specialist_policy_stability_v20 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_robust_source_v20 as source_runner,
)
from jax_experiments.analysis import (
    run_regime_polarity_robust_warmstart_specialist_v11 as historical,
)
from jax_experiments.algos.sac_policy_stability import SACPolicyStability
from jax_experiments.common.checkpoint import (
    _patch_flax_variablestate_unpickle,
    _restore_tree_like,
    _to_numpy_tree,
    load_checkpoint,
    save_checkpoint,
)
from jax_experiments.common.logging import Logger
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.train import make_algo, make_env


def _restore_params(module, raw, name: str) -> None:
    restored = _restore_tree_like(
        nnx.state(module, nnx.Param),
        raw,
        name,
        allow_fallback=False,
    )
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
        _restore_params(agent.policy, policy_state, "v20 source policy")
        _restore_params(
            agent.critic, controller_state["critic"], "v20 source critic")
        _restore_params(
            agent.target_critic,
            controller_state["target_critic"],
            "v20 source target critic",
        )
        agent.log_alpha = jnp.asarray(controller_state["log_alpha"])
        agent.update_count = int(controller_state["update_count"])
        if agent.update_count != protocol.SOURCE_UPDATE_COUNT:
            raise ValueError("v20 source update count mismatch")
        return config, agent
    finally:
        if hasattr(env, "close"):
            env.close()


def _controller_equivalence(source_agent, target_agent) -> dict[str, Any]:
    observations = jax.random.normal(
        jax.random.PRNGKey(320_101), (64, source_agent.obs_dim))
    actions = jax.random.uniform(
        jax.random.PRNGKey(320_102),
        (64, source_agent.act_dim),
        minval=-1.0,
        maxval=1.0,
    )
    source_actions = source_agent.policy.deterministic(observations)
    target_actions = target_agent.policy.deterministic(observations)
    source_q = source_agent.critic(observations, actions)
    target_q = target_agent.critic(observations, actions)
    source_target_q = source_agent.target_critic(observations, actions)
    target_target_q = target_agent.target_critic(observations, actions)
    errors = {
        "actor_max_abs_error": float(jnp.max(jnp.abs(
            source_actions - target_actions))),
        "critic_max_abs_error": float(jnp.max(jnp.abs(
            source_q - target_q))),
        "target_critic_max_abs_error": float(jnp.max(jnp.abs(
            source_target_q - target_target_q))),
        "log_alpha_abs_error": float(jnp.abs(
            source_agent.log_alpha - target_agent.log_alpha)),
    }
    return {
        "pass": bool(max(errors.values()) <= 1e-7),
        **errors,
        "atol": 1e-7,
    }


def _target_config(source_config, run_dir: Path, mode: int, variant: str):
    config = historical._target_config(source_config, run_dir, mode)
    config.sac_actor_update_period = protocol.actor_update_period(variant)
    config.sac_select_best_eval = protocol.select_best_validation(variant)
    return config


def _make_stability_agent(config, obs_dim: int, act_dim: int):
    return SACPolicyStability(
        obs_dim, act_dim, config, seed=config.seed)


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
            or payload.get("controller_equivalence", {}).get("pass")
            is not True
        ):
            raise ValueError(f"invalid existing v20 bootstrap: {manifest_path}")
        return payload
    if run_dir.exists():
        raise RuntimeError(
            f"partial v20 branch exists without bootstrap: {run_dir}")

    source_config, source_agent = _load_source(seed)
    config = _target_config(source_config, run_dir, mode, variant)
    env = make_env(config, seed_offset=0)
    try:
        agent = _make_stability_agent(config, env.obs_dim, env.act_dim)
        nnx.update(agent.policy, nnx.state(source_agent.policy, nnx.Param))
        nnx.update(agent.critic, nnx.state(source_agent.critic, nnx.Param))
        nnx.update(
            agent.target_critic,
            nnx.state(source_agent.target_critic, nnx.Param),
        )
        agent.log_alpha = jnp.asarray(source_agent.log_alpha)
        historical._reset_optimizers(agent)
        agent.update_count = protocol.SOURCE_UPDATE_COUNT
        agent.reset_selection_anchor()
        equivalence = _controller_equivalence(source_agent, agent)
        if not equivalence["pass"]:
            raise RuntimeError(
                f"v20 full-state warm-start mismatch: {equivalence}")

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
            "controller_equivalence": equivalence,
            "replay_reset": True,
            "optimizer_reset": True,
            "controller_initialization": [
                "actor", "critic", "target_critic", "alpha"
            ],
            "actor_update_period": protocol.actor_update_period(variant),
            "policy_selection": (
                "best_validation"
                if protocol.select_best_validation(variant) else "final"
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


def training_command(
    variant: str, seed: int, mode: int,
    run_dir: Path | None = None,
) -> list[str]:
    variant = protocol.require_variant(variant)
    run_dir = run_dir or protocol.run_dir(variant, seed, mode)
    historical.protocol = protocol
    command = historical.training_command(variant, seed, mode, run_dir)
    module_index = command.index("-m") + 1
    command[module_index] = (
        "jax_experiments.analysis."
        "train_regime_polarity_sac_policy_stability_v20"
    )
    return command


def expected_config(variant: str, seed: int, mode: int) -> dict[str, Any]:
    variant = protocol.require_variant(variant)
    values = source_runner.expected_config(seed)
    values.update({
        "max_iters": protocol.FINAL_NEXT_ITERATION,
        "stochastic_mode_fixed_id": protocol.require_mode(mode),
        "start_train_steps": 0,
        "sac_actor_update_period": protocol.actor_update_period(variant),
        "sac_select_best_eval": protocol.select_best_validation(variant),
    })
    return values


def _signature_path(variant: str, seed: int, mode: int) -> Path:
    bundle_path = (
        protocol.bundle_dir(variant, seed, mode)
        / "logs" / "protocol_signature.json"
    )
    if bundle_path.is_file():
        return bundle_path
    return (
        protocol.run_dir(variant, seed, mode)
        / "logs" / "protocol_signature.json"
    )


def validate_signature(variant: str, seed: int, mode: int) -> dict[str, Any]:
    variant = protocol.require_variant(variant)
    seed = protocol.require_training_seed(seed)
    mode = protocol.require_mode(mode)
    signature = protocol.read_json(_signature_path(variant, seed, mode))
    config = signature.get("config") or {}
    mismatches = {
        key: {"actual": config.get(key), "expected": expected}
        for key, expected in expected_config(variant, seed, mode).items()
        if config.get(key) != expected
    }
    for key, expected in {
        "checkpoint_loaded": True,
        "start_iteration": protocol.SOURCE_NEXT_ITERATION,
        "total_steps_at_start": protocol.SOURCE_TOTAL_STEPS,
    }.items():
        if signature.get(key) != expected:
            mismatches[key] = {
                "actual": signature.get(key), "expected": expected}
    if mismatches:
        raise ValueError(f"v20 specialist config mismatch: {mismatches}")
    return signature


def _load_final_agent(variant: str, seed: int, mode: int):
    run_dir = protocol.run_dir(variant, seed, mode)
    config = final_task_sweep.load_config(run_dir)
    config.sac_actor_update_period = protocol.actor_update_period(variant)
    config.sac_select_best_eval = protocol.select_best_validation(variant)
    env = make_env(config, seed_offset=0)
    agent = _make_stability_agent(config, env.obs_dim, env.act_dim)
    replay = ReplayBuffer(
        env.obs_dim,
        env.act_dim,
        capacity=1,
        belief_dim=getattr(agent, "belief_dim", 0),
    )
    with tempfile.TemporaryDirectory() as temporary:
        logger = Logger(temporary)
        next_iteration, total_steps = load_checkpoint(
            str(run_dir / "checkpoints"),
            agent,
            replay,
            logger,
            "sac",
            load_replay_buffer=False,
        )
    return config, env, agent, next_iteration, total_steps


def validate_bundle(variant: str, seed: int, mode: int) -> dict[str, Any]:
    variant = protocol.require_variant(variant)
    seed = protocol.require_training_seed(seed)
    mode = protocol.require_mode(mode)
    directory = protocol.bundle_dir(variant, seed, mode)
    payload = protocol.read_json(directory / "bundle_manifest.json")
    expected_files = {
        "policy/" + protocol.POLICY_NAME,
        "logs/protocol_signature.json",
        "provenance/" + protocol.BOOTSTRAP_NAME,
        "provenance/" + protocol.SELECTION_NAME,
    }
    if (
        payload.get("schema") != protocol.BUNDLE_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("identity")
        != protocol.identity(variant, seed, mode)
        or payload.get("checkpoint") != protocol.expected_checkpoint()
        or payload.get("source_bundle_manifest")
        != protocol.file_record(protocol.source_manifest(seed))
        or set(payload.get("files") or {}) != expected_files
    ):
        raise ValueError(f"invalid v20 policy bundle: {directory}")
    for relative, record in payload["files"].items():
        path = directory / relative
        if not path.is_file() or protocol.file_record(path) != record:
            raise ValueError(f"changed v20 policy bundle file: {path}")
    selection = protocol.read_json(
        directory / "provenance" / protocol.SELECTION_NAME)
    if (
        selection.get("selection")
        != ("best_validation" if protocol.select_best_validation(variant)
            else "final")
        or int(selection.get("actor_update_period", 0))
        != protocol.actor_update_period(variant)
        or int(selection.get("final_update_count", -1))
        != protocol.FINAL_UPDATE_COUNT
        or int(selection.get("validation_observations", 0)) <= 0
    ):
        raise ValueError("invalid v20 policy selection record")
    if protocol.select_best_validation(variant):
        selected = int(selection.get("selected_update_count", -1))
        if not protocol.SOURCE_UPDATE_COUNT < selected <= protocol.FINAL_UPDATE_COUNT:
            raise ValueError("v20 selected update count is outside fine-tuning")
    validate_signature(variant, seed, mode)
    return payload


def publish_bundle(variant: str, seed: int, mode: int) -> dict[str, Any]:
    variant = protocol.require_variant(variant)
    seed = protocol.require_training_seed(seed)
    mode = protocol.require_mode(mode)
    destination = protocol.bundle_dir(variant, seed, mode)
    if protocol.bundle_manifest(variant, seed, mode).is_file():
        return validate_bundle(variant, seed, mode)
    run_dir = protocol.run_dir(variant, seed, mode)
    config, env, agent, next_iteration, total_steps = _load_final_agent(
        variant, seed, mode)
    try:
        checkpoint = protocol.checkpoint_record(run_dir)
        if (
            checkpoint != protocol.expected_checkpoint()
            or next_iteration != protocol.FINAL_NEXT_ITERATION
            or total_steps != protocol.FINAL_TOTAL_STEPS
            or int(agent.update_count) != protocol.FINAL_UPDATE_COUNT
            or int(config.stochastic_mode_fixed_id) != mode
        ):
            raise ValueError(f"incomplete v20 training output: {run_dir}")
        signature = run_dir / "logs" / "protocol_signature.json"
        bootstrap = run_dir / "checkpoints" / protocol.BOOTSTRAP_NAME
        validate_signature(variant, seed, mode)

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp.", dir=destination.parent))
        try:
            policy_path = temporary / "policy" / protocol.POLICY_NAME
            policy_path.parent.mkdir(parents=True, exist_ok=True)
            with policy_path.open("wb") as handle:
                pickle.dump(
                    _to_numpy_tree(agent.selected_policy_state()), handle)
            signature_target = temporary / "logs" / signature.name
            signature_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(signature, signature_target)
            bootstrap_target = temporary / "provenance" / bootstrap.name
            bootstrap_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(bootstrap, bootstrap_target)
            selection_path = (
                temporary / "provenance" / protocol.SELECTION_NAME)
            protocol.write_json_atomic(
                selection_path, agent.selection_record())
            files = {
                path.relative_to(temporary).as_posix():
                    protocol.file_record(path)
                for path in (
                    policy_path,
                    signature_target,
                    bootstrap_target,
                    selection_path,
                )
            }
            protocol.write_json_atomic(
                temporary / "bundle_manifest.json",
                {
                    "schema": protocol.BUNDLE_SCHEMA,
                    "status": "complete",
                    "identity": protocol.identity(variant, seed, mode),
                    "checkpoint": checkpoint,
                    "source_bundle_manifest": protocol.file_record(
                        protocol.source_manifest(seed)),
                    "files": files,
                    "contents": (
                        "selected policy parameters and provenance only; "
                        "no critic, optimizer, or replay"
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
    return validate_bundle(variant, seed, mode)


def run(variant: str, seed: int, mode: int) -> None:
    variant = protocol.require_variant(variant)
    seed = protocol.require_training_seed(seed)
    mode = protocol.require_mode(mode)
    protocol.validate_registration()
    if protocol.bundle_manifest(variant, seed, mode).is_file():
        validate_bundle(variant, seed, mode)
        print(
            "V20 POLICY-STABILITY CONTROLLER ALREADY COMPLETE: "
            f"variant={variant} seed={seed} mode={mode}",
            flush=True,
        )
        return
    run_dir = protocol.run_dir(variant, seed, mode)
    _bootstrap(variant, seed, mode, run_dir)
    command = training_command(variant, seed, mode, run_dir)
    print("V20 POLICY-STABILITY TRAIN:", " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(protocol.ROOT)
    environment["BAPR_SAC_ACTOR_UPDATE_PERIOD"] = str(
        protocol.actor_update_period(variant))
    environment["BAPR_SAC_SELECT_BEST_EVAL"] = (
        "1" if protocol.select_best_validation(variant) else "0")
    subprocess.run(
        command,
        cwd=protocol.ROOT,
        env=environment,
        check=True,
    )
    payload = publish_bundle(variant, seed, mode)
    replay = run_dir / "checkpoints" / "replay_buffer.npz"
    if replay.is_file():
        replay.unlink()
    print(
        "V20 POLICY-STABILITY CONTROLLER COMPLETE: "
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
