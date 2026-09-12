"""Prepare one canonical late-base adapter checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np

from jax_experiments.analysis import final_task_sweep
from jax_experiments.analysis import regime_adapter_fork as source_protocol
from jax_experiments.analysis import regime_adapter_latebase_min as protocol
from jax_experiments.analysis import run_regime_adapter_branch as source_runner
from jax_experiments.common.checkpoint import load_checkpoint, save_checkpoint
from jax_experiments.common.logging import Logger
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.train import make_algo, make_env


REQUIRED_FILES = {
    "checkpoints/params.pkl",
    "checkpoints/train_state.pkl",
    "checkpoints/replay_buffer.npz",
    "checkpoints/" + protocol.CANONICAL_BOOTSTRAP_NAME,
}


def _array_record(value) -> dict:
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "value": float(array.reshape(())),
    }


def _source_manifest(seed: int) -> Path:
    return (
        protocol.source_bundle_dir(seed)
        / source_protocol.BUNDLE_MANIFEST_NAME)


def _load_source(seed: int):
    seed = protocol.require_seed(seed)
    source_runner.validate_published_bundle(seed, "robust_continue")
    directory = protocol.source_bundle_dir(seed)
    config = final_task_sweep.load_config(directory)
    config.stochastic_mode_fixed_id = -1
    env = make_env(config, seed_offset=0)
    tasks = env.sample_tasks(config.task_num)
    agent = make_algo(config.algo, env.obs_dim, env.act_dim, config)
    agent.set_task_metadata(tasks)
    replay = ReplayBuffer(
        env.obs_dim, env.act_dim, capacity=1,
        belief_dim=agent.belief_dim)
    temporary = tempfile.TemporaryDirectory()
    logger = Logger(temporary.name)
    next_iteration, total_steps = load_checkpoint(
        str(directory / "checkpoints"), agent, replay, logger,
        config.algo, load_replay_buffer=False)
    if (next_iteration != protocol.SOURCE_NEXT_ITERATION
            or total_steps != protocol.SOURCE_TOTAL_STEPS
            or int(agent.update_count) != protocol.SOURCE_UPDATE_COUNT):
        temporary.cleanup()
        if hasattr(env, "close"):
            env.close()
        raise ValueError("late-base source checkpoint has the wrong budget")
    return config, env, tasks, agent, logger, temporary


def adapter_config(source_config, output: Path):
    config = deepcopy(source_config)
    config.algo = "bapr_regime"
    config.save_root = str(output.parent)
    config.run_name = output.name
    config.max_iters = protocol.FINAL_NEXT_ITERATION
    config.start_train_steps = 0
    config.stochastic_mode_fixed_id = -1
    config.bapr_v2_mode = "supervised"
    config.bapr_v2_latent_dim = len(protocol.MODES)
    config.bapr_v2_policy_context_source = "stored"
    config.bapr_v2_policy_mode = "residual"
    config.bapr_v2_training_schedule = "joint"
    config.bapr_v2_base_pretrain_iters = 1
    config.bapr_v2_teacher_iters = 0
    config.bapr_v2_student_iters = 0
    config.bapr_v2_context_hidden_dim = 128
    config.bapr_v2_context_length = 64
    config.bapr_v2_context_chunks = 8
    config.bapr_v2_context_burnin = 16
    config.bapr_v2_min_history = 16
    config.bapr_v2_switch_rollout_steps = protocol.DWELL_STEPS
    config.bapr_v2_residual_delta = protocol.DELTA
    config.bapr_v2_action_deviation_weight = 0.01
    config.bapr_v2_base_aux_weight = 0.0
    config.bapr_v2_context_dropout = 0.0
    config.bapr_v2_advantage_gate = False
    config.bapr_v2_actor_objective = "mean"
    config.bapr_v2_critic_target_mode = "min"
    config.bapr_v2_freeze_alpha = True
    config.bapr_v2_beta_ood = 0.0
    config.bapr_v2_reg_weight = 0.0
    config.bapr_v3_context_ensemble_size = 5
    config.bapr_v3_variance_model = "mode_empirical"
    config.bapr_v3_variance_ceiling = 0.5
    config.bapr_v3_variance_ema = 0.05
    config.bapr_regime_inference_iters = 1
    config.bapr_regime_adaptation_source = "oracle"
    config.bapr_regime_freeze_context_after_inference = True
    config.bapr_regime_clear_replay_on_adaptation = False
    config.bapr_regime_zero_residual_init = True
    config.bapr_regime_advantage_fallback = False
    config.context_warmup_iters = 0
    config.log_interval = 25
    config.eval_episodes = 3
    config.save_interval = 25
    config.resume = True
    return config


def validate_canonical(seed: int) -> dict:
    seed = protocol.require_seed(seed)
    directory = protocol.canonical_dir(seed)
    payload = protocol.read_json(protocol.canonical_manifest(seed))
    if (payload.get("schema") != protocol.CANONICAL_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("identity") != protocol.canonical_identity(seed)
            or payload.get("source_bundle_manifest")
            != protocol.file_record(_source_manifest(seed))
            or set(payload.get("files") or {}) != REQUIRED_FILES):
        raise ValueError(f"invalid late-base canonical bundle: {directory}")
    for relative, expected in payload["files"].items():
        path = directory / relative
        if not path.is_file() or protocol.file_record(path) != expected:
            raise ValueError(f"canonical file changed: {path}")
    checkpoint = payload.get("checkpoint") or {}
    if checkpoint != {
            "iteration": protocol.SOURCE_NEXT_ITERATION - 1,
            "next_iteration": protocol.SOURCE_NEXT_ITERATION,
            "total_steps": protocol.SOURCE_TOTAL_STEPS,
            "update_count": protocol.SOURCE_UPDATE_COUNT,
            "algo": "bapr_regime"}:
        raise ValueError("canonical checkpoint budget changed")
    return payload


def prepare(seed: int) -> dict:
    seed = protocol.require_seed(seed)
    destination = protocol.canonical_dir(seed)
    if protocol.canonical_manifest(seed).is_file():
        return validate_canonical(seed)

    (source_config, source_env, tasks, source_agent, source_logger,
     source_temporary) = _load_source(seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp.", dir=destination.parent))
    try:
        config = adapter_config(source_config, destination)
        agent = make_algo(
            config.algo, source_env.obs_dim, source_env.act_dim, config)
        agent.set_task_metadata(tasks)
        source_protocol.copy_source_controller_to_adapter(
            source_agent, agent)
        source_runner._reset_optimizers(agent)
        equivalence = source_runner._equivalence(source_agent, agent)
        if not equivalence["pass"]:
            raise RuntimeError(
                f"late-base source-to-adapter mismatch: {equivalence}")
        if not np.array_equal(
                np.asarray(agent.log_alpha),
                np.asarray(source_agent.log_alpha)):
            raise RuntimeError("late-base entropy temperature was not copied")

        replay = ReplayBuffer(
            source_env.obs_dim, source_env.act_dim,
            capacity=config.replay_size, belief_dim=agent.belief_dim)
        save_checkpoint(
            str(temporary / "checkpoints"), agent, replay, source_logger,
            iteration=protocol.SOURCE_NEXT_ITERATION - 1,
            total_steps=protocol.SOURCE_TOTAL_STEPS, algo=config.algo)
        bootstrap = {
            "schema": "bapr.regime-adapter-latebase-bootstrap.v1",
            "status": "complete",
            "identity": protocol.canonical_identity(seed),
            "source_bundle_manifest": protocol.file_record(
                _source_manifest(seed)),
            "empty_replay": True,
            "optimizer_states_reset": True,
            "function_equivalence": equivalence,
            "initial_components": {
                "source_policy": source_protocol.source_policy_sha256(
                    source_agent.policy),
                "frozen_base": source_protocol.base_policy_sha256(
                    agent.policy),
                "residual": source_protocol.residual_policy_sha256(
                    agent.policy),
                "critic": source_protocol.critic_sha256(agent.critic),
                "target_critic": source_protocol.critic_sha256(
                    agent.target_critic),
                "log_alpha": _array_record(agent.log_alpha),
            },
        }
        protocol.write_json_atomic(
            temporary / "checkpoints"
            / protocol.CANONICAL_BOOTSTRAP_NAME,
            bootstrap)

        records = {}
        for relative in sorted(REQUIRED_FILES):
            records[relative] = protocol.file_record(temporary / relative)
        payload = {
            "schema": protocol.CANONICAL_SCHEMA,
            "status": "complete",
            "identity": protocol.canonical_identity(seed),
            "source_bundle_manifest": protocol.file_record(
                _source_manifest(seed)),
            "checkpoint": {
                "iteration": protocol.SOURCE_NEXT_ITERATION - 1,
                "next_iteration": protocol.SOURCE_NEXT_ITERATION,
                "total_steps": protocol.SOURCE_TOTAL_STEPS,
                "update_count": protocol.SOURCE_UPDATE_COUNT,
                "algo": "bapr_regime",
            },
            "initial_components": bootstrap["initial_components"],
            "files": records,
        }
        protocol.write_json_atomic(
            temporary / protocol.CANONICAL_MANIFEST_NAME, payload)
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    finally:
        source_temporary.cleanup()
        if hasattr(source_env, "close"):
            source_env.close()
        if temporary.exists():
            shutil.rmtree(temporary)
    return validate_canonical(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, choices=protocol.TRAINING_SEEDS, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for scheduler staging")
    payload = prepare(args.seed)
    print(
        "LATE-BASE CANONICAL COMPLETE: "
        f"seed={args.seed} alpha="
        f"{payload['initial_components']['log_alpha']['value']:.9g}",
        flush=True)


if __name__ == "__main__":
    main()
