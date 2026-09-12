"""Bootstrap, train, validate, and publish one anchored-controller branch."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.analysis import final_task_sweep
from jax_experiments.analysis import (
    regime_polarity_anchored_residual as protocol,
)
from jax_experiments.analysis import (
    run_regime_control_headroom_controller as source_common,
)
from jax_experiments.common.checkpoint import load_checkpoint, save_checkpoint
from jax_experiments.common.logging import Logger
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.train import make_algo, make_env


source_common.protocol = protocol

REQUIRED_BUNDLE_FILES = {
    "checkpoints/params.pkl",
    "checkpoints/train_state.pkl",
    "checkpoints/" + protocol.BOOTSTRAP_NAME,
    "logs/protocol_signature.json",
}
ACTION_ATOL = 1e-5
CRITIC_ATOL = 2e-3
CRITIC_RTOL = 2e-3


def _copy_value(destination, value) -> None:
    destination.value = jnp.asarray(value, dtype=destination.value.dtype)


def _reset_optimizers(agent) -> None:
    agent.policy_opt_state = agent.policy_opt.init(
        nnx.state(agent.policy, nnx.Param))
    agent.critic_opt_state = agent.critic_opt.init(
        nnx.state(agent.critic, nnx.Param))
    agent.alpha_opt_state = agent.alpha_opt.init(agent.log_alpha)
    if hasattr(agent, "context_net"):
        agent.context_opt_state = agent.context_opt.init(
            nnx.state(agent.context_net, nnx.Param))


def copy_source_policy(source_agent, target_agent) -> None:
    source = source_agent.policy
    target = target_agent.policy
    if len(source.layers) != len(target.base_layers):
        raise ValueError("source and anchored actor depths differ")
    for index, (source_layer, target_layer) in enumerate(
            zip(source.layers, target.base_layers)):
        kernel = source_layer.kernel.value
        if index == 0:
            expected = source_agent.obs_dim + source_agent.context_dim
            if kernel.shape[0] != expected:
                raise ValueError("source actor input width changed")
            kernel = kernel[:source_agent.obs_dim]
        if kernel.shape != target_layer.kernel.value.shape:
            raise ValueError(
                f"actor layer {index} shape mismatch: "
                f"{kernel.shape} != {target_layer.kernel.value.shape}")
        _copy_value(target_layer.kernel, kernel)
        _copy_value(target_layer.bias, source_layer.bias.value)
    for source_head, target_head in (
            (source.mean_head, target.base_mean),
            (source.log_std_head, target.base_log_std)):
        _copy_value(target_head.kernel, source_head.kernel.value)
        _copy_value(target_head.bias, source_head.bias.value)
    if not target.zero_residual_output():
        raise ValueError("anchored policy residual is not exactly zero")


def copy_source_ensemble_critic(
    source,
    target,
    *,
    obs_dim: int,
    source_context_dim: int,
    target_context_dim: int,
) -> None:
    """Copy the source zero-context function into an obs/context critic."""
    if len(source.layers) != len(target.layers):
        raise ValueError("source and target critic depths differ")
    for index, (source_layer, target_layer) in enumerate(
            zip(source.layers, target.layers)):
        source_kernel = source_layer.kernel.value
        if index == 0:
            target_kernel = jnp.zeros_like(target_layer.kernel.value)
            source_action = obs_dim + source_context_dim
            target_action = obs_dim + target_context_dim
            action_dim = source_kernel.shape[1] - source_action
            if target_kernel.shape[1] - target_action != action_dim:
                raise ValueError("source and target critic actions differ")
            target_kernel = target_kernel.at[:, :obs_dim].set(
                source_kernel[:, :obs_dim])
            target_kernel = target_kernel.at[
                :, target_action:target_action + action_dim
            ].set(source_kernel[:, source_action:])
        else:
            target_kernel = source_kernel
        if target_kernel.shape != target_layer.kernel.value.shape:
            raise ValueError(
                f"critic layer {index} shape mismatch: "
                f"{target_kernel.shape} != {target_layer.kernel.value.shape}")
        _copy_value(target_layer.kernel, target_kernel)
        _copy_value(target_layer.bias, source_layer.bias.value)


def copy_source_controller(source_agent, target_agent) -> None:
    copy_source_policy(source_agent, target_agent)
    for source_critic, target_dual in (
            (source_agent.critic, target_agent.critic),
            (source_agent.target_critic, target_agent.target_critic)):
        copy_source_ensemble_critic(
            source_critic,
            target_dual.base_critic,
            obs_dim=source_agent.obs_dim,
            source_context_dim=source_agent.context_dim,
            target_context_dim=0,
        )
        copy_source_ensemble_critic(
            source_critic,
            target_dual.adaptive_critic,
            obs_dim=source_agent.obs_dim,
            source_context_dim=source_agent.context_dim,
            target_context_dim=len(protocol.MODES),
        )
    source_log_alpha = jnp.asarray(source_agent.log_alpha)
    if source_log_alpha.ndim != 0:
        raise ValueError("robust source alpha must be scalar")
    target_agent.log_alpha = jnp.full(
        (len(protocol.MODES) + 1,),
        source_log_alpha,
        dtype=source_log_alpha.dtype,
    )
    target_agent.update_count = int(source_agent.update_count)


def _load_source(seed: int):
    seed = protocol.require_training_seed(seed)
    source_common.validate_bundle(protocol.ENV, "robust", seed)
    bundle = protocol.bundle_dir(protocol.ENV, "robust", seed)
    config = final_task_sweep.load_config(bundle)
    if (config.algo != "regime_sac"
            or config.regime_context_source != "robust"
            or config.env_name != protocol.ENV
            or config.stochastic_mode_family != protocol.FAMILY):
        raise ValueError("anchored source must be the fresh robust bundle")
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
        str(bundle / "checkpoints"), agent, replay, logger,
        config.algo, load_replay_buffer=False)
    if (next_iteration != protocol.SOURCE_NEXT_ITERATION
            or total_steps != protocol.SOURCE_TOTAL_STEPS
            or agent.update_count != protocol.SOURCE_UPDATE_COUNT):
        temporary.cleanup()
        raise ValueError("anchored source has the wrong training budget")
    return config, env, tasks, agent, logger, temporary


def _robust_config(source, run_dir: Path):
    config = deepcopy(source)
    config.save_root = str(run_dir.parent)
    config.run_name = run_dir.name
    config.max_iters = protocol.BRANCH_FINAL_NEXT_ITERATION
    config.start_train_steps = 0
    config.stochastic_mode_fixed_id = -1
    config.log_interval = 50
    config.eval_episodes = 3
    config.save_interval = 25
    config.resume = True
    return config


def _anchored_config(source, run_dir: Path):
    config = deepcopy(source)
    config.algo = "anchored_regime_sac"
    config.save_root = str(run_dir.parent)
    config.run_name = run_dir.name
    config.max_iters = protocol.BRANCH_FINAL_NEXT_ITERATION
    config.start_train_steps = 0
    config.stochastic_mode_fixed_id = -1
    config.bapr_v2_mode = "supervised"
    config.bapr_v2_latent_dim = len(protocol.MODES)
    config.bapr_v2_policy_context_source = "stored"
    config.bapr_v2_training_schedule = "joint"
    config.bapr_v2_policy_mode = "residual"
    config.bapr_v2_context_hidden_dim = 128
    config.bapr_v2_context_length = 64
    config.bapr_v2_context_chunks = 8
    config.bapr_v2_context_burnin = 16
    config.bapr_v2_min_history = 16
    config.bapr_v2_switch_rollout_steps = protocol.DWELL_STEPS
    config.bapr_v2_residual_delta = protocol.RESIDUAL_DELTA
    config.bapr_v2_action_deviation_weight = 0.01
    config.bapr_v2_base_aux_weight = 0.0
    config.bapr_v2_context_dropout = 0.0
    config.bapr_v2_advantage_gate = False
    config.bapr_v2_actor_objective = "mean"
    config.bapr_v2_critic_target_mode = "min"
    config.bapr_v2_beta_ood = 0.0
    config.bapr_v2_reg_weight = 0.0
    config.bapr_v3_context_ensemble_size = 5
    config.bapr_v3_variance_model = "mode_empirical"
    config.bapr_v3_variance_ceiling = 0.5
    config.bapr_v3_variance_ema = 0.05
    config.bapr_v4_training_source_period = 2
    config.bapr_v4_training_robust_slots = 1
    config.bapr_v4_context_bootstrap_model = ""
    config.bapr_v4_context_bootstrap_manifest = ""
    config.context_warmup_iters = 0
    config.log_interval = 25
    config.eval_episodes = 3
    config.save_interval = 25
    config.resume = True
    return config


def _equivalence(source_agent, target_agent) -> dict[str, object]:
    obs_key, act_key = jax.random.split(jax.random.PRNGKey(20260729))
    obs = jax.random.normal(obs_key, (64, source_agent.obs_dim))
    act = jnp.tanh(jax.random.normal(
        act_key, (64, source_agent.act_dim)))
    source_context = jnp.zeros(
        (64, source_agent.context_dim), dtype=obs.dtype)
    source_action = source_agent.policy.deterministic(obs, source_context)
    source_q = source_agent.critic(
        jnp.concatenate([obs, source_context], axis=-1), act)
    action_errors = []
    critic_errors = []
    for context in (
            jnp.zeros((64, len(protocol.MODES) + 1), dtype=obs.dtype),
            *[
                jnp.broadcast_to(
                    jnp.concatenate([
                        jax.nn.one_hot(mode, len(protocol.MODES)),
                        jnp.ones((1,), dtype=obs.dtype),
                    ])[None],
                    (64, len(protocol.MODES) + 1),
                )
                for mode in protocol.MODES
            ]):
        action = target_agent.policy.deterministic(obs, context)
        q_value = target_agent.critic(
            jnp.concatenate([obs, context], axis=-1), act)
        action_errors.append(float(jnp.max(jnp.abs(
            source_action - action))))
        critic_errors.append(float(jnp.max(jnp.abs(source_q - q_value))))
    action_error = max(action_errors)
    critic_error = max(critic_errors)
    critic_scale = float(jnp.max(jnp.abs(source_q)))
    critic_threshold = CRITIC_ATOL + CRITIC_RTOL * max(critic_scale, 1.0)
    return {
        "pass": bool(
            action_error <= ACTION_ATOL
            and critic_error <= critic_threshold),
        "max_abs_action_error": action_error,
        "max_abs_critic_error": critic_error,
        "max_abs_source_q": critic_scale,
        "critic_threshold": critic_threshold,
        "action_atol": ACTION_ATOL,
        "critic_atol": CRITIC_ATOL,
        "critic_rtol": CRITIC_RTOL,
        "contexts_checked": len(protocol.MODES) + 1,
    }


def _component_hashes(agent) -> dict[str, str]:
    policy = protocol.anchored_policy_hashes(agent.policy)
    critic = protocol.anchored_critic_hashes(agent.critic)
    target = protocol.anchored_critic_hashes(agent.target_critic)
    return {
        "base_policy": policy["base"],
        "residual_policy": policy["residual"],
        "base_critic": critic["base"],
        "adaptive_critic": critic["adaptive"],
        "base_target_critic": target["base"],
        "adaptive_target_critic": target["adaptive"],
    }


def _bootstrap(seed: int, role: str, run_dir: Path) -> dict:
    role = protocol.require_branch_role(role)
    manifest = run_dir / "checkpoints" / protocol.BOOTSTRAP_NAME
    if manifest.is_file():
        payload = protocol.read_json(manifest)
        if (payload.get("schema")
                != "bapr.regime-polarity-anchored-bootstrap.v1"
                or payload.get("identity")
                != protocol.branch_identity(role, seed)):
            raise ValueError(f"invalid anchored bootstrap: {manifest}")
        return payload
    if run_dir.exists():
        raise RuntimeError(
            f"partial anchored branch lacks bootstrap: {run_dir}")

    source_config, env, tasks, source_agent, logger, temporary = (
        _load_source(seed))
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
        if role == "robust_continue":
            config = _robust_config(source_config, run_dir)
            agent = source_agent
            agent.config = config
            _reset_optimizers(agent)
            equivalence = {
                "pass": True,
                "max_abs_action_error": 0.0,
                "max_abs_critic_error": 0.0,
                "contexts_checked": 1,
            }
            components = {
                "source_policy": protocol.source_policy_sha256(agent.policy),
            }
        else:
            config = _anchored_config(source_config, run_dir)
            agent = make_algo(
                config.algo, env.obs_dim, env.act_dim, config)
            agent.set_task_metadata(tasks)
            copy_source_controller(source_agent, agent)
            _reset_optimizers(agent)
            equivalence = _equivalence(source_agent, agent)
            if not equivalence["pass"]:
                raise RuntimeError(
                    f"anchored bootstrap changed the source function: "
                    f"{equivalence}")
            components = _component_hashes(agent)

        replay = ReplayBuffer(
            env.obs_dim, env.act_dim, capacity=config.replay_size,
            belief_dim=getattr(agent, "belief_dim", 0))
        save_checkpoint(
            str(run_dir / "checkpoints"), agent, replay, logger,
            iteration=protocol.SOURCE_FINAL_ITERATION,
            total_steps=protocol.SOURCE_TOTAL_STEPS,
            algo=config.algo,
        )
        source_manifest = protocol.bundle_manifest(
            protocol.ENV, "robust", seed)
        payload = {
            "schema": "bapr.regime-polarity-anchored-bootstrap.v1",
            "status": "complete",
            "identity": protocol.branch_identity(role, seed),
            "semantics": (
                "common-controller rebootstrap with empty replay and reset "
                "optimizer states; both branches start at 5.6M transitions"),
            "source": {
                "bundle_manifest": protocol.file_record(source_manifest),
                "checkpoint_next_iteration": protocol.SOURCE_NEXT_ITERATION,
                "checkpoint_total_steps": protocol.SOURCE_TOTAL_STEPS,
                "checkpoint_update_count": protocol.SOURCE_UPDATE_COUNT,
            },
            "empty_replay": True,
            "optimizer_states_reset": True,
            "function_equivalence": equivalence,
            "initial_components": components,
        }
        protocol.write_json_atomic(manifest, payload)
        return payload
    except Exception:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    finally:
        temporary.cleanup()
        if hasattr(env, "close"):
            env.close()


def _training_command(seed: int, role: str, run_dir: Path) -> list[str]:
    role = protocol.require_branch_role(role)
    if role == "robust_continue":
        values = [
            "--algo", "regime_sac",
            "--regime_context_source", "robust",
        ]
    else:
        values = [
            "--algo", "anchored_regime_sac",
            "--bapr_v2_mode", "supervised",
            "--bapr_v2_latent_dim", "4",
            "--bapr_v2_policy_context_source", "stored",
            "--bapr_v2_training_schedule", "joint",
            "--bapr_v2_policy_mode", "residual",
            "--bapr_v2_context_hidden_dim", "128",
            "--bapr_v2_context_length", "64",
            "--bapr_v2_context_chunks", "8",
            "--bapr_v2_context_burnin", "16",
            "--bapr_v2_min_history", "16",
            "--bapr_v2_switch_rollout_steps", str(protocol.DWELL_STEPS),
            "--bapr_v2_residual_delta", str(protocol.RESIDUAL_DELTA),
            "--bapr_v2_action_deviation_weight", "0.01",
            "--bapr_v2_base_aux_weight", "0.0",
            "--bapr_v2_context_dropout", "0.0",
            "--bapr_v2_actor_objective", "mean",
            "--bapr_v2_critic_target_mode", "min",
            "--bapr_v2_beta_ood", "0.0",
            "--bapr_v2_reg_weight", "0.0",
            "--bapr_v3_context_ensemble_size", "5",
            "--bapr_v3_variance_model", "mode_empirical",
            "--bapr_v3_variance_ceiling", "0.5",
            "--bapr_v3_variance_ema", "0.05",
            "--bapr_v4_training_source_period", "2",
            "--bapr_v4_training_robust_slots", "1",
        ]
    return [
        sys.executable, "-u", "-m", "jax_experiments.train",
        *values,
        "--env", protocol.ENV,
        "--seed", str(protocol.require_training_seed(seed)),
        "--max_iters", str(protocol.BRANCH_FINAL_NEXT_ITERATION),
        "--save_root", str(run_dir.parent),
        "--run_name", run_dir.name,
        "--env_type", "stochastic_mode",
        "--stochastic_mode_family", protocol.FAMILY,
        "--stochastic_mode_dwell_steps", str(protocol.DWELL_STEPS),
        "--stochastic_mode_dwell_distribution", "fixed",
        "--task_num", "4",
        "--test_task_num", "4",
        "--samples_per_iter", str(protocol.SAMPLES_PER_ITER),
        "--updates_per_iter", str(protocol.UPDATES_PER_ITER),
        "--start_train_steps", "0",
        "--context_warmup_iters", "0",
        "--ensemble_size", "10",
        "--hidden_dim", "256",
        "--lr", "0.0003",
        "--max_episode_steps", str(protocol.MAX_EPISODE_STEPS),
        "--eval_protocol", "stationary",
        "--log_interval", "25" if role == "anchored" else "50",
        "--eval_episodes", "3",
        "--save_interval", "25",
        "--min_resume_iteration", str(protocol.SOURCE_NEXT_ITERATION),
        "--resume",
    ]


def _load_final(run_dir: Path):
    config = final_task_sweep.load_config(run_dir)
    env = make_env(config, seed_offset=0)
    tasks = env.sample_tasks(config.task_num)
    agent = make_algo(config.algo, env.obs_dim, env.act_dim, config)
    if hasattr(agent, "set_task_metadata"):
        agent.set_task_metadata(tasks)
    replay = ReplayBuffer(
        env.obs_dim, env.act_dim, capacity=1,
        belief_dim=getattr(agent, "belief_dim", 0))
    temporary = tempfile.TemporaryDirectory()
    logger = Logger(temporary.name)
    next_iteration, total_steps = load_checkpoint(
        str(run_dir / "checkpoints"), agent, replay, logger,
        config.algo, load_replay_buffer=False)
    return config, env, agent, next_iteration, total_steps, temporary


def validate_branch(seed: int, role: str, run_dir: Path) -> dict:
    role = protocol.require_branch_role(role)
    bootstrap = protocol.read_json(
        run_dir / "checkpoints" / protocol.BOOTSTRAP_NAME)
    if (bootstrap.get("schema")
            != "bapr.regime-polarity-anchored-bootstrap.v1"
            or bootstrap.get("identity")
            != protocol.branch_identity(role, seed)
            or bootstrap.get("empty_replay") is not True
            or bootstrap.get("optimizer_states_reset") is not True):
        raise ValueError(f"invalid anchored bootstrap: {run_dir}")
    config, env, agent, next_iteration, total_steps, temporary = (
        _load_final(run_dir))
    try:
        checkpoint = protocol.checkpoint_record(run_dir)
        expected = protocol.expected_branch_checkpoint(role)
        if checkpoint != expected:
            raise ValueError(
                f"anchored branch has wrong final budget: {checkpoint}")
        if (next_iteration != expected["next_iteration"]
                or total_steps != expected["total_steps"]):
            raise ValueError("loaded anchored state disagrees with manifest")
        signature = protocol.read_json(
            run_dir / "logs" / "protocol_signature.json")
        config_record = signature.get("config") or {}
        if (signature.get("checkpoint_loaded") is not True
                or signature.get("start_iteration")
                != protocol.SOURCE_NEXT_ITERATION
                or signature.get("total_steps_at_start")
                != protocol.SOURCE_TOTAL_STEPS
                or config_record.get("start_train_steps") != 0):
            raise ValueError("anchored branch missed its audited boundary")
        result = {
            "identity": protocol.branch_identity(role, seed),
            "checkpoint": checkpoint,
            "bootstrap": bootstrap,
            "protocol_signature": signature,
        }
        if role == "anchored":
            required_config = {
                "algo": "anchored_regime_sac",
                "bapr_v2_policy_mode": "residual",
                "bapr_v2_training_schedule": "joint",
                "bapr_v2_critic_target_mode": "min",
                "bapr_v4_training_source_period": 2,
                "bapr_v4_training_robust_slots": 1,
                "bapr_v2_residual_delta": protocol.RESIDUAL_DELTA,
            }
            mismatches = {
                key: {"actual": getattr(config, key), "expected": value}
                for key, value in required_config.items()
                if getattr(config, key) != value
            }
            if mismatches:
                raise ValueError(
                    f"anchored final config changed: {mismatches}")
            final_components = _component_hashes(agent)
            initial = bootstrap["initial_components"]
            for name in final_components:
                if final_components[name] == initial[name]:
                    raise ValueError(
                        f"anchored component did not update: {name}")
            mode_log = np.load(
                run_dir / "logs" / "mode_id.npy", allow_pickle=False)
            observed = set(
                int(value)
                for value in mode_log[protocol.SOURCE_NEXT_ITERATION:])
            if observed != set(protocol.MODES):
                raise ValueError(
                    f"anchored rollout missed modes: {observed}")
            context_log = run_dir / "logs" / "v4_training_context_source.npy"
            if context_log.is_file():
                sources = np.load(context_log, allow_pickle=False)
                tail = np.asarray(sources[-protocol.BRANCH_EXTRA_ITERS:])
                expected_sources = np.asarray([
                    int(iteration % 2 != 0)
                    for iteration in range(
                        protocol.SOURCE_NEXT_ITERATION,
                        protocol.BRANCH_FINAL_NEXT_ITERATION)
                ])
                if not np.array_equal(tail.astype(np.int32), expected_sources):
                    raise ValueError(
                        "anchored rollout-source cycle is not 1:1")
            result["final_components"] = final_components
        elif (config.algo != "regime_sac"
              or config.regime_context_source != "robust"):
            raise ValueError("robust continuation config changed")
        return result
    finally:
        temporary.cleanup()
        if hasattr(env, "close"):
            env.close()


def validate_published(seed: int, role: str) -> dict:
    directory = protocol.branch_bundle_dir(role, seed)
    payload = protocol.read_json(directory / "bundle_manifest.json")
    if (payload.get("schema") != protocol.BRANCH_BUNDLE_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("identity")
            != protocol.branch_identity(role, seed)
            or payload.get("checkpoint")
            != protocol.expected_branch_checkpoint(role)
            or set(payload.get("files") or {}) != REQUIRED_BUNDLE_FILES):
        raise ValueError(f"invalid anchored bundle: {directory}")
    for relative, record in payload["files"].items():
        path = directory / relative
        if not path.is_file() or protocol.file_record(path) != record:
            raise ValueError(f"anchored bundle changed: {path}")
    return payload


def publish_bundle(seed: int, role: str, run_dir: Path) -> dict:
    validation = validate_branch(seed, role, run_dir)
    destination = protocol.branch_bundle_dir(role, seed)
    if (destination / "bundle_manifest.json").is_file():
        return validate_published(seed, role)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp.", dir=destination.parent))
    try:
        records = {}
        for relative in (
                Path("checkpoints") / "params.pkl",
                Path("checkpoints") / "train_state.pkl",
                Path("checkpoints") / protocol.BOOTSTRAP_NAME,
                Path("logs") / "protocol_signature.json"):
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(run_dir / relative, target)
            records[relative.as_posix()] = protocol.file_record(target)
        payload = {
            "schema": protocol.BRANCH_BUNDLE_SCHEMA,
            "status": "complete",
            "identity": protocol.branch_identity(role, seed),
            "checkpoint": validation["checkpoint"],
            "budget_semantics": {
                "shared_source_steps": protocol.SOURCE_TOTAL_STEPS,
                "additional_steps": (
                    protocol.BRANCH_EXTRA_ITERS
                    * protocol.SAMPLES_PER_ITER),
                "controller_updates": protocol.BRANCH_UPDATE_COUNT,
                "anchored_dual_context_relabel": role == "anchored",
            },
            "components": validation.get("final_components"),
            "files": records,
        }
        protocol.write_json_atomic(
            temporary / "bundle_manifest.json", payload)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return validate_published(seed, role)


def run(seed: int, role: str) -> None:
    seed = protocol.require_training_seed(seed)
    role = protocol.require_branch_role(role)
    run_dir = protocol.branch_run_dir(role, seed)
    destination = protocol.branch_bundle_dir(role, seed)
    if (destination / "bundle_manifest.json").is_file():
        validate_published(seed, role)
        print(f"ANCHORED BRANCH ALREADY COMPLETE: {destination}")
        return
    _bootstrap(seed, role, run_dir)
    expected = protocol.expected_branch_checkpoint(role)
    current = protocol.checkpoint_record(run_dir)
    if current["next_iteration"] < expected["next_iteration"]:
        command = _training_command(seed, role, run_dir)
        print("ANCHORED TRAIN:", " ".join(command), flush=True)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(protocol.ROOT)
        subprocess.run(
            command, cwd=protocol.ROOT, env=environment, check=True)
    payload = publish_bundle(seed, role, run_dir)
    replay = run_dir / "checkpoints" / "replay_buffer.npz"
    if replay.is_file():
        replay.unlink()
    print("ANCHORED BRANCH COMPLETE: " + json.dumps(
        payload["identity"], sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument(
        "--role", choices=protocol.BRANCH_ROLES, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for checkpoint-safe execution")
    run(args.seed, args.role)


if __name__ == "__main__":
    main()
