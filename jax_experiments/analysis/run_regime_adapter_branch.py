"""Bootstrap, train, validate, and publish one regime-adapter branch."""
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
from jax_experiments.analysis import regime_adapter_fork as protocol
from jax_experiments.analysis.run_regime_control_headroom_controller import (
    validate_bundle as validate_source_bundle,
)
from jax_experiments.common.checkpoint import load_checkpoint, save_checkpoint
from jax_experiments.common.logging import Logger
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.train import make_algo, make_env


REQUIRED_BUNDLE_FILES = {
    "checkpoints/params.pkl",
    "checkpoints/train_state.pkl",
    "checkpoints/" + protocol.BOOTSTRAP_NAME,
    "logs/protocol_signature.json",
}

# CUDA GEMM reduction order changes when zero context columns widen an
# otherwise identical network. Exact parameter-block, zero-coordinate, and
# residual checks remain binding; these tolerances cover only the forward
# diagnostic observed across the five source seeds and GPU generations.
ACTION_EQUIVALENCE_ATOL = 1e-5
CRITIC_EQUIVALENCE_ATOL = 1e-3
CRITIC_EQUIVALENCE_GLOBAL_RTOL = 2e-3


def _reset_optimizers(agent) -> None:
    agent.policy_opt_state = agent.policy_opt.init(
        nnx.state(agent.policy, nnx.Param))
    agent.critic_opt_state = agent.critic_opt.init(
        nnx.state(agent.critic, nnx.Param))
    agent.alpha_opt_state = agent.alpha_opt.init(agent.log_alpha)
    if hasattr(agent, "context_net"):
        agent.context_opt_state = agent.context_opt.init(
            nnx.state(agent.context_net, nnx.Param))


def _load_source(seed: int):
    seed = protocol.require_seed(seed)
    validate_source_bundle(protocol.ENV, "robust", seed)
    bundle = protocol.source_bundle_dir(seed)
    config = final_task_sweep.load_config(bundle)
    if (config.algo != "regime_sac"
            or config.regime_context_source != "robust"
            or config.env_name != protocol.ENV):
        raise ValueError("adapter source must be the robust RegimeSAC bundle")
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
        raise ValueError("source bundle has the wrong training budget")
    return config, env, tasks, agent, logger, temporary


def _adapter_config(source_config, run_dir: Path, delta: float, mode: int):
    config = deepcopy(source_config)
    config.algo = "bapr_regime"
    config.save_root = str(run_dir.parent)
    config.run_name = run_dir.name
    config.max_iters = protocol.ADAPTER_FINAL_NEXT_ITERATION
    config.start_train_steps = 0
    config.stochastic_mode_fixed_id = protocol.require_mode(mode)
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
    config.bapr_v2_residual_delta = protocol.require_delta(delta)
    config.bapr_v2_action_deviation_weight = 0.01
    config.bapr_v2_base_aux_weight = 0.0
    config.bapr_v2_context_dropout = 0.0
    config.bapr_v2_advantage_gate = False
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


def _robust_config(source_config, run_dir: Path):
    config = deepcopy(source_config)
    config.save_root = str(run_dir.parent)
    config.run_name = run_dir.name
    config.max_iters = protocol.ROBUST_FINAL_NEXT_ITERATION
    config.start_train_steps = 0
    config.stochastic_mode_fixed_id = -1
    config.log_interval = 50
    config.eval_episodes = 3
    config.save_interval = 25
    config.resume = True
    return config


def _equivalence(source_agent, target_agent) -> dict[str, float | bool]:
    key_obs, key_act = jax.random.split(jax.random.PRNGKey(220722))
    obs = jax.random.normal(key_obs, (64, source_agent.obs_dim))
    act = jnp.tanh(jax.random.normal(
        key_act, (64, source_agent.act_dim)))
    source_context = jnp.zeros(
        (64, source_agent.context_dim), dtype=obs.dtype)
    target_context = jnp.zeros(
        (64, target_agent.context_dim), dtype=obs.dtype)
    source_action = source_agent.policy.deterministic(obs, source_context)
    target_action = target_agent.policy.base_deterministic(obs)
    source_q = source_agent.critic(
        jnp.concatenate([obs, source_context], axis=-1), act)
    target_q = target_agent.critic(
        jnp.concatenate([obs, target_context], axis=-1), act)
    action_atol = ACTION_EQUIVALENCE_ATOL
    critic_atol = CRITIC_EQUIVALENCE_ATOL
    critic_global_rtol = CRITIC_EQUIVALENCE_GLOBAL_RTOL
    action_error = float(jnp.max(jnp.abs(source_action - target_action)))
    critic_error = float(jnp.max(jnp.abs(source_q - target_q)))
    critic_scale = float(jnp.max(jnp.abs(source_q)))
    critic_global_relative_error = critic_error / max(critic_scale, 1.0)
    critic_threshold = (
        critic_atol + critic_global_rtol * max(critic_scale, 1.0))
    critic_scaled_error = float(jnp.max(
        jnp.abs(source_q - target_q) / jnp.maximum(jnp.abs(source_q), 1.0)))
    return {
        "pass": bool(
            action_error <= action_atol
            and critic_error <= critic_threshold),
        "max_abs_action_error": action_error,
        "max_abs_critic_error": critic_error,
        "max_abs_source_q": critic_scale,
        "max_scaled_critic_error": critic_scaled_error,
        "global_relative_critic_error": critic_global_relative_error,
        "critic_threshold": critic_threshold,
        "observations": int(obs.shape[0]),
        "action_atol": action_atol,
        "critic_atol": critic_atol,
        "critic_global_rtol": critic_global_rtol,
    }


def _bootstrap(
        seed: int, role: str, run_dir: Path,
        delta: float | None = None, mode: int | None = None) -> dict:
    manifest_path = run_dir / "checkpoints" / protocol.BOOTSTRAP_NAME
    if manifest_path.is_file():
        payload = protocol.read_json(manifest_path)
        expected = protocol.identity(seed, role, delta, mode)
        if (payload.get("schema") != "bapr.regime-adapter-bootstrap.v1"
                or payload.get("identity") != expected):
            raise ValueError(f"invalid existing bootstrap: {manifest_path}")
        return payload
    if run_dir.exists():
        raise RuntimeError(
            f"partial branch exists without bootstrap manifest: {run_dir}")

    source_config, source_env, tasks, source_agent, source_logger, temporary = (
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
                "observations": 0,
                "tolerance": 0.0,
            }
            component_hashes = {
                "source_policy": protocol.source_policy_sha256(agent.policy),
            }
        elif role == "adapter":
            if delta is None or mode is None:
                raise ValueError("adapter bootstrap requires delta and mode")
            config = _adapter_config(source_config, run_dir, delta, mode)
            agent = make_algo(
                config.algo, source_env.obs_dim, source_env.act_dim, config)
            agent.set_task_metadata(tasks)
            protocol.copy_source_controller_to_adapter(source_agent, agent)
            _reset_optimizers(agent)
            equivalence = _equivalence(source_agent, agent)
            if not equivalence["pass"]:
                raise RuntimeError(
                    f"source-to-adapter function mismatch: {equivalence}")
            component_hashes = {
                "source_policy": protocol.source_policy_sha256(
                    source_agent.policy),
                "frozen_base": protocol.base_policy_sha256(agent.policy),
                "initial_residual": protocol.residual_policy_sha256(
                    agent.policy),
                "initial_critic": protocol.critic_sha256(agent.critic),
                "initial_target_critic": protocol.critic_sha256(
                    agent.target_critic),
            }
        else:
            raise ValueError(f"unknown branch role {role!r}")

        replay = ReplayBuffer(
            source_env.obs_dim, source_env.act_dim,
            capacity=config.replay_size,
            belief_dim=getattr(agent, "belief_dim", 0))
        save_checkpoint(
            str(run_dir / "checkpoints"), agent, replay, source_logger,
            iteration=protocol.SOURCE_NEXT_ITERATION - 1,
            total_steps=protocol.SOURCE_TOTAL_STEPS, algo=config.algo)
        source_manifest = (
            protocol.source_bundle_dir(seed) / "bundle_manifest.json")
        payload = {
            "schema": "bapr.regime-adapter-bootstrap.v1",
            "status": "complete",
            "semantics": (
                "common-controller rebootstrap with empty replay and reset "
                "optimizer states; not exact trajectory continuation"),
            "identity": protocol.identity(seed, role, delta, mode),
            "source": {
                "bundle_manifest": protocol.file_record(source_manifest),
                "checkpoint_next_iteration": protocol.SOURCE_NEXT_ITERATION,
                "checkpoint_total_steps": protocol.SOURCE_TOTAL_STEPS,
                "checkpoint_update_count": protocol.SOURCE_UPDATE_COUNT,
            },
            "empty_replay": True,
            "optimizer_states_reset": True,
            "function_equivalence": equivalence,
            "component_hashes": component_hashes,
        }
        protocol.write_json_atomic(manifest_path, payload)
        return payload
    except Exception:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    finally:
        temporary.cleanup()
        if hasattr(source_env, "close"):
            source_env.close()


def _training_command(
        seed: int, role: str, run_dir: Path,
        delta: float | None = None, mode: int | None = None) -> list[str]:
    if role == "robust_continue":
        max_iters = protocol.ROBUST_FINAL_NEXT_ITERATION
        values = [
            "--algo", "regime_sac",
            "--regime_context_source", "robust",
        ]
    elif role == "adapter":
        if delta is None or mode is None:
            raise ValueError("adapter command requires delta and mode")
        max_iters = protocol.ADAPTER_FINAL_NEXT_ITERATION
        values = [
            "--algo", "bapr_regime",
            "--stochastic_mode_fixed_id", str(protocol.require_mode(mode)),
            "--bapr_v2_mode", "supervised",
            "--bapr_v2_latent_dim", "4",
            "--bapr_v2_policy_context_source", "stored",
            "--bapr_v2_policy_mode", "residual",
            "--bapr_v2_training_schedule", "joint",
            "--bapr_v2_base_pretrain_iters", "1",
            "--bapr_v2_teacher_iters", "0",
            "--bapr_v2_student_iters", "0",
            "--bapr_v2_context_hidden_dim", "128",
            "--bapr_v2_context_length", "64",
            "--bapr_v2_context_chunks", "8",
            "--bapr_v2_context_burnin", "16",
            "--bapr_v2_min_history", "16",
            "--bapr_v2_switch_rollout_steps", str(protocol.DWELL_STEPS),
            "--bapr_v2_residual_delta", str(protocol.require_delta(delta)),
            "--bapr_v2_action_deviation_weight", "0.01",
            "--bapr_v2_base_aux_weight", "0.0",
            "--bapr_v2_context_dropout", "0.0",
            "--bapr_v3_context_ensemble_size", "5",
            "--bapr_v3_variance_model", "mode_empirical",
            "--bapr_v3_variance_ceiling", "0.5",
            "--bapr_v3_variance_ema", "0.05",
            "--bapr_regime_inference_iters", "1",
            "--bapr_regime_adaptation_source", "oracle",
            "--no_bapr_regime_advantage_fallback",
            "--no_bapr_regime_clear_replay",
        ]
    else:
        raise ValueError(f"unknown branch role {role!r}")
    return [
        sys.executable, "-u", "-m", "jax_experiments.train",
        *values,
        "--env", protocol.ENV,
        "--seed", str(protocol.require_seed(seed)),
        "--max_iters", str(max_iters),
        "--save_root", str(run_dir.parent),
        "--run_name", run_dir.name,
        "--env_type", "stochastic_mode",
        "--stochastic_mode_family", protocol.FAMILY,
        "--stochastic_mode_dwell_steps", str(protocol.DWELL_STEPS),
        "--stochastic_mode_dwell_distribution", "fixed",
        "--task_num", "4", "--test_task_num", "4",
        "--samples_per_iter", str(protocol.SAMPLES_PER_ITER),
        "--updates_per_iter", str(protocol.UPDATES_PER_ITER),
        "--start_train_steps", "0",
        "--context_warmup_iters", "0",
        "--ensemble_size", "10", "--hidden_dim", "256",
        "--lr", "0.0003",
        "--max_episode_steps", str(protocol.MAX_EPISODE_STEPS),
        "--eval_protocol", "stationary",
        "--log_interval", "25" if role == "adapter" else "50",
        "--eval_episodes", "3", "--save_interval", "25",
        "--min_resume_iteration", str(protocol.SOURCE_NEXT_ITERATION),
        "--resume",
    ]


def _load_final_agent(run_dir: Path):
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


def validate_branch(
        seed: int, role: str, run_dir: Path,
        delta: float | None = None, mode: int | None = None) -> dict:
    bootstrap = protocol.read_json(
        run_dir / "checkpoints" / protocol.BOOTSTRAP_NAME)
    expected_identity = protocol.identity(seed, role, delta, mode)
    if (bootstrap.get("schema") != "bapr.regime-adapter-bootstrap.v1"
            or bootstrap.get("identity") != expected_identity
            or bootstrap.get("empty_replay") is not True
            or bootstrap.get("optimizer_states_reset") is not True):
        raise ValueError(f"invalid branch bootstrap: {run_dir}")
    config, env, agent, next_iteration, total_steps, temporary = (
        _load_final_agent(run_dir))
    try:
        checkpoint = protocol.source.checkpoint_record(run_dir)
        if checkpoint != protocol.expected_checkpoint(role):
            raise ValueError(
                f"branch has wrong final budget: {checkpoint}")
        if (next_iteration != checkpoint["next_iteration"]
                or total_steps != checkpoint["total_steps"]):
            raise ValueError("loaded branch state disagrees with checkpoint")
        signature = protocol.read_json(
            run_dir / "logs" / "protocol_signature.json")
        signature_config = signature.get("config") or {}
        if (signature.get("checkpoint_loaded") is not True
                or signature.get("start_iteration")
                != protocol.SOURCE_NEXT_ITERATION
                or signature.get("total_steps_at_start")
                != protocol.SOURCE_TOTAL_STEPS
                or signature_config.get("start_train_steps") != 0):
            raise ValueError("branch did not start from the audited boundary")
        result = {
            "identity": expected_identity,
            "checkpoint": checkpoint,
            "bootstrap": bootstrap,
            "protocol_signature": signature,
        }
        if role == "adapter":
            assert delta is not None and mode is not None
            if (config.algo != "bapr_regime"
                    or int(config.stochastic_mode_fixed_id) != int(mode)
                    or float(config.bapr_v2_residual_delta) != float(delta)):
                raise ValueError("adapter final config changed")
            final_components = {
                "frozen_base": protocol.base_policy_sha256(agent.policy),
                "residual": protocol.residual_policy_sha256(agent.policy),
                "critic": protocol.critic_sha256(agent.critic),
                "target_critic": protocol.critic_sha256(
                    agent.target_critic),
            }
            initial = bootstrap["component_hashes"]
            if final_components["frozen_base"] != initial["frozen_base"]:
                raise ValueError("frozen robust actor changed during adaptation")
            if final_components["residual"] == initial["initial_residual"]:
                raise ValueError("adapter residual did not update")
            if final_components["critic"] == initial["initial_critic"]:
                raise ValueError("adapter critic did not update")
            mode_log = np.load(
                run_dir / "logs" / "mode_id.npy", allow_pickle=False)
            if (len(mode_log) != protocol.ADAPTER_FINAL_NEXT_ITERATION
                    or not np.all(
                        mode_log[protocol.SOURCE_NEXT_ITERATION:] == mode)):
                raise ValueError("adapter rollout escaped its fixed mode")
            result["final_components"] = final_components
        else:
            if (config.algo != "regime_sac"
                    or config.regime_context_source != "robust"):
                raise ValueError("robust continuation config changed")
        return result
    finally:
        temporary.cleanup()
        if hasattr(env, "close"):
            env.close()


def validate_published_bundle(
        seed: int, role: str, delta: float | None = None,
        mode: int | None = None) -> dict:
    directory = protocol.branch_bundle_dir(seed, role, delta, mode)
    payload = protocol.read_json(directory / protocol.BUNDLE_MANIFEST_NAME)
    if (payload.get("schema") != protocol.BUNDLE_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("identity")
            != protocol.identity(seed, role, delta, mode)
            or set(payload.get("files") or {}) != REQUIRED_BUNDLE_FILES
            or payload.get("checkpoint")
            != protocol.expected_checkpoint(role)):
        raise ValueError(f"invalid regime-adapter bundle: {directory}")
    for relative, expected in payload["files"].items():
        path = directory / relative
        if not path.is_file() or protocol.file_record(path) != expected:
            raise ValueError(f"regime-adapter bundle file changed: {path}")
    return payload


def publish_bundle(
        seed: int, role: str, run_dir: Path,
        delta: float | None = None, mode: int | None = None) -> dict:
    validation = validate_branch(seed, role, run_dir, delta, mode)
    destination = protocol.branch_bundle_dir(seed, role, delta, mode)
    manifest = destination / protocol.BUNDLE_MANIFEST_NAME
    if manifest.is_file():
        return validate_published_bundle(seed, role, delta, mode)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp.", dir=destination.parent))
    try:
        files = (
            Path("checkpoints") / "params.pkl",
            Path("checkpoints") / "train_state.pkl",
            Path("checkpoints") / protocol.BOOTSTRAP_NAME,
            Path("logs") / "protocol_signature.json",
        )
        records = {}
        for relative in files:
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(run_dir / relative, target)
            records[relative.as_posix()] = protocol.file_record(target)
        payload = {
            "schema": protocol.BUNDLE_SCHEMA,
            "status": "complete",
            "identity": protocol.identity(seed, role, delta, mode),
            "checkpoint": validation["checkpoint"],
            "budget_semantics": {
                "shared_pretrain_steps": protocol.SOURCE_TOTAL_STEPS,
                "branch_additional_steps": (
                    protocol.ROBUST_EXTRA_ITERS * protocol.SAMPLES_PER_ITER
                    if role == "robust_continue" else
                    protocol.ADAPTER_EXTRA_ITERS_PER_MODE
                    * protocol.SAMPLES_PER_ITER),
                "adapter_bank_aggregate_additional_steps": (
                    protocol.ROBUST_EXTRA_ITERS * protocol.SAMPLES_PER_ITER),
            },
            "components": validation.get("final_components"),
            "files": records,
        }
        protocol.write_json_atomic(
            temporary / protocol.BUNDLE_MANIFEST_NAME, payload)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return validate_published_bundle(seed, role, delta, mode)


def run(seed: int, role: str, delta: float | None, mode: int | None) -> None:
    seed = protocol.require_seed(seed)
    if role == "robust_continue":
        delta = None
        mode = None
        run_dir = protocol.robust_run_dir(seed)
    else:
        delta = protocol.require_delta(float(delta))
        mode = protocol.require_mode(int(mode))
        run_dir = protocol.adapter_run_dir(seed, delta, mode)
    destination = protocol.branch_bundle_dir(seed, role, delta, mode)
    if (destination / protocol.BUNDLE_MANIFEST_NAME).is_file():
        validate_published_bundle(seed, role, delta, mode)
        print(f"REGIME ADAPTER BRANCH ALREADY COMPLETE: {destination}")
        return
    _bootstrap(seed, role, run_dir, delta, mode)
    expected = protocol.expected_checkpoint(role)
    current = protocol.source.checkpoint_record(run_dir)
    if current["next_iteration"] < expected["next_iteration"]:
        command = _training_command(seed, role, run_dir, delta, mode)
        print("REGIME ADAPTER TRAIN:", " ".join(command), flush=True)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(protocol.ROOT)
        xla_flags = environment.get("XLA_FLAGS", "").split()
        for flag in (
                "--xla_gpu_enable_triton_gemm=false",
                "--xla_cpu_multi_thread_eigen=false",
                "intra_op_parallelism_threads=1"):
            if flag not in xla_flags:
                xla_flags.append(flag)
        environment["XLA_FLAGS"] = " ".join(xla_flags)
        for name in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "JAX_NUM_THREADS", "TF_NUM_INTRAOP_THREADS",
                "TF_NUM_INTEROP_THREADS"):
            environment[name] = "1"
        environment["JAX_CPU_ENABLE_ASYNC_DISPATCH"] = "false"
        subprocess.run(
            command, cwd=protocol.ROOT, env=environment, check=True)
    payload = publish_bundle(seed, role, run_dir, delta, mode)
    replay = run_dir / "checkpoints" / "replay_buffer.npz"
    if replay.is_file():
        replay.unlink()
    print("REGIME ADAPTER BRANCH COMPLETE: " + json.dumps(
        payload["identity"], sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=protocol.TRAINING_SEEDS,
                        required=True)
    parser.add_argument(
        "--role", choices=("robust_continue", "adapter"), required=True)
    parser.add_argument("--delta", type=float)
    parser.add_argument("--mode", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for checkpoint-safe execution")
    if args.role == "adapter" and (
            args.delta is None or args.mode is None):
        raise SystemExit("adapter role requires --delta and --mode")
    run(args.seed, args.role, args.delta, args.mode)


if __name__ == "__main__":
    main()
