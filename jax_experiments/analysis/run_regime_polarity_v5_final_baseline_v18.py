"""Train one v18 baseline and publish evaluation-only parameters."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from flax import nnx

from jax_experiments.analysis import final_task_sweep
from jax_experiments.analysis import (
    regime_polarity_v5_final_comparison_v18 as protocol,
)
from jax_experiments.common.checkpoint import _to_numpy_tree, load_checkpoint
from jax_experiments.common.logging import Logger
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.train import make_algo, make_env


EVAL_PARAMS_SCHEMA = "bapr.evaluation-parameters.v18"


def _run_dir(kind: str, seed: int, slot: int | None) -> Path:
    if kind == "sac_replica":
        if slot is None:
            raise ValueError("SAC replica training requires --slot")
        return protocol.sac_run_dir(seed, slot)
    if slot is not None:
        raise ValueError("--slot is only valid for SAC replicas")
    return protocol.baseline_run_dir(protocol.require_method(kind), seed)


def _training_seed(kind: str, seed: int, slot: int | None) -> int:
    if kind == "sac_replica":
        if slot is None:
            raise ValueError("SAC replica training requires --slot")
        return protocol.replica_training_seed(seed, slot)
    return protocol.require_training_seed(seed)


def _algo(kind: str) -> str:
    return "sac" if kind == "sac_replica" else protocol.algo_for(kind)


def training_command(
    kind: str, seed: int, slot: int | None = None,
) -> list[str]:
    seed = protocol.require_training_seed(seed)
    run = _run_dir(kind, seed, slot)
    train_seed = _training_seed(kind, seed, slot)
    values = [
        sys.executable, "-u", "-m", "jax_experiments.train",
        "--algo", _algo(kind),
        "--env", protocol.ENV,
        "--seed", str(train_seed),
        "--max_iters", str(protocol.MAX_ITERS),
        "--save_root", str(run.parent),
        "--run_name", run.name,
        "--env_type", "stochastic_mode",
        "--stochastic_mode_family", protocol.FAMILY,
        "--stochastic_mode_dwell_steps", str(protocol.DWELL_STEPS),
        "--stochastic_mode_dwell_distribution", "fixed",
        "--task_num", "4",
        "--test_task_num", "4",
        "--samples_per_iter", str(protocol.SAMPLES_PER_ITER),
        "--updates_per_iter", str(protocol.UPDATES_PER_ITER),
        "--start_train_steps", str(protocol.START_TRAIN_STEPS),
        "--initial_random_steps", "0",
        "--hidden_dim", "256",
        "--max_episode_steps", str(protocol.MAX_EPISODE_STEPS),
        "--backend", "spring",
        "--eval_protocol", "stationary",
        "--log_interval", "50",
        "--eval_episodes", "3",
        "--save_interval", "25",
        "--resume",
    ]
    if kind == "sac_replica":
        values += [
            "--context_warmup_iters", "50",
            "--ensemble_size", "2",
            "--lr", "0.0003",
        ]
    elif kind == "escp_recurrent":
        config = protocol.ESCP_CONFIG
        values += [
            "--ensemble_size", str(config["ensemble_size"]),
            "--ep_dim", str(config["ep_dim"]),
            "--lr", str(config["policy_lr"]),
            "--clip_norm", str(config["clip_norm"]),
            "--context_warmup_iters", "0",
            "--rmdm_max_tasks", "4",
            "--rbf_radius", str(config["rbf_radius"]),
            "--consistency_loss_weight", str(config["consistency_weight"]),
            "--diversity_loss_weight", str(config["diversity_weight"]),
            "--escp_context_mode", "recurrent",
            "--escp_history_length", str(config["history_length"]),
            "--escp_target_mode", "twin_min",
            "--escp_actor_mode", "twin_min",
            "--escp_context_min_steps", str(config["context_min_steps"]),
            "--escp_context_min_tasks", str(config["context_min_tasks"]),
            "--escp_alpha_max", "1.0",
            "--escp_policy_lr", str(config["policy_lr"]),
            "--escp_critic_lr", str(config["critic_lr"]),
            "--escp_context_lr", str(config["context_lr"]),
            "--escp_alpha_lr", str(config["alpha_lr"]),
            "--escp_target_entropy_ratio",
            str(config["target_entropy_ratio"]),
            "--escp_bottleneck_sigma", str(config["bottleneck_sigma"]),
            "--escp_prototype_tau", str(config["prototype_tau"]),
        ]
    elif kind == "resac_b0":
        config = protocol.RESAC_CONFIG
        values += [
            "--ensemble_size", str(config["ensemble_size"]),
            "--lr", str(config["lr"]),
            "--beta", str(config["beta_start"]),
            "--weight_reg", "0",
            "--beta_ood", "0",
            "--resac_independent_ratio", str(config["independent_ratio"]),
            "--resac_anchor_lambda", str(config["anchor_lambda"]),
            "--resac_adaptive_beta",
            "--resac_beta_start", str(config["beta_start"]),
            "--resac_beta_end", str(config["beta_end"]),
            "--resac_beta_warmup", str(config["beta_warmup"]),
            "--resac_critic_actor_ratio", "1",
            "--resac_beta_bc", "0",
            "--resac_clip_norm", "0",
            "--ema_tau", str(config["ema_tau"]),
            "--use_ema_eval",
        ]
    else:
        raise ValueError(f"unsupported v18 training kind {kind!r}")
    return values


def expected_config(
    kind: str, seed: int, slot: int | None = None,
) -> dict[str, Any]:
    seed = protocol.require_training_seed(seed)
    expected: dict[str, Any] = {
        "algo": _algo(kind),
        "env_name": protocol.ENV,
        "seed": _training_seed(kind, seed, slot),
        "max_iters": protocol.MAX_ITERS,
        "env_type": "stochastic_mode",
        "stochastic_mode_family": protocol.FAMILY,
        "stochastic_mode_dwell_steps": protocol.DWELL_STEPS,
        "stochastic_mode_dwell_distribution": "fixed",
        "stochastic_mode_fixed_id": -1,
        "task_num": 4,
        "test_task_num": 4,
        "samples_per_iter": protocol.SAMPLES_PER_ITER,
        "updates_per_iter": protocol.UPDATES_PER_ITER,
        "start_train_steps": protocol.START_TRAIN_STEPS,
        "initial_random_steps": 0,
        "hidden_dim": 256,
        "max_episode_steps": protocol.MAX_EPISODE_STEPS,
        "brax_backend": "spring",
    }
    if kind == "sac_replica":
        expected.update({
            "context_warmup_iters": 50,
            "ensemble_size": 2,
            "lr": 3e-4,
        })
    elif kind == "escp_recurrent":
        config = protocol.ESCP_CONFIG
        expected.update({
            "ensemble_size": config["ensemble_size"],
            "ep_dim": config["ep_dim"],
            "lr": config["policy_lr"],
            "clip_norm": config["clip_norm"],
            "context_warmup_iters": 0,
            "rmdm_max_tasks": 4,
            "rbf_radius": config["rbf_radius"],
            "consistency_loss_weight": config["consistency_weight"],
            "diversity_loss_weight": config["diversity_weight"],
            "escp_context_mode": "recurrent",
            "escp_history_length": config["history_length"],
            "escp_target_mode": "twin_min",
            "escp_actor_mode": "twin_min",
            "escp_context_min_steps": config["context_min_steps"],
            "escp_context_min_tasks": config["context_min_tasks"],
            "escp_alpha_max": 1.0,
            "escp_policy_lr": config["policy_lr"],
            "escp_critic_lr": config["critic_lr"],
            "escp_context_lr": config["context_lr"],
            "escp_alpha_lr": config["alpha_lr"],
            "escp_target_entropy_ratio": config["target_entropy_ratio"],
            "escp_bottleneck_sigma": config["bottleneck_sigma"],
            "escp_prototype_tau": config["prototype_tau"],
        })
    elif kind == "resac_b0":
        config = protocol.RESAC_CONFIG
        expected.update({
            "ensemble_size": config["ensemble_size"],
            "lr": config["lr"],
            "beta": config["beta_start"],
            "weight_reg": 0.0,
            "beta_ood": 0.0,
            "resac_independent_ratio": config["independent_ratio"],
            "resac_anchor_lambda": config["anchor_lambda"],
            "resac_adaptive_beta": True,
            "resac_beta_start": config["beta_start"],
            "resac_beta_end": config["beta_end"],
            "resac_beta_warmup": config["beta_warmup"],
            "resac_critic_actor_ratio": 1,
            "resac_beta_bc": 0.0,
            "resac_clip_norm": 0.0,
            "ema_tau": config["ema_tau"],
            "use_ema_eval": True,
            "use_ema_rollout": False,
        })
    else:
        raise ValueError(f"unsupported v18 training kind {kind!r}")
    return expected


def validate_signature(
    kind: str, seed: int, slot: int | None = None,
) -> dict[str, Any]:
    path = _run_dir(kind, seed, slot) / "logs/protocol_signature.json"
    signature = protocol.read_json(path)
    config = signature.get("config") or {}
    mismatches = {
        key: {"actual": config.get(key), "expected": value}
        for key, value in expected_config(kind, seed, slot).items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"v18 baseline config mismatch: {mismatches}")
    return signature


def _expected_eval_keys(kind: str) -> set[str]:
    keys = {"schema", "identity", "policy", "update_count"}
    if kind == "escp_recurrent":
        keys.update({"context_net", "custom_agent_state"})
    elif kind == "resac_b0":
        keys.add("ema_policy")
    return keys


def validate_bundle(
    kind: str, seed: int, slot: int | None = None,
) -> dict[str, Any]:
    destination = protocol.bundle_dir(kind, seed, slot)
    manifest = protocol.read_json(destination / "bundle_manifest.json")
    expected_files = {
        "runtime/" + protocol.EVAL_PARAMS_NAME,
        "logs/protocol_signature.json",
    }
    if (
        manifest.get("schema") != protocol.BUNDLE_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("identity") != protocol.identity(kind, seed, slot)
        or manifest.get("checkpoint") != protocol.expected_checkpoint(kind)
        or manifest.get("registration")
        != protocol.file_record(protocol.REGISTRATION_PATH)
        or set(manifest.get("files") or {}) != expected_files
    ):
        raise ValueError(f"invalid v18 baseline bundle: {destination}")
    for relative, record in manifest["files"].items():
        path = destination / relative
        if not path.is_file() or protocol.file_record(path) != record:
            raise ValueError(f"changed v18 bundle file: {path}")
    with (destination / "runtime" / protocol.EVAL_PARAMS_NAME).open("rb") as handle:
        payload = pickle.load(handle)
    if (
        set(payload) != _expected_eval_keys(kind)
        or payload.get("schema") != EVAL_PARAMS_SCHEMA
        or payload.get("identity") != protocol.identity(kind, seed, slot)
        or int(payload.get("update_count", -1)) != protocol.FINAL_UPDATE_COUNT
    ):
        raise ValueError(f"invalid v18 evaluation parameters: {destination}")
    return manifest


def _load_final_agent(kind: str, seed: int, slot: int | None):
    run = _run_dir(kind, seed, slot)
    config = final_task_sweep.load_config(run)
    env = make_env(config, seed_offset=0)
    tasks = env.sample_tasks(len(protocol.MODES))
    agent = make_algo(_algo(kind), env.obs_dim, env.act_dim, config)
    if hasattr(agent, "set_task_metadata"):
        agent.set_task_metadata(tasks)
    replay = ReplayBuffer(
        env.obs_dim,
        env.act_dim,
        capacity=1,
        belief_dim=getattr(agent, "belief_dim", 0),
    )
    with tempfile.TemporaryDirectory() as temporary:
        logger = Logger(temporary)
        next_iteration, total_steps = load_checkpoint(
            str(run / "checkpoints"),
            agent,
            replay,
            logger,
            _algo(kind),
            load_replay_buffer=False,
        )
    return config, env, agent, next_iteration, total_steps


def publish_bundle(
    kind: str, seed: int, slot: int | None = None,
) -> dict[str, Any]:
    destination = protocol.bundle_dir(kind, seed, slot)
    if (destination / "bundle_manifest.json").is_file():
        return validate_bundle(kind, seed, slot)
    run = _run_dir(kind, seed, slot)
    config, env, agent, next_iteration, total_steps = _load_final_agent(
        kind, seed, slot)
    try:
        checkpoint = protocol.checkpoint_record(run)
        if (
            checkpoint != protocol.expected_checkpoint(kind)
            or int(next_iteration) != protocol.MAX_ITERS
            or int(total_steps) != protocol.FINAL_TOTAL_STEPS
            or int(agent.update_count) != protocol.FINAL_UPDATE_COUNT
            or int(config.stochastic_mode_fixed_id) != -1
        ):
            raise ValueError(f"incomplete v18 training output: {run}")
        validate_signature(kind, seed, slot)
        payload: dict[str, Any] = {
            "schema": EVAL_PARAMS_SCHEMA,
            "identity": protocol.identity(kind, seed, slot),
            "policy": _to_numpy_tree(nnx.state(agent.policy, nnx.Param)),
            "update_count": int(agent.update_count),
        }
        if kind == "escp_recurrent":
            payload["context_net"] = _to_numpy_tree(
                nnx.state(agent.context_net, nnx.Param))
            payload["custom_agent_state"] = agent.checkpoint_state()
        elif kind == "resac_b0":
            payload["ema_policy"] = _to_numpy_tree(
                nnx.state(agent.ema_policy, nnx.Param))

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp.", dir=destination.parent))
        try:
            params_path = temporary / "runtime" / protocol.EVAL_PARAMS_NAME
            params_path.parent.mkdir(parents=True, exist_ok=True)
            with params_path.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            signature_source = run / "logs/protocol_signature.json"
            signature_target = temporary / "logs/protocol_signature.json"
            signature_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(signature_source, signature_target)
            files = {
                path.relative_to(temporary).as_posix(): protocol.file_record(path)
                for path in (params_path, signature_target)
            }
            protocol.write_json_atomic(
                temporary / "bundle_manifest.json",
                {
                    "schema": protocol.BUNDLE_SCHEMA,
                    "status": "complete",
                    "identity": protocol.identity(kind, seed, slot),
                    "checkpoint": checkpoint,
                    "registration": protocol.file_record(
                        protocol.REGISTRATION_PATH),
                    "files": files,
                    "contents": (
                        "evaluation-only policy/context parameters; no critic, "
                        "optimizer, replay, or full checkpoint"
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
    return validate_bundle(kind, seed, slot)


def run(kind: str, seed: int, slot: int | None = None) -> None:
    seed = protocol.require_training_seed(seed)
    if kind == "sac_replica":
        if slot is None:
            raise ValueError("SAC replica run requires --slot")
        slot = protocol.require_replica_slot(slot)
    else:
        protocol.require_method(kind)
        if slot is not None:
            raise ValueError("--slot is only valid for SAC replicas")
    protocol.validate_registration()
    manifest = protocol.bundle_manifest(kind, seed, slot)
    if manifest.is_file():
        validate_bundle(kind, seed, slot)
        print(f"V18 BASELINE ALREADY COMPLETE: {manifest}", flush=True)
        return
    command = training_command(kind, seed, slot)
    print("V18 BASELINE TRAIN:", " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(protocol.ROOT)
    subprocess.run(command, cwd=protocol.ROOT, env=environment, check=True)
    payload = publish_bundle(kind, seed, slot)
    print("V18 BASELINE COMPLETE: " + json.dumps(
        payload["identity"], sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind", choices=("sac_replica", *protocol.TRAINED_METHODS),
        required=True)
    parser.add_argument(
        "--seed", choices=protocol.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--slot", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit("--resume is required for checkpoint-safe execution")
    run(args.kind, args.seed, args.slot)


if __name__ == "__main__":
    main()
