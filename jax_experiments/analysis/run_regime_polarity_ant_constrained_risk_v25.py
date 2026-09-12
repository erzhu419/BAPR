"""Train and publish one V25 constrained-risk Ant specialist."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from jax_experiments.algos.sac_switch_recovery_risk import (
    SACSwitchRecoveryRisk,
)
from jax_experiments.analysis import final_task_sweep
from jax_experiments.analysis import (
    regime_polarity_ant_constrained_risk_v25 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_ant_switch_recovery_v24 as base,
)
from jax_experiments.common.checkpoint import load_checkpoint
from jax_experiments.common.logging import Logger
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.train import make_env


_V24_TARGET_CONFIG = base._target_config


def _apply_risk_config(config, variant: str):
    config.switch_recovery_risk_objective = protocol.risk_objective(variant)
    config.switch_recovery_risk_lambda = protocol.RISK_LAMBDA
    config.switch_recovery_risk_actor_start_update = (
        protocol.RISK_ACTOR_START_UPDATE)
    return config


def _target_config(source_config, run_dir: Path, mode: int, variant: str):
    config = _V24_TARGET_CONFIG(source_config, run_dir, mode, variant)
    return _apply_risk_config(config, variant)


def _make_agent(config, obs_dim: int, act_dim: int) -> SACSwitchRecoveryRisk:
    return SACSwitchRecoveryRisk(obs_dim, act_dim, config, seed=config.seed)


def _runtime_payload(
    variant: str, seed: int, mode: int,
) -> dict[str, Any]:
    return {
        "schema": "bapr.ant-constrained-risk-runtime.v25",
        "identity": protocol.identity(variant, seed, mode),
        "physical_samples_per_iter": protocol.PHYSICAL_SAMPLES_PER_ITER,
        "candidate_samples_per_iter": protocol.CANDIDATE_SAMPLES_PER_ITER,
        "target_mode_samples_per_iter": protocol.TARGET_MODE_SAMPLES_PER_ITER,
        "target_behavior_mix": {
            "candidate_policy": 0.5,
            "frozen_robust_policy": 0.5,
        },
        "robust_prefix_samples_per_iter": (
            protocol.PHYSICAL_SAMPLES_PER_ITER
            - protocol.CANDIDATE_SAMPLES_PER_ITER),
        "segment_steps": protocol.SWITCH_SEGMENT_STEPS,
        "termination_penalty": 0.0,
        "risk_target": "discounted_termination_probability",
        "risk_objective": protocol.risk_objective(variant),
        "risk_lambda": protocol.RISK_LAMBDA,
        "risk_actor_start_update": protocol.RISK_ACTOR_START_UPDATE,
        "predecessor_policy": "frozen_matched_robust_actor",
        "specialist_replay": "post_switch_target_mode_only",
    }


def _bind() -> None:
    base.protocol = protocol
    base._target_config = _target_config
    base._make_agent = _make_agent
    base._runtime_payload = _runtime_payload
    base._load_final_agent = _load_final_agent


def _load_source(seed: int):
    _bind()
    return base._load_source(seed)


def training_command(
    variant: str,
    seed: int,
    mode: int,
    run_dir: Path | None = None,
) -> list[str]:
    variant = protocol.require_variant(variant)
    seed = protocol.require_training_seed(seed)
    mode = protocol.require_mode(mode)
    run_dir = run_dir or protocol.run_dir(variant, seed, mode)
    return [
        sys.executable,
        "-u",
        "-m",
        "jax_experiments.analysis."
        "train_regime_polarity_ant_constrained_risk_v25",
        "--algo", "sac",
        "--env", protocol.ENV,
        "--seed", str(seed),
        "--max_iters", str(protocol.FINAL_NEXT_ITERATION),
        "--save_root", str(run_dir.parent),
        "--run_name", run_dir.name,
        "--env_type", "stochastic_mode",
        "--stochastic_mode_family", protocol.FAMILY,
        "--stochastic_mode_dwell_steps", str(protocol.DWELL_STEPS),
        "--stochastic_mode_dwell_distribution", "fixed",
        "--stochastic_mode_fixed_id", str(mode),
        "--task_num", "4",
        "--test_task_num", "4",
        "--samples_per_iter", str(protocol.PHYSICAL_SAMPLES_PER_ITER),
        "--updates_per_iter", str(protocol.UPDATES_PER_ITER),
        "--start_train_steps", "0",
        "--context_warmup_iters", "50",
        "--ensemble_size", "2",
        "--hidden_dim", "256",
        "--lr", "0.0003",
        "--max_episode_steps", str(protocol.MAX_EPISODE_STEPS),
        "--backend", "spring",
        "--eval_protocol", "stationary",
        "--log_interval", "50",
        "--eval_episodes", "3",
        "--save_interval", "50",
        "--resume",
        "--min_resume_iteration", str(protocol.SOURCE_NEXT_ITERATION),
    ]


def _runtime_environment(variant: str, mode: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(protocol.ROOT)
    environment["BAPR_SWITCH_RECOVERY_TARGET_MODE"] = str(
        protocol.require_mode(mode))
    environment["BAPR_SWITCH_RECOVERY_SEGMENT_STEPS"] = str(
        protocol.SWITCH_SEGMENT_STEPS)
    environment["BAPR_SWITCH_RECOVERY_RISK_OBJECTIVE"] = (
        protocol.risk_objective(variant))
    environment["BAPR_SWITCH_RECOVERY_RISK_LAMBDA"] = str(
        protocol.RISK_LAMBDA)
    environment["BAPR_SWITCH_RECOVERY_RISK_ACTOR_START_UPDATE"] = str(
        protocol.RISK_ACTOR_START_UPDATE)
    return environment


def _load_final_agent(variant: str, seed: int, mode: int):
    run_dir = protocol.run_dir(variant, seed, mode)
    config = final_task_sweep.load_config(run_dir)
    config.switch_recovery_target_mode = protocol.require_mode(mode)
    config.switch_recovery_segment_steps = protocol.SWITCH_SEGMENT_STEPS
    config.switch_recovery_termination_penalty = 0.0
    _apply_risk_config(config, variant)
    env = make_env(config, seed_offset=0)
    agent = _make_agent(config, env.obs_dim, env.act_dim)
    replay = ReplayBuffer(env.obs_dim, env.act_dim, capacity=1)
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
    _bind()
    return base.validate_bundle(variant, seed, mode)


def publish_bundle(variant: str, seed: int, mode: int) -> dict[str, Any]:
    _bind()
    return base.publish_bundle(variant, seed, mode)


def run(variant: str, seed: int, mode: int) -> None:
    _bind()
    variant = protocol.require_variant(variant)
    seed = protocol.require_training_seed(seed)
    mode = protocol.require_mode(mode)
    protocol.validate_registration()
    if protocol.bundle_manifest(variant, seed, mode).is_file():
        validate_bundle(variant, seed, mode)
        print(
            f"V25 CONSTRAINED RISK ALREADY COMPLETE: {variant} "
            f"seed={seed} mode={mode}",
            flush=True,
        )
        return
    run_dir = protocol.run_dir(variant, seed, mode)
    base._bootstrap(variant, seed, mode, run_dir)
    command = training_command(variant, seed, mode, run_dir)
    print("V25 CONSTRAINED RISK TRAIN:", " ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=protocol.ROOT,
        env=_runtime_environment(variant, mode),
        check=True,
    )
    payload = publish_bundle(variant, seed, mode)
    print(
        "V25 CONSTRAINED RISK COMPLETE: "
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
