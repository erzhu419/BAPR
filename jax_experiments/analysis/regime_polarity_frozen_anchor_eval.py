"""Bind strict anchored evaluation to one frozen-anchor v2 variant."""
from __future__ import annotations

import tempfile

from flax import nnx

from jax_experiments.analysis import final_task_sweep
from jax_experiments.analysis import (
    regime_polarity_anchored_eval as base,
)
from jax_experiments.analysis import (
    regime_polarity_frozen_anchor as protocol_module,
)
from jax_experiments.analysis import (
    run_regime_polarity_frozen_anchor_branch as branch_runner,
)
from jax_experiments.common.checkpoint import load_checkpoint
from jax_experiments.common.logging import Logger
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.train import make_algo, make_env


CALIBRATION_ARMS = base.CALIBRATION_ARMS
AUDIT_ARMS = base.AUDIT_ARMS
LEARNED_ARMS = base.LEARNED_ARMS
ANCHORED_ARMS = base.ANCHORED_ARMS
InverseSystemIDEstimator = base.InverseSystemIDEstimator

protocol = None


def bind_variant(variant: str):
    global protocol
    protocol = protocol_module.variant_view(variant)
    base.protocol = protocol
    return protocol


def _require_bound():
    if protocol is None:
        raise RuntimeError("bind_variant must be called before evaluation")
    return protocol


def load_controller(role: str, seed: int):
    view = _require_bound()
    role = view.require_branch_role(role)
    actual_role = (
        "robust_long" if role == "robust_continue" else view.variant)
    seed = view.require_training_seed(seed)
    branch_runner.validate_published(seed, actual_role)
    directory = protocol_module.branch_bundle_dir(actual_role, seed)
    config = final_task_sweep.load_config(directory)
    expected_algo = protocol_module.branch_algo(actual_role)
    if (
        config.algo != expected_algo
        or config.env_name != view.ENV
        or config.env_type != "stochastic_mode"
        or config.stochastic_mode_family != view.FAMILY
    ):
        raise ValueError(
            f"wrong frozen-anchor controller config: {directory}")
    env = make_env(config, seed_offset=0)
    tasks = env.sample_tasks(config.task_num)
    agent = make_algo(config.algo, env.obs_dim, env.act_dim, config)
    agent.set_task_metadata(tasks)
    replay = ReplayBuffer(
        env.obs_dim,
        env.act_dim,
        capacity=1,
        belief_dim=getattr(agent, "belief_dim", 0),
    )
    with tempfile.TemporaryDirectory() as log_dir:
        logger = Logger(log_dir)
        next_iteration, total_steps = load_checkpoint(
            str(directory / "checkpoints"),
            agent,
            replay,
            logger,
            config.algo,
            load_replay_buffer=False,
        )
    if (
        next_iteration != view.BRANCH_FINAL_NEXT_ITERATION
        or total_steps != view.BRANCH_TOTAL_STEPS
        or agent.update_count != view.BRANCH_UPDATE_COUNT
    ):
        raise ValueError(f"stale frozen-anchor controller: {directory}")
    if hasattr(env, "close"):
        env.close()
    return config, agent, nnx.state(agent.policy, nnx.Param)


policy_action_fn = base.policy_action_fn
arm_source = base.arm_source
arm_context = base.arm_context
strict_stationary = base.strict_stationary
strict_switching = base.strict_switching


def load_estimator():
    _require_bound()
    return base.load_estimator()
