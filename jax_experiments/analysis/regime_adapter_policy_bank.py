"""Evaluation-only policy bank for independently trained residual adapters."""
from __future__ import annotations

import tempfile
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.analysis import final_task_sweep
from jax_experiments.analysis import regime_adapter_fork as protocol
from jax_experiments.analysis.run_regime_adapter_branch import (
    validate_published_bundle,
)
from jax_experiments.common.checkpoint import load_checkpoint
from jax_experiments.common.logging import Logger
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.networks.residual_policy import ResidualGaussianPolicy
from jax_experiments.train import make_algo, make_env


def _module_list(values):
    list_cls = getattr(nnx, "List", None)
    return list_cls(values) if list_cls is not None else values


class ModePolicyBank(nnx.Module):
    """Route one-hot controller context to one frozen-base residual policy."""

    def __init__(self, policies):
        policies = list(policies)
        if len(policies) != len(protocol.MODES):
            raise ValueError(
                f"policy bank requires {len(protocol.MODES)} adapters")
        if any(policy.policy_mode != "residual" for policy in policies):
            raise ValueError("policy bank accepts residual policies only")
        self.policies = _module_list(policies)
        self.num_modes = len(policies)
        self.context_dim = self.num_modes + 1

    def _split_context(self, obs, context):
        if context is None:
            return (
                jnp.zeros(obs.shape[:-1] + (self.num_modes,), obs.dtype),
                jnp.zeros(obs.shape[:-1] + (1,), obs.dtype),
            )
        if context.shape[-1] != self.context_dim:
            raise ValueError(
                f"policy-bank context width must be {self.context_dim}")
        weights = jnp.clip(context[..., :self.num_modes], 0.0)
        total = jnp.sum(weights, axis=-1, keepdims=True)
        weights = weights / jnp.maximum(total, 1e-8)
        gate = jnp.clip(context[..., -1:], 0.0, 1.0)
        gate = gate * (total > 1e-8).astype(gate.dtype)
        return weights, gate

    def __call__(self, obs, ep_tensor=None):
        base_mean, base_log_std = self.policies[0](obs, None)
        weights, gate = self._split_context(obs, ep_tensor)
        adaptive_means = []
        adaptive_log_stds = []
        for mode, policy in enumerate(self.policies):
            latent = jax.nn.one_hot(
                mode, self.num_modes, dtype=obs.dtype)
            latent = jnp.broadcast_to(
                latent, obs.shape[:-1] + (self.num_modes,))
            policy_context = jnp.concatenate([
                latent,
                jnp.ones(obs.shape[:-1] + (1,), dtype=obs.dtype),
            ], axis=-1)
            mean, log_std = policy(obs, policy_context)
            adaptive_means.append(mean)
            adaptive_log_stds.append(log_std)
        stacked_mean = jnp.stack(adaptive_means, axis=-2)
        stacked_log_std = jnp.stack(adaptive_log_stds, axis=-2)
        selected_mean = jnp.sum(
            stacked_mean * weights[..., :, None], axis=-2)
        selected_log_std = jnp.sum(
            stacked_log_std * weights[..., :, None], axis=-2)
        return (
            base_mean + gate * (selected_mean - base_mean),
            base_log_std + gate * (selected_log_std - base_log_std),
        )

    def sample(self, obs, key, ep_tensor=None):
        mean, log_std = self(obs, ep_tensor)
        return ResidualGaussianPolicy._sample_distribution(
            mean, log_std, key)

    def deterministic(self, obs, ep_tensor=None):
        mean, _ = self(obs, ep_tensor)
        return jnp.tanh(mean)


class ModePolicyBankAgent:
    """Direct-context agent exposing a frozen controller map to eval code."""

    uses_regime_context = True

    def __init__(self, policy: ModePolicyBank, config,
                 controller_map=protocol.MODES):
        self.policy = policy
        self.config = config
        self.context_dim = len(protocol.MODES) + 1
        self.belief_dim = self.context_dim
        self.rngs = nnx.Rngs(int(config.seed) + 9821)
        self._current_mode_id = 0
        self._task_metadata_ready = False
        self.set_controller_map(controller_map)

    def set_task_metadata(self, tasks) -> None:
        mode_ids = sorted(int(task["mode_id"]) for task in tasks)
        if mode_ids != list(protocol.MODES):
            raise ValueError(f"policy bank requires all modes, got {mode_ids}")
        self._task_metadata_ready = True

    def set_controller_map(self, controller_map) -> None:
        values = tuple(int(value) for value in controller_map)
        if len(values) != len(protocol.MODES):
            raise ValueError("controller map must have one entry per mode")
        if any(value not in (*protocol.MODES, -1) for value in values):
            raise ValueError("controller map entries must be -1 or mode ids")
        self.controller_map = values

    def set_oracle_task_id(self, mode_id: int) -> None:
        mode_id = protocol.require_mode(mode_id)
        self._current_mode_id = mode_id

    def set_eval_task(self, task) -> None:
        self.set_oracle_task_id(int(task["mode_id"]))

    def context_for_task_id(self, mode_id: int):
        if not self._task_metadata_ready:
            raise RuntimeError("set_task_metadata must run before evaluation")
        controller = self.controller_map[protocol.require_mode(mode_id)]
        if controller < 0:
            return jnp.zeros((self.context_dim,), dtype=jnp.float32)
        return jnp.concatenate([
            jax.nn.one_hot(
                controller, len(protocol.MODES), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
        ])

    def rollout_context(self, iteration: int | None = None):
        del iteration
        return self.context_for_task_id(self._current_mode_id)

    def _build_belief_jax(self):
        return self.rollout_context()

    def select_action(self, obs, deterministic: bool = False):
        obs = jnp.asarray(obs, dtype=jnp.float32)
        if obs.ndim == 1:
            obs = obs[None]
        context = jnp.broadcast_to(
            self.rollout_context()[None],
            obs.shape[:-1] + (self.context_dim,))
        if deterministic:
            action = self.policy.deterministic(obs, context)
        else:
            action, _ = self.policy.sample(
                obs, self.rngs.params(), context)
        return np.asarray(action[0])


@dataclass
class LoadedPolicyBank:
    config: object
    tasks: list
    agent: ModePolicyBankAgent
    source_agents: list
    environments: list
    temporary_logs: list[tempfile.TemporaryDirectory]

    def close(self) -> None:
        for temporary in self.temporary_logs:
            temporary.cleanup()
        for env in self.environments:
            if hasattr(env, "close"):
                env.close()


def load_policy_bank_from(
        seed: int,
        delta: float,
        *,
        validate_bundle,
        adapter_bundle_dir,
        expected_next_iteration: int,
        expected_total_steps: int,
        source_record_from_bundle=None) -> LoadedPolicyBank:
    seed = protocol.require_seed(seed)
    delta = protocol.require_delta(delta)
    policies = []
    agents = []
    environments = []
    temporary_logs = []
    base_hashes = set()
    source_records = set()
    config = None
    tasks = None
    if source_record_from_bundle is None:
        source_record_from_bundle = jsonable_source_record
    try:
        for mode in protocol.MODES:
            payload = validate_bundle(
                seed, "adapter", delta, mode)
            directory = adapter_bundle_dir(seed, delta, mode)
            current_config = final_task_sweep.load_config(directory)
            env = make_env(current_config, seed_offset=0)
            current_tasks = env.sample_tasks(current_config.task_num)
            agent = make_algo(
                current_config.algo, env.obs_dim, env.act_dim,
                current_config)
            agent.set_task_metadata(current_tasks)
            replay = ReplayBuffer(
                env.obs_dim, env.act_dim, capacity=1,
                belief_dim=agent.belief_dim)
            temporary = tempfile.TemporaryDirectory()
            logger = Logger(temporary.name)
            next_iteration, total_steps = load_checkpoint(
                str(directory / "checkpoints"), agent, replay, logger,
                current_config.algo, load_replay_buffer=False)
            if (next_iteration != expected_next_iteration
                    or total_steps != expected_total_steps):
                raise ValueError("adapter bank contains a stale checkpoint")
            base_hashes.add(protocol.base_policy_sha256(agent.policy))
            source_records.add(source_record_from_bundle(directory))
            policies.append(agent.policy)
            agents.append(agent)
            environments.append(env)
            temporary_logs.append(temporary)
            if config is None:
                config = current_config
                tasks = current_tasks
        if len(base_hashes) != 1 or len(source_records) != 1:
            raise ValueError(
                "adapter bank does not share one frozen robust controller")
        bank = ModePolicyBank(policies)
        bank_agent = ModePolicyBankAgent(bank, config)
        bank_agent.set_task_metadata(tasks)
        return LoadedPolicyBank(
            config, tasks, bank_agent, agents, environments, temporary_logs)
    except Exception:
        for temporary in temporary_logs:
            temporary.cleanup()
        for env in environments:
            if hasattr(env, "close"):
                env.close()
        raise


def load_policy_bank(seed: int, delta: float) -> LoadedPolicyBank:
    return load_policy_bank_from(
        seed,
        delta,
        validate_bundle=validate_published_bundle,
        adapter_bundle_dir=protocol.adapter_bundle_dir,
        expected_next_iteration=protocol.ADAPTER_FINAL_NEXT_ITERATION,
        expected_total_steps=protocol.ADAPTER_FINAL_TOTAL_STEPS,
    )


def jsonable_source_record(directory) -> str:
    bootstrap = protocol.read_json(
        directory / "checkpoints" / protocol.BOOTSTRAP_NAME)
    source_record = (bootstrap.get("source") or {}).get("bundle_manifest")
    if not isinstance(source_record, dict):
        raise ValueError("adapter bundle lacks source provenance")
    return str(sorted(source_record.items()))
