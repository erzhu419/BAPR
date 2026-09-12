"""SAC specialist with an immutable robust rollout policy."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from flax import nnx

from jax_experiments.algos.sac_base import SACBase
from jax_experiments.common.checkpoint import (
    _restore_tree_like,
    _to_numpy_tree,
)


class SACSwitchRecovery(SACBase):
    """Train a specialist while retaining a frozen robust behavior policy."""

    CHECKPOINT_SCHEMA = "bapr.sac-switch-recovery.v1"
    robust_actor_frozen = True

    def __init__(self, obs_dim: int, act_dim: int, config, seed: int = 0):
        super().__init__(obs_dim, act_dim, config, seed=seed)
        self.fallback_policy = deepcopy(self.policy)
        self.switch_recovery_target_mode = int(
            config.switch_recovery_target_mode)
        self.switch_recovery_segment_steps = int(
            config.switch_recovery_segment_steps)
        self.switch_recovery_termination_penalty = float(
            config.switch_recovery_termination_penalty)
        self._last_switch_recovery_termination_rate = 0.0
        self._last_switch_recovery_raw_reward = 0.0

    def set_fallback_policy_state(self, state) -> None:
        nnx.update(self.fallback_policy, state)

    def multi_update(self, stacked_batch: dict, **kwargs):
        metrics = super().multi_update(stacked_batch, **kwargs)
        metrics.update({
            "switch_recovery_target_mode": float(
                self.switch_recovery_target_mode),
            "switch_recovery_termination_penalty": float(
                self.switch_recovery_termination_penalty),
            "switch_recovery_termination_rate": float(
                self._last_switch_recovery_termination_rate),
            "switch_recovery_raw_reward": float(
                self._last_switch_recovery_raw_reward),
            "frozen_robust_actor": 1.0,
        })
        return metrics

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "schema": self.CHECKPOINT_SCHEMA,
            "target_mode": self.switch_recovery_target_mode,
            "segment_steps": self.switch_recovery_segment_steps,
            "termination_penalty": self.switch_recovery_termination_penalty,
            "fallback_policy": _to_numpy_tree(
                nnx.state(self.fallback_policy, nnx.Param)),
        }

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        expected = (
            state.get("schema") == self.CHECKPOINT_SCHEMA
            and int(state.get("target_mode", -1))
            == self.switch_recovery_target_mode
            and int(state.get("segment_steps", -1))
            == self.switch_recovery_segment_steps
            and float(state.get("termination_penalty", -1.0))
            == self.switch_recovery_termination_penalty
        )
        if not expected:
            raise ValueError("incompatible switch-recovery SAC checkpoint")
        restored = _restore_tree_like(
            nnx.state(self.fallback_policy, nnx.Param),
            state["fallback_policy"],
            "switch-recovery frozen robust policy",
            allow_fallback=False,
        )
        nnx.update(self.fallback_policy, restored)
