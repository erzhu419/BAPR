"""SAC variants that keep a pretrained robust actor byte-for-byte frozen."""
from __future__ import annotations

from jax_experiments.algos.anchored_regime_sac import AnchoredRegimeSAC
from jax_experiments.algos.bapr_v5 import BAPRv5
from jax_experiments.networks.frozen_mode_residual_policy import (
    FrozenModeResidualGaussianPolicy,
)


class _FrozenBaseMixin:
    robust_actor_frozen = True

    def controller_update_flags(
        self,
        iteration: int | None = None,
    ) -> tuple[bool, bool, bool]:
        del iteration
        return False, True, True

    def train_policy_gate(self, iteration: int | None = None) -> bool:
        del iteration
        return False


class FrozenAnchoredRegimeSAC(_FrozenBaseMixin, AnchoredRegimeSAC):
    """Shared bounded residual over a frozen robust actor."""

    def context_checkpoint_signature(self) -> dict[str, object]:
        signature = super().context_checkpoint_signature()
        signature.update({
            "kind": "frozen_anchored_regime_sac_v2",
            "robust_actor_frozen": True,
            "oracle_rollouts_only": True,
            "train_advantage_constraint": bool(
                self.config.bapr_v2_train_advantage_constraint),
            "train_advantage_lcb_scale": float(
                self.config.bapr_v2_train_advantage_lcb_scale),
            "train_advantage_margin": float(
                self.config.bapr_v2_train_advantage_margin),
            "train_advantage_temperature": float(
                self.config.bapr_v2_train_advantage_temperature),
            "train_advantage_weight": float(
                self.config.bapr_v2_train_advantage_weight),
            "train_update_filter": bool(
                self.config.bapr_v2_train_update_filter),
            "train_update_tolerance": float(
                self.config.bapr_v2_train_update_tolerance),
            "train_update_floor": float(
                self.config.bapr_v2_train_update_floor),
        })
        return signature


class FrozenModeResidualSAC(_FrozenBaseMixin, BAPRv5):
    """Independent full-capacity mode heads over a frozen robust actor."""

    uses_per_context_alpha = True
    uses_frozen_mode_residual = True

    def __init__(self, obs_dim: int, act_dim: int, config, seed: int = 0):
        if config.bapr_v2_critic_target_mode != "min":
            raise ValueError(
                "frozen_mode_residual_sac must match SAC's minimum target")
        if config.bapr_v2_training_schedule != "joint":
            raise ValueError(
                "frozen_mode_residual_sac requires joint residual training")
        super().__init__(obs_dim, act_dim, config, seed=seed)
        if not self.policy.zero_mode_residual_output():
            raise RuntimeError(
                "mode residuals were not initialized as robust copies")

    def _make_policy(self):
        return FrozenModeResidualGaussianPolicy(
            self.obs_dim,
            self.act_dim,
            self.config.hidden_dim,
            self.latent_dim,
            rngs=self.rngs,
        )

    def context_checkpoint_signature(self) -> dict[str, object]:
        signature = super().context_checkpoint_signature()
        signature.update({
            "kind": "frozen_mode_residual_sac_v2",
            "robust_actor_frozen": True,
            "zero_initialized_mode_residuals": True,
            "independent_mode_heads": True,
            "per_context_alpha": True,
            "critic_target_mode": "min",
            "oracle_rollouts_only": True,
        })
        return signature

    def multi_update(self, *args, **kwargs):
        metrics = super().multi_update(*args, **kwargs)
        metrics.update({
            "frozen_robust_actor": 1.0,
            "independent_mode_residuals": 1.0,
        })
        return metrics
