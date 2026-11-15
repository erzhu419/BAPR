"""Joint robust SAC plus a zero-initialized mode residual."""
from __future__ import annotations

from jax_experiments.algos.bapr_v5 import BAPRv5
from jax_experiments.networks.anchored_dual_critic import (
    AnchoredDualEnsembleCritic,
)
from jax_experiments.networks.anchored_residual_policy import (
    AnchoredResidualGaussianPolicy,
)


class AnchoredRegimeSAC(BAPRv5):
    """Train robust fallback and shared residual without gradient leakage."""

    uses_hard_option_heads = False
    uses_anchored_residual = True
    uses_per_context_alpha = True

    def __init__(self, obs_dim: int, act_dim: int, config, seed: int = 0):
        if config.bapr_v2_policy_mode != "residual":
            raise ValueError(
                "anchored_regime_sac requires residual policy mode")
        if config.bapr_v2_critic_target_mode != "min":
            raise ValueError(
                "anchored_regime_sac must match SAC's minimum target")
        if config.bapr_v2_training_schedule != "joint":
            raise ValueError(
                "anchored_regime_sac requires joint controller training")
        super().__init__(obs_dim, act_dim, config, seed=seed)

    def _make_policy(self):
        return AnchoredResidualGaussianPolicy(
            self.obs_dim,
            self.act_dim,
            self.config.hidden_dim,
            self.latent_dim,
            residual_delta=self.config.bapr_v2_residual_delta,
            rngs=self.rngs,
        )

    def _make_critic(self):
        return AnchoredDualEnsembleCritic(
            self.obs_dim,
            self.act_dim,
            self.context_dim,
            self.config.hidden_dim,
            self.latent_dim,
            self.config.ensemble_size,
            n_layers=3,
            rngs=self.rngs,
        )

    def context_checkpoint_signature(self) -> dict[str, object]:
        signature = super().context_checkpoint_signature()
        signature.update({
            "kind": "anchored_regime_sac_v1",
            "zero_initialized_shared_residual": True,
            "gradient_isolated_robust_actor": True,
            "gradient_isolated_robust_critic": True,
            "dual_context_relabel": True,
            "per_context_alpha": True,
            "critic_target_mode": "min",
        })
        return signature

    def multi_update(self, *args, **kwargs):
        metrics = super().multi_update(*args, **kwargs)
        metrics.update({
            "anchored_residual": 1.0,
            "anchored_robust_actor_isolated": 1.0,
            "anchored_robust_critic_isolated": 1.0,
        })
        return metrics

