"""Isolated robust and mode-conditioned critic ensembles."""
from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

from jax_experiments.networks.ensemble_critic import EnsembleCritic


class AnchoredDualEnsembleCritic(nnx.Module):
    """Select between independent robust and adaptive Q ensembles.

    The robust critic sees observations only.  The adaptive critic sees the
    observation and soft mode posterior.  A zero valid bit therefore prevents
    adaptive critic gradients from changing robust value estimates.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        context_dim: int,
        hidden_dim: int,
        num_modes: int,
        ensemble_size: int,
        n_layers: int = 3,
        *,
        rngs: nnx.Rngs,
    ):
        if int(context_dim) != int(num_modes) + 1:
            raise ValueError(
                "anchored critic context must be posterior plus valid bit")
        self.obs_dim = int(obs_dim)
        self.context_dim = int(context_dim)
        self.num_modes = int(num_modes)
        self.base_critic = EnsembleCritic(
            obs_dim,
            act_dim,
            hidden_dim,
            ensemble_size=ensemble_size,
            n_layers=n_layers,
            rngs=rngs,
        )
        self.adaptive_critic = EnsembleCritic(
            obs_dim + num_modes,
            act_dim,
            hidden_dim,
            ensemble_size=ensemble_size,
            n_layers=n_layers,
            rngs=rngs,
        )

    def __call__(self, obs_aug, act):
        obs = obs_aug[..., :self.obs_dim]
        context = obs_aug[..., -self.context_dim:]
        posterior = context[..., :self.num_modes]
        valid = jnp.clip(context[..., -1], 0.0, 1.0)
        robust_q = self.base_critic(obs, act)
        adaptive_q = self.adaptive_critic(
            jnp.concatenate([obs, posterior], axis=-1), act)
        return robust_q + valid[None] * (adaptive_q - robust_q)

    def compute_reg_norm(self):
        return 0.5 * (
            self.base_critic.compute_reg_norm()
            + self.adaptive_critic.compute_reg_norm())

