"""Residual SAC actor with gradient-isolated robust anchoring."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from jax_experiments.networks.policy import LOG_STD_MAX, LOG_STD_MIN
from jax_experiments.networks.residual_policy import ResidualGaussianPolicy


class AnchoredResidualGaussianPolicy(ResidualGaussianPolicy):
    """Keep robust and adaptive actor gradients on separate parameter paths.

    A zero context is the continually trained robust actor.  A unit-valid
    context adds a bounded mode-conditioned mean residual while treating the
    robust mean and log standard deviation as constants for that adaptive
    loss.  The residual output is exactly zero at initialization.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int,
        latent_dim: int,
        residual_delta: float = 0.5,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__(
            obs_dim,
            act_dim,
            hidden_dim,
            latent_dim,
            residual_delta=residual_delta,
            policy_mode="residual",
            rngs=rngs,
        )
        if not self.zero_residual_output():
            raise RuntimeError("anchored residual could not initialize at zero")

    def __call__(self, obs, ep_tensor=None):
        base = obs
        for layer in self.base_layers:
            base = nnx.relu(layer(base))
        base_mean = self.base_mean(base)
        base_log_std = jnp.clip(
            self.base_log_std(base), LOG_STD_MIN, LOG_STD_MAX)

        latent, gate = self._split_context(obs, ep_tensor)
        residual = jnp.concatenate([obs, latent], axis=-1)
        for layer in self.residual_layers:
            residual = nnx.relu(layer(residual))
        delta = self.residual_delta * jnp.tanh(
            self.residual_mean(residual))

        # Oracle/adaptive rows must not modify the robust actor.  Robust rows
        # have gate=0 and retain the ordinary SAC gradient exactly.
        anchored_mean = (
            (1.0 - gate) * base_mean
            + gate * jax.lax.stop_gradient(base_mean)
        )
        anchored_log_std = (
            (1.0 - gate) * base_log_std
            + gate * jax.lax.stop_gradient(base_log_std)
        )
        return anchored_mean + gate * delta, anchored_log_std

