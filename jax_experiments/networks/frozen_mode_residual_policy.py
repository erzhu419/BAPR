"""Mode-specific additive residuals with an immutable robust fallback."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from jax_experiments.networks.ensemble_critic import VectorizedLinear
from jax_experiments.networks.policy import LOG_STD_MAX, LOG_STD_MIN


def _module_list(layers):
    list_cls = getattr(nnx, "List", None)
    return list_cls(layers) if list_cls is not None else layers


class FrozenModeResidualGaussianPolicy(nnx.Module):
    """Full-capacity mode residuals initialized at exactly zero.

    The base always follows one common computation graph. Vectorized
    mode-specific networks add unrestricted pre-tanh mean and log-standard-
    deviation residuals. Zero output heads make every context bitwise equal
    to the base at bootstrap without relying on numerically non-identical
    copies of a batched GEMM.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int,
        num_modes: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.latent_dim = int(num_modes)
        self.ep_dim = self.latent_dim + 1
        self.policy_mode = "frozen_mode_residual"

        base_layers = []
        mode_layers = []
        width = self.obs_dim
        for _ in range(2):
            base = nnx.Linear(width, hidden_dim, rngs=rngs)
            mode = VectorizedLinear(
                width, hidden_dim, self.latent_dim, rngs=rngs)
            base_layers.append(base)
            mode_layers.append(mode)
            width = int(hidden_dim)
        self.base_layers = _module_list(base_layers)
        self.mode_layers = _module_list(mode_layers)

        self.base_mean = nnx.Linear(hidden_dim, act_dim, rngs=rngs)
        self.base_log_std = nnx.Linear(hidden_dim, act_dim, rngs=rngs)
        self.mode_mean = VectorizedLinear(
            hidden_dim, act_dim, self.latent_dim, rngs=rngs)
        self.mode_log_std = VectorizedLinear(
            hidden_dim, act_dim, self.latent_dim, rngs=rngs)
        for mode in (self.mode_mean, self.mode_log_std):
            mode.kernel.value = jnp.zeros_like(mode.kernel.value)
            mode.bias.value = jnp.zeros_like(mode.bias.value)

    def _split_context(self, obs, context):
        shape = obs.shape[:-1]
        if context is None:
            posterior = jnp.zeros(
                shape + (self.latent_dim,), dtype=obs.dtype)
            valid = jnp.zeros(shape + (1,), dtype=obs.dtype)
            return posterior, valid
        posterior = jnp.asarray(
            context[..., :self.latent_dim], dtype=obs.dtype)
        valid = jnp.clip(
            context[..., self.latent_dim:self.latent_dim + 1],
            0.0,
            1.0,
        )
        return posterior, valid

    def _base_distribution(self, obs):
        hidden = obs
        for layer in self.base_layers:
            hidden = nnx.relu(layer(hidden))
        return self.base_mean(hidden), jnp.clip(
            self.base_log_std(hidden), LOG_STD_MIN, LOG_STD_MAX)

    def _mode_distributions(self, obs):
        leading = obs.shape[:-1]
        flat = obs.reshape((-1, self.obs_dim))
        hidden = jnp.broadcast_to(
            flat[None], (self.latent_dim,) + flat.shape)
        for layer in self.mode_layers:
            hidden = jax.nn.relu(layer(hidden))
        means = self.mode_mean(hidden)
        log_stds = self.mode_log_std(hidden)
        shape = (self.latent_dim,) + leading + (self.act_dim,)
        return means.reshape(shape), log_stds.reshape(shape)

    @staticmethod
    def _select_mode(values, posterior):
        values = jnp.moveaxis(values, 0, -2)
        nonnegative = jnp.clip(posterior, 0.0)
        total = jnp.sum(nonnegative, axis=-1, keepdims=True)
        normalized = nonnegative / jnp.maximum(total, 1e-8)
        fallback = jax.nn.one_hot(
            jnp.argmax(posterior, axis=-1),
            posterior.shape[-1],
            dtype=values.dtype,
        )
        weights = jnp.where(total > 1e-8, normalized, fallback)
        return jnp.sum(values * weights[..., :, None], axis=-2)

    def __call__(self, obs, ep_tensor=None):
        base_mean, base_log_std = self._base_distribution(obs)
        mode_means, mode_log_stds = self._mode_distributions(obs)
        posterior, valid = self._split_context(obs, ep_tensor)
        delta_mean = self._select_mode(mode_means, posterior)
        delta_log_std = self._select_mode(mode_log_stds, posterior)

        # The base is also masked out by the optimizer, but stop_gradient
        # makes the separation explicit in the policy graph.
        frozen_mean = jax.lax.stop_gradient(base_mean)
        frozen_log_std = jax.lax.stop_gradient(base_log_std)
        anchored_mean = (
            (1.0 - valid) * base_mean + valid * frozen_mean)
        anchored_log_std = (
            (1.0 - valid) * base_log_std + valid * frozen_log_std)
        mean = anchored_mean + valid * delta_mean
        log_std = anchored_log_std + valid * delta_log_std
        return mean, jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)

    @staticmethod
    def _sample_distribution(mean, log_std, key, noise=None):
        std = jnp.exp(log_std)
        if noise is None:
            noise = jax.random.normal(key, mean.shape)
        pre_tanh = mean + std * noise
        action = jnp.tanh(pre_tanh)
        log_prob = -0.5 * (
            jnp.square((pre_tanh - mean) / std)
            + 2.0 * log_std + jnp.log(2.0 * jnp.pi))
        log_prob = log_prob.sum(axis=-1)
        log_prob -= jnp.sum(
            jnp.log(1.0 - jnp.square(action) + 1e-6), axis=-1)
        return action, log_prob

    def sample(self, obs, key, ep_tensor=None):
        mean, log_std = self(obs, ep_tensor)
        return self._sample_distribution(mean, log_std, key)

    def sample_pair(self, obs, key, ep_tensor):
        mode_mean, mode_log_std = self(obs, ep_tensor)
        base_mean, base_log_std = self._base_distribution(obs)
        noise = jax.random.normal(key, mode_mean.shape)
        mode = self._sample_distribution(
            mode_mean, mode_log_std, key, noise=noise)
        base = self._sample_distribution(
            base_mean, base_log_std, key, noise=noise)
        return mode[0], base[0], mode[1], base[1]

    def deterministic(self, obs, ep_tensor=None):
        mean, _ = self(obs, ep_tensor)
        return jnp.tanh(mean)

    def base_deterministic(self, obs):
        mean, _ = self._base_distribution(obs)
        return jnp.tanh(mean)

    def adaptation_strength(self, obs, context):
        adaptive = self.deterministic(obs, context)
        robust = self.base_deterministic(obs)
        _, valid = self._split_context(obs, context)
        return valid * jnp.mean(
            jnp.abs(adaptive - robust), axis=-1, keepdims=True)

    def learned_adaptation_gate(self, obs, context):
        _, valid = self._split_context(obs, context)
        return jnp.ones_like(valid)

    def warmstart_conditioned_from_base(self) -> bool:
        return True

    def zero_mode_residual_output(self) -> bool:
        for mode in (self.mode_mean, self.mode_log_std):
            if not jnp.array_equal(
                    mode.kernel.value,
                    jnp.zeros_like(mode.kernel.value)):
                return False
            if not jnp.array_equal(
                    mode.bias.value,
                    jnp.zeros_like(mode.bias.value)):
                return False
        return True
