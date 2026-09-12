"""Shared FiLM actor for persistent BAPR options."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from jax_experiments.networks.policy import LOG_STD_MAX, LOG_STD_MIN


def _module_list(layers):
    list_cls = getattr(nnx, "List", None)
    return list_cls(layers) if list_cls is not None else layers


class PersistentOptionGaussianPolicy(nnx.Module):
    """One shared SAC actor modulated by a persistent one-hot option.

    The last context coordinate is an option-valid bit. A zero context is the
    robust option. Nonzero contexts FiLM-modulate the shared state features;
    they do not select independent actor networks or interpolate actions from
    separately trained controllers.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int,
                 num_options: int, *, rngs: nnx.Rngs):
        self.latent_dim = int(num_options)
        self.ep_dim = self.latent_dim + 1
        self.act_dim = int(act_dim)
        self.policy_mode = "persistent_option"

        layers = []
        width = int(obs_dim)
        for _ in range(2):
            layers.append(nnx.Linear(width, hidden_dim, rngs=rngs))
            width = int(hidden_dim)
        self.base_layers = _module_list(layers)
        self.option_scale = nnx.Linear(
            self.latent_dim, hidden_dim, rngs=rngs)
        self.option_shift = nnx.Linear(
            self.latent_dim, hidden_dim, rngs=rngs)
        self.base_mean = nnx.Linear(hidden_dim, act_dim, rngs=rngs)
        self.base_log_std = nnx.Linear(hidden_dim, act_dim, rngs=rngs)

        # All options initially reproduce the robust actor exactly. The option
        # coordinates separate only through the joint SAC return objective.
        for layer in (self.option_scale, self.option_shift):
            layer.kernel.value = jnp.zeros_like(layer.kernel.value)
            layer.bias.value = jnp.zeros_like(layer.bias.value)

    def _split_context(self, obs, context):
        shape = obs.shape[:-1]
        if context is None:
            option = jnp.zeros(shape + (self.latent_dim,), obs.dtype)
            valid = jnp.zeros(shape + (1,), obs.dtype)
            return option, valid
        option = jnp.asarray(context[..., :self.latent_dim], obs.dtype)
        valid = jnp.clip(
            context[..., self.latent_dim:self.latent_dim + 1], 0.0, 1.0)
        return option, valid

    def _features(self, obs, context):
        hidden = obs
        for layer in self.base_layers:
            hidden = nnx.silu(layer(hidden))
        option, valid = self._split_context(obs, context)
        scale = jnp.tanh(self.option_scale(option))
        shift = jnp.tanh(self.option_shift(option))
        modulated = hidden * (1.0 + valid * scale) + valid * shift
        return modulated, hidden, valid

    def __call__(self, obs, ep_tensor=None):
        hidden, _, _ = self._features(obs, ep_tensor)
        mean = self.base_mean(hidden)
        log_std = jnp.clip(
            self.base_log_std(hidden), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

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
        option_mean, option_log_std = self(obs, ep_tensor)
        robust_mean, robust_log_std = self(obs, None)
        noise = jax.random.normal(key, option_mean.shape)
        option = self._sample_distribution(
            option_mean, option_log_std, key, noise=noise)
        robust = self._sample_distribution(
            robust_mean, robust_log_std, key, noise=noise)
        return option[0], robust[0], option[1], robust[1]

    def deterministic(self, obs, ep_tensor=None):
        mean, _ = self(obs, ep_tensor)
        return jnp.tanh(mean)

    def base_deterministic(self, obs):
        return self.deterministic(obs, None)

    def adaptation_strength(self, obs, context):
        modulated, robust, valid = self._features(obs, context)
        return valid * jnp.mean(jnp.abs(modulated - robust), axis=-1,
                                keepdims=True)

    def learned_adaptation_gate(self, obs, context):
        _, valid = self._split_context(obs, context)
        return jnp.ones_like(valid)

    def warmstart_conditioned_from_base(self) -> bool:
        # Zero FiLM parameters already make every option equal to robust.
        return True
