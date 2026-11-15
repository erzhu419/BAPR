"""Robust-isolated, hard option-specific Gaussian policy."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from jax_experiments.networks.ensemble_critic import VectorizedLinear
from jax_experiments.networks.policy import LOG_STD_MAX, LOG_STD_MIN


def _module_list(layers):
    list_cls = getattr(nnx, "List", None)
    return list_cls(layers) if list_cls is not None else layers


class HardOptionGaussianPolicy(nnx.Module):
    """Separate robust MLP plus jointly trained option-specific MLPs.

    A binary valid bit selects either the robust actor or exactly one option
    actor. The policy never averages actions from separately trained experts.
    Option MLPs are vectorized for one XLA graph and initialize as exact copies
    of the robust actor, then specialize only through return gradients.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int,
                 num_options: int, *, rngs: nnx.Rngs):
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.latent_dim = int(num_options)
        self.ep_dim = self.latent_dim + 1
        self.policy_mode = "hard_persistent_option"

        base_layers = []
        option_layers = []
        width = self.obs_dim
        for _ in range(2):
            base = nnx.Linear(width, hidden_dim, rngs=rngs)
            option = VectorizedLinear(
                width, hidden_dim, self.latent_dim, rngs=rngs)
            option.kernel.value = jnp.broadcast_to(
                base.kernel.value[None], option.kernel.value.shape)
            option.bias.value = jnp.broadcast_to(
                base.bias.value[None, None], option.bias.value.shape)
            base_layers.append(base)
            option_layers.append(option)
            width = int(hidden_dim)
        self.base_layers = _module_list(base_layers)
        self.option_layers = _module_list(option_layers)

        self.base_mean = nnx.Linear(hidden_dim, act_dim, rngs=rngs)
        self.base_log_std = nnx.Linear(hidden_dim, act_dim, rngs=rngs)
        self.option_mean = VectorizedLinear(
            hidden_dim, act_dim, self.latent_dim, rngs=rngs)
        self.option_log_std = VectorizedLinear(
            hidden_dim, act_dim, self.latent_dim, rngs=rngs)
        for base, option in (
                (self.base_mean, self.option_mean),
                (self.base_log_std, self.option_log_std)):
            option.kernel.value = jnp.broadcast_to(
                base.kernel.value[None], option.kernel.value.shape)
            option.bias.value = jnp.broadcast_to(
                base.bias.value[None, None], option.bias.value.shape)

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

    def _base_distribution(self, obs):
        hidden = obs
        for layer in self.base_layers:
            hidden = nnx.silu(layer(hidden))
        return self.base_mean(hidden), jnp.clip(
            self.base_log_std(hidden), LOG_STD_MIN, LOG_STD_MAX)

    def _option_distributions(self, obs):
        leading = obs.shape[:-1]
        flat = obs.reshape((-1, self.obs_dim))
        hidden = jnp.broadcast_to(
            flat[None], (self.latent_dim,) + flat.shape)
        for layer in self.option_layers:
            hidden = nnx.silu(layer(hidden))
        means = self.option_mean(hidden)
        log_stds = jnp.clip(
            self.option_log_std(hidden), LOG_STD_MIN, LOG_STD_MAX)
        shape = (self.latent_dim,) + leading + (self.act_dim,)
        return means.reshape(shape), log_stds.reshape(shape)

    @staticmethod
    def _select_option(values, option):
        # [O, ..., A] -> [..., O, A], then hard one-hot selection.
        values = jnp.moveaxis(values, 0, -2)
        return jnp.sum(values * option[..., :, None], axis=-2)

    def __call__(self, obs, ep_tensor=None):
        base_mean, base_log_std = self._base_distribution(obs)
        option_mean, option_log_std = self._option_distributions(obs)
        option, valid = self._split_context(obs, ep_tensor)
        selected_mean = self._select_option(option_mean, option)
        selected_log_std = self._select_option(option_log_std, option)
        mean = base_mean + valid * (selected_mean - base_mean)
        log_std = base_log_std + valid * (selected_log_std - base_log_std)
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
        mean, _ = self._base_distribution(obs)
        return jnp.tanh(mean)

    def adaptation_strength(self, obs, context):
        option = self.deterministic(obs, context)
        robust = self.base_deterministic(obs)
        _, valid = self._split_context(obs, context)
        return valid * jnp.mean(
            jnp.abs(option - robust), axis=-1, keepdims=True)

    def learned_adaptation_gate(self, obs, context):
        _, valid = self._split_context(obs, context)
        return jnp.ones_like(valid)

    def warmstart_conditioned_from_base(self) -> bool:
        return True
