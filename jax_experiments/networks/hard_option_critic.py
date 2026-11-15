"""Vectorized robust-isolated critic bank for hard persistent options."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from jax_experiments.networks.ensemble_critic import (
    EnsembleCritic,
    VectorizedLinear,
)


def _module_list(layers):
    list_cls = getattr(nnx, "List", None)
    return list_cls(layers) if list_cls is not None else layers


class HardOptionEnsembleCritic(nnx.Module):
    """Independent robust and option Q heads selected by a hard context."""

    def __init__(self, obs_dim: int, act_dim: int, context_dim: int,
                 hidden_dim: int, num_options: int, ensemble_size: int,
                 n_layers: int = 3, *, rngs: nnx.Rngs):
        self.obs_dim = int(obs_dim)
        self.context_dim = int(context_dim)
        self.num_options = int(num_options)
        self.ensemble_size = int(ensemble_size)
        self.n_hidden = int(n_layers)
        self.base_critic = EnsembleCritic(
            obs_dim, act_dim, hidden_dim,
            ensemble_size=ensemble_size, n_layers=n_layers, rngs=rngs)

        total_heads = self.num_options * self.ensemble_size
        option_layers = []
        width = int(obs_dim + act_dim)
        for index in range(n_layers + 1):
            out = 1 if index == n_layers else int(hidden_dim)
            layer = VectorizedLinear(width, out, total_heads, rngs=rngs)
            base = self.base_critic.layers[index]
            layer.kernel.value = jnp.tile(
                base.kernel.value, (self.num_options, 1, 1))
            layer.bias.value = jnp.tile(
                base.bias.value, (self.num_options, 1, 1))
            option_layers.append(layer)
            width = out
        self.option_layers = _module_list(option_layers)

    def __call__(self, obs_aug, act):
        leading = obs_aug.shape[:-1]
        obs = obs_aug[..., :self.obs_dim]
        context = obs_aug[..., -self.context_dim:]
        option = context[..., :self.num_options]
        valid = jnp.clip(
            context[..., self.num_options:self.num_options + 1], 0.0, 1.0)

        base_q = self.base_critic(obs, act)
        flat = jnp.concatenate([obs, act], axis=-1).reshape(
            (-1, self.obs_dim + act.shape[-1]))
        total_heads = self.num_options * self.ensemble_size
        hidden = jnp.broadcast_to(flat[None], (total_heads,) + flat.shape)
        for index, layer in enumerate(self.option_layers):
            hidden = layer(hidden)
            if index < self.n_hidden:
                hidden = jax.nn.relu(hidden)
        option_q = hidden.squeeze(-1).reshape(
            (self.num_options, self.ensemble_size, -1))
        option_flat = option.reshape((-1, self.num_options))
        selected = jnp.einsum("oeb,bo->eb", option_q, option_flat)
        selected = selected.reshape((self.ensemble_size,) + leading)
        gate = jnp.moveaxis(valid, -1, 0)
        return base_q + gate * (selected - base_q)

    def compute_reg_norm(self):
        base = self.base_critic.compute_reg_norm()
        option = jnp.zeros(
            (self.num_options, self.ensemble_size), dtype=base.dtype)
        for layer in self.option_layers:
            norm = jnp.sum(jnp.abs(layer.kernel.value), axis=(1, 2))
            norm += jnp.sum(jnp.abs(layer.bias.value), axis=(1, 2))
            option += norm.reshape((self.num_options, self.ensemble_size))
        return (base + jnp.sum(option, axis=0)) / (self.num_options + 1)
